"""Equiangular cube-sphere grid (DESIGN §3).

Cells are numbered with one flat index across all six faces::

    idx = face * N * N + i * N + j

``i`` is the row (along the face's v axis), ``j`` the column (along u).

Geometry is computed on the fly by small Numba functions (decision D1);
only cell areas are cached. All functions that take ``(face, i, j)`` work in
``float64`` internally and return unit-sphere quantities unless they say
otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numba import njit, prange

# Face frames: centre axis c, u axis (+j direction), v axis (+i direction).
# Each frame is right-handed: u x v = c (checked by tests).
FACE_FRAMES = np.array(
    [
        # centre        u              v
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 0: +X
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # 1: -X
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # 2: +Y
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # 3: -Y
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # 4: +Z
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # 5: -Z
    ],
    dtype=np.float64,
)

# Edge ids: which side of a face a step leaves through.
EDGE_W, EDGE_E, EDGE_S, EDGE_N = 0, 1, 2, 3  # j<0, j>=N, i<0, i>=N

# The 8 neighbour offsets (di, dj). Order is fixed: 4 edge neighbours first.
NEIGHBOUR_OFFSETS = np.array(
    [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    dtype=np.int64,
)


def _build_edge_table() -> np.ndarray:
    """Work out, for every (face, edge), how cells map onto the next face.

    Returns int64 array (6, 4, 4) of
    ``[neighbour_face, fixed_is_last, along_is_i, along_reversed]``:

    - ``fixed_is_last``: the crossed-into row/column is index N-1 (else 0);
    - ``along_is_i``: the along-edge index becomes the neighbour's i (else j);
    - ``along_reversed``: along index k maps to N-1-k (else k).

    Derived exactly from points on the shared edge, so it holds for any N.
    """
    table = np.zeros((6, 4, 4), dtype=np.int64)
    for f in range(6):
        c, u, v = FACE_FRAMES[f]
        for edge, (axis, sign) in enumerate([(u, -1), (u, 1), (v, -1), (v, 1)]):
            other = v if axis is u else u
            out_dir = sign * axis
            g = next(k for k in range(6) if np.allclose(FACE_FRAMES[k, 0], out_dir))
            cg, ug, vg = FACE_FRAMES[g]
            t = 0.5  # any interior point along the edge
            p = c + sign * axis + t * other
            a_g = p @ ug / (p @ cg)
            b_g = p @ vg / (p @ cg)
            if math.isclose(abs(a_g), 1.0):
                # Fixed coordinate is j on the neighbour; along index is its i.
                fixed_is_last = int(a_g > 0)
                along_is_i = 1
                along_reversed = int(b_g / t < 0)
            else:
                fixed_is_last = int(b_g > 0)
                along_is_i = 0
                along_reversed = int(a_g / t < 0)
            table[f, edge] = (g, fixed_is_last, along_is_i, along_reversed)
    return table


EDGE_TABLE = _build_edge_table()


# --- Numba kernels -----------------------------------------------------------


@njit(cache=True, inline="always")
def cell_index(face: int, i: int, j: int, n: int) -> int:
    return face * n * n + i * n + j


@njit(cache=True, inline="always")
def cell_fij(idx: int, n: int) -> tuple[int, int, int]:
    face = idx // (n * n)
    rem = idx - face * n * n
    return face, rem // n, rem - (rem // n) * n


@njit(cache=True, inline="always")
def _angle(k: float, n: int) -> float:
    """Equiangular angle of a (possibly fractional) cell coordinate.

    ``k`` is a cell index; ``k + 0.5`` is the cell centre. Edges are at
    integer values: 0 -> -pi/4, n -> +pi/4.
    """
    return -0.25 * math.pi + k * (0.5 * math.pi) / n


@njit(cache=True, inline="always")
def tangent_xyz(face: int, tu: float, tv: float, frames: np.ndarray) -> tuple[float, float, float]:
    """Unit vector for tangent-plane coordinates (tan(alpha), tan(beta)) on a face."""
    x = frames[face, 0, 0] + tu * frames[face, 1, 0] + tv * frames[face, 2, 0]
    y = frames[face, 0, 1] + tu * frames[face, 1, 1] + tv * frames[face, 2, 1]
    z = frames[face, 0, 2] + tu * frames[face, 1, 2] + tv * frames[face, 2, 2]
    r = math.sqrt(x * x + y * y + z * z)
    return x / r, y / r, z / r


@njit(cache=True, inline="always")
def cell_xyz(face: int, i: int, j: int, n: int, frames: np.ndarray) -> tuple[float, float, float]:
    """Unit vector of a cell centre."""
    tu = math.tan(_angle(j + 0.5, n))
    tv = math.tan(_angle(i + 0.5, n))
    return tangent_xyz(face, tu, tv, frames)


@njit(cache=True, inline="always")
def corner_xyz(
    face: int, ci: int, cj: int, n: int, frames: np.ndarray
) -> tuple[float, float, float]:
    """Unit vector of a cell corner; ``ci, cj`` in 0..n (grid-line indices)."""
    return tangent_xyz(face, math.tan(_angle(cj, n)), math.tan(_angle(ci, n)), frames)


@njit(cache=True)
def xyz_to_cell(x: float, y: float, z: float, n: int, frames: np.ndarray) -> tuple[int, int, int]:
    """Cell containing the direction (x, y, z). Need not be normalised."""
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        face = 0 if x > 0 else 1
    elif ay >= az:
        face = 2 if y > 0 else 3
    else:
        face = 4 if z > 0 else 5
    dc = x * frames[face, 0, 0] + y * frames[face, 0, 1] + z * frames[face, 0, 2]
    du = x * frames[face, 1, 0] + y * frames[face, 1, 1] + z * frames[face, 1, 2]
    dv = x * frames[face, 2, 0] + y * frames[face, 2, 1] + z * frames[face, 2, 2]
    a = math.atan(du / dc)
    b = math.atan(dv / dc)
    j = math.floor((a + 0.25 * math.pi) / (0.5 * math.pi) * n)
    i = math.floor((b + 0.25 * math.pi) / (0.5 * math.pi) * n)
    j = min(max(j, 0), n - 1)
    i = min(max(i, 0), n - 1)
    return face, i, j


@njit(cache=True)
def neighbour(face: int, i: int, j: int, di: int, dj: int, n: int, edge_table: np.ndarray) -> int:
    """Flat index of the neighbour at offset (di, dj), or -1 if none.

    Offsets must be in {-1, 0, 1}. The only missing neighbours are the
    diagonals pointing into a cube corner (both coordinates leave the face).
    """
    ni = i + di
    nj = j + dj
    in_i = 0 <= ni < n
    in_j = 0 <= nj < n
    if in_i and in_j:
        return cell_index(face, ni, nj, n)
    if not in_i and not in_j:
        return -1
    if not in_j:
        edge = 0 if nj < 0 else 1
        k = ni
    else:
        edge = 2 if ni < 0 else 3
        k = nj
    g = edge_table[face, edge, 0]
    fixed = n - 1 if edge_table[face, edge, 1] == 1 else 0
    along = n - 1 - k if edge_table[face, edge, 3] == 1 else k
    if edge_table[face, edge, 2] == 1:
        return cell_index(g, along, fixed, n)
    return cell_index(g, fixed, along, n)


@njit(cache=True, inline="always")
def arc(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    """Great-circle angle between two unit vectors (numerically stable)."""
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), ax * bx + ay * by + az * bz)


@njit(cache=True, inline="always")
def _solid_angle_f(x: float, y: float) -> float:
    return math.atan2(x * y, math.sqrt(1.0 + x * x + y * y))


@njit(cache=True)
def cell_solid_angle(i: int, j: int, n: int) -> float:
    """Exact solid angle (steradians) of a cell; the same on every face."""
    x0 = math.tan(_angle(j, n))
    x1 = math.tan(_angle(j + 1, n))
    y0 = math.tan(_angle(i, n))
    y1 = math.tan(_angle(i + 1, n))
    return (
        _solid_angle_f(x1, y1)
        - _solid_angle_f(x0, y1)
        - _solid_angle_f(x1, y0)
        + _solid_angle_f(x0, y0)
    )


@njit(cache=True)
def shared_edge_angle(face: int, i: int, j: int, k: int, n: int, frames: np.ndarray) -> float:
    """Arc length (radians) of the edge between a cell and edge neighbour k (0..3).

    ``k`` indexes the first four entries of NEIGHBOUR_OFFSETS: W, E, S, N.
    """
    if k == 0:  # west edge: grid line cj = j, rows i..i+1
        p = corner_xyz(face, i, j, n, frames)
        q = corner_xyz(face, i + 1, j, n, frames)
    elif k == 1:
        p = corner_xyz(face, i, j + 1, n, frames)
        q = corner_xyz(face, i + 1, j + 1, n, frames)
    elif k == 2:
        p = corner_xyz(face, i, j, n, frames)
        q = corner_xyz(face, i, j + 1, n, frames)
    else:
        p = corner_xyz(face, i + 1, j, n, frames)
        q = corner_xyz(face, i + 1, j + 1, n, frames)
    return arc(p[0], p[1], p[2], q[0], q[1], q[2])


@njit(cache=True, parallel=True)
def _all_xyz(n: int, frames: np.ndarray) -> np.ndarray:
    out = np.empty((6 * n * n, 3), dtype=np.float64)
    for idx in prange(6 * n * n):
        f, i, j = cell_fij(idx, n)
        x, y, z = cell_xyz(f, i, j, n, frames)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return out


@njit(cache=True, parallel=True)
def _all_areas(n: int) -> np.ndarray:
    face_area = np.empty(n * n, dtype=np.float64)
    for k in prange(n * n):
        face_area[k] = cell_solid_angle(k // n, k % n, n)
    out = np.empty(6 * n * n, dtype=np.float64)
    for f in range(6):
        out[f * n * n : (f + 1) * n * n] = face_area
    return out


@njit(cache=True, parallel=True)
def _all_neighbours(n: int, offsets: np.ndarray, edge_table: np.ndarray) -> np.ndarray:
    out = np.empty((6 * n * n, 8), dtype=np.int32)
    for idx in prange(6 * n * n):
        f, i, j = cell_fij(idx, n)
        for k in range(8):
            out[idx, k] = neighbour(f, i, j, offsets[k, 0], offsets[k, 1], n, edge_table)
    return out


@njit(cache=True, parallel=True)
def _xyz_to_index(xyz: np.ndarray, n: int, frames: np.ndarray) -> np.ndarray:
    out = np.empty(xyz.shape[0], dtype=np.int64)
    for k in prange(xyz.shape[0]):
        f, i, j = xyz_to_cell(xyz[k, 0], xyz[k, 1], xyz[k, 2], n, frames)
        out[k] = cell_index(f, i, j, n)
    return out


# --- Python API ----------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A cube-sphere grid with N x N cells per face on a sphere of radius_m."""

    n: int
    radius_m: float = 6_371_000.0

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"N must be >= 1, got {self.n}")
        if 6 * self.n * self.n > np.iinfo(np.int32).max:
            raise ValueError(f"N={self.n} is too large for int32 cell indices")
        if self.radius_m <= 0:
            raise ValueError(f"radius must be positive, got {self.radius_m}")

    @property
    def ncells(self) -> int:
        return 6 * self.n * self.n

    @cached_property
    def cell_area_m2(self) -> np.ndarray:
        """Exact area of every cell in m^2 (float32; the one cached array, D1)."""
        return (_all_areas(self.n) * self.radius_m**2).astype(np.float32)

    def xyz(self) -> np.ndarray:
        """Unit vectors of all cell centres, shape (ncells, 3), float64."""
        return _all_xyz(self.n, FACE_FRAMES)

    def lonlat_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """Longitude (-180..180) and latitude (-90..90) of all cell centres."""
        p = self.xyz()
        lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(p[:, 2], -1.0, 1.0)))
        return lon, lat

    def neighbours(self) -> np.ndarray:
        """All 8 neighbours per cell, shape (ncells, 8), int32, -1 if none.

        For tests and small grids. Kernels call ``neighbour`` directly (D1).
        """
        return _all_neighbours(self.n, NEIGHBOUR_OFFSETS, EDGE_TABLE)

    def index_of(self, xyz: np.ndarray) -> np.ndarray:
        """Flat cell index containing each direction in ``xyz`` (shape (k, 3))."""
        pts = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
        return _xyz_to_index(pts, self.n, FACE_FRAMES)
