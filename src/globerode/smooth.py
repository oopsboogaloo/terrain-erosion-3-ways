"""Masked heat-diffusion smoothing on the cube sphere (DESIGN §6.2, decision D2).

Finite-volume Laplacian over the 4 edge neighbours::

    h_i += dt * sum_e  w_ie * (h_e - h_i),   w_ie = L_e / (d_e * A_i)

on the unit sphere (L_e shared edge length, d_e centre distance, A_i cell
area). Diffusing for time t = (sigma / R)^2 / 2 is a Gaussian blur of width
sigma. Cells outside the mask are frozen and excluded (no flow across the
mask edge).

The weights depend only on a cell's (i, j) within its face: every face is
the same grid rotated by a cube symmetry. So one (N, N, 4) float32 table
serves all six faces.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from globerode.grid import (
    EDGE_TABLE,
    FACE_FRAMES,
    NEIGHBOUR_OFFSETS,
    Grid,
    arc,
    cell_fij,
    cell_solid_angle,
    cell_xyz,
    neighbour,
    shared_edge_angle,
)

STABILITY = 0.9  # fraction of the explicit stability limit


@njit(cache=True, parallel=True)
def face_weights(n: int, face: int = 0) -> np.ndarray:
    """(n, n, 4) float32 table of w = L / (d * A) for the W, E, S, N neighbours."""
    out = np.empty((n, n, 4), dtype=np.float32)
    for i in prange(n):
        for j in range(n):
            p = cell_xyz(face, i, j, n, FACE_FRAMES)
            area = cell_solid_angle(i, j, n)
            for k in range(4):
                b = neighbour(
                    face, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE
                )
                fb, ib, jb = cell_fij(b, n)
                q = cell_xyz(fb, ib, jb, n, FACE_FRAMES)
                d = arc(p[0], p[1], p[2], q[0], q[1], q[2])
                length = shared_edge_angle(face, i, j, k, n, FACE_FRAMES)
                out[i, j, k] = length / (d * area)
    return out


@njit(cache=True, parallel=True)
def edge_neighbours(n: int) -> np.ndarray:
    """(ncells, 4) int32 table of W, E, S, N neighbours (16 B/cell)."""
    nbrs = np.empty((6 * n * n, 4), dtype=np.int32)
    for c in prange(6 * n * n):
        f, i, j = cell_fij(c, n)
        for k in range(4):
            nbrs[c, k] = neighbour(
                f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE
            )
    return nbrs


@njit(cache=True, parallel=True)
def _diffuse(
    h: np.ndarray,
    mask: np.ndarray,
    w: np.ndarray,
    nbrs: np.ndarray,
    n: int,
    dt: float,
    steps: int,
) -> np.ndarray:
    cur = h.copy()
    nxt = h.copy()
    ncells = h.shape[0]
    for _ in range(steps):
        for c in prange(ncells):
            if not mask[c]:
                continue
            rem = c % (n * n)
            i = rem // n
            j = rem - i * n
            hc = cur[c]
            acc = 0.0
            for k in range(4):
                b = nbrs[c, k]
                if mask[b]:
                    acc += w[i, j, k] * (cur[b] - hc)
            nxt[c] = hc + dt * acc
        cur, nxt = nxt, cur
    return cur


class Diffuser:
    """Masked diffusion on one grid, with its lookup tables built once (D12).

    Tables: (N, N, 4) float32 weights shared by all faces, and a
    (ncells, 4) int32 neighbour table. Reuse one Diffuser for repeated calls
    (the erosion loop) to avoid rebuilding them.
    """

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.weights = face_weights(grid.n)
        self.nbrs = edge_neighbours(grid.n)
        self.dt_max = STABILITY / float(self.weights.sum(axis=2).max())

    def diffuse(self, values: np.ndarray, mask: np.ndarray, t_unit: float) -> np.ndarray:
        """Diffuse for unit-sphere time ``t_unit`` (= metres^2 / R^2) over masked cells."""
        g = self.grid
        if values.shape != (g.ncells,) or mask.shape != (g.ncells,):
            raise ValueError("values and mask must have one value per cell")
        h = np.ascontiguousarray(values, dtype=np.float64)
        if t_unit <= 0:
            return h.copy()
        steps = max(1, math.ceil(t_unit / self.dt_max))
        out: np.ndarray = _diffuse(
            h,
            np.ascontiguousarray(mask, np.bool_),
            self.weights,
            self.nbrs,
            g.n,
            t_unit / steps,
            steps,
        )
        return out

    def gaussian(self, values: np.ndarray, mask: np.ndarray, sigma_m: float) -> np.ndarray:
        """Blur of width sigma_m (diffusion time sigma^2 / 2)."""
        return self.diffuse(values, mask, 0.5 * (sigma_m / self.grid.radius_m) ** 2)


def gaussian_smooth(grid: Grid, values: np.ndarray, mask: np.ndarray, sigma_m: float) -> np.ndarray:
    """Gaussian-like blur of width sigma_m over masked cells; others unchanged.

    Builds a temporary Diffuser; use one directly for repeated calls.
    """
    if values.shape != (grid.ncells,) or mask.shape != (grid.ncells,):
        raise ValueError("values and mask must have one value per cell")
    if sigma_m <= 0:
        return np.asarray(values, dtype=np.float64).copy()
    return Diffuser(grid).gaussian(values, mask, sigma_m)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    out: np.ndarray = t * t * (3.0 - 2.0 * t)
    return out


def coast_ramp(
    values: np.ndarray, coast_dist_m: np.ndarray, coast_value: float, ramp_m: float
) -> np.ndarray:
    """Blend land values toward coast_value near the coast (SPEC §8.2 coast ramp)."""
    if ramp_m <= 0:
        return values
    t = smoothstep(0.0, ramp_m, coast_dist_m)
    out: np.ndarray = coast_value + (values - coast_value) * t
    return out
