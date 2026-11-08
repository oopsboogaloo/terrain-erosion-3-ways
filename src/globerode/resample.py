"""Moving data between equirectangular images and the cube sphere (DESIGN §5, §9.1).

- Levels image -> per-cell class: 4x4 supersampled majority vote.
- Cube field -> equirect: Catmull-Rom bicubic on halo-padded faces.
- Equirect classes -> equirect classes at another size: majority vote
  (used to re-lock the coast at output resolution).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit, prange
from PIL import Image

from globerode.config import Config, ConfigError
from globerode.grid import FACE_FRAMES, Grid, cell_fij, tangent_xyz

SUB = 4  # supersampling per axis for categorical resampling
HALO = 2  # cells of halo for bicubic interpolation


# --- reading the levels image -----------------------------------------------------


def read_levels(path: Path, cfg: Config) -> np.ndarray:
    """Read the levels image and map pixel values to class indices (int16).

    Class index = position in ``cfg.input.classes``. Raises ConfigError for a
    bad shape, an unsupported image mode, or pixel values with no class.
    """
    try:
        with Image.open(path) as im:
            if im.mode not in ("L", "P", "I;16", "I"):
                raise ConfigError(
                    f"{path}: image mode {im.mode!r} not supported; "
                    "use 8/16-bit grayscale or a palette image"
                )
            values = np.asarray(im).astype(np.int64)
    except FileNotFoundError:
        raise ConfigError(f"levels image not found: {path}") from None
    h, w = values.shape
    if w != 2 * h:
        raise ConfigError(f"{path}: levels image must be 2:1 (equirectangular), got {w}x{h}")

    lut = np.full(65536, -1, dtype=np.int16)
    for k, c in enumerate(cfg.input.classes):
        lut[c.value] = k
    if values.min() < 0 or values.max() > 65535:
        raise ConfigError(f"{path}: pixel values must be in 0..65535")
    classes = lut[values]
    bad = classes < 0
    if bad.any():
        vals, counts = np.unique(values[bad], return_counts=True)
        listing = ", ".join(f"{v} ({n} px)" for v, n in zip(vals[:20], counts[:20], strict=True))
        more = "" if len(vals) <= 20 else f" and {len(vals) - 20} more"
        raise ConfigError(
            f"{path}: pixel values with no class in the config: {listing}{more}. "
            "Check for anti-aliased edges or add these values to [[input.classes]]."
        )
    return classes


# --- equirect -> cube (categorical) ---------------------------------------------------


@njit(cache=True, inline="always")
def _lonlat_to_pixel(x: float, y: float, z: float, w: int, h: int) -> tuple[int, int]:
    lon = math.atan2(y, x)
    lat = math.asin(min(1.0, max(-1.0, z)))
    px = math.floor((lon + math.pi) / (2.0 * math.pi) * w) % w
    py = math.floor((0.5 * math.pi - lat) / math.pi * h)
    return px, min(max(py, 0), h - 1)


@njit(cache=True, inline="always")
def _majority(votes: np.ndarray) -> int:
    best = 0
    for k in range(1, votes.shape[0]):
        if votes[k] > votes[best]:
            best = k
    return best


@njit(cache=True, parallel=True)
def _equirect_to_cube_classes(img: np.ndarray, nclasses: int, n: int) -> np.ndarray:
    h, w = img.shape
    out = np.empty(6 * n * n, dtype=np.int16)
    for idx in prange(6 * n * n):
        f, i, j = cell_fij(idx, n)
        votes = np.zeros(nclasses, dtype=np.int32)
        for a in range(SUB):
            tv = math.tan(-0.25 * math.pi + (i + (a + 0.5) / SUB) * 0.5 * math.pi / n)
            for b in range(SUB):
                tu = math.tan(-0.25 * math.pi + (j + (b + 0.5) / SUB) * 0.5 * math.pi / n)
                x, y, z = tangent_xyz(f, tu, tv, FACE_FRAMES)
                px, py = _lonlat_to_pixel(x, y, z, w, h)
                votes[img[py, px]] += 1
        out[idx] = _majority(votes)
    return out


def equirect_to_cube_classes(grid: Grid, classes: np.ndarray, nclasses: int) -> np.ndarray:
    """Per-cell class index by 4x4 supersampled majority vote (ties -> lower index)."""
    return _equirect_to_cube_classes(np.ascontiguousarray(classes, np.int16), nclasses, grid.n)


# --- equirect -> equirect (categorical) -----------------------------------------------


@njit(cache=True, parallel=True)
def _equirect_majority(img: np.ndarray, nclasses: int, ow: int) -> np.ndarray:
    ih, iw = img.shape
    oh = ow // 2
    out = np.empty((oh, ow), dtype=np.int16)
    for oy in prange(oh):
        votes = np.zeros(nclasses, dtype=np.int32)
        for ox in range(ow):
            votes[:] = 0
            for a in range(SUB):
                fy = (oy + (a + 0.5) / SUB) / oh
                py = min(int(fy * ih), ih - 1)
                for b in range(SUB):
                    fx = (ox + (b + 0.5) / SUB) / ow
                    px = min(int(fx * iw), iw - 1)
                    votes[img[py, px]] += 1
            out[oy, ox] = _majority(votes)
    return out


def equirect_majority(classes: np.ndarray, nclasses: int, out_width: int) -> np.ndarray:
    """Resample a class image to ``out_width`` x ``out_width/2`` by majority vote.

    Same size returns an exact copy, so the coast is pixel-identical.
    """
    if classes.shape == (out_width // 2, out_width):
        return classes.astype(np.int16, copy=True)
    return _equirect_majority(np.ascontiguousarray(classes, np.int16), nclasses, out_width)


# --- cube -> equirect (continuous) ----------------------------------------------------------


@njit(cache=True, inline="always")
def _face_coords(x: float, y: float, z: float, n: int) -> tuple[int, float, float]:
    """Face and continuous (row, col) in centre-index units (cell k centre = k)."""
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        f = 0 if x > 0 else 1
    elif ay >= az:
        f = 2 if y > 0 else 3
    else:
        f = 4 if z > 0 else 5
    fr = FACE_FRAMES
    dc = x * fr[f, 0, 0] + y * fr[f, 0, 1] + z * fr[f, 0, 2]
    du = x * fr[f, 1, 0] + y * fr[f, 1, 1] + z * fr[f, 1, 2]
    dv = x * fr[f, 2, 0] + y * fr[f, 2, 1] + z * fr[f, 2, 2]
    col = (math.atan(du / dc) + 0.25 * math.pi) / (0.5 * math.pi) * n - 0.5
    row = (math.atan(dv / dc) + 0.25 * math.pi) / (0.5 * math.pi) * n - 0.5
    return f, row, col


@njit(cache=True, inline="always")
def _bilinear_face(field: np.ndarray, f: int, row: float, col: float, n: int) -> float:
    r = min(max(row, 0.0), n - 1.0)
    c = min(max(col, 0.0), n - 1.0)
    r0 = min(math.floor(r), n - 2) if n > 1 else 0
    c0 = min(math.floor(c), n - 2) if n > 1 else 0
    tr = r - r0
    tc = c - c0
    base = f * n * n
    if n == 1:
        return float(field[base])
    v00 = field[base + r0 * n + c0]
    v01 = field[base + r0 * n + c0 + 1]
    v10 = field[base + (r0 + 1) * n + c0]
    v11 = field[base + (r0 + 1) * n + c0 + 1]
    return float((v00 * (1 - tc) + v01 * tc) * (1 - tr) + (v10 * (1 - tc) + v11 * tc) * tr)


@njit(cache=True, inline="always")
def _bilinear_padded(padded: np.ndarray, g: int, row: float, col: float, n: int) -> float:
    """Bilinear sample of a padded face at centre-index coords (may be slightly < 0 or > n-1)."""
    m = n + 2 * HALO
    r = min(max(row + HALO, 0.0), m - 1.000001)
    c = min(max(col + HALO, 0.0), m - 1.000001)
    r0 = math.floor(r)
    c0 = math.floor(c)
    tr = r - r0
    tc = c - c0
    v00 = padded[g, r0, c0]
    v01 = padded[g, r0, c0 + 1]
    v10 = padded[g, r0 + 1, c0]
    v11 = padded[g, r0 + 1, c0 + 1]
    return float((v00 * (1 - tc) + v01 * tc) * (1 - tr) + (v10 * (1 - tc) + v11 * tc) * tr)


@njit(cache=True)
def _halo_value(
    field: np.ndarray, padded: np.ndarray, f: int, i: int, j: int, n: int, use_padded: bool
) -> float:
    tv = math.tan(-0.25 * math.pi + (i + 0.5) * 0.5 * math.pi / n)
    tu = math.tan(-0.25 * math.pi + (j + 0.5) * 0.5 * math.pi / n)
    x, y, z = tangent_xyz(f, tu, tv, FACE_FRAMES)
    g, row, col = _face_coords(x, y, z, n)
    if use_padded:
        return _bilinear_padded(padded, g, row, col, n)
    return _bilinear_face(field, g, row, col, n)


CORNER_REFINE = 2  # refinement passes for corner halo blocks


@njit(cache=True, parallel=True)
def _pad_faces(field: np.ndarray, n: int) -> np.ndarray:
    """Faces with a HALO-cell border sampled from the neighbouring faces.

    Edge halos (one coordinate outside the face) sample points well inside
    the neighbouring face. Corner blocks (both outside) sample points near
    another face's corner, which needs that face's own corner halo: they get
    a rough first value and are then refined from the padded arrays a few
    times, which keeps cube corners second-order accurate.
    """
    m = n + 2 * HALO
    out = np.zeros((6, m, m), dtype=np.float64)
    for f in prange(6):
        for pi in range(m):
            i = pi - HALO
            for pj in range(m):
                j = pj - HALO
                in_i = 0 <= i < n
                in_j = 0 <= j < n
                if in_i and in_j:
                    out[f, pi, pj] = field[f * n * n + i * n + j]
                else:
                    out[f, pi, pj] = _halo_value(field, out, f, i, j, n, False)
    corner = np.empty((6, m, m), dtype=np.float64)
    for _ in range(CORNER_REFINE):
        for f in prange(6):
            for pi in range(m):
                i = pi - HALO
                for pj in range(m):
                    j = pj - HALO
                    if not (0 <= i < n) and not (0 <= j < n):
                        corner[f, pi, pj] = _halo_value(field, out, f, i, j, n, True)
        for f in prange(6):
            for pi in range(m):
                i = pi - HALO
                for pj in range(m):
                    j = pj - HALO
                    if not (0 <= i < n) and not (0 <= j < n):
                        out[f, pi, pj] = corner[f, pi, pj]
    return out


@njit(cache=True, inline="always")
def _cr_weights(t: float) -> tuple[float, float, float, float]:
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * (-t3 + 2 * t2 - t),
        0.5 * (3 * t3 - 5 * t2 + 2),
        0.5 * (-3 * t3 + 4 * t2 + t),
        0.5 * (t3 - t2),
    )


@njit(cache=True, parallel=True)
def _cube_to_equirect(padded: np.ndarray, n: int, w: int) -> np.ndarray:
    h = w // 2
    out = np.empty((h, w), dtype=np.float32)
    for py in prange(h):
        lat = 0.5 * math.pi - (py + 0.5) / h * math.pi
        cl = math.cos(lat)
        sl = math.sin(lat)
        for px in range(w):
            lon = (px + 0.5) / w * 2.0 * math.pi - math.pi
            f, row, col = _face_coords(cl * math.cos(lon), cl * math.sin(lon), sl, n)
            r0 = math.floor(row)
            c0 = math.floor(col)
            wr = _cr_weights(row - r0)
            wc = _cr_weights(col - c0)
            acc = 0.0
            for a in range(4):
                pr = r0 - 1 + a + HALO
                for b in range(4):
                    acc += wr[a] * wc[b] * padded[f, pr, c0 - 1 + b + HALO]
            out[py, px] = acc
    return out


def cube_to_equirect(grid: Grid, field: np.ndarray, width: int) -> np.ndarray:
    """Bicubic (Catmull-Rom) resample of a cell field to a width x width/2 image."""
    if field.shape != (grid.ncells,):
        raise ValueError("field must have one value per cell")
    if width % 2:
        raise ValueError("width must be even")
    padded = _pad_faces(np.ascontiguousarray(field, np.float64), grid.n)
    return _cube_to_equirect(padded, grid.n, width)


# --- cube -> cube (continuous, for the cascade) ---------------------------------------------


@njit(cache=True, parallel=True)
def _cube_to_cube(padded: np.ndarray, n: int, nf: int) -> np.ndarray:
    out = np.empty(6 * nf * nf, dtype=np.float64)
    for c in prange(6 * nf * nf):
        f, i, j = cell_fij(c, nf)
        tu = math.tan(-0.25 * math.pi + (j + 0.5) * 0.5 * math.pi / nf)
        tv = math.tan(-0.25 * math.pi + (i + 0.5) * 0.5 * math.pi / nf)
        x, y, z = tangent_xyz(f, tu, tv, FACE_FRAMES)
        g, row, col = _face_coords(x, y, z, n)
        r0 = math.floor(row)
        c0 = math.floor(col)
        wr = _cr_weights(row - r0)
        wc = _cr_weights(col - c0)
        acc = 0.0
        for a in range(4):
            pr = r0 - 1 + a + HALO
            for b in range(4):
                acc += wr[a] * wc[b] * padded[g, pr, c0 - 1 + b + HALO]
        out[c] = acc
    return out


def cube_to_cube(coarse: Grid, field: np.ndarray, fine: Grid) -> np.ndarray:
    """Bicubic resample of a cell field from one cube-sphere grid to another."""
    if field.shape != (coarse.ncells,):
        raise ValueError("field must have one value per coarse cell")
    padded = _pad_faces(np.ascontiguousarray(field, np.float64), coarse.n)
    out: np.ndarray = _cube_to_cube(padded, coarse.n, fine.n)
    return out
