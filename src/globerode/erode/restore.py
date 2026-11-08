"""Local height restore (decision D16, replaces the global hypsometry restore of D6).

Erosion and uplift change heights unevenly. To keep land about as high as
painted *where it was painted*, a smooth offset is added to eroded land::

    offset = G((h_init - h_eroded) * w) / G(w),   h_out = h_eroded + offset

G is a Gaussian of width sigma (~200 km) and w is the land fraction, so the
coast needs no special handling (normalised convolution). Each region's
mean height then matches the prepared surface, and the eroded relief
(valley depths, peak heights above the valleys) is kept as it is. An
additive offset is used rather than a multiplicative factor: a factor
stretches relief where valleys are deep and overshot painted peaks by ~2x.

The offset is smooth, so it is computed on a coarse grid (cell <= sigma / 4)
from area-weighted block averages, then upsampled bicubically. This keeps
the cost small at 8K-16K.
"""

from __future__ import annotations

import numpy as np

from globerode.grid import Grid
from globerode.resample import cube_to_cube
from globerode.smooth import Diffuser


def _coarse_factor(grid: Grid, sigma_m: float) -> int:
    cell_m = grid.radius_m * (np.pi / 2) / grid.n
    k = max(1, int(sigma_m / 4 / cell_m))
    while grid.n % k:
        k -= 1
    return k


def block_average(grid: Grid, values: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Weighted k x k block average per face -> field on the N/k grid."""
    nc = grid.n // k
    v = (values * weights).reshape(6, nc, k, nc, k).sum(axis=(2, 4))
    w = weights.reshape(6, nc, k, nc, k).sum(axis=(2, 4))
    out: np.ndarray = (v / np.where(w > 0, w, 1.0)).reshape(6 * nc * nc)
    return out


def local_restore(
    grid: Grid,
    h_eroded: np.ndarray,
    h_init: np.ndarray,
    land: np.ndarray,
    sigma_m: float,
    min_land_m: float,
) -> np.ndarray:
    """Shift eroded land so its local mean height matches ``h_init`` (sea unchanged)."""
    k = _coarse_factor(grid, sigma_m)
    coarse = Grid(grid.n // k, radius_m=grid.radius_m)
    area = grid.cell_area_m2.astype(np.float64)
    diff_c = block_average(grid, np.where(land, h_init - h_eroded, 0.0), area, k)
    frac_c = block_average(grid, land.astype(np.float64), area, k)
    everywhere = np.ones(coarse.ncells, dtype=np.bool_)
    diffuser = Diffuser(coarse)
    num = diffuser.gaussian(diff_c, everywhere, sigma_m)
    wsum = diffuser.gaussian(frac_c, everywhere, sigma_m)
    ok = wsum > 1e-6
    off_coarse = np.where(ok, num / np.where(ok, wsum, 1.0), 0.0)
    off = cube_to_cube(coarse, off_coarse, grid) if k > 1 else off_coarse
    out: np.ndarray = np.where(land, np.maximum(h_eroded + off, min_land_m), h_eroded)
    return out


def cap_peaks(
    h: np.ndarray,
    h_init: np.ndarray,
    land: np.ndarray,
    ratio: float,
    soft: float,
) -> np.ndarray:
    """Softly compress land rising above ``ratio`` x its prepared height (D24).

    Threshold T = ratio * h_init per cell. Above it, the excess e = h - T is
    mapped to s * tanh(e / s) with s = soft * T, so peaks approach at most
    T * (1 + soft) and nothing below T changes. The mapping is monotone for a
    given T; where T changes steeply between neighbours a capped cell can, in
    rare cases, end slightly below a downhill neighbour. Flow and river masks
    are computed before this step, so they are unaffected.
    """
    if ratio <= 0:
        return h
    t = ratio * np.maximum(h_init, 1.0)
    s = soft * t
    e = h - t
    over = land & (e > 0)
    out = np.array(h, dtype=np.float64)
    out[over] = t[over] + s[over] * np.tanh(e[over] / s[over])
    return out
