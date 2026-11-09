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


def _coarse_factor(grid: Grid, sigma_m: float, cells_per_sigma: float = 4.0) -> int:
    cell_m = grid.radius_m * (np.pi / 2) / grid.n
    k = max(1, int(sigma_m / cells_per_sigma / cell_m))
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


def smooth_over_land(
    grid: Grid,
    values: np.ndarray,
    land: np.ndarray,
    sigma_m: float,
    cells_per_sigma: float = 4.0,
) -> np.ndarray:
    """Gaussian mean of ``values`` over land cells only (normalised convolution).

    Computed on a coarse grid (cell <= sigma / cells_per_sigma) and upsampled
    bicubically.
    """
    k = _coarse_factor(grid, sigma_m, cells_per_sigma)
    coarse = Grid(grid.n // k, radius_m=grid.radius_m)
    area = grid.cell_area_m2.astype(np.float64)
    val_c = block_average(grid, np.where(land, values, 0.0), area, k)
    frac_c = block_average(grid, land.astype(np.float64), area, k)
    everywhere = np.ones(coarse.ncells, dtype=np.bool_)
    diffuser = Diffuser(coarse)
    num = diffuser.gaussian(val_c, everywhere, sigma_m)
    wsum = diffuser.gaussian(frac_c, everywhere, sigma_m)
    ok = wsum > 1e-6
    out_c = np.where(ok, num / np.where(ok, wsum, 1.0), 0.0)
    out: np.ndarray = cube_to_cube(coarse, out_c, grid) if k > 1 else out_c
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
    off = smooth_over_land(grid, h_init - h_eroded, land, sigma_m)
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


def cap_peaks_regional(
    grid: Grid,
    h: np.ndarray,
    h_init: np.ndarray,
    land: np.ndarray,
    ratio: float,
    soft: float,
    sigma_m: float,
    min_land_m: float,
) -> np.ndarray:
    """Cap the regional (smoothed) height instead of each cell (D34).

    S = land mean of h over ~sigma, T = ratio * land mean of h_init. Where S
    rises above T it is compressed with the same tanh as ``cap_peaks``, and
    every cell is shifted down by S - cap(S). Local relief (peaks above their
    valleys) is kept, so tops are not flattened; summits can exceed the cap
    by their local relief.
    """
    if ratio <= 0:
        return h
    # 2 cells per sigma: at 8K this halves the grid, 5x faster and ~0.8 GB less
    # memory than 4 cells, with height differences under 10 m (D34).
    s_h = smooth_over_land(grid, h, land, sigma_m, cells_per_sigma=2.0)
    t = ratio * np.maximum(smooth_over_land(grid, h_init, land, sigma_m, 2.0), 1.0)
    e = s_h - t
    capped = np.where(e > 0, t + soft * t * np.tanh(np.maximum(e, 0.0) / (soft * t)), s_h)
    out: np.ndarray = np.where(land, np.maximum(h - (s_h - capped), min_land_m), h)
    return out
