"""Local height restore (decision D16, replaces the global hypsometry restore of D6).

Erosion and uplift change heights unevenly. To keep mountains about as high
as painted *where they were painted*, eroded land is scaled by a smooth
factor::

    S = G(h_init * w) / G(h_eroded * w),   h_out = h_eroded * S

G is a Gaussian of width sigma (~200 km) and w is the land fraction, so the
coast needs no special handling (normalised convolution). Each region's
local mean height then matches the prepared surface, while valleys and
ridges keep their shape.

S is smooth, so it is computed on a coarse grid (cell <= sigma / 4) from
area-weighted block averages, then upsampled bicubically. This keeps the cost
small at 8K-16K.
"""

from __future__ import annotations

import numpy as np

from globerode.grid import Grid
from globerode.resample import cube_to_cube
from globerode.smooth import Diffuser

S_MIN, S_MAX = 0.1, 10.0


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
    """Scale eroded land so its local mean height matches ``h_init`` (sea unchanged)."""
    k = _coarse_factor(grid, sigma_m)
    coarse = Grid(grid.n // k, radius_m=grid.radius_m)
    area = grid.cell_area_m2.astype(np.float64)
    num_c = block_average(grid, np.where(land, h_init, 0.0), area, k)
    den_c = block_average(grid, np.where(land, h_eroded, 0.0), area, k)
    everywhere = np.ones(coarse.ncells, dtype=np.bool_)
    diff = Diffuser(coarse)
    num = diff.gaussian(num_c, everywhere, sigma_m)
    den = diff.gaussian(den_c, everywhere, sigma_m)
    ok = den > 1e-6 * max(float(np.abs(den).max()), 1.0)
    s_coarse = np.where(ok, num / np.where(ok, den, 1.0), 1.0)
    s_coarse = np.clip(s_coarse, S_MIN, S_MAX)
    s = cube_to_cube(coarse, s_coarse, grid) if k > 1 else s_coarse
    out: np.ndarray = np.where(land, np.maximum(h_eroded * s, min_land_m), h_eroded)
    return out
