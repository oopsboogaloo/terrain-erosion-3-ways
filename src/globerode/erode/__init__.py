"""Erosion (DESIGN §7): iteration loop and uplift.

One iteration:
  1. route (Priority-Flood+eps): filled heights, receivers, stack, area
  2. h <- filled (pits fill instantly, D4); accumulate lake fill
  3. stream power (Braun-Willett, n = 1) with uplift, and optional
     deposition (Yuan et al. 2019, D30)
  4. landslide slope limit (optional, D27)
  5. hillslope diffusion
  6. coast clamp: land >= min_land_height

Erodibility and rainfall may vary per cell (D26): K = k * erodibility, and
drainage area is rain-weighted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from globerode.erode.landslide import limit_slopes
from globerode.erode.routing import Routing, route
from globerode.erode.stream_power import stream_power_step
from globerode.erode.variation import Strata
from globerode.grid import Grid
from globerode.smooth import Diffuser


@dataclass(frozen=True)
class ErosionParams:
    k: float = 2e-6
    m: float = 0.5
    dt_yr: float = 1e5
    kappa_m2_per_yr: float = 1.0
    iterations: int = 100
    epsilon_m: float = 1e-3
    min_land_height_m: float = 1.0
    max_slope_deg: float = 0.0  # 0 = no slope limit
    deposition_g: float = 0.0  # 0 = no deposition (D30)
    deposition_passes: int = 5
    rho8_seed: int | None = None  # None = D8 receivers (D32)


@dataclass
class ErosionResult:
    height: np.ndarray  # float64, land eroded; sea cells unchanged
    routing: Routing  # routing of the final surface
    lake_fill_m: np.ndarray  # float64, total fill added by pit filling


def erode(
    grid: Grid,
    height_m: np.ndarray,
    land: np.ndarray,
    uplift_m_per_yr: np.ndarray,
    params: ErosionParams,
    diffuser: Diffuser | None = None,
    progress: Callable[[int, int], None] | None = None,
    erodibility: np.ndarray | None = None,
    rainfall: np.ndarray | None = None,
    strata: Strata | None = None,
    uplift_before_m: np.ndarray | None = None,
) -> ErosionResult:
    """Run ``params.iterations`` erosion iterations on land cells.

    ``erodibility`` and ``rainfall`` are optional per-cell factors (~1).
    ``strata`` multiplies K by a layered-rock factor of rock elevation, i.e.
    height minus total uplift: ``uplift_before_m`` (per cell, from earlier
    cascade levels) plus the uplift during this call (D33).
    """
    is_land = np.ascontiguousarray(land, np.bool_)
    h = np.array(height_m, dtype=np.float64)
    h[is_land] = np.maximum(h[is_land], params.min_land_height_m)
    k_field = np.full(grid.ncells, params.k, dtype=np.float64)
    if erodibility is not None:
        k_field *= erodibility
    lake = np.zeros(grid.ncells, dtype=np.float64)
    uplift_rock = np.asarray(uplift_m_per_yr, dtype=np.float64)
    before = 0.0 if uplift_before_m is None else np.asarray(uplift_before_m, dtype=np.float64)
    cell_area = grid.cell_area_m2.astype(np.float64) if params.deposition_g > 0 else None
    t_diff = params.kappa_m2_per_yr * params.dt_yr / grid.radius_m**2
    if params.kappa_m2_per_yr > 0 and diffuser is None:
        diffuser = Diffuser(grid)

    for it in range(params.iterations):
        r = route(grid, h, is_land, params.epsilon_m, rainfall, params.rho8_seed)
        lake += r.filled - h
        h = r.filled
        stream_power_step(
            h,
            uplift_m_per_yr,
            k_field
            if strata is None
            else k_field * strata.factor(h - before - uplift_rock * (it * params.dt_yr)),
            r.area,
            r.receiver,
            r.stack,
            is_land,
            grid.n,
            grid.radius_m,
            params.dt_yr,
            params.m,
            cell_area,
            params.deposition_g,
            params.deposition_passes,
        )
        limit_slopes(grid, h, is_land, r.stack, params.max_slope_deg)
        if diffuser is not None:
            h = diffuser.diffuse(h, is_land, t_diff)
        h[is_land] = np.maximum(h[is_land], params.min_land_height_m)
        if progress is not None:
            progress(it + 1, params.iterations)

    final = route(grid, h, is_land, params.epsilon_m, rainfall, params.rho8_seed)
    return ErosionResult(height=h, routing=final, lake_fill_m=lake)


def uplift_from_height(
    h_init: np.ndarray, land: np.ndarray, u_per_km_m_per_yr: float
) -> np.ndarray:
    """Uplift proportional to the painted height in absolute terms (D23).

    ``u_per_km_m_per_yr`` is the uplift for each km of painted height, so
    steady-state relief scales with what was painted: a 2.7 km plateau gets
    a third of the uplift of an 8 km ridge.
    """
    out: np.ndarray = np.where(land, u_per_km_m_per_yr * np.maximum(h_init, 0.0) / 1000.0, 0.0)
    return out
