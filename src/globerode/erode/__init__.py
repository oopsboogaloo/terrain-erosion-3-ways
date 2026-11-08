"""Erosion (DESIGN §7): iteration loop and uplift.

One iteration:
  1. route (Priority-Flood+eps): filled heights, receivers, stack, area
  2. h <- filled (pits fill instantly, D4); accumulate lake fill
  3. stream power (Braun-Willett, n = 1) with uplift
  4. hillslope diffusion
  5. coast clamp: land >= min_land_height
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from globerode.erode.routing import Routing, route
from globerode.erode.stream_power import stream_power_step
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
) -> ErosionResult:
    """Run ``params.iterations`` erosion iterations on land cells."""
    is_land = np.ascontiguousarray(land, np.bool_)
    h = np.array(height_m, dtype=np.float64)
    h[is_land] = np.maximum(h[is_land], params.min_land_height_m)
    erodibility = np.full(grid.ncells, params.k, dtype=np.float64)
    lake = np.zeros(grid.ncells, dtype=np.float64)
    t_diff = params.kappa_m2_per_yr * params.dt_yr / grid.radius_m**2
    if params.kappa_m2_per_yr > 0 and diffuser is None:
        diffuser = Diffuser(grid)

    for it in range(params.iterations):
        r = route(grid, h, is_land, params.epsilon_m)
        lake += r.filled - h
        h = r.filled
        stream_power_step(
            h,
            uplift_m_per_yr,
            erodibility,
            r.area,
            r.receiver,
            r.stack,
            is_land,
            grid.n,
            grid.radius_m,
            params.dt_yr,
            params.m,
        )
        if diffuser is not None:
            h = diffuser.diffuse(h, is_land, t_diff)
        h[is_land] = np.maximum(h[is_land], params.min_land_height_m)
        if progress is not None:
            progress(it + 1, params.iterations)

    final = route(grid, h, is_land, params.epsilon_m)
    return ErosionResult(height=h, routing=final, lake_fill_m=lake)


def uplift_from_height(h_init: np.ndarray, land: np.ndarray, u_max_m_per_yr: float) -> np.ndarray:
    """Uplift proportional to the painted height (ranges stay roughly as painted)."""
    top = float(h_init[land].max()) if land.any() else 1.0
    out: np.ndarray = np.where(land, u_max_m_per_yr * np.maximum(h_init, 0.0) / max(top, 1.0), 0.0)
    return out
