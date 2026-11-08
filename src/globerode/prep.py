"""Pre-processing: painted levels -> initial heights on the sphere (DESIGN §6, §8).

land:  class base -> masked smoothing -> coast ramp -> height curve -> + noise
ocean: class depth -> masked smoothing -> shelf ramp -> + floor noise
"""

from __future__ import annotations

import numpy as np

from globerode.coast import coast_distance
from globerode.config import Config
from globerode.curve import HeightCurve
from globerode.fields import Fields
from globerode.grid import Grid
from globerode.noise import NoiseSpec, fbm_on_grid, resolvable_wavelength_m
from globerode.ocean import ocean_floor
from globerode.smooth import coast_ramp, gaussian_smooth


def coast_distance_cap_m(cfg: Config) -> float:
    """Furthest distance from the coast any stage needs."""
    km = max(
        cfg.prep.coast_ramp_km,
        3.0 * cfg.prep.sigma_km,
        3.0 * cfg.ocean.sigma_km,
        cfg.ocean.shelf_width_km + cfg.ocean.slope_width_km,
        1.0,
    )
    return km * 1000.0


def noise_spec(cfg: Config) -> NoiseSpec:
    nz = cfg.prep.noise
    return NoiseSpec(
        seed=nz.seed, octaves=nz.octaves, base_m=nz.base_km * 1000.0, gain=nz.gain, warp=nz.warp
    )


def prep_fields(grid: Grid, classes: np.ndarray, cfg: Config) -> Fields:
    """Initial heights from per-cell class indices.

    Adds fields: ``class``, ``land``, ``coast_dist`` (m), ``noise``, ``height`` (m).
    Noise includes every octave the grid can resolve (>= 4 cells).
    """
    cls_cfg = cfg.input.classes
    is_land_class = np.array([c.kind == "land" for c in cls_cfg])
    base_of = np.array([c.base if c.base is not None else 0.0 for c in cls_cfg])
    depth_of = np.array([c.depth_m if c.depth_m is not None else 0.0 for c in cls_cfg])

    f = Fields(grid)
    f["class"] = classes
    land = is_land_class[classes]
    f["land"] = land
    f["coast_dist"] = coast_distance(grid, land, coast_distance_cap_m(cfg))
    f["noise"] = fbm_on_grid(grid, noise_spec(cfg), min_wavelength_m=resolvable_wavelength_m(grid))

    # Land.
    land_bases = [c.base for c in cls_cfg if c.base is not None]
    coast_base = cfg.prep.coast_base if cfg.prep.coast_base is not None else min(land_bases)
    smoothed = gaussian_smooth(grid, base_of[classes], land, cfg.prep.sigma_km * 1000.0)
    ramped = coast_ramp(smoothed, f["coast_dist"], coast_base, cfg.prep.coast_ramp_km * 1000.0)
    h_land = HeightCurve(cfg.prep.curve)(ramped)
    nz = cfg.prep.noise
    h_land = h_land + (nz.amp_m + nz.amp_per_m * h_land) * f["noise"]
    h_land = np.maximum(h_land, cfg.coast.min_land_height_m)

    # Ocean.
    sea = ~land
    painted = gaussian_smooth(grid, depth_of[classes], sea, cfg.ocean.sigma_km * 1000.0)
    oc = cfg.ocean
    h_sea = ocean_floor(
        painted,
        -f["coast_dist"].astype(np.float64),
        f["noise"].astype(np.float64),
        shelf_depth_m=oc.shelf_depth_m,
        shelf_width_m=oc.shelf_width_km * 1000.0,
        slope_width_m=oc.slope_width_km * 1000.0,
        floor_noise_m=oc.floor_noise_m,
        max_sea_height_m=cfg.coast.max_sea_height_m,
    )
    f["height"] = np.where(land, h_land, h_sea)
    return f
