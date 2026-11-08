"""Coarse-to-fine erosion (DESIGN §7.7).

For each level N in ``resolution.levels``:

- prep at this level gives the target surface ``h_init`` (fine coast,
  uplift, hypsometry target) and the ocean;
- the starting surface is ``h_init`` at the first level, else the previous
  level's eroded land upsampled (bicubic), plus the noise octaves this level
  can show but the previous one couldn't;
- ``erode.iterations[level]`` iterations run, with erodibility, rainfall
  and uplift varied by smooth noise fields (D26).

After the last level, eroded land is optionally rescaled so each region's
mean height matches ``h_init`` (local height restore, D16).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from globerode.config import Config
from globerode.erode import ErosionParams, erode, uplift_from_height
from globerode.erode.restore import cap_peaks, local_restore
from globerode.erode.routing import Routing
from globerode.erode.variation import erosion_fields
from globerode.fields import Fields
from globerode.grid import Grid
from globerode.noise import fbm_on_grid, resolvable_wavelength_m
from globerode.prep import noise_spec, prep_fields
from globerode.resample import cube_to_cube, equirect_to_cube_classes
from globerode.smooth import Diffuser

Log = Callable[..., None]
RESTORE_ROUNDS = 3


def _params(cfg: Config, level: int) -> ErosionParams:
    e = cfg.erode
    return ErosionParams(
        k=e.k,
        m=e.m,
        dt_yr=e.dt_yr,
        kappa_m2_per_yr=e.kappa_m2_per_yr,
        iterations=e.iterations[level],
        epsilon_m=e.epsilon_m,
        min_land_height_m=cfg.coast.min_land_height_m,
        max_slope_deg=e.max_slope_deg,
    )


def run_cascade(cfg: Config, levels_img: np.ndarray, log: Log) -> tuple[Grid, Fields, Routing]:
    """Prep + erosion at every level. Returns the final grid, fields and routing.

    Final fields: class, land, coast_dist, noise, height (eroded land + ocean),
    h_init (prepared surface), lake_fill.
    """
    nclasses = len(cfg.input.classes)
    prev_grid: Grid | None = None
    prev_land_h: np.ndarray | None = None
    routing: Routing | None = None
    fields: Fields | None = None

    for level, n in enumerate(cfg.resolution.levels):
        grid = Grid(n, radius_m=cfg.radius_m)
        classes = equirect_to_cube_classes(grid, levels_img, nclasses)
        fields = prep_fields(grid, classes, cfg)
        land = fields["land"]
        h_init = fields["height"].copy()
        fields["h_init"] = h_init

        if prev_grid is None or prev_land_h is None:
            start = h_init
        else:
            up = cube_to_cube(prev_grid, prev_land_h, grid)
            band = fbm_on_grid(
                grid,
                noise_spec(cfg),
                min_wavelength_m=resolvable_wavelength_m(grid),
                max_wavelength_m=resolvable_wavelength_m(prev_grid),
            )
            nz = cfg.prep.noise
            up = up + (nz.amp_m + nz.amp_per_m * np.maximum(up, 0.0)) * band
            start = np.where(land, up, h_init)

        var = erosion_fields(grid, cfg)
        uplift = uplift_from_height(h_init, land, cfg.erode.uplift_per_km_m_per_yr)
        uplift *= var["uplift"]
        params = _params(cfg, level)
        log("erosion level start", level=level, n=n, iterations=params.iterations)

        def progress(it: int, total: int, level: int = level) -> None:
            if it == total or it % max(1, total // 5) == 0:
                log("erosion progress", level=level, iteration=it, of=total)

        result = erode(
            grid,
            start,
            land,
            uplift,
            params,
            Diffuser(grid),
            progress,
            erodibility=var["erodibility"],
            rainfall=var["rainfall"],
        )
        h = np.where(land, result.height, h_init)
        fields["lake_fill"] = result.lake_fill_m
        routing = result.routing
        fields["height"] = h
        prev_grid = grid
        prev_land_h = np.where(land, h, 0.0)  # sea at sea level for clean upsampling

    assert fields is not None and routing is not None and prev_grid is not None
    if cfg.erode.restore_heights == "local":
        # The peak cap removes height where hard rock stands high, lowering the
        # regional mean again; a few restore + cap rounds bring it back (D26).
        for _ in range(RESTORE_ROUNDS):
            fields["height"] = local_restore(
                prev_grid,
                fields["height"],
                fields["h_init"],
                fields["land"],
                cfg.erode.restore_sigma_km * 1000.0,
                cfg.coast.min_land_height_m,
            )
            fields["height"] = cap_peaks(
                fields["height"],
                fields["h_init"],
                fields["land"],
                cfg.erode.peak_cap_ratio,
                cfg.erode.peak_cap_soft,
            )
    return prev_grid, fields, routing
