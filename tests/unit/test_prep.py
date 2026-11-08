from pathlib import Path

import numpy as np

from globerode.config import load_config
from globerode.grid import Grid
from globerode.prep import prep_fields
from globerode.resample import equirect_to_cube_classes, read_levels
from globerode.testworlds import write_world


def test_prep_heights_respect_coast_and_curve(tmp_path: Path) -> None:
    cfg = load_config(write_world("mini_earth", tmp_path, width=256))
    grid = Grid(64, radius_m=cfg.radius_m)
    classes = equirect_to_cube_classes(grid, read_levels(cfg.input.levels, cfg), 8)
    f = prep_fields(grid, classes, cfg)
    land, h = f["land"], f["height"]
    assert (h[land] >= cfg.coast.min_land_height_m).all()
    assert (h[~land] <= cfg.coast.max_sea_height_m).all()
    assert h[land].max() < 7500 * 1.3  # curve top plus noise headroom
    # Near the coast land is low; far inland it is higher on average.
    d = f["coast_dist"]
    assert h[land & (d < 160e3)].mean() < h[land & (d > 300e3)].mean()
    # Near the coast the sea is shallow (shelf); far out it is deep.
    assert h[~land & (d > -100e3)].mean() > -300
    assert h[~land & (d < -400e3)].mean() < -3000


def test_prep_is_deterministic(tmp_path: Path) -> None:
    cfg = load_config(write_world("ring", tmp_path, width=128))
    grid = Grid(32, radius_m=cfg.radius_m)
    classes = equirect_to_cube_classes(grid, read_levels(cfg.input.levels, cfg), 8)
    a = prep_fields(grid, classes, cfg)["height"]
    b = prep_fields(grid, classes, cfg)["height"]
    assert np.array_equal(a, b)
