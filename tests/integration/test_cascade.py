"""Cascade-level checks on small grids."""

from pathlib import Path

import numpy as np
import pytest

from globerode.config import Config, load_config
from globerode.erode.cascade import run_cascade
from globerode.erode.variation import erosion_fields
from globerode.prep import prep_fields
from globerode.resample import equirect_to_cube_classes, read_levels
from globerode.testworlds import write_world


def _cfg(tmp_path: Path, world: str = "mini_earth", *overrides: str) -> Config:
    return load_config(write_world(world, tmp_path, width=256), list(overrides))


def _noop(*_: object, **__: object) -> None:
    return None


def test_fine_level_keeps_the_fine_coast(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "mini_earth", "erode.iterations=[20, 10]")
    img = read_levels(cfg.input.levels, cfg)
    grid, fields, _ = run_cascade(cfg, img, _noop)
    assert grid.n == 64
    prep = prep_fields(grid, equirect_to_cube_classes(grid, img, 8), cfg)
    assert np.array_equal(fields["land"], prep["land"])
    h, land = fields["height"], fields["land"]
    assert (h[land] >= cfg.coast.min_land_height_m).all()
    assert np.array_equal(h[~land], prep["height"][~land])  # ocean untouched


def test_local_restore_keeps_regional_heights(tmp_path: Path) -> None:
    from globerode.erode.restore import block_average

    cfg = _cfg(tmp_path, "mini_earth", "erode.iterations=[20, 10]")
    img = read_levels(cfg.input.levels, cfg)
    grid, fields, _ = run_cascade(cfg, img, _noop)
    land = fields["land"].astype(np.float64)
    area = grid.cell_area_m2.astype(np.float64)
    w = area * land
    # Regional (8x8-cell block) mean land heights match the prepared surface.
    got = block_average(grid, fields["height"], w, 8)
    want = block_average(grid, fields["h_init"], w, 8)
    frac = block_average(grid, land, area, 8)
    inland = frac > 0.99
    assert np.median(np.abs(got[inland] - want[inland]) / want[inland]) < 0.1
    # The highest cells stay where the highest land was painted.
    h, h0 = fields["height"], fields["h_init"]
    top = h >= np.percentile(h[land > 0], 99.5)
    assert np.median(h0[top]) > np.percentile(h0[land > 0], 90)


def test_erosion_changes_the_surface_and_every_land_cell_drains(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "ring", "erode.iterations=[30, 20]")
    img = read_levels(cfg.input.levels, cfg)
    grid, fields, routing = run_cascade(cfg, img, _noop)
    land = fields["land"]
    assert not np.allclose(fields["height"][land], fields["h_init"][land])
    rec = routing.receiver[land]
    assert (rec >= 0).all()
    outlets = np.nonzero(land)[0][~land[rec]]
    rain = erosion_fields(grid, cfg)["rainfall"]  # area is rain-weighted (D26)
    want = (grid.cell_area_m2.astype(np.float64) * rain)[land].sum()
    assert routing.area[outlets].sum() == pytest.approx(want, rel=1e-9)


def test_single_level_and_disabled_erosion(tmp_path: Path) -> None:
    from globerode.cli import main

    cfg_path = write_world("ring", tmp_path / "w", width=64)
    runs = tmp_path / "runs"
    for extra in (
        ["--set", "erode.enabled=false"],
        ["--set", "resolution.levels=[16]", "--set", "erode.iterations=[3]"],
    ):
        assert main(["run", str(cfg_path), "--runs-dir", str(runs), "--quiet", *extra]) == 0
    disabled, single = sorted(runs.iterdir())
    assert not (disabled / "outputs/mask_rivers.png").exists()
    assert (single / "outputs/mask_rivers.png").exists()


def test_iterations_must_match_levels(tmp_path: Path) -> None:
    from globerode.config import ConfigError

    with pytest.raises(ConfigError, match="one count per level"):
        _cfg(tmp_path, "ring", "erode.iterations=[5]")
