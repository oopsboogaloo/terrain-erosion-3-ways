"""Cascade-level checks on small grids."""

from pathlib import Path

import numpy as np
import pytest

from globerode.config import Config, load_config
from globerode.erode.cascade import run_cascade
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


def test_hypsometry_restore_matches_prepared_heights(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "mini_earth", "erode.iterations=[20, 10]")
    img = read_levels(cfg.input.levels, cfg)
    _, fields, _ = run_cascade(cfg, img, _noop)
    land = fields["land"]
    got = np.percentile(fields["height"][land], [10, 50, 90, 100])
    want = np.percentile(fields["h_init"][land], [10, 50, 90, 100])
    assert got == pytest.approx(want, rel=1e-3, abs=1.0)


def test_erosion_changes_the_surface_and_every_land_cell_drains(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "ring", "erode.iterations=[30, 20]")
    img = read_levels(cfg.input.levels, cfg)
    grid, fields, routing = run_cascade(cfg, img, _noop)
    land = fields["land"]
    assert not np.allclose(fields["height"][land], fields["h_init"][land])
    rec = routing.receiver[land]
    assert (rec >= 0).all()
    outlets = np.nonzero(land)[0][~land[rec]]
    assert routing.area[outlets].sum() == pytest.approx(
        grid.cell_area_m2[land].sum(dtype=np.float64), rel=1e-9
    )


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
