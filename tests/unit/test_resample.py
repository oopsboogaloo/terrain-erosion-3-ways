from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from globerode.config import Config, ConfigError, load_config
from globerode.grid import Grid
from globerode.resample import (
    cube_to_equirect,
    equirect_majority,
    equirect_to_cube_classes,
    read_levels,
)
from globerode.testworlds import make_levels, write_world


def _pixel_xyz(width: int) -> np.ndarray:
    h = width // 2
    lon = np.radians((np.arange(width) + 0.5) / width * 360 - 180)
    lat = np.radians(90 - (np.arange(h) + 0.5) / h * 180)
    lo, la = np.meshgrid(lon, lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


@pytest.mark.parametrize("n", [16, 48])
def test_smooth_field_round_trip_has_no_seams(n: int) -> None:
    grid = Grid(n)
    p = grid.xyz()
    field = p[:, 0] * p[:, 2] + 0.5 * p[:, 1]  # Y21-like plus a Y11 term
    out = cube_to_equirect(grid, field, 4 * n)
    q = _pixel_xyz(4 * n)
    err = out - (q[..., 0] * q[..., 2] + 0.5 * q[..., 1])
    cell = (np.pi / 2) / n
    assert np.abs(err).max() < 0.5 * cell**2
    # No jumps: neighbouring pixel errors differ by far less than the field step.
    assert np.abs(np.diff(err, axis=1)).max() < 0.1 * cell
    assert np.abs(np.diff(err, axis=0)).max() < 0.1 * cell


def test_constant_field_is_exact() -> None:
    grid = Grid(8)
    out = cube_to_equirect(grid, np.full(grid.ncells, 3.25), 32)
    assert np.allclose(out, 3.25)


def test_categorical_cube_matches_geometry() -> None:
    grid = Grid(32)
    width = 512
    img = make_levels("cap_equator", width)
    land = (img >= 90).astype(np.int16)
    cls = equirect_to_cube_classes(grid, land, 2)
    p = grid.xyz()
    ang = np.degrees(np.arccos(np.clip(p[:, 0], -1, 1)))
    cell_deg = 90 / 32
    assert (cls[ang < 40 - 2 * cell_deg] == 1).all()
    assert (cls[ang > 40 + 2 * cell_deg] == 0).all()


def test_categorical_handles_poles_and_seam() -> None:
    grid = Grid(16)
    for name in ["cap_pole", "seam"]:
        img = (make_levels(name, 256) >= 90).astype(np.int16)
        cls = equirect_to_cube_classes(grid, img, 2)
        lon, lat = grid.lonlat_deg()
        if name == "cap_pole":
            assert (cls[lat > 60] == 1).all()
            assert (cls[lat < 40] == 0).all()
        else:
            assert (cls[(np.abs(lon) > 165) & (np.abs(lat - 10) < 15)] == 1).all()


def test_equirect_majority_identity_and_downsize() -> None:
    img = (make_levels("islands", 512) >= 90).astype(np.int16)
    same = equirect_majority(img, 2, 512)
    assert np.array_equal(same, img)
    small = equirect_majority(img, 2, 256)
    assert small.shape == (128, 256)
    assert abs(small.mean() - img.mean()) < 0.01


def _cfg(tmp_path: Path) -> Config:
    return load_config(write_world("ring", tmp_path, width=64))


def test_read_levels_maps_classes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cls = read_levels(cfg.input.levels, cfg)
    assert cls.shape == (32, 64)
    assert cls.min() >= 0 and cls.max() < len(cfg.input.classes)


def test_read_levels_rejects_unknown_values(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    img = np.array(Image.open(cfg.input.levels))
    img[0, :3] = 7
    Image.fromarray(img).save(cfg.input.levels)
    with pytest.raises(ConfigError, match=r"7 \(3 px\)"):
        read_levels(cfg.input.levels, cfg)


def test_read_levels_rejects_bad_shape_and_mode(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    Image.fromarray(np.zeros((32, 32), np.uint8)).save(cfg.input.levels)
    with pytest.raises(ConfigError, match="2:1"):
        read_levels(cfg.input.levels, cfg)
    Image.fromarray(np.zeros((32, 64, 3), np.uint8)).save(cfg.input.levels)
    with pytest.raises(ConfigError, match="mode"):
        read_levels(cfg.input.levels, cfg)
    with pytest.raises(ConfigError, match="not found"):
        read_levels(tmp_path / "missing.png", cfg)


def test_reads_16bit_levels(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    img = np.array(Image.open(cfg.input.levels)).astype(np.uint16)
    Image.fromarray(img).save(cfg.input.levels)
    assert Image.open(cfg.input.levels).mode == "I;16"
    assert read_levels(cfg.input.levels, cfg).shape == (32, 64)
