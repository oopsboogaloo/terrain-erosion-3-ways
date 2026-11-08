from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from globerode.config import load_config
from globerode.testworlds import CLASSES, WORLDS, make_levels, write_world

VALUES = {int(c["value"]) for c in CLASSES}  # type: ignore[call-overload]


@pytest.mark.parametrize("name", list(WORLDS))
def test_world_uses_known_classes_and_has_land_and_sea(name: str) -> None:
    img = make_levels(name, 256)
    assert img.shape == (128, 256)
    assert set(np.unique(img).tolist()) <= VALUES
    assert (img >= 90).any() and (img < 90).any()


def test_cap_pole_covers_north_pole_only() -> None:
    img = make_levels("cap_pole", 256)
    assert (img[0] >= 90).all()
    assert (img[-1] < 90).all()


def test_seam_world_wraps_across_180() -> None:
    img = make_levels("seam", 256)
    mid = img.shape[0] // 2 - 8
    assert img[mid, 0] >= 90 and img[mid, -1] >= 90
    assert img[mid, 128] < 90


def test_write_world_produces_loadable_config(tmp_path: Path) -> None:
    cfg_path = write_world("mini_earth", tmp_path, width=128)
    cfg = load_config(cfg_path)
    assert cfg.resolution.output_width == 128
    assert cfg.resolution.levels == [8, 32]
    assert Image.open(cfg.input.levels).size == (128, 64)


def test_bad_inputs() -> None:
    with pytest.raises(ValueError, match="unknown test world"):
        make_levels("atlantis", 64)
    with pytest.raises(ValueError, match="multiple of 8"):
        make_levels("ring", 60)
