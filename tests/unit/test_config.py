import tomllib
from pathlib import Path

import pytest

from globerode.config import ConfigError, apply_overrides, dump_toml, load_config

GOOD = """
name = "t"
[resolution]
output_width = 256
levels = [16, 64]
[input]
levels = "levels.png"
[[input.classes]]
value = 0
name = "deep"
kind = "ocean"
depth_m = -4000
[[input.classes]]
value = 255
name = "high"
kind = "land"
base = 1.0
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "w.toml"
    p.write_text(text)
    return p


def test_loads_and_resolves_paths(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, GOOD))
    assert cfg.input.levels == (tmp_path / "levels.png").resolve()
    assert cfg.resolution.levels == [16, 64]
    assert cfg.radius_m == 6_371_000.0
    assert set(cfg.class_by_value()) == {0, 255}


def test_overrides(tmp_path: Path) -> None:
    cfg = load_config(
        _write(tmp_path, GOOD),
        ["name=renamed", "resolution.output_width=512", "resolution.levels=[32, 128]"],
    )
    assert cfg.name == "renamed"
    assert cfg.resolution.output_width == 512


def test_override_needs_equals() -> None:
    with pytest.raises(ConfigError):
        apply_overrides({}, ["nonsense"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("resolution.output_width=250", "multiple of 8"),
        ("resolution.levels=[64, 16]", "increasing"),
        ("resolution.levels=[16, 32]", "output_width/4"),
        ("bogus=1", "bogus"),
        ("coast.mode=deltas", "coast.mode"),
    ],
)
def test_invalid_configs_give_readable_errors(tmp_path: Path, override: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, GOOD), [override])


def test_class_rules(tmp_path: Path) -> None:
    bad = GOOD.replace("base = 1.0", "depth_m = -5")
    with pytest.raises(ConfigError, match="needs 'base'"):
        load_config(_write(tmp_path, bad))
    only_land = GOOD.replace('kind = "ocean"\ndepth_m = -4000', 'kind = "land"\nbase = 0.1')
    with pytest.raises(ConfigError, match="land class and one ocean"):
        load_config(_write(tmp_path, only_land))


def test_missing_file_and_bad_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(_write(tmp_path, "x = = 1"))


def test_dump_round_trips(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, GOOD))
    text = dump_toml(cfg)
    again = load_config(_write(tmp_path, text))
    assert again == cfg
    assert tomllib.loads(text)["input"]["classes"][0]["name"] == "deep"


def test_preview_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, GOOD)).preview()
    assert cfg.resolution.output_width == 2048
    assert cfg.resolution.levels == [512]
