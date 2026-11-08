"""The example config in the README stays valid."""

import re
from pathlib import Path

from globerode.config import load_config

README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_example_config_loads(tmp_path: Path) -> None:
    blocks = re.findall(r"```toml\n(.*?)```", README.read_text(), re.S)
    assert blocks, "README has no toml example"
    cfg_path = tmp_path / "world.toml"
    cfg_path.write_text(blocks[0])
    cfg = load_config(cfg_path)
    assert cfg.resolution.levels[-1] == cfg.resolution.output_width // 4
    assert {c.kind for c in cfg.input.classes} == {"land", "ocean"}
