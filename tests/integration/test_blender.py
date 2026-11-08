"""Blender globe script: generation (fast) and headless render (slow, needs bpy)."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from globerode.blender import SCRIPT_NAME
from globerode.cli import main
from globerode.testworlds import write_world


def _run(tmp_path: Path, world: str = "mini_earth") -> Path:
    cfg = write_world(world, tmp_path / world, width=256)
    runs = tmp_path / "runs"
    assert main(["run", str(cfg), "--runs-dir", str(runs), "--quiet"]) in (0, 1)
    return sorted(runs.iterdir())[-1]


def test_script_is_written_with_settings(tmp_path: Path) -> None:
    run = _run(tmp_path)
    text = (run / SCRIPT_NAME).read_text()
    tree = ast.parse(text)  # valid Python
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "SETTINGS"
    )
    assert node.value is not None
    s = ast.literal_eval(node.value)
    assert s["height_exr"] == "outputs/height_m.exr"
    assert s["rivers_png"] == "outputs/mask_rivers.png"
    assert s["exaggeration"] == 20.0 and s["radius_m"] == 6_371_000.0
    assert (run / s["height_exr"]).exists()


@pytest.mark.slow
def test_headless_render_shows_a_globe_with_land_and_sea(tmp_path: Path) -> None:
    pytest.importorskip("bpy")
    run = _run(tmp_path, "mini_earth")
    out = tmp_path / "globe.png"
    size = 160
    assert (
        main(
            [
                "render",
                str(run),
                "--out",
                str(out),
                "--size",
                str(size),
                "--samples",
                "4",
                "--lon",
                "20",
                "--lat",
                "20",
            ]
        )
        == 0
    )
    img = np.asarray(Image.open(out).convert("RGB")).astype(float)
    lit = img.sum(axis=2) > 15
    assert lit.mean() > 0.2, "image is (nearly) blank"
    rows = np.nonzero(lit.any(axis=1))[0]
    diameter = rows[-1] - rows[0] + 1
    assert diameter == pytest.approx(0.8 * size, abs=6)  # ortho_scale 2.5 -> 80 % of frame
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    sea = lit & (b > r + 20) & (b > g)
    land = lit & (r + g > 2 * b)
    assert sea.mean() > 0.05 and land.mean() > 0.02


@pytest.mark.slow
def test_script_runs_standalone_and_saves_a_blend(tmp_path: Path) -> None:
    pytest.importorskip("bpy")
    run = _run(tmp_path, "ring")
    blend = tmp_path / "globe.blend"
    proc = subprocess.run(
        [sys.executable, str(run / SCRIPT_NAME), "--save", str(blend)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert blend.stat().st_size > 10_000
    assert json.loads(proc.stdout.strip().splitlines()[-1])["ok"] is True
