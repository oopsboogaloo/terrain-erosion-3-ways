"""End-to-end runs of the CLI on every test world at small size."""

import json
from pathlib import Path

import numpy as np
import OpenEXR
import pytest
from PIL import Image

from globerode.cli import main
from globerode.testworlds import WORLDS

WIDTH = 256  # output 256x128, N = 64


def _run(tmp_path: Path, world: str) -> tuple[int, Path]:
    cfg = tmp_path / world / "world.toml"
    if not cfg.exists():
        assert main(["testworld", world, str(tmp_path / world), "--width", str(WIDTH)]) == 0
    runs = tmp_path / "runs"
    return main(["run", str(cfg), "--runs-dir", str(runs), "--quiet"]), runs


def _latest(runs: Path) -> Path:
    return sorted(runs.iterdir())[-1]


def _read_exr(path: Path) -> np.ndarray:
    with OpenEXR.File(str(path)) as f:
        return np.array(f.channels()["Y"].pixels)


@pytest.mark.parametrize("world", list(WORLDS))
def test_run_writes_all_outputs_with_exact_coast(tmp_path: Path, world: str) -> None:
    code, runs = _run(tmp_path, world)
    assert code == 0
    run_dir = _latest(runs)
    for rel in [
        "config.resolved.toml",
        "log.jsonl",
        "metadata.json",
        "summary.json",
        "outputs/height_m.exr",
        "outputs/height16.png",
        "outputs/mask_land.png",
        "previews/hillshade_equirect.png",
    ]:
        assert (run_dir / rel).is_file(), rel

    h = _read_exr(run_dir / "outputs/height_m.exr")
    assert h.shape == (WIDTH // 2, WIDTH)
    levels = np.array(Image.open(tmp_path / world / "levels.png"))
    land_in = levels >= 90  # land classes in the test-world palette
    assert np.array_equal(h > 0, land_in), "coast must match the input exactly"
    assert (h[land_in] >= 1.0).all() and (h[~land_in] <= -1.0).all()

    png = np.array(Image.open(run_dir / "outputs/height16.png"))
    assert png.dtype == np.uint16 and png.shape == h.shape

    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["n"] == WIDTH // 4
    assert set(meta["timings_s"]) >= {"resample", "prep", "export", "previews", "total"}


def test_stdout_is_one_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["testworld", "ring", str(tmp_path / "w"), "--width", "64"]) == 0
    capsys.readouterr()
    code = main(
        ["run", str(tmp_path / "w/world.toml"), "--runs-dir", str(tmp_path / "r"), "--quiet"]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["ok"] is True
    assert Path(out["run_dir"]).is_dir()


def test_runs_are_deterministic(tmp_path: Path) -> None:
    code, runs = _run(tmp_path, "mini_earth")
    first = _latest(runs)
    code2, _ = _run(tmp_path, "mini_earth")
    second = _latest(runs)
    assert code == code2 == 0 and first != second
    for rel in ["outputs/height_m.exr", "outputs/height16.png", "outputs/mask_land.png"]:
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), rel


def test_bad_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", str(tmp_path / "missing.toml"), "--runs-dir", str(tmp_path)])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_unknown_pixel_values_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["testworld", "ring", str(tmp_path / "w"), "--width", "64"])
    img = np.array(Image.open(tmp_path / "w/levels.png"))
    img[3, 3] = 77
    Image.fromarray(img).save(tmp_path / "w/levels.png")
    capsys.readouterr()
    code = main(
        ["run", str(tmp_path / "w/world.toml"), "--runs-dir", str(tmp_path / "r"), "--quiet"]
    )
    assert code == 2
    assert "77 (1 px)" in json.loads(capsys.readouterr().out)["error"]


def test_estimate_and_preview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["testworld", "ring", str(tmp_path / "w"), "--width", "64"])
    capsys.readouterr()
    assert main(["estimate", str(tmp_path / "w/world.toml"), "--preview"]) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["n"] == 512 and est["output_size"] == [2048, 1024]
    assert est["peak_memory_gb"] > 0


def test_set_override_changes_output_size(tmp_path: Path) -> None:
    main(["testworld", "ring", str(tmp_path / "w"), "--width", "64"])
    code = main(
        [
            "run",
            str(tmp_path / "w/world.toml"),
            "--runs-dir",
            str(tmp_path / "r"),
            "--quiet",
            "--set",
            "resolution.output_width=128",
            "--set",
            "resolution.levels=[32]",
        ]
    )
    assert code == 0
    h = _read_exr(_latest(tmp_path / "r") / "outputs/height_m.exr")
    assert h.shape == (64, 128)
