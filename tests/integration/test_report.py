"""Report, `globerode report`, `globerode compare`, exit codes, polar consistency."""

import json
from pathlib import Path

import pytest

from globerode.cli import main
from globerode.testworlds import write_world


def _run(tmp_path: Path, world: str, width: int = 256, *extra: str) -> Path:
    cfg = write_world(world, tmp_path / world, width=width)
    runs = tmp_path / f"runs-{world}-{width}"
    code = main(["run", str(cfg), "--runs-dir", str(runs), "--quiet", *extra])
    assert code in (0, 1)
    return sorted(runs.iterdir())[-1]


def test_report_files_and_previews(tmp_path: Path) -> None:
    run = _run(tmp_path, "mini_earth")
    rep = json.loads((run / "report.json").read_text())
    names = {c["name"]: c["status"] for c in rep["checks"]}
    assert names["coast"] == "pass" and names["drainage"] == "pass"
    assert set(names) == {"coast", "seams", "max_height", "drainage", "hack", "horton"}
    for p in [
        "report.md",
        "state.npz",
        "previews/globe_north.png",
        "previews/globe_south.png",
        "previews/closeup_seam180.png",
        "previews/closeup_cubecorner.png",
        "previews/masks_overlay.png",
        "previews/curve.png",
        "previews/hist.png",
    ]:
        assert (run / p).is_file(), p
    assert "![globe_north](previews/globe_north.png)" in (run / "report.md").read_text()


def test_report_command_reproduces_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _run(tmp_path, "ring")
    before = json.loads((run / "report.json").read_text())
    (run / "report.json").unlink()
    capsys.readouterr()
    assert main(["report", str(run)]) in (0, 1)
    after = json.loads((run / "report.json").read_text())
    # state.npz stores float32 heights, so values match to ~7 digits.
    for b, a in zip(before["checks"], after["checks"], strict=True):
        assert (b["name"], b["status"]) == (a["name"], a["status"])
        assert a["value"] == pytest.approx(b["value"], rel=1e-5, nan_ok=True)


def test_failed_hard_check_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_world("ring", tmp_path / "w", width=128)
    capsys.readouterr()
    code = main(
        [
            "run",
            str(cfg),
            "--runs-dir",
            str(tmp_path / "r"),
            "--quiet",
            "--set",
            "report.hard_checks=['coast', 'max_height']",
            "--set",
            "report.max_height_ratio=[5.0, 6.0]",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ok"] is False and out["checks"]["max_height"] == "fail"
    assert (sorted((tmp_path / "r").iterdir())[-1] / "outputs/height_m.exr").exists()


def test_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _run(tmp_path, "ring")
    b = _run(tmp_path, "ring", 256, "--set", "erode.k=4e-6")
    capsys.readouterr()
    assert main(["compare", str(a), str(b)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert Path(out["image"]).is_file()
    assert "hack" in out["metrics"] and "b_minus_a" in out["metrics"]["hack"]


def test_polar_consistency(tmp_path: Path) -> None:
    """The same continent at the equator and at the pole gives matching statistics."""
    eq = json.loads((_run(tmp_path, "cap_equator", 512) / "report.json").read_text())
    po = json.loads((_run(tmp_path, "cap_pole", 512) / "report.json").read_text())
    he, hp = eq["metrics"]["heights"], po["metrics"]["heights"]
    assert hp["mean_m"] == pytest.approx(he["mean_m"], rel=0.1)
    assert hp["max_m"] == pytest.approx(he["max_m"], rel=0.1)
    de, dp = eq["metrics"]["drainage_density_per_km"], po["metrics"]["drainage_density_per_km"]
    assert dp == pytest.approx(de, rel=0.25)
    assert po["metrics"]["hack"]["h"] == pytest.approx(eq["metrics"]["hack"]["h"], abs=0.1)
