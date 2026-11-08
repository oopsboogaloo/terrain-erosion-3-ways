"""`globerode report` and `globerode compare` (DESIGN §11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import OpenEXR
from PIL import Image, ImageDraw

from globerode.config import ConfigError, load_config
from globerode.report.report import STATE_FILE, RunState, build_report, write_previews, write_report


def _read_exr(path: Path) -> np.ndarray:
    with OpenEXR.File(str(path)) as f:
        out: np.ndarray = np.array(f.channels()["Y"].pixels, dtype=np.float32)
    return out


def rerun_report(run_dir: Path) -> dict[str, Any]:
    """Recompute report.json / report.md / previews from a run directory."""
    cfg_path = run_dir / "config.resolved.toml"
    state_path = run_dir / STATE_FILE
    exr = run_dir / "outputs" / "height_m.exr"
    for p in (cfg_path, state_path, exr):
        if not p.exists():
            raise ConfigError(f"{p} not found (the run needs exr32 output and save_state)")
    cfg = load_config(cfg_path)
    state = RunState.load(state_path)
    height_eq = _read_exr(exr)
    land_eq = np.array(Image.open(run_dir / "outputs" / "mask_land.png")) > 0
    rivers = run_dir / "outputs" / "mask_rivers.png"
    erosion = run_dir / "outputs" / "mask_erosion.png"
    rivers_eq = np.array(Image.open(rivers)) > 0 if rivers.exists() else None
    erosion_eq = np.array(Image.open(erosion)) / 65535.0 if erosion.exists() else None
    report = build_report(state, height_eq, land_eq, cfg)
    previews = write_previews(run_dir / "previews", state, height_eq, rivers_eq, erosion_eq, cfg)
    write_report(run_dir, report, previews)
    return {
        "ok": report["status"] != "fail",
        "report_status": report["status"],
        "checks": {c["name"]: c["status"] for c in report["checks"]},
    }


def _flat_metrics(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in report["checks"]:
        if isinstance(c["value"], (int, float)):
            out[c["name"]] = float(c["value"])
    dd = report["metrics"].get("drainage_density_per_km")
    if isinstance(dd, (int, float)):
        out["drainage_density_per_km"] = float(dd)
    for k in ("mean_m", "max_m"):
        v = report["metrics"].get("heights", {}).get(k)
        if isinstance(v, (int, float)):
            out[f"height_{k}"] = float(v)
    return out


def compare_runs(run_a: Path, run_b: Path, out: Path | None = None) -> dict[str, Any]:
    """Metric differences and a side-by-side hillshade image of two runs."""
    reports = []
    for r in (run_a, run_b):
        p = r / "report.json"
        if not p.exists():
            raise ConfigError(f"{p} not found")
        reports.append(json.loads(p.read_text()))
    ma, mb = (_flat_metrics(x) for x in reports)
    diff = {
        k: {"a": ma[k], "b": mb[k], "b_minus_a": mb[k] - ma[k]} for k in sorted(set(ma) & set(mb))
    }
    imgs = [
        Image.open(r / "previews" / "hillshade_equirect.png").convert("RGB") for r in (run_a, run_b)
    ]
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + 40
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)
    y = 0
    for name, im in zip((run_a.name, run_b.name), imgs, strict=True):
        d.text((4, y + 2), name, fill="black")
        sheet.paste(im, (0, y + 20))
        y += im.height + 20
    out = out or run_b.parent / f"compare-{run_a.name}-vs-{run_b.name}.png"
    sheet.save(out)
    return {"ok": True, "image": str(out), "metrics": diff}
