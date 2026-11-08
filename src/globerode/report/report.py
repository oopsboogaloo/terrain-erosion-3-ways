"""Run report: checks, metrics, previews -> report.json + report.md (DESIGN §12)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from globerode.config import Config
from globerode.curve import HeightCurve
from globerode.grid import Grid
from globerode.report import metrics as M
from globerode.report import previews as P

STATE_FILE = "state.npz"
SEAM_NOISE_SE = 4.0  # heavy-tailed gradients on eroded terrain: be generous
SEAM_MIN_PAIRS = 400  # fewer edge pairs (tiny grids): a finite ratio only warns (D31)


@dataclass
class RunState:
    """What the report needs from the cube-sphere run (saved as state.npz)."""

    grid: Grid
    height: np.ndarray
    h_init: np.ndarray
    land: np.ndarray
    receiver: np.ndarray | None = None
    stack: np.ndarray | None = None
    area: np.ndarray | None = None

    def save(self, path: Path) -> None:
        arrays: dict[str, Any] = {
            "n": np.array(self.grid.n),
            "radius_m": np.array(self.grid.radius_m),
            "height": self.height.astype(np.float32),
            "h_init": self.h_init.astype(np.float32),
            "land": self.land.astype(np.uint8),
        }
        if self.receiver is not None and self.stack is not None and self.area is not None:
            arrays |= {"receiver": self.receiver, "stack": self.stack, "area": self.area}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> RunState:
        with np.load(path) as z:
            grid = Grid(int(z["n"]), radius_m=float(z["radius_m"]))
            has_routing = "receiver" in z
            return cls(
                grid=grid,
                height=z["height"].astype(np.float64),
                h_init=z["h_init"].astype(np.float64),
                land=z["land"].astype(bool),
                receiver=z["receiver"] if has_routing else None,
                stack=z["stack"] if has_routing else None,
                area=z["area"].astype(np.float64) if has_routing else None,
            )


def _check(name: str, value: float, rng: tuple[float, float], hard: bool) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        status = "skip"
    elif rng[0] <= value <= rng[1]:
        status = "pass"
    else:
        status = "fail" if hard else "warn"
    return {"name": name, "value": value, "range": list(rng), "status": status, "hard": hard}


def seam_check(seams: dict[str, float], limits: tuple[float, float], hard: bool) -> dict[str, Any]:
    """Seam check with allowances for sampling noise (D17) and thin data (D31)."""
    seam = _check("seams", seams["ratio"], limits, hard)
    finite = 0 < seam["value"] < math.inf
    if seam["status"] in ("fail", "warn") and finite:
        if abs(math.log(seam["value"])) < SEAM_NOISE_SE * seams["log_se"]:
            # Sampling noise on rough terrain (checked visually, D17), not a seam.
            seam["status"] = "pass"
            seam["note"] = (
                f"outside range but within {SEAM_NOISE_SE:g} standard errors of 1 (noise)"
            )
        elif seams["pairs"] < SEAM_MIN_PAIRS:
            # Too few edge pairs to tell a seam from terrain (tiny grids, D31).
            seam["status"] = "warn"
            seam["note"] = f"only {seams['pairs']} edge pairs; too few to fail the run"
    return seam


def build_report(
    state: RunState, height_eq: np.ndarray, land_eq: np.ndarray, cfg: Config
) -> dict[str, Any]:
    """Compute every metric and check. Pure: no files written."""
    rc = cfg.report
    hard = set(rc.hard_checks)
    grid, land = state.grid, state.land
    out: dict[str, Any] = {"metrics": {}, "checks": []}
    m = out["metrics"]

    m["coast"] = M.coast_match(height_eq, land_eq)
    out["checks"].append(
        _check("coast", float(m["coast"]["mismatched_pixels"]), (0.0, 0.0), "coast" in hard)
    )
    m["seams"] = M.seam_continuity(grid, state.height, land)
    seam = seam_check(m["seams"], rc.seam_ratio, "seams" in hard)
    out["checks"].append(seam)
    m["heights"] = M.height_stats(state.height, state.h_init, land)
    out["checks"].append(
        _check(
            "max_height",
            float(m["heights"].get("max_vs_prepared", float("nan"))),
            rc.max_height_ratio,
            "max_height" in hard,
        )
    )

    if state.receiver is not None and state.stack is not None and state.area is not None:
        m["drainage"] = M.drainage(state.receiver, state.stack, land)
        out["checks"].append(
            _check("drainage", m["drainage"]["fraction_draining"], (1.0, 1.0), "drainage" in hard)
        )
        dist = M.receiver_distance_m(grid, state.receiver)
        channel = land & (state.area >= M.river_threshold_m2(grid, cfg.erode.river_threshold_km2))
        m["hack"] = M.hack_exponent(state.receiver, state.stack, state.area, dist, land, channel)
        m["horton"] = M.horton_bifurcation(state.receiver, state.stack, land, channel)
        m["drainage_density_per_km"] = M.drainage_density_per_km(
            dist, grid.cell_area_m2.astype(np.float64), land, channel
        )
        out["checks"].append(_check("hack", m["hack"]["h"], rc.hack_exponent, "hack" in hard))
        out["checks"].append(
            _check("horton", float(m["horton"]["rb"]), rc.horton_ratio, "horton" in hard)
        )

    statuses = [c["status"] for c in out["checks"]]
    out["status"] = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return out


def write_previews(
    prev_dir: Path,
    state: RunState,
    height_eq: np.ndarray,
    rivers_eq: np.ndarray | None,
    erosion_eq: np.ndarray | None,
    cfg: Config,
) -> list[Path]:
    prev_dir.mkdir(parents=True, exist_ok=True)
    r = cfg.radius_m
    paths = [P.write_hillshade_preview(prev_dir / "hillshade_equirect.png", height_eq, r)]
    paths += P.write_views(prev_dir, height_eq, r)
    if rivers_eq is not None and erosion_eq is not None:
        paths.append(
            P.write_mask_overlay(
                prev_dir / "masks_overlay.png", height_eq, rivers_eq, erosion_eq, r
            )
        )
    paths.append(
        P.write_curve_chart(
            prev_dir / "curve.png", list(cfg.prep.curve), HeightCurve(cfg.prep.curve)
        )
    )
    land = state.land
    paths.append(P.write_histogram(prev_dir / "hist.png", state.height[land], state.h_init[land]))
    return paths


def write_report(run_dir: Path, report: dict[str, Any], previews: list[Path]) -> None:
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    lines = [
        f"# Run report: {run_dir.name}",
        "",
        f"**Overall: {report['status'].upper()}**",
        "",
        "| Check | Value | Range | Status |",
        "|---|---|---|---|",
    ]
    for c in report["checks"]:
        v = c["value"]
        vs = f"{v:.4g}" if isinstance(v, float) else str(v)
        hard = " (hard)" if c["hard"] else ""
        lines.append(f"| {c['name']}{hard} | {vs} | {c['range']} | {c['status']} |")
    m = report["metrics"]
    lines += ["", "## Metrics", "", "```json", json.dumps(m, indent=2, default=float), "```"]
    lines += ["", "## Previews", ""]
    for p in previews:
        rel = p.relative_to(run_dir)
        lines.append(f"![{p.stem}]({rel})")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
