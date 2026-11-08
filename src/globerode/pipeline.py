"""Runs the stages in order and writes a run directory (DESIGN §1, §10).

Stages: resample -> erode (prep + erosion at each cascade level; or prep only
when erosion is disabled) -> export (heights, masks) -> previews.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from globerode import __version__
from globerode.config import Config, dump_toml
from globerode.erode.cascade import run_cascade
from globerode.export import relock_coast, write_heights, write_mask_png8, write_mask_png16
from globerode.grid import Grid
from globerode.prep import prep_fields
from globerode.report.previews import write_hillshade_preview
from globerode.resample import (
    cube_to_equirect,
    equirect_majority,
    equirect_to_cube_classes,
    read_levels,
)

BYTES_PER_CELL = 45  # DESIGN §4 peak estimate
BYTES_PER_OUTPUT_PIXEL = 4 + 4 + 2 + 2 + 8  # height f32, relock f32, classes, uint16, temp


class ResourceError(RuntimeError):
    """The machine is too small for the requested run (CLI exit code 3)."""


# --- logging ----------------------------------------------------------------------------


class RunLog:
    """JSON-lines log file plus short human-readable lines on stderr."""

    def __init__(self, path: Path | None, quiet: bool = False) -> None:
        self.path = path
        self.quiet = quiet
        self.timings: dict[str, float] = {}

    def event(self, message: str, **data: Any) -> None:
        record = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "message": message}
        record.update(data)
        if self.path is not None:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        if not self.quiet:
            extra = " ".join(f"{k}={v}" for k, v in data.items())
            print(f"[globerode] {message} {extra}".rstrip(), file=sys.stderr)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.event("stage start", stage=name)
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.timings[name] = round(dt, 3)
        self.event("stage done", stage=name, seconds=round(dt, 3))


# --- estimates ------------------------------------------------------------------------------


def estimate(cfg: Config) -> dict[str, Any]:
    """Rough peak memory and output disk use for a run."""
    n = cfg.resolution.levels[-1]
    w = cfg.resolution.output_width
    cells = 6 * n * n
    pixels = w * (w // 2)
    mem = cells * BYTES_PER_CELL + pixels * BYTES_PER_OUTPUT_PIXEL
    disk = pixels * (4 * ("exr32" in cfg.output.formats) + 4 * ("tiff32" in cfg.output.formats))
    disk += pixels * 2 * sum(f in cfg.output.formats for f in ("png16", "tiff16"))
    return {
        "n": n,
        "cells": cells,
        "output_size": [w, w // 2],
        "peak_memory_gb": round(mem / 1e9, 3),
        "output_disk_gb_max": round(disk / 1e9, 3),
        "available_memory_gb": round(available_memory_bytes() / 1e9, 3),
    }


def available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    pages = os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") else 0
    return int(pages * os.sysconf("SC_PAGE_SIZE")) if pages else 1 << 62


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --- the pipeline ---------------------------------------------------------------------------


@dataclass
class RunResult:
    run_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)


def make_run_dir(runs_dir: Path, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    base = runs_dir / f"{stamp}-{name}"
    path = base
    k = 2
    while path.exists():
        path = Path(f"{base}-{k}")
        k += 1
    path.mkdir(parents=True)
    return path


def run(cfg: Config, run_dir: Path, quiet: bool = False) -> RunResult:
    """Run the whole pipeline for ``cfg``, writing everything into ``run_dir``."""
    est = estimate(cfg)
    if est["peak_memory_gb"] * 1e9 > 0.9 * available_memory_bytes():
        raise ResourceError(
            f"run needs ~{est['peak_memory_gb']} GB but only "
            f"{est['available_memory_gb']} GB is available"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.toml").write_text(dump_toml(cfg))
    log = RunLog(run_dir / "log.jsonl", quiet=quiet)
    log.event("run start", name=cfg.name, n=est["n"], output_width=cfg.resolution.output_width)
    t_start = time.perf_counter()

    classes_cfg = cfg.input.classes
    nclasses = len(classes_cfg)
    is_land_class = np.array([c.kind == "land" for c in classes_cfg])
    n = cfg.resolution.levels[-1]

    with log.stage("resample"):
        levels_img = read_levels(cfg.input.levels, cfg)

    mask_info: dict[str, Any] = {}
    if cfg.erode.enabled:
        with log.stage("erode"):
            grid, fields, routing = run_cascade(cfg, levels_img, log.event)
    else:
        with log.stage("prep"):
            grid = Grid(n, radius_m=cfg.radius_m)
            classes = equirect_to_cube_classes(grid, levels_img, nclasses)
            fields = prep_fields(grid, classes, cfg)
            routing = None

    out_dir = run_dir / "outputs"
    with log.stage("export"):
        width = cfg.resolution.output_width
        height_eq = cube_to_equirect(grid, fields["height"], width)
        land_out = is_land_class[equirect_majority(levels_img, nclasses, width)]
        height_eq = relock_coast(height_eq, land_out, cfg.coast)
        written = write_heights(out_dir, height_eq, cfg.output.formats, cfg.output.png_range_m)
        written.append(write_mask_png8(out_dir / "mask_land.png", land_out))
        if routing is not None:
            written += _write_erosion_masks(
                grid, fields, routing.area, land_out, cfg, out_dir, mask_info
            )

    with log.stage("previews"):
        prev_dir = run_dir / "previews"
        prev_dir.mkdir(exist_ok=True)
        write_hillshade_preview(prev_dir / "hillshade_equirect.png", height_eq, cfg.radius_m)

    total = round(time.perf_counter() - t_start, 3)
    land_px = height_eq[land_out]
    sea_px = height_eq[~land_out]
    metadata = {
        "globerode_version": __version__,
        "git_commit": _git_commit(),
        "name": cfg.name,
        "n": n,
        "output_size": [width, width // 2],
        "radius_m": cfg.radius_m,
        "height_range_m": [float(height_eq.min()), float(height_eq.max())],
        "land_fraction": float(land_out.mean()),
        "land_height_m": {
            "max": float(land_px.max()) if land_px.size else None,
            "mean": float(land_px.mean()) if land_px.size else None,
        },
        "sea_depth_m": {"min": float(sea_px.min()) if sea_px.size else None},
        "png16_mapping_m": list(cfg.output.png_range_m),
        "masks": mask_info,
        "timings_s": dict(log.timings) | {"total": total},
        "peak_rss_gb": _peak_rss_gb(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {
        "ok": True,
        "run_dir": str(run_dir),
        "outputs": [str(p.relative_to(run_dir)) for p in written],
        "height_range_m": metadata["height_range_m"],
        "land_fraction": metadata["land_fraction"],
        "seconds": total,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.event("run done", seconds=total)
    return RunResult(run_dir=run_dir, summary=summary)


def _write_erosion_masks(
    grid: Grid,
    fields: Any,
    area_m2: np.ndarray,
    land_out: np.ndarray,
    cfg: Config,
    out_dir: Path,
    info: dict[str, Any],
) -> list[Path]:
    """Flow (log drainage area), rivers (area threshold) and erosion depth masks."""
    width = cfg.resolution.output_width
    land = fields["land"]
    cell_min = float(grid.cell_area_m2.min())
    log_a = np.log10(np.where(land, np.maximum(area_m2, cell_min), cell_min))
    log_a_eq = cube_to_equirect(grid, log_a, width)
    lo = float(np.log10(cell_min))
    hi = float(log_a[land].max()) if land.any() else lo + 1.0
    flow = np.where(land_out, (log_a_eq - lo) / max(hi - lo, 1e-9), 0.0)
    thresh = cfg.erode.river_threshold_km2 * 1e6
    rivers = land_out & (log_a_eq >= np.log10(thresh))
    eroded = np.maximum(fields["h_init"] - fields["height"], 0.0) * land
    eroded_eq = cube_to_equirect(grid, eroded, width)
    emax = float(eroded.max()) if eroded.size else 0.0
    erosion = np.where(land_out, eroded_eq / max(emax, 1e-9), 0.0)
    info.update(
        {
            "flow": {"log10_area_m2_range": [lo, hi]},
            "rivers": {"threshold_km2": cfg.erode.river_threshold_km2},
            "erosion": {"max_depth_m": emax},
        }
    )
    return [
        write_mask_png16(out_dir / "mask_flow.png", flow),
        write_mask_png8(out_dir / "mask_rivers.png", rivers),
        write_mask_png16(out_dir / "mask_erosion.png", erosion),
    ]


def _peak_rss_gb() -> float | None:
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    except ImportError:
        return None
