"""Blender integration (DESIGN §12.3): per-run globe script and headless renders."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from globerode.config import ConfigError

SCRIPT_NAME = "blender_globe.py"
_PLACEHOLDER = "SETTINGS: dict = {}  # globerode replaces this line with the run's settings"


def write_globe_script(run_dir: Path, settings: dict[str, Any]) -> Path:
    """Write ``blender_globe.py`` into the run directory with settings baked in."""
    template = files("globerode.blender").joinpath("globe_script.py").read_text()
    if _PLACEHOLDER not in template:
        raise RuntimeError("globe script template is missing its SETTINGS line")
    line = f"SETTINGS: dict = {json.dumps(settings, indent=4)}"
    path = run_dir / SCRIPT_NAME
    path.write_text(template.replace(_PLACEHOLDER, line))
    return path


def render(
    run_dir: Path,
    out: Path | None = None,
    size: int = 1024,
    samples: int = 32,
    lon: float = 0.0,
    lat: float = 20.0,
    timeout_s: float = 3600.0,
) -> dict[str, Any]:
    """Render the run's globe headless with the ``bpy`` module (optional extra)."""
    script = run_dir / SCRIPT_NAME
    if not script.exists():
        raise ConfigError(f"{script} not found")
    try:
        import bpy  # noqa: F401
    except ImportError:
        raise ConfigError(
            "rendering needs the bpy module: `uv sync --extra blender` "
            "(or run `blender -b -P blender_globe.py -- --render out.png`)"
        ) from None
    out = out or run_dir / "previews" / "render.png"
    cmd = [
        sys.executable,
        str(script),
        "--render",
        str(out),
        "--size",
        str(size),
        "--samples",
        str(samples),
        "--lon",
        str(lon),
        "--lat",
        str(lat),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout)[-2000:]
        raise RuntimeError(f"Blender render failed (exit {proc.returncode}):\n{tail}")
    return {"ok": True, "render": str(out)}
