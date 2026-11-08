"""Synthetic test worlds (DESIGN §13.2).

Each world is an equirectangular levels image plus a config, generated from
simple geometry in code, so no hand-painted fixtures are needed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

# Shared class set: 3 ocean levels, 5 land levels.
CLASSES: list[dict[str, object]] = [
    {"value": 0, "name": "deep", "kind": "ocean", "depth_m": -4500.0},
    {"value": 30, "name": "ocean", "kind": "ocean", "depth_m": -2500.0},
    {"value": 60, "name": "shelf", "kind": "ocean", "depth_m": -150.0},
    {"value": 90, "name": "coastal", "kind": "land", "base": 0.05},
    {"value": 120, "name": "lowland", "kind": "land", "base": 0.15},
    {"value": 160, "name": "midland", "kind": "land", "base": 0.4},
    {"value": 210, "name": "highland", "kind": "land", "base": 0.7},
    {"value": 255, "name": "peak", "kind": "land", "base": 1.0},
]
V = {str(c["name"]): int(c["value"]) for c in CLASSES}  # type: ignore[call-overload]


def _lonlat_grid(width: int) -> tuple[np.ndarray, np.ndarray]:
    h = width // 2
    lon = np.radians((np.arange(width) + 0.5) / width * 360.0 - 180.0)
    lat = np.radians(90.0 - (np.arange(h) + 0.5) / h * 180.0)
    return np.meshgrid(lon, lat)


def _unit(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _angle_deg(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Great-circle angle (degrees) from (lon0, lat0) in degrees."""
    c = _unit(np.radians(np.array(lon0)), np.radians(np.array(lat0)))
    p = _unit(lon, lat)
    out: np.ndarray = np.degrees(np.arccos(np.clip(p @ c, -1.0, 1.0)))
    return out


def _azimuth(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lo0, la0 = np.radians(lon0), np.radians(lat0)
    y = np.sin(lon - lo0) * np.cos(lat)
    x = np.cos(la0) * np.sin(lat) - np.sin(la0) * np.cos(lat) * np.cos(lon - lo0)
    out: np.ndarray = np.arctan2(y, x)
    return out


def _continent(img: np.ndarray, inland_deg: np.ndarray) -> None:
    """Paint levels by distance inland (degrees); ocean levels by distance offshore."""
    img[inland_deg > -8] = np.maximum(img[inland_deg > -8], V["ocean"])
    img[inland_deg > -3] = np.maximum(img[inland_deg > -3], V["shelf"])
    for thresh, name in [(0, "coastal"), (2, "lowland"), (6, "midland"), (11, "highland")]:
        mask = inland_deg > thresh
        img[mask] = np.maximum(img[mask], V[name])


def _cap(lon0: float, lat0: float, radius: float) -> Callable[[int], np.ndarray]:
    def make(width: int) -> np.ndarray:
        lon, lat = _lonlat_grid(width)
        img = np.full(lon.shape, V["deep"], dtype=np.uint8)
        _continent(img, radius - _angle_deg(lon, lat, lon0, lat0))
        return img

    return make


def _islands(width: int) -> np.ndarray:
    lon, lat = _lonlat_grid(width)
    img = np.full(lon.shape, V["deep"], dtype=np.uint8)
    for k in range(12):
        lon0 = -60.0 + 11.0 * k
        lat0 = -10.0 + 8.0 * np.sin(k)
        radius = 1.5 + 2.5 * ((k * 7) % 5) / 4
        inland = radius - _angle_deg(lon, lat, lon0, lat0)
        img[inland > -2] = np.maximum(img[inland > -2], V["shelf"])
        img[inland > 0] = np.maximum(img[inland > 0], V["lowland"])
        img[inland > radius * 0.6] = np.maximum(img[inland > radius * 0.6], V["midland"])
    return img


def _ring(width: int) -> np.ndarray:
    lon, lat = _lonlat_grid(width)
    img = np.full(lon.shape, V["deep"], dtype=np.uint8)
    ang = _angle_deg(lon, lat, 60.0, -20.0)
    inland = np.minimum(ang - 18.0, 36.0 - ang)  # ring between 18 and 36 deg
    _continent(img, inland)
    img[ang < 15] = V["shelf"]  # inner sea: shallow
    return img


def _mini_earth(width: int) -> np.ndarray:
    lon, lat = _lonlat_grid(width)
    img = np.full(lon.shape, V["deep"], dtype=np.uint8)
    # (lon0, lat0, radius, wobble amplitude, wobble seed)
    continents = [
        (-100.0, 40.0, 32.0, 7.0, 1),
        (-60.0, -20.0, 25.0, 6.0, 2),
        (20.0, 20.0, 38.0, 8.0, 3),
        (130.0, -25.0, 18.0, 4.0, 4),
        (0.0, -80.0, 15.0, 3.0, 5),
    ]
    for lon0, lat0, radius, amp, seed in continents:
        az = _azimuth(lon, lat, lon0, lat0)
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, 2 * np.pi, 4)
        wobble = sum(amp / (k + 1) * np.sin((k + 2) * az + phases[k]) for k in range(4))
        _continent(img, radius + wobble - _angle_deg(lon, lat, lon0, lat0))
    # A mountain range: narrow band along a great circle in the big continent.
    ridge = np.abs(_angle_deg(lon, lat, 20.0, -70.0) - 90.0)  # band 90 deg from a pole point
    in_big = _angle_deg(lon, lat, 20.0, 20.0) < 28.0
    img[(ridge < 2.5) & in_big] = V["peak"]
    return img


WORLDS: dict[str, Callable[[int], np.ndarray]] = {
    "cap_equator": _cap(0.0, 0.0, 40.0),
    "cap_pole": _cap(0.0, 90.0, 40.0),
    "seam": _cap(180.0, 10.0, 35.0),
    "corner": _cap(45.0, 35.264, 30.0),  # centred on the cube corner (1,1,1)
    "islands": _islands,
    "ring": _ring,
    "mini_earth": _mini_earth,
}


def make_levels(name: str, width: int) -> np.ndarray:
    """Levels image (uint8, shape (width/2, width)) for a named world."""
    if name not in WORLDS:
        raise ValueError(f"unknown test world {name!r}; choose from {', '.join(WORLDS)}")
    if width % 8 or width < 16:
        raise ValueError("width must be a multiple of 8 and at least 16")
    return WORLDS[name](width)


def config_text(name: str, output_width: int, levels_file: str = "levels.png") -> str:
    n = output_width // 4
    levels = [n // 4, n] if n // 4 >= 4 else [n]
    lines = [
        f'name = "{name}"',
        "",
        "[resolution]",
        f"output_width = {output_width}",
        f"levels = {levels}",
        "",
        "[input]",
        f'levels = "{levels_file}"',
    ]
    for c in CLASSES:
        lines += ["", "[[input.classes]]"]
        for k in ("value", "name", "kind", "base", "depth_m"):
            if k in c:
                v = c[k]
                lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
    return "\n".join(lines) + "\n"


def write_world(
    name: str, out_dir: Path, width: int = 512, output_width: int | None = None
) -> Path:
    """Write ``levels.png`` and ``world.toml`` into out_dir; returns the config path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_levels(name, width)).save(out_dir / "levels.png")
    cfg = out_dir / "world.toml"
    cfg.write_text(config_text(name, output_width or width))
    return cfg
