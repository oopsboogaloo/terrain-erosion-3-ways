"""Preview images Claude (and the user) can look at (DESIGN §12.2).

Step 4 provides the hillshaded equirect; globe views and close-ups come in
step 7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MAX_PREVIEW_WIDTH = 1024

# Hypsometric tint stops: (metres, RGB).
_STOPS = [
    (-8000.0, (8, 20, 60)),
    (-3000.0, (20, 50, 120)),
    (-150.0, (70, 130, 190)),
    (-1.0, (120, 180, 220)),
    (0.0, (90, 150, 80)),
    (500.0, (140, 170, 90)),
    (1500.0, (180, 160, 100)),
    (3000.0, (150, 110, 80)),
    (5000.0, (220, 220, 220)),
    (8000.0, (255, 255, 255)),
]


def downsample(image: np.ndarray, max_width: int = MAX_PREVIEW_WIDTH) -> np.ndarray:
    """Block-average an equirect image to at most max_width (integer factor)."""
    h, w = image.shape
    f = max(1, -(-w // max_width))
    while w % f or h % f:
        f += 1
    return image.reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def tint(height_m: np.ndarray) -> np.ndarray:
    """Hypsometric colours, float RGB in 0..1, shape (..., 3)."""
    xs = np.array([s[0] for s in _STOPS])
    cols = np.array([s[1] for s in _STOPS], dtype=np.float64) / 255.0
    return np.stack([np.interp(height_m, xs, cols[:, k]) for k in range(3)], axis=-1)


def hillshade_equirect(
    height_m: np.ndarray,
    radius_m: float,
    exaggeration: float = 20.0,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    """Lambertian hillshade (0..1) of an equirect height map, metric-correct in x."""
    h, w = height_m.shape
    lat = np.radians(90.0 - (np.arange(h) + 0.5) / h * 180.0)
    dy = radius_m * np.pi / h
    dx = radius_m * 2 * np.pi / w * np.maximum(np.cos(lat), 1e-3)[:, None]
    z = height_m * exaggeration
    gx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
    gy = np.zeros_like(z)
    gy[1:-1] = (z[:-2] - z[2:]) / (2 * dy)  # north minus south
    az = np.radians(azimuth_deg)
    alt = np.radians(altitude_deg)
    lx, ly, lz = np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)
    nz = 1.0 / np.sqrt(1 + gx**2 + gy**2)
    shade = (-gx * lx - gy * ly + lz) * nz
    out: np.ndarray = np.clip(shade, 0.0, 1.0)
    return out


def write_hillshade_preview(path: Path, height_m: np.ndarray, radius_m: float) -> Path:
    small = downsample(height_m)
    shade = hillshade_equirect(small, radius_m)
    rgb = tint(small) * (0.35 + 0.65 * shade[..., None])
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).save(path)
    return path
