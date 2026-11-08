"""Preview images Claude (and the user) can look at (DESIGN §12.2).

Hillshaded equirect, orthographic globe views and close-ups of the places
grid problems would show (poles, the 180 degree seam, a cube corner), a mask
overlay, and simple charts drawn with Pillow (no plotting dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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


# --- globe views and close-ups --------------------------------------------------------------

GLOBE_VIEWS: dict[str, tuple[float, float, float]] = {
    # name: (lon_deg, lat_deg, half-width of the view in radians; 1.0 = whole disc)
    "globe_lon0": (0.0, 0.0, 1.0),
    "globe_lon90": (90.0, 0.0, 1.0),
    "globe_lon180": (180.0, 0.0, 1.0),
    "globe_lon-90": (-90.0, 0.0, 1.0),
    "globe_north": (0.0, 90.0, 1.0),
    "globe_south": (0.0, -90.0, 1.0),
    "closeup_northpole": (0.0, 90.0, 0.25),
    "closeup_southpole": (0.0, -90.0, 0.25),
    "closeup_seam180": (180.0, 0.0, 0.25),
    "closeup_cubecorner": (45.0, 35.264, 0.25),
}


def shaded_rgb(height_m: np.ndarray, radius_m: float, max_width: int = 4096) -> np.ndarray:
    """Tinted, hillshaded equirect image (float RGB 0..1) used for all views."""
    small = downsample(height_m, max_width)
    shade = hillshade_equirect(small, radius_m)
    out: np.ndarray = tint(small) * (0.35 + 0.65 * shade[..., None])
    return out


def orthographic(
    rgb: np.ndarray, lon0_deg: float, lat0_deg: float, half_width: float, size: int = 512
) -> np.ndarray:
    """Orthographic view of an equirect RGB image (uint8), centred on (lon0, lat0)."""
    h, w, _ = rgb.shape
    lo, la = np.radians(lon0_deg), np.radians(lat0_deg)
    c = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.cross(c, east)
    u = ((np.arange(size) + 0.5) / size * 2 - 1) * half_width
    x, y = np.meshgrid(u, -u)
    r2 = x**2 + y**2
    inside = r2 < 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    p = x[..., None] * east + y[..., None] * north + z[..., None] * c
    lon = np.arctan2(p[..., 1], p[..., 0])
    lat = np.arcsin(np.clip(p[..., 2], -1.0, 1.0))
    px = ((lon + np.pi) / (2 * np.pi) * w - 0.5) % w
    py = np.clip((np.pi / 2 - lat) / np.pi * h - 0.5, 0, h - 1)
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    tx = (px - x0)[..., None]
    ty = (py - y0)[..., None]
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    img = (rgb[y0, x0] * (1 - tx) + rgb[y0, x1] * tx) * (1 - ty) + (
        rgb[y1, x0] * (1 - tx) + rgb[y1, x1] * tx
    ) * ty
    limb = 0.55 + 0.45 * z[..., None] if half_width >= 1.0 else 1.0
    img = np.where(inside[..., None], img * limb, 0.0)
    out: np.ndarray = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return out


def write_views(out_dir: Path, height_m: np.ndarray, radius_m: float) -> list[Path]:
    rgb = shaded_rgb(height_m, radius_m)
    paths = []
    for name, (lon, lat, hw) in GLOBE_VIEWS.items():
        path = out_dir / f"{name}.png"
        Image.fromarray(orthographic(rgb, lon, lat, hw)).save(path)
        paths.append(path)
    return paths


def write_mask_overlay(
    path: Path, height_m: np.ndarray, rivers: np.ndarray, erosion01: np.ndarray, radius_m: float
) -> Path:
    """Hillshade with eroded depth in red and rivers in blue."""
    small = downsample(height_m)
    f = height_m.shape[1] // small.shape[1]
    h, w = rivers.shape
    riv = rivers.reshape(h // f, f, w // f, f).max(axis=(1, 3))
    ero = downsample(erosion01.astype(np.float64), small.shape[1])
    shade = hillshade_equirect(small, radius_m)
    base = np.repeat((0.25 + 0.6 * shade)[..., None], 3, axis=-1)
    base[..., 0] = np.clip(base[..., 0] + 0.8 * ero, 0, 1)
    base[riv] = [0.15, 0.35, 1.0]
    Image.fromarray((base * 255).astype(np.uint8)).save(path)
    return path


# --- simple charts (Pillow only) -----------------------------------------------------------


def _axes(size: tuple[int, int], title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 6), title, fill="black")
    return img, draw


def write_curve_chart(path: Path, points: list[tuple[float, float]], curve: object) -> Path:
    """Height curve: configured points and the PCHIP curve between them."""
    w, h, m = 480, 320, 40
    img, d = _axes((w, h), "height curve (x: smoothed level 0..1, y: metres)")
    xs = np.linspace(0, 1, 200)
    ys = np.asarray(curve(xs))  # type: ignore[operator]
    top = max(float(ys.max()), max(p[1] for p in points), 1.0)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return m + x * (w - 2 * m), h - m - y / top * (h - 2 * m)

    d.rectangle([m, m, w - m, h - m], outline="grey")
    d.line([to_px(float(x), float(y)) for x, y in zip(xs, ys, strict=True)], fill="blue", width=2)
    for x, y in points:
        px, py = to_px(x, y)
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill="red")
    d.text((m, h - m + 6), "0", fill="black")
    d.text((w - m - 10, h - m + 6), "1", fill="black")
    d.text((4, m - 4), f"{top:.0f} m", fill="black")
    img.save(path)
    return path


def write_histogram(path: Path, final_m: np.ndarray, prepared_m: np.ndarray) -> Path:
    """Land height histograms: final (blue) vs prepared (grey)."""
    w, h, m = 480, 320, 40
    img, d = _axes((w, h), "land heights: final (blue) vs prepared (grey)")
    top = float(max(final_m.max(initial=1.0), prepared_m.max(initial=1.0)))
    bins = np.linspace(0, top, 41)
    hf, _ = np.histogram(final_m, bins)
    hp, _ = np.histogram(prepared_m, bins)
    peak = max(int(hf.max(initial=1)), int(hp.max(initial=1)), 1)
    bw = (w - 2 * m) / 40
    for k in range(40):
        x0 = m + k * bw
        for counts, col, off in ((hp, "lightgrey", 0.0), (hf, "royalblue", bw / 2)):
            yh = counts[k] / peak * (h - 2 * m)
            d.rectangle([x0 + off, h - m - yh, x0 + off + bw / 2 - 1, h - m], fill=col)
    d.text((m, h - m + 6), "0 m", fill="black")
    d.text((w - m - 50, h - m + 6), f"{top:.0f} m", fill="black")
    img.save(path)
    return path
