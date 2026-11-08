"""Output files (DESIGN §9).

Heights are written as equirectangular images. Before writing, the coast is
re-locked at output resolution (SPEC §10.3): every pixel's land/sea state
comes from the input levels map, and heights are clamped to match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import OpenEXR
import tifffile
from PIL import Image

from globerode.config import CoastConfig, Format


def relock_coast(height_m: np.ndarray, land: np.ndarray, coast: CoastConfig) -> np.ndarray:
    """Clamp heights so land pixels are >= min_land and sea pixels <= max_sea."""
    out: np.ndarray = np.where(
        land,
        np.maximum(height_m, coast.min_land_height_m),
        np.minimum(height_m, coast.max_sea_height_m),
    ).astype(np.float32)
    return out


def to_uint16(height_m: np.ndarray, png_range_m: tuple[float, float]) -> np.ndarray:
    """Map metres linearly onto 0..65535 over png_range_m (clipped)."""
    lo, hi = png_range_m
    scaled = (np.asarray(height_m, np.float64) - lo) / (hi - lo) * 65535.0
    out: np.ndarray = np.clip(np.round(scaled), 0, 65535).astype(np.uint16)
    return out


def write_exr(path: Path, image: np.ndarray) -> None:
    """Single-channel ('Y') float32 EXR, ZIP compressed."""
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    OpenEXR.File(header, {"Y": np.ascontiguousarray(image, np.float32)}).write(str(path))


def write_png16(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(image, np.uint16)).save(path)


def write_tiff(path: Path, image: np.ndarray) -> None:
    tifffile.imwrite(path, np.ascontiguousarray(image), compression="zlib")


FILENAMES: dict[Format, str] = {
    "exr32": "height_m.exr",
    "tiff32": "height_m.tif",
    "png16": "height16.png",
    "tiff16": "height16.tif",
}


def write_heights(
    out_dir: Path,
    height_m: np.ndarray,
    formats: list[Format],
    png_range_m: tuple[float, float],
) -> list[Path]:
    """Write the height image in each requested format; returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    h16: np.ndarray | None = None
    for fmt in formats:
        path = out_dir / FILENAMES[fmt]
        if fmt == "exr32":
            write_exr(path, height_m)
        elif fmt == "tiff32":
            write_tiff(path, height_m.astype(np.float32))
        else:
            if h16 is None:
                h16 = to_uint16(height_m, png_range_m)
            if fmt == "png16":
                write_png16(path, h16)
            else:
                write_tiff(path, h16)
        written.append(path)
    return written


def write_mask_png8(path: Path, mask: np.ndarray) -> Path:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)
    return path
