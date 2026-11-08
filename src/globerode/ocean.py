"""Ocean floor: painted depths + shelf ramp + deep-floor noise (DESIGN §8).

    h_sea = min(max_sea, max(painted_smoothed, ramp(distance offshore))) + noise

The ramp only ever makes the sea *shallower* near the coast: 0 m at the
coast down to the shelf depth over the shelf width, then a cosine-eased drop
to the deepest painted depth over the slope width.
"""

from __future__ import annotations

import numpy as np

from globerode.smooth import smoothstep


def shelf_ramp(
    offshore_m: np.ndarray,
    shelf_depth_m: float,
    shelf_width_m: float,
    slope_width_m: float,
    abyss_m: float,
) -> np.ndarray:
    """Depth bound (negative metres) as a function of distance offshore."""
    d = np.maximum(offshore_m, 0.0)
    shelf = shelf_depth_m * np.minimum(d / shelf_width_m, 1.0)
    t = np.clip((d - shelf_width_m) / slope_width_m, 0.0, 1.0)
    eased = 0.5 - 0.5 * np.cos(np.pi * t)
    out: np.ndarray = shelf + (abyss_m - shelf_depth_m) * eased * (d > shelf_width_m)
    return out


def ocean_floor(
    painted_m: np.ndarray,
    offshore_m: np.ndarray,
    noise: np.ndarray,
    *,
    shelf_depth_m: float,
    shelf_width_m: float,
    slope_width_m: float,
    floor_noise_m: float,
    max_sea_height_m: float,
) -> np.ndarray:
    """Sea-cell heights. ``painted_m`` should already be smoothed over the sea."""
    abyss = float(np.min(painted_m)) if painted_m.size else shelf_depth_m
    ramp = shelf_ramp(offshore_m, shelf_depth_m, shelf_width_m, slope_width_m, min(abyss, -1.0))
    h = np.maximum(painted_m, ramp)
    deep = smoothstep(shelf_width_m, shelf_width_m + slope_width_m, offshore_m)
    h = h + floor_noise_m * noise * deep
    out: np.ndarray = np.minimum(h, max_sea_height_m)
    return out
