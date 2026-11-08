"""Height curve: smoothed 0..1 land value -> metres (DESIGN §6.3).

Monotone cubic (PCHIP) through the configured points, so it never
overshoots between them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


class HeightCurve:
    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            raise ValueError("curve needs at least 2 [x, metres] points")
        xs, ys = pts[:, 0], pts[:, 1]
        if np.any(np.diff(xs) <= 0):
            raise ValueError("curve x values must be strictly increasing")
        if np.any(np.diff(ys) < 0):
            raise ValueError("curve heights must not decrease")
        self.xs = xs
        self.ys = ys
        self._f = PchipInterpolator(xs, ys, extrapolate=False)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xc = np.clip(np.asarray(x, dtype=np.float64), self.xs[0], self.xs[-1])
        out: np.ndarray = self._f(xc)
        return out
