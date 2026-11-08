"""Implicit stream-power incision, n = 1 (DESIGN §7.4; Braun & Willett 2013; D5).

For each land cell, in stack order (receiver already updated)::

    F   = K * dt * A^m / d(c, r)
    h_c = (h_c + dt * U_c + F * h_r) / (1 + F)

A sea receiver counts as sea level (0 m), never the ocean floor, so rivers
don't cut down toward ocean depth at the coast. Unconditionally stable.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from globerode.grid import FACE_FRAMES, arc, cell_fij, cell_xyz


@njit(cache=True)
def _stream_power(
    h: np.ndarray,
    uplift: np.ndarray,
    erodibility: np.ndarray,
    area: np.ndarray,
    receiver: np.ndarray,
    stack: np.ndarray,
    land: np.ndarray,
    n: int,
    radius_m: float,
    dt: float,
    m: float,
) -> None:
    for k in range(stack.shape[0]):
        c = stack[k]
        r = receiver[c]
        hr = h[r] if land[r] else 0.0
        fc, ic, jc = cell_fij(c, n)
        fr, ir, jr = cell_fij(r, n)
        p = cell_xyz(fc, ic, jc, n, FACE_FRAMES)
        q = cell_xyz(fr, ir, jr, n, FACE_FRAMES)
        d = arc(p[0], p[1], p[2], q[0], q[1], q[2]) * radius_m
        f = erodibility[c] * dt * area[c] ** m / d
        h[c] = (h[c] + dt * uplift[c] + f * hr) / (1.0 + f)


def stream_power_step(
    h: np.ndarray,
    uplift_m_per_yr: np.ndarray,
    erodibility: np.ndarray,
    area_m2: np.ndarray,
    receiver: np.ndarray,
    stack: np.ndarray,
    land: np.ndarray,
    n: int,
    radius_m: float,
    dt_yr: float,
    m: float,
) -> None:
    """One implicit uplift + incision step, updating ``h`` (float64) in place."""
    _stream_power(
        h,
        np.ascontiguousarray(uplift_m_per_yr, np.float64),
        np.ascontiguousarray(erodibility, np.float64),
        area_m2,
        receiver,
        stack,
        np.ascontiguousarray(land, np.bool_),
        n,
        radius_m,
        dt_yr,
        m,
    )
