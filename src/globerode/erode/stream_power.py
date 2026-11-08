"""Implicit stream-power incision with optional deposition, n = 1.

Without deposition (Braun & Willett 2013; DESIGN §7.4; D5), for each land
cell in stack order (receiver already updated)::

    F   = K * dt * A^m / d(c, r)
    h_c = (h_c + dt * U_c + F * h_r) / (1 + F)

With deposition (Yuan et al. 2019, JGR Earth Surface, doi:10.1029/2018JF004867;
D30), rivers drop part of the sediment eroded upstream, in proportion to
sediment flux over water flux (Davy & Lague 2009). The target height before
incision is raised by the deposit::

    elev_c = h_c + dt * U_c + G * V_up(c) / A(c)
    h_c    = (elev_c + F * h_r) / (1 + F)

V_up is the net volume eroded strictly upstream of c in this step. It depends
on the new heights, so the step is repeated (Gauss-Seidel) a fixed number of
times, as in FastScape. G = 0 gives the plain scheme in one pass.

A sea receiver counts as sea level (0 m), never the ocean floor, so rivers
don't cut down toward ocean depth at the coast. Sediment reaching the sea is
removed. Unconditionally stable.
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
    cell_area: np.ndarray,
    n: int,
    radius_m: float,
    dt: float,
    m: float,
    g: float,
    passes: int,
) -> None:
    ns = stack.shape[0]
    fac = np.empty(ns, dtype=np.float64)
    h_t = np.empty(ns, dtype=np.float64)  # height after uplift, before incision
    for k in range(ns):
        c = stack[k]
        r = receiver[c]
        fc, ic, jc = cell_fij(c, n)
        fr, ir, jr = cell_fij(r, n)
        p = cell_xyz(fc, ic, jc, n, FACE_FRAMES)
        q = cell_xyz(fr, ir, jr, n, FACE_FRAMES)
        d = arc(p[0], p[1], p[2], q[0], q[1], q[2]) * radius_m
        fac[k] = erodibility[c] * dt * area[c] ** m / d
        h_t[k] = h[c] + dt * uplift[c]

    v_up = np.zeros(h.shape[0], dtype=np.float64)  # sediment volume from upstream
    v_tot = np.zeros(h.shape[0], dtype=np.float64)
    for it in range(passes if g > 0 else 1):
        for k in range(ns):
            c = stack[k]
            r = receiver[c]
            hr = h[r] if land[r] else 0.0
            elev = h_t[k]
            if g > 0:
                elev += g * v_up[c] / area[c]
            h[c] = (elev + fac[k] * hr) / (1.0 + fac[k])
        if g <= 0 or it == passes - 1:
            break
        # net eroded volume, summed downstream (reverse stack order)
        for k in range(ns):
            v_tot[stack[k]] = 0.0
        for k in range(ns - 1, -1, -1):
            c = stack[k]
            e = (h_t[k] - h[c]) * cell_area[c]
            v_up[c] = max(v_tot[c], 0.0)
            v_tot[c] += e
            r = receiver[c]
            if land[r]:
                v_tot[r] += v_tot[c]


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
    cell_area_m2: np.ndarray | None = None,
    deposition_g: float = 0.0,
    passes: int = 1,
) -> None:
    """One implicit uplift + incision (+ deposition) step, updating ``h`` (float64) in place.

    ``deposition_g`` > 0 needs ``cell_area_m2`` and uses ``passes`` Gauss-Seidel passes.
    """
    if deposition_g > 0 and cell_area_m2 is None:
        raise ValueError("deposition needs cell areas")
    ca = (
        np.ascontiguousarray(cell_area_m2, np.float64)
        if cell_area_m2 is not None
        else np.empty(0, dtype=np.float64)
    )
    _stream_power(
        h,
        np.ascontiguousarray(uplift_m_per_yr, np.float64),
        np.ascontiguousarray(erodibility, np.float64),
        area_m2,
        receiver,
        stack,
        np.ascontiguousarray(land, np.bool_),
        ca,
        n,
        radius_m,
        dt_yr,
        m,
        float(deposition_g),
        max(1, int(passes)),
    )
