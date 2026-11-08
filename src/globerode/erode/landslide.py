"""Landslide slope limit (threshold hillslopes, D27).

Real mountain slopes stop steepening at a critical angle: beyond it they
fail by landsliding, giving straight slopes and sharp ridges (e.g. Anand
et al. 2023, GRL, eikonal threshold hillslopes). Here each land cell, in
stack order (outlets first), is lowered so it is at most ``tan(angle) * d``
above every neighbour already processed (sea counts as 0 m)::

    h_c = min(h_c, min_b (h_b + tan(angle) * d(c, b)))

The removed material is assumed carried away by rivers. Because the stack
runs from low to high, one pass limits most slopes; repeated every
iteration, the surface converges to the limit. The angle applies at the
grid's cell scale: slopes look gentler on coarser grids, so the useful
angle depends on resolution (about 15-20 deg at 8K, ~4.9 km cells).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from globerode.erode.routing import _dist
from globerode.grid import EDGE_TABLE, NEIGHBOUR_OFFSETS, Grid, cell_fij, neighbour


@njit(cache=True)
def _limit(h: np.ndarray, land: np.ndarray, stack: np.ndarray, n: int, s_max_m: float) -> int:
    done = np.zeros(h.shape[0], dtype=np.bool_)
    changed = 0
    # Neighbour centres are never closer than this (unit sphere; the true
    # minimum is ~0.7 cell widths), so a drop below s_max * d_lo can't be too
    # steep and the costly distance is skipped.
    d_lo = 0.5 * (math.pi / 2) / n
    for k in range(stack.shape[0]):
        c = stack[k]
        f, i, j = cell_fij(c, n)
        cap = h[c]
        for q in range(8):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[q, 0], NEIGHBOUR_OFFSETS[q, 1], n, EDGE_TABLE)
            if b < 0:
                continue
            if land[b]:
                if not done[b]:
                    continue
                hb = h[b]
            else:
                hb = 0.0
            if cap - hb <= s_max_m * d_lo:
                continue
            lim = hb + s_max_m * _dist(c, b, n)
            if lim < cap:
                cap = lim
        if cap < h[c]:
            h[c] = cap
            changed += 1
        done[c] = True
    return changed


def limit_slopes(
    grid: Grid, h: np.ndarray, land: np.ndarray, stack: np.ndarray, max_slope_deg: float
) -> int:
    """Lower land cells in place so no slope exceeds ``max_slope_deg``.

    ``h`` must be float64. Returns the number of cells lowered.
    """
    if max_slope_deg <= 0:
        return 0
    # _dist is on the unit sphere, so fold the radius into the slope.
    s = math.tan(math.radians(max_slope_deg)) * grid.radius_m
    return int(_limit(h, np.ascontiguousarray(land, np.bool_), stack, grid.n, s))
