"""Signed geodesic distance to the coast (DESIGN §6.1).

Positive inland, negative at sea, in metres. Computed with a
nearest-coast-point Dijkstra: every cell keeps the coast point it is closest
to, and its distance is the great-circle distance to that point. This avoids
the up-to-8% overestimate of plain 8-neighbour graph distance.

Coast points are the midpoints between edge-adjacent land and sea cell
centres, so the coastline lies half a cell from each side.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from globerode.grid import (
    EDGE_TABLE,
    FACE_FRAMES,
    NEIGHBOUR_OFFSETS,
    Grid,
    arc,
    cell_fij,
    cell_xyz,
    neighbour,
)
from globerode.heap import ipop, ipush_or_decrease


@njit(cache=True)
def _coast_distance(land: np.ndarray, n: int, cap: float) -> np.ndarray:
    ncells = land.shape[0]
    dist = np.full(ncells, np.inf)
    seed = np.zeros((ncells, 3), dtype=np.float32)
    hk = np.empty(ncells, dtype=np.float64)
    hi = np.empty(ncells, dtype=np.int32)
    pos = np.full(ncells, -1, dtype=np.int32)
    done = np.zeros(ncells, dtype=np.uint8)
    size = 0

    # Seeds: midpoints to edge neighbours of the other kind.
    for c in range(ncells):
        f, i, j = cell_fij(c, n)
        p = cell_xyz(f, i, j, n, FACE_FRAMES)
        for k in range(4):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE)
            if land[b] == land[c]:
                continue
            fb, ib, jb = cell_fij(b, n)
            q = cell_xyz(fb, ib, jb, n, FACE_FRAMES)
            mx, my, mz = p[0] + q[0], p[1] + q[1], p[2] + q[2]
            r = math.sqrt(mx * mx + my * my + mz * mz)
            mx, my, mz = mx / r, my / r, mz / r
            d = arc(p[0], p[1], p[2], mx, my, mz)
            if d < dist[c]:
                dist[c] = d
                seed[c, 0], seed[c, 1], seed[c, 2] = mx, my, mz
        if dist[c] < np.inf:
            size = ipush_or_decrease(hk, hi, pos, size, dist[c], c)

    while size > 0:
        d, c, size = ipop(hk, hi, pos, size)
        done[c] = 1
        if d > cap:
            break
        f, i, j = cell_fij(c, n)
        sx, sy, sz = seed[c, 0], seed[c, 1], seed[c, 2]
        for k in range(8):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE)
            if b < 0 or done[b] or land[b] != land[c]:
                continue
            fb, ib, jb = cell_fij(b, n)
            q = cell_xyz(fb, ib, jb, n, FACE_FRAMES)
            nd = arc(q[0], q[1], q[2], sx, sy, sz)
            if nd < dist[b]:
                dist[b] = nd
                seed[b, 0], seed[b, 1], seed[b, 2] = sx, sy, sz
                size = ipush_or_decrease(hk, hi, pos, size, nd, b)

    out = np.empty(ncells, dtype=np.float32)
    for c in range(ncells):
        d = min(dist[c], cap)
        out[c] = d if land[c] else -d
    return out


def coast_distance(grid: Grid, land: np.ndarray, cap_m: float) -> np.ndarray:
    """Signed distance to the coast in metres (float32), clipped to +-cap_m.

    A world with no coast at all returns +-cap_m everywhere.
    """
    is_land = np.ascontiguousarray(land, dtype=np.bool_)
    if is_land.shape != (grid.ncells,):
        raise ValueError("land must have one value per cell")
    cap = cap_m / grid.radius_m
    out: np.ndarray = _coast_distance(is_land, grid.n, cap) * np.float32(grid.radius_m)
    return out
