"""Quality metrics for a run (DESIGN §12.1).

Most functions take plain arrays (receiver, stack, area, distance to
receiver) so they can be tested on synthetic networks without a grid.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

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

# --- helpers ----------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _receiver_distance(receiver: np.ndarray, n: int, radius_m: float) -> np.ndarray:
    out = np.zeros(receiver.shape[0], dtype=np.float64)
    for c in prange(receiver.shape[0]):
        r = receiver[c]
        if r < 0:
            continue
        fc, ic, jc = cell_fij(c, n)
        fr, ir, jr = cell_fij(r, n)
        p = cell_xyz(fc, ic, jc, n, FACE_FRAMES)
        q = cell_xyz(fr, ir, jr, n, FACE_FRAMES)
        out[c] = arc(p[0], p[1], p[2], q[0], q[1], q[2]) * radius_m
    return out


def receiver_distance_m(grid: Grid, receiver: np.ndarray) -> np.ndarray:
    """Great-circle distance from each cell to its receiver (0 where none)."""
    out: np.ndarray = _receiver_distance(receiver, grid.n, grid.radius_m)
    return out


MIN_RIVER_CELLS = 10  # a river drains at least this many cells' worth of area


def river_threshold_m2(grid: Grid, threshold_km2: float) -> float:
    """Drainage area that makes a cell a river: the configured threshold, but never
    less than MIN_RIVER_CELLS mean cells (otherwise every cell of a coarse grid is a river)."""
    mean_cell = 4 * math.pi * grid.radius_m**2 / grid.ncells
    return max(threshold_km2 * 1e6, MIN_RIVER_CELLS * mean_cell)


# --- correctness ----------------------------------------------------------------------------


def coast_match(height_eq: np.ndarray, land_eq: np.ndarray) -> dict[str, float]:
    """Exported land mask (height > 0) vs the input land mask, pixel for pixel."""
    mismatch = int(np.count_nonzero((height_eq > 0) != land_eq))
    return {"mismatched_pixels": mismatch, "mismatch_fraction": mismatch / land_eq.size}


@njit(cache=True)
def _drains(receiver: np.ndarray, stack: np.ndarray, land: np.ndarray) -> int:
    """Count land cells whose receiver chain does not reach the sea."""
    ok = np.zeros(receiver.shape[0], dtype=np.uint8)
    bad = 0
    for k in range(stack.shape[0]):  # receivers come first
        c = stack[k]
        r = receiver[c]
        if r < 0:
            bad += 1
        elif not land[r] or ok[r]:
            ok[c] = 1
        else:
            bad += 1
    return bad


def drainage(receiver: np.ndarray, stack: np.ndarray, land: np.ndarray) -> dict[str, float]:
    """Fraction of land cells whose flow path reaches the sea."""
    nland = int(land.sum())
    in_stack = np.zeros(land.shape[0], dtype=bool)
    in_stack[stack] = True
    missing = int(np.count_nonzero(land & ~in_stack))
    bad = int(_drains(receiver, stack, land)) + missing
    return {"land_cells": nland, "not_draining": bad, "fraction_draining": 1 - bad / max(nland, 1)}


@njit(cache=True, parallel=True)
def _seam_pairs(h: np.ndarray, land: np.ndarray, n: int, radius_m: float) -> np.ndarray:
    """For each step across a face edge, the gradient across it and the gradient
    of the next step back inside the face along the same line.

    Returns (ncells * 4, 2): (cross, inside); unused rows are -1.
    """
    ncells = h.shape[0]
    out = np.full((ncells * 4, 2), -1.0)
    for c in prange(ncells):
        if not land[c]:
            continue
        f, i, j = cell_fij(c, n)
        p = cell_xyz(f, i, j, n, FACE_FRAMES)
        for k in range(4):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE)
            fb, ib, jb = cell_fij(b, n)
            if fb == f or not land[b]:
                continue
            k2 = k ^ 1  # opposite direction: W<->E, S<->N
            o = neighbour(
                f, i, j, NEIGHBOUR_OFFSETS[k2, 0], NEIGHBOUR_OFFSETS[k2, 1], n, EDGE_TABLE
            )
            fo, io, jo = cell_fij(o, n)
            if fo != f or not land[o]:
                continue
            q = cell_xyz(fb, ib, jb, n, FACE_FRAMES)
            r = cell_xyz(fo, io, jo, n, FACE_FRAMES)
            out[c * 4 + k, 0] = abs(h[b] - h[c]) / (
                arc(p[0], p[1], p[2], q[0], q[1], q[2]) * radius_m
            )
            out[c * 4 + k, 1] = abs(h[c] - h[o]) / (
                arc(p[0], p[1], p[2], r[0], r[1], r[2]) * radius_m
            )
    return out


def seam_continuity(grid: Grid, height: np.ndarray, land: np.ndarray) -> dict[str, float]:
    """Mean |gradient| across cube-face edges / mean |gradient| one step inside.

    Each step over a face edge is paired with the next step back inside the
    face along the same line, so a smooth surface gives ~1 and a step or
    crease at the seams pushes it up. ``log_se`` is the standard error of
    log(ratio) (delta method over the pairs), so callers can tell a real
    seam from sampling noise on rough, eroded terrain.
    """
    g = _seam_pairs(np.ascontiguousarray(height, np.float64), land, grid.n, grid.radius_m)
    g = g[g[:, 0] >= 0]
    nan = float("nan")
    if g.shape[0] < 10:
        return {"ratio": nan, "log_se": nan, "pairs": int(g.shape[0])}
    x, y = g[:, 0], g[:, 1]
    if y.mean() == 0:
        # Flat inside the faces: seamless only if also flat across the edges.
        ratio = 1.0 if x.mean() == 0 else float("inf")
        return {"ratio": ratio, "log_se": 0.0, "pairs": int(g.shape[0])}
    if x.mean() == 0:
        return {"ratio": 0.0, "log_se": 0.0, "pairs": int(g.shape[0])}
    mx, my = x.mean(), y.mean()
    cov = np.cov(x, y)
    n = x.size
    var_log = cov[0, 0] / (n * mx**2) + cov[1, 1] / (n * my**2) - 2 * cov[0, 1] / (n * mx * my)
    return {
        "ratio": float(mx / my),
        "log_se": float(np.sqrt(max(var_log, 0.0))),
        "pairs": int(n),
    }


# --- river network ----------------------------------------------------------------------------


@njit(cache=True)
def _longest_upstream(
    receiver: np.ndarray, stack: np.ndarray, dist: np.ndarray, land: np.ndarray
) -> np.ndarray:
    length = np.zeros(receiver.shape[0], dtype=np.float64)
    for k in range(stack.shape[0] - 1, -1, -1):  # donors before receivers
        c = stack[k]
        r = receiver[c]
        if r >= 0 and land[r]:
            v = length[c] + dist[c]
            if v > length[r]:
                length[r] = v
    return length


def hack_exponent(
    receiver: np.ndarray,
    stack: np.ndarray,
    area_m2: np.ndarray,
    dist_m: np.ndarray,
    land: np.ndarray,
    channel: np.ndarray,
) -> dict[str, float]:
    """Fit L = c * A^h over channel cells, L = longest flow path upstream.

    Real rivers: h ~ 0.5-0.6 (Hack 1957).
    """
    length = _longest_upstream(receiver, stack, dist_m, land)
    sel = channel & (length > 0)
    if sel.sum() < 10:
        return {"h": float("nan"), "cells": int(sel.sum())}
    slope, _ = np.polyfit(np.log(area_m2[sel]), np.log(length[sel]), 1)
    return {"h": float(slope), "cells": int(sel.sum())}


@njit(cache=True)
def _strahler(
    receiver: np.ndarray, stack: np.ndarray, land: np.ndarray, channel: np.ndarray
) -> np.ndarray:
    ncells = receiver.shape[0]
    top = np.zeros(ncells, dtype=np.int32)  # highest donor order so far
    count = np.zeros(ncells, dtype=np.int32)  # donors with that order
    order = np.zeros(ncells, dtype=np.int32)
    for k in range(stack.shape[0] - 1, -1, -1):
        c = stack[k]
        if not channel[c]:
            continue
        if top[c] == 0:
            order[c] = 1
        elif count[c] >= 2:
            order[c] = top[c] + 1
        else:
            order[c] = top[c]
        r = receiver[c]
        if r >= 0 and land[r] and channel[r]:
            if order[c] > top[r]:
                top[r] = order[c]
                count[r] = 1
            elif order[c] == top[r]:
                count[r] += 1
    return order


def horton_bifurcation(
    receiver: np.ndarray, stack: np.ndarray, land: np.ndarray, channel: np.ndarray
) -> dict[str, object]:
    """Strahler stream counts N_k and bifurcation ratio Rb = N_k / N_{k+1}.

    A stream of order k ends where its receiver has a different order (or is
    the sea). Rb from a log-linear fit of N_k against k. Real rivers: 3-5.
    """
    order = _strahler(receiver, stack, land, channel)
    ch = np.nonzero(channel & land)[0]
    if ch.size == 0:
        return {"rb": float("nan"), "counts": []}
    rec = receiver[ch]
    rec_order = np.where((rec >= 0) & land[np.maximum(rec, 0)], order[np.maximum(rec, 0)], 0)
    ends = order[ch] != rec_order
    kmax = int(order.max())
    counts = [int(np.count_nonzero(ends & (order[ch] == k))) for k in range(1, kmax + 1)]
    ks = [k + 1 for k, c in enumerate(counts) if c > 0]
    if len(ks) < 2:
        return {"rb": float("nan"), "counts": counts}
    slope, _ = np.polyfit(ks, np.log([counts[k - 1] for k in ks]), 1)
    return {"rb": float(math.exp(-slope)), "counts": counts}


def drainage_density_per_km(
    dist_m: np.ndarray, area_m2: np.ndarray, land: np.ndarray, channel: np.ndarray
) -> float:
    """Total channel length / land area, in km^-1."""
    length_km = float(dist_m[channel & land].sum()) / 1000.0
    area_km2 = float(area_m2[land].sum()) / 1e6
    return length_km / area_km2 if area_km2 > 0 else float("nan")


# --- heights ------------------------------------------------------------------------------


def height_stats(height: np.ndarray, h_init: np.ndarray, land: np.ndarray) -> dict[str, object]:
    pcts = [10, 50, 90, 99, 100]
    if not land.any():
        return {}
    got = np.percentile(height[land], pcts)
    want = np.percentile(h_init[land], pcts)
    return {
        "percentiles": pcts,
        "final_m": [round(float(x), 1) for x in got],
        "prepared_m": [round(float(x), 1) for x in want],
        "mean_m": round(float(height[land].mean()), 1),
        "max_m": round(float(height[land].max()), 1),
        "max_vs_prepared": round(float(got[-1] / max(want[-1], 1.0)), 3),
    }
