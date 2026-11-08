"""Flow routing on the cube sphere (DESIGN §7.2-7.3).

One call does:

1. **Priority-Flood+epsilon** (Barnes et al. 2014): flood inward from the
   coast, raising every pit so each land cell has a strictly lower path to
   the sea. Pop order is a valid processing order (receivers before donors).
2. **Steepest-descent receivers** on the filled surface. A strictly lower
   neighbour was always popped earlier, so the order stays valid. Optionally
   Rho8 (Fairfield & Leymarie 1991, doi:10.1029/90WR02658): each diagonal
   distance is stretched by a random (2 - r) / sqrt(2), r in [0, 1), which
   breaks D8's straight, grid-aligned channels (D32). r comes from a hash of
   (cell, direction, seed), so runs are repeatable.
3. **Drainage area**, summed down the tree in reverse order.

Heights are float64 here: epsilon steps of 1e-3 m are below float32
resolution at mountain heights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
from globerode.heap import pop as _heap_pop
from globerode.heap import push as _heap_push

DEFAULT_EPSILON_M = 1e-3


@dataclass(frozen=True)
class Routing:
    """Result of one routing pass. Arrays are over all cells unless noted."""

    filled: np.ndarray  # float64; filled height (land), input height elsewhere
    receiver: np.ndarray  # int32; downstream cell (may be a sea cell), -1 for sea
    stack: np.ndarray  # int32; land cells only, receivers before donors
    area: np.ndarray  # float64 m^2; drainage area (cell's own area included), 0 at sea


# --- kernels -------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _dist(a: int, b: int, n: int) -> float:
    fa, ia, ja = cell_fij(a, n)
    fb, ib, jb = cell_fij(b, n)
    p = cell_xyz(fa, ia, ja, n, FACE_FRAMES)
    q = cell_xyz(fb, ib, jb, n, FACE_FRAMES)
    return arc(p[0], p[1], p[2], q[0], q[1], q[2])


@njit(cache=True)
def _priority_flood(
    h: np.ndarray, land: np.ndarray, n: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ncells = h.shape[0]
    filled = h.copy()
    receiver = np.full(ncells, -1, dtype=np.int32)
    stack = np.empty(ncells, dtype=np.int32)
    visited = np.zeros(ncells, dtype=np.uint8)
    hk = np.empty(ncells, dtype=np.float64)
    hi = np.empty(ncells, dtype=np.int32)
    size = 0

    # Seed: land cells touching the sea drain to their nearest sea neighbour.
    for c in range(ncells):
        if not land[c]:
            continue
        f, i, j = cell_fij(c, n)
        best = -1
        best_d = math.inf
        for k in range(8):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE)
            if b >= 0 and not land[b]:
                d = _dist(c, b, n)
                if d < best_d:
                    best_d = d
                    best = b
        if best >= 0:
            receiver[c] = best
            visited[c] = 1
            size = _heap_push(hk, hi, size, filled[c], c)

    count = 0
    while size > 0:
        key, c, size = _heap_pop(hk, hi, size)
        stack[count] = c
        count += 1
        f, i, j = cell_fij(c, n)
        for k in range(8):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE)
            if b < 0 or visited[b] or not land[b]:
                continue
            visited[b] = 1
            if filled[b] < key + eps:
                filled[b] = key + eps
            receiver[b] = c
            size = _heap_push(hk, hi, size, filled[b], b)
    return filled, receiver, stack, count


@njit(cache=True, inline="always")
def _unit_hash(c: int, k: int, seed: int) -> float:
    """Uniform [0, 1) from (cell, direction, seed) via splitmix64."""
    z = np.uint64(c * 8 + k) ^ (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15))
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return float(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, parallel=True)
def _steepest_receivers(
    filled: np.ndarray, land: np.ndarray, receiver: np.ndarray, n: int, rho8_seed: int
) -> None:
    """Point each land cell at its steepest strictly-lower neighbour (in place).

    Sea neighbours count as height 0 (sea level). If none is lower, the flood
    receiver is kept (it is always lower by construction). ``rho8_seed`` >= 0
    stretches diagonal distances at random (Rho8).
    """
    rt2 = math.sqrt(2.0)
    for c in prange(filled.shape[0]):
        if not land[c]:
            continue
        f, i, j = cell_fij(c, n)
        hc = filled[c]
        best = receiver[c]
        best_s = -1.0
        for k in range(8):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE)
            if b < 0:
                continue
            hb = filled[b] if land[b] else 0.0
            if hb >= hc:
                continue
            d = _dist(c, b, n)
            if rho8_seed >= 0 and k >= 4:  # diagonal
                d *= (2.0 - _unit_hash(c, k, rho8_seed)) / rt2
            s = (hc - hb) / d
            if s > best_s or (s == best_s and b < best):
                best_s = s
                best = b
        receiver[c] = best


@njit(cache=True)
def _drainage_area(
    stack: np.ndarray, receiver: np.ndarray, land: np.ndarray, cell_area: np.ndarray
) -> np.ndarray:
    area = np.zeros(cell_area.shape[0], dtype=np.float64)
    for k in range(stack.shape[0]):
        c = stack[k]
        area[c] = cell_area[c]
    for k in range(stack.shape[0] - 1, -1, -1):
        c = stack[k]
        r = receiver[c]
        if land[r]:
            area[r] += area[c]
    return area


# --- Python API ---------------------------------------------------------------------------


def route(
    grid: Grid,
    height_m: np.ndarray,
    land: np.ndarray,
    epsilon_m: float = DEFAULT_EPSILON_M,
    rainfall: np.ndarray | None = None,
    rho8_seed: int | None = None,
) -> Routing:
    """Fill pits and route flow over land cells to the sea.

    ``height_m`` should be positive on land (the coast clamp guarantees it).
    ``rainfall`` (optional, per cell, ~1 on average) weights each cell's
    area, so ``area`` becomes rain-weighted drainage area (discharge, D26).
    ``rho8_seed`` turns on Rho8 receivers (D32); None gives plain D8.
    Raises ValueError if some land cannot reach the sea (e.g. no sea at all).
    """
    h = np.ascontiguousarray(height_m, dtype=np.float64)
    is_land = np.ascontiguousarray(land, dtype=np.bool_)
    if h.shape != (grid.ncells,) or is_land.shape != (grid.ncells,):
        raise ValueError("height and land must have one value per grid cell")
    if epsilon_m <= 0:
        raise ValueError("epsilon must be positive")

    filled, receiver, stack_full, count = _priority_flood(h, is_land, grid.n, epsilon_m)
    nland = int(is_land.sum())
    if count != nland:
        raise ValueError(
            f"{nland - count} land cells cannot reach the sea; "
            "the map needs at least some ocean touching every landmass"
        )
    stack = stack_full[:count].copy()
    seed = -1 if rho8_seed is None else rho8_seed
    _steepest_receivers(filled, is_land, receiver, grid.n, seed)
    cell_area = grid.cell_area_m2.astype(np.float64)
    if rainfall is not None:
        cell_area = cell_area * np.asarray(rainfall, dtype=np.float64)
    area = _drainage_area(stack, receiver, is_land, cell_area)
    return Routing(filled=filled, receiver=receiver, stack=stack, area=area)
