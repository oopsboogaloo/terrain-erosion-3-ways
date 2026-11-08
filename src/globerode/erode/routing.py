"""Flow routing on the cube sphere (DESIGN §7.2-7.3).

One call does:

1. **Priority-Flood+epsilon** (Barnes et al. 2014): flood inward from the
   coast, raising every pit so each land cell has a strictly lower path to
   the sea. Pop order is a valid processing order (receivers before donors).
2. **Steepest-descent receivers** on the filled surface. A strictly lower
   neighbour was always popped earlier, so the order stays valid.
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

DEFAULT_EPSILON_M = 1e-3


@dataclass(frozen=True)
class Routing:
    """Result of one routing pass. Arrays are over all cells unless noted."""

    filled: np.ndarray  # float64; filled height (land), input height elsewhere
    receiver: np.ndarray  # int32; downstream cell (may be a sea cell), -1 for sea
    stack: np.ndarray  # int32; land cells only, receivers before donors
    area: np.ndarray  # float64 m^2; drainage area (cell's own area included), 0 at sea


# --- min-heap keyed on (height, index) ---------------------------------------------


@njit(cache=True, inline="always")
def _less(hk: np.ndarray, hi: np.ndarray, a: int, b: int) -> bool:
    return bool(hk[a] < hk[b] or (hk[a] == hk[b] and hi[a] < hi[b]))


@njit(cache=True)
def _heap_push(hk: np.ndarray, hi: np.ndarray, size: int, key: float, idx: int) -> int:
    k = size
    hk[k] = key
    hi[k] = idx
    while k > 0:
        parent = (k - 1) >> 1
        if _less(hk, hi, k, parent):
            hk[k], hk[parent] = hk[parent], hk[k]
            hi[k], hi[parent] = hi[parent], hi[k]
            k = parent
        else:
            break
    return size + 1


@njit(cache=True)
def _heap_pop(hk: np.ndarray, hi: np.ndarray, size: int) -> tuple[float, int, int]:
    key = hk[0]
    idx = hi[0]
    size -= 1
    hk[0] = hk[size]
    hi[0] = hi[size]
    k = 0
    while True:
        left = 2 * k + 1
        if left >= size:
            break
        child = left
        right = left + 1
        if right < size and _less(hk, hi, right, left):
            child = right
        if _less(hk, hi, child, k):
            hk[k], hk[child] = hk[child], hk[k]
            hi[k], hi[child] = hi[child], hi[k]
            k = child
        else:
            break
    return key, idx, size


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


@njit(cache=True, parallel=True)
def _steepest_receivers(filled: np.ndarray, land: np.ndarray, receiver: np.ndarray, n: int) -> None:
    """Point each land cell at its steepest strictly-lower neighbour (in place).

    Sea neighbours count as height 0 (sea level). If none is lower, the flood
    receiver is kept (it is always lower by construction).
    """
    for c in prange(filled.shape[0]):
        if not land[c]:
            continue
        f, i, j = cell_fij(c, n)
        hc = filled[c]
        best = receiver[c]
        hb = filled[best] if land[best] else 0.0
        best_s = (hc - hb) / _dist(c, best, n)
        for k in range(8):
            b = neighbour(f, i, j, NEIGHBOUR_OFFSETS[k, 0], NEIGHBOUR_OFFSETS[k, 1], n, EDGE_TABLE)
            if b < 0:
                continue
            hb = filled[b] if land[b] else 0.0
            if hb >= hc:
                continue
            s = (hc - hb) / _dist(c, b, n)
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
) -> Routing:
    """Fill pits and route flow over land cells to the sea.

    ``height_m`` should be positive on land (the coast clamp guarantees it).
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
    _steepest_receivers(filled, is_land, receiver, grid.n)
    area = _drainage_area(stack, receiver, is_land, grid.cell_area_m2.astype(np.float64))
    return Routing(filled=filled, receiver=receiver, stack=stack, area=area)
