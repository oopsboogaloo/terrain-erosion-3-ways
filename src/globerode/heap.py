"""Numba binary heaps keyed on (float key, int index).

- ``push`` / ``pop``: plain min-heap; ties broken by index (deterministic).
- ``IndexedHeap`` functions: min-heap with one entry per cell and
  decrease-key, so its size never exceeds the number of cells (used by
  Dijkstra-style searches).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def less(hk: np.ndarray, hi: np.ndarray, a: int, b: int) -> bool:
    return bool(hk[a] < hk[b] or (hk[a] == hk[b] and hi[a] < hi[b]))


@njit(cache=True)
def push(hk: np.ndarray, hi: np.ndarray, size: int, key: float, idx: int) -> int:
    k = size
    hk[k] = key
    hi[k] = idx
    while k > 0:
        parent = (k - 1) >> 1
        if less(hk, hi, k, parent):
            hk[k], hk[parent] = hk[parent], hk[k]
            hi[k], hi[parent] = hi[parent], hi[k]
            k = parent
        else:
            break
    return size + 1


@njit(cache=True)
def pop(hk: np.ndarray, hi: np.ndarray, size: int) -> tuple[float, int, int]:
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
        if right < size and less(hk, hi, right, left):
            child = right
        if less(hk, hi, child, k):
            hk[k], hk[child] = hk[child], hk[k]
            hi[k], hi[child] = hi[child], hi[k]
            k = child
        else:
            break
    return key, idx, size


# --- indexed heap (decrease-key) --------------------------------------------------------


@njit(cache=True, inline="always")
def _swap(hk: np.ndarray, hi: np.ndarray, pos: np.ndarray, a: int, b: int) -> None:
    hk[a], hk[b] = hk[b], hk[a]
    hi[a], hi[b] = hi[b], hi[a]
    pos[hi[a]] = a
    pos[hi[b]] = b


@njit(cache=True)
def _sift_up(hk: np.ndarray, hi: np.ndarray, pos: np.ndarray, k: int) -> None:
    while k > 0:
        parent = (k - 1) >> 1
        if less(hk, hi, k, parent):
            _swap(hk, hi, pos, k, parent)
            k = parent
        else:
            break


@njit(cache=True)
def _sift_down(hk: np.ndarray, hi: np.ndarray, pos: np.ndarray, k: int, size: int) -> None:
    while True:
        left = 2 * k + 1
        if left >= size:
            break
        child = left
        if left + 1 < size and less(hk, hi, left + 1, left):
            child = left + 1
        if less(hk, hi, child, k):
            _swap(hk, hi, pos, k, child)
            k = child
        else:
            break


@njit(cache=True)
def ipush_or_decrease(
    hk: np.ndarray, hi: np.ndarray, pos: np.ndarray, size: int, key: float, idx: int
) -> int:
    """Insert idx, or lower its key if already present. ``pos[idx] = -1`` means absent."""
    k = pos[idx]
    if k < 0:
        k = size
        hk[k] = key
        hi[k] = idx
        pos[idx] = k
        size += 1
    elif key < hk[k]:
        hk[k] = key
    else:
        return size
    _sift_up(hk, hi, pos, k)
    return size


@njit(cache=True)
def ipop(hk: np.ndarray, hi: np.ndarray, pos: np.ndarray, size: int) -> tuple[float, int, int]:
    key = hk[0]
    idx = hi[0]
    pos[idx] = -1
    size -= 1
    if size > 0:
        hk[0] = hk[size]
        hi[0] = hi[size]
        pos[hi[0]] = 0
        _sift_down(hk, hi, pos, 0, size)
    return key, idx, size
