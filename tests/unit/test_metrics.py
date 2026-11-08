import numpy as np
import pytest

from globerode.grid import Grid
from globerode.report.metrics import (
    coast_match,
    drainage,
    drainage_density_per_km,
    hack_exponent,
    height_stats,
    horton_bifurcation,
    seam_continuity,
)


def _binary_tree(depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete binary tree of cells draining to cell 0, which drains to sea cell -> index n."""
    n = 2**depth - 1
    receiver = np.empty(n + 1, dtype=np.int32)
    receiver[0] = n  # sea
    for c in range(1, n):
        receiver[c] = (c - 1) // 2
    receiver[n] = -1
    stack = np.arange(n, dtype=np.int32)  # parents before children
    land = np.ones(n + 1, dtype=bool)
    land[n] = False
    return receiver, stack, land


def test_horton_ratio_of_binary_tree_is_two() -> None:
    receiver, stack, land = _binary_tree(7)
    channel = land.copy()
    res = horton_bifurcation(receiver, stack, land, channel)
    assert res["counts"] == [64, 32, 16, 8, 4, 2, 1]
    assert res["rb"] == pytest.approx(2.0)


def test_hack_exponent_of_straight_channel_is_one() -> None:
    n = 200
    receiver = np.append(np.arange(-1, n - 1, dtype=np.int32), -1)
    receiver[0] = n  # cell 0 drains to sea cell n
    stack = np.arange(n, dtype=np.int32)
    land = np.ones(n + 1, dtype=bool)
    land[n] = False
    # Cell c receives from c+1; area counts the cells upstream including itself.
    receiver[1:n] = np.arange(0, n - 1)
    area = np.append(np.arange(n, 0, -1, dtype=float), 0.0)
    dist = np.ones(n + 1)
    res = hack_exponent(receiver, stack, area, dist, land, land & (area >= 20))
    assert res["h"] == pytest.approx(1.0, abs=0.03)  # L = A - 1 exactly


def test_drainage_detects_broken_chains() -> None:
    receiver, stack, land = _binary_tree(4)
    assert drainage(receiver, stack, land)["fraction_draining"] == 1.0
    broken = receiver.copy()
    broken[3] = -1
    assert drainage(broken, stack, land)["not_draining"] >= 1


def test_coast_match() -> None:
    land = np.array([[True, False], [False, True]])
    assert coast_match(np.where(land, 5.0, -5.0), land)["mismatched_pixels"] == 0
    assert coast_match(np.full((2, 2), 5.0), land)["mismatched_pixels"] == 2


def test_seam_ratio_is_one_for_smooth_field_and_flags_a_seam() -> None:
    grid = Grid(48)
    p = grid.xyz()
    land = np.ones(grid.ncells, dtype=bool)
    smooth = 1000 * (p[:, 0] * p[:, 2] + np.sin(3 * p[:, 1]))
    r = seam_continuity(grid, smooth, land)["ratio"]
    assert 0.85 < r < 1.15
    bumped = smooth + np.repeat(np.arange(6) * 50.0, 48 * 48)  # step between faces
    assert seam_continuity(grid, bumped, land)["ratio"] > 1.5
    rough = smooth + 300 * np.random.default_rng(0).random(grid.ncells)  # noisy but seamless
    assert 0.85 < seam_continuity(grid, rough, land)["ratio"] < 1.15


def test_density_and_height_stats() -> None:
    land = np.array([True, True, False])
    channel = np.array([True, False, False])
    d = drainage_density_per_km(
        np.array([2000.0, 1000.0, 0.0]), np.array([1e6, 1e6, 1e6]), land, channel
    )
    assert d == pytest.approx(1.0)
    st = height_stats(np.array([10.0, 20.0, -5.0]), np.array([10.0, 30.0, -5.0]), land)
    assert st["max_m"] == 20.0
    assert st["max_vs_prepared"] == pytest.approx(20 / 30, abs=1e-3)


def test_river_threshold_is_at_least_ten_cells() -> None:
    from globerode.report.metrics import river_threshold_m2

    coarse = Grid(16)  # mean cell ~ 260 000 km^2
    assert river_threshold_m2(coarse, 10_000.0) == pytest.approx(
        10 * 4 * np.pi * coarse.radius_m**2 / coarse.ncells
    )
    fine = Grid(2048)
    assert river_threshold_m2(fine, 10_000.0) == pytest.approx(1e10)


def test_seam_steps_between_flat_faces_are_flagged() -> None:
    grid = Grid(16)
    land = np.ones(grid.ncells, dtype=bool)
    stepped = np.repeat(np.arange(6) * 100.0, 16 * 16)  # flat faces at different heights
    assert seam_continuity(grid, stepped, land)["ratio"] == float("inf")
    flat = np.full(grid.ncells, 7.0)
    assert seam_continuity(grid, flat, land)["ratio"] == 1.0
