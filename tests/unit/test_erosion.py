import numpy as np
import pytest

from globerode.erode import ErosionParams, erode, uplift_from_height
from globerode.grid import FACE_FRAMES, Grid, arc, cell_fij, cell_xyz


def _cap(n: int, radius_deg: float = 50.0, seed: int = 0) -> tuple[Grid, np.ndarray, np.ndarray]:
    grid = Grid(n)
    ang = np.degrees(np.arccos(np.clip(grid.xyz()[:, 0], -1, 1)))
    land = ang < radius_deg
    rng = np.random.default_rng(seed)
    h = np.where(land, 10.0 + rng.random(grid.ncells), -3000.0)
    return grid, h, land


def _dist_m(grid: Grid, a: int, b: int) -> float:
    p = cell_xyz(*cell_fij(a, grid.n), grid.n, FACE_FRAMES)
    q = cell_xyz(*cell_fij(b, grid.n), grid.n, FACE_FRAMES)
    return float(arc(*p, *q)) * grid.radius_m


def test_reaches_stream_power_steady_state() -> None:
    grid, h0, land = _cap(24)
    u, k, m = 1e-3, 1e-5, 0.5
    params = ErosionParams(k=k, m=m, dt_yr=1e6, kappa_m2_per_yr=0.0, iterations=300)
    res = erode(grid, h0, land, np.where(land, u, 0.0), params)
    r = res.routing
    ratios = []
    for c in np.nonzero(land)[0]:
        rc = int(r.receiver[c])
        hr = res.height[rc] if land[rc] else 0.0
        slope = (res.height[c] - hr) / _dist_m(grid, int(c), rc)
        ratios.append(slope / (u / (k * r.area[c] ** m)))
    ratios_a = np.array(ratios)
    assert np.mean(np.abs(ratios_a - 1) < 0.02) > 0.99, np.percentile(ratios_a, [1, 50, 99])


def test_coast_clamp_and_sea_untouched() -> None:
    grid, h0, land = _cap(16)
    res = erode(grid, h0, land, np.zeros(grid.ncells), ErosionParams(iterations=5, dt_yr=1e7))
    assert (res.height[land] >= 1.0).all()
    assert np.array_equal(res.height[~land], h0[~land])


def test_without_uplift_erosion_only_lowers_relief() -> None:
    grid, h0, land = _cap(16)
    h0 = np.where(land, 500.0 + 1000.0 * np.random.default_rng(1).random(grid.ncells), -3000.0)
    res = erode(grid, h0, land, np.zeros(grid.ncells), ErosionParams(iterations=10))
    assert res.height[land].mean() < h0[land].mean()
    assert res.height[land].max() <= h0[land].max() + 1.0  # fills can't exceed the rim


def test_deterministic() -> None:
    grid, h0, land = _cap(16, seed=4)
    up = np.where(land, 1e-3, 0.0)
    a = erode(grid, h0, land, up, ErosionParams(iterations=5))
    b = erode(grid, h0, land, up, ErosionParams(iterations=5))
    assert np.array_equal(a.height, b.height)


def test_uplift_from_height() -> None:
    land = np.array([True, True, False])
    u = uplift_from_height(np.array([1000.0, 4000.0, -50.0]), land, 2e-4)
    assert u.tolist() == pytest.approx([2e-4, 8e-4, 0.0])
