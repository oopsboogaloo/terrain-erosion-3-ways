import math

import numpy as np
import pytest

from globerode.coast import coast_distance
from globerode.grid import Grid


def _cap(grid: Grid, lon0: float, lat0: float, radius_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lo, la = math.radians(lon0), math.radians(lat0)
    c = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    ang = np.arccos(np.clip(grid.xyz() @ c, -1, 1))
    exact = (math.radians(radius_deg) - ang) * grid.radius_m
    return ang < math.radians(radius_deg), exact


@pytest.mark.parametrize(("lon0", "lat0"), [(0.0, 0.0), (0.0, 90.0), (180.0, 0.0), (45.0, 35.264)])
def test_matches_exact_distance_to_a_circle(lon0: float, lat0: float) -> None:
    grid = Grid(48)
    land, exact = _cap(grid, lon0, lat0, 30.0)
    d = coast_distance(grid, land, cap_m=1e9)
    cell = grid.radius_m * (math.pi / 2) / grid.n
    err = np.abs(d - exact)
    assert err.max() < 0.75 * cell, f"max error {err.max() / cell:.2f} cells"
    assert np.median(err) < 0.3 * cell
    assert (d[land] > 0).all() and (d[~land] < 0).all()


def test_cap_clips_far_values() -> None:
    grid = Grid(32)
    land, _ = _cap(grid, 0, 0, 30.0)
    d = coast_distance(grid, land, cap_m=500_000)
    assert d.max() == pytest.approx(500_000)
    assert d.min() == pytest.approx(-500_000)


def test_no_coast_gives_cap_everywhere() -> None:
    grid = Grid(8)
    d = coast_distance(grid, np.ones(grid.ncells, bool), cap_m=1000)
    assert np.allclose(d, 1000)


def test_shape_check() -> None:
    with pytest.raises(ValueError):
        coast_distance(Grid(4), np.ones(5, bool), 1.0)
