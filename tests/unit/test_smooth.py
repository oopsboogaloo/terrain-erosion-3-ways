import math

import numpy as np
import pytest

from globerode.grid import Grid
from globerode.smooth import coast_ramp, face_weights, gaussian_smooth, smoothstep


def test_weights_identical_on_every_face() -> None:
    w0 = face_weights(12, 0)
    for f in range(1, 6):
        assert np.allclose(face_weights(12, f), w0, rtol=1e-6)


def _impulse_width(grid: Grid, lon: float, lat: float, sigma_m: float) -> float:
    lo, la = math.radians(lon), math.radians(lat)
    target = np.array([[math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    c = int(grid.index_of(target)[0])
    v = np.zeros(grid.ncells)
    v[c] = 1.0 / grid.cell_area_m2[c]
    out = gaussian_smooth(grid, v, np.ones(grid.ncells, bool), sigma_m)
    p = grid.xyz()
    theta = np.arccos(np.clip(p @ p[c], -1, 1)) * grid.radius_m
    mass = out * grid.cell_area_m2
    # 2-D Gaussian: E[r^2] = 2 sigma^2
    return float(math.sqrt((mass * theta**2).sum() / mass.sum() / 2))


@pytest.mark.parametrize("lat", [0.0, 60.0, 85.0])
def test_blur_width_in_km_is_the_same_at_any_latitude(lat: float) -> None:
    grid = Grid(128)
    sigma = 600_000.0
    est = _impulse_width(grid, 30.0, lat, sigma)
    assert est == pytest.approx(sigma, rel=0.05)


def test_mass_is_conserved_and_constant_stays_constant() -> None:
    grid = Grid(32)
    rng = np.random.default_rng(0)
    v = rng.random(grid.ncells)
    out = gaussian_smooth(grid, v, np.ones(grid.ncells, bool), 300_000.0)
    a = grid.cell_area_m2.astype(np.float64)
    assert (out * a).sum() == pytest.approx((v * a).sum(), rel=1e-9)
    assert out.std() < v.std() / 3
    const = gaussian_smooth(grid, np.full(grid.ncells, 2.5), np.ones(grid.ncells, bool), 3e5)
    assert np.allclose(const, 2.5)


def test_mask_stops_leaks_across_the_coast() -> None:
    grid = Grid(32)
    land = grid.xyz()[:, 0] > 0.2
    v = np.where(land, 1.0, -1000.0)
    out = gaussian_smooth(grid, v, land, 500_000.0)
    assert np.allclose(out[land], 1.0)  # sea values never leak in
    assert np.array_equal(out[~land], v[~land])  # outside the mask: untouched


def test_zero_sigma_is_identity_and_shapes_checked() -> None:
    grid = Grid(4)
    v = np.arange(grid.ncells, dtype=float)
    assert np.array_equal(gaussian_smooth(grid, v, np.ones(grid.ncells, bool), 0.0), v)
    with pytest.raises(ValueError):
        gaussian_smooth(grid, v[:3], np.ones(3, bool), 1.0)


def test_coast_ramp() -> None:
    d = np.array([0.0, 5_000.0, 10_000.0, 50_000.0])
    out = coast_ramp(np.full(4, 1.0), d, coast_value=0.1, ramp_m=10_000.0)
    assert out[0] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.55)
    assert out[2] == pytest.approx(1.0) and out[3] == pytest.approx(1.0)
    assert smoothstep(0, 1, np.array([-1.0, 2.0])).tolist() == [0.0, 1.0]
