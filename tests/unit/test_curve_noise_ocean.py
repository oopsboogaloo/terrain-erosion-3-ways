import math

import numpy as np
import pytest

from globerode.curve import HeightCurve
from globerode.grid import Grid
from globerode.noise import NoiseSpec, fbm_on_grid, perlin3, permutation, resolvable_wavelength_m
from globerode.ocean import ocean_floor, shelf_ramp

# --- curve -------------------------------------------------------------------------

POINTS = [(0.0, 0.0), (0.2, 150.0), (0.5, 900.0), (0.8, 3500.0), (1.0, 7500.0)]


def test_curve_passes_through_points_and_is_monotone() -> None:
    c = HeightCurve(POINTS)
    for x, y in POINTS:
        assert c(np.array([x]))[0] == pytest.approx(y)
    xs = np.linspace(0, 1, 2001)
    assert (np.diff(c(xs)) >= 0).all()


def test_curve_clamps_outside_range() -> None:
    c = HeightCurve(POINTS)
    assert c(np.array([-1.0, 2.0])) == pytest.approx([0.0, 7500.0])


@pytest.mark.parametrize("bad", [[(0, 0)], [(0, 0), (0, 1)], [(0, 5), (1, 1)]])
def test_curve_rejects_bad_points(bad: list[tuple[float, float]]) -> None:
    with pytest.raises(ValueError):
        HeightCurve(bad)


# --- noise -------------------------------------------------------------------------


def test_perlin_is_zero_on_lattice_and_bounded() -> None:
    perm = permutation(1)
    assert perlin3(perm, 3.0, -2.0, 7.0) == 0.0
    rng = np.random.default_rng(0)
    vals = np.array([perlin3(perm, *rng.uniform(-50, 50, 3)) for _ in range(5000)])
    assert np.abs(vals).max() <= 1.05
    assert abs(vals.mean()) < 0.02
    assert 0.1 < vals.std() < 0.4


def test_fbm_repeatable_and_seed_dependent() -> None:
    grid = Grid(24)
    a = fbm_on_grid(grid, NoiseSpec(seed=7))
    assert np.array_equal(a, fbm_on_grid(grid, NoiseSpec(seed=7)))
    assert not np.array_equal(a, fbm_on_grid(grid, NoiseSpec(seed=8)))
    assert np.abs(a).max() < 1.0


def test_fbm_bands_sum_to_full() -> None:
    grid = Grid(32)
    spec = NoiseSpec(octaves=6, base_m=2_000_000.0)
    split = 300_000.0
    coarse = fbm_on_grid(grid, spec, min_wavelength_m=split)
    fine = fbm_on_grid(grid, spec, max_wavelength_m=split)
    full = fbm_on_grid(grid, spec)
    assert np.allclose(coarse + fine, full, atol=1e-6)
    assert not np.allclose(fine, 0)


def test_fbm_is_smooth_across_cube_edges() -> None:
    grid = Grid(64)
    v = fbm_on_grid(grid, NoiseSpec(octaves=3, base_m=3_000_000.0)).astype(np.float64)
    nb = grid.neighbours()
    diffs = np.abs(v[:, None] - v[np.where(nb >= 0, nb, np.arange(grid.ncells)[:, None])])
    face = np.arange(grid.ncells) // (64 * 64)
    cross = face[:, None] != face[np.where(nb >= 0, nb, 0)]
    # Neighbour differences across face edges look like those inside faces.
    assert diffs[cross].max() < 1.5 * diffs[~cross].max()


def test_resolvable_wavelength() -> None:
    grid = Grid(100, radius_m=1.0)
    assert resolvable_wavelength_m(grid) == pytest.approx(4 * (math.pi / 2) / 100)


# --- ocean -------------------------------------------------------------------------


def test_shelf_ramp_profile() -> None:
    d = np.array([0.0, 40e3, 80e3, 155e3, 230e3, 1e6])
    r = shelf_ramp(d, -130.0, 80e3, 150e3, -4000.0)
    assert r[0] == 0.0
    assert r[1] == pytest.approx(-65.0)
    assert r[2] == pytest.approx(-130.0)
    assert r[3] == pytest.approx(-130 + (-4000 + 130) * 0.5)
    assert r[4] == pytest.approx(-4000.0) and r[5] == pytest.approx(-4000.0)
    assert (np.diff(r) <= 0).all()


def test_ocean_is_never_deeper_than_ramp_near_coast() -> None:
    d = np.linspace(0, 500e3, 200)
    painted = np.full(200, -4500.0)
    h = ocean_floor(
        painted,
        d,
        np.zeros(200),
        shelf_depth_m=-130,
        shelf_width_m=80e3,
        slope_width_m=150e3,
        floor_noise_m=150,
        max_sea_height_m=-1,
    )
    ramp = shelf_ramp(d, -130, 80e3, 150e3, -4500)
    assert (h >= np.minimum(ramp, -1.0) - 1e-9).all()  # shallow-clamped at -1 m
    assert (h <= -1).all()
    assert h[-1] == pytest.approx(-4500.0)
