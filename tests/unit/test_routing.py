import numpy as np
import pytest

from globerode.erode.routing import route
from globerode.grid import Grid


def _cap_world(
    n: int, seed: int = 0, radius_deg: float = 50.0
) -> tuple[Grid, np.ndarray, np.ndarray]:
    """A round continent at lon=0, lat=0 with rough random heights."""
    grid = Grid(n)
    p = grid.xyz()
    ang = np.degrees(np.arccos(np.clip(p[:, 0], -1, 1)))
    land = ang < radius_deg
    rng = np.random.default_rng(seed)
    h = np.where(land, 1.0 + 2000.0 * rng.random(grid.ncells), -3000.0)
    return grid, h, land


def _check_invariants(
    grid: Grid, h: np.ndarray, land: np.ndarray, eps: float = 1e-3, rho8_seed: int | None = None
) -> None:
    r = route(grid, h, land, epsilon_m=eps, rho8_seed=rho8_seed)
    land_idx = np.nonzero(land)[0]

    # Every land cell appears exactly once in the stack.
    assert np.array_equal(np.sort(r.stack), land_idx)

    # Filling only raises land; sea is untouched.
    assert (r.filled[land] >= h[land]).all()
    assert np.array_equal(r.filled[~land], h[~land])

    # Each land cell drains to a strictly lower cell (sea counts as 0 m).
    rec = r.receiver[land_idx]
    assert (rec >= 0).all()
    h_rec = np.where(land[rec], r.filled[rec], 0.0)
    assert (h_rec < r.filled[land_idx]).all()

    # Receivers are neighbours.
    nb = grid.neighbours()
    assert all(rec[k] in nb[c] for k, c in enumerate(land_idx))

    # Stack order: every land receiver comes before its donor.
    pos = np.full(grid.ncells, -1)
    pos[r.stack] = np.arange(r.stack.size)
    land_rec = land[rec]
    assert (pos[rec[land_rec]] < pos[land_idx[land_rec]]).all()

    # Drainage area: outlets carry all land area; nothing at sea.
    outlets = land_idx[~land[rec]]
    total = grid.cell_area_m2[land].sum(dtype=np.float64)
    assert r.area[outlets].sum() == pytest.approx(total, rel=1e-9)
    assert (r.area[~land] == 0).all()
    assert (r.area[land] >= grid.cell_area_m2[land] * (1 - 1e-6)).all()


@pytest.mark.parametrize("n", [4, 16, 48])
@pytest.mark.parametrize("rho8_seed", [None, 7])
def test_invariants_on_rough_continent(n: int, rho8_seed: int | None) -> None:
    grid, h, land = _cap_world(n)
    _check_invariants(grid, h, land, rho8_seed=rho8_seed)


def test_invariants_with_flat_plateau() -> None:
    grid, _, land = _cap_world(24)
    h = np.where(land, 500.0, -100.0)
    _check_invariants(grid, h, land)


def test_continent_across_seam_and_pole() -> None:
    grid = Grid(24)
    lon, lat = grid.lonlat_deg()
    land = (lat > 40) | ((np.abs(lon) > 150) & (np.abs(lat) < 30))
    h = np.where(land, 10.0 + (90 - np.abs(lat)) * 20, -1000.0)
    _check_invariants(grid, h, land)


def test_pit_is_filled_to_its_spill_level() -> None:
    grid, _, land = _cap_world(32, radius_deg=60)
    p = grid.xyz()
    ang = np.degrees(np.arccos(np.clip(p[:, 0], -1, 1)))
    # A crater: rim at 30-40 deg high, floor inside lower than the rim.
    h = np.where(land, 100.0, -500.0)
    h[(ang > 30) & (ang < 40)] = 1000.0
    h[ang <= 30] = 200.0
    r = route(grid, h, land)
    inside = ang <= 25
    # The floor must be raised to at least the rim height.
    assert (r.filled[inside] >= 1000.0).all()
    lake = r.filled - h
    assert lake[inside].min() >= 800.0
    _check_invariants(grid, h, land)


def test_deterministic() -> None:
    grid, h, land = _cap_world(32, seed=3)
    a = route(grid, h, land)
    b = route(grid, h, land)
    for x, y in [
        (a.filled, b.filled),
        (a.receiver, b.receiver),
        (a.stack, b.stack),
        (a.area, b.area),
    ]:
        assert np.array_equal(x, y)


def test_all_land_world_is_rejected() -> None:
    grid = Grid(4)
    with pytest.raises(ValueError, match="cannot reach the sea"):
        route(grid, np.ones(grid.ncells), np.ones(grid.ncells, dtype=bool))


def test_shape_and_epsilon_validation() -> None:
    grid = Grid(4)
    with pytest.raises(ValueError):
        route(grid, np.ones(3), np.ones(3, dtype=bool))
    with pytest.raises(ValueError):
        route(grid, np.ones(grid.ncells), np.ones(grid.ncells, dtype=bool), epsilon_m=0)


def test_rho8_breaks_parallel_flow_on_a_plane() -> None:
    # A plane tilted at ~30 deg to the grid on face 0: D8 sends every cell the
    # same way (parallel channels); Rho8 mixes the two nearest directions.
    grid = Grid(48)
    p = grid.xyz()
    land = p[:, 0] > 0.6  # a cap around the centre of face 0
    h = np.where(land, 1000.0 + 5e5 * (0.87 * p[:, 1] + 0.5 * p[:, 2]), -100.0)
    interior = land & (p[:, 0] > 0.8)
    nb = grid.neighbours()

    def modal_share(seed: int | None) -> float:
        rec = route(grid, h, land, rho8_seed=seed).receiver
        k = np.array([int(np.nonzero(nb[c] == rec[c])[0][0]) for c in np.nonzero(interior)[0]])
        return float(np.bincount(k).max() / k.size)

    assert modal_share(None) > 0.95
    assert modal_share(3) < 0.8
    a = route(grid, h, land, rho8_seed=3).receiver
    assert np.array_equal(a, route(grid, h, land, rho8_seed=3).receiver)  # repeatable
    assert not np.array_equal(a, route(grid, h, land, rho8_seed=4).receiver)
