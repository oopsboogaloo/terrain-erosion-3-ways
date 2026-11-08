import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from globerode import grid as g
from globerode.grid import FACE_FRAMES, Grid

SIZES = [1, 2, 3, 16, 64]


def test_face_frames_are_right_handed_and_orthonormal() -> None:
    for c, u, v in FACE_FRAMES:
        assert np.allclose(np.cross(u, v), c)
        assert math.isclose(float(u @ v), 0.0)
    centres = {tuple(f[0]) for f in FACE_FRAMES}
    assert len(centres) == 6


@pytest.mark.parametrize("n", SIZES)
def test_cell_areas_sum_to_sphere(n: int) -> None:
    grid = Grid(n, radius_m=1.0)
    total = float(np.sum(g._all_areas(n)))
    assert total == pytest.approx(4 * math.pi, rel=1e-12)
    assert grid.cell_area_m2.dtype == np.float32
    assert grid.cell_area_m2.shape == (grid.ncells,)


def test_cell_area_ratio_is_about_1_4() -> None:
    areas = g._all_areas(256)
    assert 1.35 < areas.max() / areas.min() < 1.42


def test_area_scales_with_radius_squared() -> None:
    a = Grid(8, radius_m=2.0).cell_area_m2.sum(dtype=np.float64)
    assert a == pytest.approx(4 * math.pi * 4.0, rel=1e-6)


@pytest.mark.parametrize("n", SIZES)
def test_neighbours_are_symmetric_unique_and_not_self(n: int) -> None:
    nb = Grid(n).neighbours()
    for a in range(nb.shape[0]):
        row = [b for b in nb[a] if b >= 0]
        assert a not in row
        assert len(set(row)) == len(row)
        for b in row:
            assert a in nb[b], f"cell {a} -> {b} but not back"


@pytest.mark.parametrize("n", [2, 3, 16, 64])
def test_exactly_24_corner_cells_have_7_neighbours(n: int) -> None:
    nb = Grid(n).neighbours()
    missing = (nb < 0).sum(axis=1)
    assert int((missing == 1).sum()) == 24
    assert int((missing > 1).sum()) == 0
    # The missing one is always a diagonal (slots 4..7).
    assert (nb[:, :4] >= 0).all()


def test_single_cell_faces_have_only_edge_neighbours() -> None:
    nb = Grid(1).neighbours()
    assert (nb[:, :4] >= 0).all()
    assert (nb[:, 4:] < 0).all()


@pytest.mark.parametrize("n", [3, 16, 64])
def test_neighbours_are_geometrically_adjacent(n: int) -> None:
    grid = Grid(n)
    p = grid.xyz()
    nb = grid.neighbours()
    cell = (math.pi / 2) / n
    for k in range(8):
        valid = nb[:, k] >= 0
        cos = np.einsum("ij,ij->i", p[valid], p[nb[valid, k]])
        ang = np.arccos(np.clip(cos, -1, 1))
        # Near cube corners cells are skewed (gridlines meet at ~120 deg), so one
        # diagonal is ~1.61 cells long even within a face.
        limit = 1.05 * cell if k < 4 else 1.65 * cell
        assert ang.max() < limit, f"offset {k}"
        assert ang.min() > 0.6 * cell


@pytest.mark.parametrize("n", [3, 16])
def test_shared_edges_agree_from_both_sides(n: int) -> None:
    grid = Grid(n)
    nb = grid.neighbours()
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}
    for a in range(grid.ncells):
        fa, ia, ja = g.cell_fij(a, n)
        for k in range(4):
            b = int(nb[a, k])
            fb, ib, jb = g.cell_fij(b, n)
            # Find which edge of b points back at a.
            kb = next(m for m in range(4) if nb[b, m] == a)
            la = g.shared_edge_angle(fa, ia, ja, k, n, FACE_FRAMES)
            lb = g.shared_edge_angle(fb, ib, jb, kb, n, FACE_FRAMES)
            assert la == pytest.approx(lb, rel=1e-9)
            if fa == fb:
                assert kb == opposite[k]


@pytest.mark.parametrize("n", [1, 2, 7, 64])
def test_cell_centre_round_trip(n: int) -> None:
    grid = Grid(n)
    idx = grid.index_of(grid.xyz())
    assert np.array_equal(idx, np.arange(grid.ncells))


def test_lonlat_ranges_and_face_centres() -> None:
    grid = Grid(2)
    lon, lat = grid.lonlat_deg()
    assert lon.min() >= -180 and lon.max() <= 180
    assert lat.min() >= -90 and lat.max() <= 90
    # +Z face cells are all in the northern hemisphere, above 35 degrees.
    face4 = slice(4 * 4, 5 * 4)
    assert (lat[face4] > 35).all()


unit_vectors = st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)).filter(
    lambda v: math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) > 1e-3
)


@settings(max_examples=300, deadline=None)
@given(v=unit_vectors, n=st.sampled_from([1, 5, 32, 200]))
def test_point_lies_inside_its_cell(v: tuple[float, float, float], n: int) -> None:
    grid = Grid(n)
    p = np.array(v) / np.linalg.norm(v)
    idx = int(grid.index_of(p[None, :])[0])
    f, i, j = g.cell_fij(idx, n)
    c = np.array(g.cell_xyz(f, i, j, n, FACE_FRAMES))
    # Within the cell's half-diagonal of its centre (generous bound).
    half_diag = 0.75 * (math.pi / 2) / n * math.sqrt(2)
    assert math.acos(min(1.0, float(p @ c))) <= half_diag
    # And no neighbour centre is clearly closer than the cell's own centre.
    for k in range(8):
        di, dj = (int(x) for x in g.NEIGHBOUR_OFFSETS[k])
        b = g.neighbour(f, i, j, di, dj, n, g.EDGE_TABLE)
        if b < 0:
            continue
        fb, ib, jb = g.cell_fij(b, n)
        q = np.array(g.cell_xyz(fb, ib, jb, n, FACE_FRAMES))
        assert float(p @ q) <= float(p @ c) + 0.35 * ((math.pi / 2) / n) ** 2


def test_invalid_grids_rejected() -> None:
    with pytest.raises(ValueError):
        Grid(0)
    with pytest.raises(ValueError):
        Grid(4, radius_m=-1)
    with pytest.raises(ValueError):
        Grid(20_000)
