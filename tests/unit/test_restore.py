import numpy as np
import pytest

from globerode.erode.restore import block_average, local_restore
from globerode.grid import Grid


def _world(n: int) -> tuple[Grid, np.ndarray, np.ndarray, np.ndarray]:
    grid = Grid(n)
    p = grid.xyz()
    ang = np.degrees(np.arccos(np.clip(p[:, 0], -1, 1)))
    land = ang < 60
    ridge = land & (np.abs(p[:, 2]) < 0.08)  # a high band through the continent
    h_init = np.where(ridge, 6000.0, np.where(land, 800.0, -3000.0))
    return grid, land, ridge, h_init


def test_uniform_lowering_is_undone() -> None:
    grid, land, _, h_init = _world(64)
    out = local_restore(grid, np.where(land, h_init - 300.0, h_init), h_init, land, 400e3, 1.0)
    assert np.allclose(out[land], h_init[land], atol=1.0)
    assert np.array_equal(out[~land], h_init[~land])


def test_peaks_stay_where_they_were_painted() -> None:
    grid, land, ridge, h_init = _world(64)
    rng = np.random.default_rng(0)
    # Erosion lowered the ridge a lot; the plateau kept rough divides.
    h_er = np.where(ridge, 0.4 * h_init, h_init * (1 + 0.3 * rng.random(grid.ncells)))
    h_er = np.where(land, h_er, h_init)
    out = local_restore(grid, h_er, h_init, land, 300e3, 1.0)
    top = out >= np.percentile(out[land], 99)
    assert (ridge[top & land]).mean() > 0.9, "the highest cells must be on the painted ridge"
    plateau = land & ~ridge & (np.abs(grid.xyz()[:, 2]) > 0.4)
    assert out[plateau].max() < 2500.0
    # Relief is shifted, not stretched: no cell far above the painted maximum.
    assert out[land].max() < 1.15 * h_init[land].max()


def test_block_average_preserves_weighted_mean() -> None:
    grid = Grid(16)
    v = np.random.default_rng(1).random(grid.ncells)
    a = grid.cell_area_m2.astype(np.float64)
    coarse = block_average(grid, v, a, 4)
    assert coarse.shape == (6 * 4 * 4,)
    assert coarse.mean() == pytest.approx(v.mean(), rel=0.05)


def test_cap_peaks_compresses_only_above_threshold() -> None:
    from globerode.erode.restore import cap_peaks

    h_init = np.full(5, 1000.0)
    land = np.array([True, True, True, True, False])
    h = np.array([500.0, 1100.0, 1500.0, 9000.0, 9000.0])
    out = cap_peaks(h, h_init, land, ratio=1.1, soft=0.1)
    assert out[0] == 500.0 and out[1] == pytest.approx(1100.0)  # at/below threshold
    assert 1100.0 < out[2] < 1210.0 and out[3] <= 1210.0  # at most T * (1 + soft)
    assert out[2] < out[3]  # monotone
    assert out[4] == 9000.0  # sea untouched
    assert np.array_equal(cap_peaks(h, h_init, land, 0.0, 0.1), h)  # off


def test_regional_cap_keeps_local_relief() -> None:
    from globerode.erode.restore import cap_peaks, cap_peaks_regional, smooth_over_land

    grid, land, ridge, h_init = _world(64)
    rng = np.random.default_rng(0)
    relief = 600.0 * (rng.random(grid.ncells) - 0.5)
    h = np.where(land, 1.5 * h_init + relief, h_init)  # region raised 50 % + local relief
    out = cap_peaks_regional(grid, h, h_init, land, 1.1, 0.1, 300e3, 1.0)
    sigma = 300e3
    s_in = smooth_over_land(grid, h, land, sigma)
    t = 1.1 * smooth_over_land(grid, h_init, land, sigma)
    shift = h - out
    # the regional height after the shift stays under the ceiling T (1 + soft)
    assert (s_in - shift)[land].max() <= (t * 1.1)[land].max() + 1.0
    assert ((s_in - shift)[ridge] <= t[ridge] * 1.1 + 1.0).all()
    # one smooth shift per region: neighbouring cells move together, relief kept
    nb = grid.neighbours()[:, 0]
    both = ridge & (nb >= 0) & ridge[np.maximum(nb, 0)]
    assert np.abs(shift[both] - shift[nb[both]]).max() < 0.05 * shift[both].max()
    per_cell = cap_peaks(h, h_init, land, 1.1, 0.1)
    assert np.std((out - per_cell)[ridge]) > 0  # differs from the per-cell cap
    assert np.array_equal(out[~land], h[~land])  # sea untouched
    assert np.array_equal(cap_peaks_regional(grid, h, h_init, land, 0.0, 0.1, sigma, 1.0), h)
