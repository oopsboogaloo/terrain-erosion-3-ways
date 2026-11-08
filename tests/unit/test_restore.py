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
