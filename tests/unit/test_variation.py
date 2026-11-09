"""Varying erodibility, rainfall and uplift (D26) and the landslide slope limit (D27)."""

import math

import numpy as np
import pytest

from globerode.config import Config, VariationConfig
from globerode.erode import ErosionParams, erode
from globerode.erode.landslide import limit_slopes
from globerode.erode.routing import route
from globerode.erode.variation import erosion_fields, variation_factor
from globerode.grid import Grid
from tests.unit.test_erosion import _cap, _dist_m


def _cfg(**erode: object) -> Config:
    data = Config.model_validate(
        {
            "name": "t",
            "input": {
                "levels": "x.png",
                "classes": [
                    {"value": 0, "name": "sea", "kind": "ocean", "depth_m": -1000.0},
                    {"value": 1, "name": "land", "kind": "land", "base": 0.5},
                ],
            },
        }
    ).model_dump()
    data["erode"].update(erode)
    return Config.model_validate(data)


def test_factor_off_is_ones() -> None:
    grid = Grid(8)
    assert np.array_equal(variation_factor(grid, VariationConfig(), 1), np.ones(grid.ncells))


def test_factor_bounds_and_geometric_mean() -> None:
    grid = Grid(32)
    f = variation_factor(grid, VariationConfig(contrast=9.0, scale_km=2000.0, sharpness=2.0), 7)
    assert f.min() >= 1 / 3 - 1e-9 and f.max() <= 3 + 1e-9
    assert f.max() / f.min() > 5  # sharp: most of the range is used
    assert abs(float(np.log(f).mean())) < 0.2


def test_fields_independent_and_consistent_across_levels() -> None:
    v = {"contrast": 4.0, "scale_km": 8000.0, "octaves": 2}  # both grids resolve both
    cfg = _cfg(erodibility=v, rainfall=v, uplift=v)
    fine, coarse = erosion_fields(Grid(32), cfg), erosion_fields(Grid(16), cfg)
    assert not np.allclose(fine["erodibility"], fine["rainfall"])
    # same field sampled at two resolutions: block means agree closely
    fk = np.log(fine["erodibility"]).reshape(6, 16, 2, 16, 2).mean(axis=(2, 4)).ravel()
    assert np.corrcoef(fk, np.log(coarse["erodibility"]))[0, 1] > 0.95


def test_rainfall_weights_drainage_area() -> None:
    grid, h, land = _cap(16)
    a = route(grid, h, land)
    b = route(grid, h, land, rainfall=np.full(grid.ncells, 2.0))
    assert np.allclose(b.area, 2 * a.area)


def test_slope_limit_caps_every_slope() -> None:
    grid, _, land = _cap(24)
    rng = np.random.default_rng(3)
    h = np.where(land, 1.0 + 20_000.0 * rng.random(grid.ncells), -3000.0)
    h = route(grid, h, land).filled
    angle = 5.0
    for _ in range(3):
        limit_slopes(grid, h, land, route(grid, h, land).stack, angle)
    s_max = math.tan(math.radians(angle))
    nb = grid.neighbours()
    worst = 0.0
    for c in np.nonzero(land)[0]:
        for b in nb[c]:
            if b < 0:
                continue
            hb = h[b] if land[b] else 0.0
            worst = max(worst, (h[c] - hb) / _dist_m(grid, int(c), int(b)))
    assert worst <= s_max * (1 + 1e-9), worst / s_max


def test_slope_limit_off_or_loose_changes_nothing() -> None:
    grid, h0, land = _cap(16)
    stack = route(grid, h0, land).stack
    h = h0.copy()
    assert limit_slopes(grid, h, land, stack, 0.0) == 0
    assert limit_slopes(grid, h, land, stack, 89.0) == 0
    assert np.array_equal(h, h0)


def test_erode_with_fields_is_deterministic_and_uses_them() -> None:
    grid, h0, land = _cap(16, seed=4)
    h0 = np.where(land, 2000.0 + h0, h0)
    up = np.where(land, 1e-3, 0.0)
    k = variation_factor(grid, VariationConfig(contrast=10.0, scale_km=3000.0), 5)
    p = ErosionParams(iterations=5, max_slope_deg=20.0)
    a = erode(grid, h0, land, up, p, erodibility=k, rainfall=k)
    b = erode(grid, h0, land, up, p, erodibility=k, rainfall=k)
    c = erode(grid, h0, land, up, p)
    assert np.array_equal(a.height, b.height)
    assert not np.allclose(a.height, c.height)


def test_partial_variation_table_keeps_field_defaults() -> None:
    cfg = _cfg(erodibility={"scale_km": 500.0}, rainfall={"contrast": 2.0})
    assert cfg.erode.erodibility.scale_km == 500.0
    assert cfg.erode.erodibility.contrast == 8.0  # not reset to 1 (off)
    assert cfg.erode.erodibility.sharpness == 1.5
    assert cfg.erode.rainfall.scale_km == 1500.0
    assert cfg.erode.uplift.contrast == 3.0


def test_strata_factor_bands() -> None:
    from globerode.config import StrataConfig
    from globerode.erode.variation import strata_for

    grid = Grid(8)
    assert strata_for(grid, StrataConfig(contrast=1.0), 1) is None
    st = strata_for(grid, StrataConfig(contrast=9.0, thickness_m=800.0, fold_amp_m=0.0), 1)
    assert st is not None
    h = np.linspace(0.0, 1600.0, 401)[:, None] * np.ones((1, grid.ncells))
    f = np.stack([st.factor(row) for row in h])
    assert f.min() == pytest.approx(1 / 3) and f.max() == pytest.approx(3.0)
    assert np.allclose(st.factor(h[0] + 800.0), st.factor(h[0]))  # one band pair = 800 m
    folded = strata_for(
        grid, StrataConfig(contrast=9.0, fold_amp_m=1000.0, fold_scale_km=8000.0), 1
    )
    assert folded is not None and folded.fold_m.std() == pytest.approx(1000.0)


def test_strata_change_erosion() -> None:
    from globerode.config import StrataConfig
    from globerode.erode.variation import strata_for

    grid, h0, land = _cap(16, seed=4)
    h0 = np.where(land, 2000.0 + h0, h0)
    up = np.where(land, 1e-3, 0.0)
    st = strata_for(grid, StrataConfig(contrast=8.0, thickness_m=300.0), 1)
    p = ErosionParams(iterations=5)
    a = erode(grid, h0, land, up, p, strata=st)
    b = erode(grid, h0, land, up, p)
    assert np.array_equal(a.height, erode(grid, h0, land, up, p, strata=st).height)
    assert not np.allclose(a.height, b.height)


def test_strata_follow_accumulated_uplift() -> None:
    from globerode.config import StrataConfig
    from globerode.erode.variation import strata_for

    grid, h0, land = _cap(16, seed=4)
    h0 = np.where(land, 2000.0 + h0, h0)
    up = np.where(land, 1e-3, 0.0)
    st = strata_for(grid, StrataConfig(contrast=8.0, thickness_m=400.0, fold_amp_m=0.0), 1)
    p = ErosionParams(iterations=4)

    def run(before: float) -> np.ndarray:
        return erode(
            grid, h0, land, up, p, strata=st, uplift_before_m=np.full(grid.ncells, before)
        ).height

    base = run(0.0)
    assert np.allclose(run(400.0), base)  # one full band pair: same rock layers
    assert not np.allclose(run(200.0), base)  # half a band: hard and soft swap
