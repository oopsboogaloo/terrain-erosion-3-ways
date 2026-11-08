"""Spatially varying erosion parameters (D26).

Real landscapes are uneven because rock hardness, rainfall and uplift
change from place to place. Each is modelled as a multiplicative factor
from smooth warped fBm::

    z = fbm / std(fbm)                 # unit variance
    v = tanh(sharpness * z)            # -1..1; high sharpness -> patchy bands
    factor = contrast ** (v / 2)       # 1/sqrt(contrast) .. sqrt(contrast)

Every field has its own seed (offset from ``prep.noise.seed``), so maps
are reproducible and the fields are independent. The noise is sampled at
fixed positions on the sphere, so each cascade level sees the same field.
"""

from __future__ import annotations

import numpy as np

from globerode.config import Config, VariationConfig
from globerode.grid import Grid
from globerode.noise import NoiseSpec, fbm_on_grid, resolvable_wavelength_m

# Seed offsets so the fields differ from each other and from the height noise.
SEED_OFFSETS = {"erodibility": 101, "rainfall": 202, "uplift": 303}


def variation_factor(grid: Grid, var: VariationConfig, seed: int) -> np.ndarray:
    """Per-cell multiplicative factor (float64); all ones when contrast is 1."""
    if var.contrast <= 1.0:
        return np.ones(grid.ncells, dtype=np.float64)
    spec = NoiseSpec(
        seed=seed, octaves=var.octaves, base_m=var.scale_km * 1000.0, gain=0.5, warp=0.3
    )
    z = fbm_on_grid(grid, spec, min_wavelength_m=resolvable_wavelength_m(grid)).astype(np.float64)
    std = float(z.std())
    z = (z - float(z.mean())) / (std if std > 0 else 1.0)
    v = np.tanh(var.sharpness * z)
    out: np.ndarray = var.contrast ** (v / 2.0)
    return out


def erosion_fields(grid: Grid, cfg: Config) -> dict[str, np.ndarray]:
    """Factors for erodibility (K), rainfall (drainage weight) and uplift."""
    seed = cfg.prep.noise.seed
    e = cfg.erode
    return {
        "erodibility": variation_factor(grid, e.erodibility, seed + SEED_OFFSETS["erodibility"]),
        "rainfall": variation_factor(grid, e.rainfall, seed + SEED_OFFSETS["rainfall"]),
        "uplift": variation_factor(grid, e.uplift, seed + SEED_OFFSETS["uplift"]),
    }
