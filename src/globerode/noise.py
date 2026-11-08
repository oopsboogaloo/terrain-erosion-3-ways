"""3D gradient noise on the sphere (DESIGN §6.4, decision D3).

Our own "improved Perlin" noise, so the same seed always gives the same
world, whatever library versions are installed. Sampled at each cell's 3D
position, so there are no seams or polar pinching.

fBm octave k has wavelength ``base_m / 2**k``. ``fbm_on_grid`` can return
just a band of octaves (by wavelength) so the cascade can add coarse
octaves at the coarse level and the rest after upsampling; the bands sum to
the full fBm because every band is normalised by the full octave weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from globerode.grid import FACE_FRAMES, Grid, cell_fij, cell_xyz

# 12 gradient directions (cube edge midpoints), as in improved Perlin noise.
_GRAD = np.array(
    [
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ],
    dtype=np.float64,
)  # fmt: skip


def permutation(seed: int) -> np.ndarray:
    """Doubled permutation table (512 entries) from a seed."""
    p = np.random.default_rng(seed).permutation(256).astype(np.int64)
    return np.concatenate([p, p])


@njit(cache=True, inline="always")
def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, inline="always")
def _dot(perm: np.ndarray, h: int, x: float, y: float, z: float) -> float:
    g = perm[h] % 12
    return float(_GRAD[g, 0] * x + _GRAD[g, 1] * y + _GRAD[g, 2] * z)


@njit(cache=True)
def perlin3(perm: np.ndarray, x: float, y: float, z: float) -> float:
    """Improved Perlin noise, roughly in [-1, 1], zero at lattice points."""
    xf = math.floor(x)
    yf = math.floor(y)
    zf = math.floor(z)
    xi = int(xf) & 255
    yi = int(yf) & 255
    zi = int(zf) & 255
    x -= xf
    y -= yf
    z -= zf
    u, v, w = _fade(x), _fade(y), _fade(z)
    a = perm[xi] + yi
    aa = perm[a] + zi
    ab = perm[a + 1] + zi
    b = perm[xi + 1] + yi
    ba = perm[b] + zi
    bb = perm[b + 1] + zi
    x1 = _dot(perm, aa, x, y, z) * (1 - u) + _dot(perm, ba, x - 1, y, z) * u
    x2 = _dot(perm, ab, x, y - 1, z) * (1 - u) + _dot(perm, bb, x - 1, y - 1, z) * u
    y1 = x1 * (1 - v) + x2 * v
    x3 = _dot(perm, aa + 1, x, y, z - 1) * (1 - u) + _dot(perm, ba + 1, x - 1, y, z - 1) * u
    x4 = _dot(perm, ab + 1, x, y - 1, z - 1) * (1 - u) + _dot(perm, bb + 1, x - 1, y - 1, z - 1) * u
    y2 = x3 * (1 - v) + x4 * v
    return y1 * (1 - w) + y2 * w


@dataclass(frozen=True)
class NoiseSpec:
    seed: int = 42
    octaves: int = 8
    base_m: float = 800_000.0
    gain: float = 0.5
    warp: float = 0.3  # domain-warp offset, in units of the base wavelength


@njit(cache=True, parallel=True)
def _fbm_grid(
    n: int,
    perm: np.ndarray,
    scales: np.ndarray,
    amps: np.ndarray,
    base_scale: float,
    warp: float,
    norm: float,
) -> np.ndarray:
    out = np.empty(6 * n * n, dtype=np.float32)
    for c in prange(6 * n * n):
        f, i, j = cell_fij(c, n)
        x, y, z = cell_xyz(f, i, j, n, FACE_FRAMES)
        if warp != 0.0:
            bx, by, bz = x * base_scale, y * base_scale, z * base_scale
            dx = perlin3(perm, bx + 31.7, by + 11.3, bz + 5.9)
            dy = perlin3(perm, bx + 7.1, by + 43.9, bz + 23.3)
            dz = perlin3(perm, bx + 19.3, by + 3.7, bz + 61.1)
            k = warp / base_scale
            x, y, z = x + k * dx, y + k * dy, z + k * dz
        acc = 0.0
        for o in range(scales.shape[0]):
            s = scales[o]
            acc += amps[o] * perlin3(perm, x * s, y * s, z * s)
        out[c] = acc / norm
    return out


def fbm_on_grid(
    grid: Grid,
    spec: NoiseSpec,
    min_wavelength_m: float = 0.0,
    max_wavelength_m: float = math.inf,
) -> np.ndarray:
    """fBm (float32, roughly -1..1) with octaves whose wavelength is in
    (min_wavelength_m, max_wavelength_m]."""
    k = np.arange(spec.octaves)
    wavelengths = spec.base_m / 2.0**k
    amps_all = spec.gain**k
    norm = float(amps_all.sum())
    keep = (wavelengths > min_wavelength_m) & (wavelengths <= max_wavelength_m)
    scales = grid.radius_m / wavelengths[keep]
    if not keep.any():
        return np.zeros(grid.ncells, dtype=np.float32)
    out: np.ndarray = _fbm_grid(
        grid.n,
        permutation(spec.seed),
        scales.astype(np.float64),
        amps_all[keep].astype(np.float64),
        grid.radius_m / spec.base_m,
        spec.warp,
        norm,
    )
    return out


def resolvable_wavelength_m(grid: Grid, cells: float = 4.0) -> float:
    """Shortest wavelength a grid can show: ``cells`` cell widths."""
    return cells * grid.radius_m * (math.pi / 2) / grid.n
