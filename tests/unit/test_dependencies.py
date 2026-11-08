"""Smoke test: every runtime dependency imports and the key features work."""

import numpy as np


def test_numba_compiles_a_kernel() -> None:
    from numba import njit

    @njit(cache=False)
    def add_one(a: np.ndarray) -> np.ndarray:
        return a + 1.0

    assert add_one(np.zeros(3)).tolist() == [1.0, 1.0, 1.0]


def test_image_writers_import() -> None:
    import OpenEXR
    import tifffile
    from PIL import Image

    assert OpenEXR is not None
    assert tifffile is not None
    assert Image is not None


def test_scipy_pchip_available() -> None:
    from scipy.interpolate import PchipInterpolator

    f = PchipInterpolator([0.0, 1.0], [0.0, 2.0])
    assert float(f(0.5)) == 1.0
