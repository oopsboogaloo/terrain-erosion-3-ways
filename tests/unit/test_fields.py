import numpy as np
import pytest

from globerode.fields import Fields
from globerode.grid import Grid


def test_set_get_drop() -> None:
    f = Fields(Grid(2))
    f["h"] = np.zeros(24, dtype=np.float32)
    assert "h" in f
    assert f["h"].dtype == np.float32
    assert list(f) == ["h"]
    assert f.nbytes() == 96
    f.drop("h", "missing")
    assert "h" not in f


def test_wrong_shape_rejected() -> None:
    f = Fields(Grid(2))
    with pytest.raises(ValueError, match="expected"):
        f["h"] = np.zeros(10)


def test_missing_field_error_lists_present() -> None:
    f = Fields(Grid(2))
    f["a"] = np.zeros(24)
    with pytest.raises(KeyError, match="have: a"):
        _ = f["b"]
