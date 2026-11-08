"""Named arrays on a cube-sphere grid (DESIGN §4)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from globerode.grid import Grid


@dataclass
class Fields:
    """Named flat arrays, one value per grid cell.

    Units throughout: metres, m^2, years.
    """

    grid: Grid
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def __setitem__(self, name: str, values: np.ndarray) -> None:
        arr = np.ascontiguousarray(values)
        if arr.shape != (self.grid.ncells,):
            raise ValueError(
                f"field {name!r} has shape {arr.shape}, expected ({self.grid.ncells},)"
            )
        self.arrays[name] = arr

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError:
            have = ", ".join(sorted(self.arrays)) or "none"
            raise KeyError(f"field {name!r} not present (have: {have})") from None

    def __contains__(self, name: object) -> bool:
        return name in self.arrays

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def drop(self, *names: str) -> None:
        """Free fields no later stage needs."""
        for name in names:
            self.arrays.pop(name, None)

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())
