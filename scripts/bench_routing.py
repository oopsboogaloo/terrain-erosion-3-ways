"""Routing speed check (PLAN step 3).

Usage: uv run python scripts/bench_routing.py 512 2048
"""

import resource
import sys
import time

import numpy as np

from globerode.erode.routing import route
from globerode.grid import Grid


def world(n: int, kind: str) -> tuple[Grid, np.ndarray, np.ndarray]:
    g = Grid(n)
    p = g.xyz()
    land = p[:, 0] > -0.3  # ~65% land, one big ocean
    rng = np.random.default_rng(0)
    if kind == "white":
        h = 1.0 + 3000 * rng.random(g.ncells)
    else:  # smooth-ish: a few sinusoids plus small noise
        h = (
            1500
            + 800 * np.sin(7 * p[:, 0]) * np.cos(5 * p[:, 1])
            + 500 * np.sin(13 * p[:, 2] + 2)
            + 50 * rng.random(g.ncells)
        )
    return g, np.where(land, np.maximum(h, 1.0), -3000.0), land


def main() -> None:
    route(*world(8, "white"))  # JIT warm-up
    for n in [int(a) for a in sys.argv[1:]]:
        for kind in ["smooth", "white"]:
            g, h, land = world(n, kind)
            _ = g.cell_area_m2
            t = time.perf_counter()
            route(g, h, land)
            dt = time.perf_counter() - t
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"N={n} cells={g.ncells / 1e6:.1f}M {kind}: {dt:.1f} s  maxrss={rss_gb:.2f} GB")


if __name__ == "__main__":
    main()
