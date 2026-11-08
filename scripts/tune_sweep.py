"""Parameter sweep for PLAN step 9 (tuning) on a test world at preview size.

Usage: uv run python scripts/tune_sweep.py WORLD_TOML OUT_DIR "key=value;key=value" ...

Each argument is one candidate: a ';'-separated list of --set overrides.
For every candidate it runs prep + erosion (no height restore, so the raw
erosion strength is visible), computes the report metrics, and writes a
hillshade crop. Results go to OUT_DIR/results.json and OUT_DIR/<k>.png.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from globerode.config import load_config
from globerode.erode.cascade import run_cascade
from globerode.report import metrics as M
from globerode.report.previews import hillshade_equirect, tint
from globerode.resample import cube_to_equirect, read_levels


def main() -> None:
    cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for k, cand in enumerate(sys.argv[3:]):
        overrides = [s for s in cand.split(";") if s] + ["erode.restore_heights=none"]
        cfg = load_config(cfg_path, overrides).preview()
        t0 = time.perf_counter()
        grid, f, r = run_cascade(cfg, read_levels(cfg.input.levels, cfg), lambda *a, **kw: None)
        land = f["land"]
        dist = M.receiver_distance_m(grid, r.receiver)
        ch = land & (r.area >= M.river_threshold_m2(grid, cfg.erode.river_threshold_km2))
        h, h0 = f["height"], f["h_init"]
        mean_ratio = float(h[land].mean() / h0[land].mean())
        rb = M.horton_bifurcation(r.receiver, r.stack, land, ch)["rb"]
        horton_rb = float(rb) if isinstance(rb, (int, float)) else float("nan")
        res = {
            "k": k,
            "set": cand,
            "seconds": round(time.perf_counter() - t0, 1),
            "mean_ratio": round(mean_ratio, 3),
            "max_ratio": round(float(h[land].max() / h0[land].max()), 3),
            "relief_p90_p10_m": round(
                float(np.percentile(h[land], 90) - np.percentile(h[land], 10)), 1
            ),
            "hack": round(M.hack_exponent(r.receiver, r.stack, r.area, dist, land, ch)["h"], 3),
            "horton": round(horton_rb, 3),
            "river_frac": round(float(ch.sum() / land.sum()), 4),
        }
        results.append(res)
        print(json.dumps(res), flush=True)
        eq = cube_to_equirect(grid, np.where(land, h, -100.0), 2048)
        crop = eq[160:560, 1000:1500]  # big continent + ridge
        sh = hillshade_equirect(eq, cfg.radius_m, exaggeration=8)[160:560, 1000:1500]
        rgb = tint(crop * max(1.0, 1.0 / max(mean_ratio, 0.1))) * (0.3 + 0.7 * sh[..., None])
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out / f"{k}.png")
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
