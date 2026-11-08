"""Parameter sweep for PLAN step 9 (tuning) on a test world at preview size.

Usage: uv run python scripts/tune_sweep.py [--full] WORLD_TOML OUT_DIR "key=value;key=value" ...

Each argument is one candidate: a ';'-separated list of --set overrides.
For every candidate it runs prep + erosion, computes the report metrics, and
writes a hillshade crop. Results go to OUT_DIR/results.json and OUT_DIR/<k>.png.

By default it runs at preview size with no height restore, so the raw erosion
strength is visible. With --full it uses the config's resolution (set it with
overrides) and keeps the restore and peak cap, as a real run would.
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
    args = sys.argv[1:]
    full = args[0] == "--full"
    if full:
        args = args[1:]
    cfg_path, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for k, cand in enumerate(args[2:]):
        overrides = [s for s in cand.split(";") if s]
        if full:
            cfg = load_config(cfg_path, overrides)
        else:
            cfg = load_config(cfg_path, [*overrides, "erode.restore_heights=none"]).preview()
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
        w = cfg.resolution.output_width
        eq = cube_to_equirect(grid, np.where(land, h, -100.0), w)
        win = (slice(160 * w // 2048, 560 * w // 2048), slice(1000 * w // 2048, 1500 * w // 2048))
        crop = eq[win]  # big continent + ridge
        sh = hillshade_equirect(eq, cfg.radius_m, exaggeration=8)[win]
        rgb = tint(crop * max(1.0, 1.0 / max(mean_ratio, 0.1))) * (0.3 + 0.7 * sh[..., None])
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out / f"{k}.png")
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
