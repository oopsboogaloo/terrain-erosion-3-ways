# globerode — MVP build plan (v0.2, review policy A)

How the MVP in [SPEC.md](SPEC.md) §2 gets built, following
[DESIGN.md](DESIGN.md). Ten steps. Each step is **one pull request** that
leaves the tool working and fully tested.

## How each step is run

1. Claude writes the code and its tests together. Every function gets a
   known-answer or property test, as listed in DESIGN §13.
2. Before opening the PR, Claude runs the full test suite (including slow
   tests) in the cloud sandbox, plus the step's own checks listed below.
3. The PR description lists what was built, the test results, timings,
   and preview images where relevant.
4. If the code had to depart from the design, the PR updates DESIGN.md
   and adds an entry to `docs/decisions.md`. Anything that adds scope or
   complexity is raised as a proposal first, under the complexity gate
   (SPEC §3.2).
5. CI must be green. Then the PR is merged according to the review policy
   (§3).

## 1. Steps

### Step 1: Project skeleton and CI

- `pyproject.toml` (uv, Python 3.13, entry point `globerode`), `uv.lock`,
  ruff and mypy configuration, pytest setup with a `slow` marker.
- Empty package layout from DESIGN §2. `globerode --version` works.
- `.github/workflows/ci.yml`: ruff, mypy, `pytest -m "not slow"`.
- `.gitignore` gains `runs/` and caches.

**Done when**: CI is green on the PR. `uv sync && uv run globerode
--version` works from a fresh clone in the sandbox.

### Step 2: Cube-sphere grid

- `grid.py`: face frames, index ↔ (face,i,j), xyz, lon/lat, inverse
  (xyz → cell), 8 neighbours with cross-edge and corner handling,
  great-circle distances, shared edge lengths, exact cell areas (D1).
- `fields.py`: the `Fields` container.

**Tests**:
- areas sum to 4πR²;
- neighbour symmetry at N = 1, 2, 3, 16, 64;
- exactly 24 corner cells have 7 neighbours;
- round trip xyz → cell → xyz;
- Hypothesis: random points and rotations;
- face frames are right-handed.

**Check**: a preview image of face IDs and the cell-area ratio on an
equirect map, to confirm the layout by eye.

**Done when**: all tests pass; the face-ID image shows 6 faces with no gaps.

### Step 3: Routing, plus the speed check

Pulled forward because it is the biggest risk (DESIGN §15).

- `erode/routing.py`: Priority-Flood+ε with index tie-breaks, stack
  order, steepest-descent receiver refinement, drainage area, `lake_fill`.

**Tests**:
- no pits remain;
- every receiver is lower;
- the stack is a valid order;
- outlet areas sum to the land area;
- deterministic.

**Speed check**: routing on a random-noise globe at N = 512 and 2048 in
the sandbox. Record time per iteration and peak memory.

**Decision point**: the full 8K run needs about 100 routing iterations
at N=2048. If one takes more than ~60 s (so more than ~2 h of routing),
Claude stops and brings options to the user before going further.

**Done when**: tests pass; timings are recorded in `docs/decisions.md`.

### Step 4: Walking skeleton (input → output, end to end)

The thinnest version of the whole pipeline, so every later step plugs
into something that already runs.

- `config.py`: pydantic models for the MVP config sections, TOML loading,
  `--set` overrides, clear error messages.
- `testworlds.py`: all test worlds from DESIGN §13.2, written as equirect
  PNG + config.
- `resample.py`: categorical equirect → cube (4×4 majority), halo-padded
  faces, bicubic cube → equirect with polar supersampling.
- `export.py`: EXR / PNG16 / TIFF writers, coast re-lock, `metadata.json`.
- `pipeline.py` + `cli.py`: `run`, `testworld`, `estimate`. Run directory,
  JSON log, stdout summary, exit codes.
- At this step, height = class base value with no smoothing or erosion.
  A first `hillshade_equirect.png` preview is written.

**Tests**:
- analytic field (Y₂₁) round trip with no seam jumps;
- coast re-lock exact on `islands`;
- unknown pixel values are rejected with a clear message;
- determinism;
- CLI integration test on every test world at N=64.

**Done when**: `globerode run` on `mini_earth` produces all files in a run
directory, and the coast matches exactly.

### Step 5: Prep, giving a smoothed globe with no erosion

- `coast.py`: nearest-coast-point Dijkstra, signed distance, capped search.
- `smooth.py`: masked diffusion (D2) plus the coast ramp.
- `curve.py`: PCHIP height curve.
- `noise.py`: own 3D gradient noise with fBm, warp and octave split (D3).
- `ocean.py`: painted ocean depths, shelf ramp, deep-floor noise.

**Tests**:
- coast distance vs the exact circle on `cap_equator`, within half a cell;
- smoothing width in km matches at 0°, 60° and 85°;
- the curve is monotone and passes through its points;
- noise statistics and repeatability;
- the ocean is never deeper than the ramp near the coast.

**Check**: hillshaded globe previews of `mini_earth` and both poles. They
should show no polar pinching and no seams.

**Done when**: tests pass; this already gives SPEC milestone (b), a usable
smoothed globe.

### Step 6: Erosion

- `erode/stream_power.py`: Braun–Willett with n = 1, sea level as base
  level (D5).
- `erode/diffusion.py`: hillslope diffusion.
- `erode/__init__.py`: iteration loop, coast clamp, uplift ∝ height,
  hypsometry restore (D6), cascade 512 → 2048 with the fine noise octaves.
- `mask_flow`, `mask_rivers`, `mask_erosion`.

**Tests**:
- steady state matches S = (U/K)·A^(−m) within 2%;
- the coast clamp holds;
- hypsometry restore keeps every downhill direction and matches the
  height percentiles;
- the cascade's fine coast equals the prep coast;
- determinism.

**Check**: preview of `mini_earth` and the polar pair at N=512.

**Done when**: tests pass; the previews show branching river valleys.

### Step 7: Quality report and previews

- `report/metrics.py`: the metrics in DESIGN §12.1 (coast match, drainage,
  seam continuity, Hack exponent, Horton ratio, drainage density,
  hypsometry, relief, timings).
- `report/previews.py`: globe views from 6 directions, close-ups of both
  poles, the ±180° seam and a cube corner, mask overlay, curve and
  histogram.
- `report.json` + `report.md`; exit code 1 on failed checks;
  `globerode report` and `globerode compare` commands.

**Tests**:
- each metric on a case with a known answer (e.g. a synthetic binary tree
  for the Horton ratio, straight channels for Hack);
- polar consistency: `cap_equator` vs `cap_pole` metrics agree within
  tolerance.

**Done when**: every test world gets a report, and Claude has reviewed the
images.

### Step 8: Blender

- `blender/globe_script.py` template, written into each run as
  `blender_globe.py` (DESIGN §12.3).
- `globerode render` through the `bpy` module (optional extra).

**Tests** (slow): a 256 px render of a test world is not blank, the globe
outline is the expected size, and land and sea differ.

**Done when**: a render of `mini_earth` looks right to Claude, and the
script also runs as `python blender_globe.py`.

### Step 9: Tuning defaults

- Sweep K, Δt, u_scale, κ and iteration counts on `mini_earth` and the
  polar pair at N=512, using `compare`. Aim: report metrics in range, and
  previews that read as eroded.
- Final defaults go into config and `docs/decisions.md` with the reasons
  and the metrics.

**Done when**: all test worlds pass every report check with the new
defaults.

### Step 10: Full 8K run, docs, MVP sign-off

- An 8K run of `mini_earth` in the sandbox, in the background. Measured
  per-stage timings and memory replace the estimates in SPEC §13.
- README section: install, run, outputs, Blender, with a worked example.
- SPEC §2.3 checklist ticked with evidence (report, renders).

**Done when**: SPEC §2.3 is met; you review the result images and the
Blender render.

## 2. Order and dependencies

```
1 skeleton ─► 2 grid ─► 3 routing (speed check)
                  └───► 4 walking skeleton ─► 5 prep ─► 6 erosion ─► 7 report ─► 9 tuning ─► 10 8K + sign-off
                                                                 └─► 8 blender ─┘
```

Steps 3 and 4 both need only step 2. Step 8 needs only step 4's outputs,
but is placed after 7 so renders can be checked alongside the reports.

## 3. Review policy

**Chosen: A** (user, 2026-11-08).

Options considered for the implementation PRs:

- **A (recommended)**: you review the milestone PRs (**4** walking
  skeleton, **5** smoothed globe, **6** erosion, **10** sign-off). Claude
  merges the others once CI is green, and summarises them for you.
- **B**: you review every PR before it merges.
- **C**: Claude merges everything on green CI; you review only step 10.

Whatever the policy, Claude stops and asks if a step hits a decision
point (like step 3's speed check), needs to depart from the design, or
would add complexity beyond the spec.

## 4. What's not in this plan

Everything after the MVP (SPEC §16: v2 texture maps, deposition, optional
maps; v3/v4; GPU only if needed). Each gets its own plan once the MVP is
done.
