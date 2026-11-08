# Spherical Terrain Erosion Tool — Specification (draft v0.1)

Working name: **`globerode`**

## 1. Purpose

Turn a rough, hand-painted world height map (equirectangular, 3 or so height
levels) into a believable, eroded planetary height map: rivers, valleys,
ridges, alluvial plains and offshore deposits. The erosion runs **on the
sphere**, so the result has no polar stretching and no seam at ±180°.

The workflow copies the one the author uses in Gaea, moved onto the sphere:

1. Paint land/sea and rough levels (lowland / midland / highland).
2. Gaussian-smooth everything above the lowlands, keeping the coastline crisp.
3. Apply a height curve so elevations feel right.
4. Simulate erosion to cut rivers and valleys and lay down sediment.
5. Export height maps and masks for Blender and other renderers.

The goal is visual plausibility, not geomorphological accuracy.

### 1.1 Non-goals (v1)

- Plate tectonics, climate or ocean simulation (aspirational, see §18).
- Fine droplet-style erosion detail (a possible later add-on, §9.6).
- A GUI or our own globe viewer. **Blender is the viewer** (§12.1). The tool
  is a command-line program with a config file and preview images (§11).
- Real-world GIS accuracy (datums, ellipsoids). The planet is a perfect sphere.

## 2. MVP (version 1)

v1 is the smallest tool that gives a usable eroded globe. Everything else in
this spec comes after it (§16). The rule for v1: **if the result still looks
eroded without a feature, the feature waits.**

### 2.1 In the MVP

1. **Input**: one equirectangular `levels` image with **any number of land
   and ocean levels** (e.g. deep ocean / shelf / lowland / midland /
   highland), defined in config. Its land/sea boundary is the coastline.
2. **Grid**: cube sphere, equirect → cube resampling, seam handling (§6, §7.1).
3. **Prep**: coast distance field, spherical smoothing with σ in km, height
   curve, 3D seed noise (§8).
4. **Erosion**: single-flow-direction routing, Priority-Flood pit filling,
   implicit stream power (Braun–Willett), hillslope diffusion, uplift taken
   from the curved height map so ranges keep roughly their painted height (§9.1–9.2).
   Two-level cascade (preview N=512 → N=2048).
5. **Coast**: `locked` mode only (§10.3).
6. **Ocean**: painted ocean levels, smoothed, combined with a shelf ramp
   near the coast (§10.1). Not eroded.
7. **Output**: 32-bit EXR (metres) and 16-bit PNG equirect, plus `mask_flow`,
   `mask_rivers` and `mask_erosion` (all nearly free), hillshaded previews and
   `metadata.json`. A Blender script that builds a globe scene from the EXR
   (§12.1).
8. **Interface**: `globerode run world.toml [--preview]`. One TOML config.
9. **Compute**: **CPU only** (NumPy + Numba). Overnight runs are acceptable.
10. **Testing**: test suite, synthetic worlds, quality report, headless
    Blender render check and CI (§3.1).
11. **Resolution**: 8192×4096. Nothing is done that blocks 16K later.

### 2.2 Not in the MVP (planned, §16)

sediment deposition and deposition mask · marine fans ·
delta coast mode · separate high-res coast mask · hardness / rainfall /
uplift / protect / forced rivers / forced lakes maps · texture and colour maps ·
cube-face export · lake mask · 16K · checkpoint resume · age slider (§9.4).

### 2.3 MVP done when

- An 8K run of a test world finishes overnight on CPU.
- Acceptance tests 1–6 (§14) pass.
- The EXR displaces a sphere in Blender, and you judge it "reads as eroded".

**Status (2026-11-08, PLAN step 10):**

| Criterion | Evidence | Status |
|---|---|---|
| 8K run overnight on CPU | mini_earth, 8192×4096, sandbox (4 cores): **25.4 min**, peak **4.1 GB** (decisions M4) | done |
| §14 test 1, seams | 8K report: seam ratio 1.10 (pass); `test_seam_*`; close-up previews | done |
| §14 test 2, polar consistency | `test_polar_consistency`; equator vs pole caps within 1 % on max height ratio (1.20 / 1.21) | done |
| §14 test 3, coast lock | 8K report: 0 mismatched pixels; tested on every test world | done |
| §14 test 4, drainage | 8K report: 100 % of land drains to the sea | done |
| §14 test 5, determinism | `test_runs_are_deterministic` (byte-identical outputs) | done |
| §14 test 6, smoothing scale | `test_blur_width_in_km_is_the_same_at_any_latitude` (within 0.3 % at 0°, 60°, 85°) | done |
| Blender displaces a sphere; "reads as eroded" | `globerode render` of the 8K EXR (`docs/images/mvp_8k_render.png`) | **your call** |

Known limitations carried past the MVP: evenly spaced parallel valleys on the
flanks of narrow painted ridges (needs a routing change); Horton ratio ~5.8
(real rivers 3–5; warning only).

## 3. Engineering requirements

These apply to every version, including the MVP.

### 3.1 Claude must be able to run and test everything

The user is not the tester. Claude builds, runs and checks the tool,
in the cloud sandbox and on the user's PC.

- **Small-N runs**: every stage runs at tiny resolution (N = 32–128) in
  seconds, so the whole pipeline can be tested on each change.
- **Synthetic test worlds**, generated in code with no hand-painted
  fixtures: a round continent at the equator, the same at a pole, a
  continent across the ±180° seam, an island chain, a ring continent, and
  a multi-level "mini Earth".
- **Unit tests checked against known answers**:
  - grid: cell areas sum to 4πR²; neighbour links are symmetric; seam cells line up;
  - resampling: equirect → cube → equirect round trip within tolerance;
  - coast distance: matches the exact distance to a circular coast;
  - smoothing: matches a Gaussian of the same σ in km at any latitude;
  - pit filling: no pits remain; drainage area sums to total land area;
  - stream power: matches the known steady-state slope–area solution
    S = (U/K)^(1/n) · A^(−m/n) on a simple test profile;
  - determinism: same config + seed gives byte-identical output.
- **Property tests** (Hypothesis) for grid and resampling code: random
  rotations and positions must not change results beyond tolerance.
- **Regression snapshots**: small-N outputs stored with tolerances, so
  unintended changes show up.
- **Automatic quality report** for every run (`report.json` +
  `report.md` with images), with pass/fail thresholds:
  - coast match (§14 test 3), drainage completeness, seam discontinuity,
    smoothing scale by latitude, mass balance;
  - height histogram compared with the configured curve;
  - river-network statistics against real-world ranges: **Hack's law**
    exponent (stream length ∝ area^h, h ≈ 0.5–0.6; [Hack's law][hack]),
    **Horton bifurcation ratio** ≈ 3–5 ([Strahler number][strahler]), and
    drainage density;
  - equator vs pole statistics (relief, river density, slope
    distribution) for polar consistency.
- **Visual checks Claude can look at**: hillshaded previews, globe views
  from 6 directions, close-ups of both poles, the ±180° seam and cube
  corners, plus mask overlays, all at sizes small enough to inspect.
- **Headless Blender renders** via the `bpy` package from PyPI. It runs
  without the Blender app (bpy 5.2.3 wheel confirmed available for this
  sandbox's Python 3.13). Cycles on CPU at low resolution, used in tests
  and to check the Blender script.
- **CI**: GitHub Actions runs lint, type checks and the small-N test suite
  on every push.

### 3.2 Code quality

- **Python 3.13**, matching the sandbox and the current `bpy` wheel.
  Environment and lockfile managed with `uv`.
- **Package layout**: one module per stage (`grid`, `resample`, `prep`,
  `erode`, `ocean`, `export`, `report`, `blender`). Stages are mostly pure
  functions over a shared `Fields` container on the cube-sphere grid.
- **Typed** (mypy), **linted and formatted** (ruff). Docstrings on public
  functions. Numba kernels are kept small and separate from orchestration.
- **Few dependencies**: numpy, scipy, numba, pydantic (config validation
  with clear errors), tifffile, OpenEXR, Pillow. Dev: pytest, hypothesis,
  ruff, mypy. bpy is optional (Blender extras only).
- **Decision log**: `docs/decisions.md` records each design choice and why.
- **Complexity gate**: any feature beyond the MVP list, or any
  optimisation, starts as a short proposal: benefit, added complexity,
  how it is tested. The user can say no. Marginal gains that add
  significant complexity default to no.
- The repo's existing scripts stay untouched as reference.

### 3.3 Interface (built for Claude to drive)

- CLI is primary. Its main user is Claude, so it favours
  **machine-readable** behaviour: exit codes, a JSON summary on stdout,
  logs to file, no interactive prompts.
- Every parameter lives in the TOML config. `--set key=value` overrides it
  for quick experiments.
- **Run directories**: each run writes `runs/<timestamp>-<name>/` with
  the resolved config, outputs, report and timings, so runs can be compared
  side by side (`globerode compare runA runB`).
- A thin Python API mirrors the CLI, for tests and notebooks.

### 3.4 Environments

| | Cloud sandbox (measured 2026-11-08) | User PC |
|---|---|---|
| CPU / RAM | 4 cores / 15 GB | many cores / large RAM |
| GPU | none | RTX (NVIDIA) |
| Disk | ~30 GB free | large |
| Used for | development, all tests, previews, 8K runs | full 8K and 16K runs |

- The same code and the same commands run in both places.
- Before a run, the tool estimates memory and disk use and stops with a
  clear message if the machine is too small (e.g. 16K in the sandbox).
- Checkpoints are optional and cleaned up by default, to save disk.

### 3.5 Performance approach: keep it simple

- **One code path: CPU with Numba**, parallel across cores where the
  algorithm allows. It runs everywhere, including the sandbox, which has no GPU.
- Some core steps (Priority-Flood, Braun–Willett ordering) are largely
  serial. That is acceptable because overnight runs are fine.
- **A GPU backend is not planned by default.** It is considered only if
  measured runs on the user's PC are too slow, and then goes through the
  complexity gate (§3.2). If built, it sits behind the same interface and is
  tested against the CPU version.
- Every run records per-stage timings, so optimisation targets come from
  data, not guesses.

## 4. Decisions so far

| Topic | Decision |
|---|---|
| Input projection | Equirectangular (plate carrée) only. |
| Height levels | Any number of land and ocean levels, set in config. Same code as 3 levels. |
| Ocean depth | Painted as ocean levels in the `levels` map; generated shelf ramp near the coast. |
| Rivers | Carved valleys + `mask_rivers`; water drawn in Blender from the mask. No water surface in the height map. |
| Mountain height | Ranges stay about as high as painted (uplift from painted height). |
| Viewer / renderer | Blender (5.0+), usually driven by Claude through the Blender MCP. No own GUI. |
| Hardware | CPU (Numba) everywhere: cloud sandbox and user PC. GPU only if measured need (§3.5). |
| Resolution | 8192×4096 equirectangular output first; design must scale to 16384×8192. |
| Interface | CLI + config file + preview images. |
| Erosion model | Stream power law + hillslope diffusion + simple sediment deposition. |
| Pre-processing | Smoothing and height curve run on the sphere, inside the tool. |
| Ocean deposits | Offshore sediment fans (after MVP). |
| Inland pits | Filled: every river reaches the sea. Filled areas exported as a lake mask. |
| Coastline | Config option: `locked` (exact, default) or `deltas` (sediment may build land). |
| Optional painted inputs | Rock hardness, rainfall, uplift/protect, fixed rivers and lakes, high-res exact coastline. |
| Outputs | 16-bit PNG/TIFF equirect, 32-bit float EXR/TIFF (metres), cube-map faces, masks. |
| Planet | Radius and elevation range configurable; Earth defaults. |
| Run time | Overnight acceptable at full resolution; previews should take minutes. |

## 5. Glossary

- **Equirect**: equirectangular (plate carrée) map, longitude → x, latitude → y, 2:1.
- **Cube sphere**: the sphere split into 6 square faces by projecting a cube
  outward. The working grid for all simulation steps.
- **N**: cells along one cube-face edge. The equator crosses 4 faces, so it
  has 4N cells.
- **Drainage area A**: total surface area upstream of a cell, i.e. the
  catchment that drains through it. With rainfall it becomes discharge Q.
- **Stream power law**: erosion rate = K · A^m · S^n, where S is the downhill
  slope and K is erodibility.

## 6. Coordinate system and grid

### 6.1 Working grid: equiangular cube sphere

All processing after input resampling runs on an **equiangular gnomonic cube
sphere** ([Ronchi, Iacono & Paolucci 1996][ronchi]). Reasons:

- Near-uniform cell size everywhere, with no poles to special-case. The
  equiangular grid is more uniform than the plain (equidistant) gnomonic one.
- Each face is a regular 2D array, so kernels stay simple.
- Neighbours are 8-connected within a face. Across face edges they are found
  through a precomputed lookup table, or by padding each face with a halo of
  cells from its neighbours.

Every cell stores (precomputed once):

- its unit vector on the sphere (float32 ×3, or recomputed on the fly);
- its true **area** on the sphere (used in drainage area and smoothing);
- the true **geodesic distance** to each of its 8 neighbours (used in slopes).

The exact area and distance values matter because cube-sphere cells are not
all the same size. Using them keeps erosion consistent everywhere on the globe.

### 6.2 Resolution

The simulation N is set so the equator resolution matches the output map:
`N = output_width / 4`.

| Output equirect | N | Cells (6N²) | Cell size at Earth radius |
|---|---|---|---|
| 2048×1024 (preview) | 512 | 1.6 M | ~19.5 km |
| 8192×4096 | 2048 | 25.2 M | ~4.9 km |
| 16384×8192 | 4096 | 100.7 M | ~2.4 km |

Memory budget at N=4096: one float32 field is ~400 MB. The roughly 10–14
working fields (height, uplift, K, rain, discharge, receiver index, sediment,
masks, …) total about 4–6 GB, plus the pit-filling queue and 64-bit index
arrays, so roughly 6–12 GB. At N=2048 (8K) it is about a quarter of that,
which fits the 15 GB cloud sandbox. 16K therefore runs on the user's PC
(§3.4).

Optional `sim_supersample` (e.g. 1.25×) lets the simulation run finer than
the output resolution.

### 6.3 Conventions

- Longitude −180°…+180°, left→right. Latitude +90° (top) … −90° (bottom).
- Cube face order and orientation: fixed and documented in the code, with a
  diagram, matching what Blender expects for cube-map displacement (to be
  checked during Phase 1).
- Elevation in metres relative to sea level (0 m).

## 7. Inputs

All raster inputs are equirectangular images. Size can be anything with a
2:1 ratio; they need not match the output size.

| Input | Required | Format | Meaning |
|---|---|---|---|
| `levels` | yes | 8/16-bit PNG/TIFF, palette or grayscale | Painted classes, any number, land and ocean (e.g. deep ocean, shelf, lowland, midland, highland). Config maps each pixel value to a class. |
| `coast` | no | Grayscale, ideally higher-res and anti-aliased | Exact land mask. Overrides the land/sea boundary taken from `levels`. The 0.5 iso-line is the coast, so it is sub-pixel accurate. |
| `hardness` | no | Grayscale 0–1 | Rock resistance. Scales K down (harder = less erosion). |
| `rainfall` | no | Grayscale 0–1 | Precipitation. Weights drainage area into discharge. |
| `uplift` | no | Grayscale 0–1 | Tectonic uplift rate. Keeps ranges high while erosion cuts them. |
| `protect` | no | Grayscale 0–1 | 1 = no erosion or deposition here. |
| `rivers` | no | Black/white lines | Forced river courses. Flow is routed along the line and the bed carved (§9.5). |
| `lakes` | no | Black/white areas | Forced lakes with a flat surface (§9.5). |

Inputs are always equirectangular (decided). A later option is vector
coastlines (SVG/GeoJSON).

### 7.1 Resampling input onto the cube sphere

- Continuous maps (hardness, rainfall, …): bilinear sampling of the equirect
  at each cell's longitude/latitude, wrapping at ±180°.
- Categorical `levels`: each class becomes a 0–1 indicator, which is
  resampled and then turned back into a class by majority. Painted edges must
  not produce in-between classes by accident.
- Coast: the anti-aliased mask is resampled bilinearly. A **signed geodesic
  distance field** to the coast is built on the cube sphere (positive inland,
  negative at sea), e.g. with jump flooding or a fast-marching method. Coast
  locking, shelf generation and smoothing falloff all use this field.

## 8. Pre-processing pipeline (on the sphere)

### 8.1 Base height from levels

Each class is a config entry with its pixel value, whether it is land or
ocean, and a base value:

- **Land** classes: 0–1 value, smoothed and then mapped through the height
  curve (§8.3). Example: lowland 0.15, midland 0.45, highland 1.0.
- **Ocean** classes: depth in metres (e.g. shelf −150, ocean −4000,
  trench −8000), handled in §10.1.

Any number of classes is supported. The code is the same for 3 or 10, so
this costs nothing extra. Start with 3–4 land levels and 2–3 ocean levels.

### 8.2 Mask-aware spherical Gaussian smoothing

Goal: smooth land above the lowlands without moving the coast.

- Implemented as **heat diffusion on the cube-sphere grid**. Diffusing for
  time t equals a Gaussian blur with σ = √(2t). σ is given in **km**, so the
  blur is the same physical size at the equator and the poles. This is the
  main reason the step must run on the sphere.
- Sea cells are excluded from the diffusion (no-flux boundary), so ocean
  values never leak into land.
- A **coast ramp** keeps coastal lowlands low: within `coast_ramp_km` of the
  coast, height is blended toward the lowland value using the distance field.
- Optional separate σ per class (e.g. heavier smoothing for highlands).
- Large σ uses multiple resolution levels (blur at lower N, then upsample) to
  keep the cost manageable.

### 8.3 Height curve

- Maps the 0–1 smoothed value to metres with a monotone spline through
  control points given in config, like Gaea's Curve node:
  `curve = [[0,0],[0.2,150],[0.5,900],[0.8,3500],[1.0,7500]]`.
- The CLI saves a preview image of the curve.

### 8.4 Seed detail (noise)

Erosion on a smooth surface gives straight, parallel channels. A little noise
lets drainage patterns branch.

- **3D** fractal noise (simplex/fBm, optionally ridged or domain-warped),
  sampled at each cell's 3D position. Sampling in 3D means no seams or polar
  pinching.
- Amplitude scales with height (e.g. `amp = a0 + a1·h`), so lowlands stay
  calm and mountains rough.
- Reproducible from a `seed` value.

## 9. Erosion model

### 9.1 Governing equation (land cells)

```
∂h/∂t = U(x)                         uplift (optional map)
      − K(x) · Q^m · S^n              fluvial incision (stream power)
      + κ ∇²h                          hillslope diffusion (soil creep)
      + D(x)                           deposition (§9.3)
```

- `Q = Σ_upstream (rain · cell_area)`, which is plain drainage area A when no
  rainfall map is given.
- `K(x) = K0 · (1 − hardness·hardness_strength) · (1 − protect)`.
- Defaults `m = 0.5`, `n = 1`, the usual values in landscape-evolution work.
- Sea level is the base level: coast cells are where rivers leave the land.

This is the model used in graphics by [Cordonnier et al. 2016][cordonnier]
and [Schott et al. 2023][schott]. Schott et al. also model terrain through
the uplift field, which suits the "paint the mountains" workflow. An uplift
map can be derived from the curved height map (h_target / response time).
Erosion then cuts valleys into ranges that stay roughly as high as painted.

### 9.2 Numerical method

- **Flow routing**: each cell drains to its steepest-descent neighbour among
  the 8, using true geodesic distances, including across face edges (single
  flow direction). This builds a drainage tree.
- **Pit filling**: every iteration, inland depressions are routed to the sea
  so all land drains to the ocean ([Barnes et al. 2014 Priority-Flood][barnes]).
  Filled volume is kept as the lake mask.
- **Drainage area**: summed down the tree.
- **Incision**: solved implicitly with the O(n) [Braun & Willett 2013][bw]
  scheme. It is stable for large time steps, so few iterations are needed.
- **Diffusion**: explicit or implicit step on the cube-sphere Laplacian with
  true cell areas.
- **If a GPU backend is ever built** (§3.5): flow routing, drainage
  accumulation and depression routing would follow the GPU-parallel approach
  of [FastFlow, Jain et al. 2024][fastflow] (O(log n) iterations for flow
  routing).
- **Multigrid / cascade**: run first at low N (e.g. 512), upsample, continue at
  2048, then (optionally) 4096. Big drainage basins form cheaply at coarse
  resolution, and fine levels add detail. The coarse level is also the preview.
- **Alternative to evaluate**: the analytical stream-power solution of
  [Tzathas et al. 2024][tzathas], where time works as an "erosion amount"
  slider instead of an iteration count. Evaluate in Phase 3 if iteration
  counts at 16K get painful.

### 9.3 Deposition (simple)

- Each cell carries a sediment flux `Qs` downstream (eroded material
  accumulated along the drainage tree).
- Deposition rate `G · Qs / Q` (G = deposition coefficient). Material drops
  where slope or flow falls: valley floors, plains, river mouths.
- Sediment reaching the coast goes to the marine step (§10.2).
- Total eroded and deposited volume is logged as a sanity check.

### 9.4 Stopping

- Set by `iterations` per level, or `erosion_strength` (0–1) mapped to total
  simulated time.
- **Age slider (after MVP)**: `age` (0–1) scales uplift down so ranges
  wear lower and broader over time. MVP default: ranges stay about as
  painted.
- Checkpoints (all fields) are saved per level, so a run can resume or be
  re-run from any stage with new parameters.

### 9.5 Forced rivers and lakes

- **Rivers**: painted lines are converted to cell paths on the cube sphere,
  ordered by distance to the coast. Along a path, the receiver is forced to be
  the next cell downstream, and height is clamped to be monotone decreasing.
  A small extra K multiplier carves a clear valley. Lines must reach the sea
  or another river; the tool checks this and reports paths that don't.
- **Lakes**: painted areas are held at a flat water level, the lowest rim
  height or an explicit value. Flow is routed through to their lowest outlet.
  The lake mask is exported.

### 9.6 Later (not v1)

- Thermal/talus erosion pass for scree slopes.
- Droplet erosion detail pass at final resolution (the repo's
  `simulation.py` is a flat-grid reference).
- Multiple flow direction routing for softer, braided rivers in flat areas.
- Glacial erosion (U-shaped valleys) at high latitude or altitude.

## 10. Ocean and coastline

### 10.1 Generated bathymetry

1. Painted ocean classes give a depth per cell. It is smoothed on the sphere
   over sea cells only (`ocean.sigma_km`), the same way as land (§8.2).
2. A **shelf ramp** r(d) from the signed coast distance d (km offshore):
   0 m at the coast, down to `shelf_depth` (default −130 m) over
   `shelf_width_km` (default 80 km), then steeply down over `slope_width_km`.
3. Final depth = `max(painted, r(d))`. Near the coast the ramp keeps water
   shallow. Further out the painted depth takes over. If you paint a shelf
   class, it shows wherever it is shallower than the ramp.
4. Low-amplitude 3D noise on the deep floor.

### 10.2 Marine deposition

- Sediment flux arriving at each river mouth is spread over sea cells by
  diffusion within `marine_diffusion_km`, building submarine fans and
  shelf deposits.

### 10.3 Coastline modes (`coast.mode`)

- `locked` (default): after every step, land cells are clamped to ≥
  `min_land_height` (e.g. +1 m) and sea cells to ≤ `max_sea_height`
  (e.g. −1 m) on the simulation grid.
  - Clamping simulation cells is not enough on its own: islands or channels
    smaller than a cell can be lost when resampling, and interpolating
    height to the output map can shift where it crosses 0 m. So the
    **original coast mask is kept** and enforced again **at output
    resolution, after** cube → equirect resampling. Each output pixel's
    land/sea state comes from the input mask resampled to the output grid
    (threshold 0.5), and its height is clamped to the land or sea range to
    match. Islands smaller than a simulation cell survive as small
    low-lying land.
  - Cube-face exports get the same treatment, using the mask sampled at
    each face pixel.
- `deltas`: sea cells near river mouths may rise above 0 m through
  deposition, within `delta_max_km` of the original coast. Elsewhere the
  coast stays locked.

## 11. CLI and configuration

### 11.1 Commands

```
globerode run   world.toml              # full pipeline
globerode run   world.toml --preview    # N=512, fast; writes previews only
globerode run   world.toml --from erode # resume from a saved stage checkpoint
globerode stage world.toml prep         # run one stage: resample | prep | erode | export
globerode export world.toml --format exr,png16,cube,masks
globerode inspect checkpoint.npz        # stats, histograms, quick images
```

### 11.2 Example config

```toml
[planet]
radius_km      = 6371
sea_level_m    = 0

[resolution]
output_width   = 8192          # 16384 later
levels         = [512, 2048]   # cascade N values; append 4096 for 16K
sim_supersample = 1.0

[input]
levels   = "paint/levels.png"
coast    = "paint/coast_16k.png"   # optional
rainfall = "paint/rain.png"        # optional, likewise hardness/uplift/protect/rivers/lakes

[[input.classes]]                  # any number; pixel value -> class
value = 0
name = "deep"
kind = "ocean"
depth_m = -4500

[[input.classes]]
value = 40
name = "shelf"
kind = "ocean"
depth_m = -150

[[input.classes]]
value = 85
name = "lowland"
kind = "land"

[[input.classes]]
value = 170
name = "midland"
kind = "land"

[[input.classes]]
value = 255
name = "highland"
kind = "land"

[prep]
# classes listed above; base values per class:
base       = { lowland = 0.15, midland = 0.45, highland = 1.0 }
sigma_km   = 120
coast_ramp_km = 40
curve      = [[0,0],[0.2,150],[0.5,900],[0.8,3500],[1.0,7500]]
noise      = { seed = 42, octaves = 8, base_km = 800, amp_m = 50, amp_per_m = 0.15, warp = 0.3 }

[erode]
K0 = 2e-5
m = 0.5
n = 1.0
kappa = 0.01
deposition_G = 1.0
uplift_from_height = true
iterations = { 512 = 200, 2048 = 100 }
hardness_strength = 0.8

[ocean]
sigma_km = 150
shelf_depth_m = -130
shelf_width_km = 80
slope_width_km = 150
marine_diffusion_km = 100

[coast]
mode = "locked"        # or "deltas"
min_land_height_m = 1
delta_max_km = 30

[output]
dir = "out/"
formats = ["png16", "tiff16", "exr32", "cube", "masks", "previews"]
png_range_m = [-6000, 9000]   # mapping of 16-bit values; also written to metadata.json
```

Parameter values above are starting guesses, to be tuned in Phase 2.

## 12. Outputs

All rasters are written as equirect at `output_width`, plus cube faces when
requested.

| File | Format | Content |
|---|---|---|
| `height_m.exr` / `.tif` | 32-bit float | Elevation in metres. |
| `height16.png` / `.tif` | 16-bit unsigned | Elevation mapped by `png_range_m`. |
| `height_land16.png` | 16-bit | Land-only (0 at sea level), for land displacement. |
| `cube/height_{px,nx,py,ny,pz,nz}.exr` | 32-bit float | Cube-map faces, no polar distortion. |
| `mask_flow.png` | 16-bit | log(discharge), normalised. Gaea "Flow" style. |
| `mask_rivers.png` | 8-bit | Discharge above a threshold, optional width ∝ Q^0.5. |
| `mask_erosion.png` | 16-bit | Total material removed ("wear"). |
| `mask_deposition.png` | 16-bit | Total sediment deposited. |
| `mask_lakes.png` | 8-bit | Filled pits and painted lakes. |
| `mask_land.png` | 8-bit | Final land/sea mask. |
| `slope.png`, `coast_distance.exr` | | Helper maps for texturing. |
| `previews/*.png` | 8-bit | Hillshaded equirect, globe views (both poles + 4 sides), curve plot, histogram. |
| `metadata.json` | | Config used, value ranges, run times, software version, seed. |

Cube → equirect resampling uses bicubic interpolation that handles face edges.
Rows near the poles are supersampled so they aren't blocky.

### 12.1 Blender setup (recommended)

Blender is the globe viewer and renderer. Claude will usually drive it
through the Blender MCP, but the tool also writes a plain script
(`blender_globe.py`), so the scene can be rebuilt without Claude.
Colour and material maps follow in v2 (§12.2).

- **Blender 5.0+, Cycles.** Adaptive subdivision is no longer experimental
  in 5.0, and it has an object-space edge-length option
  ([5.0 Cycles notes][b50]).
- **Mesh**: UV sphere, smooth-shaded, radius 1 (scaled units).
- **Render displacement**: Subdivision Surface modifier with adaptive
  subdivision, plus a material Displacement node fed by `height_m.exr`
  (equirect, Non-Color, Linear interpolation), displacement mode
  "Displacement only".
- **Viewport preview**: a lower-res `height_m.exr` (2K) on a Displace
  modifier, so the scene stays responsive.
- **Vertical exaggeration**: real relief is tiny next to the radius
  (~9 km / 6371 km ≈ 0.14%). The script sets the displacement scale from
  `planet.radius_km` and an `exaggeration` value (default 20×).
- **Shading starter**: colour ramp on height plus the flow and erosion masks
  (rivers, valley floors), and a sea-level sphere for water.
- **Later (v4)**: quad-sphere mesh using the cube-face EXRs, which avoids
  the UV sphere's pinching at the poles.

### 12.2 Texture maps (v2)

Beyond height, the tool exports **colour and material maps derived from
the terrain**, so valleys, eroded slopes, deposits and peaks can look
different. All are equirect at output resolution, with cube faces when
cube export exists.

**Terrain attribute maps** (inputs to colouring, also exported on their own):

| Map | Meaning | Typical use |
|---|---|---|
| height | elevation (already exported) | colour bands, snow line |
| slope | steepness | bare rock on cliffs, soil on flats |
| curvature | concave (valleys) vs convex (ridges), at a few scales in km | dark valley floors, light ridges |
| flow / rivers | discharge (already exported) | river beds, wet valleys, vegetation |
| erosion | material removed (already exported) | exposed rock |
| deposition | sediment laid down | alluvial plains, fans, beaches |
| aspect | direction a slope faces | sunny vs shaded slopes |
| latitude | from grid position | polar ice, simple climate banding |
| coast distance | already computed | beaches, coastal vegetation |

**Colour map ("albedo")**: rules in config, applied in order, each with
a colour or gradient and a blend weight from one or more attribute maps.
Example: base colour by height → blend rock where slope is high → darken
where curvature is concave → sand where deposition is high near the coast
→ snow above a line that falls with latitude. Output: 8-bit sRGB PNG
(16-bit option).

**Material masks**: the same rule weights packed as RGBA "splat" maps,
plus an optional roughness map, so Blender can mix real materials (rock,
grass, sand, snow) instead of one baked colour.

**Two ways to use them** (both supported):
1. Baked colour map from the tool: simple, works in any renderer, and
   testable in the quality report.
2. Attribute maps into Blender shader nodes: more flexible, tweak colours
   without rerunning. The Blender script (§12.1) wires up a starter
   material.

**Testing**: attribute maps have known-answer tests (e.g. curvature of a
sphere-cap bump, slope of a cone). Colour rules are checked on synthetic
worlds by sampling known locations (a valley floor gets the valley colour).
Rendered previews let Claude check them visually.

## 13. Performance targets

| Run | Cloud sandbox (4 cores) | User PC (many cores) |
|---|---|---|
| Test suite (small N) | < 5 min | < 2 min |
| Preview (N=512) end-to-end | < 15 min | < 5 min |
| 8K (N=2048) full run | overnight (< 12 h) | a few hours |
| 16K (N=4096) full run | not supported (RAM) | overnight (< 12 h) |

These are estimates, to be replaced by measured timings from the MVP.

## 14. Acceptance tests

1. **Seams**: no visible step or crease across cube-face edges or at ±180°
   (automatic check on height gradients along seams).
2. **Polar consistency**: the same test continent placed at the equator and
   at a pole gives statistically matching results (river density, relief).
3. **Coast lock**: in `locked` mode, the exported land mask (height > 0)
   equals the input coast mask resampled to the output grid (threshold 0.5),
   pixel for pixel. If the input is higher-res than the output, the
   comparison is made at output resolution; if lower-res, the input is
   upsampled bilinearly first.
4. **Drainage**: every land cell drains to the sea or a painted lake.
5. **Determinism**: same config + seed gives an identical result.
6. **Smoothing scale**: a σ = 100 km blur measured on the output has the same
   km width at 0°, 60° and 85° latitude.
7. **Mass balance**: deposited + exported sediment ≈ eroded volume (within
   tolerance).
8. **Forced rivers**: painted river paths appear in `mask_rivers` with no gaps.

## 15. Technology

See §3.2 for language, tools and dependencies, and §3.5 for compute.
New code goes in a `globerode/` package. The repo's existing scripts
(`simulation.py`, `river_network.py`, `util.py`) stay as reference.

## 16. Milestones

| Version | Deliverable |
|---|---|
| **v1 (MVP)** | Everything in §2.1. Built in steps: (a) project skeleton, CI, test worlds, cube-sphere grid, resampling, seam tests; (b) prep: coast distance, smoothing, curve, noise, EXR/PNG export, which already gives a usable smoothed globe; (c) CPU erosion + cascade to 8K, flow/erosion masks, previews. |
| v2 | **Texture maps** (§12.2): attribute maps, rule-based colour map, material masks. Sediment deposition + deposition mask. Lake mask. Checkpoint resume. Evaluate the Tzathas-style analytical option. |
| v3 | Optional painted maps (hardness, rainfall, uplift, protect, forced rivers/lakes), separate high-res coast mask, marine fans, delta mode. |
| v4 | 16K support on the user PC, cube-face export with a quad-sphere Blender scene. |
| If needed | GPU backend, only if measured PC runs are too slow (§3.5). |
| Later | Aspirational directions in §18. |

## 17. Open questions (all answered)

1. ~~Input projection?~~ **Decided: equirectangular.**
2. ~~Ocean depths?~~ **Decided: painted, as ocean levels in the `levels` map.**
3. ~~Number of levels?~~ **Decided: any number.**
4. ~~Rivers as water?~~ **Decided: valleys + river mask.** Blender draws the
   water from the mask; no flat water surface in the height map.
5. ~~Mountain height?~~ **Decided: stay about as painted.** An age slider
   can come later.
6. ~~Blender setup?~~ **Decided: recommended setup in §12.1.**
7. ~~GUI?~~ **Decided: Blender is the viewer.**

## 18. Future directions (aspirational, not scheduled)

Noted so the design doesn't rule them out. Neither has been researched yet.

### 18.1 Climate and weather zones

Given the finished globe, estimate an Earth-like climate:

- Overlay standard latitude-based zones: circulation cells (Hadley,
  Ferrel, polar), prevailing winds, ITCZ, subtropical deserts.
- Model how continents, oceans and mountains change them: rain shadows,
  continental versus maritime climates, monsoons, how ocean currents move heat.
- Outputs could include temperature and rainfall maps, a Köppen-style climate
  classification map, and biome masks for texturing.
- Feeds back into erosion: the generated rainfall map can drive the
  `rainfall` input (§7), so wet coasts erode more than dry interiors.

### 18.2 Tectonics and geology

Work out the terrain from plate history instead of painting it:

- Mark continents/plates and their movements; derive where collisions,
  rifts and subduction would raise mountains, trenches and volcanic arcs.
- Use that to produce the `uplift` and `hardness` maps that erosion already
  takes, so tectonics plugs into the existing pipeline.
- Produce geological maps underneath the surface: rock types, ages, fault
  lines, for cartography and texturing.

### 18.3 Design notes that keep these possible

- Keep every field on the shared cube-sphere grid with a common data
  container, so new stages (climate, tectonics) can read and write fields
  without new plumbing.
- Keep the pipeline as separate stages with clear inputs and outputs
  (`resample → prep → erode → export`), so stages can be inserted before
  `erode` (tectonics → uplift) or after it (climate → biomes).

## References

- Ronchi, Iacono & Paolucci (1996), *The "Cubed Sphere": a new method for the solution of partial differential equations in spherical geometry*, J. Comput. Phys. 124(1). https://doi.org/10.1006/jcph.1996.0047
- Braun & Willett (2013), *A very efficient O(n), implicit and parallel method to solve the stream power equation*, Geomorphology 180–181. https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.stream_power.fastscape_stream_power.html
- Barnes, Lehman & Mulla (2014), *Priority-Flood: an optimal depression-filling and watershed-labeling algorithm for DEMs*, Computers & Geosciences 62. https://arxiv.org/abs/1511.04463
- Cordonnier et al. (2016), *Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion*, CGF 35(2). https://diglib.eg.org/handle/10.1111/cgf12820
- Schott, Paris, Fournier, Guérin & Galin (2023), *Large-scale Terrain Authoring through Interactive Erosion Simulation*, ACM TOG 42(5). https://h-schott.github.io/p/uplift — code: https://github.com/H-Schott/StreamPowerErosion
- Tzathas, Gailleton, Steer & Cordonnier (2024), *Physically-based Analytical Erosion for fast Terrain Generation*, CGF 43(2). https://diglib.eg.org/handle/10.1111/cgf15033
- Hack's law: https://en.wikipedia.org/wiki/Hack%27s_law
- Strahler number / bifurcation ratio: https://en.wikipedia.org/wiki/Strahler_number
- Blender 5.0 Cycles release notes (adaptive subdivision). https://developer.blender.org/docs/release_notes/5.0/cycles/
- Jain, Kerbl, Gain, Finley & Cordonnier (2024), *FastFlow: GPU Acceleration of Flow and Depression Routing for Landscape Simulation*, CGF 43(7). https://diglib.eg.org/handle/10.1111/cgf15243

[ronchi]: https://doi.org/10.1006/jcph.1996.0047
[bw]: https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.stream_power.fastscape_stream_power.html
[barnes]: https://arxiv.org/abs/1511.04463
[cordonnier]: https://diglib.eg.org/handle/10.1111/cgf12820
[schott]: https://h-schott.github.io/p/uplift
[tzathas]: https://diglib.eg.org/handle/10.1111/cgf15033
[fastflow]: https://diglib.eg.org/handle/10.1111/cgf15243
[b50]: https://developer.blender.org/docs/release_notes/5.0/cycles/
[hack]: https://en.wikipedia.org/wiki/Hack%27s_law
[strahler]: https://en.wikipedia.org/wiki/Strahler_number
