# globerode — Design (v0.2, decisions accepted)

How the MVP in [SPEC.md](SPEC.md) §2 will be built. The spec says *what*; this
document says *how*. Later versions (v2+) are covered only where an MVP
choice must leave room for them.

Items marked **[Decision Dn]** are design choices for review. Each has a
recommendation. The full list is in §14.

## 1. Overview

```
levels.png ──► resample ──► prep ──► erode ──► ocean ──► export ──► report
 (equirect)    (cube        (coast    (stream   (shelf   (equirect   (metrics,
               sphere)      dist,     power     ramp +   + coast     previews,
                            smooth,   cascade)  painted  re-lock,    pass/fail)
                            curve,              depth)   masks)
                            noise)                                   blender ──► globe render
```

- All stages after `resample` work on one grid: an **equiangular cube
  sphere** with N×N cells per face.
- Stages are plain functions: `fields_out = stage(grid, fields_in, config)`.
  No hidden state. Each stage can be tested alone.
- The pipeline driver runs the stages in order, times each one, and writes
  everything to a run directory (§10).

## 2. Repository layout

```
pyproject.toml            uv-managed; Python 3.13; entry point `globerode`
uv.lock
src/globerode/
  __init__.py
  cli.py                  argparse commands (§11)
  config.py               pydantic models, TOML loading, --set overrides
  pipeline.py             runs stages, timings, run directory
  grid.py                 cube-sphere geometry: index ↔ (face,i,j) ↔ xyz ↔ lon/lat,
                          neighbours, cell areas, distances
  fields.py               Fields container (named arrays on the grid)
  resample.py             equirect → cube (bilinear, categorical);
                          cube → equirect (bicubic with halos)
  coast.py                land mask, signed geodesic coast distance
  smooth.py               masked heat-diffusion smoothing
  curve.py                monotone spline height curve
  noise.py                3D gradient noise (own implementation)
  erode/
    __init__.py           cascade driver
    routing.py            Priority-Flood+ε, receivers, stack order, drainage area
    stream_power.py       Braun–Willett implicit solve
    diffusion.py          hillslope diffusion
  ocean.py                painted depth + shelf ramp
  export.py               EXR / PNG16 / TIFF writers, coast re-lock, masks
  report/
    metrics.py            coast match, drainage, seams, Hack, Horton, etc.
    previews.py           hillshade, globe views, close-ups
    report.py             report.json + report.md
  blender/
    globe_script.py       template for blender_globe.py (§12)
  testworlds.py           synthetic input generators (§13.2)
tests/
  unit/ …                 one file per module
  integration/ …          small-N end-to-end runs
  snapshots/ …            regression reference data (small)
docs/
  SPEC.md  DESIGN.md  PLAN.md  decisions.md
.github/workflows/ci.yml
```

The existing scripts at the repo root are left untouched.

## 3. Grid: equiangular cube sphere

### 3.1 Coordinates

- **Cell index**: one flat `int32` index across all faces:
  `idx = face·N² + i·N + j`, with `face ∈ 0..5`, `i` = row, `j` = column.
  `int32` holds 6N² up to N≈18900, which covers 16K (N=4096).
- **Face frames**: each face has a centre axis and two tangent axes:

  | face | centre | u axis (j) | v axis (i) |
  |---|---|---|---|
  | 0 | +X | +Y | +Z |
  | 1 | −X | −Y | +Z |
  | 2 | +Y | −X | +Z |
  | 3 | −Y | +X | +Z |
  | 4 | +Z | +Y | −X |
  | 5 | −Z | +Y | +X |

  (Final table fixed and diagrammed in `grid.py`. The ±X/±Y faces are
  oriented with +Z "up", like an equator band. Tests check that it is
  right-handed and that neighbours match.)
- **Equiangular mapping**: cell centre angles
  `α_j = −π/4 + (j+½)·(π/2)/N` (same for `β_i`). Point on the cube
  `p = centre + tan(α)·u + tan(β)·v`, normalised to the unit sphere.
  Multiplied by radius R for metres.
- **Lon/lat**: `lon = atan2(y, x)`, `lat = asin(z)`. Equirect pixel (x,y)
  ↔ lon/lat at pixel centres.

### 3.2 Geometry is computed, not stored **[Decision D1]**

At 16K, neighbour tables (8 × int32) and xyz (3 × float32) would cost
~44 bytes/cell, about 4.4 GB. So instead:

- Small Numba functions compute xyz, lon/lat, neighbours and distances from
  `(face, i, j)` on the fly. It's a few `tan` calls and a normalise, cheap
  next to the memory traffic it saves.
- **Cell area** is the one cached array (float32), since it's used
  constantly. Exact formula for the solid angle of a gnomonic cell
  between tangent-plane corners (x₀..x₁, y₀..y₁):
  `Ω = F(x₁,y₁) − F(x₀,y₁) − F(x₁,y₀) + F(x₀,y₀)`, with
  `F(x,y) = atan2(x·y, √(1+x²+y²))`. Checked numerically: the 6 faces sum
  to 4π exactly, and the largest/smallest cell area ratio is ≈1.40.

### 3.3 Neighbours

- 8 neighbours: 4 edge neighbours (N, S, E, W) plus 4 diagonals.
- **Across a face edge**, cells match one to one: the angle along the
  shared edge is the same coordinate on both faces. Crossing an edge is a
  lookup in a 6×4 table giving (neighbour face, how i/j map: identity,
  flipped, or swapped).
- **At cube corners**, where 3 faces meet, the diagonal neighbour that would
  sit "inside the corner" doesn't exist. Those 24 corner cells have 7
  neighbours. Functions return −1 for a missing neighbour.
- **Distance** between neighbours: great-circle distance between cell
  centres, `R·acos(p·q)`, computed with the numerically stable `atan2`
  form.
- **Edge length** between edge neighbours (for diffusion): great-circle
  length of the shared cell edge, also computed on the fly.

### 3.4 Halo-padded faces (for interpolation only)

Bicubic interpolation near face edges needs cells from the next face.
`resample.py` builds a temporary padded copy of each face with a 2-cell
halo. Halo values are interpolated from the neighbouring face at the halo
cell's exact sphere position, so they are smooth across edges. Halos are
used only for resampling, never for the simulation. Corner blocks (both
coordinates outside the face) are refined iteratively so cube corners stay
second-order accurate (D13).

## 4. Fields container

```python
@dataclass
class Fields:
    grid: Grid
    arrays: dict[str, np.ndarray]   # each shape (6*N*N,), contiguous
```

- Named arrays, all flat over the cell index. Main dtypes: `float32` for
  heights and maps (but heights are `float64` during routing and erosion, D9), `int32` for indices, `uint8` for masks.
- Stages read the fields they need and add new ones. The pipeline drops
  fields once no later stage needs them, to save memory.
- Units everywhere: **metres, m², years**.

**Memory estimate** (MVP erosion stage, at its peak): height, initial height,
uplift, drainage area, receivers, stack order, filled height, cell area,
cumulative erosion, land mask, coast distance, plus the Priority-Flood
heap (≈8 bytes/cell worst case): about **45 bytes/cell**.

| Output | N | Cells | Peak RAM (est.) |
|---|---|---|---|
| 2K preview | 512 | 1.6 M | ~0.1 GB |
| 8K | 2048 | 25 M | ~1.2 GB |
| 16K | 4096 | 101 M | ~4.6 GB |

8K fits easily in the 15 GB sandbox. The pipeline prints this estimate
before running, and refuses to start if it's over the available RAM (spec §3.4).

## 5. Resample (equirect → cube)

- **Levels map**: read as 8- or 16-bit grayscale, or palette indices. Every
  pixel value must be in `input.classes`. Any unknown value stops the run
  with a list of the bad values and their pixel counts, so a stray
  anti-aliased pixel is never guessed at.
- **Categorical resampling**: for each cell, sample the equirect at a
  small fixed pattern of points inside the cell (4×4 supersampling, using
  the cell's own lon/lat footprint) and take the **majority class**. This
  avoids blended classes and handles the strongly oversampled polar rows of
  the equirect.
- **Continuous resampling** (later optional maps): bilinear, wrapping at
  ±180°, clamped at the poles.

## 6. Prep

### 6.1 Land mask and coast distance

- Land = class kind `land`.
- **Signed geodesic distance** to the coast, `coast_dist` (positive inland,
  negative at sea), via a **nearest-coast-point Dijkstra**. Seeds are
  cells on the coast. Each cell stores the xyz of its nearest seed. When a
  cell is updated from a neighbour, it takes that neighbour's seed and its
  distance becomes the great-circle distance to that seed. This is close
  to exact, unlike plain 8-neighbour graph distance, which overestimates
  by up to ~8%.
- Only needed out to `max_needed_km` (the largest of: coast ramp, shelf
  ramp width, smoothing σ × 3). The search stops there, so most of the
  ocean is never visited.

### 6.2 Base value and smoothing

- Land cells get their class `base` (0–1).
- **Masked heat diffusion** on land cells only, finite-volume form:

  `h_i ← h_i + Δt/A_i · Σ_e (L_e / d_e)·(h_e − h_i)`

  summed over the 4 edge neighbours that are land. `A_i` is the cell area,
  `L_e` the shared edge length, `d_e` the centre distance. Sea neighbours
  are skipped (no flow across the coast).
- Total diffusion time `t = σ²/2` gives a Gaussian blur of width σ.
  Explicit stepping with `Δt = 0.2·d_min²` (stable, below the 0.25 limit).
  The number of steps follows from `t / Δt`.
- **Brute force at full resolution [Decision D2]**: about 2 000 steps at
  8K with σ=120 km. Measured ≈110 s in the sandbox (M2), using a
  temporary neighbour table (D12). A multi-resolution
  version would be faster but more complex and could lose narrow land.
  Brute force is simpler, and the time is small next to erosion.
- **Coast ramp**: after smoothing,
  `h = lerp(base_lowland, h, smoothstep(0, coast_ramp_km, coast_dist))`.
  Coastal land is therefore lowland, whatever was painted next to it.
- Optional per-class σ is a v2 item, and only if it's wanted.

### 6.3 Height curve

- **Monotone cubic spline** through the config points (PCHIP: it never
  overshoots, so the curve stays monotone), from SciPy.
- Applied per cell: `h_m = curve(h)`. The spline's knots and the result's
  range go into the report.

### 6.4 Noise

- **Own 3D gradient noise in Numba [Decision D3]** (Perlin-style, about 80
  lines), with fBm octaves and optional domain warp. Our own copy means a
  dependency update can never change your world for the same seed, and
  it can be tested directly (mean ≈0, variance, repeatability).
- Sampled at each cell's 3D position × (R / `base_km`), so there are no
  seams or polar pinching.
- Amplitude: `amp = amp_m + amp_per_m · h_m`.
- **Octave split for the cascade**: noise is one function over the
  sphere. At the coarse level only octaves with wavelength ≥ 4 cells are
  added. After upsampling to the fine level, the remaining octaves (down
  to 4 fine cells) are added. Fine levels thus get the small-scale seed
  detail they need for small tributaries.
- Output of prep: `h_init` (metres, land), plus `land`, `coast_dist`.

## 7. Erosion

### 7.1 One iteration

```
1. routing:  Priority-Flood+ε on h  →  h_filled, stack (order), receivers
2. h ← h_filled                       (fill pits: lakes fill instantly)
3. drainage area A: accumulate cell areas (× rainfall) down the tree (reverse stack order)
4. stream power: Braun–Willett implicit update along the stack (K varies, §7.10)
4b. landslide slope limit along the stack (§7.10)
5. diffusion: hillslope diffusion step
6. clamp: land cells ≥ min_land_height (coast lock)
7. accumulate erosion (h_before − h_after, if positive) into `eroded`
```

### 7.2 Routing: Priority-Flood+ε

Follows [Barnes et al. 2014][barnes], ε variant:

- Seed a min-heap with all **land cells next to the sea**, at their own
  height. Sea cells are the outlet (base level = sea level, 0 m).
- Pop the lowest cell `c`. For each land neighbour `n` not yet visited:
  `h_f[n] = max(h[n], h_f[c] + ε)`. Push `n`. The **receiver** of `n` is
  initially `c`.
- **Pop order is a valid processing order**: every cell is popped after
  its receiver. It becomes the `stack` array directly, so no separate
  topological sort is needed.
- After the flood, each cell's receiver is refined to its **steepest
  downhill neighbour on h_f**. (It can only be one already popped, because
  it's lower, so the order stays valid.) This gives natural dendritic
  directions instead of flood-order directions.
- ε = 1e-3 m per cell step: enough to give every flat a drainage
  direction, too small to be visible.
- Cost: O(n log n), serial. Roughly 5–15 s per iteration at 8K
  (estimate; to be measured). This is the main cost of the MVP.

**Lakes [Decision D4]**: filling pits each iteration (step 2) means lakes
fill with sediment instantly. It is the simplest rule that keeps every
river connected to the sea, which the spec requires. Filled depth is
summed into a `lake_fill` field, which becomes the v2 lake mask almost for
free.

### 7.3 Drainage area

`A[c] = cell_area[c]` initially, then for cells in **reverse** stack order,
`A[receiver[c]] += A[c]`. O(n), serial, exact and deterministic.

### 7.4 Stream power (Braun–Willett, n = 1)

For cells in stack order (receiver already updated):

```
F   = K · Δt · A^m / d(c, r)
h_c = (h_c + Δt·U_c + F · h_r) / (1 + F)
```

- If the receiver is a sea cell, `h_r = 0` (sea level), **not** the ocean
  floor. Otherwise rivers would cut down toward ocean depth at the coast.
- This is unconditionally stable for n = 1, so Δt can be large.
- MVP supports **n = 1 only [Decision D5]**. n ≠ 1 needs a Newton solve per
  cell. It's rarely needed for the look we want, so it waits for a request.
- **Uplift**: `U = u_scale · h_init / max(h_init)` (spec: ranges stay roughly
  as painted).

### 7.5 Hillslope diffusion

Same finite-volume operator as smoothing (§6.2), on land cells, with
diffusivity κ (m²/yr) and explicit sub-steps chosen for stability. It
rounds ridges and stops one-cell spikes.

### 7.6 Keeping heights as painted: local height restore **[D16, replaces D6]**

**Superseded.** The global hypsometry restore below matched the height
*distribution* over the whole globe, which at 8K moved mountain-ridge
heights onto plateau divides. It was replaced by a **local restore** (D16):
a smooth offset G((h_init − h_eroded)·w) / G(w) is added to eroded land, with
G a Gaussian of `restore_sigma_km` (default 200 km) and w the land fraction.
The offset is computed on a coarse grid (cell ≤ σ/4) from area-weighted
block averages and upsampled bicubically. Each region keeps its painted mean
height and the eroded relief is kept as is. (A multiplicative factor was
tried first; it stretched relief and overshot painted peaks ~2×.) Original text, kept for the record:

Uplift ∝ painted height keeps ranges roughly in place, but final heights
depend on K, U and time, which are fiddly to tune. So an optional final
step (default on) remaps land heights with a **monotone function** so that
their distribution (hypsometry) matches `h_init`'s.

- A monotone remap keeps every downhill direction, so drainage and valleys
  survive. Only the vertical scale changes.
- Result: the tallest ranges come out about as tall as painted, while
  keeping the eroded shape.
- Can be switched off for the "age" look later.

### 7.7 Cascade

- Level 0 (N=512): `iterations[512]` iterations.
- Upsample h to N=2048 (bicubic via halos, §3.4, with sea cells set to
  0 m first, D14), add the fine noise octaves (§6.4), recompute the fine
  land mask and coast clamp, run `iterations[2048]` iterations.
- Uplift and `h_init` are recomputed at the fine level from the
  fine-resolution prep (not upsampled), so the fine coastline stays sharp.
- Δt and K are the same at every level, so parameters mean the same thing
  at any resolution.

### 7.8 Default parameters (to be tuned)

| Param | Default | Note |
|---|---|---|
| m | 0.5 | |
| n | 1 | fixed in MVP |
| K | 2e-6 /yr·m^(1−2m) | |
| Δt | 1e5 yr | |
| u_scale | 1e-3 m/yr | |
| κ | 1e-2 m²/yr | |
| iterations | 200 @512, 100 @2048 | |

Tuning happens on test worlds at preview size, against the report metrics
(Hack exponent, Horton ratio, relief retained). The final defaults go into
`docs/decisions.md` with the reasons.

### 7.9 Determinism

- Priority-Flood ties are broken by cell index, so the order never
  depends on how the heap handles equal heights.
- Numba `prange` is used only for per-cell independent work (noise, curve,
  diffusion stencil, clamps), never for reductions whose float sum order
  could vary. Sums (drainage area, totals) are serial.
- Same config + seed ⇒ byte-identical output, checked by a test.

### 7.10 Varied terrain **[D26–D28]**

Uniform parameters make every slope erode the same way, which reads as
"too neat". Three additions:

- **Varied erodibility, rainfall and uplift (D26).** Each is a per-cell
  factor `contrast ** (v / 2)`, `v = tanh(sharpness · z)`, z = unit-variance
  warped fBm with its own seed (offset from `prep.noise.seed`). The factor
  lies in 1/√contrast … √contrast with geometric mean ≈ 1. K = k · erodibility;
  rainfall weights each cell's area in the drainage sum, so A becomes
  discharge; uplift multiplies the painted-height uplift. Octaves finer than
  4 cells are dropped per level, and the field is sampled at fixed positions,
  so every cascade level sees the same field.
- **Landslide slope limit (D27).** After stream power, in stack order, each
  land cell is lowered to at most `tan(max_slope) · d` above every neighbour
  already processed (sea counts as 0 m). Material removed is assumed carried
  away. The angle is at cell scale: ~8° at 8K (4.9 km cells), steeper at 16K.
- **Restore rounds (D28).** Local restore + peak cap run three times so
  hard-rock highs clipped by the cap don't lower the regional mean.

## 8. Ocean

- Each ocean class gives `depth_m`. Smoothed over sea cells with the same
  masked diffusion (§6.2) and `ocean.sigma_km`.
- Shelf ramp `r(d)` from `−coast_dist`: linear 0 → `shelf_depth` over
  `shelf_width_km`, then a cosine-eased drop over `slope_width_km` to the
  deepest painted depth.
- `h_sea = min(−1 m, max(painted, r(d)))`. The ramp only makes the sea
  *shallower* near the coast, never deeper. Then low-amplitude deep-floor
  noise.
- Ocean is computed once, at the final level, after erosion. It doesn't
  interact with erosion in the MVP.

## 9. Export

### 9.1 Cube → equirect

- For each output pixel: lon/lat → xyz → face and fractional (i, j) →
  **bicubic (Catmull-Rom)** sample from the halo-padded face. Numba,
  parallel over rows.
- No polar supersampling (D10): output width is always 4N, so output
  pixels are never larger than cells.

### 9.2 Coast re-lock (spec §10.3)

- The input levels map is resampled straight to the output grid, giving
  `land_out`. If the input and output sizes match this is the image
  itself. Otherwise it's resampled with the same majority rule.
- Then land pixels are clamped to ≥ `min_land_height` and sea pixels to
  ≤ −1 m. The exported coastline is therefore exactly the input's, at
  output resolution.

### 9.3 Files

- `height_m.exr`: float32, single channel, via the `OpenEXR` package (3.5.x
  on PyPI, with numpy support).
- `height16.png`: uint16, mapped by `png_range_m`, via Pillow. The range
  goes in `metadata.json`.
- `height_m.tif` / `height16.tif`: via `tifffile`.
- `mask_flow.png` (16-bit log-normalised A), `mask_rivers.png` (8-bit,
  A ≥ `river_threshold_km2`), `mask_erosion.png` (16-bit, cumulative
  eroded depth, normalised).
- `previews/`: §12.2. `metadata.json`: resolved config, ranges, timings,
  versions, seed, git commit.

## 10. Run directory

```
runs/2026-11-08T0930-myworld/
  config.resolved.toml   exact config used, after --set overrides
  log.jsonl              one JSON object per log line (stage, time, message)
  outputs/               §9.3 files
  previews/              §12.2 images
  report.json / report.md
  blender_globe.py
  summary.json           also printed to stdout at the end
```

`runs/` is git-ignored.

## 11. CLI

Built on argparse from the standard library [Decision D7]. No dependency,
and enough for a tool that Claude drives.

```
globerode run CONFIG [--preview] [--set k=v ...] [--name NAME]
globerode report RUN_DIR             # recompute report on an existing run
globerode compare RUN_A RUN_B        # side-by-side metrics + images
globerode render RUN_DIR [--size 1024] [--samples 32]   # headless Blender
globerode testworld NAME OUT_DIR     # write a synthetic input + config
globerode estimate CONFIG            # memory/disk/time estimate only
```

- Exit codes: 0 ok; 1 failed quality checks (outputs still written); 2
  bad config/input; 3 resource limits; 4 internal error.
- **stdout**: one final JSON summary. Human-readable progress goes to
  stderr and to `log.jsonl`.
- `--preview` = N=512 only, no cascade, output 2048×1024.

## 12. Previews, report, Blender

### 12.1 Report metrics (`report/metrics.py`)

| Metric | How | Pass rule (initial) |
|---|---|---|
| Coast match | exported land mask vs `land_out` | exactly equal |
| Drainage | every land cell's receiver chain reaches the sea | 100% |
| Seam continuity | gradient magnitude across face edges vs interior cells | ratio within 1 ± 0.15 |
| Hack exponent h | fit log(main-stream length) vs log(A) over channel cells | 0.45–0.65 |
| Horton bifurcation ratio | Strahler orders on channels; geometric mean of N_k/N_{k+1} | 3–5 |
| Drainage density | channel length / land area | reported only |
| Hypsometry | land height percentiles vs `h_init` | reported; ≤ 10% diff with restore on |
| Relief | max, p99, mean land height | reported |
| Timing / memory | per stage | reported |

Thresholds live in config (`[report]`), so they can be tightened as we
learn.

### 12.2 Previews (`report/previews.py`)

All 8-bit PNG, small enough to view (≤ 1024 px):

- `hillshade_equirect.png`: hillshade + hypsometric tint.
- `globe_{+x,−x,+y,−y,n,s}.png`: orthographic globe views, rendered in
  NumPy (ray-sphere, hillshaded), without Blender.
- `closeup_northpole.png`, `closeup_southpole.png`, `closeup_seam180.png`,
  `closeup_cubecorner.png`: the areas most likely to show grid problems.
- `masks_overlay.png`: rivers in blue, erosion in red over the hillshade.
- `curve.png`, `hist.png`.

### 12.3 Blender (`blender/globe_script.py`)

- The tool writes `blender_globe.py` into the run directory with that run's
  paths and settings filled in. It works three ways: with the full Blender
  app (`blender -b -P blender_globe.py`), with the `bpy` module
  (`python blender_globe.py`), or sent by Claude through the Blender MCP.
- It builds what the spec §12.1 describes: a UV sphere, adaptive
  subdivision, a displacement material from `height_m.exr`, exaggeration
  from config, a height colour ramp, a sea sphere, sun and camera.
- `globerode render` runs it headless through `bpy` (optional extra
  `globerode[blender]`). Cycles CPU, low samples. It writes `render.png`.
- Test: render a 256 px image of a test world. Check it isn't blank, the
  globe's outline matches the expected radius, and land and sea show
  different colours.

## 13. Testing

### 13.1 Layers

| Layer | What | Speed |
|---|---|---|
| Unit | one function, known answer (spec §3.1 list) | ms–s |
| Property | Hypothesis: random rotations/points for grid & resample | s |
| Integration | full `run` on test worlds at N=32–128 | < 1 min each |
| Snapshot | small-N outputs vs stored reference, with tolerance | s |
| Render | headless Blender at 256 px (marked `slow`, optional in CI) | ~1 min |

`pytest -m "not slow"` runs in CI on every push. The full suite, including
render and an N=512 run, is run locally by Claude before each PR.

### 13.2 Test worlds (`testworlds.py`)

Generated as equirect images + config:

- `cap_equator` / `cap_pole`: the same circular continent centred at 0° and
  at 90° (polar consistency test: compare metrics).
- `seam`: continent centred on 180° longitude.
- `corner`: continent centred on a cube corner.
- `islands`: chain of small islands (coast lock, small features).
- `ring`: ring continent with an inland sea (pit filling, drainage).
- `mini_earth`: 5 land levels + 3 ocean levels, several continents.

### 13.3 Known-answer tests (examples)

- Cell areas sum to 4πR² (rel. error < 1e-6). Neighbour relation is
  symmetric for all cells at N = 1, 2, 3, 16.
- Round trip xyz → (face,i,j) → xyz for random points.
- Coast distance on `cap_equator` matches the exact distance to a circle,
  within half a cell.
- Smoothing a delta spike gives a profile matching a Gaussian of σ km,
  at latitude 0°, 60° and 85°.
- Priority-Flood: no pits remain. Every cell's receiver is lower (or ε
  lower). `stack` is a valid order.
- Drainage area: sum over outlets = total land area.
- Stream power: on a 1D-like slope with uniform U, the steady state
  matches `S = (U/K)^(1/n) · A^(−m/n)` within 2%.
- Export: equirect of a smooth analytic field (e.g. spherical harmonic
  Y₂₁) matches the analytic values, with no seam jumps.
- Determinism: two runs → identical bytes.

## 14. Decisions for review

All eight accepted by the user on 2026-11-08 (recorded in
[decisions.md](decisions.md)).

| # | Decision | Recommendation | Alternative |
|---|---|---|---|
| D1 | Compute geometry on the fly; cache only cell area | **Yes** (saves ~4 GB at 16K) | Store neighbour/xyz tables: simpler kernels, much more memory |
| D2 | Smoothing by brute-force diffusion at full resolution | **Yes** (simple; ~1 min) | Multi-resolution: faster, more code, risk to narrow land |
| D3 | Own 3D noise implementation | **Yes** (repeatable forever, testable) | `opensimplex` package: less code, results may change on upgrade |
| D4 | Fill pits every iteration (lakes fill instantly) | **Yes** (simple, guarantees drainage) | Persistent lakes: more realistic, more code |
| D5 | Stream power with n = 1 only | **Yes** | General n with per-cell Newton solve |
| D6 | Hypsometry restore after erosion, default on | **Yes** (keeps heights as painted, simple) | Rely on tuning K/U only |
| D7 | argparse CLI | **Yes** (no dependency) | typer/click: nicer help, extra dependency |
| D8 | Previews rendered in NumPy; Blender only for `render` | **Yes** (fast, no Blender needed in CI) | Every preview via Blender: slower, heavier |

## 15. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Priority-Flood too slow at 8K × 100 iterations | Run longer than overnight | Measure at Plan step 1; fewer fine iterations; the cascade does most of the work at N=512 |
| Grid artefacts at cube edges/corners | Visible lines in the map | Seam/corner metrics and close-up previews from the first erosion run |
| Parameters hard to tune | Results look wrong | Report metrics + `compare` for quick A/B; hypsometry restore |
| Numba compile time slows tests | Slow CI | `cache=True`; keep kernels few and small |
| `bpy` wheel tied to one Python version | Render breaks on upgrade | Optional extra; Python pinned to 3.13 |

## References

Same as SPEC.md. Key ones for this design:

- Barnes, Lehman & Mulla (2014), Priority-Flood. https://arxiv.org/abs/1511.04463
- Braun & Willett (2013), implicit O(n) stream power. https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.stream_power.fastscape_stream_power.html
- Ronchi, Iacono & Paolucci (1996), the cubed sphere. https://doi.org/10.1006/jcph.1996.0047

[barnes]: https://arxiv.org/abs/1511.04463
