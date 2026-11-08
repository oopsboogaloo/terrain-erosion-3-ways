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

- Plate tectonics, climate or ocean simulation.
- Fine droplet-style erosion detail (a possible later add-on, §7.6).
- A GUI. v1 is a command-line tool with a config file and preview images
  (§9). The core is a library so a GUI can be added later.
- Real-world GIS accuracy (datums, ellipsoids). The planet is a perfect sphere.

## 2. Decisions so far

| Topic | Decision |
|---|---|
| Hardware | NVIDIA GPU (CUDA). CPU fallback for small runs and testing. |
| Resolution | 8192×4096 equirectangular output first; design must scale to 16384×8192. |
| Interface | CLI + config file + preview images. |
| Erosion model | Stream power law + hillslope diffusion + simple sediment deposition. |
| Pre-processing | Smoothing and height curve run on the sphere, inside the tool. |
| Ocean | Generated continental shelf and slope, plus offshore sediment deposits. |
| Inland pits | Filled: every river reaches the sea. Filled areas exported as a lake mask. |
| Coastline | Config option: `locked` (exact, default) or `deltas` (sediment may build land). |
| Optional painted inputs | Rock hardness, rainfall, uplift/protect, fixed rivers and lakes, high-res exact coastline. |
| Outputs | 16-bit PNG/TIFF equirect, 32-bit float EXR/TIFF (metres), cube-map faces, masks. |
| Planet | Radius and elevation range configurable; Earth defaults. |
| Run time | Overnight acceptable at full resolution; previews should take minutes. |

## 3. Glossary

- **Equirect**: equirectangular (plate carrée) map, longitude → x, latitude → y, 2:1.
- **Cube sphere**: the sphere split into 6 square faces by projecting a cube
  outward. The working grid for all simulation steps.
- **N**: cells along one cube-face edge. The equator crosses 4 faces, so it
  has 4N cells.
- **Drainage area A**: total surface area upstream of a cell, i.e. the
  catchment that drains through it. With rainfall it becomes discharge Q.
- **Stream power law**: erosion rate = K · A^m · S^n, where S is the downhill
  slope and K is erodibility.

## 4. Coordinate system and grid

### 4.1 Working grid: equiangular cube sphere

All processing after input resampling runs on an **equiangular gnomonic cube
sphere** ([Ronchi, Iacono & Paolucci 1996][ronchi]). Reasons:

- Near-uniform cell size everywhere, with no poles to special-case. The
  equiangular grid is more uniform than the plain (equidistant) gnomonic one.
- Each face is a regular 2D array, so GPU kernels stay simple.
- Neighbours are 8-connected within a face. Across face edges they are found
  through a precomputed lookup table, or by padding each face with a halo of
  cells from its neighbours.

Every cell stores (precomputed once):

- its unit vector on the sphere (float32 ×3, or recomputed on the fly);
- its true **area** on the sphere (used in drainage area and smoothing);
- the true **geodesic distance** to each of its 8 neighbours (used in slopes).

The exact area and distance values matter because cube-sphere cells are not
all the same size. Using them keeps erosion consistent everywhere on the globe.

### 4.2 Resolution

The simulation N is set so the equator resolution matches the output map:
`N = output_width / 4`.

| Output equirect | N | Cells (6N²) | Cell size at Earth radius |
|---|---|---|---|
| 2048×1024 (preview) | 512 | 1.6 M | ~19.5 km |
| 8192×4096 | 2048 | 25.2 M | ~4.9 km |
| 16384×8192 | 4096 | 100.7 M | ~2.4 km |

Memory budget at N=4096: one float32 field is ~400 MB. The roughly 10–14
working fields (height, uplift, K, rain, discharge, receiver index, sediment,
masks, …) total about 4–6 GB. That fits a 12 GB+ GPU. Supporting smaller GPUs
means processing faces in tiles, which is deferred unless needed.

Optional `sim_supersample` (e.g. 1.25×) lets the simulation run finer than
the output resolution.

### 4.3 Conventions

- Longitude −180°…+180°, left→right. Latitude +90° (top) … −90° (bottom).
- Cube face order and orientation: fixed and documented in the code, with a
  diagram, matching what Blender expects for cube-map displacement (to be
  checked during Phase 1).
- Elevation in metres relative to sea level (0 m).

## 5. Inputs

All raster inputs are equirectangular images. Size can be anything with a
2:1 ratio; they need not match the output size.

| Input | Required | Format | Meaning |
|---|---|---|---|
| `levels` | yes | 8/16-bit PNG/TIFF, palette or grayscale | Painted terrain classes: ocean, lowland, midland, highland (more levels allowed). Mapping from pixel value to class set in config. |
| `coast` | no | Grayscale, ideally higher-res and anti-aliased | Exact land mask. Overrides the land/sea boundary taken from `levels`. The 0.5 iso-line is the coast, so it is sub-pixel accurate. |
| `ocean_depth` | no | Grayscale | Optional painted bathymetry hints (trenches, ridges) added to the generated shelf. |
| `hardness` | no | Grayscale 0–1 | Rock resistance. Scales K down (harder = less erosion). |
| `rainfall` | no | Grayscale 0–1 | Precipitation. Weights drainage area into discharge. |
| `uplift` | no | Grayscale 0–1 | Tectonic uplift rate. Keeps ranges high while erosion cuts them. |
| `protect` | no | Grayscale 0–1 | 1 = no erosion or deposition here. |
| `rivers` | no | Black/white lines | Forced river courses. Flow is routed along the line and the bed carved (§7.5). |
| `lakes` | no | Black/white areas | Forced lakes with a flat surface (§7.5). |

Later options: other cylindrical projections for input (Mercator up to a
latitude limit, Lambert cylindrical equal-area), and vector coastlines
(SVG/GeoJSON).

### 5.1 Resampling input onto the cube sphere

- Continuous maps (hardness, rainfall, …): bilinear sampling of the equirect
  at each cell's longitude/latitude, wrapping at ±180°.
- Categorical `levels`: each class becomes a 0–1 indicator, which is
  resampled and then turned back into a class by majority. Painted edges must
  not produce in-between classes by accident.
- Coast: the anti-aliased mask is resampled bilinearly. A **signed geodesic
  distance field** to the coast is built on the cube sphere (positive inland,
  negative at sea), e.g. with jump flooding or a fast-marching method. Coast
  locking, shelf generation and smoothing falloff all use this field.

## 6. Pre-processing pipeline (on the sphere)

### 6.1 Base height from levels

Each class gets a base value from config (e.g. lowland 0.15, midland 0.45,
highland 1.0 on a 0–1 scale; ocean handled separately in §8).

### 6.2 Mask-aware spherical Gaussian smoothing

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

### 6.3 Height curve

- Maps the 0–1 smoothed value to metres with a monotone spline through
  control points given in config, like Gaea's Curve node:
  `curve = [[0,0],[0.2,150],[0.5,900],[0.8,3500],[1.0,7500]]`.
- The CLI saves a preview image of the curve.

### 6.4 Seed detail (noise)

Erosion on a smooth surface gives straight, parallel channels. A little noise
lets drainage patterns branch.

- **3D** fractal noise (simplex/fBm, optionally ridged or domain-warped),
  sampled at each cell's 3D position. Sampling in 3D means no seams or polar
  pinching.
- Amplitude scales with height (e.g. `amp = a0 + a1·h`), so lowlands stay
  calm and mountains rough.
- Reproducible from a `seed` value.

## 7. Erosion model

### 7.1 Governing equation (land cells)

```
∂h/∂t = U(x)                         uplift (optional map)
      − K(x) · Q^m · S^n              fluvial incision (stream power)
      + κ ∇²h                          hillslope diffusion (soil creep)
      + D(x)                           deposition (§7.3)
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

### 7.2 Numerical method

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
- **GPU strategy**: flow routing, drainage accumulation and depression routing
  follow the GPU-parallel approach of [FastFlow, Jain et al. 2024][fastflow]
  (O(log n) iterations for flow routing). The Braun–Willett solve
  parallelises the same way, by walking up the drainage tree.
- **Multigrid / cascade**: run first at low N (e.g. 512), upsample, continue at
  2048, then (optionally) 4096. Big drainage basins form cheaply at coarse
  resolution, and fine levels add detail. The coarse level is also the preview.
- **Alternative to evaluate**: the analytical stream-power solution of
  [Tzathas et al. 2024][tzathas], where time works as an "erosion amount"
  slider instead of an iteration count. Evaluate in Phase 3 if iteration
  counts at 16K get painful.

### 7.3 Deposition (simple)

- Each cell carries a sediment flux `Qs` downstream (eroded material
  accumulated along the drainage tree).
- Deposition rate `G · Qs / Q` (G = deposition coefficient). Material drops
  where slope or flow falls: valley floors, plains, river mouths.
- Sediment reaching the coast goes to the marine step (§8.2).
- Total eroded and deposited volume is logged as a sanity check.

### 7.4 Stopping

- Set by `iterations` per level, or `erosion_strength` (0–1) mapped to total
  simulated time.
- Checkpoints (all fields) are saved per level, so a run can resume or be
  re-run from any stage with new parameters.

### 7.5 Forced rivers and lakes

- **Rivers**: painted lines are converted to cell paths on the cube sphere,
  ordered by distance to the coast. Along a path, the receiver is forced to be
  the next cell downstream, and height is clamped to be monotone decreasing.
  A small extra K multiplier carves a clear valley. Lines must reach the sea
  or another river; the tool checks this and reports paths that don't.
- **Lakes**: painted areas are held at a flat water level, the lowest rim
  height or an explicit value. Flow is routed through to their lowest outlet.
  The lake mask is exported.

### 7.6 Later (not v1)

- Thermal/talus erosion pass for scree slopes.
- Droplet erosion detail pass at final resolution (the repo's
  `simulation.py` is a flat-grid reference).
- Multiple flow direction routing for softer, braided rivers in flat areas.
- Glacial erosion (U-shaped valleys) at high latitude or altitude.

## 8. Ocean and coastline

### 8.1 Generated bathymetry

From the signed coast distance d (km offshore):

- Continental **shelf** from 0 down to `shelf_depth` (default −130 m) over
  `shelf_width_km` (default 80 km, optionally varied with noise).
- **Slope** down to `abyss_depth` (default −4000 m) over `slope_width_km`.
- Abyssal plain with low-amplitude 3D noise.
- Optional `ocean_depth` painted map added on top.

### 8.2 Marine deposition

- Sediment flux arriving at each river mouth is spread over sea cells by
  diffusion within `marine_diffusion_km`, building submarine fans and
  shelf deposits.

### 8.3 Coastline modes (`coast.mode`)

- `locked` (default): after every step, land cells are clamped to ≥
  `min_land_height` (e.g. +1 m) and sea cells to ≤ `max_sea_height`
  (e.g. −1 m). The output land mask exactly matches the input coast.
- `deltas`: sea cells near river mouths may rise above 0 m through
  deposition, within `delta_max_km` of the original coast. Elsewhere the
  coast stays locked.

## 9. CLI and configuration

### 9.1 Commands

```
globerode run   world.toml              # full pipeline
globerode run   world.toml --preview    # N=512, fast; writes previews only
globerode run   world.toml --from erode # resume from a saved stage checkpoint
globerode stage world.toml prep         # run one stage: resample | prep | erode | export
globerode export world.toml --format exr,png16,cube,masks
globerode inspect checkpoint.npz        # stats, histograms, quick images
```

### 9.2 Example config

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

[input.classes]                    # pixel value -> class
0   = "ocean"
85  = "lowland"
170 = "midland"
255 = "highland"

[prep]
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
shelf_depth_m = -130
shelf_width_km = 80
slope_width_km = 150
abyss_depth_m = -4000
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

## 10. Outputs

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

Blender notes go in the README: displacing a UV sphere with the equirect EXR,
using the cube faces on a quad-sphere mesh, and recommended vertical
exaggeration.

## 11. Performance targets

| Run | Target |
|---|---|
| Preview (N=512) end-to-end | < 2 min |
| 8K (N=2048) full run | < 1 h (overnight acceptable) |
| 16K (N=4096) full run | overnight (< 12 h) |

Reference hardware: one NVIDIA GPU with 12 GB+ VRAM, 32 GB system RAM.

## 12. Acceptance tests

1. **Seams**: no visible step or crease across cube-face edges or at ±180°
   (automatic check on height gradients along seams).
2. **Polar consistency**: the same test continent placed at the equator and
   at a pole gives statistically matching results (river density, relief).
3. **Coast lock**: in `locked` mode, final land mask equals input land mask
   exactly.
4. **Drainage**: every land cell drains to the sea or a painted lake.
5. **Determinism**: same config + seed gives an identical result.
6. **Smoothing scale**: a σ = 100 km blur measured on the output has the same
   km width at 0°, 60° and 85° latitude.
7. **Mass balance**: deposited + exported sediment ≈ eroded volume (within
   tolerance).
8. **Forced rivers**: painted river paths appear in `mask_rivers` with no gaps.

## 13. Technology (proposed)

- **Python 3.11+** CLI and orchestration. Config via TOML.
- **GPU kernels**: one of NVIDIA Warp, CuPy (raw CUDA kernels) or Numba-CUDA.
  Chosen in Phase 0 by a small flow-routing benchmark.
- **CPU reference** implementation (NumPy + Numba) of every stage, used for
  tests and for GPU-less previews.
- I/O: `imageio`/`tifffile`/`OpenEXR` (16-bit and float support required).
- The repo's existing scripts (`simulation.py`, `river_network.py`, `util.py`)
  stay as reference. New code goes in a `globerode/` package.

## 14. Milestones

| Phase | Deliverable |
|---|---|
| 0 | Cube-sphere grid, equirect↔cube resampling, seam tests, GPU library choice. |
| 1 | Pre-processing: levels, coast SDF, spherical smoothing, curve, noise, export. Gives a usable smoothed globe with no erosion. |
| 2 | CPU reference erosion: flow routing, priority-flood, Braun–Willett, diffusion, deposition, masks. Tuned at preview resolution. |
| 3 | GPU erosion + cascade to 8K. Performance targets. Tzathas-style analytical option evaluated. |
| 4 | Optional maps (hardness, rain, uplift, protect, forced rivers/lakes), ocean shelf + marine deposits, delta mode. |
| 5 | 16K support, memory tiling if needed, Blender docs and example scene. |

## 15. Open questions

1. Are your painted maps plate carrée (equirectangular), or another
   cylindrical projection such as Mercator?
2. Would you also paint ocean levels (shelf / deep), or should the generated
   shelf be enough?
3. How many height levels at most: exactly 3, or should any number be
   supported?
4. Do you want rivers to be visible water in the height map (a flat water
   surface in wide rivers), or only as valleys plus a mask?
5. Is "uplift keeps mountains as painted" the right feel, or should erosion
   be allowed to wear ranges down a lot (an "age" slider)?
6. Which Blender setup do you use for rendering (displacement modifier,
   adaptive subdivision in Cycles, or something else)? This sets cube-face
   orientation and file format details.
7. Interest in a later GUI/globe viewer, so the core API can plan for it?

## References

- Ronchi, Iacono & Paolucci (1996), *The "Cubed Sphere": a new method for the solution of partial differential equations in spherical geometry*, J. Comput. Phys. 124(1). https://doi.org/10.1006/jcph.1996.0047
- Braun & Willett (2013), *A very efficient O(n), implicit and parallel method to solve the stream power equation*, Geomorphology 180–181. https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.stream_power.fastscape_stream_power.html
- Barnes, Lehman & Mulla (2014), *Priority-Flood: an optimal depression-filling and watershed-labeling algorithm for DEMs*, Computers & Geosciences 62. https://arxiv.org/abs/1511.04463
- Cordonnier et al. (2016), *Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion*, CGF 35(2). https://diglib.eg.org/handle/10.1111/cgf12820
- Schott, Paris, Fournier, Guérin & Galin (2023), *Large-scale Terrain Authoring through Interactive Erosion Simulation*, ACM TOG 42(5). https://h-schott.github.io/p/uplift — code: https://github.com/H-Schott/StreamPowerErosion
- Tzathas, Gailleton, Steer & Cordonnier (2024), *Physically-based Analytical Erosion for fast Terrain Generation*, CGF 43(2). https://diglib.eg.org/handle/10.1111/cgf15033
- Jain, Kerbl, Gain, Finley & Cordonnier (2024), *FastFlow: GPU Acceleration of Flow and Depression Routing for Landscape Simulation*, CGF 43(7). https://diglib.eg.org/handle/10.1111/cgf15243

[ronchi]: https://doi.org/10.1006/jcph.1996.0047
[bw]: https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.stream_power.fastscape_stream_power.html
[barnes]: https://arxiv.org/abs/1511.04463
[cordonnier]: https://diglib.eg.org/handle/10.1111/cgf12820
[schott]: https://h-schott.github.io/p/uplift
[tzathas]: https://diglib.eg.org/handle/10.1111/cgf15033
[fastflow]: https://diglib.eg.org/handle/10.1111/cgf15243
