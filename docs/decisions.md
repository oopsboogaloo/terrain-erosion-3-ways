# Decision log

Each entry: what was decided, why, and when. Newest last. Changing a
decision means adding a new entry that replaces the old one, not editing
history.

| # | Date | Decision | Why | Source |
|---|---|---|---|---|
| D1 | 2026-11-08 | Compute grid geometry on the fly; cache only cell area | Saves ~4 GB at 16K; trig is cheap compared with memory traffic | DESIGN §3.2 |
| D2 | 2026-11-08 | Smoothing by brute-force masked diffusion at full resolution | Simple; about 1 min at 8K; multi-resolution risks losing narrow land | DESIGN §6.2 |
| D3 | 2026-11-08 | Own 3D gradient-noise implementation | Same seed gives the same world forever; directly testable | DESIGN §6.4 |
| D4 | 2026-11-08 | Fill pits every erosion iteration (lakes fill instantly) | Simplest rule that guarantees every river reaches the sea | DESIGN §7.2 |
| D5 | 2026-11-08 | Stream power with n = 1 only | Unconditionally stable closed-form update; general n waits for a need | DESIGN §7.4 |
| D6 | 2026-11-08 | Hypsometry restore after erosion, default on | Keeps heights about as painted without fine K/U tuning | DESIGN §7.6 |
| D7 | 2026-11-08 | argparse CLI | No dependency; enough for a Claude-driven tool | DESIGN §11 |
| D8 | 2026-11-08 | Previews rendered in NumPy; Blender only for `render` | Fast, no Blender needed in CI | DESIGN §12 |
| P1 | 2026-11-08 | Review policy A: user reviews milestone PRs (steps 4, 5, 6, 10); Claude merges the others on green CI | Keeps the user on the key results without blocking routine steps | PLAN §3 |
| M1 | 2026-11-08 | Routing speed check (PLAN step 3) passed: Priority-Flood+ε + receivers + drainage area take 0.5–0.7 s at N=512 and 12–18 s at N=2048 (smooth vs white-noise terrain), peak RSS 1.5–2.2 GB, in the 4-core sandbox | ~100 iterations at 8K ≈ 20–30 min of routing, far under the 60 s/iteration stop threshold | PLAN §1 step 3 |
| D9 | 2026-11-08 | Heights are float64 during routing and erosion (other fields stay float32) | ε = 1e-3 m fill steps are below float32 resolution at mountain heights (~0.5 mm at 5 km); costs ~100 MB extra at 8K | DESIGN §4, §7.2 |
| D10 | 2026-11-08 | No polar supersampling in cube → equirect export; plain bicubic | Output width is always 4N, so no output pixel covers more than about one cell anywhere; near the poles pixels are smaller than cells, so bicubic is already smooth. Less code | DESIGN §9.1 |
| D11 | 2026-11-08 | Each level class carries its own `base` (land) or `depth_m` (ocean) in `[[input.classes]]`, instead of a separate `prep.base` table | One place per level; any number of levels | DESIGN §7.1, SPEC §11.2 |
| D12 | 2026-11-08 | Smoothing precomputes a temporary (ncells, 4) neighbour table and one (N, N, 4) weight table shared by all faces (exception to D1) | ~2 000 diffusion steps at 8K make on-the-fly geometry far too slow; the table is 16 B/cell (0.4 GB at 8K, 1.6 GB at 16K), freed after each smoothing call. Faces share weights because every face is the same grid under a cube rotation (tested) | DESIGN §6.2 |
| M2 | 2026-11-08 | Measured, 8K mini_earth in the sandbox: land smoothing ≈110 s (2 010 steps, σ=120 km), whole prep ≈270 s, full run 4.7 min, peak 2.9 GB. Coast distance accuracy 0.4–0.6 cell max error (limited by the cell-sized coastline), blur width within 0.3% of σ at 0°, 60°, 85° | Replaces the DESIGN §6.2 estimate of 20–60 s; still small next to erosion | DESIGN §6.2 |
| D13 | 2026-11-08 | Halo padding fills cube-corner blocks with a rough value, then refines them twice from the padded arrays | The first version was only first-order accurate at the 8 cube corners (error grew with N); now ≈0.1 cell² max at every N. Affects export and cascade upsampling | DESIGN §3.4 |
| D14 | 2026-11-08 | Cascade: before upsampling, sea cells are set to sea level (0 m); the fine level's start = upsampled land + the noise octaves only the fine grid can show; uplift and coast come from the fine prep | Ocean depths would otherwise drag coastal land down through bicubic overshoot; fine coast stays sharp | DESIGN §7.7 |
| D15 | 2026-11-08 | `mask_erosion` = max(h_init − h_final, 0) at the final level (depth removed relative to the prepared surface), normalised by its maximum | Simple and meaningful after hypsometry restore; per-iteration accumulation adds state for no visible gain | DESIGN §9.3 |
