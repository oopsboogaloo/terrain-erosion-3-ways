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
