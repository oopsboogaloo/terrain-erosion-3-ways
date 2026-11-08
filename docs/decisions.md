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
