---
name: qr-v2-run-best-commit
description: The selected best commit for the 2026-06-22-09-10-03-qr_v2 run is 396b1ff8 (kernel blob 58aa21b3) and it IS correct
metadata: 
  node_type: memory
  type: project
  originSessionId: eddc7272-2635-4c98-821c-2891107f63aa
---

For the `2026-06-22-09-10-03-qr_v2` autocuda run, the selected best is commit
`396b1ff8d59c14539c3a5774c4f758578038efb5` (branch
`autocuda/best/2026-06-22-09-10-03-qr_v2`, "brief-67 COMBINE: fp16x3 sweep1 onto
the full host-axis stack best"), qr_v2 submission blob `58aa21b3`. This IS the
genuine fastest VALIDATING kernel of the run — the selection is correct.

The apparent 2040(dashboard) vs 2162(other node)/2197(here) gap is NOT a
wrong-commit problem; it is the toolchain baseline shift — see
[[qr-v2-toolchain-baseline-shift]], now CONFIRMED by A/B on this node:
- new stack (current): 2197.4 / 2196.0us
- old stack (pre-upgrade torch2.11/triton3.6): 2045.6 / 2045.9us
i.e. ~2040 was a valid old-stack number; same exact kernel bytes either way.

NOTE the separate `e34e57da` / blob `67444374` ("optimized peak", ~1763us) is a
LEADERBOARD-regime reference from a different (06-23-17-54-06) run, not
comparable to the local-us numbers above.
