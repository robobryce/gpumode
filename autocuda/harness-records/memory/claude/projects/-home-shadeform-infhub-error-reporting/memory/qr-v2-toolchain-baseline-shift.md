---
name: qr-v2-toolchain-baseline-shift
description: qr_v2 local-benchmark numbers shifted ~2040us->~2196us mid-run when the venv was upgraded triton 3.6->3.7.1 / torch 2.12.1+cu130; cross-regime comparisons are invalid
metadata: 
  node_type: memory
  type: project
  originSessionId: eddc7272-2635-4c98-821c-2891107f63aa
---

During the `2026-06-22-09-10-03-qr_v2` autocuda optimize-tree run, the venv was
upgraded mid-run (~2026-06-24 16:34-16:48 UTC) from torch 2.11.0+cu128 /
triton 3.6.0 to torch 2.12.1+cu130 / triton 3.7.1 / nvrtc-13.0. The
BYTE-IDENTICAL winning kernel (qr_v2 submission blob `58aa21b3`, commit
`396b1ff8` = branch `autocuda/best/2026-06-22-09-10-03-qr_v2`) measures:
- ~2040.9us pre-upgrade (triton 3.6)
- ~2195.8us post-upgrade (worker logged this as the "BASELINE-SHIFT finding",
  commit a0a7877b)

CONFIRMED EXPERIMENTALLY on this node (2026-06-25, exclusive lock, two runs each):
- old stack (torch 2.11.0+cu128 / triton 3.6.0): 2045.6 / 2045.9us
- new stack (torch 2.12.1+cu130 / triton 3.7.1): 2197.4 / 2196.0us
Per-shape, the regression is real: n=512 3746->3909us, n=1024 3451->3875us. So
the toolchain upgrade IS the cause; the kernel is genuinely ~7.4% slower under
the new stack on identical bytes. The ~2040 dashboard global-best is a valid
old-stack number, not a wrong commit.

To A/B the two stacks WITHOUT disturbing the live fleet's shared `.venv`:
built an isolated `/home/shadeform/gpumode/.venv-old-stack` (uv, old wheels were
cached) and redirected the harness via `GPUMODE_ENV=.../.gpumode-oldstack.env`
(a custom gpumode.env that sets GPUMODE_VENV_PYTHON to the old venv). GOTCHA:
exporting `GPUMODE_VENV_PYTHON` directly does NOT work — `harness/env.sh` sources
`~/.config/gpumode/gpumode.env` which plain-assigns (clobbers) it; but env.sh
checks `$GPUMODE_ENV` FIRST and breaks, so a custom file via GPUMODE_ENV wins.
Always verify the stack actually loaded (print torch/triton, or check the
isolated TRITON_CACHE_DIR populated) — a silent fallback to the new venv looks
identical in the result block. Both the old venv and env file were kept.

So a dashboard/global-best of ~2040 is a STALE pre-upgrade number; the true
current local score for that exact kernel is ~2196us. The ~2162us seen on
another node is the same post-upgrade regime + node variance, NOT a different
kernel. **Why:** never compare qr_v2 us-numbers across the upgrade boundary or
across nodes without re-benchmarking. **How to apply:** to compare a kernel's
speed, re-run `autocuda run exclusive --data-dir <data> -- bash harness/benchmark.sh
linalg/qr_v2` in the CURRENT toolchain; don't trust logged numbers from a
different regime.

Second hazard: profiling/kill briefs re-log the ambient "best-so-far" number
onto commits that don't contain that kernel (e.g. `e812d9d6`, a scalar
megakernel whose real score is 6561us, also shows 2040.9). So "lowest logged
number" is NOT a safe best-commit selector — verify the submission blob.
Related: [[qr-v2-run-best-commit]]
