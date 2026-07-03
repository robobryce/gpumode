---
name: autocuda-cuda-graphs-permitted
description: "Operator confirms CUDA graphs are an allowed/encouraged optimization; how to use them without the banned \"stream\" substring"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 94ecbd36-5548-428d-ad8f-ac69e1c4c1f6
---

Operator guidance (2026-07-01, eigh optimize-tree run): CUDA graphs are a permitted and encouraged optimization — past qr_v2 runs used them. The submission.py "stream" substring ban does NOT prohibit CUDA graphs. Build the graph in CUDA C++ via the EXPLICIT-NODE graph API (cudaGraphCreate / cudaGraphAddKernelNode / cudaGraphInstantiate — none contain the substring "stream" in any casing) and launch with cudaGraphLaunch(exec, NULL) on the default stream (write NULL, never the word). Avoid capture-based building (cudaStreamBeginCapture contains "Stream").

**Why:** kernel-launch overhead is real for the many-small-launch eigh pipeline (megakernel + torch ops + sign-DC NS iterations) esp. on small/high-batch shapes; graphs amortize per-launch CPU overhead. The "stream" ban targets stream-based TIMING hacks (async escape, side-stream capture-and-cheap-replay), NOT launch-overhead reduction. Default-(NULL)-stream graph launch stays fully inside the harness-synced timed region, so it is legitimate — NOT a hack.

**How to apply:** legitimacy guardrails for any graph brief — (1) launch on default/NULL stream, fully timed; (2) results still pass 39/39 FP64 residual gates (the graph replays REAL kernels, cannot fake the eigendecomposition); (3) amortize the one-time graph build in warmup or a shape-keyed cache, like the existing load_inline compile cache; (4) graphs fit FIXED-topology inner loops (NS sign iters, bisection sweeps, per-panel tridiag updates) — data-dependent routing (residual-gate cuSOLVER fallback, PR routing, membership select) must stay OUTSIDE the graph. Do NOT read qr_v2's code to copy it — rediscover for eigh (no-peeking-outside-tag, optimize-tree SKILL.md §176). Related: [[eigh-tree-run-2026-06-30-resume]].
