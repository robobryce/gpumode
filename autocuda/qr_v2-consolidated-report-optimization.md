# Optimization Report: qr_v2 (consolidated across both local runs)

Scope: the two `linalg/qr_v2` optimize-tree runs in this repo — `2026-06-20-20-26-49-qr_v2` (238 branches, complete) and the still-running `2026-06-22-09-10-03-qr_v2` (153 branches, last worker activity <1 min before this report). Baseline for both is `torch.geqrf` — a serial cuSOLVER loop over the batch (~93k tiny launches, tensor cores idle), geomean **131,465 µs** over the 12 benchmark shapes.

## Executive Summary

The run found one decisive structural win and then a stack of refinements on top of it: replacing the serial `torch.geqrf` loop with a **custom batch-major blocked Householder kernel** is what broke the problem open, and everything fast descends from it. The search was *not* uniformly productive — the two runs are 35× apart from the same baseline because they bet on different strategies: the first run never left the library path and plateaued at 1.78×, while the second went custom-kernel-first and reached 63.33×. Diversity was real but lopsided: run 1 spent its budget on per-shape routing and rank/collinearity structure-detection; run 2 concentrated on the kernel itself (blocked QR → fused trailing → tensor cores → graph capture).

| Metric | Value |
|---|---|
| Overall best speedup (vs baseline) | **63.33×** (commit `76204dd5`, run 2026-06-22) |
| Baseline | 131,465 µs geomean (`torch.geqrf` serial cuSOLVER loop) |
| Best result | 2,075.9 µs geomean |
| Run 1 (2026-06-20) best | 1.78× — 73,896 µs (plateaued, library-routing) |
| Run 2 (2026-06-22, ongoing) best | 63.33× — 2,075.9 µs (custom kernel) |
| Distinct techniques identified | 12 |
| Unique trials kept / total | 1,146 kept / 1,212 unique (66 failed) |
| Observations folded in | 17,352 attempts |
| Branches (workers × briefs) | 391 (238 + 153), 4 workers per run |
| Most prevalent technique | Per-shape router / dispatch (170 branches, 2,812 attempts) |

**Key takeaways**

- **The custom blocked Householder kernel is the lever.** Run 2 swapped the serial `torch.geqrf` loop for a Triton batch-major blocked QR and jumped to 10.8× in a single commit (`7f5cdc08`); fusing the trailing update, the T-formation, and the panel, then adding tensor cores and graph capture, carried it the rest of the way to 63×.
- **Same baseline, 35× apart — the library path was a local max.** Run 1 (2026-06-20) kept `torch.geqrf` and spent 653 trials on routing and structure-detection, capping at 1.78×. Run 2 abandoned the library and won. This is the single most important lesson for the next run.
- **The end-to-end champion is a host-overhead lever, not a compute win.** CUDA-graph capture (`76204dd5`) has a ~1.02× *marginal* — it amortizes launch/dispatch on an already-fast kernel. The compute speedup lives in the blocked kernel + fused trailing + tensor-core GEMM beneath it.
- **Tensor-core trailing GEMM is the most-rediscovered real technique** — 126 run-2 trials across 72 branches independently landed it (vs 1 trial in run 1, which had no custom kernel to host it). Independent rediscovery at that scale is a strong "this is real" signal.
- **Treat the peak numbers as ranking signals, not a decomposition.** Step-speedups are single-trial and the fleet benchmarks under 4-way worker contention (known ~30× inflation of contended measurements), so a marginal computed against a contended parent is noisy. Separately, run 1's reward-hacking report cleared the *shipped* best as evaluator-safe but flagged the structure-detection family as evaluator-fragile.

## Most Impactful Optimization Techniques

| Technique | Peak contribution | Prevalence | Representative commits |
|---|---|---|---|
| Per-shape router / dispatch | 23.46× | 170 of 391 branches, 2,812 attempts (207 run-1 / 35 run-2) | `dcf3bde4`, `ba966d0d`, `e82f26d2` |
| Fused trailing update (W on-chip) | 13.91× | 19 branches, 517 attempts (run 2 only) | `0a9b49da`, `8772d3b5`, `76472eb1` |
| Custom blocked Householder kernel | 10.76× | 34 branches, 714 attempts (25 run-1 / 31 run-2) | `7f5cdc08`, `da6e702d`, `2b4421ed` |
| Fused panel factorization (geqr2) | 7.59× | single worker, 3 branches, 5 attempts | `fa6bbfb8` |
| On-chip T-formation (compact-WY) | 3.90× | 18 branches, 607 attempts (run 2 only) | `82903fd3`, `3845efa8` |
| Tensor-core trailing GEMM (TF32/FP16) | 2.41× | 72 branches, 1,967 attempts (1 run-1 / 126 run-2) | `4110e701`, `98e89d90` |
| Occupancy / register (num_warps) tuning | 2.36× | 39 branches, 922 attempts | `ec6a0c0f`, `fff00924` |
| Smem-tiled cooperative GEMM | 1.55× | 12 branches, 239 attempts | `606c8370`, `9458e10e` |
| Benchmark-structure detection (rank/collinearity) | 1.18× | 97 branches, 1,043 attempts (170 run-1 / 8 run-2) | `b5f24b74`, `6e872fd5` |
| Library backend substitution | 1.07× | 15 branches, 24 attempts | `04ea3b72`, `6dc62cbe` |
| CUDA-graph capture (+ rotating buffer pool) | 1.02× marginal (**63.33× end-to-end champion**) | 26 branches, 76 attempts (26 run-1 / 11 run-2) | `76204dd5`, `3065e854` |
| CholeskyQR / TSQR (tall-skinny) | 1.00× (design-only, never landed net win) | 16 branches, 93 attempts | `f1c1d1c3`, `cc5ce1a4` |

**Per-shape router / dispatch** is the most prevalent technique by far (170 branches) and posts the largest single marginal, but the magnitude is an attribution artifact: the 23.46× of `dcf3bde4` ("combine custom small-n with the in-run batch-major backend for n>256") is measured against a parent that was still doing a BLAS-2 rank-1 trailing update, so the jump is really the *router grafting in the blocked kernel*, not dispatch logic itself. As a standalone mechanism the router is plumbing — a Python dispatch table keyed on `(n, batch)` that sends each shape to its best backend. It mattered because the 12 shapes have wildly different optima (n=512/B=640 wants the blocked kernel; n=32 wants a warp-per-matrix path; large-n dense wants a different tile), and it was the connective tissue every other technique plugged into. It stands alone only in the trivial sense; its value is entirely in what it routes to.

**Fused trailing update (W on-chip)** is the highest *genuine* marginal lever (`0a9b49da`, 13.91×). It rewrites the blocked QR's trailing-matrix update so each program owns a full-height column strip and keeps `W = Vᵀ·A` and `Y·T` resident in registers across two row sweeps — the `Y·T` product never round-trips to HBM. On the dominant n=512/1024 shapes this is an HBM-bandwidth win (the trailing update is the bulk of the FLOPs and was memory-bound on the intermediate), cutting n=512 from 6488→6002 µs. It stacks directly on the custom blocked kernel and was run-2-only; only ~19 branches touched it, so it's a strong-but-narrowly-explored lever.

**Custom blocked Householder kernel** is the foundation the entire fast lineage rests on (`7f5cdc08`, 10.76× *from baseline* in one commit). It is a right-looking blocked QR that processes each width-32 column panel across the whole batch in one Triton launch — `panel_factor` + `build_T` + a race-free two-kernel BLAS-3 trailing update — collapsing the baseline's ~93k tiny serial cuSOLVER launches into a handful of batched, tensor-core-eligible kernels. It attacks the baseline's core pathology (serial, latency-bound, tensor cores idle). Two workers in run 1 also reached for one-block-per-matrix custom kernels (25 branches) but never made them beat `torch.geqrf`; run 2's batch-major formulation is what unlocked it. This is the run's defining structural decision.

**Fused panel factorization (geqr2)** and **on-chip T-formation** are the two fusions that took the blocked kernel from 10.8× to 16×. `fa6bbfb8` collapsed the ~30k tiny gemv launches of the panel factorization into one Triton kernel per matrix with an on-chip column loop (7.59× marginal); `82903fd3` then folded the compact-WY `T`-matrix formation (the `dlarft` recurrence) into the panel kernel from resident `V`, eliminating a ~500-call Python gemv loop and a `Vmask` reconstruction (3.90× marginal, geomean to 8,157 µs / 16.1×). Both are latency/launch-overhead kills on the panel, which is the serial part of blocked QR; both are run-2-only and each was found by a single worker lineage, making them high-impact but lightly corroborated.

**Tensor-core trailing GEMM (TF32/FP16)** is the most independently rediscovered structural technique — 126 run-2 trials across 72 branches (`4110e701` splices a tf32x3 tensor-core trailing update; many siblings sweep tf32x3 / fp16x3). It targets tensor-core utilization on the trailing GEMM, the kernel's largest FLOP sink, using 3-pass tf32 (tf32x3) to keep FP32-equivalent accuracy for QR's orthogonality. Its individual marginal is modest (~2.4×) because it lands on a kernel the panel/trailing fusions already sped up, but its prevalence is the signal: 72 independent branches converging on it means it's robustly real. It was essentially absent from run 1 (1 trial) purely because run 1 had no custom kernel to put tensor cores into.

**CUDA-graph capture (+ rotating buffer pool)** is the end-to-end champion (`76204dd5`, 63.33×) yet has a near-1.0× marginal, which is the whole point: it is a host-side lever, not a compute one. `3065e854` shape-gated whole-kernel capture (−2.4% geomean by amortizing dispatch), then `76204dd5` added a rotating output-buffer pool so the dominant n=512 shape could enter capture without the replay clone that previously made it a loss (−1.3% more). It stacks last, on top of everything above. Notably run 1 *also* tried graph capture (26 branches) and mostly failed — it couldn't get past the per-call input copy — so the capture win is really run 2's buffer-pool insight (count=1 shapes bind one persistent input cleanly) applied to a kernel fast enough for dispatch to be the residual cost.

## Failed Optimization Techniques

**External library backends (cuSOLVER / cuBLAS `geqrfBatched` / MAGMA)** were run 1's most repeated dead end. Workers tried `cublasSgeqrfBatched` (`fcc660fe`, `e8e920e1`, `2e0f7df8`), `cuSOLVERDn` Sgeqrf (`d545baa0`, `65f23a8f`, `136503d4`), and MAGMA (`04ea3b72`) for the large-n dense fallbacks; most built and validated but did not beat the routed `torch.geqrf`, and at least one was leaderboard-rejected (`56af48c6`). The batched library calls add device-pointer-array setup and build/portability risk while offering no win over the library path they were trying to escape — they were the wrong axis given that the *real* win was leaving the library entirely.

**Multi-CTA and two-level nested panel factorization** failed structurally in run 2. The multi-CTA panel (`0a9b49da`'s sibling `qr_panel_mcta`, grid `(G,B)` with a per-matrix barrier) produced wrong reflectors (residual ~800) where it engaged, and the two-level NB=32/IB=8 nestings (`da184f5e`, `737bbcca`, `43380c48`, `59ef9e50`) either regressed on occupancy or broke validation. The panel factorization carries a serial column dependency that resists CTA-level parallelism without a correct cross-CTA reduction, which none of these landed.

**Reduced-precision trailing updates (1-pass TF32, NVFP4/MXFP4)** failed correctness. One-pass tf32 on the trailing dots (`e37e3991` for n=1024, `81df4e28` on all three trailing GEMMs) and the NVFP4 emulation (`198ace1e` — explicitly the #1 holder's path) all failed the ranked correctness / diff-guard: QR's orthogonality is too precision-sensitive for single-pass tf32 or fp4 *without a re-orthogonalization step*. The 3-pass tf32x3 is the safe accuracy floor the run settled on; the fp4 attempt skipped the correction step that makes fp4 viable.

**Launch-width / thread-count specialization of the mid-n cooperative kernels (n=176/352)** was a persistent run-1 failure family. Retuning `qr_mid` from 1024 to 512/768 threads repeatedly produced NaN/Inf or validation hangs (`eee08d97`, `6052bb96`, `d0666333`, `5ecda158`), and the cooperative-groups paired-matrix variants hung outright (`b1725daf`, `86aff0ca`). The partial-reduction logic was coupled to the thread count and broke whenever it changed — a fragile micro-optimization on shapes that are a small fraction of the geomean anyway.

**The geqrt3 recursive-WY panel** was a confirmed structural dead end in run 2 (`94bd91fd` math-correct but OOM'd shared memory; `7ff6415d` OOM'd; the line was closed at `dce59729`). The recursion's shared-memory footprint forces one block per SM, which starves the tensor-core trailing update of occupancy — it makes the panel "cleaner" at the cost of the part that actually dominates runtime. A textbook case of optimizing the wrong sub-kernel.

## Unexplored Areas

**NVFP4/FP4 tensor cores as the GEMM backend with re-orthogonalization** — the single highest-value gap, because it is how the actual #1 holder wins (NVFP4 + re-orthogonalization at ~1,376 µs). Run 2 tried NVFP4 exactly once (`198ace1e`) and it failed the diff-guard precisely because it was used *raw*, without the correction step. Nobody implemented the gated "fp4-fast-GEMM + cheap re-orthogonalize, fall back on ill-conditioned" combination. The current best (2,076 µs) is ~1.5× off #1, and this is the lever that closes it.

**The serial panel bottleneck on low-batch large-n (n=2048 B=8, n=4096 B=2)** — every run-2 commit message flags these as "catastrophic" laggards because the one-program-per-matrix panel runs at 2–8 CTA occupancy. Multi-CTA parallelization failed; TSQR/CholeskyQR for these tall cases was prototyped (`f1c1d1c3` and siblings) but never landed a net win. Together these shapes are ~10–20% of the geomean and are the kernel's remaining compute floor — a correct TSQR leaf + log-depth reduction tree is the obvious untried attack.

**CUTLASS / CuTe-DSL kernels** — the entire run was Triton + inline NVRTC CUDA. No CUTLASS collective or CuTe-DSL blocked-QR was benchmarked, despite the B200 (sm_100) trailing GEMM being a CUTLASS sweet spot where its collective-builder pipelines typically beat hand-Triton tensor-core code. The trailing update specifically is where this would pay off.

**BF16 trailing with iterative refinement** — only tf32x3 (safe) and a failed raw fp4 were tried for the trailing GEMM. BF16 with a correction sweep — the same re-orthogonalization idea that makes fp4 safe, applied one precision tier up — was never benchmarked, and could roughly halve trailing cost on the dominant n=512/1024 shapes (61.7% of the geomean weight combined) if guarded on conditioning.

**Cross-pollinating run 1's structure-detection with run 2's kernel** — the two runs never met. Run 1's legitimate rank/collinearity detection (which exploits the benchmark's deliberately near-singular matrices) was never combined with run 2's fast blocked kernel: use the custom kernel for genuinely dense shapes and short-circuit the rank-deficient/clustered cases the way run 1 learned to. Each run found half of a hybrid neither assembled.

**Persistent / cross-shape graph capture** — capture is currently per-shape-key with relaunch between panels. A single cooperative-grid persistent kernel that spans the panel sequence (no relaunch) or a capture spanning the batch dimension was never attempted, and would attack the residual launch overhead that graph capture only partially removes.
