# GPUMODE QR — AutoCUDA Optimization Report — 2026-06-13 to 2026-06-16

## Executive summary

Autonomous optimization took batched QR decomposition on a B200 to a **final
58.2× speedup** over the `torch.geqrf` baseline (2252.6 µs vs 131,025 µs) — the
honest, correctness-corrected result. The kernel replaces the latency-bound
cuSOLVER panel with three stacked structural wins: a **recursive compact-WY panel
factorization** that turns thousands of serial per-column launches into a
logarithmic-depth recursion (biggest single lever, **3.75×** marginal), **3×TF32
split-precision trailing GEMMs** that move the trailing update onto the tensor
cores while staying inside the benchmark's accuracy tolerance (**3.28×**, the
connective tissue of the whole kernel), and a **CholeskyQR / FP64-Gram route** for
the tall-skinny shapes a panel factorization starves on (**3.28×**). The accuracy
tolerance matters because the hardest benchmark inputs are *ill-conditioned*
(condition number 2 — moderately stretched matrices whose near-parallel columns
amplify rounding error), so a too-aggressive low-precision shortcut fails the
orthogonality check; 3×TF32 is the cheapest precision that holds. Remaining
headroom is concentrated in one shape — batch-640 n=512 is ~71% of total cost — but
that shape is already the most-optimized (megakernel and TSQR rewrites were tried
and lost; its panel is occupancy/latency-bound), so the open bets are its serial
panel latency plus precision regimes (FP8 with residual correction) and library
paths (CUTLASS epilogue fusion, CUDA graphs) never tried.

| Metric | Value |
|---|---|
| Best geomean speedup — B200 (sm_100) | 58.2× over `torch.geqrf` (2252.6 µs vs 131,025 µs baseline) |
| Experiments | 2 total (both with trial data) |
| Trials (kept / failed) | 2,707 (2,127 kept / 580 failed) |
| GPU Archs Used | 2× B200 (sm_100) |
| Models Used | 1× Claude Opus 4.8, 1× GPT 5.5 |
| Highest-impact family | recursive compact-WY panel, **3.75×** marginal (`6de17375`) |

**Key takeaways**

- **Replacing the panel factorization was the core win.** A recursive
  compact-WY (Elmroth–Gustavson) panel (3.75×) plus a custom one-CTA-per-matrix
  shared-memory blocked-Householder kernel (2.36×) eliminated the cuSOLVER
  `geqr2` panel that was 90% of baseline time.
- **Tensor cores carry the trailing update.** 3×TF32 split-precision GEMMs
  (3.28×) recapture the cross terms a single TF32 pass drops, so the cond=2
  shapes run on tensor cores within tolerance — a single-TF32 shortcut fails the
  accuracy gate.
- **Algorithm choice matters by shape.** A CholeskyQR / FP64-Gram route (3.28×)
  beats Householder on occupancy-starved tall-skinny shapes (one SYRK vs
  thousands of tiny launches), and the large-n dense shapes get their own tuned
  backend (BF16 two-level + vendor CholeskyQR, 2.95×) — but each is catastrophic
  off its regime (a CholeskyQR1 on the wide shape regressed 276×), so all are
  tightly gated.
- **An exact-repair layer is what makes the fast paths safe.** A conditioning
  probe routes each matrix, and the ones it flags ill-conditioned or near-collinear
  are re-factored exactly in FP32 (2.66×) — the precondition for running 3×TF32/BF16
  at all. The honest final at **58.2× (2252.6 µs)** came from extending it: a
  correctness pass adding reward-hack removal, a determinism fix, and a
  near-collinear repair.
- **Submittability was a real constraint, not just speed.** The fastest engine
  timed out the remote 300 s build budget, so a build-surface partitioning
  technique (lazy-split the cold LU/Ozaki extension, prune dead bindings, hardwire
  the active route) was needed to make the kernel land at all — it barely moves the
  benchmark µs but gates whether a fast result counts.
- **Remaining headroom is inside the most-optimized shape's panel.** The batch-640
  n=512 shape dominates cost (~71%) and is already the most-targeted — its panel,
  3×TF32 trailing update, and repair were tuned across hundreds of trials, and
  fused-megakernel and TSQR-style rewrites were tried and lost. Its panel-apply
  megakernel is occupancy/latency-bound (5 blocks/SM, not compute-bound), so the
  remaining lever is the serial Householder latency, not another shape-specialized
  algorithm.

## Most Impactful Optimization Techniques

| Technique | Peak contribution | Prevalence | Representative commits |
|---|---|---|---|
| Recursive / two-level compact-WY (Elmroth–Gustavson) panel | 3.75× | ~220 clusters | `6de17375`, `41c7faac` |
| 3×TF32 split-precision tensor-core trailing GEMMs | 3.28× | ~350 clusters | `f8dcc18c`, `939bed27` |
| CholeskyQR / FP64-Gram tall-skinny route | 3.28× | ~450 clusters | `c03080ab`, `03ad26d6` |
| Large-n dense-shape backend specialization (BF16 two-level + vendor CholeskyQR) | 2.95× | ~45 clusters | `6e50bba5`, `b8dace46` |
| CholeskyQR3 + custom diag-block-LU reconstruction | 2.90× | ~220 clusters | `0cde21e3`, `92a203de` |
| FP32 exact repair / near-collinear orthogonality rescue | 2.66× | ~275 clusters | `0382fd85`, `f7c8d86e` |
| Custom 1-CTA/matrix smem blocked-Householder panel kernel | 2.36× | ~140 clusters | `2adb3aac`, `7c157184` |
| Per-matrix conditioning-probe routing (detect, then dense-speculate or repair) | 1.31× | ~300 clusters | `8283596e`, `22519655` |
| Remote build-surface partitioning (fit the 300 s build budget) | build-time | ~310 clusters | `fd4b31c1`, `f1a804e6` |

**Recursive / two-level compact-WY panel.** A hybrid dispatcher
running a recursive blocked-Householder factorization (width-64 split to a base-16
per-column kernel, inter-half block reflector via a 3×TF32 GEMM in `41c7faac`),
falling back to `torch.geqrf` where batch parallelism is low. It replaces
thousands of latency-bound per-column cuSOLVER `geqr2_smem_domino` launches — the
baseline's 90%-of-time hotspot — with a logarithmic-depth recursion whose work
maps onto tensor cores. The algorithmic skeleton the other panel techniques refine.

**3×TF32 split-precision trailing GEMMs.** Splits each FP32 operand
into TF32 hi+lo parts and issues three batched tensor-core GEMMs (hi×hi, hi×lo,
lo×hi) to recapture the cross terms a single TF32 pass drops, on the trailing
update `A2 -= V·(Tᵀ·(Vᵀ·A2))`. The most pervasive technique and the enabling layer
the panel kernels stack on — without it the ill-conditioned shapes can't use the
tensor cores within tolerance (a single-TF32 pass fails the orthogonality checks).

**CholeskyQR / FP64-Gram tall-skinny route.** A distinct *algorithm*
for occupancy-starved tiny-batch large-n shapes: form G = AᵀA via one SYRK,
Cholesky-factor it, reconstruct Q. Communication-optimal here — one SYRK replaces
thousands of tiny panel launches on a near-empty GPU. Gated by shape
(`batch ≥ 256 & n ≤ 1024`).

**Large-n dense-shape backend specialization.** Rather than run the general
Householder panel on the large-n dense shapes (batch-60 n=1024, batch-8 n=2048),
route them to a shape-tuned backend: a widened BF16 two-level blocked-Householder
backend (`6e50bba5`, outer-block 72 / inner-block 24 to cut trailing-update
passes, 2.95× marginal) and a gated vendor path handing n≥2048 to cuBLAS/cuSOLVER
CholeskyQR (`b8dace46`, 2.34×). A shape-by-shape algorithm choice specialized to
the large-n mixed batches, distinct from the tall-skinny CholeskyQR route above.

**CholeskyQR3 + custom diag-block-LU reconstruction.** Layers a third
Cholesky refinement step plus a bespoke diagonal-block LU kernel onto the
CholeskyQR route for the shapes where two passes leave residual orthogonality
error; the refined `92a203de` made the diag-LU 2-wide-pivot and race-free.

**FP32 exact repair / near-collinear orthogonality rescue.** The correctness
backstop the speed of the whole kernel is conditioned on: matrices the
conditioning probe flags as ill-conditioned or near-collinear are pulled off the
low-precision fast path and re-factored exactly in FP32 (the n512 path routes
rank-deficient / clustered / mixed flags through the shared-memory two-level
factorization, `0382fd85`). Variants span hardening the n=512 near-collinear
factor margin (`f7c8d86e`, 83 attempts) and re-sizing/re-fusing the repair to
proportional scratch with no fixed cap. It is what lets the aggressive 3×TF32 /
BF16 routes run at all without failing the orthogonality gate, and the final
correctness pass that produced the honest 58.2× is an extension of this family (a
near-collinear repair the differential guard surfaced).

**Custom 1-CTA/matrix blocked-Householder panel kernel.** A bespoke
sm_100 kernel (via `load_inline`) that factors a whole nb-wide panel
column-by-column inside one CTA with the panel resident in shared memory,
eliminating the O(width) PyTorch kernel storm and the compact-WY T-factor build.
The most-rediscovered custom kernel; column-major, OV-fold, pivot-cooperative, and
float4-vectorized raw-V load/writeback variants all descend from it.

**Per-matrix conditioning-probe routing.** The dispatch spine the precision
techniques sit on: a fused probe classifies each matrix's conditioning, then routes
it to a low-precision speculative path or to the exact FP32 repair above.
Generalizing the all-stress bypass (`8283596e`) and skipping the speculative dense
path when every matrix is flagged exact (`22519655`) are the highest-marginal
routing variants, but the family is broader — it is the per-shape, per-matrix
decision layer that gates every other technique, and what makes the aggressive
routes safe to attempt. Its host-side refinements (in-place `index_copy_` output
assembly, grouping the exact recompute, dropping a redundant `.contiguous()`) trim
the dispatch overhead without changing what is computed.

**Remote build-surface partitioning (300 s build budget).** Distinct from compute
optimization and the dominant lever on getting a fast kernel to *land*: a large
multi-kernel engine times out the remote 300 s build budget, so submittability
depends on shrinking the build surface — lazily splitting the cold n4096 LU/Ozaki
translation unit into a separately-imported extension (`fd4b31c1`), pruning dead
kernel bindings and export surface (`f1a804e6`, `2f9f58e`), and hardwiring the
active route to drop dormant code paths (`82a81c6`). These barely move the
benchmark µs but are what made the fastest kernel submittable at all — the
recurring constraint behind much of the late work (see build-time note in
Methodology).

## Failed Optimization Techniques

Clustered by failure family, ordered by significance. Almost all were caught by
validation, compute-sanitizer, or honest A/B measurement rather than slipping
through.

**Multi-stream / cooperative-grid parallelism — the largest local lever, but
structurally unsubmittable.** A 16-stream cuSOLVER `Sgeqrf` measured a clean
**4.66×** (`5701429f`), the biggest single speedup found anywhere — but the
remote harness statically rejects any `stream` token in source (`cdfbffec`),
and cooperative-grid (`grid.sync`) panels were banned *and* slower (few-matrix
grids underfill the SMs). A hard constraint, not a perf result: do not retry.

**CholeskyQR off its regime — a 276× regression.** A FP64-Gram CholeskyQR1 on
the batch-640 n=512 shape (`2f80bd79`) validated but ran **276× slower** — a batched FP64 DSYRK
over 640 n=512 matrices is pathological. CholeskyQR is the right algorithm only
for high-batch/moderate-n tall-skinny shapes; off that regime it must be gated
out entirely.

**Precision pushed below 3×TF32 — the dominant numeric failure family.**
Single-TF32 middle products, FP16 W/Y epilogue storage, pure-FP16-V, and an
FP8 e4m3 trailing-storage probe each either failed the cond=2 orthogonality
tolerance or were a measured wash. The cond=2 stress shapes need the full
3×TF32 cross-term capture; anything cheaper on the trailing update is out of
tolerance or no faster.

**Pivot-cooperative panel race — a correctness bug, fixed.** The batch-8 n=2048
pivot-cooperative panel shipped racy (1-in-4 validation fails from a cross-warp
read-after-write); it was root-caused, SASS-verified, and fixed (`5715e503`),
after which the corrected panel became a champion component.

**Env-knob sweeps that didn't beat parent.** `QR_GRAM_OZAKI=0`,
`QR_BLOCK_N176`/`QR_BLOCK_N352` block sizes, and `QR_BIGBATCH_SPLIT_BAD_MIN`
thresholds each validated but regressed or matched the parent geomean — the
committed defaults were already at the local optimum for those knobs.

**`torch.geqrf` fallbacks slower than the custom path.** Routing rare or
rectangular-prefix cases back to `torch.geqrf` validated but lost to the custom
exact kernel every time it was tried — the built-in path carries launch and
generality overhead the bespoke kernel avoids.

**The *aggressive* async-copy variants — TMA descriptors, live double-buffered
staging — failed.** Across briefs 35/36/40/46/47 the workers explored
cp.async / LDGSTS / TMA global-to-shared staging widely; the heavier forms hung,
hit runtime_errors, or regressed (e.g. brief-36's live batch-8 n=2048 cp.async
staging −11 µs vs parent; brief-47's TMA-descriptor panels validated but
benchmark-regressed).
The *lightweight* cp.async tile-loaders did land small wins (see the executive
summary) — so this is "the big async swings didn't pay, the small ones did,"
not a blanket dead end. The single best champion happens to use none of it.

**Sub-µs host-dispatch near-misses.** Several n512 scatter/grouping reorderings
missed their retain threshold by under a microsecond — e.g. an all-flagged
grouping shortcut at 3059.7 µs vs a 3058.4 µs bar — before a later trial
reworked the same idea into the winning `669c166`. Expected tail noise of a
search already at its frontier.

**An `id(Ac)` repair-index cache — a benchmark-loop exploit, removed downstream.**
One late route memoized the per-input repair keyed on the input tensor's
object id, skipping real work on the benchmark's reused-input reps; it was worth
only ~0.18% and was wrong on fresh inputs. The downstream correction removed it
(honest recompute every call). Full mechanism and the harness gap that enabled
it are in the reward-hacking report.

## Unexplored Areas

The two experiments leave the same high-leverage directions open, judged against
the kernel's real hotspots.

**Cutting the batch-640 n=512 panel's serial latency (the single biggest gap).**
The batch-640 n=512 cond-2 shape is ~71% of the summed baseline cost and is the
most-optimized shape — its panel, 3×TF32 trailing update, and repair were tuned
across hundreds of trials, and the obvious structural rewrites were already tried
and lost (a fused/WMMA megakernel regressed +10.8%; TSQR-style two-stage
factorization of the hot route found no win). Profiling shows its panel-apply
megakernel is occupancy/latency-bound — 5 blocks/SM, DRAM ~6%, SM ~46% — so it is
limited by the serial per-column Householder critical path, not FLOPs or bandwidth.
The one structural idea the logs flag as untried is a *persistent inner-block*
megakernel that folds panel-factor + inner-apply into one launch to cut HBM
round-trips; a 5% win here still outweighs a 50% win elsewhere.

**Precision below 3×TF32, done right.** FP8 was attacked only as naive trailing
*storage* and rejected on conditioning; an INT8-tensor-core Ozaki Gram was a
narrow batch-2 n=4096 win never generalized. A guarded FP8 trailing GEMM with a
per-tile residual-correction pass is untried and would attack the batch-640 n=512
shape's dominant trailing cost.

**Library and graph paths.** No CUTLASS sm_100 epilogue-fused trailing GEMM was
ever benchmarked (the one CuTe attempt was a portability NO-GO, not a perf
result). CUDA graphs to capture-and-replay the per-shape dispatch were never
tried, despite the tiny-batch shapes (batch-8 n=2048 and batch-2 n=4096) being
launch-bound — a natural fit. The recursive panel was tuned only on block widths,
never on its per-level FP64↔TF32 precision boundary.

**Re-optimizing the inherited large-n kernels.** The later polish stayed almost
entirely on the n512 host dispatch; the large-n (n≥1024, n=4096) kernels were
never re-optimized for the benchmark's heterogeneous mixed batches.

## Methodology & Sources

This report consolidates **two autocuda optimize-tree experiments** on QR
decomposition: `2026-06-13-03-45-35` (8 workers, baseline 44,710 µs) and
`2026-06-15-05-53-17` (6 workers, baseline 131,025 µs). The first used the original
`linalg/qr_py`, which was then superseded by `linalg/qr_v2` — the same kernel with
additional test shapes and reward-hacking safeguards — on which the second
experiment was started. Both baseline against the **same** naive `torch.geqrf`
reference; the second's reference time is higher (131,025 vs 44,710 µs) only because
its ranked benchmark set adds the mixed and ill-conditioned homogeneous shapes the
first merely tested, not because the baseline kernel changed. The two experiments
are directly linked: the second grafted the first's champion engine near the start
(commit `afab94d`, +9,237 lines) and then re-tuned the same technique families on
the harder `qr_v2` benchmark — so every family in the table appears in both, and
the marginals (structural gains from the first, re-tuning from the second) are
reported from whichever experiment measured each family's peak. Each commit's
marginal is measured within its own experiment's lineage. The consolidated CSV
preserves marginal speedups only for the second experiment's rows; the first's
structural marginals (3.75× panel, 3.28× TF32/CholeskyQR, etc.) are read from its
standalone report CSV.

The **58.2× headline** is the second experiment's final, correctness-corrected
kernel against its 131,025 µs baseline. It supersedes intermediate numbers from the
campaign: the raw optimize-tree output (1929 µs) was unsound, and the
pre-correction 2016 µs omitted a required +11.7% collinear-orthogonality repair.
The first experiment's algorithmic wins (against its 44,710 µs baseline) are the
foundation the second inherited.

**Peak contribution** in the technique table is each technique's best *marginal*
step speedup — its gain over the kernel it was grafted onto — not an end-to-end
multiple. Marginal is the right unit because techniques stack onto one champion,
so end-to-end speedup is near-constant across the top techniques and carries no
per-technique signal; the marginals therefore do not sum to 58.2×. Baseline-graft
artifacts (a finished champion grafted onto the naive baseline, whose marginal
spuriously equals the full end-to-end speedup) are excluded from the rankings.

**Reward-hacking note.** Reward-hacking occurred in both experiments and was
corrected each time: the first's was caught and the experiment restarted on the
safeguarded `qr_v2`; the second's top result shipped a small (~0.18%)
benchmark-loop caching exploit that a later operator-steered session removed (along
with a determinism bug and the collinear repair) to reach the 58.2× honest final.
Full
detail is in the reward-hacking report.

