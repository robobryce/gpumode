# Epoch 3 worker launch prompts

Use a fresh `optimize-tree-worker` for each prompt after briefs 122-126 are
present in the manager log. Every worker must run `autocuda:explore-brief` and
obey the run's locking, validation, logging, and no-outside-run rules.

## Brief 122

`brief-id=122 tag=2026-07-19-01-44-06-cholesky-resumed-local data-dir=/home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local`

Start by reading same-run Briefs 117, 118, and 121 plus the installed SM100
block-scaled template. Preserve FP32 frontier factorization. Use MXF8 E4M3 with
E8M0 scales and FP32 accumulation only where one-term error is bounded; retain
selective correction near every diagonal. Do not spend the brief merely
reintegrating the existing mixed-precision baseline. For in-place Schur
subtraction, merge the block-scaled mainloop with the installed
`dense_gemm_alpha_beta_persistent.py` C-load epilogue (one epilogue-load warp,
TMA C pipeline, alpha=-1, beta=1) rather than materializing a full scratch GEMM.
Fuse blockwise max/E8M0-scale generation and E4M3 conversion into factor/panel
publication; do not add a separate whole-panel quantization pass unless its
measured cost is amortized by the target shape.
Compile one shape-scoped block-scaled operation lazily for the first target;
only add more `(extent, panel, region)` variants after that path validates and
benchmarks. The parent already imports nine CuTe operations.
Minimal template merge: keep the block-scaled mainloop unchanged; add the
alpha/beta kernel's epilogue-load warp, `PipelineTmaAsync` C loader,
load/store shared partitions, and `D = -acc + C` vector expression. Avoid
rewriting scheduler or scale-factor pipelines in the first trial.
Host capability is proven: the installed `blockwise_gemm.py` passed a
512x512x512 FP8-E4M3, FP32-accumulating 2-CTA `(2,1)` cluster run on this B200
at 2026-07-30T08:31Z, including compilation and reference check.

## Brief 123

`brief-id=123 tag=2026-07-19-01-44-06-cholesky-resumed-local data-dir=/home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local`

Read same-run Brief 100 completely before editing and use its measured donors,
especially the staggered relay, fused batch4 first wave, and batch8-only graph.
The exact donors are `06cbd50b0b01e81997be858765235eceb4cd55c2`
(staggered relay), `005ddb3b756faf91a8a6d2f7a0b5348e49f5bb7d`
(~663 us batch4/n1024 fused route), and
`fad9323cb9b558bfd00397d629d65bff3c4036d4` (measured batch8 graph).
Do not route batch60/n1024 through the slow unified TF32 frontier from Brief
117. Preserve the integrated leader's neighboring dispatches and fresh-input
ordering while combining only the four low-batch library islands.

## Brief 124

`brief-id=124 tag=2026-07-19-01-44-06-cholesky-resumed-local data-dir=/home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local`

Read same-run Briefs 107, 114, and 119 before editing. Keep compact LD32 global
sidecars but use bank-safe shared/TMEM layouts. Respect the generic loader's
128-producer bound discovered in Brief 119. Pursue an actual two-CTA cluster
publication mechanism rather than another host-launched pair of independent
kernels or a fine-grained global queue. Implement the clustered region in CuTe
DSL: the installed examples expose cluster launch, block rank, TMA multicast
masks, and cluster-aware pipelines. Do not emulate DSM/publication with global
atomic polling inside a Numba persistent kernel.
The installed CuTe `blockwise_gemm.py` passed an actual 2-CTA `(2,1)` cluster
launch with FP8 inputs, tcgen05, TMA, FP32 accumulation, and reference checking
on this B200 at 2026-07-30T08:31Z.
Use `0b18f368ac286b8646e796cd0736552c48e11a41` as the measured donor:
LD32 global sidecars expanded directly into LD40 shared operands. Brief 114's
corrected TMA line (`8dd32949…` through `7cadb7e…`) required LD40 for every
consumer and still measured ~1.93-2.05 ms at batch640, so avoid a separate
packed-stage/expansion phase.

## Brief 125

`brief-id=125 tag=2026-07-19-01-44-06-cholesky-resumed-local data-dir=/home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local`

Read same-run Brief 120 completely before editing. Its task-zero lookahead was
incorrect because the next diagonal requires multiple Schur contributions.
Count the exact full contribution set for each next-diagonal tile before
publishing readiness, separately from off-diagonal completion. Also read Brief
115 to avoid incomplete-stage DAGs and owner/consumer residency deadlocks.
Prefer two independent CuTe cluster pipeline/barrier states—diagonal-ready and
stage-complete—over one global generation counter. The installed DSL supports
multiple PipelineAsync/TMA/UMMA states inside a cluster.
The validated epoch foundation is
`f8aace9cf969ec6e846893f391c137964c0d1013`. The incomplete lookahead variants
`f6c8b438…`, `be59eb6f…`, and `f7a905d9…` all improved b2/n4096 but failed
b8/n2048 because assigning one task/worker did not wait for every contribution
to the next diagonal. Preserve the foundation and replace only that readiness
condition first.

## Brief 126

`brief-id=126 tag=2026-07-19-01-44-06-cholesky-resumed-local data-dir=/home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local`

Read same-run Briefs 116 and 121 before editing. Reuse Brief 121's corrected
standard scheduler for rectangular far-to-diagonal regions; triangular
coordinate mapping caused NaN/Inf there. Integrate MXF8/BF16 products at the
existing large trailing-product boundary, with FP32 output accumulation and
extra terms only for next-diagonal/frontier regions. Adapt the installed
alpha/beta persistent GEMM C-load epilogue into the block-scaled template so
far updates subtract directly into the FP32 trailing matrix without a full
scratch output. Generate E8M0 scales and E4M3 values while the solved 4096
panel is already being compacted/published; a separate full-panel quantizer is
unlikely to amortize below the largest shapes.
The checker permits relative reconstruction error of about 1.95%, 3.91%, and
7.81% at n8192, n16384, and n32768 respectively, versus below 0.25% through
n1024. Use that asymmetry: test one-term MXF8 first on far large-shape regions,
but keep compensated/FP32 work on every next-diagonal region.
Compile fixed-shape operations lazily and begin with n32768 far-trailing work;
do not instantiate all large extents before one block-scaled path is proven.
Use `d1560d514ee561710a4a4ab260f646c4ae8a9bee` for the validated three-region
scheduler and `6a7497d80262cedeee010475080c262da84f78a2` as the monolithic compact-FP16
comparison. Keep the region partition fixed while changing product precision
so the first measurement isolates MXF8/BF16 arithmetic and quantization cost.
For the first kernel, keep the block-scaled scheduler/mainloop unchanged and
port only the alpha/beta epilogue-load warp, C pipeline, split load/store
partitions, and `D = -acc + C` expression.
