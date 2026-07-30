# Epoch 3 launch plan

Eligible after `2026-07-30T09:19:11Z`, when `autocuda status` reports
`progress.new_epoch=true`. Use the run policy verbatim:

`min-macro=.25 min-micro=0 min-combine=0 min-simplify=0`

Toolchain evidence checked during cooldown: the authoritative
`/home/shadeform/gpumode/.venv/bin/python` is Python 3.13 with
PyTorch `2.12.0+cu130`; it exposes `torch.float8_e4m3fn` and
`torch.float8_e5m2`. CUTLASS DSL `4.5.2` maps both FP8 formats, and
CUDA Tile `1.4.0` plus nvMath `0.9.0` import successfully on the local B200.
PyTorch also vendors a production-matched SM100 block-scaled GEMM skeleton at
`.venv/lib/python3.13/site-packages/torch/_inductor/kernel/vendored_templates/cutedsl/dense_blockscaled_gemm_persistent.py`.
It demonstrates TMA multicast, one- and two-CTA `tcgen05` block-scaled MMA,
TMEM scale-factor/accumulator handling, and persistent tile scheduling; reuse
that supported pipeline architecture rather than inventing descriptors.
In the current leader, all CuTe trailing-product integration is centralized in
`submission.py` around `_TriangularGemm`, `_BatchedTriangularGemm`,
`_G2048TriangularGemm`, `_left_looking_operation`, and their compiled host
wrappers (roughly lines 9070-9620). Add shape-selective block-scaled operators
at that boundary; do not rewrite unrelated factor/solve state merely to test
FP8 products.

1. Parent `52427ff70e52ec6262c50b0bdbdbf9531ac09e42`, kind `macro`, with
   `--new-epoch`:

   Build a genuinely different shape-dispatched Cholesky from the baseline
   around compensated FP8/BF16 tensor-core products: keep FP32 POTRF/TRSM on
   diagonal frontiers, use one block-scaled FP8 product for far trailing tiles,
   add bounded residual terms only on next-diagonal/frontier tiles, and use
   communication-avoiding recursive supernodes with selective FP32 repair
   across the full benchmark.

2. Parent `fd956a9d484692c4dc7a4ec20d94bc371d50e7ed`, kind `macro`:

   Replace the leader's low-batch library islands at batch16/n512,
   batch4/n1024, batch2/n2048, and batch1/n4096 with one shape-scoped recursive
   portfolio. Reintegrate the current-run staggered copy/factor relay, fused
   batch4 first wave, and batch8-only batched graph while preserving the
   leader's neighboring dispatches and ordered fresh-input state.

3. Parent `0b18f368ac286b8646e796cd0736552c48e11a41`, kind `macro`:

   Rebuild both n512 regimes as coarse two-CTA SM100 clusters: one factor/panel
   owner and one update owner share diagonal and panel tiles through DSM or
   cluster multicast, retain correction accumulators in TMEM, and keep compact
   LD32 global sidecars without row-wise expansion or fine-grained global task
   queues.

4. Parent `fd956a9d484692c4dc7a4ec20d94bc371d50e7ed`, kind `macro`:

   Replace the low-batch n2048/n4096 graph frontier with correctness-complete
   next-diagonal dependency counters. Count every Schur contribution to the
   next diagonal tile, publish a diagonal-ready generation before off-diagonal
   completion, factor it immediately, and use 128x128 tcgen05 updates with
   per-matrix barriers rather than whole-grid synchronization.

5. Parent `96be23f43abd103dc13f7ebd3229d1715b939be8`, kind `macro`:

   Replace large-shape FP16-history trailing products with compensated FP8 or
   BF16 two/three-term lower-triangular products while retaining FP32
   POTRF/TRSM. Specialize n8192/n16384/n32768 schedules, keep the 4096
   supernode interface, use a single block-scaled FP8 product on far tiles,
   and spend extra decomposition terms only on the next diagonal and frontier
   so the dominant products retain Blackwell's higher-throughput mode.
