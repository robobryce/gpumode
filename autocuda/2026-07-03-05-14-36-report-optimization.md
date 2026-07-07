# Optimization Techniques Report - linalg/eigh_py (`2026-07-03-05-14-36`)

## Executive Summary

This run searched broadly across 12 briefs, but the durable result was a stack of
small routing, reuse, and scheduling wins rather than a new eigensolver. The best
local metric moved from 25,511.859 us to 25,150.416 us (1.014x); several much
larger marginal steps below are recovery deltas after deliberately regressive
experiments, not comparable net gains over the strong inherited baseline.

| Metric | Value |
|---|---|
| Overall best speedup | 1.014x vs baseline |
| Baseline geomean | 25,511.859 us |
| Best geomean | 25,150.416 us (`a8747145e`) |
| Distinct techniques identified | 7 landed/candidate clusters + 7 failed families |
| Succeeded / attempted trial rows | 238 / 255 |
| Search breadth | 12 briefs, 3 concurrent workers |
| Most prevalent current-run mechanism | Structure-aware routing/gating in 4 of 12 briefs |
| Final leaderboard evidence | Accepted 39/39; 25,245.619 us public score |

**Key takeaways:**

- The final stack combines the BF16 rank-32 trace sketch, PSD-only `A^2*Omega`
  reuse, three-way n=2048 sign divide-and-conquer, and B200 programmatic
  dependent launch (PDL); no one component accounts for the full 1.014x.
- Conservative classification mattered more than aggressive arithmetic. The
  largest reported marginal step, 1.385x at `817a6fbf`, came from restricting a
  bad broad Krylov-reuse route to gate-clean structure bands.
- PDL produced the clearest final-shape win: the n=512 mixed path fell from
  about 104.1 ms on the combined parent to 98.1 ms in the selected lineage.
- Radical alternatives were explored deeply and rejected: FP8 sign, persistent
  full-size Householder, block Jacobi, full-spectrum block Lanczos, and
  Chebyshev spectral windows all lost to the residual-gated inherited portfolio.
- The companion reward-hacking audit finds the run clean; all remotely selected
  candidates passed the invariant checker and the final candidate was accepted
  by the leaderboard after reseeding.

## Most Impactful Optimization Techniques

| Technique | Peak contribution | Prevalence | Representative commits |
|---|---|---|---|
| Conservative structure gates and early fallback | 1.39x | 4 of 12 briefs, 42 route/gate attempts | `817a6fbf`, `f26242b0`, `ccfad948` |
| Fused compact-WY bisection/twisted leaf stages | 1.26x | 1 of 12 briefs, 15 attempts | `c839dd3e`, `04b1ab9b`, `0179c68b` |
| Three-way matrix-sign spectral divide-and-conquer | 1.09x | 1 worker plus combine brief, 32 attempts | `6102542b`, `7e5ecf8f`, `1990761f` |
| O(n) diagonal-dispersion structure routing | 1.07x | 1 of 12 briefs, 5 focused attempts | `c629f83b`, `3fb87a35` |
| PDL-overlapped heterogeneous solver chains | 1.05x | 2 of 12 briefs, 16 direct scheduling attempts | `86fd967f`, `eec985dc`, `3c275b0b` |
| TF32 backtransform plus FP32 repair/gating | 1.04x | 1 of 12 briefs, 5 direct precision attempts | `1dc40e7c`, `69b7840f`, `98ea3ef8` |
| Orthogonal BF16 trace sketch and fused summary | 1.02x | 1 of 12 briefs, 19 sketch/summary attempts | `8bfe2cd6`, `f46f7482`, `d6b73a53` |

The peak-contribution column is a marginal measurement against the prior kept
trial, so it is a ranking signal rather than an additive decomposition. In this
run especially, the top two values include recovery from broken or deliberately
over-broad predecessors; their end-to-end contribution to the final stack is
much smaller.

**Conservative structure gates and early fallback.** Runtime statistics such as
participation ratio, diagonal dispersion, and homogeneity decide whether a
matrix enters a specialized low-rank/persistent solver or goes directly to
cuSOLVER. `817a6fbf` limited BF16 Krylov reuse to stable dense bands,
`f26242b0` retained only the PSD k256 `A^2*Omega` seed, and `ccfad948` bypassed
the slow persistent n=1024 path for heterogeneous batches. The 1.39x peak is a
recovery from an over-broad reuse trial; the durable value is avoiding a custom
solve followed by an expensive residual-gated fallback.

**Fused compact-WY bisection/twisted leaf stages.** Brief 10 removed scratch
traffic and synchronization inside `mega_eigh_med_split_k` and
`mega_eigh_clust_split_k`: eigenvalue owners carry bisection results directly
into twisted-factor recurrences, Gram work is upper triangular, and compact-T
construction uses a warp-local recurrence. `0179c68b` also separated shared
Gram/T tiles. The 1.26x peak at `c839dd3e` largely restored a disabled extension;
the branch ended at 25,204.208 us, slightly behind its already strong parent,
so this remains a promising local kernel improvement rather than a global win.

**Three-way matrix-sign spectral divide-and-conquer.** Brief 4 used stochastic
Lanczos/Ritz quantiles, concurrent shifted sign filters, CQR invariant
subspaces, and three reduced megakernel leaves for dense n=2048. Tuning sign
depth and shrinking boundary slack to one vector made the custom path pass its
FP32 residual gate at roughly 86-90 ms. `1990761f` was combined with the PSD
sketch reuse in `0ed0e959`; it stands alone algorithmically and is one of the
few radical paths retained in the final lineage.

**O(n) diagonal-dispersion structure routing.** `c629f83b` first replaced a
full `A^2` probe with row-energy/trace reductions; `3fb87a35` reduced the hot
n=1024 mixed decision to diagonal coefficient-of-variation statistics and a
threshold that separates dense from heterogeneous batches. This removes an
O(n^3) classification product before a direct cuSOLVER call. Its peak 1.07x is
mostly recovery from a misrouting threshold, while the confirmed end-to-end
gain over the run baseline was about 0.4%.

**PDL-overlapped heterogeneous solver chains.** Briefs 8 and 11 use SM100
programmatic dependent launch on the default queue: a long high-SMEM cluster
grid launches first, then k608/k384 or dense/PSD reduced grids become eligible
as its CTAs are resident. One fused compaction/scatter preserves bucket order.
The n=512 chain delivered the selected `a8747145e`; `3c275b0b` extended the
mechanism to n=1024 and halved mixed1024 versus its prior routed trial. Multiple
cluster dependents and split primary grids regressed, so the winning form is one
cluster primary followed by a short ordered chain.

**Orthogonal BF16 trace sketch and fused summary.** The final classifier uses a
cached orthogonal rank-32 probe, BF16-resident `A*Omega` and `A^2*Omega`, FP32
norm accumulation, and a Triton summary kernel that emits all route populations
with one host synchronization. This replaces a full `A^2` materialization while
keeping mixed-route decisions stable. It was discovered in one long brief, but
its repeated 25.34-25.27 ms measurements make the approximately 1% gain the
run's most credible isolated improvement.

## Failed Optimization Techniques

**FP8 and lower-bit factor-producing paths.** E4M3 matrix-sign iterations
rotated eigenspaces enough to miss the eigen residual gate, while an FP8
complement Gram needed TF32 cleanup and still raised the geomean to 26,132 us.
Routing-only FP8 spent its gain on conversion and normalization. The two first
backends also hit an SM100 launch incompatibility or a banned queue accessor
(`f4f499d9`, `cb49bf0e`). BF16 succeeded only for the narrow classifier sketch;
no NVFP4 kernel reached execution.

**Persistent full-size n=1024 Householder/TMA.** Brief 2 built the requested
16-column V/W factorizer with CCCL TMA double buffering and three-term TF32
rank-2k updates (`ad497fc0`, `2db6a6e6`). It was valid but mixed1024 remained
about 219 ms versus roughly 106 ms for direct cuSOLVER, with low occupancy and
barrier stalls. `ccfad948` recovered the aggregate only by routing heterogeneous
batches away from the custom path. C=8/C=12 partitions, CUDA graphs, padded MMA
GEMV, and mirror publication either regressed, failed policy/build, or hung.

**Block-cyclic Jacobi and persistent Jacobi.** Twenty trials covered 16/32/64
blocks, exact and custom local pivots, WMMA two-sided updates, dynamic pivot
ordering, FP32 cleanup, and a one-CTA persistent round. Projection-energy
assignment fixed cycling, but FP16 update error plateaued near the residual
gate and hundreds of matrices fell back. The best custom result was still about
30.1 ms, roughly 18% slower than baseline; the persistent form was slower yet.

**Full-spectrum block Lanczos and band reduction.** Twenty-six trials tried
32/64/128-column bases, BF16/TF32 products, selective reorthogonalization,
cluster/sign/cuSOLVER projected solves, exact band Givens reduction, compressed
rotation storage, and TF32 polar repair. Full-history FP32 reorthogonalization
and the projected eigensolve dominated, while cheaper variants lost basis
quality and triggered fallback. No result beat 26.9 ms.

**Chebyshev spectral windows without leaf solves.** Brief 9 replaced exact
reduced leaves with Rayleigh diagonals, orthogonal iteration, or 24/48
Jackson-Chebyshev windows. Even exact eigenvalue-count and spectral-radius
oracles did not fix the selected union's eigen residual. All benchmark matrices
fell back after 40-80 ms of filter work, producing 27.2-28.8 ms geomeans.

**Fused classifier and scalar CQR prototypes.** A one-CTA BF16 WMMA classifier
and a persistent scalar Cholesky/right-solve kernel underutilized SM100; the
former was 0.5-0.8 ms slower on n=512 routes and the latter roughly doubled
dense512. A Triton tcgen05 classifier and inverse-GEMM CQR recovered most of the
loss but remained neutral, so brief 5 restored the compact PSD-reuse source at
`32d88ef1`.

**Policy-blocked concurrency and unstable queues.** Explicit CUDA streams were
rejected by the submission policy in briefs 0, 2, and 8. A graph without an
explicit stream executed but remained slower, and the capped persistent leaf
queue in brief 10 did not load. These are implementation/policy failures rather
than evidence that overlap itself is useless; PDL was the admissible successful
alternative.

## Unexplored Areas

**Native NVFP4 with calibrated refinement.** The run reached FP8 and BF16 but
never compiled or benchmarked an NVFP4/tcgen05 factor-producing kernel. A future
attempt needs per-block scaling, a narrow use such as projector membership or a
well-conditioned reduced update, and an explicit FP32 reorthogonalization plan;
raw low-bit sign iterations were already refuted.

**TMEM-native persistent leaf scheduling.** The profile in `environment.md`
shows the inherited cuSOLVER path is SIMT, uses no tensor cores, and is
grid-starved at large n. TMA staging was implemented, but no production kernel
combined tcgen05 TMEM accumulators, a stable global work queue, and variable
leaf widths; the only queue attempt failed to load. This is the remaining
Blackwell-specific megakernel direction rather than another Python launch
rearrangement.

**Production-batch robustness as a design objective.** Tests use b16/b4 for
n512/n1024 while ranked shapes use b640/b60. Several kernels passed all 39 tests
then failed, hung, or fell back at scale. Dedicated route-activation tests at
production occupancy and multiple seed offsets would let future workers tune
the actual custom path without using a full benchmark as the first stress test.

**Small and medium structural kernels.** The run concentrated on n512-n2048.
Brief 10 touched n176/n352 only incidentally and one warp-specialized variant
faulted on n176. Stable warp-specialized diagonal/banded kernels for these
shapes remain shallowly explored and may be more tractable than another dense
full-spectrum replacement.

**A fundamentally different route to the leaderboard frontier.** The accepted
final score was 25,245.619 us while first place was 15,252.461 us, a 65.5% gap.
The 1.4% local improvement and failure of five broad algorithmic replacements
show that more threshold tuning on the current portfolio is unlikely to close
that gap; the next run needs a new end-to-end batched solver or a strong new
shape-class decomposition, not another sub-percent scheduling graft.
