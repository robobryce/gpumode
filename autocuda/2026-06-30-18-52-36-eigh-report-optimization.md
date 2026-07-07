# Optimization Techniques Report — linalg/eigh_py (`2026-06-30-18-52-36-eigh`)

## Executive Summary

This run found a **portfolio of structural wins**, not one big lever: it moved
six of the run's ~eight cuSOLVER-floor dense shapes onto faster custom paths,
each a distinct eigensolver algorithm chosen by a runtime structure probe. The
search was **diverse** — it attacked five different bottleneck angles (small-n
launch overhead, medium-n back-transform, low-rank subspace projection,
clustered/2-level spectra, and gapless-dense divide-and-conquer) — and the
biggest gains came from *replacing the algorithm* for a shape class, then
compounding with tensor-core precision grafts. All techniques stack on one
runtime router that defaults to cuSOLVER, so the trajectory is a smooth
47,824 → 32,182 µs (1.49×). (The companion reward-hacking report finds the run
**clean** — every win is genuine and residual-gated.)

| Metric | Value |
|---|---|
| Overall best speedup | 1.49× vs baseline |
| Baseline geomean | 47,824 µs |
| Best geomean (commit `e23884fa48`) | 32,182 µs |
| Distinct techniques identified | 8 landed + 4 failed families |
| Kept / attempted trial rows | 394 / 398 |
| Workers (parallel briefs) | 3 workers, 49 briefs |
| Most impactful single technique | Split batched-GEMM back-transform (1.45× marginal) |
| Leaderboard gap to #1 | ~19% (best 32,182 µs vs #1 27,023 µs) |

**Key takeaways:**

- **The single biggest marginal lever is the split back-transform** — moving the
  medium-megakernel's Householder back-transform out to a torch-level batched
  compact-WY GEMM, **1.45×** on the kernel it was grafted onto (`b778e8e6`),
  compounding across shapes 2/3/8/12.
- **3×TF32 "Ozaki" tensor-core GEMMs are the run's most *pervasive* technique** —
  the same hi+lo FP32-accurate split was independently applied to the low-rank
  Gram (1.32×, `e3ef14b4`), the orthogonality gate, and the two-level path;
  it's the workhorse that made low-precision safe under the FP64 checker.
- **Algorithm-selection-by-structure is the run's spine.** Five separate solvers
  (fused megakernel, randomized low-rank, two-level projector, thread-block
  cluster kernel, matrix-sign D&C) each win one spectrum class, dispatched by
  cheap participation-ratio / `A²≈I` probes — no single kernel wins everything.
- **The largest shape (lapack-dense-even512, ~208 ms) fell last and hardest**, to
  a matrix-sign spectral divide-and-conquer (`23fb18c8`), after every simpler
  dense approach was refuted — the run's clearest "algorithm departure" win.
- **Combine briefs did real work**: grafting disjoint-shape wins onto one lineage
  (`391107f7`, `85a450df`, `6c2cf8cd`) is how the portfolio stacked to 1.49×.

## Most Impactful Optimization Techniques

| Technique | Peak contribution | Prevalence | Representative commits |
|---|---|---|---|
| Split back-transform → torch batched compact-WY GEMM | 1.45x | 1 worker, brief-22/24 lineage, ~8 attempts | `b778e8e6`, `391107f7` |
| 3×TF32 (Ozaki hi+lo) tensor-core GEMMs | 1.32x | 3 briefs (16/44/45), ~25 attempts | `e3ef14b4`, `3b5806a7`, `d6ce9b52` |
| Randomized low-rank dominant-subspace solve | 1.15x | 4 briefs (7/11/16/26), ~40 attempts | `ce4c5eb9`, `966fb4e5`, `7eb7a5d1` |
| Matrix-sign spectral divide-and-conquer (dense-even) | 1.13x | brief-0 (refuted) → brief-43 (won), ~15 attempts | `23fb18c8`, `4f7abed0`, `e23884fa48` |
| Thread-block cluster kernel (k>448 / full-n=512) | 1.15x | 3 briefs (14/21/35/42), ~20 attempts | `a79df8e0`, `f8e44f9d`, `9b54ac47` |
| Block-diagonal CQR + Newton-Schulz (two-level shape-9) | 1.01x* | brief-18/20, ~11 attempts | `cba54eb2`, `4c1e8876` |
| Mixed-batch structural peel (dense/psd subsets) | 1.02x | 3 briefs (28/33/37), ~10 attempts | `3adba24b`, `5c66a54b`, `c0dd1324` |
| Fused megakernel + per-matrix gate trim | 1.09x | briefs 3/11/12/13/27, ~50 attempts | `9d38abc2`, `a9bf63d9`, `8d30bf97` |

\* Peak marginal reads ~1.0× because block-CQR/NS *replaced* an already-winning
joint version and its gain (54→46 ms on shape 9) is diluted in the 13-shape
geomean; its value is robustness (nbad=0 across reseeds), not raw step size.

**Split back-transform → torch batched GEMM** is the highest-marginal single
change. The medium-n fused megakernel (`mega_eigh_med`) formed eigenvectors by
applying accumulated Householder reflectors in-kernel in FP32-SIMT — ~70% of the
kernel, zero tensor cores. `b778e8e6` has the kernel *return* the tridiagonal
eigenvectors Z plus the reflector panel V and block-T, then forms
Q = (I − V T Vᵀ)Z as one batched compact-WY GEMM at the torch level
(`_mega_med_backtransform` / `_bt_bmm`). This sidesteps the in-kernel hand-WMMA
"cast-bound" wall entirely — the GEMM is compute-bound and the Z/V/T round-trip
amortizes. Because the med kernel serves shape 2 directly *and* is the low-rank
inner solve for shapes 3/8/12, one change compounds across four shapes
(shape2 1.51×, shape3 1.20×, shape8 1.22×). Single-worker, but the manager
immediately combined it (`391107f7`), making it a load-bearing part of the
final stack.

**3×TF32 Ozaki-split GEMMs** are the run's most-rediscovered mechanism and the
key that unlocked low precision under a tight checker. A plain TF32 GEMM's
~3e-4/op error inflates the measured orthogonality residual just above the gate
(→ spurious mass fallback), so the workers built `_matmul_3xtf32` /
`_gram_3xtf32` — a hi+lo bit-split that runs three TF32 tensor-core GEMMs to
reach ~6e-6 (≈FP32) accuracy at ~1.6× the FP32-SIMT rate. It was applied to the
low-rank orthogonality-gate GEMM (`3b5806a7`), the CholeskyQR2 dominant Gram
(`e3ef14b4`, 1.32× scoped to n≥1024 and n=512/k≥384), and a symmetric-aware
2-GEMM variant (`d6ce9b52`). Three independent briefs converged on it — a strong
signal it is a real, reusable lever on this checker's loose invariant gates.

**Randomized low-rank dominant-subspace solve** is the run's most *prevalent*
algorithm win (shapes 3/4/8/10/12). Instead of cuSOLVER's serial per-matrix
syevd, it does batched-GEMM subspace iteration for the k dominant eigenpairs +
CholeskyQR2 + a lumped Rayleigh tail, with k chosen by the participation-ratio
probe `PR = ‖A‖_F⁴/‖A²‖_F²` (`_lowrank_route_k`). brief-7 established the per-n
PR-band table and the cheap A@V-reusing gate; brief-11 made its inner k×k solve
cheap by routing it through the bare megakernel; brief-16/26 tuned the outer
CQR2 precision and k downward. It wins whenever the spectrum is concentrated
enough that rank-k + tail clears the gate — the dominant HBM-bound bottleneck
for the concentrated dense/rankdef shapes.

**Matrix-sign spectral divide-and-conquer** is the run's marquee algorithm
departure and the one that finally cracked the biggest term (lapack-dense-even
n=512, ~208 ms). This is a gapless, evenly-spaced signed spectrum — not low-rank
(PR ~284), not 2-level — so every projection/deflation approach was refuted
(brief-0 first tried it at 82 ms, worse than baseline). brief-43 made it win:
Newton-Schulz iterate sign(A) on tensor cores, build ± invariant-subspace bases
via oversized fixed-width (K=300) CholeskyQR probes, solve the two reduced K×K
blocks in one stacked megakernel launch, and select the true n eigenpairs by a
projector-membership rank-select. Its cost is *spectrum-independent* and
tensor-core-bound, so it beats cuSOLVER's deflation-starved d&c on the flat
spectrum (208→109 ms). Subsequent trials fused the NS via `baddbmm`, cached the
Ω probes, and split the block-eigh stage-1 (`e23884fa48`, the final best).

**Thread-block cluster kernel** extends the fused megakernel past the 228 KB
SMEM cliff by distributing the packed-FP16 triangle across 2–3 CTAs' distributed
shared memory (`cudaLaunchKernelEx`, `map_shared_rank`, `cg::cluster`). brief-14
built the infra; brief-35 de-raced the cross-CTA tridiag reduction
(`a79df8e0`, +8.8%) and fused the symv (`f8e44f9d`) to win the k=608/768
low-rank *inner* blocks for shapes 4/10 that overflow a single CTA; brief-42
then routed full-rank n=512 through it at C=3 (`9b54ac47`, 1.15×), excluding
2-level batches via the cheap `_twolevel_mask` probe. It's the one technique
that required genuine sm_100 cluster programming.

## Failed Optimization Techniques

**Custom batched dense→band→tridiag reduction (two-stage SBR).** The most-retried
dead end: briefs 1, 4, 6, 9 (and the Triton SBR of brief-15) all built batched
Householder/bulge-chasing reductions to replace cuSOLVER's syevd on the large
dense shapes, and all lost. Root cause is structural, not a bug: at n=512 b640
cuSOLVER's per-matrix syevd already fills the GPU (640 concurrent matrices), and
a one-CTA-per-matrix reduction is latency-bound at small batch (n=2048 b8 =
8 CTAs). The band→tridiag stays O(n·bandwidth) and remains ~46% of the kernel,
so the reduction cannot match cuSOLVER's fused multi-CTA sytrd
(commits `ce73b670`, `2f64c26b` errored/regressed; brief-4/9 landed non-regressing
empty routes). This tells the next run: don't rebuild the full reduction —
attack via subspace/structure instead.

**FP8 / low-bit storage for the megakernel.** brief-17 tried FP8-e4m3 (and a
hybrid FP16-band + FP8-far layout) to halve the packed-triangle SMEM and double
occupancy. It was refuted twice over: the 2-CTA ALU/barrier contention made the
kernel slower even when it fit, and FP8's 3-bit mantissa forced ~100% orth-gate
fallback (`58637`/`65919` µs, worse than baseline). Storage-only FP8 with
dequant didn't recover it. FP8 compute for the reduction was independently dead
in brief-3.

**cuSOLVER-Jacobi parameter lever.** brief-7 probed calling syevj/syevjBatched
directly with loosened tol + capped sweeps (exploiting the 1.2% gate). Measured
negative: syevj at n=512 b640 (158 ms) did not beat syevd, and syevjBatched is
not genuinely batched at k≤448. A clean vendor-parameter idea that the hardware
simply didn't reward here.

**Whole-batch / gather-split routing on mixed batches.** brief-10 (and the
mixed1024 probe in brief-39) tried classifying a heterogeneous mixed batch and
running two solvers. It lost because cuSOLVER stays above its 64-matrix knee so
its marginal cost drops only slowly, and the classifier + second-call overhead
exceeded the thin margin. The *later* mixed512 peel (brief-28) succeeded only by
peeling a razor-tight dense PR window with the now-faster split-mega inner solve;
mixed1024 was cleanly refuted (too few dense matrices, tiny n=1024 marginal).
The failed composition here is the strided/sampled orthogonality gate (brief-42
t12, `25051f15`) — it broke validation on the repeated-eigenvalue test shape and
was reverted the same trial.

## Unexplored Areas

**Shape 5 (n=2048 b8, ~186 ms) — the second-largest term, still on the floor.**
No landed technique targets it; brief-47 is only now (in-flight) trying the
matrix-sign D&C recursed to n=2048. This shape is where cuSOLVER *most*
under-fills the GPU (8 matrices, ~8 of 148 SMs), so a multi-CTA-cooperative
reduction or a deeper sign-split is the highest-EV unexplored lever — the
brief-9 cooperative-bulge attempt stalled at latency, but a tensor-core sign
recursion (spectrum-independent, fills the GPU) has not been carried to
completion.

**Shape 7 (mixed1024 b60, ~108 ms) — mapped as irreducible, worth re-probing.**
brief-39 refuted per-subset peeling because the n=1024 cuSOLVER marginal
(~1.4 ms/mat) is too small to beat the ~15 ms low-rank floor at b60. But this
was under the current inner-solve speed; if shape-5's multi-CTA work lands, the
n=1024 cluster inner solve could shift that break-even. Currently the only big
shape with zero winning route.

**CUDA graphs / launch-overhead elimination.** brief-23 found the n=32 path is
occupancy-bound compute (not host dispatch as hypothesized) and — critically —
that CUDA-graph capture is **banned** (the harness rejects any `stream`
substring, and capture needs a stream). So the entire graph-fusion angle for the
small shapes is closed by rule, not by measurement; the ~40–50 µs launch gap on
n=32 is unreclaimable without a kernel-launch change. Worth noting so no future
brief re-attempts it.

**CUTLASS grouped/batched FP8 GEMM.** Every low-precision win used torch bmm +
the 3×TF32 Ozaki trick because `torch._scaled_mm` is 2D-only and `torch.bmm` has
no FP8 path. A custom CUTLASS grouped FP8 GEMM (noted but never built in brief-43)
is the untried path to genuine FP8 throughput on the batched reduced blocks —
the one library lever the run identified but never benchmarked.

**Closing the 19% gap to #1.** The leader (27,023 µs) sits ~19% below this run's
best via a fundamentally different approach — the leader `file_name` hints
(`submission_eigh_n32_parallel`, `triangular_diagonal_fast_path`) point at
aggressive small-n and structural fast-paths. shape 0 (n=32) remains essentially
unwon here (CUDA-graph route blocked); a Triton/warp-parallel n=32 kernel and
deeper diagonal/banded fast-paths are the frontier the current cuSOLVER-floored
line cannot reach by micro-optimization alone.
