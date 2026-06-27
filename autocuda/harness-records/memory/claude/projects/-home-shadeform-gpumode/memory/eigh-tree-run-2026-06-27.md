---
name: eigh-tree-run-2026-06-27
description: "Active optimize-tree run on linalg/eigh_py started 2026-06-27, TAG 2026-06-27-13-12-42, baseline/leaderboard facts and the 3 seed-brief directions"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3858be5a-d08f-43a7-9d97-6107105fcb93
---

Optimize-tree run on **linalg/eigh_py** (batched real symmetric eigh, `custom_kernel(A)->(Q,L)`), started 2026-06-27. `TAG=2026-06-27-13-12-42`, 3 workers, `DATA_DIR=/home/shadeform/gpumode/autocuda`. Host = 1×B200/CUDA13.3(nvcc)/torch2.12+cu130(Triton 3.7)/30CPU. Baseline commit `424ba1894` (main tip), local geomean **≈56,255 µs**, leaderboard submission ACCEPTED.

**Leaderboard gap (read 2026-06-27):** baseline lands ~#12/13. Frontier: #1 `az` best.py **33,954 µs**, #2 34,982, #3 35,579, #4 `triton_diagonal_fast_path.py` 38,305, #5 40,616 — a ~1.66× gap reached by a fundamentally different (batched / tensor-core) approach, not micro-tuning. Everyone ≥53k is stock eigh.

**Baseline bottleneck (profiled):** torch.linalg.eigh = serial cuSOLVER syevd per matrix — sytrd4_cta tridiag 22%, gemvx/symv/gemv2 matrix-VECTOR panel updates ~42%, laed D&C ~20%, **zero tensor-core**. Big headroom for any genuinely batched / tensor-core / reduced-precision solver.

**3 seed macro briefs (all fork off baseline):**
- W0: batched cuSOLVER `cusolverDnXsyevBatched` (CUDA-13, no longer n≤32-capped) via load_inline; pivot to multi-stream syevd if it serializes at n≥512.
- W1: tensor-core batched two-sided/block-Jacobi (torch.bmm BF16/TF32 or Triton tl.dot tf32x3), host-visible off-norm convergence break, FP32 CholeskyQR2/Newton-Schulz cleanup + Rayleigh-quotient L.
- W2: batched two-step band reduction (full→band via BLAS-3 batched GEMM/syr2k on tensor cores, band→tridiag cheap), then tridiagonal eigensolve.

**Correctness gates (loose, residual-based):** eigen ‖AQ−Qdiag(L)‖, recon ‖Qdiag(L)Qᵀ−A‖, orthogonality ‖QᵀQ−I‖ (max-over-batch) each vs FP64 matrix L1 norm; PLUS eigenvalues sorted **ascending** (hard gate). ~1-2 digits at n=512. Low-precision interior OK if Q is FP32-orthonormalized and L from FP32 Rayleigh quotient. Landmines: clustered/repeated/mixed/rankdef shapes (slow Jacobi convergence; sign-function splits break on eigenvalues at the contour — that's why the sign-function D&C direction was NOT seeded). See [[autocuda-eigh-hack-surface]] for which reward hacks the harness catches.

Research scouts (4) all converged: cusolverDnXsyevBatched = highest-EV easy win; one-sided block-Jacobi (arXiv 2601.17979) best custom-kernel ref; QDWH/Zolo de-prioritized (slow vs MAGMA on GPU); no published Triton eigh exists. Refs: RIKEN-RCCS/EigenG-Batched, KingJamesSong/BatchED.

**DEAD ENDS measured by ~15:30 (6 families, all confirmed cannot beat 1.10x):**
1. Pure batched cusolverDnXsyevBatched = 1.10x CEILING (`2332f1a0`, global best). Rejects BF16/FP16 input; can't tensor-core the FP32 BLAS-2 sytrd (~49%).
2. Tensor-core block-Jacobi (torch.bmm) = 10.5x SLOWER. Even a FREE subproblem solver caps it ~1.2x (rotation-application bmm 139ms vs baseline 170ms on n512 b640); O(sweeps·n^3).
3. Band-reduction + torch.linalg.eigh stage2 = 6.4x regression. torch eigh ignores band/tridiag structure (saves ~0 without bulge-chase); torch reductions launch-bound.
4. QDWH/polar D&C (Chol-DWH) = 2.4x slower. potrf/trsm-bound (CUDA-core), tensor-core bmm <1%; subspace extraction needs slow pivoted QR or a 2nd full eigh.
5. Rank-deficiency fast path + small-shape squeeze = no win. Batched-syevd pays NO rank penalty (rankdef 153ms == dense 158ms); cuSOLVER Xsyev already fastest small-shape path.
6. NS-sign-function spectral D&C = 15% regression. Ragged per-matrix rank (syevd needs uniform blocks); median-find = 14+ sign fns; ALL 13 benchmark spectra have zero median gap so balanced split never engages.

**META-PATTERN:** every dead end delegated final diagonalization to cuSOLVER/torch.linalg.eigh (per-matrix launch serialization + FP32 BLAS-2 sytrd wall) OR round-tripped HBM via torch.bmm between library calls. Frontier needs a UNIFORMLY-BATCHABLE CUSTOM tensor-core eigensolver with NO spectrum-dependent branching. Reusable gem (W0): BF16 sign loop spent 51% in FP32<->BF16 CONVERSION, only 7% in tensor-core GEMM — staying BF16 end-to-end (one convert in/out) cut cost 4x. Submission must be leaderboard-portable: torch + STOCK CUDA only (no MAGMA/external libs). validate.sh bans substring "stream". Leaderboard custom load_inline C++/cuSOLVER extension CONFIRMED remote-portable (1.10x best ACCEPTED on Modal).

**BREAKTHROUGH (~16:15) — the winning architecture = tensor-core reduction -> custom tridiagonal eigensolver. BOTH halves proven fast in isolation:**
- W1 brief-3 (commit 59f2aaec): GEMM full->band reduction. Profile (n2048 b8) = parent 70% BLAS-2 GEMV sytrd; W1's reduction ~1.6x faster implemented, idealized panel A@V+syr2k ~30x (6.8ms TF32). THE WALL was the band SOLVE: cuSOLVER exploits NO bandedness (bw=1 eigh still 197ms), no batched tridiagonal eigensolver in its API.
- W2 brief-2 (commit 91edc84b): custom Triton tridiagonal eigensolver = Sturm-bisection eigenvalues + inverse-iteration eigenvectors, ONE program/matrix vectorized over n eigenvalues. **16.6x faster than torch.linalg.eigh(tridiag)** at n512 b640 (8.4k vs 140k us); bisection alone 22x faster than cuSOLVER eigvalsh. Validated 39/39. Robustness: random RHS (anti degenerate-cluster collapse), shifted CholeskyQR3 (rank-deficient bases), FP32 tail, per-matrix eigh fallback for rankdef at B=640. Its remaining wall = stage-1 eager per-column flood (62%, ~10,200 launches; torch.compile can't fuse it).
- **COST MODEL: fast reduction + W2 solver(8k) + tail(15k) ~= 25k us = 2.2x vs baseline, BEATS the 1.66x frontier.** Combine brief written (W2 brief-3, parents 59f2aaec+91edc84b). W0 (D&C solver) + W1 (bisection solver) independently build the same 2-stage shape with different solvers as de-risking redundancy. Reusable asset: W2's Triton bisection + inverse-iteration kernels work on ANY tridiagonal.
