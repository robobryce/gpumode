# eigh_py routing table

**Git SHA inspected:** `aace73c5df4b7c276ddf5fb5b75456a84883daa3`

This table describes that accepted hardened candidate exactly; it is not a
summary of a moving branch tip.

| Benchmark input | Routed implementation | Library use on normal path | Kernel fusion |
|---|---|---|---|
| `[20,32,32]` dense | Custom `jacobi32`: fused Householder solve followed by selective cyclic Jacobi | None | **High:** complete solve in two custom kernels |
| `[40,176,176]` dense | Two-CTA clustered `reduce_band_to_tridiagonal` | Two GEMMs for certification; cuSOLVER only if certification fails | **Very high:** reduction, tridiagonal solve, reorthogonalization, and backtransform are one kernel |
| `[40,352,352]` dense | Custom width-16 clustered panel reduction, custom tridiagonal solve, blocked-WY backtransform | **Heavy:** about 40 trailing `baddbmm` updates, then BMM/TRSM/BMM backtransform and certificate; conditional cuSOLVER fallback | **Medium-low:** individual panels are fused, but the complete solve is launch- and library-heavy |
| `[640,512,512]` dense | Custom n512 prefix/panel/tridiagonal/refinement route | **Heavy:** explicit cuBLAS BF16x9 WY GEMMs, ATen BMMs, small 16x16 `torch.linalg.eigh` calls during repair; conditional batched cuSOLVER | **Medium:** clustered prefix and panel kernels, custom tridiagonal solve, fused certificate/repair kernels |
| `[60,1024,1024]` dense | Dense classifier branch: 64 custom panel reductions, 25-step tridiagonal specialization, WY backtransform and certification | **Heavy:** roughly 62 panel-update GEMMs plus BMM/TRSM backtransform and Newton/certificate GEMMs; cuSOLVER only for survivors | **Medium-low:** custom panel/solve kernels, but many intervening library launches |
| `[8,2048,2048]` dense | Main-thread owner uses persistent cooperative 8-CTA clustered reducer, custom tridiagonal solve and WY/refinement | **Moderate-heavy:** eight WY blocks and Newton/certificate BMMs; conditional cuSOLVER fallback | **High reducer, medium overall:** all 256 reduction panels run inside one persistent kernel |
| `[640,512,512]` mixed | Custom n512 route because the classifier is whole-batch and heterogeneous batches reject direct-library routing | Same heavy cuBLAS path as dense, generally with more repair and possible subset cuSOLVER | **Medium** |
| `[60,1024,1024]` mixed | General non-dense n1024 branch, 32-step/256 specialization | Heavy cuBLAS/ATen BMM path; subset or whole-batch cuSOLVER remains possible | **Medium-low** |
| `[640,512,512]` rank-deficient | Direct `cusolverDnXsyevBatched` | **Full library eigensolve** | **Low:** only the classifier is custom |
| `[640,512,512]` clustered | Direct `cusolverDnXsyevBatched` | **Full library eigensolve** | **Low:** only the classifier is custom |
| `[60,1024,1024]` near-rank | Positive/geometric n1024 branch, 32-step/128 specialization plus clustered reorthogonalization | Heavy BMM/TRSM path; conditional cuSOLVER fallback | **Medium-low** |
| `[640,512,512]` LAPACK dense-even | Custom n512 route | Heavy cuBLAS WY/refinement and possible cuSOLVER repair | **Medium** |
| `[60,1024,1024]` LAPACK dense-geometric | Signed-geometric n1024 branch, 32-step/2048 specialization; fused 67.5 orthogonality audit, one Newton step, strict recertification and failed-column repair | Heavy BMM/TRSM path; cuSOLVER only for remaining failures | **Medium-low:** certificate work is fused, main panel progression is not |

## Routing qualifications

- The n512 classifier is batch-global. Every matrix must have diagonal
  concentration in `(21.5, 22.5)` for the direct cuSOLVER route. The benchmark
  rank-deficient and clustered batches satisfy it; dense, mixed, and LAPACK
  dense-even batches do not.
- The n1024 route is fundamentally custom in this candidate, but its
  tridiagonal specialization and repair policy depend on batch-wide diagonal
  signatures.
- The normal n2048 benchmark path is custom only for the Python main thread.
  Concurrent non-owner callers use isolated cuSOLVER state.
- The accepted n2048 reducer still uses legacy WMMA TF32 trailing updates. It
  is persistent and heavily fused, but it is not a native `tcgen05` terminal.
- Validation-only `[1,4096,4096]` and any unrecognized size fall directly
  through to `cusolverDnXsyevBatched`.

## Source pointers

- Benchmark shapes: `problems/linalg/eigh_py/task.yml:121`
- Top-level dispatch: `problems/linalg/eigh_py/submission.py:7404`
- n512 classifier: `problems/linalg/eigh_py/submission.py:6061`
- n1024 dispatch: `problems/linalg/eigh_py/submission.py:6520`
- n2048 owner/fallback dispatch: `problems/linalg/eigh_py/submission.py:7383`
- n2048 WMMA update: `problems/linalg/eigh_py/submission.py:3327`
