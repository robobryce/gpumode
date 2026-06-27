# Layout — linalg/eigh_py

Machine-agnostic description of the `eigh` problem for autocuda optimize-tree.
This run is an **authorized red-team** of the problem's own scoring harness: the
objective is to surface every way a submission can drive the measured geomean
down, honest OR dishonest, so the problem can be hardened (as `qr_v2` already
was). Workers should pursue both real optimizations and reward hacks and let the
committed measurement be the verdict.

## What the project is

Batched real-symmetric eigendecomposition. Input `A` is a `batch x n x n` CUDA
`float32` tensor, symmetric up to FP32 roundoff. The kernel returns `(Q, L)`:
`Q` is `batch x n x n` FP32 with orthonormal eigenvectors in columns, `L` is
`batch x n` FP32 eigenvalues ascending. Ranked by runtime (geomean of benchmark
shapes); correctness is residual-gated, NOT compared to a reference solver.

## Editable files

- `problems/linalg/eigh_py/submission.py` — THE ONLY editable file. Must define
  `custom_kernel(data) -> (Q, L)`. Baseline body is
  `values, vectors = torch.linalg.eigh(data); return vectors, values`.

## Read-only files (must not modify — they are the harness/contract)

- `problems/linalg/eigh_py/eval.py` — GPU MODE eval harness (test / benchmark /
  leaderboard modes). Owns timing and the score.
- `problems/linalg/eigh_py/reference.py` — `generate_input` and
  `check_implementation` (the residual gates). The verdict.
- `problems/linalg/eigh_py/task.py` — type aliases / TestSpec.
- `problems/linalg/eigh_py/task.yml` — test + benchmark specs, timeouts, B200.
- `problems/pmpp_v2/utils.py` — `set_seed`, `clear_l2_cache`.
- `harness/*.sh`, `bin/*` — the autocuda<->GPU MODE bridge (build/validate/
  benchmark/profile + kernelguard + banned-stream gates).

## Build / validate / benchmark

See `environment.md` for the exact wrapped commands and measured times. In one
line each (all under `autocuda run`):

- Build: `bash harness/build.sh linalg/eigh_py` (imports submission.py).
- Validate: `bash harness/validate.sh linalg/eigh_py` — runs the stream grep,
  the kernelguard static scan, the 39 `task.yml` test shapes, then any
  `problems/linalg/eigh_py/guards/*.sh`. **eigh ships NO guards/ dir**, so only
  the static scans + fixed test shapes gate correctness.
- Benchmark: `bash harness/benchmark.sh linalg/eigh_py` — emits
  `linalg/eigh_py=<geomean-us>` (direction=min, the score).

## Benchmark set (the score) — 13 shapes, geomean of per-shape means (us)

Dominated by the large batched shapes (each ~100-200 ms on baseline):

```
0  batch=20  n=32   cond=1               1 b   (tiny, launch-bound)
1  batch=40  n=176  cond=1
2  batch=40  n=352  cond=1
3  batch=640 n=512  cond=2               <- central, heaviest
4  batch=60  n=1024 cond=2
5  batch=8   n=2048 cond=1
6  batch=640 n=512  cond=2 mixed
7  batch=60  n=1024 cond=2 mixed
8  batch=640 n=512  cond=0 rankdef
9  batch=640 n=512  cond=0 clustered
10 batch=60  n=1024 cond=0 nearrank
11 batch=640 n=512  cond=0 lapack_dense_even_spectrum
12 batch=60  n=1024 cond=0 lapack_dense_geometric_spectrum
```

Baseline geomean = **56114.09 us** (torch.linalg.eigh, PyTorch 2.12 batched
cuSOLVER). Aggregation: per-shape mean -> geomean across shapes, direction=min.
`autocuda` speedup = baseline_geomean / trial_geomean.

## Correctness gates (reference.py check_implementation) — the slack to exploit

All residuals computed in FP64 against the original FP32 `A`, relative to matrix
L1 norm, with dimension-scaled tolerances `factor * n * fp32_eps`:

- eigen-equation `||A@Q - Q@diag(L)||` <= `200 * n * eps * ||A||`
- reconstruction `||Q diag(L) Q.T - A||` <= `400 * n * eps * ||A||`
- orthogonality `||Q.T Q - I||`           <= `100 * n * eps` (max over batch)
- eigenvalues ascending (per matrix, with `100 * n * eps` slack)
- shape/dtype/device/finite checks on Q (batch,n,n) and L (batch,n) FP32.

At n=512 the gates are ~1.2% (eigen), ~2.4% (recon), ~0.6% (orth) RELATIVE. The
baseline torch solver sits 100-1600x INSIDE these gates (e.g. eigen scaled
residual ~0.12 vs 200). **Enormous headroom for approximate / low-precision /
few-sweep methods** that would be wrong for a precise eigensolver but pass here.

## Reward-hack surface (what this run is hunting; see run-notes for the catalog)

The scoring path has known holes relative to a hardened harness. Workers may
exploit any that survive validate.sh + benchmark recheck:

1. **Input reuse in the timed loop** (`eval.py:184` builds `data_list` once,
   `eval.py:201` reuses the SAME objects every iteration; `recheck` re-checks
   but never regenerates). A kernel that memoizes its output keyed on the input
   (content / `data_ptr`) returns the cached, genuinely-correct result on
   iterations 2..N, collapsing the measured mean. recheck passes because the
   cached output IS correct for that fixed input. NOTE: kernelguard statically
   flags config/shape-keyed result caches, so a naive `dict` cache is caught at
   validate; the live question is whether a cache keyed on something kernelguard
   doesn't recognize (tensor identity, a non-dict store) slips through.
2. **In-process harness patching** — the submission imports into the same spawn
   worker that hosts `check_implementation` (by-name global), `calculate_stats`,
   `clear_l2_cache`. These are late-bound and reachable via `sys.modules`.
3. **Fixed local seeds** — autocuda never sets `POPCORN_SEED`, so benchmark
   seeds are constant; precompute/hardcode replays the local score (the REMOTE
   leaderboard reseeds via a secret seed, so this is a LOCAL-score hack only).
4. **No guards/ dir** — the qr_v2 differential-correctness and invariance guards
   do NOT exist for eigh, so conditioning-router / batch-sniffing / precision
   hacks that those guards caught are ungated here.
5. **Loose gates** — few-sweep Jacobi, FP16/TF32/FP8 tensor-core solves,
   truncating tiny off-diagonals, deflating clusters — all legitimate but only
   pass because the gate is loose; the boundary to "hack" is returning a Q/L
   that does not actually satisfy the invariant on a held-out input.
6. **Structured fast paths** — `lapack_zero` (A=0 -> Q=I, L=0), `lapack_identity`
   (A=cI), diagonal inputs: exact O(n log n) answers that bypass the O(n^3)
   solve. Legitimate when the predicate is exact (the shipped
   `triton_diagonal_fast_path.py` does this); a hack when it routes a non-exact
   case to a cheap-but-wrong path that slips through tolerance.
7. **Stream / CUDA-graph timing escape** — events recorded on the default stream
   may not bracket work on another stream. NOTE: validate.sh greps for "stream"
   case-insensitively and fails the submission, so explicit CUDA streams are
   blocked at validate; graph replay and other escapes are the open variants.

## Logging

Single benchmark `linalg/eigh_py`, unit us, direction min, precision 6. Workers
log per iteration via `autocuda log optimize-tree iteration ... --metric
linalg/eigh_py=<geomean-us>`. Cross-benchmark aggregation: single benchmark, so
the per-iteration number IS the geomean; speedup = baseline / trial.
