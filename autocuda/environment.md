# Environment

Per-machine half of the project description (the machine-agnostic half is the
committed `autocuda/layout.md`). Written for the **`linalg/eigh_py`** target
(batched real symmetric eigendecomposition, `custom_kernel(A) -> (Q, L)`).

## GPU and environment

- **Host GPUs:** 2× NVIDIA B200 (183 GB each), compute capability **10.0 (sm_100)**, Blackwell. The fleet round-robins workers across both: even `agent-id` → GPU 0, odd → GPU 1 (`autocuda/worker-limits.csv`).
- **Driver / CUDA:** NVIDIA driver **580.126.09**; CUDA toolkit **13.0** (`/usr/local/cuda`, `nvcc V13.0.88`). (Note: newer than the 12.8 some older notes assume.)
- **PyTorch:** `2.12.1+cu130` in the workspace venv `/home/shadeform/gpumode/.venv` (path recorded in `~/.config/gpumode/gpumode.env` as `GPUMODE_VENV_PYTHON`). `torch.cuda.is_available()` → True, device "NVIDIA B200", capability (10, 0).
- **Custom-CUDA target arch:** compile submissions for **Blackwell sm_100**. Plain `-gencode arch=compute_100,code=sm_100` works for WMMA / `mma.sync`; the **`a` suffix (`compute_100a,code=sm_100a`) is REQUIRED** for any `tcgen05` / TMA / thread-block-cluster PTX (set `TORCH_CUDA_ARCH_LIST="10.0a"` for `load_inline`). B200 FP64 is **weak** (~37–40 TFLOPS tensor, ~half of H100; ~37–45 FP64 CUDA-core) — **avoid FP64 on the hot path**. FP16/BF16 tensor ≈ 2250 TFLOPS, TF32 ≈ 1100, FP8 ≈ 4500; cuBLAS BF16x9 emulation gives ~FP32-accurate GEMM at up to ~3× native FP32. 148 SMs, ~228 KB opt-in SMEM/block, 192 MB L2, ~8 TB/s HBM3e.
- **eigh tolerances admit reduced precision.** The checker (`reference.py::check_implementation`) gates three *normwise* residuals relative to the FP64 matrix L1 norm: eigen-equation `‖AQ−Qdiag(L)‖₁` (rtol `200·n·eps32`), reconstruction `‖Qdiag(L)Qᵀ−A‖₁` (`400·n·eps32`), orthogonality `‖QᵀQ−I‖₁` (`100·n·eps32`, **max over the batch**), plus an ascending-sort gate. Relative budget grows with n: ~2 correct digits needed at n=512, ~1.3 at n=4096. TF32/FP16 internal math is clearly admissible; BF16 is fine for n≥176 well-conditioned cases. Standard recipe: do the bulk O(n³) work in low precision but **emit `Q` through an FP32 orthonormalization** (cheapest way to pass the max-over-batch ORTH gate) and **compute `L` via an FP32 Rayleigh quotient `qᵢᵀAqᵢ`**. Power-of-two prescale `A`→O(1) for the high/low-magnitude LAPACK *test* cases (FP16 would overflow/underflow otherwise — those are in `tests:`, not `benchmarks:`).
- **Toolchain caps:** 60 CPUs, 334 GB RAM on the host. `harness/env.sh` exports `MAX_JOBS=3` per worker (worst case 6 workers × 3 = 18 ≤ 59). Per-worker limits: 7 CPUs, ~42 GB RAM (`systemd-run --user --scope` via `autocuda run slice`, sized in `autocuda/worker-limits.csv`).
- **⚠️ ninja PATH fix (REQUIRED before any custom-CUDA worker):** `bin/install.sh` installs `ninja` into `.venv/bin/ninja`, but the harness runs the venv python without activating the venv, so torch's `load_inline` can't find the ninja binary and fails with `RuntimeError: Ninja is required`. Fixed for this host by `ln -sf /home/shadeform/gpumode/.venv/bin/ninja /home/shadeform/.local/bin/ninja` (`~/.local/bin` is on PATH inside `autocuda run`). Already in place and verified.

## Baseline

The unmodified `submission.py` is `torch.linalg.eigh` (cuSOLVER). Baseline commit
**`a6939aedc`** (= upstream `main` + the eigh_py profiler-range fix; the bare
`torch.linalg.eigh` submission is byte-identical to `f7b786a30`). **Workers must
fork off a commit that contains the profiler range** (`a6939aedc` or later) or
their `ncu`/`nsys` captures record nothing — see *Profiling*.

- **Local benchmark geomean (`harness/benchmark.sh`, 13 shapes): ≈ 254,990 µs** (≈ 255 ms). Re-measured A/B at 254,872 / 254,993 µs → run-to-run noise well under 1%; treat geomean deltas below ~1% as noise (eval.py's internal `err/mean < 0.1%` early-exit bounds per-shape variance).
- **Leaderboard (`--mode leaderboard`, remote): public geomean 0.0539 s, accepted** (submission id 837859, all runs passed). The remote `leaderboard` mode uses different repeat counts than the local `benchmark` mode, so the remote scalar (~53,900 µs) is **not** comparable to the local 255,000 µs geomean — compare local-to-local and remote-to-remote only.

Per-shape baseline (one full `benchmark.sh` invocation; each line is one geomean input; B = batch). **The geomean weights every shape equally (1/13)**, so a 2× on the tiny n=32 shape counts as much as 2× on n=512 B=640:

| # | spec | µs (this shape) |
|---|---|---|
| 0 | batch 20, n 32, cond 1 | 4,586.0 |
| 1 | batch 40, n 176, cond 1 | 19,410.0 |
| 2 | batch 40, n 352, cond 1 | 58,121.9 |
| 3 | batch 640, n 512, cond 2 (dense) | 556,620.5 |
| 4 | batch 60, n 1024, cond 2 (dense) | 598,461.8 |
| 5 | batch 8, n 2048, cond 1 | 623,767.2 |
| 6 | batch 640, n 512, cond 2, **mixed** | 530,339.3 |
| 7 | batch 60, n 1024, cond 2, **mixed** | 597,704.9 |
| 8 | batch 640, n 512, cond 0, **rankdef** | 542,902.7 |
| 9 | batch 640, n 512, cond 0, **clustered** | 448,902.3 |
| 10 | batch 60, n 1024, cond 0, **nearrank** | 597,368.0 |
| 11 | batch 640, n 512, cond 0, **lapack_dense_even_spectrum** | 657,792.3 |
| 12 | batch 60, n 1024, cond 0, **lapack_dense_geometric_spectrum** | 586,856.5 |

The geomean is dominated by the **n=512 B=640 group (5 shapes: 3/6/8/9/11)** and the **n=1024 B=60 group (4 shapes: 4/7/10/12)** — but shapes 0–2 (n≤352) are launch/overhead-bound and still carry 3/13 of the geomean, so a fast small-n / many-small-matrices path matters too. eval.py replicates each input to ~256 MB / (B·n²·4) copies (cap 50): n=32 → 50 copies (1000 matrices/timed call), n=176 → 50 (2000), n=352 → 13 (520); all n≥512 shapes are replication-capped to 1–2 copies (matrices/call = nominal batch).

### Baseline bottleneck (profile-confirmed)

`torch.linalg.eigh` does **not** use a batched path for n>32: it loops cuSOLVER
**`syevd` per matrix** (a known PyTorch dispatch cliff at n=33). nsys on the
n=512 B=640 shape (`profiles/<tag>/baseline-n512-f7b786a30.nsys-rep`) shows a
classic two-stage symmetric solve, **all SIMT / CUDA-core, zero tensor-core**:

- `sytrd4_cta` Householder **tridiagonalization** — 22%
- `cuds_symv_lo_direct` / `gemvx` / `gemv2T` **matrix-VECTOR** products — ~41% combined (memory/latency bound; the panel updates are GEMV, not GEMM)
- `laed3/laed4/laed2` **divide-and-conquer** on the tridiagonal — ~20%
- only stray `cutlass3x_sm100_simt_sgemm` (SIMT SGEMM, *not* tensor-core)

At **n≤32** the baseline instead takes cuSOLVER's true batched-Jacobi path
(`batch_parallel_jacobi_algo1`, `syevj_music_kernel`) — fast and genuinely
batched, so shape 0 has the least headroom of the big-ratio levers.

**Implication:** the baseline is latency-bound serial cuSOLVER with no tensor-core
use, so almost any genuinely *batched* kernel — and especially one that routes the
O(n³) work onto FP16/BF16/TF32 tensor cores — has large headroom. Research-favored
directions: batched one-sided/block Jacobi (SMEM-resident at n≤512, tensor-core
block updates at n≥1024), genuinely-batched tridiag→D&C, and GEMM-heavy spectral
divide-and-conquer (QDWH/Zolo-eig). Finish any low-precision interior with one
FP32 Newton-Schulz / Gram-Schmidt reorthogonalization of `Q`.

## Timeouts

Measured wall-clock on this host (one B200, via `autocuda run`):

- **Build (pure-PyTorch import):** ~1 s. A fresh custom-CUDA `load_inline` (sm_100) compile is tens of seconds; cached rebuild (unchanged source) ~1 s (content-hash keyed in per-worktree `problems/linalg/eigh_py/.torch_ext`).
- **Validate (39 test shapes, no problem-specific guards):** ~12 s for the `torch.linalg.eigh` baseline. A custom kernel that runs the LAPACK high/low-magnitude and rankdef/clustered test cases will take longer; budget up to ~240 s (`test_timeout` in task.yml is 240 s).
- **Benchmark (13 shapes, geomean):** ~94 s for the slow baseline. A faster custom kernel runs shorter; a slow one can run much longer — budget up to ~300 s.
- **nsys profile (one shape, capture-range):** ~10–20 s wall.
- **ncu profile (one shape, `--set full`):** tens of seconds to minutes depending on kernel count and the `--kernel-name` filter; budget ≥600 s and always pass a focusing regex.
- **Leaderboard submit (`harness/submit.sh`):** highly variable and intermittently times out — observed ~75 s for a clean round-trip but also full-timeout failures. Budget ≥600 s and **retry on timeout up to 3×** (a timeout is inconclusive; a parsed `verdict=REJECTED` is a real failure). `harness/submit.sh` now parses the popcorn-cli JSON result as the authoritative verdict and prints `submission_id` + `public_score`.

## Profiling

Profile through the harness scripts (see `layout.md` § Profiling). Both run
`eval.py benchmark` on one shape under the `torch.cuda.profiler` range that
eigh_py/eval.py now wraps its timed `custom_kernel` launches in (committed in
`a6939aedc` / PR gpu-mode/reference-kernels#157), so the capture holds only the
submission's kernels — not input generation (which calls cuSOLVER `qr` to build
the random basis), the warmup, the L2 flush, or the per-repeat FP64 reference
checker. **A worktree forked before `a6939aedc` lacks the range and will capture
nothing** ("No reports were generated" from nsys).

- **nsys** (timeline, no sudo) — default shape is the first benchmark line; pass a `<shape-spec>` (a `task.yml` benchmarks line) to pick another. Set `NSYS_OUT` to keep the `.nsys-rep`:
  ```
  NSYS_OUT="$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-rep" \
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_nsys.sh linalg/eigh_py "batch: 640; n: 512; cond: 2; seed: 1029" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
  ```
- **ncu** (per-kernel; needs root — **passwordless `sudo -n` works on this host**, `ncu` at `/usr/local/cuda/bin/ncu`). The optional 3rd arg is an `--kernel-name` regex — **always pass one** or ncu replays every kernel for ~40 passes each:
  ```
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_ncu.sh linalg/eigh_py "batch: 640; n: 512; cond: 2; seed: 1029" "regex:sytrd|gemv|laed" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
  ```
  Kernel names to filter on depend on the path: **n≤32** baseline → `syevj`, `jacobi`; **n>32** baseline → `sytrd`, `symv`, `gemv`, `laed`. For a custom kernel, profile it once on iteration 1 to learn its kernel names, then filter to the dominant one.
