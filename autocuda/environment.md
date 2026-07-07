# Environment

Per-machine half of the project description (the machine-agnostic half is the
committed `autocuda/layout.md`). Written for the **`linalg/eigh_py`** target
(batched real symmetric eigendecomposition, `custom_kernel(A) -> (Q, L)`); the
GPU / toolchain / profiler facts apply to any problem on this host.

## GPU and environment

- **Host GPU:** 1× NVIDIA B200 (Blackwell, ~179 GB HBM3e), compute capability
  **10.0 (sm_100)**. Single device — `torch.cuda.device_count() == 1`, no
  round-robin; `harness/env.sh` pins `CUDA_VISIBLE_DEVICES=0` and every
  benchmark/profile serializes on `gpu-0`.
- **Driver / CUDA:** driver 580.x; CUDA toolkit **13.3** at `/usr/local/cuda`
  (`nvcc V13.3`). torch's bundled CUDA *runtime* is 13.0 (`torch.version.cuda`).
- **PyTorch:** `2.12.1+cu130`, Python 3.12, in the workspace venv
  `/home/shadeform/gpumode/.venv` (path also recorded in
  `~/.config/gpumode/gpumode.env` as `GPUMODE_VENV_PYTHON`).
- **ninja:** already usable (`torch.utils.cpp_extension.is_ninja_available()` →
  True); no symlink fix needed on this host.
- **Triton:** 3.7.0 (bundled with torch 2.12). `tl.dot` lowers to 5th-gen
  `tcgen05` MMA automatically on sm_100. FP32 inputs route via
  `input_precision=`: `"tf32"` (default, single pass, ~1100 TFLOPS),
  `"tf32x3"` (≈FP32 accuracy at ~1/3 the rate), `"ieee"` (true FP32, no tensor
  cores — slow, reference only).
- **Compile arch:** target **sm_100**. Plain
  `-gencode arch=compute_100,code=sm_100` works for WMMA / `mma.sync`; the
  **`a` suffix (`compute_100a,code=sm_100a`) is REQUIRED** for `tcgen05` / TMA /
  thread-block-cluster PTX (set `TORCH_CUDA_ARCH_LIST="10.0a"` for
  `load_inline`).
- **Hardware perf (stable facts to plan around):** 148 SMs, ~228 KB opt-in
  SMEM/block, 192 MB L2, ~8 TB/s HBM3e. **FP64 is weak** (~37–40 TFLOPS tensor)
  — avoid it on the hot path. Dense tensor-core peaks: FP16/BF16 ≈ 2250 TFLOPS,
  TF32 ≈ 1100, FP8 ≈ 4500, FP4 ≈ 9000; cuBLAS BF16x9 emulation gives
  ~FP32-accurate GEMM at up to ~3× native FP32.
- **Toolchain caps:** 30 CPUs (Intel Xeon 6960P), 180 GB RAM. `harness/env.sh`
  exports `MAX_JOBS=3` per worker; per-worker slices in
  `autocuda/worker-limits.csv` are 8 CPUs / 48 GB each (`autocuda run slice`).

## Baseline

The unmodified `submission.py` is stock `torch.linalg.eigh` (cuSOLVER). Fork
workers off the **`main`** tip (or later): that is the baseline, and it carries
the eigh `eval.py` `torch.cuda.profiler` capture range the profilers need. The
baseline geomean is re-measured at the start of each run (the manager logs it) —
don't rely on a hardcoded number. Benchmark shapes come from
`problems/linalg/eigh_py/task.yml`.

For where the baseline sits on the leaderboard and the gap to #1, read the
standings with the **`leaderboard-rankings`** skill (no auth, no submission) —
rankings move, so don't hardcode them here.

- **Bottleneck (stable):** for n>32 `torch.linalg.eigh` loops cuSOLVER `syevd`
  per matrix (no batched path), all SIMT / **zero tensor-core**, latency-bound
  (Householder tridiagonalization + GEMV panel updates + divide-and-conquer). At
  n≤32 it takes the genuinely-batched Jacobi path. So any truly *batched* kernel
  — especially one routing the O(n³) work onto FP16/BF16/TF32 tensor cores — has
  large headroom.
- **Precision budget (stable — frozen `reference.py`):** the checker gates
  normwise residuals (eigen-equation, reconstruction, orthogonality) relative to
  the FP64 L1 norm, with the tolerance growing in n, so reduced-precision
  interior math is admissible. Safe recipe: do the bulk O(n³) work in low
  precision but emit **`Q` through an FP32 orthonormalization** (Newton–Schulz /
  Gram–Schmidt) and **`L` via an FP32 Rayleigh quotient `diag(QᵀAQ)`**.

## Noise

Treat geomean deltas below **~1%** as noise (eval.py early-exits each shape at
`err/mean < 0.1%`). Re-measure rather than trust a sub-1% improvement.

## Timeouts

Measured on this host (one B200, via `autocuda run`); budget with margin:

- **Build (pure-PyTorch import):** ~2 s. A fresh custom-CUDA `load_inline`
  (sm_100) compile is tens of seconds; a cached rebuild (unchanged source) ~1 s.
- **Validate (test shapes):** ~7 s for the baseline; a custom kernel running the
  LAPACK / rankdef test cases is slower — budget up to ~240 s (`test_timeout`).
- **Benchmark (geomean over the benchmark shapes):** ~30 s for the baseline;
  budget up to ~300 s (`benchmark_timeout` is 480 s).
- **nsys profile (one shape):** ~15–20 s.
- **ncu profile (one shape, focused):** tens of seconds to minutes; budget
  ≥600 s and always pass a `--kernel-name` regex.
- **Leaderboard submit (`harness/submit.sh`):** variable and intermittently
  times out — budget ≥600 s and **retry on timeout up to 3×** (a timeout is
  inconclusive; a parsed `verdict=REJECTED` is a real failure).

## Profiling

Profile through the harness scripts (see `layout.md` § Profiling). Both run
`eval.py benchmark` on one shape under the `torch.cuda.profiler` range eval.py
wraps its timed launches in (present on **`main`**), so the capture holds only
the submission's kernels — not input generation, warmup, the L2 flush, or the
per-repeat FP64 checker. Profile from a base that includes that range.

- **nsys** (timeline, no sudo) — default shape is the first benchmark line; pass
  a `<shape-spec>` (a `task.yml` benchmarks line) to pick another. Set
  `NSYS_OUT` to keep the `.nsys-rep`:
  ```
  NSYS_OUT="$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-rep" \
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_nsys.sh linalg/eigh_py "batch: 640; n: 512; cond: 2; seed: 1029" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
  ```
- **ncu** (per-kernel; needs root — passwordless `sudo -n` works here, `ncu` at
  `/usr/local/cuda/bin/ncu`). **Always pass** the optional 3rd-arg
  `--kernel-name` regex or ncu replays every kernel ~40× each:
  ```
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_ncu.sh linalg/eigh_py "batch: 640; n: 512; cond: 2; seed: 1029" "regex:sytrd|gemv|laed" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
  ```
  Filter by path: **n≤32** baseline → `syevj`, `jacobi`; **n>32** baseline →
  `sytrd`, `symv`, `gemv`, `laed`. For a custom kernel, profile it once to learn
  its kernel names, then filter to the dominant one.
