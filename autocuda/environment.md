# Environment

Per-machine half of the project description (the machine-agnostic half is the
committed `autocuda/layout.md`). Written for the **`linalg/cholesky_py`** target
(batched dense Cholesky factorization, `custom_kernel(A) -> L`); the
GPU / toolchain / profiler facts apply to any problem on this host. This node
uses the local `harness/` scripts while its B200 is uncontended. Modal is an
authorized fallback when the local GPU is unavailable or materially contended,
using the same production-matched execution environment described below.

## GPU and environment

- **Host GPU:** 1× NVIDIA B200 (Blackwell, 183359 MiB HBM3e), compute capability
  **10.0 (sm_100)**. Single device — `torch.cuda.device_count() == 1`, no
  round-robin; `autocuda run exclusive` claims `gpu-0`, pins
  `CUDA_VISIBLE_DEVICES=0`, and serializes every validation, benchmark, and
  profile on that GPU (`harness/env.sh` also defaults direct calls to device 0).
- **Driver / CUDA:** driver **580.126.09**; the system default is the leaderboard-matched CUDA toolkit **13.3.0** at `/usr/local/cuda` (`nvcc V13.3.33`). The `cuda` alternative is in automatic mode and resolves to `/usr/local/cuda-13.3`; the CUDA 13.3 packages are held to prevent an unattended upgrade to the 13.3.1 maintenance stack. Torch's bundled CUDA runtime is 13.0 (`torch.version.cuda`).
- **Leaderboard runtime:** Python **3.13.14** and PyTorch **2.12.0+cu130** live in the freshly rebuilt venv `/home/shadeform/gpumode/.venv`. Its direct dependency groups and install order mirror the production KernelBot image, with Torch installed last. `/home/shadeform/.config/gpumode/gpumode.env` selects this venv and `/usr/local/cuda` for every harness invocation.
- **nvJitLink:** PyTorch resolves its bundled `nvidia/cu13/lib/libnvJitLink.so.13` (13.0.88), while native CUDA 13.3 compilation resolves `/usr/local/cuda-13.3/targets/x86_64-linux/lib/libnvJitLink.so.13` (13.3.33). A cold extension rebuild, full validation, and full benchmark all pass without `LD_PRELOAD`; do not reintroduce the old preload workaround.
- **ninja:** already usable (`torch.utils.cpp_extension.is_ninja_available()` →
  True); no symlink fix needed on this host.
- **Triton:** 3.7.1 (bundled with torch 2.12). `tl.dot` lowers to 5th-gen
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
  exports `MAX_JOBS=3` per worker. `autocuda run slice --tag "$TAG"` reads the
  run's immutable `<tag>-config.json` and enforces its per-worker CPU/memory
  limits; four simultaneous builds use at most 12 compiler jobs, leaving ample
  host headroom.
- **Local execution:** build with `autocuda run slice --data-dir "$DATA_DIR" --tag "$TAG" -- bash harness/build.sh <set>/<problem>`. Run validation, benchmarking, and profiling under `autocuda run exclusive --data-dir "$DATA_DIR" --`. Use an absolute shared `DATA_DIR` from worker worktrees and wrap each script exactly once.
- **Static validation:** `kernelguard` is installed in the local control venv alongside the production runtime packages so the existing harness performs the same local policy gate before evaluation.

## Local and Modal runtime parity

- **Canonical manifest:** `harness/modal_runner.py`, synchronized to KernelBot commit `4bdd839ad76eaac0cdab986d30f8d77f989a289b`. Both backends use Python 3.13, CUDA toolkit 13.3.0, Torch 2.12 with CUDA runtime 13.0, and Torch is installed after every other direct dependency so its CUDA/NCCL dependency set wins.
- **Shared dependency setup:** Ninja, Wheel, Requests, Packaging, NumPy,
  PyTest, PyYAML, Tinygrad, Helion, CUTLASS DSL 4.5.2, CUDA Core/Python,
  CUDA Tile 1.4.0, nvMath Python 0.9.0, MathDx cu13 0.3.2.6, and CUDA Toolkit
  13.0.2 follow the same groups and constraints in both environments. The
  current clean local venv was rebuilt and audited against that manifest on 2026-07-30;
  `import cuda.tile, nvmath, cutlass.cute, torch` passes and Torch reports
  `2.12.0+cu130`.
- **Common runtime only:** submissions must use dependencies and headers
  available in both environments. Modal CLI 1.5.2 and local `kernelguard` are
  controller/static-check tools and intentionally stay outside the clean
  execution runtime. Do not add a convenience package to only one backend.
- **Drift rule:** after changing either manifest or venv, verify the imports,
  Python/Torch/CUDA versions, and direct dependency constraints before using
  Modal fallback. If they differ, rebuild the stale environment and do not
  compare its timings with the other backend.

## Execution backend selection

- **Preferred backend on this host: local B200 when uncontended.** On 2026-07-29,
  `nvidia-smi -L` reported one NVIDIA B200 and the workspace venv reported
  `torch.cuda.is_available() == True`, `torch.cuda.device_count() == 1`, and
  `torch.cuda.get_device_name(0) == "NVIDIA B200"`. Use the local
  `run exclusive` commands above whenever that device is available without
  material competing compute or memory use.
- **Modal fallback:** select it when no compatible local GPU is accessible or
  when sustained competing use of every compatible local GPU would invalidate
  timing or block useful worker progress. Keep builds local under `run slice`,
  and run
  `harness/modal.sh validate|benchmark` under `run slice` as documented in
  `layout.md`. Record `backend=modal`, the observed contention or availability
  evidence, and the candidate commit in the trial description.
- **Consistency:** validate and benchmark one candidate on one backend. Treat
  a one-off slow result as noise rather than contention, and repeat marginal
  improvements on the comparison candidate's backend. A broken toolchain or
  missing credentials remains an infrastructure issue, not a fallback signal.

## Baseline

The unmodified `submission.py` is stock
`torch.linalg.cholesky_ex(data, check_errors=False).L` (cuSOLVER). Measure and
log the baseline on the run's selected parity-matched backend; record that
backend with the reference and do not reuse a measurement from a mismatched
runtime. Benchmark shapes come from
`problems/linalg/cholesky_py/task.yml` and span thousands of 32×32 matrices
through one 32768×32768 matrix.

For where the baseline sits on the leaderboard and the gap to #1, read the
standings with the **`leaderboard-rankings`** skill (no auth, no submission) —
rankings move, so don't hardcode them here.

- **Bottleneck (stable):** the stock path leaves shape-specific batch
  parallelism, launch fusion, lower-only storage, and Blackwell tensor-core
  updates available to a custom implementation. Measure individual shapes
  on the selected backend before choosing between batched small-matrix and
  blocked large-matrix mechanisms.
- **Precision budget (stable — frozen `reference.py`):** output must be finite
  FP32, lower triangular with a strictly positive diagonal, and satisfy the
  dimension-scaled FP32 L1 reconstruction gate for `L @ L.T == A`. Reduced
  precision may be used internally only when the final factor passes those
  property-based checks across dense, spectral, diagonal, low-rank, row-scaled,
  and tridiagonal inputs.

## Noise

Re-measure backend-specific noise for Cholesky before accepting marginal
changes. Until a fresh multi-run estimate is logged, treat geomean deltas below
**~1%** as noise and confirm them independently on the same backend as the
comparison candidate.

## Timeouts

Measured on this host (one B200, via `autocuda run`); budget with margin:

- **Build (pure-PyTorch import):** ~2 s. A fresh custom-CUDA `load_inline`
  (sm_100) compile is tens of seconds; a cached rebuild (unchanged source) ~1 s.
- **Validate (test shapes):** budget up to 240 s (`test_timeout`).
- **Benchmark (15-shape geomean):** budget up to 600 s
  (`benchmark_timeout`).
- **nsys profile (one shape):** ~15–20 s.
- **ncu profile (one shape, focused):** tens of seconds to minutes; budget
  ≥600 s and always pass a `--kernel-name` regex.
- **Leaderboard submit (`harness/submit.sh`):** variable and intermittently
  times out — budget ≥1200 s and **retry on timeout up to 3×** (a timeout is
  inconclusive; a parsed `verdict=REJECTED` is a real failure).

## Profiling

Profile through the harness scripts (see `layout.md` § Profiling). Both run
`eval.py benchmark` on one shape under the `torch.cuda.profiler` range eval.py
wraps its timed launches in (present on **`main`**), so the capture holds only
the submission's kernels — not input generation, warmup, the L2 flush, or the
per-repeat FP64 checker. Profile from a base that includes that range.

Nsight Systems **2026.1.3** and Nsight Compute **2026.2.0** are installed.
`nsys` works without privilege escalation. `profile_ncu.sh` uses the working
passwordless `sudo -n` path and forwards `PATH`, `PYTHONPATH`, `CUDA_HOME`,
`CUDA_VISIBLE_DEVICES`, `TORCH_EXTENSIONS_DIR`, `PYTHONNOUSERSITE`, and
`MAX_JOBS` across the privilege boundary.

- **nsys** (timeline, no sudo) — default shape is the first benchmark line; pass
  a `<shape-spec>` (a `task.yml` benchmarks line) to pick another. Set
  `NSYS_OUT` to keep the `.nsys-rep`:
  ```
  NSYS_OUT="$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-rep" \
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_nsys.sh linalg/cholesky_py "batch: 640; n: 512; cond: 2; seed: 510512" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
  ```
- **ncu** (per-kernel; needs root — passwordless `sudo -n` works here, `ncu` at
  `/usr/local/cuda/bin/ncu`). **Always pass** the optional 3rd-arg
  `--kernel-name` regex or ncu replays every kernel ~40× each:
  ```
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_ncu.sh linalg/cholesky_py "batch: 640; n: 512; cond: 2; seed: 510512" "regex:potrf|cholesky|trsm|syrk|gemm" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
  ```
  For a custom kernel, profile it once to learn its kernel names, then filter to
  the dominant factor, solve, or trailing-update kernel.
