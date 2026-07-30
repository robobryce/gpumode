# Environment

Per-machine half of the project description (the machine-agnostic half is the
committed `autocuda/layout.md`). This file records the GPU, toolchain, runtime,
resource, and profiler facts shared by GPU MODE problems on this host. This node
uses the local `harness/` scripts while its B200 is uncontended. If Modal
credentials are available, Modal is an authorized fallback when the local GPU
is unavailable or materially contended, using the same production-matched
execution environment described below.

## GPU and environment

- **Provisioning:** run `bash bin/install.sh` from the repository root. It
  installs/selects CUDA 13.3.0 with leaderboard-matched `nvcc` 13.3.33,
  recreates `.venv`, installs the production-matched Python packages plus
  `/opt/cutlass` and `/opt/mathdx`, writes
  `~/.config/gpumode/gpumode.env`, and executes `bin/verify_environment.py`.
  `harness/env.sh` repeats that verification for every local harness command;
  do not bypass the generated config or substitute a manually assembled venv.
- **Programming models:** the installer provides and smoke-tests CUDA C++,
  CuTe DSL 4.5.2, cuTile Python 1.4.0, Triton 3.7.0, cuTile Rust 0.2.0,
  CUDA Oxide 0.2.1, and TileLang 0.1.12. Rust lives under `~/.cargo`, the
  source-backed Rust DSLs live under `/opt/cutile-rs` and `/opt/cuda-oxide`,
  and CUDA Oxide uses its pinned Rust nightly plus LLVM/Clang 21. Run
  `bin/verify_programming_models.sh` to compile and execute all seven smoke
  kernels.

- **Host GPU:** 1× NVIDIA B200 (Blackwell, 183359 MiB HBM3e), compute capability
  **10.0 (sm_100)**. Single device — `torch.cuda.device_count() == 1`, no
  round-robin; `autocuda run exclusive` claims `gpu-0`, pins
  `CUDA_VISIBLE_DEVICES=0`, and serializes every validation, benchmark, and
  profile on that GPU (`harness/env.sh` also defaults direct calls to device 0).
- **Driver / CUDA:** driver **580.126.09**; the system default is the leaderboard-matched CUDA toolkit **13.3.0** at `/usr/local/cuda` (`nvcc V13.3.33`). The `cuda` alternative is in automatic mode and resolves to `/usr/local/cuda-13.3`; the CUDA 13.3 packages are held to prevent an unattended upgrade to the 13.3.1 maintenance stack. Torch's bundled CUDA runtime is 13.0 (`torch.version.cuda`).
- **Leaderboard runtime:** Python **3.13.14** and PyTorch **2.12.0+cu130** live in the freshly rebuilt venv `/home/shadeform/gpumode/.venv`. Its direct dependency groups and install order mirror the production KernelBot image, with Torch installed last. `/home/shadeform/.config/gpumode/gpumode.env` selects this venv, `/usr/local/cuda`, `/opt/cutlass`, and `/opt/mathdx` for every harness invocation.
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
- **Drift rule:** after changing either manifest or venv, rerun
  `bash bin/install.sh`; `bin/verify_environment.py` checks the imports,
  Python/Torch/CUDA versions, direct dependency constraints, and native header
  trees. If the environments differ, rebuild the stale environment and do not
  compare its timings with the other backend.

## Execution backend selection

- **Preferred backend on this host: local B200 when uncontended.** On 2026-07-29,
  `nvidia-smi -L` reported one NVIDIA B200 and the workspace venv reported
  `torch.cuda.is_available() == True`, `torch.cuda.device_count() == 1`, and
  `torch.cuda.get_device_name(0) == "NVIDIA B200"`. Use the local
  `run exclusive` commands above whenever that device is available without
  material competing compute or memory use.
- **Modal fallback:** if Modal credentials are available, select it when no
  compatible local GPU is accessible or when sustained competing use of every
  compatible local GPU would invalidate timing or block useful worker progress.
  Keep builds local under `run slice`, and run
  `harness/modal.sh validate|benchmark` under `run slice` as documented in
  `layout.md`. Record `backend=modal`, the observed contention or availability
  evidence, and the candidate commit in the trial description.
- **Consistency:** validate and benchmark one candidate on one backend. Treat
  a one-off slow result as noise rather than contention, and repeat marginal
  improvements on the comparison candidate's backend. A broken toolchain or
  missing credentials remains an infrastructure issue, not a fallback signal.

## Timeouts

Timeouts are problem-dependent. Measure build, validation, benchmark, and
profiling durations for each run on its selected backend, then set its config
with enough margin for cold compilation and slower test cases. The following
host-level guidance applies across problems:

- A fresh custom-CUDA build can take tens of seconds; cached or Python-only
  builds can be much faster. Do not derive a universal build timeout from one
  problem.
- Validation and benchmark duration scales with the problem's test and
  benchmark cases. Record measured values in the experiment data rather than
  this shared file.
- Nsight Systems captures usually finish much faster than focused Nsight
  Compute captures. Nsight Compute may take minutes because it replays kernels;
  always pass a narrow `--kernel-name` regex and budget accordingly.
- **Leaderboard submit (`harness/submit.sh`):** variable and intermittently
  times out — budget ≥1200 s and **retry on timeout up to 3×** (a timeout is
  inconclusive; a parsed `verdict=REJECTED` is a real failure).

## Profiling

Profile through the harness scripts (see `layout.md` § Profiling). Both run
`eval.py benchmark` on one shape under the `torch.cuda.profiler` range eval.py
wraps its timed launches in (present on **`main`**), so the capture holds only
the submission's kernels — not input generation, warmup, the L2 flush, or the
per-repeat correctness checks. Profile from a base that includes that range.

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
    bash harness/profile_nsys.sh <set>/<problem> "<shape-spec>" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
  ```
- **ncu** (per-kernel; needs root — passwordless `sudo -n` works here, `ncu` at
  `/usr/local/cuda/bin/ncu`). **Always pass** the optional 3rd-arg
  `--kernel-name` regex or ncu replays every kernel ~40× each:
  ```
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_ncu.sh <set>/<problem> "<shape-spec>" "regex:<kernel-pattern>" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
  ```
  For a custom kernel, profile it once to learn its kernel names, then filter to
  the dominant kernel or small set of kernels relevant to the trial.
