# Environment

Per-machine half of the project description (the machine-agnostic half is the
committed `autocuda/layout.md`). This file records the GPU, toolchain, runtime,
resource, and profiler facts shared by GPU MODE problems on the RTX PRO 6000
SM120 workers used for autocuda experiments.

## Provisioning and verification

- Run `bash bin/install.sh` from the repository root. The script installs or
  selects CUDA 13.3, requires `nvcc` 13.3.33, recreates the checkout-local
  `.venv` with Python 3.13 and the production-matched Python packages, installs
  the native SDKs and programming-model toolchains, and writes the generated
  machine configuration to `~/.config/gpumode/gpumode.env`.
- The generated configuration records `GPUMODE_VENV_PYTHON`, `CUDA_HOME`,
  `CUTLASS_PATH`, `MATHDX_HOME`, `CUTILE_RS_PATH`, `CUDA_OXIDE_PATH`,
  `CUDA_TOOLKIT_PATH`, and `CUDA_OXIDE_LLC`. `harness/env.sh` sources it and
  runs `bin/verify_environment.py` before every local harness command. Do not
  bypass it or substitute a manually assembled environment.
- Setup finishes by running `bin/verify_environment.py`, the kernelguard
  self-test, and `bin/verify_programming_models.sh`. The last script compiles
  its CUDA C++ smoke kernel for the compute capability reported by the active
  Torch CUDA device, which is `sm_120` on these workers, and executes smoke
  tests for every supported programming model.
- Popcorn CLI installation is enabled by default and may be disabled with
  `GPUMODE_INSTALL_POPCORN=0`. Popcorn and Modal authentication are deliberately
  separate from host provisioning.

## GPU and host

- **GPU:** 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition with 97,887 MiB of
  memory, compute capability **12.0 (`sm_120`)**, and 188 SMs. The observed
  per-block opt-in shared-memory limit is 101,376 bytes (99 KiB).
- **Architecture rule:** compile native extensions and kernels for the detected
  `sm_120` target. Do not reuse B200-specific `sm_100`, `sm_100a`, cluster, TMA,
  or shared-memory assumptions without checking that the feature is supported
  on compute capability 12.0. For Torch extensions that need an explicit list,
  use `TORCH_CUDA_ARCH_LIST="12.0"`.
- **Device allocation:** each experiment has one GPU. `harness/env.sh` defaults
  `CUDA_VISIBLE_DEVICES` to device 0. `autocuda run exclusive` claims `gpu-0`
  and serializes validation, benchmark, and profiling work on it.
- **Host resources:** the measured worker configuration has 16 AMD EPYC 9335
  CPUs, 141 GiB RAM, and 4 GiB swap. `harness/env.sh` defaults `MAX_JOBS=3` per
  worker so four simultaneous agent builds use at most 12 compiler jobs.

## Runtime and programming models

- **Driver and CUDA:** NVIDIA driver 580.126.09; CUDA toolkit 13.3.0 at
  `/usr/local/cuda`; `nvcc` 13.3.33. The installer selects
  `/usr/local/cuda-13.3` and holds the exact compiler packages. PyTorch uses its
  bundled CUDA 13.0 runtime.
- **Python:** Python 3.13.14 and PyTorch 2.12.0+cu130 in the checkout-local
  `.venv`. `bin/install.sh` installs Torch last so its CUDA and NCCL dependency
  set wins, then verifies all direct dependency constraints and imports.
- **CUDA C++ and headers:** CUTLASS/CuTe DSL 4.5.2 at `/opt/cutlass` and MathDx
  26.06.0 at `/opt/mathdx`. The installer verifies their pinned contents and
  compiles a cuBLASDx header smoke test.
- **Python programming models:** CuTe DSL 4.5.2, cuTile Python 1.4.0, Triton
  supplied by PyTorch 2.12, and TileLang 0.1.12.
- **Rust programming models:** cuTile Rust 0.2.0 at `/opt/cutile-rs`, pinned to
  commit `d89788bca7de8a9cbeabc5ded63740520a96c223`; CUDA Oxide 0.2.1 at
  `/opt/cuda-oxide`, pinned to commit
  `4514af2ca8a21a9f8feb187567f61fe67090f881`. CUDA Oxide uses Rust nightly
  `nightly-2026-04-03`, including `rust-src`, `rustc-dev`, and `llvm-tools`,
  together with LLVM/Clang 21 and `/usr/bin/llc-21`. Stable Rust is also
  installed for the cuTile Rust examples.
- **Build isolation:** `harness/env.sh` sets `PYTHONNOUSERSITE=1` and gives each
  problem worktree its own `.torch_ext` cache. This prevents concurrent
  optimize-tree workers from sharing a Torch extension build directory.
- **nvJitLink:** PyTorch resolves its bundled CUDA 13.0 nvJitLink while native
  compilation resolves the CUDA 13.3 toolkit copy. The verified environment
  does not require an `LD_PRELOAD` workaround.

## Local and Modal runtime parity

- `harness/modal_runner.py` is the canonical remote manifest. Both local and
  Modal environments use Python 3.13, CUDA toolkit 13.3, PyTorch 2.12 with its
  CUDA 13.0 runtime, and the same Python dependency groups and installation
  order.
- Shared dependencies include Ninja, Wheel, Requests, Packaging, NumPy,
  PyTest, PyYAML, Tinygrad, Helion, CUTLASS DSL 4.5.2, CUDA Core/Python, CUDA
  Tile 1.4.0, nvMath Python 0.9.0, MathDx cu13 0.3.2.6, and CUDA Toolkit Python
  packages 13.0.2.
- After changing either manifest or the local runtime, rerun
  `bash bin/install.sh`. If `bin/verify_environment.py` reports drift, rebuild
  the stale environment before collecting comparable timings.
- Modal is an authorized fallback when credentials are available and no
  compatible local GPU is accessible, or sustained competing use would make
  local measurements unreliable. Because the default Modal GPU may differ
  from the local RTX PRO 6000, validate and benchmark a comparison on one
  backend and record the backend with the trial; do not compare timings across
  GPU models.

## Harness execution

- Build under the experiment's CPU and memory slice:

  ```bash
  autocuda run slice --data-dir "$DATA_DIR" --tag "$TAG" -- \
    bash harness/build.sh <set>/<problem>
  ```

- Serialize GPU work for validation, benchmarks, and profiles:

  ```bash
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/benchmark.sh <set>/<problem>
  ```

- Use an absolute shared `DATA_DIR` from worker worktrees and wrap each harness
  script exactly once. `kernelguard` is installed in the local control runtime,
  so local runs use the same policy gate before evaluation.

## Timeouts

Timeouts are problem-dependent. Measure build, validation, benchmark, and
profiling durations for the selected problem, then allow enough margin for cold
compilation and slower cases.

- Fresh custom-kernel builds can take tens of seconds; cached or Python-only
  builds can be much faster.
- Validation and benchmark duration scales with the problem's cases. Record
  measured values in experiment data rather than treating them as host facts.
- Nsight Systems captures are normally much faster than focused Nsight Compute
  captures. Nsight Compute may replay kernels for minutes, so pass a narrow
  `--kernel-name` expression.
- Leaderboard submission is variable and can time out. When a run requires it,
  allow at least 1200 seconds and retry a timeout up to three times. A timeout
  is inconclusive; a parsed `verdict=REJECTED` is a real failure.

## Profiling

The measured workers provide Nsight Systems 2026.1.3 and Nsight Compute
2026.2.1. Use the harness profiling scripts described in `autocuda/layout.md`.
They profile one benchmark shape inside the CUDA profiler range emitted by
`eval.py`, excluding input generation, warmup, L2 flushing, and repeat-level
correctness checks.

- **Nsight Systems:** set `NSYS_OUT` when the `.nsys-rep` must be retained.

  ```bash
  NSYS_OUT="$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-rep" \
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_nsys.sh <set>/<problem> "<shape-spec>" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
  ```

- **Nsight Compute:** use the passwordless `sudo -n` path provided by the
  worker and always pass the optional kernel-name filter so unrelated kernels
  are not replayed repeatedly.

  ```bash
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_ncu.sh <set>/<problem> "<shape-spec>" \
      "regex:<kernel-pattern>" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
  ```
