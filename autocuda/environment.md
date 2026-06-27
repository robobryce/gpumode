# Environment

Per-machine half of the project description (the machine-agnostic half is the
committed `autocuda/layout.md`). Written for the **`linalg/eigh_py`** target
(batched real symmetric eigendecomposition, `custom_kernel(A) -> (Q, L)`).

> **Rewritten 2026-06-27 for THIS host; re-verified 2026-06-27 13:12 at the
> start of run `2026-06-27-13-12-42`.** The prior copy was inherited from a
> different node and was wrong about the hardware/toolchain. Every fact below is
> freshly probed on this machine. Notable corrections vs the old file: this host
> has **1 B200, not 2** (no GPU round-robin); the nvcc **toolkit is 13.3** while
> torch's bundled CUDA **runtime is 13.0** (`torch.version.cuda == "13.0"`,
> `2.12.1+cu130`); **30 CPUs (Intel Xeon 6960P) / 180 GB, not 60 / 334**; the
> `~/.local/bin/ninja` symlink fix is **obsolete here** (ninja is already
> available and torch finds it). Baseline commit is **`424ba1894` (current
> `main` tip)**; the freshly measured baseline geomean is **≈ 56,255 µs** this
> run (≈ 56,314 last probe; both well within run noise — the old inherited file's
> ≈ 255,000 µs was that other node).

## GPU and environment

- **Host GPU:** **1× NVIDIA B200** (183,359 MiB / ~179 GB), compute capability
  **10.0 (sm_100)**, Blackwell. `nvidia-smi -L` shows exactly one device
  (`GPU 0`), and `torch.cuda.device_count() == 1`. There is **no second GPU and
  no even/odd round-robin** — `autocuda run` writes `gpu-count=1`, every
  benchmark/profile serializes on `gpu-0`, and `harness/env.sh` pins
  `CUDA_VISIBLE_DEVICES=0`.
- **Driver / CUDA:** NVIDIA driver **580.126.09**; CUDA toolkit **13.3**
  (`/usr/local/cuda` → `cuda-13.3`, `nvcc V13.3.33`, built Apr 2026). Also
  installed but unused: `/usr/local/cuda-12.8`, `cuda-13`. (The old file said
  13.0 / `nvcc V13.0.88` — wrong for this host.)
- **PyTorch:** `2.12.1+cu130` in the workspace venv `/home/shadeform/gpumode/.venv`
  (path recorded in `~/.config/gpumode/gpumode.env` as `GPUMODE_VENV_PYTHON`).
  Python **3.12.13**. `torch.cuda.is_available()` → True, device "NVIDIA B200",
  capability `(10, 0)`.
- **Custom-CUDA target arch:** compile submissions for **Blackwell sm_100**.
  Plain `-gencode arch=compute_100,code=sm_100` works for WMMA / `mma.sync`; the
  **`a` suffix (`compute_100a,code=sm_100a`) is REQUIRED** for any `tcgen05` /
  TMA / thread-block-cluster PTX (set `TORCH_CUDA_ARCH_LIST="10.0a"` for
  `load_inline`). B200 FP64 is **weak** (~37–40 TFLOPS tensor; ~half of H100) —
  **avoid FP64 on the hot path**. Dense tensor-core peaks (NOT the 2× sparse
  numbers): **FP16/BF16 ≈ 2250 TFLOPS, TF32 ≈ 1100, FP8 ≈ 4500, FP4 ≈ 9000**;
  cuBLAS BF16x9 emulation gives ~FP32-accurate GEMM at up to ~3× native FP32.
  148 SMs, ~228 KB opt-in SMEM/block (a 512×512 FP32 tile is 1 MB — does NOT
  fit, you tile), 192 MB L2, ~8 TB/s HBM3e. `tl.dot` on Triton ≥3.3 emits
  `tcgen05` automatically; FP32 inputs route to TF32 tensor cores by default.
- **eigh tolerances admit reduced precision.** The checker
  (`reference.py::check_implementation`) gates three *normwise* residuals
  relative to the FP64 matrix L1 norm: eigen-equation `‖AQ−Qdiag(L)‖₁`
  (rtol `200·n·eps32`), reconstruction `‖Qdiag(L)Qᵀ−A‖₁` (`400·n·eps32`),
  orthogonality `‖QᵀQ−I‖₁` (`100·n·eps32`, **max over the batch**), plus an
  ascending-sort gate. Relative budget grows with n: ~2 correct digits at
  n=512, ~1.3 at n=4096. **Measured on this host, the bare `torch.linalg.eigh`
  baseline sits 100–1600× inside the gate** (validate prints e.g.
  `scaled_eigen_residual≈0.05…0.25`, `scaled_orthogonality≈0.5…0.65`, against a
  gate threshold of ~100). TF32/FP16 internal math is clearly admissible; BF16
  is fine for n≥176 well-conditioned cases. Standard recipe: do the bulk O(n³)
  work in low precision but **emit `Q` through an FP32 orthonormalization**
  (Newton–Schulz / Gram–Schmidt — cheapest way to pass the max-over-batch ORTH
  gate) and **compute `L` via an FP32 Rayleigh quotient `diag(QᵀAQ)`** (an
  O(ε²)-accurate eigenvalue from an O(ε) eigenvector). Power-of-two prescale
  `A`→O(1) for the high/low-magnitude LAPACK *test* cases (FP16 would
  overflow/underflow; those are in `tests:`, not `benchmarks:`).
- **Toolchain caps:** **30 CPUs, 180 GB RAM** on the host (the old file's
  60 / 334 was the other node). `harness/env.sh` exports `MAX_JOBS=3` per worker
  (3 workers × 3 = 9 ≤ 29). Per-worker limits in
  `autocuda/worker-limits.csv`: **8 CPUs, 48 GB RAM** each
  (`systemd-run --user --scope` via `autocuda run slice`).
- **✅ ninja is already usable — no fix needed on this host.** The active ninja
  is the venv's **`1.13.0`** at `/home/shadeform/gpumode/.venv/bin/ninja` (a
  system `ninja 1.11` also exists at `/usr/bin/ninja`, earlier on some PATHs, but
  both work); `torch.utils.cpp_extension.is_ninja_available()` → **True**. The old
  file's "REQUIRED `ln -sf … ~/.local/bin/ninja`" step does **not** apply here
  (that symlink does not exist and is unnecessary — `load_inline` compiles fine).
- **Triton:** PyTorch 2.12 bundles **Triton 3.7.0** (matured Blackwell support).
  `tl.dot` lowers to 5th-gen tensor-core `tcgen05` MMA **automatically** on
  sm_100 — no PTX needed. FP32×FP32 `tl.dot` is routed by `input_precision=`:
  default **`"tf32"`** (single TF32 pass, ~1100 TFLOPS, truncates without
  rounding), **`"tf32x3"`** (3×TF32, ≈FP32 accuracy at ~1/3 the TF32 rate — the
  accuracy-safe choice for rotations/orthogonalization), or `"ieee"` (true FP32
  in **software, no tensor cores** — slow, reference only). Accumulation is always
  FP32 in TMEM. `allow_tf32=` is deprecated (maps to `input_precision="tf32"`).
  For batched work, the idiom is **one Triton program per matrix**
  (`pid=tl.program_id(0)`, offset pointers by `pid*stride_batch`), autotuned
  128–256 tiles, persistent kernels when batch ≫ 148 SMs. cuBLAS **BF16x9**
  emulated-FP32 (`torch.bmm`/cublasLt) gives ≈FP32-accurate GEMM at ~3× native
  FP32 — the high-accuracy GEMM baseline to beat for the O(n³) interior.

## Baseline

The unmodified `submission.py` is `torch.linalg.eigh` (cuSOLVER). **Baseline
commit `424ba1894`** = current `origin/main` tip (`main` + the eigh
profiler-range fix `a6939aedc` + the `submit.sh` verdict-parsing commit). **Fork
workers off `424ba1894` or later** — a worktree forked before `a6939aedc` lacks
the eigh `eval.py` `torch.cuda.profiler` range and its `ncu`/`nsys` captures
record nothing. (Note: the current `eval-profiler-capture-range` branch /
`6007add16` predates the eigh range and is NOT a valid profiling base; this run
is based on `main`.)

- **Local benchmark geomean (`harness/benchmark.sh`, 13 shapes): ≈ 56,255 µs**
  (freshly measured at run `2026-06-27-13-12-42` start; 56,314 the prior probe —
  within run noise). Treat geomean deltas below ~1% as noise (eval.py's internal
  `err/mean < 0.1%` early-exit bounds per-shape variance).
- **Leaderboard gap (read 2026-06-27 13:16, `eigh`/B200):** the stock
  `torch.linalg.eigh` baseline lands ~#12 of 13. The frontier cluster is **#1 `az`
  ≈ 33,954 µs, #2 ≈ 34,982, #3 ≈ 35,579, #4 ≈ 38,305, #5 ≈ 40,616** — a **~40%
  speedup (1.66×) over baseline at the top**. Everyone at 53–56k µs is essentially
  running stock eigh, so the frontier reflects a *fundamentally different*
  (batched / reduced-precision / tensor-core) approach, not micro-tuning. #4's
  file is literally `triton_diagonal_fast_path.py` (a copy ships under
  `problems/linalg/eigh_py/submissions/`) — a hint that special-casing structured
  shapes plus a Triton fast path is on the frontier line.
- **Measured wall-clock (this B200, via `autocuda run`):** build (pure-PyTorch
  import) ~1.7 s; validate (39 test shapes, no guards) ~6.6 s; full
  baseline benchmark (13 shapes) **~29 s**; nsys one-shape profile ~15–20 s.

Per-shape baseline geomean inputs (one full `benchmark.sh`; B = batch). **The
geomean weights every shape equally (1/13)** — a 2× on tiny n=32 counts as much
as 2× on n=512 B=640:

| # | spec | µs (this shape) |
|---|---|---|
| 0 | batch 20, n 32, cond 1 | 160.1 |
| 1 | batch 40, n 176, cond 1 | 5,808.1 |
| 2 | batch 40, n 352, cond 1 | 20,871.9 |
| 3 | batch 640, n 512, cond 2 (dense) | 170,951.2 |
| 4 | batch 60, n 1024, cond 2 (dense) | 106,305.1 |
| 5 | batch 8, n 2048, cond 1 | 188,142.5 |
| 6 | batch 640, n 512, cond 2, **mixed** | 163,213.3 |
| 7 | batch 60, n 1024, cond 2, **mixed** | 105,589.3 |
| 8 | batch 640, n 512, cond 0, **rankdef** | 166,505.0 |
| 9 | batch 640, n 512, cond 0, **clustered** | 138,048.3 |
| 10 | batch 60, n 1024, cond 0, **nearrank** | 105,125.6 |
| 11 | batch 640, n 512, cond 0, **lapack_dense_even_spectrum** | 202,662.0 |
| 12 | batch 60, n 1024, cond 0, **lapack_dense_geometric_spectrum** | 102,299.4 |

The geomean is spread across the **n=512 B=640 group (5 shapes: 3/6/8/9/11)**,
the **n=1024 B=60 group (4 shapes: 4/7/10/12)**, and **n=2048 B=8 (shape 5)** —
but in **geomean space the launch-bound small shapes (0:n=32, 1:n=176, 2:n=352)
carry 3/13 of the score**, so a fast many-small-matrices path matters as much as
the big-n path. eval.py replicates each input to ~256 MB / (B·n²·4) copies
(cap 50): n=32 → 50 copies, n=176 → 50, n=352 → 13; all n≥512 shapes are
replication-capped to 1–2 copies.

### Baseline bottleneck (profile-confirmed on this host)

`torch.linalg.eigh` does **not** use a batched path for n>32: it loops cuSOLVER
**`syevd` per matrix**. nsys on the n=512 B=640 shape
(`profiles/<tag>/baseline-n512-424ba1894.nsys-rep`) shows a classic two-stage
symmetric solve, **all SIMT / CUDA-core, ZERO tensor-core**:

- `sytrd4_cta` Householder **tridiagonalization** — 21.6%
- `gemvx::kernel` + `cuds_symv_lo_direct` + `gemv2T` + `gemv2N` **matrix-VECTOR**
  products — ~42% combined (memory/latency bound; panel updates are GEMV, not GEMM)
- `laed3/laed4/laed2` **divide-and-conquer** on the tridiagonal — ~20%
- only stray `cutlass3x_sm100_simt_sgemm` (SIMT SGEMM, *not* tensor-core) ~5%

At **n≤32** the baseline instead takes cuSOLVER's true batched-Jacobi path
(`syevj`/`batch_parallel_jacobi`) — already fast and genuinely batched, so
shape 0 has the least big-ratio headroom (but still 1/13 of the geomean).

**Implication:** the baseline is latency-bound serial cuSOLVER with no
tensor-core use, so almost any genuinely *batched* kernel — and especially one
that routes the O(n³) work onto FP16/BF16/TF32 tensor cores — has large
headroom. Research-favored directions: batched one-sided/block Jacobi
(SMEM-resident at n≤512, tensor-core block updates at n≥1024), genuinely-batched
tridiag→D&C, GEMM-heavy spectral divide-and-conquer (Newton–Schulz / QDWH /
Zolo-eig polar decomposition). Finish any low-precision interior with one FP32
Newton–Schulz / Gram–Schmidt reorthogonalization of `Q` and an FP32 Rayleigh
quotient for `L`. **The known input-reuse hole** (`eval.py` reuses the same
`data_list` objects across timed repeats and only rechecks the cached output) is
a separate, harness-level lever this red-team run is also hunting.

## Timeouts

Measured wall-clock on this host (one B200, via `autocuda run`):

- **Build (pure-PyTorch import):** ~1.7 s. A fresh custom-CUDA `load_inline`
  (sm_100) compile is tens of seconds; cached rebuild (unchanged source) ~1 s
  (content-hash keyed in per-worktree `problems/linalg/eigh_py/.torch_ext`).
- **Validate (39 test shapes, no problem-specific guards):** ~6.6 s for the
  `torch.linalg.eigh` baseline. A custom kernel that runs the LAPACK
  high/low-magnitude and rankdef/clustered test cases will take longer; budget
  up to ~240 s (`test_timeout` in task.yml is 240 s).
- **Benchmark (13 shapes, geomean):** ~29 s for the baseline. A faster custom
  kernel runs shorter; a slow one can run longer; budget up to ~300 s
  (`benchmark_timeout` is 480 s).
- **nsys profile (one shape, capture-range):** ~15–20 s wall.
- **ncu profile (one shape, `--set full`):** tens of seconds to minutes
  depending on kernel count and the `--kernel-name` filter; budget ≥600 s and
  always pass a focusing regex.
- **Leaderboard submit (`harness/submit.sh`):** highly variable and
  intermittently times out — budget ≥600 s and **retry on timeout up to 3×**
  (a timeout is inconclusive; a parsed `verdict=REJECTED` is a real failure).
  `harness/submit.sh` parses the popcorn-cli JSON result as the authoritative
  verdict and prints `submission_id` + `public_score`.

## Profiling

Profile through the harness scripts (see `layout.md` § Profiling). Both run
`eval.py benchmark` on one shape under the `torch.cuda.profiler` range that
eigh_py/eval.py wraps its timed `custom_kernel` launches in (committed in
`a6939aedc`, present on `main`), so the capture holds only the submission's
kernels — not input generation, warmup, the L2 flush, or the per-repeat FP64
checker. **A worktree forked before `a6939aedc` captures nothing.**

- **nsys** (timeline, no sudo) — default shape is the first benchmark line; pass
  a `<shape-spec>` (a `task.yml` benchmarks line) to pick another. Set
  `NSYS_OUT` to keep the `.nsys-rep`:
  ```
  NSYS_OUT="$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-rep" \
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_nsys.sh linalg/eigh_py "batch: 640; n: 512; cond: 2; seed: 1029" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
  ```
- **ncu** (per-kernel; needs root — **passwordless `sudo -n` works on this host**,
  `ncu` at `/usr/local/cuda/bin/ncu`). The optional 3rd arg is an
  `--kernel-name` regex — **always pass one** or ncu replays every kernel for
  ~40 passes each:
  ```
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    bash harness/profile_ncu.sh linalg/eigh_py "batch: 640; n: 512; cond: 2; seed: 1029" "regex:sytrd|gemv|laed" \
      > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
  ```
  Kernel names to filter on depend on the path: **n≤32** baseline → `syevj`,
  `jacobi`; **n>32** baseline → `sytrd`, `symv`, `gemv`, `laed`. For a custom
  kernel, profile it once on iteration 1 to learn its kernel names, then filter
  to the dominant one.
