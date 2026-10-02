# autocuda layout — linalg/qr_v2 (local-only benchmarking)

> Machine-agnostic description of the GPU MODE / KernelBot `linalg/qr_v2`
> problem, wired for **local-only** benchmarking on an RTX PRO 6000 — i.e. the
> KernelBot eval harness is driven directly on a local GPU and nothing is ever
> submitted to the public leaderboard. The machine-specific half (GPU model,
> measured timings, noise, profiler invocations) lives in `environment.md`,
> which is generated per-host by `/autocuda:discover` in environment-only mode
> and is **not** committed.

## Summary

`linalg/qr_v2` is a GPU MODE / KernelBot competition kernel: **batched square
compact-Householder QR factorization**. The input `A` is a `batch × n × n`
`float32` CUDA tensor; the kernel must return `(H, tau)` in the exact compact
convention of `torch.geqrf(A)` — `H` holds `R` in its upper triangle and the
Householder vectors below the diagonal, `tau` holds the `batch × n` reflector
coefficients. The single thing being optimized is
`custom_kernel(data)` in **`submission.py`**; everything else (the reference,
the checker, the timing harness) is fixed. Correctness is a hard gate measured
in FP64 against the FP32 `geqrf` contract (LAPACK-style factor residual +
orthogonality of `Q`); among correct submissions, ranking is by **runtime,
geometric-mean across the benchmark set (lower is better)**.

The reference and the starter `submission.py` both just call `torch.geqrf`, so
the starter is correct at a 1.0× baseline; the optimization target is to beat
`torch.geqrf`'s runtime on the benchmark shapes (batched `512×512` is the
headline case, with `1024`/`2048`/`4096` and a spread of ill-conditioned and
heterogeneous `mixed` batches) while still passing every correctness gate.

This layout drives the eval harness **locally** through `run_local.py` (see
**Validation** / **Benchmarks**); there is no `popcorn-cli` submission step and
no network access. To run an actual leaderboard submission you would use the
KernelBot Discord bot / `popcorn-cli` instead — explicitly out of scope here.

## Editable files

- **`submission.py`** — the one and only optimization target. Must define:

  ```python
  def custom_kernel(data: input_t) -> output_t:
      # data: torch.Tensor, shape (batch, n, n), dtype float32, on CUDA
      # returns: (H, tau)
      #   H:   torch.Tensor (batch, n, n) float32, R in upper triangle +
      #        Householder vectors below the diagonal  (geqrf convention)
      #   tau: torch.Tensor (batch, n)    float32, reflector coefficients
  ```

  Interface contract enforced by `reference.check_implementation`:
  - `H` must be `(batch, n, n)` `float32` on the same device as `data`;
    `tau` must be `(batch, n)` `float32` on the same device; both finite.
  - The checker materializes `Q = torch.linalg.householder_product(H, tau)`
    and `R = triu(H)`, then gates on the LAPACK-style factor residual
    `‖R − Qᵀ·A‖₁` with `rtol = 20·n·eps32` and on orthogonality
    `‖QᵀQ − I‖₁` with `rtol = 100·n·eps32`. Triangular leakage in `Qᵀ·A` is
    reported as a diagnostic and is implied by the factor residual.
  - Low-bit (FP16/FP8/NVFP4) work is allowed **only** as an internal strategy;
    the returned factors must be FP32 and satisfy the FP32 QR invariants. The
    `mixed` and fully-ill-conditioned benchmark batches exist specifically to
    defeat "inspect a few matrices, route the whole batch to a
    well-conditioned-only fast path" shortcuts — each matrix must be factored
    on its own merits.

  `submission.py` may pull in whatever it needs to go fast: pure PyTorch,
  `torch.compile`, Triton, or inline CUDA via `torch.utils.cpp_extension`
  (`load_inline`). Any such kernels are compiled lazily on first call (see
  **Build**).

- **New helper modules you add next to `submission.py`** (e.g. a `_kernels.py`
  it imports) are also editable. Keep `submission.py` importable as a flat
  module — the harness stages it into a flat working directory alongside
  `task.py`, `reference.py`, `utils.py`, and `eval.py`.

## Read-only files

These define the problem and the scoring; **do not modify** them (modifying
them is the kernel-competition equivalent of cheating, and locally it just
invalidates your numbers):

- **`eval.py`** — the KernelBot popcorn eval harness. Reads a `POPCORN_FD`
  file descriptor, a mode (`test` / `benchmark` / `leaderboard` / `profile`),
  and a spec file; generates inputs, runs `custom_kernel`, checks correctness,
  and (in benchmark/leaderboard mode) times it with CUDA events after an L2
  flush. `benchmark` mode re-validates every timed iteration (`recheck=True`).
- **`reference.py`** — `generate_input` (all the conditioning cases: `dense`,
  `upper`, `rankdef`, `nearrank`, `clustered`, `band`, `rowscale`,
  `nearcollinear`, and the heterogeneous `mixed` batch), the `torch.geqrf`
  reference path, and `check_implementation` (the FP64 correctness gate).
- **`task.py`** — the `TestSpec` schema (`batch`, `n`, `cond`, `seed`, and
  optional `case`).
- **`task.yml`** — the problem manifest: the file list, the `tests:` and
  `benchmarks:` case sets, timeouts, and (for the real leaderboard) the
  `gpus: [B200]` pin and `ranking_by: geom`. Source of truth for the spec
  files `run_local.py` generates. Read-only.
- **`../../pmpp_v2/utils.py`** — shared helpers (`set_seed`, `clear_l2_cache`
  used between timed iterations, allclose utilities). Staged in as `utils.py`.
- **`run_local.py`** — the local driver (staging + spec generation + the
  `POPCORN_FD` protocol). It is *tooling*, not a competition file, but treat
  it as read-only during optimization: the optimizer should change
  `submission.py`, not the harness that scores it.

## Build

**There is no compilation step for the project itself** — this is a `lang: py`
KernelBot problem, so the "artifact" is just `submission.py`, and an
"incremental rebuild" after an edit is free (re-import the module). The clean
vs. incremental rebuild ratio is therefore N/A, and the C/C++ build
accelerators the discover playbook normally checks do not apply:

- **ccache / sccache** — N/A (no `nvcc`/host-compiler invocation at build time;
  there is no build).
- **Ninja over Make** — N/A (no CMake/Make build graph). Note that if
  `submission.py` uses `torch.utils.cpp_extension.load_inline`, PyTorch itself
  shells out to Ninja to JIT-compile that inline extension **on first call**,
  inside the timed/validated run — that cost is paid lazily at runtime, not in
  a build step, and is cached under `~/.cache/torch_extensions` for subsequent
  runs.
- **Turning off non-essential targets** — N/A.

The only "build" worth running is a fast smoke check that the current
`submission.py` imports and exposes `custom_kernel`, plus (for inline-CUDA
submissions) a warm-up call that triggers the lazy JIT so its compile cost is
not charged to the first timed iteration. Recorded in wrapped form so it opts
into the per-GPU lock contract (a warm-up touches the GPU):

```
autocuda run slice --data-dir "$DATA_DIR" --agent-id <i> -- \
  python -c "import importlib, sys; sys.path.insert(0, '.'); \
             m = importlib.import_module('submission'); \
             assert hasattr(m, 'custom_kernel'), 'submission.custom_kernel missing'; \
             print('submission OK')"
```

(Run from the `linalg/qr_v2/` directory. For a pure-PyTorch submission this is
just an import check and need not touch the GPU; `autocuda run slice` is still
the right wrap because it yields to any exclusive benchmark on any GPU.)

**Per-worker build concurrency:** N/A — there is no parallel compile to cap, so
no `-j<K>` flag applies. A host that runs multiple optimize workers concurrently
contends only on the GPU (handled by the `autocuda run exclusive` lock during
validation/benchmarking), not on a build.

## Validation

Run the correctness gate over the full 22-case `tests:` set from `task.yml`.
Raw command (downstream skills wrap it; the wrapped form is shown too because
validation runs `custom_kernel` on the GPU):

```
# raw
python run_local.py test

# wrapped — claims a GPU and holds its lock for the run
autocuda run exclusive --data-dir "$DATA_DIR" -- \
  python run_local.py test
```

Run from `problems/linalg/qr_v2/`. `run_local.py` stages the manifest files
into a temp dir, generates the `test` spec file from `task.yml`'s `tests:`
block, and drives `eval.py test` via `POPCORN_FD`.

**A passing run** exits `0` and prints, for all 22 cases:

```
test-count: 22
test.0.spec: batch: 20; n: 32; cond: 1; seed: 53124
test.0.status: pass
...
test.21.status: pass
check: pass
```

Any `test.<i>.status: fail` (with a `test.<i>.error:` diagnostic) means the
submission is incorrect; exit code is `112`. Exit `113` = malformed spec,
`111` = `POPCORN_FD` unset (a `run_local.py` bug, not a submission bug).
Correctness is a **hard gate**: a submission that fails any test case is not a
valid candidate regardless of speed, so the optimize loop must treat a failed
`test` run as a revert.

## Benchmarks

One benchmark, `qr_v2` — the timed run over the 12-case `benchmarks:` set from
`task.yml`. This is the metric the leaderboard ranks on, measured locally.

#### qr_v2

- **Command** (run from `problems/linalg/qr_v2/`):

  ```
  # raw
  python run_local.py benchmark

  # wrapped — exclusive GPU claim + lock for the whole timed run
  autocuda run exclusive --data-dir "$DATA_DIR" -- \
    python run_local.py benchmark
  ```

  `run_local.py benchmark` generates the spec file from `task.yml`'s
  `benchmarks:` block and drives `eval.py benchmark`, which times each case
  with CUDA events, flushing the L2 between iterations (`clear_l2_cache`) and
  re-validating every timed iteration (`recheck=True`). Use
  `python run_local.py leaderboard` for the longer, more-repeats ranked-style
  rerun (still local; same 12 cases, no submission).

- **Metric** — wall-clock execution time of `custom_kernel`, per input matrix
  (the harness divides the measured batch time by the number of inputs).
- **Unit** — nanoseconds (`ns`). `eval.py` reports
  `cuda.Event.elapsed_time` (ms) × 1e6.
- **Direction** — **lower is better**.
- **Metric extraction** — parse the harness output emitted on `POPCORN_FD`
  (which `run_local.py` streams to stdout). For each case `i` in `0..11`, read
  the `benchmark.<i>.mean` line (mean ns per input; `best` / `std` / `err` are
  also reported). A run is valid only if it ends with `check: pass`; a
  `benchmark.<i>.status: fail` + `benchmark.<i>.error:` means the timed output
  diverged and the whole run is invalid (exit `112`).
- **Axes** — the 12 benchmark cases (each a `(batch, n, cond, case)` point;
  `case` defaults to `dense` when omitted). Treat the case spec as the single
  sweep axis:
  - `batch=20,  n=32,   cond=1, case=dense`
  - `batch=40,  n=176,  cond=1, case=dense`
  - `batch=40,  n=352,  cond=1, case=dense`
  - `batch=640, n=512,  cond=2, case=dense`   ← headline batched-512 case
  - `batch=60,  n=1024, cond=2, case=dense`
  - `batch=8,   n=2048, cond=1, case=dense`
  - `batch=2,   n=4096, cond=1, case=dense`
  - `batch=640, n=512,  cond=2, case=mixed`
  - `batch=60,  n=1024, cond=2, case=mixed`
  - `batch=640, n=512,  cond=0, case=rankdef`
  - `batch=640, n=512,  cond=0, case=clustered`
  - `batch=60,  n=1024, cond=0, case=nearrank`
- **Aggregation** — **geometric mean** across the 12 cases (matches
  `task.yml`'s `ranking_by: geom`). Per the autocuda contract the aggregator
  operates on per-case speedup-vs-baseline, not raw ns: for this
  lower-is-better metric convert each case to `baseline_mean / trial_mean`,
  then take the geomean. The baseline is the starter `submission.py`
  (`torch.geqrf`), so the starter aggregates to 1.0× and any improvement is
  `> 1.0×`.

## Cross-benchmark aggregation

There is a single benchmark (`qr_v2`), so no cross-benchmark combination is
needed — the run-level score **is** the `qr_v2` geometric-mean speedup defined
above. (If future work adds more benchmarks, default to the geometric mean of
each benchmark's per-baseline speedup.)

## Dependencies

Machine-agnostic requirements any host must provide to run this locally:

- **PyTorch ≥ 2.4 with CUDA support built for the target GPU.** The RTX PRO
  6000 is Blackwell (compute capability `sm_120`), so the torch build must ship
  (or JIT-compile against) `sm_120` kernels — i.e. a CUDA 12.8+ / Blackwell-
  capable wheel. A CPU-only torch build can import the reference and generate
  inputs but **cannot** run `test`/`benchmark`: `eval.py` calls
  `torch.cuda.synchronize()` and `clear_l2_cache()` (allocates a CUDA tensor),
  both of which require a live CUDA device.
- **NumPy** — used by `utils.set_seed`.
- **PyYAML** — used by `run_local.py` to parse `task.yml` (the harness itself
  does not need it).
- **CUDA toolkit (`nvcc`) + Ninja** — only if a submission JIT-compiles inline
  CUDA via `torch.utils.cpp_extension`; not needed for pure-PyTorch / Triton
  submissions. The toolkit version must match the GPU architecture
  (`sm_120` for RTX PRO 6000).
- **`autocuda` CLI** — for the `run`/`schema`/`log` wrappers referenced here.

Exact on-host paths (CUDA_HOME, the venv, driver version, the resolved
`sm_120` toolchain) are host-specific and belong in `environment.md`.

## Log schema

Materialize the per-project optimize log schema once at the start of a run,
then append one `autocuda log` row per trial:

```
autocuda schema define optimize \
    --data-dir "$DATA_DIR" \
    --benchmark qr_v2:ns:min:2
```

- `qr_v2:ns:min:2` — the single benchmark, metric in nanoseconds (`ns`),
  `min` direction (lower is better), 2 decimal places in the log. The name
  `qr_v2` matches the benchmark subsection in **Benchmarks**. The CLI computes
  the `pct_change` column against the baseline automatically.
