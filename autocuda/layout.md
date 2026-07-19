# Layout

## Summary

This is a fork of [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) wired for optimizing [GPU MODE](https://www.gpumode.com/) leaderboard kernels with autocuda. The official problems already live under `problems/<set>/<problem>/` — each a single editable `submission.py` (`custom_kernel(data) -> output`) beside its frozen `reference.py` / `task.yml`. The `eval.py` / `utils.py` they run against may live **either** at the set root (shared by the set's problems, e.g. `pmpp_v2`) **or** in the problem dir itself (e.g. `linalg/qr_py`, `bioml/trimul`), and a problem may even borrow `utils.py` from another set (`qr_py` uses `pmpp_v2/utils.py`). The harness resolves both from each problem's `task.yml` `files:` manifest — the same flattening KernelBot does — so the layout of any given problem is discovered, not assumed. autocuda's `optimize-tree` (or `optimize-simple` / `optimize-hill`) edits `submission.py` **in place** — nothing is copied or scaffolded — then **builds → validates → benchmarks** it. The benchmark metric is the harness's per-shape timing reduced to the geomean the leaderboard ranks by, so every trial is measured exactly the way the leaderboard measures it and a wrong kernel can never be scored.

Pick the target problem by passing its `<set>/<problem>` path — this is the single token that selects the target. It is both the `benchmark=` argument you scope the run with **and** the first positional argument to every `harness/` script (which use it to locate the editable file and put the right dirs on `PYTHONPATH`). There is no environment variable to export: the `harness/` scripts take the path as an argument, so it travels with each command instead of relying on shell state that an agent harness does not preserve between separate command invocations. There is **one** autocuda data dir for the whole repo (`autocuda/` at the root) holding the run's logs, schema, worktrees, and dashboard. The machine-agnostic project description is this `layout.md` (committed); the per-machine half — GPU, toolchain paths, measured timings/noise, profiler invocations — is `autocuda/environment.md`, which `/autocuda:discover` writes per host (run it once on a fresh machine; with this `layout.md` present it only sets up and records the environment).

Optimize a problem — pass its `<set>/<problem>` path as `benchmark=` (the one token that selects the target; nothing to export, nothing to `cd` into):

```bash
/autocuda:optimize-tree workers=4 benchmark=<set>/<problem> tag-suffix=<problem>
```

### ⚠️ MANDATORY OVERRIDE: MODAL TESTS AND BENCHMARKS USE `autocuda run slice`

**Explicit override of the autocuda skill's default guidance:** this project runs validation and benchmarking on Modal, not on a local GPU. Therefore **all tests and benchmarks MUST use `autocuda run slice`, never `autocuda run exclusive`.** The remote Modal GPU consumes no contended local GPU resource; `exclusive` would unnecessarily serialize workers and contradict this layout's project-specific execution contract.

- **Build:** `autocuda run slice --data-dir "$DATA_DIR" -- <cmd>`
- **Validate and Benchmark (remote Modal GPU):** `autocuda run slice --data-dir "$DATA_DIR" -- bash harness/modal.sh <validate|benchmark> <set>/<problem>`
- **Profile (run-specific no-GPU exception):** no local GPU is available for this run, so local profiling is waived; do not invoke the local profiler wrappers or `autocuda run exclusive` for profiling. Hosted profiling through `popcorn-cli --profile-brev` is allowed, subject to the strict rate limit in [Profiling](#profiling).

`autocuda run slice` still provides each worker's local CPU/memory allocation while allowing independent Modal jobs to overlap. `harness/modal.sh` uploads the current worktree, so each worker evaluates its own edited `submission.py`. Its remote image mirrors the production `gpu-mode/kernelbot` Modal dependency definition at the pinned commit recorded in `harness/modal_runner.py`; do not add convenience packages to that image, because doing so can hide leaderboard import failures. Local-only kernelguard/portability checks run before Modal is launched. Set `MODAL_GPU` to the Modal NVIDIA GPU type required by the problem (default `B200`); authentication is the normal Modal CLI configuration. The supplied CUDA image supports NVIDIA/CUDA problem sets; AMD/ROCm and multi-GPU problems require a different remote image/runner and are not supported by this wrapper. `autocuda run` preserves the caller's working directory, and the harness resolves the target from the `<set>/<problem>` argument rather than the cwd.

**Wrap each `harness/` script exactly once — never nest `autocuda run`.** Wrap the script *you* invoke; harness scripts do their own work directly and never call `autocuda run`. For remote validation/benchmarking, wrap `harness/modal.sh`; the local wrapper invokes Modal, and the remote function invokes `validate.sh` or `benchmark.sh` without another autocuda wrapper.

## Editable files

- **`problems/<set>/<problem>/submission.py`** — the ONLY file you may edit. It must keep the contract: a module-level `custom_kernel(data: input_t) -> output_t`. The input tensors are already on the GPU (see the problem's `reference.py::generate_input` for exact shapes/dtypes) and any output buffer is preallocated. You may add module-level code (e.g. a `load_inline`/`load` call that compiles a CUDA kernel once at import time) and helper functions, but keep everything in this one file (leaderboard submissions are single-file). Do NOT change the `custom_kernel` name or signature.

## Read-only files

Frozen GPU MODE harness — DO NOT modify. These define correctness and timing exactly as the leaderboard does. Per the selected problem:

- **`eval.py`** — the official KernelBot eval harness. Modes `test` / `benchmark` / `leaderboard` / `profile`; reads a spec file, runs `custom_kernel` in a spawned subprocess, writes `key: value` results to the fd named by `POPCORN_FD`. Times with CUDA events, clears L2 between repeats, runs an obligatory correctness check before timing. **Its location varies by problem** — shared at the set root (`problems/<set>/eval.py`, e.g. `pmpp_v2`) or problem-local (`problems/<set>/<problem>/eval.py`, e.g. `linalg/qr_py`). The harness resolves the right one from `task.yml`'s `files:` manifest (`bin/gen_specs.py --file-source eval.py`), so never hardcode the set root.
- **`utils.py`** — checkers (`verbose_allclose` / `verbose_allequal`), seeding, `clear_l2_cache`. Same dual location as `eval.py`, and may be borrowed from another set (`linalg/qr_py` → `problems/pmpp_v2/utils.py`); resolved the same way (`--file-source utils.py`).
- **`problems/<set>/<problem>/reference.py`** — `generate_input(...)` and the `check_implementation` ground truth. **`task.py`** — input/output type schema. **`task.yml`** — the official problem spec: the canonical `tests:` / `benchmarks:` shapes + leaderboard timeouts **and** the `files:` manifest that maps each runtime file (`eval.py`, `utils.py`, …) to its `source` path. The `harness/` scripts render its shapes into eval.py spec files and resolve those file locations on the fly via `bin/gen_specs.py`. Problem-specific `guards/*.sh` are also frozen validation infrastructure: QR v2 carries differential/invariance/off-grid guards, while Eigh and Cholesky carry bounded-time fresh-seed input-variation guards over published shapes and their ranked/official generator families.
- **`harness/env.sh`, `build.sh`, `validate.sh`, `benchmark.sh`, `modal.sh`, `modal_runner.py`, `profile_ncu.sh`, `submit.sh`** and **`bin/gen_specs.py`** — the autocuda↔GPU-MODE bridge. `modal.sh` uploads the current worktree and invokes validation/benchmarking on Modal; `env.sh` then resolves `eval.py` / `utils.py` per problem from the `files:` manifest (via `gen_specs.py --file-source`) and puts their dirs on `PYTHONPATH`, so the same scripts drive both the set-root and problem-local layouts unchanged. Treat as read-only infrastructure.

## Build

GPU MODE submissions compile their CUDA at *runtime* (via `load_inline`/`load` at import), so "build" here means: import `submission.py` in a fresh process to trigger that compilation now, surfacing any nvcc/compile error as a `build_error` instead of at benchmark time. A pure-PyTorch submission imports instantly. The compiled extension caches in `$TORCH_EXTENSIONS_DIR` (per-worktree: `problems/<set>/<problem>/.torch_ext`), keyed by source-content hash, so the import that validate/benchmark trigger reuses this build.

```bash
autocuda run slice --data-dir "$DATA_DIR" -- \
  bash harness/build.sh <set>/<problem>
```

**Per-worker build concurrency.** `harness/env.sh` (sourced by `build.sh` with the `<set>/<problem>` arg) exports `MAX_JOBS=3` to cap nvcc/ninja parallelism (worst case `N_workers × 3 ≤ nproc − 1`). `load_inline` recompiles only the changed translation unit(s) + relink: a fresh CUDA-kernel compile is tens of seconds, a no-op/pure-PyTorch rebuild is ~1 s. There is no CMake/Make project to tune; ccache/sccache do not apply (nvcc is driven by torch's build). `autocuda run slice` enforces per-worker CPU/memory slices via `systemd-run --user --scope`. Host-specific build timings live in `autocuda/environment.md`.

## Validation

Runs the official harness in `test` mode over the problem's test shapes from `task.yml` (the same shapes the leaderboard's `--mode test` uses), so a local pass faithfully predicts a remote test pass.

```bash
autocuda run slice --data-dir "$DATA_DIR" -- \
  bash harness/modal.sh validate <set>/<problem>
```

A passing run exits **0** and prints `validation: PASS` (`check: pass`). Before GPU evaluation, every problem is scanned by local kernelguard plus the banned-stream gate. After official test shapes pass, `harness/validate.sh` runs every problem-local `guards/*.sh`; Eigh and Cholesky use these to catch naïve shape/call-order replay and public-test seed/profile overfitting with fresh inputs across their ranked/official generator families. Any mismatch, guard failure, crash, or compile error exits non-zero (eval.py returns `112` on a correctness mismatch) — treat as `validation_error`.

## Benchmarks

The active benchmark is the problem named by the `<set>/<problem>` token; the autocuda metric name **is** that token. `harness/modal.sh benchmark` uploads the worktree and remotely runs `harness/benchmark.sh`, which uses the problem's benchmark shapes from `task.yml`. Scope an `optimize-tree` run to it with `benchmark=<set>/<problem>` — the same token you pass to the script.

```bash
autocuda run slice --data-dir "$DATA_DIR" -- \
  bash harness/modal.sh benchmark <set>/<problem>
```

- **Command:** `autocuda run slice --data-dir "$DATA_DIR" -- bash harness/modal.sh benchmark <set>/<problem>`.
- **Metric:** the **geometric mean of the per-shape mean runtimes**, in microseconds.
- **Unit:** µs. **Direction:** **min** (lower is better — it's latency). **Precision:** 3.
- **Metric name:** the `<set>/<problem>` token — the same string you pass as `benchmark=` and to the script, so the emitted key matches the autocuda schema column with no lookup. (The GPU MODE *leaderboard* name from the set yaml is a separate identifier — distinct from the `<set>/<problem>` token — used only for `popcorn-cli submit --leaderboard`; `bin/gen_specs.py … --leaderboard` resolves it on demand.)
- **Metric extraction:** the script prints, on **stdout**, a single fenced block as the LAST thing it emits:
  ```
  ===GPUMODE_RESULT_BEGIN===
  <set>/<problem>=<value>
  ===GPUMODE_RESULT_END===
  ```
  Parse the `<set>/<problem>=<value>` line. Full eval.py output + a per-shape `shape i [spec]: <us>` breakdown go to **stderr**. A non-zero exit = correctness mismatch (eval.py aborts timing and reports an error) or crash/timeout — a `runtime_error`, **never** a metric of 0. The script refuses to emit a metric unless every shape reported `check: pass`, so a fast-but-wrong kernel cannot be scored.
- **Axes:** the benchmark shapes from `task.yml`; the geomean reduces across them.
- **Aggregation:** geomean over the shapes. Because `geomean(baseline_i)/geomean(trial_i) == geomean(baseline_i/trial_i)`, this single geomean-µs scalar with direction=min makes autocuda's baseline/trial speedup the ratio-space per-shape aggregate the worker skill prescribes, and matches the leaderboard's own geomean-of-shapes ranking.

## Profiling

### ⚠️ RUN-SPECIFIC EXCEPTION: NO LOCAL GPU

No local GPU is available for this run. This is an explicit exception to any autocuda guidance that requires local profiling: **do not run `harness/profile_nsys.sh`, `harness/profile_ncu.sh`, or any local profiler command, and do not treat the absence of a local profile as a blocker or as an invalid baseline.**

Models may instead use the hosted popcorn-cli profiling service for an occasional high-value capture:

```bash
popcorn-cli submit --no-tui --leaderboard <name> --gpu <gpu> \
  --mode profile --profile-brev --benchmark-index <index> submission.py
```

Limit hosted profiling to **at most 3 requests in any rolling hour across the entire run, not per worker**. Coordinate across workers and record each request's timestamp, commit SHA, and benchmark index in the run log. Reuse saved Nsight Compute artifacts; reserve requests for the baseline or a materially distinct promising candidate, rather than profiling every trial. Resolve `<name>` and `<gpu>` as described under [Leaderboard submission & standings (manager)](#leaderboard-submission--standings-manager).

## Cross-benchmark aggregation

One problem is optimized per run (one `optimize-tree` run, scoped with `benchmark=<set>/<problem>`), so there is no cross-benchmark combination: a run's per-trial score is its single benchmark's geomean-µs compared to baseline as `baseline/trial` (direction=min). To optimize a different problem, start a fresh run with a different `benchmark=<set>/<problem>` and `tag-suffix`.

## Dependencies

- **Modal CLI ≥ 1.1**, authenticated with `modal setup` or token environment variables. `harness/modal_runner.py` defaults to B200 and pins the production KernelBot image definition commit, including its CUDA/Python/PyTorch and package versions. Only `MODAL_GPU` is configurable; dependency overrides are intentionally unsupported so local results remain representative.
- **CUDA toolkit** with `nvcc` for local builds/profiling (the harness compiles `submission.py`'s CUDA via torch `load_inline` at import). Verified on CUDA 13.0.
- **PyTorch** matching the host GPU/CUDA (the workspace venv; `bin/install.sh` builds one and writes its path into `~/.config/gpumode/gpumode.env`), plus `pyyaml` (for `bin/gen_specs.py`) and `ninja`.
- **popcorn-cli** — GPU MODE leaderboard submission client (authenticated via Discord; see the `popcorn-login` skill).
- Custom CUDA in a submission must build against **stock CUB/Thrust** to be leaderboard-portable; compiling against a local CCCL checkout via `extra_include_paths` is diagnostic-only (the remote build has only stock headers).

## Log schema

Materialise the optimize schema with the target problem's single benchmark before logging the baseline. The benchmark name is the `<set>/<problem>` token (the same one you pass as `benchmark=` and to `harness/modal.sh benchmark`):

```bash
autocuda schema define optimize --data-dir "$DATA_DIR" \
  --benchmark <set>/<problem>:us:min:3
```

Every trial row carries `--metric <set>/<problem>=<value>` (geomean µs). The only allowed `N/A` is a failure-status row.

## Leaderboard submission & standings (manager)

Leaderboard submissions are **mandatory evidence**, not optional reporting. A run without a baseline leaderboard submission is **invalid from the start**. A run that finds improvements but does not submit them regularly is also **invalid**: local geomean timings alone cannot tell whether a kernel is accepted by GPU MODE, whether it is considered cheating/reward-hacking by the remote harness, or what its real leaderboard performance is.

Automatic submission is authorized for this workspace. Do not ask the operator before submitting, and do not continue optimizing until the required submission for the current phase has either succeeded or failed with an explicit, logged infrastructure/authentication error.

Submission can be flaky. A run that comes back **failed** is a real failure — treat it as one. A run that **times out** is not conclusive: retry it, up to 3 times. If all 3 time out, treat that as a real failure.

- **At baseline setup:** after the baseline passes validation, is benchmarked, and is logged with `autocuda log optimize-tree baseline`, immediately submit that exact baseline `submission.py` with `popcorn-cli submit --no-tui --leaderboard <name> --gpu <gpu> --mode leaderboard submission.py`. Local profiling is waived for this run; hosted profiling is optional and subject to the run-wide rate limit above. If the baseline leaderboard submission is missing, the entire optimize run is invalid and workers must not be launched.
- **As the run improves:** each time the best safe committed kernel meaningfully improves over the last submitted kernel, submit that exact committed `submission.py` with `--mode leaderboard` before treating the improvement as real. Do not compare local candidates as final apples-to-apples results without corresponding leaderboard submissions.
- **Before final selection:** the chosen final candidate must have a successful leaderboard submission. If the fastest local candidate was never submitted, or was rejected remotely, it is not the final candidate.
- **Submission metadata:** resolve `<name>` with `bin/gen_specs.py problems/<set>/<problem>/task.yml --leaderboard` and `<gpu>` with `bin/gen_specs.py problems/<set>/<problem>/task.yml --gpus` (choose the token matching the host GPU). The leaderboard name is **not** the autocuda metric token.
- **Submission records:** record the submission attempt, command, commit SHA, local metric, and returned leaderboard score/rank in the manager log or a run note. If `popcorn-cli` authentication or service availability prevents submission, log that failure explicitly and treat the run as blocked/invalid for leaderboard comparison until it is fixed.
- **Check the standings — know your gap to #1:** `popcorn-cli` returns your own accepted score but **not a rank or a position** — so after every submission, read the public standings to get the rank the bullet above asks for and, more importantly, **how far you are from the #1 slot**. The leaderboard ranks by the same geomean-of-shapes the local benchmark uses, so this is directly comparable to your metric. Use the `leaderboard-rankings` skill (stdlib-only, no auth, no submission), keyed by the same leaderboard `<name>`:
  - `.claude/skills/leaderboard-rankings/scripts/rankings.py --gpu <gpu> --user <you>` prints, per active problem on the GPU, the **#1 holder, their µs, your rank, and an explicit `gap→#1` percentage** — the rank/gap to record. `<you>` is the leaderboard `user_name` `popcorn-cli` submits as (set `$GPUMODE_USER`/`$GPUMODE_GPU` once and omit the flags).
  - `.claude/skills/leaderboard-rankings/scripts/rankings.py --problem <name> --gpu <gpu> --user <you> --top 10` prints that one problem's full table — every top entry's µs **and `file_name`**, your row marked — so you can see how far above you #1 sits and what approach they took.
  Let the gap **steer strategy:** a gap within run noise means you're at the frontier and the game is micro-optimization; a gap of multiples means #1 is reaching that score with a fundamentally different approach your current line cannot match — point some workers' next briefs at algorithmic departures (the leader's `file_name` often hints at the angle) instead of grinding the same kernel.
- **Public posting:** these submissions post to the **public** leaderboard by design; do not gate them on operator confirmation. Reading standings needs no auth and no submission.

Baseline submission checklist (run from the repo root, with the baseline commit checked out):

```bash
TASK=<set>/<problem>
BASELINE=$(git rev-parse HEAD)
bash harness/submit.sh "$TASK"
echo "baseline leaderboard submission recorded for $BASELINE"
```

`harness/submit.sh` resolves the leaderboard name and supported GPU token from `task.yml`, honors `GPUMODE_GPU=<token>` when the automatic GPU-name match is ambiguous, runs `popcorn-cli submit --no-tui --mode leaderboard`, then lists recent submissions for that leaderboard. Use the same helper for every meaningful improvement and final candidate.
