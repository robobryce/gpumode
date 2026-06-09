# Layout

## Summary

This is a fork of [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) wired for optimizing [GPU MODE](https://www.gpumode.com/) leaderboard kernels with autocuda. The official problems already live under `problems/<set>/<problem>/` — each a single editable `submission.py` (`custom_kernel(data) -> output`) beside its frozen `reference.py` / `task.yml`, with the shared `eval.py` / `utils.py` at the set root. autocuda's `optimize-tree` (or `optimize-simple` / `optimize-hill`) edits `submission.py` **in place** — nothing is copied or scaffolded — then **builds → validates → benchmarks** it. The benchmark metric is the harness's per-shape timing reduced to the geomean the leaderboard ranks by, so every iteration is measured exactly the way the leaderboard measures it and a wrong kernel can never be scored.

Pick the target problem by passing its `<set>/<problem>` path (e.g. `pmpp_v2/histogram_py`) — this is the single token that selects the target. It is both the `benchmark=` argument you scope the run with **and** the first positional argument to every `harness/` script (which use it to locate the editable file and put the right dirs on `PYTHONPATH`). There is no environment variable to export: the `harness/` scripts take the path as an argument, so it travels with each command instead of relying on shell state that an agent harness does not preserve between separate command invocations. There is **one** autocuda data dir for the whole repo (`autocuda/` at the root) holding the run's logs, schema, worktrees, and dashboard. The machine-agnostic project description is this `layout.md` (committed); the per-machine half — GPU, toolchain paths, measured timings/noise, profiler invocations — is `autocuda/environment.md`, which `/autocuda:discover` writes per host (run it once on a fresh machine; with this `layout.md` present it only sets up and records the environment).

Optimize a problem — pass its `<set>/<problem>` path as `benchmark=` (the one token that selects the target; nothing to export, nothing to `cd` into):

```bash
/autocuda:optimize-tree workers=4 benchmark=pmpp_v2/histogram_py tag-suffix=histogram_py
```

### ⚠️ MANDATORY: ALL COMMANDS MUST USE `autocuda run` WRAPPERS

**Every build, validate, benchmark, and profile command MUST be wrapped with `autocuda run` — NO EXCEPTIONS.**

- **Build:** `autocuda run slice --data-dir "$DATA_DIR" --agent-id <i> -- <cmd>`
- **Validate, Benchmark, Profile:** `autocuda run exclusive --data-dir "$DATA_DIR" -- <cmd>`

Direct invocations bypass the `.gpu.lock` flock in `$DATA_DIR` and produce corrupted/noisy measurements when multiple workers run concurrently. The single repo-root data dir means one lock serializes the whole fleet, even across different problems. `autocuda run` preserves the caller's working directory, and `harness/env.sh` resolves the target from the `<set>/<problem>` argument (not the cwd) to find the editable file, so a worker operates on the right `submission.py` from inside its worktree.

## Editable files

- **`problems/<set>/<problem>/submission.py`** — the ONLY file you may edit. It must keep the contract: a module-level `custom_kernel(data: input_t) -> output_t`. The input tensors are already on the GPU (see the problem's `reference.py::generate_input` for exact shapes/dtypes) and any output buffer is preallocated. You may add module-level code (e.g. a `load_inline`/`load` call that compiles a CUDA kernel once at import time) and helper functions, but keep everything in this one file (leaderboard submissions are single-file). Do NOT change the `custom_kernel` name or signature.

## Read-only files

Frozen GPU MODE harness — DO NOT modify. These define correctness and timing exactly as the leaderboard does. Per the selected problem:

- **`problems/<set>/eval.py`** — the official KernelBot eval harness (at the set root, shared by the set's problems). Modes `test` / `benchmark` / `leaderboard` / `profile`; reads a spec file, runs `custom_kernel` in a spawned subprocess, writes `key: value` results to the fd named by `POPCORN_FD`. Times with CUDA events, clears L2 between repeats, runs an obligatory correctness check before timing.
- **`problems/<set>/utils.py`** — checkers (`verbose_allclose` / `verbose_allequal`), seeding, `clear_l2_cache`.
- **`problems/<set>/<problem>/reference.py`** — `generate_input(...)` and the `check_implementation` ground truth. **`task.py`** — input/output type schema. **`task.yml`** — the official problem spec (canonical `tests:` / `benchmarks:` shapes + leaderboard timeouts); the `harness/` scripts render its shapes into eval.py spec files on the fly via `bin/gen_specs.py`.
- **`harness/env.sh`, `build.sh`, `validate.sh`, `benchmark.sh`, `profile_ncu.sh`** and **`bin/gen_specs.py`** — the autocuda↔GPU-MODE bridge. Treat as read-only infrastructure.

## Build

GPU MODE submissions compile their CUDA at *runtime* (via `load_inline`/`load` at import), so "build" here means: import `submission.py` in a fresh process to trigger that compilation now, surfacing any nvcc/compile error as a `build_error` instead of at benchmark time. A pure-PyTorch submission imports instantly. The compiled extension caches in `$TORCH_EXTENSIONS_DIR` (per-worktree: `problems/<set>/<problem>/.torch_ext`), keyed by source-content hash, so the import that validate/benchmark trigger reuses this build.

```bash
autocuda run slice --data-dir "$DATA_DIR" --agent-id <i> -- \
  bash harness/build.sh <set>/<problem>
```

**Per-worker build concurrency.** `harness/env.sh` (sourced by `build.sh` with the `<set>/<problem>` arg) exports `MAX_JOBS=3` to cap nvcc/ninja parallelism (worst case `N_workers × 3 ≤ nproc − 1`). `load_inline` recompiles only the changed translation unit(s) + relink: a fresh CUDA-kernel compile is tens of seconds, a no-op/pure-PyTorch rebuild is ~1 s. There is no CMake/Make project to tune; ccache/sccache do not apply (nvcc is driven by torch's build). `autocuda run slice` enforces per-worker CPU/memory slices via `systemd-run --user --scope`. Host-specific build timings live in `autocuda/environment.md`.

## Validation

Runs the official harness in `test` mode over the problem's test shapes from `task.yml` (the same shapes the leaderboard's `--mode test` uses), so a local pass faithfully predicts a remote test pass.

```bash
autocuda run exclusive --data-dir "$DATA_DIR" -- \
  bash harness/validate.sh <set>/<problem>
```

A passing run exits **0** and prints `validation: PASS` (`check: pass`). Any mismatch, crash, or compile error exits non-zero (eval.py returns `112` on a correctness mismatch) — treat as `validation_error`.

## Benchmarks

The active benchmark is the problem named by the `<set>/<problem>` token; the autocuda metric name **is** that token. `harness/benchmark.sh` runs the problem's benchmark shapes from `task.yml`. Scope an `optimize-tree` run to it with `benchmark=<set>/<problem>` — the same token you pass to the script.

```bash
autocuda run exclusive --data-dir "$DATA_DIR" -- \
  bash harness/benchmark.sh <set>/<problem>
```

- **Command:** `autocuda run exclusive --data-dir "$DATA_DIR" -- bash harness/benchmark.sh <set>/<problem>`.
- **Metric:** the **geometric mean of the per-shape mean runtimes**, in microseconds.
- **Unit:** µs. **Direction:** **min** (lower is better — it's latency). **Precision:** 3.
- **Metric name:** the `<set>/<problem>` token (e.g. `pmpp_v2/histogram_py`) — the same string you pass as `benchmark=` and to the script, so the emitted key matches the autocuda schema column with no lookup. (The GPU MODE *leaderboard* name from the set yaml, e.g. `histogram_v2`, is a separate identifier used only for `popcorn-cli submit --leaderboard`; `bin/gen_specs.py … --leaderboard` resolves it on demand.)
- **Metric extraction:** the script prints, on **stdout**, a single fenced block as the LAST thing it emits:
  ```
  ===GPUMODE_RESULT_BEGIN===
  <set>/<problem>=<value>
  ===GPUMODE_RESULT_END===
  ```
  Parse the `<set>/<problem>=<value>` line. Full eval.py output + a per-shape `shape i [spec]: <us>` breakdown go to **stderr**. A non-zero exit = correctness mismatch (eval.py aborts timing and reports an error) or crash/timeout — a `runtime_error`, **never** a metric of 0. The script refuses to emit a metric unless every shape reported `check: pass`, so a fast-but-wrong kernel cannot be scored.
- **Axes:** the benchmark shapes from `task.yml`; the geomean reduces across them.
- **Aggregation:** geomean over the shapes. Because `geomean(baseline_i)/geomean(trial_i) == geomean(baseline_i/trial_i)`, this single geomean-µs scalar with direction=min makes autocuda's baseline/trial speedup the ratio-space per-shape aggregate the worker skill prescribes, and matches the leaderboard's own geomean-of-shapes ranking.

## Cross-benchmark aggregation

One problem is optimized per run (one `optimize-tree` run, scoped with `benchmark=<set>/<problem>`), so there is no cross-benchmark combination: a run's per-iteration score is its single benchmark's geomean-µs compared to baseline as `baseline/trial` (direction=min). To optimize a different problem, start a fresh run with a different `benchmark=<set>/<problem>` and `tag-suffix`.

## Dependencies

- **CUDA toolkit** with `nvcc` (the harness compiles `submission.py`'s CUDA via torch `load_inline` at import). Verified on CUDA 13.0.
- **PyTorch** matching the host GPU/CUDA (the workspace venv; `bin/install.sh` builds one and writes its path into `~/.config/gpumode/gpumode.env`), plus `pyyaml` (for `bin/gen_specs.py`) and `ninja`.
- **popcorn-cli** — GPU MODE leaderboard submission client (authenticated via Discord; see the `popcorn-login` skill).
- Custom CUDA in a submission must build against **stock CUB/Thrust** to be leaderboard-portable; compiling against a local CCCL checkout via `extra_include_paths` is diagnostic-only (the remote build has only stock headers).

## Log schema

Materialise the optimize schema with the target problem's single benchmark before logging the baseline. The benchmark name is the `<set>/<problem>` token (the same one you pass as `benchmark=` and to `harness/benchmark.sh`):

```bash
autocuda schema define optimize --data-dir "$DATA_DIR" \
  --benchmark <set>/<problem>:us:min:3
```

Every iteration row carries `--metric <set>/<problem>=<value>` (geomean µs). The only allowed `N/A` is a failure-status row.

## Leaderboard submission (manager)

Submit to the public leaderboard regularly throughout a run, on your own, without pausing to ask the operator — automatic submission is authorized for this workspace:

- **At baseline setup:** when you measure the baseline (profile it and log the baseline row), also submit the baseline `submission.py` with `popcorn-cli submit --no-tui --leaderboard <name> --gpu <gpu> --mode leaderboard submission.py` (here `<name>` is the GPU MODE leaderboard name, e.g. `histogram_v2`, which `bin/gen_specs.py problems/<set>/<problem>/task.yml --leaderboard` prints — **not** the `<set>/<problem>` metric token), then read its score back (the `leaderboard-rankings` skill, or `popcorn-cli --no-tui submissions list --leaderboard <name>`). This puts the reference's public score on record up front and shows how the (noisier — see `environment.md`'s Noise) Modal ranking maps to the local geomean, so you can calibrate before trusting small deltas.
- **As the run improves:** each time the global best kernel improves meaningfully over the last submitted one, submit the new best with `--mode leaderboard`. Submit the actual best-scoring committed `submission.py`, not work-in-progress.
- These submissions post to the **public** leaderboard by design; do not gate them on operator confirmation. The `<gpu>` token must be one the problem supports (`bin/gen_specs.py … --gpus`), matched to the host.
