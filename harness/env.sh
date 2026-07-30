#!/usr/bin/env bash
# Shared environment for the autocuda <-> GPU MODE bridge.
# Sourced by build.sh / validate.sh / benchmark.sh / profile_ncu.sh.
#
# Optimization happens IN PLACE: the editable file is the problem's own
# `submission.py` under `problems/<set>/<problem>/`, and the frozen
# `eval.py` / `utils.py` (at the set root) plus `reference.py` / `task.yml`
# (in the problem dir) are read where they already live — nothing is copied.
#
# Select the target problem by passing <set>/<problem> (e.g.
# `pmpp_v2/histogram_py`) as the FIRST positional argument to the calling
# script — build.sh / validate.sh / benchmark.sh / profile_ncu.sh each forward
# their args here via `source env.sh "$@"`. This is the same <set>/<problem>
# string you scope an optimize-tree run with (`benchmark=<set>/<problem>`), so
# the one token flows from the skill argument straight to the harness with no
# environment variable to export (a bare `export` would not survive between an
# agent harness's separate shell invocations anyway). The repo root holds ONE
# autocuda data dir (`autocuda/`).
set -uo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HARNESS_DIR/.." && pwd)"

# --- 1. machine-level config -------------------------------------------------
# Written by bin/install.sh to ~/.config/gpumode/gpumode.env. Defines
# GPUMODE_VENV_PYTHON, CUDA_HOME, CUTLASS_PATH, and MATHDX_HOME.
for cfg in "${GPUMODE_ENV:-}" "$HOME/.config/gpumode/gpumode.env"; do
    if [ -n "$cfg" ] && [ -f "$cfg" ]; then source "$cfg"; break; fi
done

# --- 2. target problem -------------------------------------------------------
# First positional arg of the calling script, forwarded via `source env.sh "$@"`.
# `${1:?...}` prints this usage and exits if it is missing (works under `set -u`).
PROBLEM="${1:?usage: <script>.sh <set>/<problem> (e.g. pmpp_v2/histogram_py)}"
PROBLEM_DIR="$REPO_DIR/problems/$PROBLEM"
SET_DIR="$(dirname "$PROBLEM_DIR")"   # the set root (problems/<set>); the
                                      # default — but NOT universal — home of
                                      # eval.py / utils.py (see EVAL_PY below).
[ -f "$PROBLEM_DIR/submission.py" ] || {
    echo "no submission.py at $PROBLEM_DIR — is '$PROBLEM' the right <set>/<problem>?" >&2; exit 1; }

# --- derived / defaults ------------------------------------------------------
# PYTHON is consumed by the scripts that source this file (not exported on
# purpose — it stays a shell var the same-process callers use as "$PYTHON").
# shellcheck disable=SC2034
PYTHON="${GPUMODE_VENV_PYTHON:-$REPO_DIR/.venv/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUTLASS_PATH="${CUTLASS_PATH:-/opt/cutlass}"
export MATHDX_HOME="${MATHDX_HOME:-/opt/mathdx}"
[ -x "$PYTHON" ] || {
    echo "GPU MODE venv not found at $PYTHON; run $REPO_DIR/bin/install.sh" >&2; exit 1; }
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
export PATH="$PYTHON_BIN_DIR:$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CPLUS_INCLUDE_PATH="$MATHDX_HOME/include:$MATHDX_HOME/external/cutlass/include:$CUTLASS_PATH/include:$CUTLASS_PATH/tools/util/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# Every harness entry point verifies the runtime before loading problem code.
# This turns stale venvs, wrong CUDA defaults, and missing production headers
# into an immediate actionable error instead of a late compile/benchmark fault.
if [ "${GPUMODE_SKIP_ENV_CHECK:-0}" != 1 ]; then
    "$PYTHON" "$REPO_DIR/bin/verify_environment.py" >/dev/null || exit 1
fi

# Locate eval.py / utils.py from the problem's task.yml `files:` manifest —
# the same flattening KernelBot does — instead of assuming the set root. Most
# problems share `../eval.py` (so EVAL_PY == SET_DIR/eval.py), but several ship
# a problem-local eval.py (linalg/qr_py, amd/mla-decode, bioml/trimul, …) and
# qr_py even borrows utils.py from another set (../../pmpp_v2/utils.py). The
# resolver returns an absolute path and falls back to the set-root sibling, so
# this is strictly more general than the old hardcode. Callers run "$EVAL_PY".
EVAL_PY="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --file-source eval.py)"
UTILS_PY="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --file-source utils.py)"
[ -f "$EVAL_PY" ] || { echo "eval.py not found for '$PROBLEM' (resolved: $EVAL_PY)" >&2; exit 1; }
EVAL_DIR="$(dirname "$EVAL_PY")"
UTILS_DIR="$(dirname "$UTILS_PY")"

# eval.py uses bare imports (`from utils import …`, `from reference import …`,
# `from submission import …`). reference.py + submission.py live in the problem
# dir; eval.py / utils.py live wherever the manifest points (resolved above).
# Put the problem dir FIRST so its submission.py/reference.py win, then the
# dirs holding eval.py and utils.py. The `spawn` Pool eval.py uses re-imports,
# so PYTHONPATH (inherited by children) is what makes the in-place layout
# resolve. Duplicate path entries are harmless when the dirs coincide.
export PYTHONPATH="$PROBLEM_DIR:$EVAL_DIR:$UTILS_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Build cache for torch load_inline, keyed by source-content hash. Per-worktree
# (under the problem dir) so concurrent optimize-tree workers never share or
# race on a cache dir.
export TORCH_EXTENSIONS_DIR="$PROBLEM_DIR/.torch_ext"
export PYTHONNOUSERSITE=1
# Cap per-worker nvcc/ninja parallelism: with N concurrent worker builds keep
# N x MAX_JOBS <= nproc-1. Default 3 suits a 4-worker run on a 16-core host.
export MAX_JOBS="${MAX_JOBS:-3}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
