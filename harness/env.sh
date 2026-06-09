# Shared environment for the autocuda <-> GPU MODE bridge.
# Sourced by build.sh / validate.sh / benchmark.sh / profile_ncu.sh.
#
# Optimization happens IN PLACE: the editable file is the problem's own
# `submission.py` under `problems/<set>/<problem>/`, and the frozen
# `eval.py` / `utils.py` (at the set root) plus `reference.py` / `task.yml`
# (in the problem dir) are read where they already live — nothing is copied.
#
# Select the target problem with GPUMODE_PROBLEM=<set>/<problem> (e.g.
# `pmpp_v2/histogram_py`), an env var the autocuda layout tells the manager to
# export. The repo root holds ONE autocuda data dir (`autocuda/`).
set -uo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HARNESS_DIR/.." && pwd)"

# --- 1. machine-level config -------------------------------------------------
# Written by bin/install.sh to ~/.config/gpumode/gpumode.env. Defines
# GPUMODE_VENV_PYTHON and CUDA_HOME. (GPUMODE_REFERENCE_KERNELS is unused here —
# this repo IS the reference-kernels checkout.)
for cfg in "${GPUMODE_ENV:-}" "$HOME/.config/gpumode/gpumode.env"; do
    if [ -n "$cfg" ] && [ -f "$cfg" ]; then source "$cfg"; break; fi
done

# --- 2. target problem -------------------------------------------------------
: "${GPUMODE_PROBLEM:?set GPUMODE_PROBLEM=<set>/<problem> (e.g. pmpp_v2/histogram_py)}"
PROBLEM_DIR="$REPO_DIR/problems/$GPUMODE_PROBLEM"
SET_DIR="$(dirname "$PROBLEM_DIR")"   # holds eval.py / utils.py
[ -f "$PROBLEM_DIR/submission.py" ] || {
    echo "no submission.py at $PROBLEM_DIR — is GPUMODE_PROBLEM=$GPUMODE_PROBLEM correct?" >&2; exit 1; }

# --- derived / defaults ------------------------------------------------------
# PYTHON is consumed by the scripts that source this file (not exported on
# purpose — it stays a shell var the same-process callers use as "$PYTHON").
# shellcheck disable=SC2034
PYTHON="${GPUMODE_VENV_PYTHON:-$HOME/gpumode/.venv/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# eval.py uses bare imports (`from utils import …`, `from reference import …`,
# `from submission import …`). reference.py + submission.py live in the problem
# dir; eval.py + utils.py at the set root. Put the problem dir FIRST so its
# submission.py/reference.py win, then the set dir for eval.py/utils.py. The
# `spawn` Pool eval.py uses re-imports, so PYTHONPATH (inherited by children)
# is what makes the in-place layout resolve.
export PYTHONPATH="$PROBLEM_DIR:$SET_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Build cache for torch load_inline, keyed by source-content hash. Per-worktree
# (under the problem dir) so concurrent optimize-tree workers never share or
# race on a cache dir.
export TORCH_EXTENSIONS_DIR="$PROBLEM_DIR/.torch_ext"
export PYTHONNOUSERSITE=1
# Cap per-worker nvcc/ninja parallelism: with N concurrent worker builds keep
# N x MAX_JOBS <= nproc-1. Default 3 suits a 4-worker run on a 16-core host.
export MAX_JOBS="${MAX_JOBS:-3}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
