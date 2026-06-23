#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"
SPEC="batch: 8; n: 2048; cond: 1; seed: 224466"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s\n' "$SPEC" > "$SPECFILE"
cd "$PROBLEM_DIR" || exit 1
NCU="$(command -v ncu || echo "$CUDA_HOME/bin/ncu")"
KEEP=(
    PATH="$PATH" PYTHONPATH="${PYTHONPATH:-}" CUDA_HOME="$CUDA_HOME"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
    PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" MAX_JOBS="$MAX_JOBS"
)
[ -n "${LD_LIBRARY_PATH:-}" ] && KEEP+=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH")
# Just the panel, mid-factorization, with scheduler + warp-state + memory sections.
sudo -n env "${KEEP[@]}" "$NCU" \
    --kernel-name "regex:_panel_factor_kernel" \
    --launch-skip 40 --launch-count 3 \
    --section Occupancy --section SpeedOfLight --section SchedulerStats \
    --section WarpStateStats --section MemoryWorkloadAnalysis \
    bash -c 'exec 3>/dev/null; export POPCORN_FD=3; exec "$@"' profile_ncu \
    "$PYTHON" "$EVAL_PY" benchmark "$SPECFILE" 3>/dev/null
