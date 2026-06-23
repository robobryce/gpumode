#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"
SPEC="batch: 8; n: 2048; cond: 1; seed: 224466"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s\n' "$SPEC" > "$SPECFILE"
cd "$PROBLEM_DIR" || exit 1
NCU="$(command -v ncu || echo "$CUDA_HOME/bin/ncu")"
# Target the panel + the two trailing helpers. Skip the first few panel launches
# (warmup / tau=0 region) and grab a mid-factorization launch of each so the
# occupancy reflects steady state. Occupancy + launch stats sections only (fast).
KEEP=(
    PATH="$PATH" PYTHONPATH="${PYTHONPATH:-}" CUDA_HOME="$CUDA_HOME"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
    PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" MAX_JOBS="$MAX_JOBS"
)
[ -n "${LD_LIBRARY_PATH:-}" ] && KEEP+=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH")
sudo -n env "${KEEP[@]}" "$NCU" \
    --kernel-name "regex:_panel_factor_kernel|_trailing_YT_kernel|_trailing_apply_kernel" \
    --launch-skip 30 --launch-count 9 \
    --section Occupancy --section LaunchStats --section SpeedOfLight \
    bash -c 'exec 3>/dev/null; export POPCORN_FD=3; exec "$@"' profile_ncu \
    "$PYTHON" "$EVAL_PY" benchmark "$SPECFILE" 3>/dev/null
