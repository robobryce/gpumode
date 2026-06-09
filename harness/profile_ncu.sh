#!/usr/bin/env bash
# PROFILE helper (Nsight Compute) — autocuda <-> GPU MODE bridge.
#
# Profiles ONE representative benchmark shape under ncu, capturing the kernel
# custom_kernel launches. The shape is $1 if given, else the first benchmark
# shape from the problem's task.yml.
#
# Usage (wrap with autocuda run exclusive):
#   autocuda run exclusive --data-dir "$DATA_DIR" -- \
#     bash harness/profile_ncu.sh > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
# Redirect to a SHA-named file so `autocuda init brief` can hand it to the next
# brief. A pure-PyTorch baseline shows torch's internal kernels; a custom-CUDA
# submission shows your single kernel cleanly.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ "${1:-}" ]; then SPEC="$1"
else SPEC="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks | head -n1)"; fi
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s\n' "$SPEC" > "$SPECFILE"

cd "$PROBLEM_DIR" || exit 1
NCU="$(command -v ncu || echo "$CUDA_HOME/bin/ncu")"
run_ncu() {
    # "$@" is an optional privilege prefix (e.g. sudo -E); empty when unset.
    POPCORN_FD=3 "$@" "$NCU" --set full --launch-skip 5 --launch-count 1 \
        "$PYTHON" "$SET_DIR/eval.py" benchmark "$SPECFILE" 3>/dev/null
}
run_ncu || run_ncu sudo -E
