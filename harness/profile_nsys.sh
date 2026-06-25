#!/usr/bin/env bash
# PROFILE helper (Nsight Systems) — autocuda <-> GPU MODE bridge.
#
# Profiles ONE benchmark shape under nsys and prints the per-kernel summary.
# The first arg is the <set>/<problem> to profile; an optional second arg pins
# the shape spec, else the first benchmark shape from the problem's task.yml.
#
# Usage (wrap with autocuda run exclusive):
#   autocuda run exclusive --data-dir "$DATA_DIR" -- \
#     bash harness/profile_nsys.sh <set>/<problem> [<shape-spec>] \
#       > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
# The .nsys-rep is written next to that path (-o without the extension); set
# NSYS_OUT to keep it at a SHA-named path so `autocuda init brief` can hand it on.
#
# eval.py wraps the timed custom_kernel launches in a torch.cuda.profiler range,
# so `--capture-range=cudaProfilerApi` records only those launches — not the
# warmup, the L2 flush, or the reference checker eval.py runs between them.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"

if [ "${2:-}" ]; then SPEC="$2"
else SPEC="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks | head -n1)"; fi
SPECFILE="$(mktemp)"; printf '%s\n' "$SPEC" > "$SPECFILE"

REP="${NSYS_OUT:-$(mktemp -u --suffix=.nsys-rep)}"
# Keep a caller-named report; clean up a throwaway one (plus the .sqlite that
# `nsys stats` materialises) and the spec file.
if [ -n "${NSYS_OUT:-}" ]; then trap 'rm -f "$SPECFILE"' EXIT
else trap 'rm -f "$SPECFILE" "$REP" "${REP%.nsys-rep}.sqlite"' EXIT; fi
NSYS="$(command -v nsys || echo "$CUDA_HOME/bin/nsys")"

cd "$PROBLEM_DIR" || exit 1
POPCORN_FD=3 "$NSYS" profile --force-overwrite true -o "${REP%.nsys-rep}" \
    --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx \
    "$PYTHON" "$EVAL_PY" benchmark "$SPECFILE" 3>/dev/null >&2

echo "===== nsys cuda_gpu_kern_sum ($SPEC) ====="
"$NSYS" stats --report cuda_gpu_kern_sum --format table "$REP"
