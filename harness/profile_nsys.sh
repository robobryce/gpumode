#!/usr/bin/env bash
# PROFILE helper (Nsight Systems) — autocuda <-> GPU MODE bridge.
#
# Profiles ONE benchmark shape's `custom_kernel` under nsys and prints the
# per-kernel summary. The first arg is the <set>/<problem> to profile; an
# optional second arg pins the shape spec, else the first benchmark shape from
# the problem's task.yml is used.
#
# Usage (wrap with autocuda run exclusive):
#   autocuda run exclusive --data-dir "$DATA_DIR" -- \
#     bash harness/profile_nsys.sh <set>/<problem> [<shape-spec>] \
#       > "$DATA_DIR/profiles/<tag>/<name>-<sha>.nsys-txt" 2>&1
# The .nsys-rep is written next to that path (-o without the extension); redirect
# stdout/stderr to a SHA-named file so `autocuda init brief` can hand it on.
#
# Why a driver instead of profiling eval.py directly: eval.py runs custom_kernel
# in a `spawn` Pool (nsys follows it, but it adds noise) AND `benchmark` mode
# interleaves the cuSOLVER/cuBLAS reference checker with every timed call, so a
# whole-benchmark trace mixes reference kernels with yours. profile_driver.py
# imports the live submission + reference and wraps ONLY the timed custom_kernel
# calls in a cudaProfiler range, so `--capture-range=cudaProfilerApi` records
# just your kernels. Correctness stays eval.py's job (validate.sh / benchmark.sh).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"

if [ "${2:-}" ]; then SPEC="$2"
else SPEC="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks | head -n1)"; fi

# -o needs an explicit path; default to the data dir's profiles area when the
# caller didn't redirect into one. The capture range (driver's cudaProfiler
# start/stop) means warmup launches are excluded automatically.
REP="${NSYS_OUT:-$(mktemp -u --suffix=.nsys-rep)}"
# When the caller named the output (the normal SHA-named worker path) keep it;
# otherwise it's a throwaway, so clean up the .nsys-rep and the .sqlite that
# `nsys stats` materialises beside it.
[ -n "${NSYS_OUT:-}" ] || trap 'rm -f "$REP" "${REP%.nsys-rep}.sqlite"' EXIT
NSYS="$(command -v nsys || echo "$CUDA_HOME/bin/nsys")"

cd "$PROBLEM_DIR" || exit 1
"$NSYS" profile --force-overwrite true -o "${REP%.nsys-rep}" \
    --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx \
    "$PYTHON" "$HARNESS_DIR/profile_driver.py" "$SPEC" 3 5 >&2

echo "===== nsys cuda_gpu_kern_sum ($SPEC) ====="
"$NSYS" stats --report cuda_gpu_kern_sum --format table "$REP"
