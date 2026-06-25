#!/usr/bin/env bash
# PROFILE helper (Nsight Compute) — autocuda <-> GPU MODE bridge.
#
# Profiles ONE benchmark shape's `custom_kernel` under ncu, capturing the
# kernels it launches. The first arg is the <set>/<problem> to profile; an
# optional second arg pins the shape spec, else the first benchmark shape from
# the problem's task.yml is used. A third optional arg is an ncu --kernel-name
# value (e.g. `regex:gemm`) to focus on the dominant kernel; default profiles
# the first few kernels in the range.
#
# Usage (wrap with autocuda run exclusive):
#   autocuda run exclusive --data-dir "$DATA_DIR" -- \
#     bash harness/profile_ncu.sh <set>/<problem> [<shape-spec>] [<kernel-filter>] \
#       > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
# (e.g. <set>/<problem> = pmpp_v2/histogram_py). Redirect to a SHA-named file so
# `autocuda init brief` can hand it to the next brief.
#
# Why a driver instead of profiling eval.py directly: eval.py runs custom_kernel
# in a `spawn` Pool AND `benchmark` mode interleaves the cuSOLVER/cuBLAS
# reference checker with every timed call, so profiling eval.py mixes reference
# kernels with yours and needs fragile --launch-skip guesswork to dodge them.
# profile_driver.py imports the live submission + reference and wraps ONLY the
# timed custom_kernel calls in a cudaProfiler range, so `--profile-from-start
# off` records just your kernels. Correctness stays eval.py's job.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"

if [ "${2:-}" ]; then SPEC="$2"
else SPEC="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks | head -n1)"; fi
KERNEL_FILTER="${3:-}"

cd "$PROBLEM_DIR" || exit 1
NCU="$(command -v ncu || echo "$CUDA_HOME/bin/ncu")"

# Profiler capture range (driver's cudaProfilerStart/Stop) excludes warmup;
# --profile-from-start off honours it. --launch-count caps collected kernels;
# the warmup runs 5 calls so the captured iters are steady-state. An optional
# --kernel-name focuses the dominant kernel (e.g. the trailing GEMM).
NCU_ARGS=(--profile-from-start off --set full --launch-count 6)
[ -n "$KERNEL_FILTER" ] && NCU_ARGS+=(--kernel-name "$KERNEL_FILTER")

# ncu must run as root to read the GPU performance counters; without it the
# profile aborts with ERR_NVGPUCTRPERM. So run it under sudo (skipped when we
# are already root). sudo's env_reset drops PYTHONPATH / LD_LIBRARY_PATH even
# under -E (they sit on its blocklist), and the driver's `spawn`-free imports
# resolve through PYTHONPATH (see env.sh), so we re-establish the load-bearing
# vars on the far side of sudo with `env`, which its filtering cannot touch.
run_ncu() {
    # "$@" is the privilege prefix (`sudo -n env VAR=val ...`); empty when root.
    "$@" "$NCU" "${NCU_ARGS[@]}" \
        "$PYTHON" "$HARNESS_DIR/profile_driver.py" "$SPEC" 5 6
}

if [ "$(id -u)" -eq 0 ]; then
    run_ncu
else
    command -v sudo >/dev/null \
        || { echo "profile_ncu: ncu needs root but sudo is not installed" >&2; exit 1; }
    # The vars env.sh set up that the profiled driver (and its torch JIT) need on
    # the far side of sudo. PATH carries $CUDA_HOME/bin and overrides sudo's
    # secure_path. LD_LIBRARY_PATH is forwarded only when the machine set it, so
    # an unset value is never turned into an empty one (an empty element means
    # "cwd" to the loader).
    KEEP=(
        PATH="$PATH"
        PYTHONPATH="$PYTHONPATH"
        CUDA_HOME="$CUDA_HOME"
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
        TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
        PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
        MAX_JOBS="$MAX_JOBS"
    )
    [ -n "${LD_LIBRARY_PATH:-}" ] && KEEP+=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH")
    # `sudo -n`: fail fast rather than hang on a password prompt under the
    # automated `autocuda run exclusive` harness (the host needs passwordless
    # sudo for ncu, or run this as root). `env` re-exports the vars sudo strips.
    run_ncu sudo -n env "${KEEP[@]}"
fi
