#!/usr/bin/env bash
# PROFILE helper (Nsight Compute) — autocuda <-> GPU MODE bridge.
#
# Profiles ONE benchmark shape under ncu, capturing the kernels custom_kernel
# launches. The first arg is the <set>/<problem>; an optional second arg pins
# the shape spec (else the first benchmark shape from task.yml); an optional
# third arg is an ncu --kernel-name value (e.g. `regex:gemm`) to focus the
# dominant kernel.
#
# Usage (wrap with autocuda run exclusive):
#   autocuda run exclusive --data-dir "$DATA_DIR" -- \
#     bash harness/profile_ncu.sh <set>/<problem> [<shape-spec>] [<kernel-filter>] \
#       > "$DATA_DIR/profiles/<tag>/<name>-<sha>.ncu-txt" 2>&1
# (e.g. <set>/<problem> = pmpp_v2/histogram_py). Redirect to a SHA-named file so
# `autocuda init brief` can hand it to the next brief.
#
# eval.py wraps the timed custom_kernel launches in a torch.cuda.profiler range,
# so `--profile-from-start off` records only those launches — not the warmup,
# the L2 flush, or the reference checker eval.py runs between them.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"

if [ "${2:-}" ]; then SPEC="$2"
else SPEC="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks | head -n1)"; fi
KERNEL_FILTER="${3:-}"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s\n' "$SPEC" > "$SPECFILE"

cd "$PROBLEM_DIR" || exit 1
NCU="$(command -v ncu || echo "$CUDA_HOME/bin/ncu")"

# --profile-from-start off honours eval.py's cudaProfiler range (so warmup and
# reference-checker kernels are excluded). --launch-count caps collected
# kernels; an optional --kernel-name focuses the dominant one (e.g. the GEMM).
NCU_ARGS=(--profile-from-start off --set full --launch-count 6)
[ -n "$KERNEL_FILTER" ] && NCU_ARGS+=(--kernel-name "$KERNEL_FILTER")

# ncu must run as root to read the GPU performance counters; without it the
# profile aborts with ERR_NVGPUCTRPERM. So run it under sudo (skipped when we
# are already root). Two sudo behaviours bite the profiler — both handled here
# so no special sudoers config is needed:
#
#   * env_reset drops PYTHONPATH / LD_LIBRARY_PATH across the privilege boundary
#     even under `sudo -E` (they sit on sudo's blocklist). eval.py's `spawn` Pool
#     re-imports submission/reference/utils through PYTHONPATH (see env.sh), so
#     dropping it breaks the profiled run. We re-establish the load-bearing vars
#     on the far side of sudo with `env`, which its filtering cannot touch.
#   * sudo closes inherited fds >= 3 (closefrom), discarding the POPCORN_FD=3
#     channel eval.py writes its protocol to — eval.py would die with EBADF
#     before timing. A one-line bash shim reopens fd 3 -> /dev/null after the
#     boundary, then execs the real command in the same process so ncu still
#     profiles the kernels.
run_ncu() {
    # "$@" is the privilege prefix (`sudo -n env VAR=val ...`); empty when root.
    "$@" "$NCU" "${NCU_ARGS[@]}" \
        bash -c 'exec 3>/dev/null; export POPCORN_FD=3; exec "$@"' profile_ncu \
        "$PYTHON" "$EVAL_PY" benchmark "$SPECFILE" 3>/dev/null
}

if [ "$(id -u)" -eq 0 ]; then
    run_ncu
else
    command -v sudo >/dev/null \
        || { echo "profile_ncu: ncu needs root but sudo is not installed" >&2; exit 1; }
    # The vars env.sh set up that the profiled eval.py (and its nvcc/torch JIT)
    # need on the far side of sudo. PATH carries $CUDA_HOME/bin (nvcc) and
    # overrides sudo's secure_path. LD_LIBRARY_PATH is forwarded only when the
    # machine actually set it, so an unset value is never turned into an empty
    # one (an empty LD_LIBRARY_PATH element means "cwd" to the loader).
    KEEP=(
        PATH="$PATH"
        PYTHONPATH="${PYTHONPATH:-}"
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
