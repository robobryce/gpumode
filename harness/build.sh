#!/usr/bin/env bash
# BUILD step — autocuda <-> GPU MODE bridge.
#
# GPU MODE submissions compile their CUDA at runtime (torch load_inline /
# cpp_extension.load), so "build" means: import the problem's submission.py in
# a fresh process to trigger that compilation, surfacing any nvcc/compile error
# as a non-zero exit NOW (an autocuda build_error) instead of at benchmark time.
# A pure-PyTorch submission imports instantly. The compiled extension caches in
# $TORCH_EXTENSIONS_DIR, so the validate / benchmark imports reuse this build.
#
# Usage:  bash harness/build.sh <set>/<problem>   # e.g. pmpp_v2/histogram_py
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"
cd "$PROBLEM_DIR" || exit 1

cleanup_stale_torch_locks() {
    local stale_after now lock lock_age ext_dir removed=0
    stale_after="${TORCH_STALE_LOCK_SECONDS:-600}"
    now="$(date +%s)"

    [ -d "$TORCH_EXTENSIONS_DIR" ] || return 0

    while IFS= read -r -d '' lock; do
        lock_age=$((now - $(stat -c %Y "$lock" 2>/dev/null || echo "$now")))
        [ "$lock_age" -ge "$stale_after" ] || continue

        ext_dir="$(dirname "$lock")"
        if fuser "$lock" >/dev/null 2>&1; then
            echo "build: keeping active torch extension lock $lock" >&2
            continue
        fi
        if command -v lsof >/dev/null 2>&1 && lsof +D "$ext_dir" >/dev/null 2>&1; then
            echo "build: keeping busy torch extension dir $ext_dir" >&2
            continue
        fi

        echo "build: removing stale torch extension lock $lock (age ${lock_age}s)" >&2
        rm -f -- "$lock"
        removed=$((removed + 1))
    done < <(find "$TORCH_EXTENSIONS_DIR" -mindepth 2 -maxdepth 2 -type f -name lock -print0 2>/dev/null)

    [ "$removed" -eq 0 ] || echo "build: removed $removed stale torch extension lock(s)" >&2
}

cleanup_stale_torch_locks

BUILD_IMPORT_TIMEOUT_SECONDS="${BUILD_IMPORT_TIMEOUT_SECONDS:-1800}"
timeout --preserve-status --signal=TERM --kill-after=30s "$BUILD_IMPORT_TIMEOUT_SECONDS" "$PYTHON" - <<'PY'
import sys, traceback
try:
    import submission  # triggers load_inline / cpp_extension.load at import
    assert hasattr(submission, "custom_kernel"), "submission.py defines no custom_kernel"
except Exception:
    traceback.print_exc()
    sys.exit(1)
print("BUILD_OK")
PY
rc=$?
if [ $rc -eq 143 ] || [ $rc -eq 124 ]; then
    echo "build: FAILED (import timed out after ${BUILD_IMPORT_TIMEOUT_SECONDS}s)" >&2
fi
[ $rc -eq 0 ] && echo "build: ok" || echo "build: FAILED (rc=$rc)"
exit $rc
