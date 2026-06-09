#!/usr/bin/env bash
# BUILD step — autocuda <-> GPU MODE bridge.
#
# GPU MODE submissions compile their CUDA at runtime (torch load_inline /
# cpp_extension.load), so "build" means: import the problem's submission.py in
# a fresh process to trigger that compilation, surfacing any nvcc/compile error
# as a non-zero exit NOW (an autocuda build_error) instead of at benchmark time.
# A pure-PyTorch submission imports instantly. The compiled extension caches in
# $TORCH_EXTENSIONS_DIR, so the validate / benchmark imports reuse this build.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROBLEM_DIR" || exit 1

"$PYTHON" - <<'PY'
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
[ $rc -eq 0 ] && echo "build: ok" || echo "build: FAILED (rc=$rc)"
exit $rc
