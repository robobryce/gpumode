#!/usr/bin/env bash
# Run validation or benchmarking on a remote Modal GPU.
# Usage: bash harness/modal.sh {validate|benchmark} <set>/<problem>
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HARNESS_DIR/.." && pwd)"
if [ "$#" -ne 2 ]; then
    echo "usage: modal.sh {validate|benchmark} <set>/<problem>" >&2
    exit 2
fi
ACTION="$1"
PROBLEM="$2"

case "$ACTION" in
    validate|benchmark) ;;
    *) echo "action must be 'validate' or 'benchmark'" >&2; exit 2 ;;
esac
[ -f "$REPO_DIR/problems/$PROBLEM/submission.py" ] || {
    echo "no submission.py for '$PROBLEM'" >&2; exit 2;
}
# Prefer an explicit command, then PATH, then the gpumode venv written by
# bin/install.sh. The Modal CLI is a console script (``python -m modal`` is not
# supported by every Modal release).
for cfg in "${GPUMODE_ENV:-}" "$HOME/.config/gpumode/gpumode.env"; do
    if [ -n "$cfg" ] && [ -f "$cfg" ]; then source "$cfg"; break; fi
done
MODAL_BIN="${MODAL_BIN:-$(command -v modal 2>/dev/null || true)}"
if [ -z "$MODAL_BIN" ] && [ -n "${GPUMODE_VENV_PYTHON:-}" ]; then
    candidate="$(dirname "$GPUMODE_VENV_PYTHON")/modal"
    [ -x "$candidate" ] && MODAL_BIN="$candidate"
fi
[ -n "$MODAL_BIN" ] || {
    echo "modal CLI not found; run bin/install.sh or install it with pip" >&2
    exit 127
}

cd "$REPO_DIR"

# Keep non-production static policy checks local. The remote Modal image mirrors
# the leaderboard dependency set exactly and therefore deliberately does not
# add kernelguard or other local-only packages.
if [ "$ACTION" = validate ]; then
    PYTHON_BIN="${GPUMODE_VENV_PYTHON:-}"
    if [ -z "$PYTHON_BIN" ] && [ -x "$HOME/.venv/bin/python" ]; then
        PYTHON_BIN="$HOME/.venv/bin/python"
    fi
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
    if grep -Eq '^[[:space:]]*(from[[:space:]]+cupy|import[[:space:]]+cupy)([[:space:]]|$)' "problems/$PROBLEM/submission.py"; then
        echo "validation: FAILED (submission.py imports CuPy, which is unavailable in the leaderboard runner)" >&2
        exit 3
    fi
    if grep -iq stream "problems/$PROBLEM/submission.py"; then
        echo "validation: FAILED (submission.py references 'stream'; CUDA streams are banned by the leaderboard)" >&2
        exit 3
    fi
    "$PYTHON_BIN" bin/kernelguard_gate.py "problems/$PROBLEM/submission.py"
fi

exec "$MODAL_BIN" run harness/modal_runner.py --action "$ACTION" --problem "$PROBLEM"
