#!/usr/bin/env bash
# DIAG helper: run eval.py benchmark on ONE shape with SIGNDC_DBG + SIGNDC_TIME
# so the sign-DC gate margins (orth/eigr/nbad) and per-stage timing print to
# stderr. Measurement-only (not a kernel proxy) -- exercises the live path.
# Usage (wrap with autocuda run exclusive):
#   autocuda run exclusive --data-dir "$DATA_DIR" -- \
#     bash harness/diag_shape.sh linalg/eigh_py "<shape-spec>"
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"
SPEC="${2:?need a shape spec}"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s\n' "$SPEC" > "$SPECFILE"
cd "$PROBLEM_DIR" || exit 1
SIGNDC_DBG=1 SIGNDC_TIME=1 POPCORN_FD=1 "$PYTHON" "$EVAL_PY" benchmark "$SPECFILE"
