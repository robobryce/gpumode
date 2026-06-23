#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"
SPEC="batch: 8; n: 2048; cond: 1; seed: 224466"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s\n' "$SPEC" > "$SPECFILE"
cd "$PROBLEM_DIR" || exit 1
exec "$PYTHON" "$EVAL_PY" benchmark "$SPECFILE"
