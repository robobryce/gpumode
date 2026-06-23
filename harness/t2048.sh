#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"
cd "$PROBLEM_DIR" || exit 1
shift  # drop the <set>/<problem> arg; rest are config groups
exec "$PYTHON" _t2048.py "$@"
