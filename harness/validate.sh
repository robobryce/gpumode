#!/usr/bin/env bash
# VALIDATE step — autocuda <-> GPU MODE bridge.
#
# Runs the official GPU MODE eval harness in `test` mode over the problem's
# test shapes (rendered to harness/<set>__<problem>-tests.txt by build-time
# gen_specs, or generated on the fly here). Correctness is whatever the
# problem's reference.py / utils.py check_implementation enforces. Exits 0 iff
# every test passes (`check: pass`); non-zero on any mismatch, crash, or
# compile error. These are the SAME shapes the leaderboard's `--mode test`
# uses, so a local pass faithfully predicts a remote test pass.
#
# Usage:  bash harness/validate.sh <set>/<problem>   # e.g. pmpp_v2/histogram_py
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"

SPECS="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit tests)"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE" "$OUT"' EXIT
printf '%s' "$SPECS" > "$SPECFILE"

cd "$PROBLEM_DIR" || exit 1
OUT="$(mktemp)"
# POPCORN_FD=3 -> eval.py writes results to fd 3; capture it. eval.py is found
# on PYTHONPATH (set dir); run it as a module so its dir isn't forced onto cwd.
POPCORN_FD=3 "$PYTHON" "$SET_DIR/eval.py" test "$SPECFILE" 3>"$OUT"
rc=$?
echo "----- eval.py test output -----"; cat "$OUT"; echo "-------------------------------"
if [ $rc -ne 0 ]; then echo "validation: FAILED (eval.py exited $rc)"; exit $rc; fi
if grep -qx "check: pass" "$OUT"; then echo "validation: PASS"; exit 0; fi
echo "validation: FAILED (no 'check: pass')"; exit 1
