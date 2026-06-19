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
# After the fixed test shapes pass, validate ALSO runs any PROBLEM-SPECIFIC
# guards the problem ships under `problems/<set>/<problem>/guards/*.sh` — the
# anti-reward-hack / anti-data-fitting checks that the fixed shapes alone cannot
# catch (e.g. linalg/qr_v2 ships a differential-correctness guard and a
# functional-invariance guard). They run IN THIS PROCESS, under whatever GPU
# lock the caller already holds: NEVER wrap a guard (or this script) in a nested
# `autocuda run` — `autocuda run exclusive` holds the fleet scheduler lock while
# it waits for a GPU, so a nested exclusive call deadlocks (it can never get the
# GPU its parent is holding) and starves the whole fleet. The one outer
# `autocuda run exclusive -- bash harness/validate.sh <set>/<problem>` already
# covers the test AND every guard under a single GPU lock. A problem with no
# guards/ dir just runs the tests, exactly as before.
#
# Usage:  bash harness/validate.sh <set>/<problem>   # e.g. pmpp_v2/histogram_py
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"

# --- banned-construct scan ---------------------------------------------------
# The GPU MODE leaderboard rejects submissions that do work on a CUDA stream
# ("Your code contains work on another stream. This is not allowed and may
# result in your disqualification."). A local benchmark cannot see this — only a
# remote submit does — so gate it here, for EVERY problem, before spending GPU
# time. Case-insensitive substring match on "stream" catches cudaStream_t,
# getCurrentCUDAStream, cublasSetStream, CUDAStreamGuard, etc.
if grep -iq "stream" "$PROBLEM_DIR/submission.py"; then
    echo "----- banned-construct scan -----"
    grep -in "stream" "$PROBLEM_DIR/submission.py" | head -20
    echo "---------------------------------"
    echo "validation: FAILED (submission.py references 'stream'; CUDA streams are banned by the leaderboard)"
    exit 3
fi

SPECS="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit tests)"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE" "$OUT"' EXIT
printf '%s' "$SPECS" > "$SPECFILE"

cd "$PROBLEM_DIR" || exit 1
OUT="$(mktemp)"
# POPCORN_FD=3 -> eval.py writes results to fd 3; capture it. $EVAL_PY is the
# manifest-resolved eval.py (set-root or problem-local); env.sh put its dir on
# PYTHONPATH so its sibling imports (utils/reference) resolve under the Pool.
POPCORN_FD=3 "$PYTHON" "$EVAL_PY" test "$SPECFILE" 3>"$OUT"
rc=$?
echo "----- eval.py test output -----"; cat "$OUT"; echo "-------------------------------"
if [ $rc -ne 0 ]; then echo "validation: FAILED (eval.py exited $rc)"; exit $rc; fi
if ! grep -qx "check: pass" "$OUT"; then echo "validation: FAILED (no 'check: pass')"; exit 1; fi
echo "validation: test shapes PASS"

# --- problem-specific guards -------------------------------------------------
# Discover and run every guard the problem ships under guards/. Each is a
# self-contained script taking the <set>/<problem> token (the qr guards are
# symlinks to harness/{diff_correctness,invariance}_guard.sh). They run in THIS
# process under the caller's existing GPU lock — do not nest `autocuda run`.
GUARD_DIR="$PROBLEM_DIR/guards"
shopt -s nullglob
GUARDS=("$GUARD_DIR"/*.sh)
shopt -u nullglob
if [ ${#GUARDS[@]} -eq 0 ]; then
    echo "validation: PASS (no problem-specific guards)"; exit 0
fi
echo "----- problem-specific guards (${#GUARDS[@]}) -----"
guard_fail=0
for g in "${GUARDS[@]}"; do
    name="$(basename "$g")"
    echo ">>> guard: $name"
    if bash "$g" "$PROBLEM"; then
        echo "<<< guard $name: OK"
    else
        grc=$?
        echo "<<< guard $name: FAILED (exit $grc)"
        guard_fail=1
    fi
done
echo "-------------------------------"
if [ $guard_fail -ne 0 ]; then echo "validation: FAILED (a problem-specific guard failed)"; exit 2; fi
echo "validation: PASS"; exit 0
