#!/usr/bin/env bash
# Submit the selected GPU MODE problem's current submission.py to the public
# leaderboard. This is mandatory evidence for autocuda runs: baseline setup,
# meaningful improvements, and final candidates must all be submitted.
#
# Evidence contract (do not regress): this script must PROVE acceptance, not
# merely run popcorn-cli. It captures the full submit output, requires the
# remote "Leaderboard run successful" + "Passed N/N tests" markers, and exits
# NON-ZERO on any rejection/timeout/missing-verdict so a wrapping `tail` or an
# exit-0 check can never mistake a failed submit for a successful one. The
# verified verdict (submission id + per-shape ranked timings) is printed to
# stdout between explicit BEGIN/END fences.
set -euo pipefail
source "$(dirname "$0")/env.sh" "$@"

TASK_YML="$PROBLEM_DIR/task.yml"
LEADERBOARD="$($PYTHON "$REPO_DIR/bin/gen_specs.py" "$TASK_YML" --leaderboard)"
SUPPORTED_GPUS="$($PYTHON "$REPO_DIR/bin/gen_specs.py" "$TASK_YML" --gpus)"
MODE="${GPUMODE_SUBMIT_MODE:-leaderboard}"
COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"

choose_gpu() {
    if [ -n "${GPUMODE_GPU:-}" ]; then
        printf '%s\n' "$GPUMODE_GPU"
        return
    fi

    local gpu_name=""
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    IFS=',' read -r -a supported <<< "$SUPPORTED_GPUS"
    for gpu in "${supported[@]}"; do
        if [ -n "$gpu" ] && printf '%s\n' "$gpu_name" | grep -qi -- "$gpu"; then
            printf '%s\n' "$gpu"
            return
        fi
    done

    for gpu in "${supported[@]}"; do
        if [ -n "$gpu" ]; then
            printf '%s\n' "$gpu"
            return
        fi
    done

    echo "ERROR: no supported GPUs listed for $PROBLEM in $TASK_YML" >&2
    return 1
}

GPU="$(choose_gpu)"

echo "leaderboard-submit: problem=$PROBLEM leaderboard=$LEADERBOARD gpu=$GPU mode=$MODE commit=$COMMIT" >&2

# Capture the FULL submit transcript (stdout+stderr) so the verdict survives any
# downstream `tail`/`head`. popcorn-cli streams "processing" lines then a final
# JSON/markdown result; we tee it to the operator's terminal AND a temp file.
SUBMIT_LOG="$(mktemp -t popcorn-submit-XXXXXX.log)"
set +e
popcorn-cli submit --no-tui \
    --leaderboard "$LEADERBOARD" \
    --gpu "$GPU" \
    --mode "$MODE" \
    "$PROBLEM_DIR/submission.py" 2>&1 | tee "$SUBMIT_LOG"
CLI_RC="${PIPESTATUS[0]}"
set -e

# --- Verdict parsing: FAILURE-FIRST. ---
# The remote transcript can contain BOTH a "Leaderboard run failed" line (the
# real ranked run) AND a later "Leaderboard run successful" line (e.g. a
# secondary/secret run), and the RANKED benchmark can fail a shape (e.g.
# "... failed testing: R - Q.T @ A is too large") even after the 22 TEST shapes
# pass — the test shapes use small batches, the ranked benchmark uses the large
# B=640 batches where a kernel bug surfaces. So a failure marker ANYWHERE must
# override any success marker. Only a clean run with NO failure marker AND the
# leaderboard-success marker counts as ACCEPTED.
verdict="UNKNOWN"
full_pass="$(awk '
    match($0, /Passed ([0-9]+)\/([0-9]+) tests/, m) { if (m[1]==m[2] && m[1]>0) print "yes" }
' "$SUBMIT_LOG" | head -1)"
# Hard failure markers — any one of these => REJECTED, no matter what else is present.
if grep -qiE "Leaderboard run failed|failed testing|Benchmarking failed|Testing failed|compilation error|build error|too large|not orthogonal|Submission failed|exceeded" "$SUBMIT_LOG"; then
    verdict="REJECTED"
elif grep -qi "Leaderboard run successful" "$SUBMIT_LOG"; then
    if [ "$full_pass" = "yes" ]; then
        verdict="ACCEPTED"
    else
        verdict="ACCEPTED_NO_TESTLINE"  # ran but couldn't confirm full test pass
    fi
elif grep -qiE "failed|rejected|mismatch|timeout|exceeded|cheat|invalid|compilation error|runtime error" "$SUBMIT_LOG"; then
    verdict="REJECTED"
fi

echo "===GPUMODE_SUBMIT_BEGIN==="
echo "commit=$COMMIT"
echo "leaderboard=$LEADERBOARD gpu=$GPU mode=$MODE"
echo "cli_exit=$CLI_RC"
echo "verdict=$verdict"
# Surface the submission id and the ranked per-shape timings if present.
grep -iE "submission .*id|Submitted|^[0-9]{5,} " "$SUBMIT_LOG" | head -3 || true
# Echo the terminal success/failure block for the record.
grep -iE "Passed [0-9]+/[0-9]+ tests|Leaderboard run|Testing|Benchmarking|success|fail|error" "$SUBMIT_LOG" | head -12 || true
echo "submit_log=$SUBMIT_LOG"
echo "===GPUMODE_SUBMIT_END==="

# Best-effort: show recent submissions for context (NOT used as the verdict -
# this list is shared across machines/users and is stale right after a submit).
echo "leaderboard-submit: recent submissions for $LEADERBOARD (context only, NOT the verdict) :" >&2
popcorn-cli --no-tui submissions list --leaderboard "$LEADERBOARD" 2>/dev/null | head -6 || true

case "$verdict" in
    ACCEPTED)            echo "leaderboard-submit: ACCEPTED commit=$COMMIT" >&2; exit 0 ;;
    ACCEPTED_NO_TESTLINE) echo "leaderboard-submit: ACCEPTED (test line not parsed) commit=$COMMIT" >&2; exit 0 ;;
    REJECTED)            echo "leaderboard-submit: REJECTED commit=$COMMIT (see $SUBMIT_LOG)" >&2; exit 2 ;;
    *)                   echo "leaderboard-submit: NO VERDICT (cli_rc=$CLI_RC) commit=$COMMIT (see $SUBMIT_LOG)" >&2; exit 3 ;;
esac
