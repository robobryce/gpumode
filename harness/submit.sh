#!/usr/bin/env bash
# Submit the selected GPU MODE problem's current submission.py to the public
# leaderboard. This is mandatory evidence for autocuda runs: baseline setup,
# meaningful improvements, and final candidates must all be submitted.
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
popcorn-cli submit --no-tui \
    --leaderboard "$LEADERBOARD" \
    --gpu "$GPU" \
    --mode "$MODE" \
    "$PROBLEM_DIR/submission.py"

echo "leaderboard-submit: submissions for $LEADERBOARD after commit=$COMMIT" >&2
popcorn-cli --no-tui submissions list --leaderboard "$LEADERBOARD"
