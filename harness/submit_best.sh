#!/usr/bin/env bash
# Submit the autocuda run's GLOBAL-BEST submission to the GPU MODE leaderboard,
# WITHOUT mutating the working tree.
#
# autocuda keeps each kept optimization on its own branch/commit; `autocuda
# status` reports the single best one as `.global_best.commit`. This script
# resolves that commit's SHA, materializes its `submission.py` straight out of
# git (`git show <SHA>:...` into a temp .py file — no checkout, no stash, no
# branch switch, so the current HEAD and any in-progress edits are untouched),
# and hands that file to popcorn-cli the same way harness/submit.sh does
# (leaderboard name + GPU resolved from the problem's task.yml via gen_specs.py).
#
# popcorn-cli needs a real, seekable .py path — it does NOT read stdin or
# /dev/stdin — hence the mktemp file (cleaned up on exit via trap).
#
# Usage:
#   harness/submit_best.sh --tag <run-tag> [--data-dir <dir>] [--problem <set>/<problem>]
#
#   --tag       REQUIRED. The optimize-tree manager run tag, e.g.
#               2026-06-26-23-45-55-eigh_py
#   --data-dir  autocuda data dir (default: <repo>/autocuda)
#   --problem   <set>/<problem> (default: linalg/eigh_py)
#
# Env:
#   DRY_RUN=1   Print the resolved SHA / leaderboard / GPU and the exact
#               popcorn-cli command line, then exit WITHOUT submitting. Use this
#               to validate the script — a real leaderboard submit is a public,
#               hard-to-reverse action.
#
# Example (dry run):
#   DRY_RUN=1 harness/submit_best.sh --tag 2026-06-26-23-45-55-eigh_py
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HARNESS_DIR/.." && pwd)"

# --- args / defaults ---------------------------------------------------------
DATA_DIR="$REPO_DIR/autocuda"
TAG="${GPUMODE_TAG:-}"
PROBLEM="${GPUMODE_PROBLEM:-linalg/eigh_py}"

usage() {
    sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir) DATA_DIR="${2:?--data-dir needs a value}"; shift 2 ;;
        --tag)      TAG="${2:?--tag needs a value}"; shift 2 ;;
        --problem)  PROBLEM="${2:?--problem needs a value}"; shift 2 ;;
        -h|--help)  usage 0 ;;
        *) echo "submit-best: unknown argument: $1" >&2; usage 1 ;;
    esac
done

[ -n "$TAG" ] || { echo "submit-best: --tag is required (e.g. 2026-06-26-23-45-55-eigh_py)" >&2; usage 1; }

PROBLEM_DIR="$REPO_DIR/problems/$PROBLEM"
TASK_YML="$PROBLEM_DIR/task.yml"
[ -f "$TASK_YML" ] || { echo "submit-best: no task.yml at $TASK_YML — is '$PROBLEM' the right <set>/<problem>?" >&2; exit 1; }

# Resolve PYTHON the same way harness/env.sh does (gpumode.env defines
# GPUMODE_VENV_PYTHON, which install.sh provisions with PyYAML for gen_specs.py).
for cfg in "${GPUMODE_ENV:-}" "$HOME/.config/gpumode/gpumode.env"; do
    if [ -n "$cfg" ] && [ -f "$cfg" ]; then source "$cfg"; break; fi
done
PYTHON="${GPUMODE_VENV_PYTHON:-$HOME/gpumode/.venv/bin/python}"

# --- 1. resolve the global-best commit SHA from `autocuda status` ------------
# Parse JSON with python3 (no jq dependency, per the harness convention).
SHA="$(autocuda status --data-dir "$DATA_DIR" --tag "$TAG" \
    | "$PYTHON" -c 'import sys, json; print(json.load(sys.stdin)["global_best"]["commit"])')"
[ -n "$SHA" ] || { echo "submit-best: could not resolve .global_best.commit for tag '$TAG'" >&2; exit 1; }

# --- 2. materialize that commit's submission.py WITHOUT touching git state ---
# `git show <SHA>:<path>` reads the blob straight out of the object store; no
# checkout / branch switch / stash, so the working tree and HEAD are unchanged.
# popcorn-cli requires a real seekable .py path, so write it to a temp file.
SUBMISSION="$(mktemp -t submit-best-XXXXXX.py)"
trap 'rm -f "$SUBMISSION"' EXIT
git -C "$REPO_DIR" show "$SHA:problems/$PROBLEM/submission.py" > "$SUBMISSION"
[ -s "$SUBMISSION" ] || { echo "submit-best: empty submission.py at $SHA:problems/$PROBLEM/submission.py" >&2; exit 1; }

# --- 3. resolve leaderboard + GPU exactly like harness/submit.sh ------------
LEADERBOARD="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$TASK_YML" --leaderboard)"
# gen_specs.py --gpus prints a CSV of supported GPUs; take the first, as instructed.
GPU="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$TASK_YML" --gpus | cut -d, -f1)"
[ -n "$LEADERBOARD" ] || { echo "submit-best: empty leaderboard name for $PROBLEM" >&2; exit 1; }
[ -n "$GPU" ]         || { echo "submit-best: no supported GPU listed for $PROBLEM in $TASK_YML" >&2; exit 1; }

echo "submit-best: tag=$TAG problem=$PROBLEM" >&2
echo "submit-best: global-best commit=$SHA leaderboard=$LEADERBOARD gpu=$GPU" >&2

# --- 4. submit (or print the command under DRY_RUN) -------------------------
set -- popcorn-cli submit --no-tui \
    --leaderboard "$LEADERBOARD" \
    --gpu "$GPU" \
    --mode leaderboard \
    "$SUBMISSION"

if [ "${DRY_RUN:-0}" != "0" ]; then
    echo "submit-best: DRY_RUN set — not submitting. Command would be:" >&2
    { printf ' '; printf ' %q' "$@"; printf '\n'; } >&2
    exit 0
fi

exec "$@"
