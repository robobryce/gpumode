#!/usr/bin/env bash
# DIFFERENTIAL-CORRECTNESS-GUARD — anti-reward-hack guard for linalg/qr_py.
#
# Detects benchmark-loop caching reward-hacks by correctness, not timing: runs a
# sequence of DISTINCT inputs of each benchmark shape through one process and
# checks every output against the reference. A kernel that caches/assumes a prior
# input's result (skipping the conditioning probe) answers a later, different
# input wrongly and fails the check. An honest kernel recomputes and passes all.
#
# This catches what eval.py's `recheck` cannot: recheck re-validates repeated
# calls on the SAME reused input (where the cache is still valid); this guard
# varies the input so a stale cache is exposed.
#
# This guard lives with its problem (problems/<set>/<problem>/guards/) so it is
# discovered and run by harness/validate.sh after the test shapes pass. It runs
# in validate's process under the GPU lock already held — do NOT wrap it in its
# own `autocuda run`. The shared checker is bin/diff_correctness_guard.py.
#
# Usage:  bash <this> <set>/<problem>   # e.g. linalg/qr_v2
# Exit 0 = clean, 2 = wrong output on a distinct input (reward-hack), else error.
set -uo pipefail
# Repo root is a fixed 4 levels up: problems/<set>/<problem>/guards/<this>.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_DIR/harness/env.sh" "$@"

SPECS="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks)"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s' "$SPECS" > "$SPECFILE"

cd "$PROBLEM_DIR" || exit 1
# env.sh consumes <set>/<problem> ($1); the guard takes the specfile plus any
# tuning flags via DIFF_GUARD_ARGS (e.g. "--seq 12").
"$PYTHON" "$REPO_DIR/bin/diff_correctness_guard.py" "$SPECFILE" ${DIFF_GUARD_ARGS:-}
