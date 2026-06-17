#!/usr/bin/env bash
# FUNCTIONAL-INVARIANCE-GUARD — data-fitting / dispatch-bug guard for linalg/qr_v2.
#
# Detects kernels whose correctness silently depends on dimensions the benchmark
# happens to fix: the batch size, the fraction of ill-conditioned matrices, or the
# position of a bad matrix within the batch. Such a kernel passes every benchmark
# and test shape (they never vary those) yet is wrong the moment the same shape is
# presented differently — e.g. a repair-capacity cap (max_bad=min(96,B//4)) that
# drops surplus ill-conditioned matrices, a batch-size-keyed fast path (if B==60),
# or a first-N conditioning sample taken as representative of the whole batch.
#
# It perturbs each of those axes and re-checks against the float64 reference:
#   - BATCH SWEEP            same (n, mix) at several B, past any plausible cap
#   - CONDITIONING SWEEP     fixed B, 0% -> 100% ill-conditioned
#   - POSITION SWEEP         one bad matrix walked across the batch
#
# QR-specific: it perturbs batch / cond / case, which only the linalg/qr*
# generate_input accepts — which is why guards live with their problem rather
# than running for every problem. Discovered and run by harness/validate.sh
# after the test shapes pass, in validate's process under the GPU lock already
# held — do NOT wrap it in its own `autocuda run`. Checker: bin/invariance_guard.py.
#
# Usage:  bash <this> <set>/<problem>   # e.g. linalg/qr_v2
# Exit 0 = invariant (clean), 2 = wrong output under perturbation (data-fitted), else error.
set -uo pipefail
# Repo root is a fixed 4 levels up: problems/<set>/<problem>/guards/<this>.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_DIR/harness/env.sh" "$@"

cd "$PROBLEM_DIR" || exit 1
# env.sh consumes <set>/<problem> ($1); extra tuning flags pass via INVARIANCE_GUARD_ARGS
# (e.g. "--n 1024 --batch 256").
"$PYTHON" "$REPO_DIR/bin/invariance_guard.py" ${INVARIANCE_GUARD_ARGS:-}
