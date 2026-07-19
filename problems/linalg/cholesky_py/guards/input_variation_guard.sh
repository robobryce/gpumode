#!/usr/bin/env bash
# Cholesky anti-memorization / input-variation guard.
#
# The benchmark reuses a bounded list of generated tensors across timed
# iterations. Rechecking those same objects cannot detect shape/call-order
# caching or a fast path fitted to the public seeds. Run every official matrix
# family at one published shape with fresh seeds and check each factor against
# its own input. No QR-specific rank-deficient/off-grid tests are copied here:
# Cholesky's contract is SPD-only and its official generator families already
# cover the meaningful structural variation.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_DIR/harness/env.sh" "$@"
cd "$PROBLEM_DIR" || exit 1

timeout --preserve-status --signal=TERM --kill-after=15s \
    "${INPUT_VARIATION_GUARD_TIMEOUT:-180}" "$PYTHON" - <<'PY'
from __future__ import annotations

import sys
import torch

from reference import check_implementation, generate_input
from submission import custom_kernel

CASES = (
    ("dense", 2),
    ("diagonal", 5),
    ("spectrum", 5),
    ("lowrank", 4),
    ("rowscale", 4),
    ("tridiagonal", 1),
)


def main() -> int:
    if not torch.cuda.is_available():
        print("cholesky input-variation guard: CUDA unavailable", file=sys.stderr)
        return 1

    failures = []
    print("cholesky input-variation guard: official shape B=8 n=256, fresh seeds/cases")
    for index, (case, cond) in enumerate(CASES):
        seed = 890_001 + 1009 * index
        data = generate_input(batch=8, n=256, cond=cond, seed=seed, case=case)
        reference_input = data.clone()
        torch.cuda.synchronize()
        output = custom_kernel(data)
        torch.cuda.synchronize()
        ok, message = check_implementation(reference_input, output)
        print(f"  [{'ok' if ok else 'FAIL':4}] case={case:11} seed={seed}")
        if not ok:
            print(f"         {message[:180]}")
            failures.append(case)
        del data, reference_input, output

    if failures:
        print(f"cholesky input-variation guard: FLAGGED {len(failures)} case(s): {', '.join(failures)}")
        return 2
    print("cholesky input-variation guard: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
