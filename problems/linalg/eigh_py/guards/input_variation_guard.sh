#!/usr/bin/env bash
# Eigh anti-memorization / input-variation guard.
#
# The benchmark keeps a bounded batch of generated tensors and reuses it across
# timed iterations. Rechecking those same objects cannot detect a submission
# that caches by shape/call order or dispatches only for the public seed's
# numerical profile. Exercise one fixed, representative shape with fresh seeds
# across every ranked generator case and validate each result against the exact
# input that produced it. The shape is an official public shape; only the seeds
# and case combinations vary. This uses only generate_input and
# check_implementation from the problem and does not expand the shape contract.
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
    ("spectrum", 4),
    ("psd", 2),
    ("rankdef", 0),
    ("nearrank", 0),
    ("repeated", 0),
    ("clustered", 0),
    ("diagonal", 4),
    ("band", 0),
    ("rowscale", 4),
    ("mixed", 2),
    ("lapack_dense_even_spectrum", 0),
    ("lapack_dense_geometric_spectrum", 0),
)


def main() -> int:
    if not torch.cuda.is_available():
        print("eigh input-variation guard: CUDA unavailable", file=sys.stderr)
        return 1

    failures = []
    print("eigh input-variation guard: official shape B=20 n=32, fresh seeds/cases")
    for index, (case, cond) in enumerate(CASES):
        seed = 880_001 + 1009 * index
        data = generate_input(batch=20, n=32, cond=cond, seed=seed, case=case)
        reference_input = data.clone()
        torch.cuda.synchronize()
        output = custom_kernel(data)
        torch.cuda.synchronize()
        ok, message = check_implementation(reference_input, output)
        print(f"  [{'ok' if ok else 'FAIL':4}] case={case:10} seed={seed}")
        if not ok:
            print(f"         {message[:180]}")
            failures.append(case)
        del data, reference_input, output

    if failures:
        print(f"eigh input-variation guard: FLAGGED {len(failures)} case(s): {', '.join(failures)}")
        return 2
    print("eigh input-variation guard: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
