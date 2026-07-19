#!/usr/bin/env bash
# Cholesky anti-memorization / input-variation guard.
#
# The benchmark reuses a bounded list of generated tensors across timed
# iterations. Rechecking those same objects cannot detect shape/call-order
# caching or a fast path fitted to the public seeds. Run every official matrix
# family at one published shape with fresh seeds and check each factor against
# its own input. Also exercise the ranked batch-64 n=256 shape twice through
# the same tensor object, replacing its contents between calls, so an
# identity-keyed result cache cannot replay the first factor. No QR-specific
# rank-deficient/off-grid tests are copied here: Cholesky's contract is
# SPD-only and its official generator families already cover the meaningful
# structural variation.
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

    print(
        "cholesky input-variation guard: exact cache shape B=64 n=256, "
        "same-object mutation"
    )
    cache_data = generate_input(
        batch=64, n=256, cond=2, seed=990_001, case="dense"
    )
    cache_identity = id(cache_data)
    cache_pointer = cache_data.data_ptr()
    cache_cases = (("fresh", 990_001), ("mutated", 991_010))
    for index, (phase, seed) in enumerate(cache_cases):
        if index:
            replacement = generate_input(
                batch=64, n=256, cond=2, seed=seed, case="dense"
            )
            cache_data.copy_(replacement)
            del replacement
        reference_input = cache_data.clone()
        torch.cuda.synchronize()
        output = custom_kernel(cache_data)
        torch.cuda.synchronize()
        ok, message = check_implementation(reference_input, output)
        identity_stable = id(cache_data) == cache_identity
        storage_stable = cache_data.data_ptr() == cache_pointer
        ok = ok and identity_stable and storage_stable
        print(
            f"  [{'ok' if ok else 'FAIL':4}] phase={phase:7} seed={seed} "
            f"same_object={identity_stable} same_storage={storage_stable}"
        )
        if not ok:
            if not identity_stable:
                message = "guard failed to preserve the input object identity"
            elif not storage_stable:
                message = "guard failed to preserve the input tensor storage"
            print(f"         {message[:180]}")
            failures.append(f"cache-{phase}")
        del reference_input, output
    del cache_data

    if failures:
        print(f"cholesky input-variation guard: FLAGGED {len(failures)} case(s): {', '.join(failures)}")
        return 2
    print("cholesky input-variation guard: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
