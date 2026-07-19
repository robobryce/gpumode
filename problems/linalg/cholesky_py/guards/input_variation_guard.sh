#!/usr/bin/env bash
# Cholesky anti-memorization / input-variation guard.
#
# The benchmark reuses a bounded list of generated tensors across timed
# iterations. Rechecking those same objects cannot detect shape/call-order
# caching or a fast path fitted to the public seeds. Run every official matrix
# family at one published shape with fresh seeds and check each factor against
# its own input. Also exercise every exact shape with a benchmark-sized cache
# through silently mutated tensor storage, alternating identities, cross-shape
# re-entry, and different caller streams. Keeping the object's identity,
# pointer, and PyTorch version unchanged prevents those values from serving as
# stale-result cache keys. No QR-specific rank-deficient/off-grid tests are
# copied here: Cholesky's contract is SPD-only and its official generator
# families already cover the meaningful structural variation.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_DIR/harness/env.sh" "$@"
cd "$PROBLEM_DIR" || exit 1

timeout --preserve-status --signal=TERM --kill-after=15s \
    "${INPUT_VARIATION_GUARD_TIMEOUT:-300}" "$PYTHON" - <<'PY'
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

CACHE_SHAPES = (
    (64, 256),
    (640, 512),
    (60, 1024),
    (8, 2048),
)

SIGNATURE_FIELDS = (
    "identity",
    "storage",
    "version",
    "shape",
    "stride",
    "dtype",
    "device",
)


def tensor_signature(data: torch.Tensor) -> tuple[object, ...]:
    return (
        id(data),
        data.data_ptr(),
        data._version,
        tuple(data.shape),
        tuple(data.stride()),
        data.dtype,
        data.device,
    )


def run_cache_case(
    *,
    data: torch.Tensor,
    expected_signature: tuple[object, ...],
    batch: int,
    n: int,
    phase: str,
    seed: int,
    failures: list[str],
    replacement: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    # `.data` is deliberate here. Its alias has an independent version counter,
    # so copying through it changes the bytes without changing data._version.
    # That makes this an adversarial check for identity-, pointer-, and
    # version-keyed result replay rather than an ordinary mutation test.
    torch.cuda.synchronize()
    if stream is None:
        if replacement is not None:
            storage_alias = data.data
            storage_alias.copy_(replacement)
            del storage_alias
        reference_input = data.clone()
        output = custom_kernel(data)
        checked_output = output.clone()
        caller_stream = "default"
    else:
        with torch.cuda.stream(stream):
            if replacement is not None:
                storage_alias = data.data
                storage_alias.copy_(replacement)
                del storage_alias
            reference_input = data.clone()
            output = custom_kernel(data)
            checked_output = output.clone()
        stream.synchronize()
        caller_stream = "alternate"
    torch.cuda.synchronize()

    # The clone was enqueued immediately on the caller stream. A submission
    # that reused a callable bound to a stale stream cannot rely on the global
    # synchronization above to conceal missing caller-stream ordering.
    correct, message = check_implementation(reference_input, checked_output)
    actual_signature = tensor_signature(data)
    changed_fields = [
        name
        for name, expected, actual in zip(
            SIGNATURE_FIELDS, expected_signature, actual_signature, strict=True
        )
        if expected != actual
    ]
    ok = correct and not changed_fields
    print(
        f"  [{'ok' if ok else 'FAIL':4}] B={batch:<3} n={n:<4} "
        f"phase={phase:12} seed={seed} stream={caller_stream:9} "
        f"same_object={actual_signature[0] == expected_signature[0]} "
        f"same_storage={actual_signature[1] == expected_signature[1]} "
        f"same_version={actual_signature[2] == expected_signature[2]}"
    )
    if not correct:
        print(f"         {message[:180]}")
    if changed_fields:
        print(f"         input metadata changed: {', '.join(changed_fields)}")
    if not ok:
        failures.append(f"B{batch}-n{n}-{phase}")
    del reference_input, output, checked_output


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
        "cholesky input-variation guard: exact cached shapes, silent mutation "
        "and alternating identities"
    )
    cache_states = []
    seed_step = 1009
    for index, (batch, n) in enumerate(CACHE_SHAPES):
        first_seed = 990_001 + 10_009 * index
        first = generate_input(
            batch=batch, n=n, cond=2, seed=first_seed, case="dense"
        )
        first_signature = tensor_signature(first)
        run_cache_case(
            data=first,
            expected_signature=first_signature,
            batch=batch,
            n=n,
            phase="A0-fresh",
            seed=first_seed,
            failures=failures,
        )

        mutation_seed = first_seed + seed_step
        replacement = generate_input(
            batch=batch, n=n, cond=2, seed=mutation_seed, case="dense"
        )
        run_cache_case(
            data=first,
            expected_signature=first_signature,
            batch=batch,
            n=n,
            phase="A1-mutated",
            seed=mutation_seed,
            failures=failures,
            replacement=replacement,
        )
        del replacement

        second_seed = first_seed + 2 * seed_step
        second = generate_input(
            batch=batch, n=n, cond=2, seed=second_seed, case="dense"
        )
        second_signature = tensor_signature(second)
        distinct_inputs = (
            first_signature[0] != second_signature[0]
            and first_signature[1] != second_signature[1]
        )
        if not distinct_inputs:
            print(f"  [FAIL] B={batch:<3} n={n:<4} A and B alias unexpectedly")
            failures.append(f"B{batch}-n{n}-distinct-inputs")
        run_cache_case(
            data=second,
            expected_signature=second_signature,
            batch=batch,
            n=n,
            phase="B0-fresh",
            seed=second_seed,
            failures=failures,
        )
        cache_states.append(
            (batch, n, first_seed, first, first_signature, second, second_signature)
        )

    print(
        "cholesky input-variation guard: cross-shape re-entry across "
        "alternate and default caller streams"
    )
    alternate_stream = torch.cuda.Stream()
    # Forward setup ends at n=2048. Start re-entry at n=1024, then alternate
    # sizes so every cached object is revisited after a different shape.
    reentry_states = cache_states[2:4] + cache_states[1::-1]
    for (
        batch,
        n,
        first_seed,
        first,
        first_signature,
        second,
        second_signature,
    ) in reentry_states:
        mutation_seed = first_seed + 3 * seed_step
        replacement = generate_input(
            batch=batch, n=n, cond=2, seed=mutation_seed, case="dense"
        )
        run_cache_case(
            data=first,
            expected_signature=first_signature,
            batch=batch,
            n=n,
            phase="A2-reentry",
            seed=mutation_seed,
            failures=failures,
            replacement=replacement,
            stream=alternate_stream,
        )
        del replacement

        # Revisit the same entry on the default stream as well. This catches a
        # callable or queue wrapper that rebinds once but remains stuck on the
        # alternate stream for later calls.
        default_mutation_seed = first_seed + 4 * seed_step
        replacement = generate_input(
            batch=batch,
            n=n,
            cond=2,
            seed=default_mutation_seed,
            case="dense",
        )
        run_cache_case(
            data=first,
            expected_signature=first_signature,
            batch=batch,
            n=n,
            phase="A3-default",
            seed=default_mutation_seed,
            failures=failures,
            replacement=replacement,
        )
        del replacement

        # The n=2048 benchmark retains two input objects. Complete its
        # two-object B,A,B alternation before another shape can clear the cache.
        if (batch, n) == (8, 2048):
            second_mutation_seed = first_seed + 5 * seed_step
            replacement = generate_input(
                batch=batch,
                n=n,
                cond=2,
                seed=second_mutation_seed,
                case="dense",
            )
            run_cache_case(
                data=second,
                expected_signature=second_signature,
                batch=batch,
                n=n,
                phase="B1-reentry",
                seed=second_mutation_seed,
                failures=failures,
                replacement=replacement,
            )
            del replacement
    del cache_states, reentry_states

    if failures:
        print(f"cholesky input-variation guard: FLAGGED {len(failures)} case(s): {', '.join(failures)}")
        return 2
    print("cholesky input-variation guard: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
