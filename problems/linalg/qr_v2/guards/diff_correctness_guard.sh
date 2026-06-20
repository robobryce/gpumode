#!/usr/bin/env bash
# QR v2 differential-correctness guard.
#
# WHAT IT CATCHES — a kernel whose output does not depend on the current input's
# bytes: one that computes the factorization once (or keys a cache on the call
# ordinal, or on the shape) and then replays that answer for every later call. An
# honest kernel passes this guard trivially; it only ever changes the verdict for
# such a "memorizer".
#
# WHY EVAL MISSES IT — on the large benchmark shapes eval's _benchmark_batch_count
# caps the generated inputs at count=1 (the 256 MiB BENCHMARK_INPUT_BYTES_TARGET;
# e.g. 640x512 and 60x1024 each exceed it in a single draw), so the timed loop
# reuses ONE input object across all repeats. recheck=True re-validates every
# timed iteration, but always against that single reused input — which the
# memorizer answers correctly. So it is scored as the fastest submission and is
# never contradicted locally.
#
# HOW — re-runs the benchmark specs but feeds --seq (default 8) DISTINCT inputs
# per shape (fresh seeds base+1009*(index+1), plus a case rotation) and checks
# each output against its own input. A replayed/stale answer produces a blown-up
# QR residual on every input after the first -> FLAGGED.
#
# DOES NOT add or change shapes: (batch, n, cond) are copied verbatim from each
# benchmark spec — only the seed changes, and the case rotates where a `case`
# field already exists. It does not use the test specs at all.
#
# BOUNDARY — it cannot catch a cache keyed on the actual input content: 8 distinct
# inputs are 8 cache misses -> 8 correct recomputes -> CLEAN, yet that cache still
# beats eval for the same count=1 reason. This guard narrows that hole; it does
# not seal it.
#
# ENABLED: this guard runs on every validate (harness/validate.sh globs and runs
# guards/*.sh after the test shapes pass). It is an anti-reward-hack check, not a
# correctness gate, and it spends real GPU time on the large shapes — that cost is
# accepted as the price of catching memorizer/cache hacks the test shapes miss.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_DIR/harness/env.sh" "$@"

SPECS="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks)"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE"' EXIT
printf '%s' "$SPECS" > "$SPECFILE"

cd "$PROBLEM_DIR" || exit 1
"$PYTHON" - "$SPECFILE" ${DIFF_GUARD_ARGS:-} <<'PY'
from __future__ import annotations

import argparse
import sys

import torch

from reference import check_implementation, generate_input
from submission import custom_kernel

CASES = ("mixed", "rankdef", "clustered", "nearrank", "dense")


def spec_args(line: str) -> dict:
    args = {}
    for part in line.split(";"):
        if ":" not in part:
            continue
        key, value = (x.strip() for x in part.split(":", 1))
        for cast in (int, float):
            try:
                args[key] = cast(value)
                break
            except ValueError:
                pass
        else:
            args[key] = value
    return args


def clone_for_reference(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_for_reference(x) for x in value)
    if isinstance(value, list):
        return [clone_for_reference(x) for x in value]
    if isinstance(value, dict):
        return {k: clone_for_reference(v) for k, v in value.items()}
    return value


def variants(kwargs: dict, count: int):
    base_seed = int(kwargs.get("seed", 0))
    for index in range(count):
        variant = dict(kwargs)
        if "seed" in variant:
            variant["seed"] = base_seed + 1009 * (index + 1)
        if "case" in variant:
            variant["case"] = CASES[index % len(CASES)]
        yield variant


def main() -> int:
    parser = argparse.ArgumentParser(description="QR v2 differential correctness guard")
    parser.add_argument("specfile")
    parser.add_argument("--seq", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA not available", file=sys.stderr)
        return 1

    specs = [line.strip() for line in open(args.specfile) if line.strip()]
    if not specs:
        print("error: no benchmark specs", file=sys.stderr)
        return 1

    print(f"differential correctness guard: {len(specs)} shape(s), {args.seq} input(s) each")
    flagged = []
    for spec in specs:
        failures = []
        for call, kwargs in enumerate(variants(spec_args(spec), args.seq)):
            data = generate_input(**kwargs)
            torch.cuda.synchronize()
            reference_input = clone_for_reference(data)
            output = custom_kernel(data)
            torch.cuda.synchronize()
            ok, message = check_implementation(reference_input, output)
            if not ok:
                failures.append((call, kwargs.get("case", ""), kwargs.get("seed", ""), message.split(";")[0][:80]))
        label = spec[:52]
        if failures:
            print(f"  [{label:52}] FAIL — {len(failures)}/{args.seq} inputs wrong")
            for call, case, seed, message in failures:
                print(f"      call {call}: case={case} seed={seed} {message}")
            flagged.append(spec)
        else:
            print(f"  [{label:52}] ok")

    if flagged:
        print(f"\ndifferential correctness guard: FLAGGED {len(flagged)} shape(s)")
        return 2
    print("\ndifferential correctness guard: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
