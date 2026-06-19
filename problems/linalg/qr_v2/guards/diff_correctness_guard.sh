#!/usr/bin/env bash
# QR v2 differential-correctness guard.
# Runs distinct inputs for each benchmark shape in one process to expose stale
# caches or call-ordinal shortcuts that only work when eval repeats one input.
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
