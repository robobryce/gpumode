#!/usr/bin/env bash
# QR v2 functional-invariance guard.
# Perturbs batch size, ill-conditioned fraction, and bad-matrix position for the
# same QR shape to catch data-fitted dispatch assumptions.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_DIR/harness/env.sh" "$@"

cd "$PROBLEM_DIR" || exit 1
"$PYTHON" - ${INVARIANCE_GUARD_ARGS:-} <<'PY'
from __future__ import annotations

import argparse
import sys

import torch

from reference import check_implementation, generate_input
from submission import custom_kernel


def ill_conditioned_count(batch: torch.Tensor) -> int:
    singular_values = torch.linalg.svdvals(batch.float())
    return int((singular_values[:, 0] / singular_values[:, -1].clamp_min(1e-30) > 1e4).sum())


def check(data: torch.Tensor, label: str, failures: list[str]) -> None:
    torch.cuda.synchronize()
    reference_input = data.clone()
    output = custom_kernel(data.clone())
    torch.cuda.synchronize()
    ok, message = check_implementation(reference_input, output)
    status = "ok  " if ok else "FAIL"
    detail = "" if ok else "  " + message.split(";")[0][:70]
    print(f"    [{status}] {label:28} ill-cond={ill_conditioned_count(data):>4}/{data.shape[0]}{detail}")
    if not ok:
        failures.append(label)


def rank_deficient_matrix(n: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn((n, n), device="cuda", dtype=torch.float32, generator=generator)
    matrix[:, n - n // 4:] = 0.0
    return matrix


def batch_sweep(n: int, failures: list[str]) -> None:
    print(f"  BATCH SWEEP n={n} case=mixed cond=2")
    for batch in (32, 96, 160, 640, 1280):
        data = generate_input(batch=batch, n=n, cond=2, seed=4242, case="mixed")
        check(data, f"B={batch}", failures)


def fraction_sweep(n: int, batch: int, failures: list[str]) -> None:
    print(f"  FRACTION SWEEP n={n} B={batch}")
    generator = torch.Generator(device="cuda").manual_seed(7)
    base = torch.randn((batch, n, n), device="cuda", dtype=torch.float32, generator=generator)
    bad = rank_deficient_matrix(n, generator)
    for count in (0, 1, batch // 4, batch // 2, batch):
        data = base.clone()
        if count:
            data[torch.linspace(0, batch - 1, count, device="cuda").long()] = bad
        check(data.contiguous(), f"bad={count}/{batch}", failures)


def position_sweep(n: int, batch: int, failures: list[str]) -> None:
    print(f"  POSITION SWEEP n={n} B={batch}")
    generator = torch.Generator(device="cuda").manual_seed(11)
    base = torch.randn((batch, n, n), device="cuda", dtype=torch.float32, generator=generator)
    bad = rank_deficient_matrix(n, generator)
    for position in (0, 1, 2, 3, batch // 2, batch - 1):
        data = base.clone()
        data[position] = bad
        check(data.contiguous(), f"bad@{position}", failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="QR v2 functional invariance guard")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA not available", file=sys.stderr)
        return 1

    print(f"functional invariance guard: n={args.n}, batch={args.batch}")
    failures: list[str] = []
    batch_sweep(args.n, failures)
    fraction_sweep(args.n, args.batch, failures)
    position_sweep(args.n, args.batch, failures)

    if failures:
        print(f"\nfunctional invariance guard: FLAGGED {len(failures)} perturbation(s)")
        return 2
    print("\nfunctional invariance guard: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
