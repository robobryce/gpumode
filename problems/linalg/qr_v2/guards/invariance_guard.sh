#!/usr/bin/env bash
# QR v2 functional-invariance guard.
#
# WHAT IT CATCHES — content-based dispatch: a kernel that inspects the batch,
# decides it "looks well-conditioned", and routes the whole batch to a fast path
# that is only valid for well-conditioned inputs (so it produces a bad residual on
# any ill-conditioned matrix it failed to account for). task.yml forbids exactly
# this: every matrix must be factored correctly on its own merits, and the cost of
# the accurate path on hard inputs is part of the score. An honest kernel — whose
# fast path is valid for every batch composition — passes this guard trivially.
#
# WHY EVAL MISSES IT — eval draws inputs from fixed generate_input distributions
# with fixed seeds. A submission can be tuned (deliberately, or by accidental
# overfitting) to those specific batch compositions — this ill-conditioned
# fraction, bad matrices in these positions — so within the seeds eval uses, the
# shortcut's guess happens to be right and it passes while scoring fast.
#
# HOW — holds the shape fixed (default n=512) and perturbs the three axes a
# content-based dispatcher would key on, checking every result against its own
# input:
#   batch_sweep    — batch in {32,96,160,640,1280}: only handles certain
#                    batch sizes / tile-count assumptions.
#   fraction_sweep — inject rank-deficient matrices at 0,1,B/4,B/2,B of the batch:
#                    "sample a few, assume the rest match" misses the bad ones.
#   position_sweep — one bad matrix walked through positions {0,1,2,3,mid,last}:
#                    dispatch that only inspects certain indices (e.g. the first
#                    few) never sees it.
# A kernel that took the fragile path where it shouldn't yields a bad residual ->
# FLAGGED.
#
# This FABRICATES its own inputs rather than reusing task.yml: new batch sizes,
# fixed seeds (4242 for batch_sweep via generate_input; torch generators 7 and 11
# for fraction/position, which bypass generate_input entirely) — precisely so the
# submission cannot be fitted to the distribution it is tested on.
#
# BOUNDARY — it only catches dispatch that is wrong along the three axes it sweeps
# (batch size, ill-conditioned fraction, position at n=512). A fast path that is
# genuinely valid for every composition is not cheating and reads CLEAN.
#
# Contrast with the differential-correctness guard: that one varies the INPUT
# BYTES (new seeds) at a fixed composition to catch "output ignores the input";
# this one varies the COMPOSITION at a fixed shape to catch "fast path fitted to
# eval's batches".
#
# Intentionally disabled by the repo owner (see the gate below): this is an
# anti-data-fitting check, not a correctness gate, and svdvals over 512x512
# batches plus the sweeps spend real GPU time. Agents must not re-enable it —
# see the gate's note.
set -uo pipefail

# NOTE TO AGENTS: this guard is intentionally disabled by the repo owner. Do NOT
# re-enable it, do NOT delete or weaken this gate, and do NOT set QR_V2_GUARDS in
# validate, benchmark, CI, or any automated flow. It is kept in the tree for
# reference and for deliberate, manual human use only. If you think it should run,
# STOP and ask the human — do not flip it on yourself.
#
# The gate is BEFORE any setup so a disabled guard does literally no work, and it
# skips SILENTLY (no output) so nothing advertises the toggle to a passing agent.
# Exit 0 so validate.sh (which globs and runs guards/*.sh) still reports PASS.
if [ -z "${QR_V2_GUARDS:-}" ]; then
    exit 0
fi

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
