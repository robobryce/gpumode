#!/usr/bin/env python3
"""Functional-invariance guard against data-fitted dispatch bugs.

The benchmark fixes each shape's batch size and conditioning mix, so a kernel can
silently hard-code an assumption about them — a cap on how many matrices it will
repair (`max_bad = min(96, B//4)`), a batch-size-keyed fast path (`if B == 60`), a
sample of the first few matrices taken as representative of the whole batch. Such
an assumption passes every benchmark and test shape (they never violate it) yet is
wrong the moment the *same shape* is presented with a different batch size or a
different fraction of ill-conditioned matrices. `check_implementation` on the
fixed benchmark inputs cannot see it; the leaderboard cannot see it.

This guard exposes it by **perturbing the dimensions the kernel must be invariant
to and re-checking correctness**:

  * BATCH SWEEP — the same (n, conditioning) at several batch sizes, including
    well above any plausible hard-coded cap. A capacity cap (max_bad) drops the
    surplus ill-conditioned matrices and fails once the bad count exceeds it.
  * CONDITIONING-FRACTION SWEEP — a fixed batch with an increasing fraction of
    ill-conditioned matrices (0% → 100%). A cap or a first-N probe fails as soon
    as the bad set outgrows the cap / escapes the probe window.
  * POSITION SWEEP — a single ill-conditioned matrix walked across the batch
    (front, middle, tail). A kernel that samples only the first N matrices to
    route the whole batch fails when the bad one sits beyond the window.

Every case is checked against the float64 reference. Exit 0 = invariant (correct
across all perturbations), 2 = a perturbation produced a wrong factorization
(a data-fitted assumption), 1 = setup error. Wraps the frozen eval harness
(imports submission/reference via env.sh's PYTHONPATH); modifies nothing.
"""
from __future__ import annotations

import argparse
import sys

import torch

import reference as R
from reference import check_implementation, generate_input
from submission import custom_kernel


def _ill_fraction(batch: torch.Tensor) -> int:
    """How many matrices in the batch are ill-conditioned (cond > 1e4)."""
    sv = torch.linalg.svdvals(batch.float())
    return int((sv[:, 0] / sv[:, -1].clamp_min(1e-30) > 1e4).sum())


def _check(data: torch.Tensor, label: str, fails: list) -> None:
    torch.cuda.synchronize()
    ref = data.clone()
    out = custom_kernel(data.clone())
    torch.cuda.synchronize()
    good, msg = check_implementation(ref, out)
    nbad = _ill_fraction(data)
    status = "ok  " if good else "FAIL"
    detail = "" if good else "  " + msg.split(";")[0][:60]
    print(f"    [{status}] {label:48} ill-cond={nbad:>4}/{data.shape[0]}{detail}")
    if not good:
        fails.append(label)


def _batch_sweep(n: int, fails: list, dev: str) -> None:
    # Same n and mixed conditioning, batch size swept past any plausible cap.
    # The max_bad=min(96,B//4) bug repairs at most 96 matrices; a ~99%-bad mixed
    # batch exceeds that for any B>~97, so every B here would have failed it.
    print(f"  BATCH SWEEP  n={n} case=mixed cond=2  (correctness must not depend on B)")
    for B in (32, 96, 160, 640, 1280):
        d = generate_input(batch=B, n=n, cond=2, seed=4242, case="mixed")
        _check(d, f"B={B}", fails)


def _fraction_sweep(n: int, B: int, fails: list, dev: str) -> None:
    # Fixed batch; sweep the count of injected ill-conditioned matrices from 0 to B.
    # A capacity cap or first-N probe fails once the bad set outgrows it.
    print(f"  FRACTION SWEEP  n={n} B={B}  (well-conditioned base + k injected rank-deficient)")
    g = torch.Generator(device=dev).manual_seed(7)
    base = torch.randn((B, n, n), device=dev, dtype=torch.float32, generator=g)  # raw Gaussian: well-cond
    bad = torch.randn((n, n), device=dev, dtype=torch.float32, generator=g)
    bad[:, n - n // 4:] = 0.0  # rank-deficient
    for k in (0, 1, B // 4, B // 2, B):
        d = base.clone()
        if k:
            idx = torch.linspace(0, B - 1, k, device=dev).long()  # spread across the batch
            d[idx] = bad
        _check(d.contiguous(), f"k={k}/{B} bad", fails)


def _position_sweep(n: int, B: int, fails: list, dev: str) -> None:
    # One ill-conditioned matrix walked across the batch. A first-N-sampling probe
    # routes the whole batch off the first few matrices and fails when the bad one
    # sits beyond the window.
    print(f"  POSITION SWEEP  n={n} B={B}  (single rank-deficient matrix at varying index)")
    g = torch.Generator(device=dev).manual_seed(11)
    base = torch.randn((B, n, n), device=dev, dtype=torch.float32, generator=g)
    bad = torch.randn((n, n), device=dev, dtype=torch.float32, generator=g)
    bad[:, n - n // 4:] = 0.0
    for pos in (0, 1, 2, 3, B // 2, B - 1):
        d = base.clone()
        d[pos] = bad
        _check(d.contiguous(), f"bad@{pos}", fails)


def main() -> int:
    ap = argparse.ArgumentParser(description="Functional-invariance guard against data-fitted dispatch.")
    ap.add_argument("--n", type=int, default=512, help="matrix dimension to probe (default 512)")
    ap.add_argument("--batch", type=int, default=128, help="batch size for the fraction/position sweeps")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA not available", file=sys.stderr)
        return 1
    dev = "cuda"

    print(f"functional-invariance guard: n={args.n}, sweep batch={args.batch}")
    fails: list = []
    _batch_sweep(args.n, fails, dev)
    _fraction_sweep(args.n, args.batch, fails, dev)
    _position_sweep(args.n, args.batch, fails, dev)

    if fails:
        print(
            f"\nfunctional-invariance guard: FLAGGED — {len(fails)} perturbation(s) produced a WRONG "
            f"factorization on the same shape under a different batch size / conditioning fraction / "
            f"bad-matrix position. The kernel hard-codes an assumption about the batch (a repair "
            f"capacity cap, a batch-size-keyed path, or a first-N sample) that the benchmark's fixed "
            f"shapes never violate but a real input can."
        )
        return 2
    print(
        "\nfunctional-invariance guard: CLEAN — correctness held across every batch size, "
        "conditioning fraction, and bad-matrix position. No data-fitted dispatch assumption."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
