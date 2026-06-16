#!/usr/bin/env python3
"""Differential correctness guard against benchmark-loop reward-hacks.

A benchmark-loop hack skips real work on the calls the eval harness times by
carrying state across calls — an input-identity cache (keyed on `id(input)` /
`input.data_ptr()` / a weakref) or a call-ordinal counter that, once armed,
*assumes* a cached result (e.g. which matrices failed a conditioning probe)
still applies to the current input. That assumption is only valid because the
harness's timed loop feeds the kernel the *same* input repeatedly.

The clean way to expose it is correctness, not timing: feed the kernel a
**sequence of distinct inputs of the same shape whose correct answers differ**,
all in one process, and check every output against the reference. An honest
kernel recomputes each input and passes them all. A kernel that reuses a cached
result / skips the probe answers a later input as if it were an earlier one and
**fails the reference check** — a hard, binary signal with no timing threshold,
no warmup sensitivity, and no false positives from JIT or autotuning.

This is what the eval harness's `recheck` cannot catch: recheck re-validates the
output of repeated calls on the *same reused input*, where the cache is still
valid. This guard deliberately varies the input between calls so a stale cache
produces a wrong answer.

It wraps the frozen `eval.py` (imports the same `submission.custom_kernel`,
`reference.generate_input`, and `reference.check_implementation` via env.sh's
PYTHONPATH); it does not modify the harness. Exit 0 = clean, 2 = a kernel failed
on a distinct input it had previously been primed against, 1 = setup error.
"""
from __future__ import annotations

import argparse
import sys

import torch

from reference import check_implementation, generate_input  # via PYTHONPATH (env.sh)
from submission import custom_kernel


def _spec_to_args(spec: str) -> dict:
    args: dict = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = (s.strip() for s in part.split(":", 1))
        for cast in (int, float):
            try:
                args[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            args[k] = v
    return args


def _ref_copy(data):
    if isinstance(data, tuple):
        return tuple(_ref_copy(x) for x in data)
    if isinstance(data, list):
        return [_ref_copy(x) for x in data]
    if isinstance(data, dict):
        return {k: _ref_copy(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    return data


# Conditioning cases the qr_v2 generator supports; alternating them between calls
# maximally varies the per-matrix bad/good pattern a probe-skipping cache assumes.
_CASES = ["mixed", "rankdef", "clustered", "nearrank", "dense"]


def _variants(kwargs: dict, n: int) -> list[dict]:
    """n distinct inputs of the same shape: step the seed every call, and rotate
    the conditioning case if the spec carries one, so consecutive inputs have
    genuinely different correct answers (different bad-matrix sets)."""
    base_seed = int(kwargs.get("seed", 0))
    has_case = "case" in kwargs
    out = []
    for i in range(n):
        kw = dict(kwargs)
        if "seed" in kw:
            kw["seed"] = base_seed + 1009 * (i + 1)
        if has_case:
            kw["case"] = _CASES[i % len(_CASES)]
        out.append(kw)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Differential correctness check against benchmark-loop caches.")
    ap.add_argument("specfile")
    ap.add_argument(
        "--seq", type=int, default=8,
        help="distinct inputs per shape, run in sequence in one process (default 8)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA not available", file=sys.stderr)
        return 1
    specs = [ln.strip() for ln in open(args.specfile) if ln.strip()]
    if not specs:
        print("error: no specs", file=sys.stderr)
        return 1

    print(f"differential correctness guard: {len(specs)} shape(s), {args.seq} distinct inputs each")
    flagged = []
    for spec in specs:
        kwargs = _spec_to_args(spec)
        fails = []
        for i, kw in enumerate(_variants(kwargs, args.seq)):
            data = generate_input(**kw)
            torch.cuda.synchronize()
            ref = _ref_copy(data)
            out = custom_kernel(data)
            torch.cuda.synchronize()
            good, msg = check_implementation(ref, out)
            if not good:
                fails.append((i, kw.get("case", ""), kw.get("seed", ""), msg.split(";")[0][:70]))
        tag = spec[:46]
        if fails:
            print(f"  [{tag:46}] FAIL — {len(fails)}/{args.seq} distinct inputs produced wrong output:")
            for i, case, seed, msg in fails:
                print(f"        call {i} (case={case} seed={seed}): {msg}")
            flagged.append(spec)
        else:
            print(f"  [{tag:46}] ok — all {args.seq} distinct inputs validated")

    if flagged:
        print(
            f"\ndifferential correctness guard: FLAGGED — {len(flagged)} shape(s) produced a WRONG "
            f"factorization on a distinct input run after a different one in the same process. The "
            f"kernel is reusing a cached/assumed result instead of recomputing — a benchmark-loop "
            f"reward-hack that only validates because the real benchmark reuses one input object."
        )
        return 2
    print(
        "\ndifferential correctness guard: CLEAN — every distinct input validated; the kernel "
        "recomputes per input and carries no stale cross-call assumption."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
