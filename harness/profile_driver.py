#!/usr/bin/env python3
# PROFILE driver — autocuda <-> GPU MODE bridge.
#
# Runs ONE benchmark shape's `custom_kernel` IN ISOLATION under a CUDA
# profiler capture range, so nsys/ncu record only the submission's kernels —
# not eval.py's spawn-Pool indirection and not the cuSOLVER/cuBLAS reference
# checker that `eval.py benchmark` interleaves with every timed call.
#
# Usage:  profile_driver.py <spec> [<warmup>] [<iters>]
#   <spec>   one task.yml benchmark line, e.g. "batch: 640; n: 512; cond: 2; seed: 1029"
#   <warmup> kernel calls before the capture range (default 3)
#   <iters>  kernel calls inside the capture range (default 1)
#
# Imports `custom_kernel` (submission.py) and `generate_input` (reference.py)
# the same way eval.py does — env.sh put their dir on PYTHONPATH — so it profiles
# the exact live code the benchmark scores. Correctness is eval.py's job; this
# driver only shapes a clean capture and is invoked by profile_nsys.sh /
# profile_ncu.sh, never directly.
import re
import sys

import torch
from torch.cuda import profiler

from submission import custom_kernel
from reference import generate_input


def parse_spec(spec: str) -> dict:
    args = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*", part)
        if not m:
            raise SystemExit(f"invalid spec part: '{part}' in '{spec}'")
        key, val = m.group(1), m.group(2)
        try:
            val = int(val)
        except ValueError:
            pass
        args[key] = val
    return args


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: profile_driver.py <spec> [<warmup>] [<iters>]")
    spec = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    args = parse_spec(spec)

    # generate_input puts tensors on the GPU; clone per call so a kernel that
    # mutates its input still sees fresh data, exactly as the benchmark does.
    data = generate_input(**args)

    def clone(d):
        if torch.is_tensor(d):
            return d.clone()
        if isinstance(d, (tuple, list)):
            return type(d)(clone(x) for x in d)
        return d

    for _ in range(warmup):
        custom_kernel(clone(data))
    torch.cuda.synchronize()

    # Only the work between start() and stop() is recorded under
    # `nsys -c cudaProfilerApi` / `ncu --profile-from-start off`.
    profiler.start()
    for _ in range(iters):
        custom_kernel(clone(data))
    torch.cuda.synchronize()
    profiler.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
