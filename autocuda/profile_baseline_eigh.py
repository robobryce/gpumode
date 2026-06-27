"""Standalone in-process eigh profiler for nsys capture-range.

Mirrors how environment.md says the baseline profile was produced: call
generate_input then the kernel inside torch.cuda.profiler.start()/stop() so
nsys --capture-range=cudaProfilerApi brackets exactly the eigh work.
Profiles the two central benchmark shapes (n=512 batch=640, n=1024 batch=60).
"""
import sys
sys.path.insert(0, "problems/linalg/eigh_py")

import torch
from reference import generate_input

torch.backends.cuda.matmul.allow_tf32 = False  # baseline default


def run_shape(batch, n, cond, seed, case="dense", iters=3):
    data = generate_input(batch=batch, n=n, cond=cond, seed=seed, case=case)
    torch.cuda.synchronize()
    # warmup
    for _ in range(2):
        v, Q = torch.linalg.eigh(data)
    torch.cuda.synchronize()
    for _ in range(iters):
        v, Q = torch.linalg.eigh(data)
    torch.cuda.synchronize()


def main():
    # warmup / compile
    run_shape(640, 512, 2, 1029, iters=1)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    run_shape(640, 512, 2, 1029, iters=3)      # central heavy shape
    run_shape(60, 1024, 2, 75342, iters=3)     # n=1024 shape
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
