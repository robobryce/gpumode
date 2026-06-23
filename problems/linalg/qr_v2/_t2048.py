"""n=2048 (B=8) focused timer: sweep submission knobs via module globals.
Run wrapped in `autocuda run exclusive`. Times custom_kernel with CUDA events,
clearing L2 between reps to mirror eval.py. Diagnostic only (not a submission)."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
import reference as ref
import submission as sub

dev = torch.device("cuda")
# n=2048 B=8 benchmark shape (cond=1, seed=224466), via reference.generate_input.
data = ref.generate_input(batch=8, n=2048, cond=1, seed=224466)
if isinstance(data, (tuple, list)):
    A = data[0]
else:
    A = data
A = A.to(dev)

# big buffer to flush L2 between reps (B200 L2 ~ 60MB)
l2 = torch.empty(int(80e6 // 4), dtype=torch.float32, device=dev)

def bench(reps=30, warmup=8):
    torch.cuda.synchronize()
    for _ in range(warmup):
        sub.custom_kernel(A)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        l2.zero_()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        sub.custom_kernel(A)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)  # ms->us
    times.sort()
    return times[len(times)//2], times[0]  # median, min (us)

def set_knobs(**kw):
    # The n=2048 trailing knobs are read from os.environ INSIDE _w2_qr each call,
    # so set the environment (not module attrs). _N2048_BLK is a module global
    # read at dispatch, so set that one as an attribute too.
    for k, v in kw.items():
        os.environ[k] = str(v)
    if "QR_N2048_BLK" in kw:
        sub._N2048_BLK = int(kw["QR_N2048_BLK"])

# Parse config list from argv: each "KEY=VAL,KEY=VAL;..." group
configs = []
if len(sys.argv) > 1:
    for grp in sys.argv[1].split(";"):
        d = {}
        label = grp
        for kv in grp.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=")
            d[k] = int(v)
        configs.append((label, d))
else:
    configs = [("default", {})]

print(f"=== n=2048 B=8 timer (median/min us over 30 reps) ===")
for label, d in configs:
    set_knobs(**d)
    med, mn = bench()
    print(f"  {label:40s}  median={med:9.1f}us  min={mn:9.1f}us")
