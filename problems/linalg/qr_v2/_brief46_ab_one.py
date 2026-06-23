"""brief-46 COMBINE A/B single-arm timer (one process = one arm).

Times the ranked benchmark shapes with the SAME L2-clear + CUDA-event timing the
harness uses (util=0 required). The _2L_PNW constexpr is read at import via
QR_2L_PNW, so each ARM is a fresh process:
  QR_2L_PNW=0  -> parent P1 (n4096 tall sub-panel keeps in-code nwp=32)
  QR_2L_PNW=8  -> candidate (n4096 tall sub-panel nwp=8; default after graft)

Prints one line per shape (us) and the geomean over all shapes, so an outer
interleave loop can compare NET geomean AND the isolated n4096 shape (the only
shape _2L_PNW touches) round-by-round. Default-context launches only.
"""
import os, sys, math, torch
import submission as S


def make(n, b, seed, case):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(b, n, n, device="cuda", dtype=torch.float32, generator=g)
    # The conditioning 'case' only changes input numerics, not which kernel path
    # runs or its timing for the nwp knob, so for a TIMING A/B a plain random
    # input at the right (n,b) is representative. (Correctness is gated by
    # validate, not here.)
    return A


# ranked benchmark shapes (from gen_specs --emit benchmarks)
SHAPES = [
    (32, 20, 43214), (176, 40, 423011), (352, 40, 123456),
    (512, 640, 1029), (1024, 60, 75342), (2048, 8, 224466),
    (4096, 2, 32412),
    (512, 640, 770001), (1024, 60, 770002),
    (512, 640, 770003), (512, 640, 770004), (1024, 60, 770005),
]


def time_shape(A, reps, warmup):
    for _ in range(warmup):
        S.custom_kernel(A)
    torch.cuda.synchronize()
    flush = torch.empty(int(64e6), dtype=torch.int8, device="cuda")
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        flush.zero_()
        torch.cuda.synchronize()
        e0.record()
        S.custom_kernel(A)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)  # us
    ts.sort()
    return sum(ts) / len(ts), ts[len(ts) // 2], ts[0]


if __name__ == "__main__":
    pnw = os.environ.get("QR_2L_PNW", "(unset)")
    # small shapes are noisy -> more reps; n4096 is the target -> generous reps
    means = {}
    out = []
    for n, b, seed in SHAPES:
        A = make(n, b, seed, None)
        reps = 50 if n <= 512 else (40 if n <= 1024 else (30 if n <= 2048 else 40))
        warm = 10
        mean, med, mn = time_shape(A, reps, warm)
        means[(n, seed)] = mean
        out.append(f"  n={n:5d} B={b:3d} seed={seed:7d}: mean={mean:9.2f} med={med:9.2f} min={mn:9.2f}")
        del A
        torch.cuda.empty_cache()
    geo = math.exp(sum(math.log(v) for v in means.values()) / len(means))
    n4096 = means[(4096, 32412)]
    print(f"ARM QR_2L_PNW={pnw}")
    print("\n".join(out))
    print(f"  GEOMEAN(all12)={geo:9.2f}us   n4096={n4096:9.2f}us")
    # machine-readable tail
    print(f"RESULT pnw={pnw} geomean={geo:.4f} n4096={n4096:.4f}")
