import torch, time
dev = "cuda"
torch.manual_seed(0)

def timeit(fn, iters=10, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

# how does batched eigh scale with n at fixed total batch?
for (B, n) in [(640, 512), (640, 256), (640, 128), (640, 64),
               (60, 1024), (60, 512), (60, 256),
               (8, 2048), (8, 1024)]:
    A = torch.randn(B, n, n, device=dev, dtype=torch.float32)
    A = 0.5 * (A + A.transpose(-1, -2))
    t = timeit(lambda: torch.linalg.eigh(A))
    print(f"B={B:4d} n={n:5d}: eigh={t:9.1f} us")
