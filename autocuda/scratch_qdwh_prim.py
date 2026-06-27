import torch, time
dev = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

def timeit(fn, iters=10, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

for (B, n) in [(640, 512), (60, 1024), (8, 2048)]:
    A = torch.randn(B, n, n, device=dev, dtype=torch.float32)
    A = 0.5 * (A + A.transpose(-1, -2))
    X = torch.randn(B, n, n, device=dev, dtype=torch.float32)
    stacked = torch.randn(B, 2 * n, n, device=dev, dtype=torch.float32)
    spd = torch.bmm(X, X.transpose(-1, -2)) + n * torch.eye(n, device=dev)  # SPD

    t_base = timeit(lambda: torch.linalg.eigh(A))
    t_qr_red = timeit(lambda: torch.linalg.qr(X, mode="reduced"))
    t_qr_stack = timeit(lambda: torch.linalg.qr(stacked, mode="reduced"))
    t_chol = timeit(lambda: torch.linalg.cholesky(spd))
    Lc = torch.linalg.cholesky(spd)
    t_trsm = timeit(lambda: torch.linalg.solve_triangular(Lc, X, upper=False))
    t_bmm = timeit(lambda: torch.bmm(X, A))
    t_xtx = timeit(lambda: torch.bmm(X.transpose(-1,-2), X))
    # one full QDWH QR-iter cost estimate: qr(stacked) + a couple bmm
    def qdwh_iter():
        Q, R = torch.linalg.qr(stacked, mode="reduced")
        Q1 = Q[:, :n, :]; Q2 = Q[:, n:, :]
        return 0.7 * X + torch.bmm(Q1, Q2.transpose(-1, -2))
    t_iter_qr = timeit(qdwh_iter)
    def qdwh_iter_chol():
        W = torch.linalg.cholesky(spd)
        Y = torch.linalg.solve_triangular(W, X, upper=False)
        Y = torch.linalg.solve_triangular(W.transpose(-1,-2), Y, upper=True)
        return 0.7 * X + Y
    t_iter_chol = timeit(qdwh_iter_chol)

    print(f"B={B} n={n}: eigh={t_base:8.1f} | qr_red={t_qr_red:8.1f} qr_stack(2n)={t_qr_stack:8.1f} "
          f"chol={t_chol:8.1f} trsm={t_trsm:7.1f} bmm={t_bmm:7.1f} | QDWH_iter_QR={t_iter_qr:8.1f} QDWH_iter_chol={t_iter_chol:8.1f}")
    print(f"          -> est 6 QR-iters={6*t_iter_qr:9.1f}  3QR+3chol={3*t_iter_qr+3*t_iter_chol:9.1f}  (vs eigh={t_base:.1f})")
