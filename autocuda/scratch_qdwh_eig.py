import torch, time
dev = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

def chol_qdwh_polar(M, iters=6):
    # Polar factor U of M (symmetric) via Cholesky-based dynamically weighted Halley.
    # M assumed prescaled so ||M||2 ~ O(1). Returns U ~ sign(M).
    B, n, _ = M.shape
    I = torch.eye(n, device=dev, dtype=M.dtype)
    # alpha estimate >= ||M||2 (Frobenius is an upper bound)
    alpha = torch.linalg.matrix_norm(M, ord='fro', dim=(-2,-1)).clamp_min(1e-20).view(B,1,1)
    X = M / alpha
    # lower bound l0 on smallest singular value of X; estimate via 1/(||X^-1||) is costly,
    # use a small floor and let DWH self-correct.
    l = torch.full((B,1,1), 1e-3, device=dev, dtype=M.dtype)
    for k in range(iters):
        # DWH weights from l
        d = (4.0*(1.0-l*l)/(l*l*l*l))**(1.0/3.0)
        a = torch.sqrt(1.0 + d) + 0.5*torch.sqrt(8.0 - 4.0*d + 8.0*(2.0-l*l)/(l*l*torch.sqrt(1.0+d)))
        b = (a - 1.0)**2 / 4.0
        c = a + b - 1.0
        # Cholesky form: Z = I + c X^T X ; W = chol(Z); X = (b/c)X + (a - b/c)(X W^-1)W^-T
        Z = I + c * torch.bmm(X.transpose(-1,-2), X)
        W = torch.linalg.cholesky(Z)
        Y = torch.linalg.solve_triangular(W, X.transpose(-1,-2), upper=False)  # solve W Y = X^T
        # X W^-1 W^-T = (W^-T (W^-1 X^T))^T ... compute T = solve(W, X^T)=Y; then solve(W^T, Y)
        Y2 = torch.linalg.solve_triangular(W.transpose(-1,-2), Y, upper=True)  # W^T Y2 = Y
        XWinv = Y2.transpose(-1,-2)  # = X (W W^T)^-1 = X Z^-1
        X = (b/c)*X + (a - b/c)*XWinv
        # update l (cubic convergence of singular values toward 1)
        l = l*(a + b*l*l)/(1.0 + c*l*l)
        l = l.clamp(max=1.0)
    # symmetrize (U should be symmetric for symmetric M)
    return 0.5*(X + X.transpose(-1,-2))

def cholqr2(Y, reg=0.0):
    # orthonormalize columns of Y (B,n,m) via CholeskyQR2 with a relative
    # diagonal shift (handles mild rank-deficiency robustly).
    for _ in range(2):
        G = torch.bmm(Y.transpose(-1,-2), Y)
        m = G.shape[-1]
        tr = torch.diagonal(G, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20).view(-1,1,1)
        G = G + (reg*tr + 1e-12)*torch.eye(m, device=dev, dtype=Y.dtype)
        R = torch.linalg.cholesky(G)
        Y = torch.linalg.solve_triangular(R, Y.transpose(-1,-2), upper=False).transpose(-1,-2)
    return Y

def qdwh_eig_single(A):
    B, n, _ = A.shape
    # prescale power-of-two
    asym = 0.5*(A + A.transpose(-1,-2))
    maxabs = asym.abs().amax(dim=(-2,-1))
    k = torch.floor(torch.log2(maxabs.clamp_min(1e-30)))
    k = torch.where(maxabs>0, k, torch.zeros_like(k))
    scale = torch.exp2(-k).view(B,1,1)
    As = asym*scale
    # shift sigma = median of Rayleigh diag estimate (use mean of Gershgorin center)
    diagA = torch.diagonal(As, dim1=-2, dim2=-1)
    sigma = diagA.median(dim=-1).values.view(B,1,1)
    I = torch.eye(n, device=dev, dtype=As.dtype)
    M = As - sigma*I
    U = chol_qdwh_polar(M, iters=6)
    Pp = 0.5*(I + U)  # projector onto eigenvalues > sigma
    Pm = 0.5*(I - U)
    kplus = torch.diagonal(Pp, dim1=-2, dim2=-1).sum(-1)  # approx rank
    # fixed half split (median shift => kplus ~ n/2 within +-1-2)
    h = n//2
    V1 = cholqr2(Pp[:, :, :h], reg=1e-2)       # (B,n,h)
    V2 = cholqr2(Pm[:, :, h:], reg=1e-2)       # (B,n,n-h)
    A1 = torch.bmm(V1.transpose(-1,-2), torch.bmm(As, V1))
    A2 = torch.bmm(V2.transpose(-1,-2), torch.bmm(As, V2))
    L1, W1 = torch.linalg.eigh(A1)
    L2, W2 = torch.linalg.eigh(A2)
    Q1 = torch.bmm(V1, W1)
    Q2 = torch.bmm(V2, W2)
    Q = torch.cat([Q1, Q2], dim=-1)
    Ls = torch.cat([L1, L2], dim=-1)
    # reortho cholqr2
    Q = cholqr2(Q)
    # Rayleigh L
    AQ = torch.bmm(As, Q)
    Lr = (Q*AQ).sum(-2) / scale.view(B,1)
    Lr, order = torch.sort(Lr, dim=-1)
    Q = torch.gather(Q, -1, order.unsqueeze(-2).expand(-1,n,-1))
    return Q, Lr, kplus

def scaled_resid(A, Q, L):
    A=A.double();Q=Q.double();L=L.double()
    n=A.shape[-1]; eps=torch.finfo(torch.float32).eps
    res=torch.linalg.matrix_norm(A@Q - Q*L.unsqueeze(-2),ord=1,dim=(-2,-1))
    scl=torch.linalg.matrix_norm(A,ord=1,dim=(-2,-1))
    sc=(res/(eps*n*scl.clamp_min(1e-30))).amax().item()
    qtq=Q.transpose(-1,-2)@Q; eye=torch.eye(n,dtype=torch.float64,device=A.device)
    osc=(torch.linalg.matrix_norm(qtq-eye,ord=1,dim=(-2,-1)).amax()/(eps*n)).item()
    return sc, osc

def make_clustered(B, n):
    # half eigenvalues near -1, half near +1 (median lands ON a cluster gap=0)
    vals = torch.where(torch.arange(n, device=dev) < n//2,
                       torch.full((n,), -1.0, device=dev),
                       torch.full((n,), 1.0, device=dev)).float()
    vals = vals + 1e-5*torch.randn(B, n, device=dev)
    Qr,_ = torch.linalg.qr(torch.randn(B,n,n,device=dev))
    return (Qr * vals.unsqueeze(-2)) @ Qr.transpose(-1,-2)

for n in [128, 256, 512]:
    B=8
    A = torch.randn(B,n,n,device=dev,dtype=torch.float32); A=0.5*(A+A.transpose(-1,-2))
    Q,L,kp = qdwh_eig_single(A)
    sc,osc = scaled_resid(A,Q,L)
    Ac = make_clustered(B, n)
    Qc,Lc,kpc = qdwh_eig_single(Ac)
    scc,oscc = scaled_resid(Ac,Qc,Lc)
    print(f"n={n:4d} dense: eig={sc:.2e} orth={osc:.2e} kp={kp.float().mean():.1f} | "
          f"CLUSTERED: eig={scc:.2e} orth={oscc:.2e} kp={kpc.float().mean():.1f} (h={n//2})")
