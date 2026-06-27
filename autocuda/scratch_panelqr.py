import torch, time
import triton
import triton.language as tl
dev = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

def timeit(fn, it=8, w=3):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6


# Panel Householder QR kernel: one program per matrix. Panel = A[:, r0:, c0:c0+b]
# shape (B, m, b). Compute b Householder reflectors (stored in V, m x b, with
# implicit unit on the diagonal) and tau (b). The panel is reduced in registers
# block-loaded. We operate column-by-column (b sequential steps), each step does
# a Householder on the current column and updates the remaining columns.
#
# To keep it simple and correct, we load the panel into SMEM-backed tensors via
# tl arises; Triton handles the (m x b) tile. b is small (<=64), m up to 512.
@triton.jit
def _panel_qr_kernel(p_ptr, v_ptr, tau_ptr, B, m, b,
                     BM: tl.constexpr, BB: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B:
        return
    base = pid * m * b
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BB)
    rmask = rows < m
    cmask = cols < b
    # load panel (BM x BB) into registers
    P = tl.load(p_ptr + base + rows[:, None] * b + cols[None, :],
                mask=rmask[:, None] & cmask[None, :], other=0.0)
    V = tl.zeros((BM, BB), tl.float32)
    for j in range(BB):
        if j < b:
            # column j below row j
            colj = tl.where(rows >= j, tl.sum(tl.where(cols[None, :] == j, P, 0.0), axis=1), 0.0)
            # ||x|| over rows >= j
            x = tl.where(rows >= j, colj, 0.0)
            alpha = tl.sum(tl.where(rows == j, x, 0.0))
            normx2 = tl.sum(x * x)
            normx = tl.sqrt(normx2)
            beta = tl.where(alpha >= 0, -normx, normx)
            # v = x; v[j] = alpha - beta ; tau = (beta-alpha)/beta
            denom = alpha - beta
            safe = normx2 > 0.0
            tau = tl.where(safe, (beta - alpha) / tl.where(beta == 0.0, 1e-30, beta), 0.0)
            vj = tl.where(rows == j, denom, x)  # v with v[j]=alpha-beta, v[rows>j]=x
            vj = tl.where(rows >= j, vj, 0.0)
            # normalize v so v[j] = 1 (store implicit). Actually store v with v[j]=1:
            vscale = tl.where(rows == j, 1.0, vj / tl.where(denom == 0.0, 1e-30, denom))
            vstore = tl.where(rows >= j, vscale, 0.0)
            vstore = tl.where(safe, vstore, tl.where(rows == j, 1.0, 0.0))
            # apply H = I - tau v v^T to remaining columns (cols > j) of P:
            # P[:, k] -= tau * v * (v^T P[:,k])  for k>j
            # compute w_k = v^T P[:,k] for all k
            vcol = vstore  # (BM,)
            w = tl.sum(vcol[:, None] * P, axis=0)  # (BB,) = v^T P[:,k]
            P = P - tau * vcol[:, None] * w[None, :]
            # store reflector column j into V (rows>j hold tail; we store full vstore with v[j]=1)
            V = tl.where(cols[None, :] == j, vstore[:, None], V)
            # store tau
            tau_store_mask = cols == j  # in 1D over BB
            # write tau to tau_ptr[pid, j]
            tl.store(tau_ptr + pid * b + j, tau)
    # write V and the reduced R (P upper triangle) back
    tl.store(v_ptr + base + rows[:, None] * b + cols[None, :], V,
             mask=rmask[:, None] & cmask[None, :])
    # overwrite panel with R (the reduced upper-triangular part in the first b rows)
    tl.store(p_ptr + base + rows[:, None] * b + cols[None, :], P,
             mask=rmask[:, None] & cmask[None, :])


def panel_qr(panel):
    # panel: (B, m, b). Returns V (B,m,b reflectors, v[j,j]=1), tau (B,b), R in panel (overwritten)
    B, m, b = panel.shape
    p = panel.contiguous().clone()
    V = torch.zeros((B, m, b), device=dev, dtype=torch.float32)
    tau = torch.zeros((B, b), device=dev, dtype=torch.float32)
    BM = triton.next_power_of_2(m); BB = triton.next_power_of_2(b)
    _panel_qr_kernel[(B,)](p, V, tau, B, m, b, BM, BB, num_warps=8)
    return V, tau, p


# verify against torch qr on a panel
B, m, b = 4, 200, 16
panel = torch.randn(B, m, b, device=dev)
V, tau, R = panel_qr(panel)
# reconstruct Q from V,tau and check Q R == panel
# H_j = I - tau_j v_j v_j^T ; Q = H_0 H_1 ... H_{b-1}
Q = torch.eye(m, device=dev).expand(B, m, m).contiguous()
for j in range(b):
    v = V[:, :, j:j+1]  # (B,m,1)
    Q = Q - tau[:, j].view(B,1,1) * (Q @ v) @ v.transpose(-1,-2)
# R is in the top b rows of the overwritten panel
Rtop = R[:, :b, :]  # (B,b,b) upper triangular
recon = Q[:, :, :b] @ Rtop  # Q[:,:,:b] (m x b) @ R (b x b)? no: panel = Q @ [R;0], R is b x b
# panel (m x b) = Q (m x m) @ Rfull (m x b) where Rfull top b rows = R, rest 0
Rfull = torch.zeros(B, m, b, device=dev); Rfull[:, :b, :] = Rtop
recon = Q @ Rfull
err = (recon - panel).abs().max().item()
print(f"panel_qr recon err = {err:.3e}  (B={B} m={m} b={b})")
print("R upper-tri check (below-diag of Rtop):", Rtop.tril(-1).abs().max().item())

# ---- full->band via panel_qr + WY GEMM two-sided update ----
def apply_block_reflector_left(V, tau, X):
    # Q^T X where Q = H_0..H_{b-1} = I - V T V^T ; Q^T = I - V T^T V^T
    # X: (B, m, k). Compute T (b x b) from V, tau, then W = V T, then X -= V (T^T (V^T X))? 
    # Simpler: apply reflectors one by one (b GEMVs) -> but that's b launches. Use WY.
    # Build T: T is upper triangular b x b with T_ii = tau_i, and
    # T = -T_prev applied... Standard: T_jj=tau_j, T[0:j,j] = -tau_j T[0:j,0:j] (V[:,0:j]^T V[:,j])
    B, m, b = V.shape
    # compute Y = V^T V (b x b)
    YtV = torch.bmm(V.transpose(-1,-2), V)  # (B,b,b)
    T = torch.zeros(B, b, b, device=dev)
    for j in range(b):
        T[:, j, j] = tau[:, j]
        if j > 0:
            z = -tau[:, j].view(B,1) * YtV[:, :j, j]  # (B,j)
            T[:, :j, j] = torch.bmm(T[:, :j, :j], z.unsqueeze(-1)).squeeze(-1)
    # Q^T X = X - V T^T (V^T X)
    VtX = torch.bmm(V.transpose(-1,-2), X)  # (B,b,k)
    return X - torch.bmm(V, torch.bmm(T.transpose(-1,-2), VtX)), V, T

def full_to_band_fast(a, b):
    B, n, _ = a.shape
    t = a.clone()
    q1 = torch.eye(n, device=dev).expand(B, n, n).contiguous()
    c0 = 0
    while c0 + b < n:
        r0 = c0 + b
        mblk = n - r0
        bb = min(b, n - c0)  # panel width (could be < b at the end)
        panel = t[:, r0:, c0:c0+b].contiguous()
        V, tau, _ = panel_qr(panel)  # V: (B, mblk, b)
        # build T
        YtV = torch.bmm(V.transpose(-1,-2), V)
        T = torch.zeros(B, b, b, device=dev)
        for j in range(b):
            T[:, j, j] = tau[:, j]
            if j > 0:
                z = -tau[:, j].view(B,1) * YtV[:, :j, j]
                T[:, :j, j] = torch.bmm(T[:, :j, :j], z.unsqueeze(-1)).squeeze(-1)
        # left: t[r0:,:] := Q^T t[r0:,:] = t - V T^T (V^T t)
        X = t[:, r0:, :]
        VtX = torch.bmm(V.transpose(-1,-2), X)
        t[:, r0:, :] = X - torch.bmm(V, torch.bmm(T.transpose(-1,-2), VtX))
        # right: t[:,r0:] := t[:,r0:] Q = t - (t V) T V^T
        Y = t[:, :, r0:]
        YV = torch.bmm(Y, V)
        t[:, :, r0:] = Y - torch.bmm(torch.bmm(YV, T), V.transpose(-1,-2))
        # accumulate q1[:,r0:] := q1[:,r0:] Q
        Yq = q1[:, :, r0:]
        YqV = torch.bmm(Yq, V)
        q1[:, :, r0:] = Yq - torch.bmm(torch.bmm(YqV, T), V.transpose(-1,-2))
        c0 += b
    return 0.5*(t + t.transpose(-1,-2)), q1

print("=== full_to_band_fast correctness + speed ===")
for (B,n) in [(640,512),(60,1024),(8,2048)]:
    A=torch.randn(B,n,n,device=dev);A=0.5*(A+A.transpose(-1,-2))
    Tb,Q1=full_to_band_fast(A,64)
    # check Q1^T A Q1 == Tb (banded) and Q1 orthogonal
    chk = (Q1.transpose(-1,-2)@A@Q1 - Tb).abs().max().item()
    orth = (Q1.transpose(-1,-2)@Q1 - torch.eye(n,device=dev)).abs().max().item()
    idx=torch.arange(n,device=dev); offb=Tb[:, (idx[:,None]-idx[None,:]).abs()>64].abs().max().item()
    t=timeit(lambda:full_to_band_fast(A,64))
    print(f"B={B} n={n}: chk={chk:.2e} orth={orth:.2e} offband={offb:.2e}  time={t:.0f}us")
