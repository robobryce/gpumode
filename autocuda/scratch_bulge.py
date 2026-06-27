import torch, time
import triton
import triton.language as tl
dev = "cuda"
torch.manual_seed(0)

# Band -> tridiagonal via Givens bulge-chasing, accumulating Q.
# Reference (torch, per-matrix loop) FIRST to nail the algorithm, then port to Triton.
# Symmetric band stored as full dense (we operate on the dense band here for the
# reference). Half-bandwidth b. Reduce to tridiagonal, accumulate G into Q.

def bulge_chase_ref(Aband, b):
    # Aband: (B, n, n) symmetric banded (half-width b). Returns d, e, Q with
    # Q^T Aband Q = tridiag(d,e). Reference using explicit Givens on dense.
    B, n, _ = Aband.shape
    A = Aband.clone()
    Q = torch.eye(n, device=dev).expand(B, n, n).contiguous()
    def rot(i, k, c, s):
        # apply Givens (rows/cols i,k) on both sides: G = [[c,s],[-s,c]] on (i,k)
        # A := G A G^T ; rows i,k then cols i,k
        Ai = A[:, i, :].clone(); Ak = A[:, k, :].clone()
        A[:, i, :] = c.view(B,1)*Ai + s.view(B,1)*Ak
        A[:, k, :] = -s.view(B,1)*Ai + c.view(B,1)*Ak
        Ai = A[:, :, i].clone(); Ak = A[:, :, k].clone()
        A[:, :, i] = c.view(B,1)*Ai + s.view(B,1)*Ak
        A[:, :, k] = -s.view(B,1)*Ai + c.view(B,1)*Ak
        Qi = Q[:, :, i].clone(); Qk = Q[:, :, k].clone()
        Q[:, :, i] = c.view(B,1)*Qi + s.view(B,1)*Qk
        Q[:, :, k] = -s.view(B,1)*Qi + c.view(B,1)*Qk
    def givens_zero(brow, bcol):
        # zero A[brow, bcol] using the entry above it A[brow-1, bcol] via a
        # symmetric Givens rotation of rows/cols (brow-1, brow). Returns the
        # position of the bulge it creates, or None.
        a = A[:, brow - 1, bcol]
        bb = A[:, brow, bcol]
        r = torch.sqrt(a * a + bb * bb)
        rsafe = torch.where(r == 0, torch.ones_like(r), r)
        c = torch.where(r > 0, a / rsafe, torch.ones_like(r))
        s = torch.where(r > 0, bb / rsafe, torch.zeros_like(r))
        rot(brow - 1, brow, c, s)
        # the rotation of cols (brow-1, brow) creates fill at (brow-1+b+1, brow-1)
        nr = brow - 1 + b + 1
        nc = brow - 1
        if nr < n and nc < n and nr > nc + 1:
            return nr, nc
        return None

    for j in range(n - 2):
        for i in range(min(j + b, n - 1), j + 1, -1):
            # annihilate A[i, j] (partner above at A[i-1, j]) and chase the bulge
            nxt = givens_zero(i, j)
            while nxt is not None:
                nxt = givens_zero(nxt[0], nxt[1])
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    e = torch.diagonal(A, offset=-1, dim1=-2, dim2=-1)
    return d, e, Q

for (B, n, b) in [(2, 16, 4), (2, 40, 8), (2, 64, 16)]:
    # build symmetric band
    A = torch.randn(B, n, n, device=dev); A = 0.5*(A+A.transpose(-1,-2))
    idx = torch.arange(n, device=dev)
    mask = (idx[:,None]-idx[None,:]).abs() <= b
    A = A * mask
    d, e, Q = bulge_chase_ref(A, b)
    T = torch.diag_embed(d)+torch.diag_embed(e,-1)+torch.diag_embed(e,1)
    chk = (Q.transpose(-1,-2)@A@Q - T).abs().max().item()
    orth = (Q.transpose(-1,-2)@Q - torch.eye(n,device=dev)).abs().max().item()
    evA = torch.linalg.eigvalsh(A); evT = torch.linalg.eigvalsh(T)
    everr = (evA - evT).abs().max().item()
    print(f"B={B} n={n} b={b}: chk={chk:.2e} orth={orth:.2e} everr={everr:.2e}")
