import torch, time
import triton
import triton.language as tl
dev="cuda"; torch.manual_seed(0)
def timeit(fn,it=6,w=2):
    for _ in range(w): fn()
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize();return (time.perf_counter()-t0)/it*1e6

# Dense band->tridiag bulge chase WITH Q accumulation, one program per matrix.
# A_ptr: (B,n,n) dense symmetric band (modified to tridiagonal). Q_ptr: (B,n,n)
# init identity, accumulates column rotations. Scalar HBM access; ports the
# verified reference exactly. b = half bandwidth.
@triton.jit
def _bulgeq_kernel(A_ptr, Q_ptr, B, n, b):
    pid = tl.program_id(0)
    if pid >= B: return
    Abase = pid*n*n
    Qbase = pid*n*n
    for j in range(0, n-2):
        # i from min(j+b,n-1) down to j+2
        hi = tl.minimum(j+b, n-1)
        i = hi
        while i >= j+2:
            brow = i
            bcol = j
            cont = 1
            while cont == 1:
                if (brow >= n) or (brow <= bcol+1):
                    cont = 0
                else:
                    p = brow-1
                    q = brow
                    a = tl.load(A_ptr + Abase + p*n + bcol)
                    bb = tl.load(A_ptr + Abase + q*n + bcol)
                    r = tl.sqrt(a*a + bb*bb)
                    rs = tl.where(r==0.0, 1.0, r)
                    c = tl.where(r>0.0, a/rs, 1.0)
                    s = tl.where(r>0.0, bb/rs, 0.0)
                    # rotate rows p,q over all columns (dense): A[p,:],A[q,:]
                    kk = 0
                    while kk < n:
                        apk = tl.load(A_ptr + Abase + p*n + kk)
                        aqk = tl.load(A_ptr + Abase + q*n + kk)
                        tl.store(A_ptr + Abase + p*n + kk, c*apk + s*aqk)
                        tl.store(A_ptr + Abase + q*n + kk, -s*apk + c*aqk)
                        kk += 1
                    # rotate cols p,q over all rows: A[:,p],A[:,q]
                    kk = 0
                    while kk < n:
                        akp = tl.load(A_ptr + Abase + kk*n + p)
                        akq = tl.load(A_ptr + Abase + kk*n + q)
                        tl.store(A_ptr + Abase + kk*n + p, c*akp + s*akq)
                        tl.store(A_ptr + Abase + kk*n + q, -s*akp + c*akq)
                        kk += 1
                    # accumulate Q cols p,q
                    kk = 0
                    while kk < n:
                        qkp = tl.load(Q_ptr + Qbase + kk*n + p)
                        qkq = tl.load(Q_ptr + Qbase + kk*n + q)
                        tl.store(Q_ptr + Qbase + kk*n + p, c*qkp + s*qkq)
                        tl.store(Q_ptr + Qbase + kk*n + q, -s*qkp + c*qkq)
                        kk += 1
                    nbrow = brow - 1 + b + 1
                    nbcol = brow - 1
                    brow = nbrow
                    bcol = nbcol
            i -= 1

def bulge_q(A, b):
    B,n,_=A.shape
    Awork = A.contiguous().clone()
    Q = torch.eye(n,device=dev).expand(B,n,n).contiguous().clone()
    _bulgeq_kernel[(B,)](Awork, Q, B, n, b, num_warps=4)
    d = torch.diagonal(Awork,dim1=-2,dim2=-1).contiguous()
    e = torch.diagonal(Awork,offset=-1,dim1=-2,dim2=-1).contiguous()
    return d, e, Q

for (B,n,b) in [(4,64,8),(8,128,16),(4,256,8)]:
    A=torch.randn(B,n,n,device=dev);A=0.5*(A+A.transpose(-1,-2))
    idx=torch.arange(n,device=dev); A=A*((idx[:,None]-idx[None,:]).abs()<=b)
    d,e,Q=bulge_q(A,b)
    T=torch.diag_embed(d)+torch.diag_embed(e,-1)+torch.diag_embed(e,1)
    chk=(Q.transpose(-1,-2)@A@Q - T).abs().max().item()
    orth=(Q.transpose(-1,-2)@Q-torch.eye(n,device=dev)).abs().max().item()
    everr=(torch.linalg.eigvalsh(A)-torch.linalg.eigvalsh(T)).abs().max().item()
    print(f"B={B} n={n} b={b}: chk={chk:.2e} orth={orth:.2e} everr={everr:.2e}")

print("=== bulge_q SPEED at realistic sizes ===")
for (B,n,b) in [(640,512,64),(640,512,16),(640,512,8),(60,1024,32),(60,1024,8),(8,2048,8)]:
    A=torch.randn(B,n,n,device=dev);A=0.5*(A+A.transpose(-1,-2))
    idx=torch.arange(n,device=dev); A=A*((idx[:,None]-idx[None,:]).abs()<=b)
    t=timeit(lambda:bulge_q(A,b))
    print(f"B={B} n={n} b={b}: bulge_q={t:.0f}us")
