import torch, time
import triton, triton.language as tl
dev="cuda"; torch.manual_seed(0)
@triton.jit
def _bulgeq_kernel(A_ptr, Q_ptr, B, n, b):
    pid=tl.program_id(0)
    if pid>=B: return
    Ab=pid*n*n; Qb=pid*n*n
    for j in range(0,n-2):
        hi=tl.minimum(j+b,n-1); i=hi
        while i>=j+2:
            brow=i; bcol=j; cont=1
            while cont==1:
                if (brow>=n) or (brow<=bcol+1): cont=0
                else:
                    p=brow-1; q=brow
                    a=tl.load(A_ptr+Ab+p*n+bcol); bb=tl.load(A_ptr+Ab+q*n+bcol)
                    r=tl.sqrt(a*a+bb*bb); rs=tl.where(r==0.0,1.0,r)
                    c=tl.where(r>0.0,a/rs,1.0); s=tl.where(r>0.0,bb/rs,0.0)
                    kk=0
                    while kk<n:
                        apk=tl.load(A_ptr+Ab+p*n+kk); aqk=tl.load(A_ptr+Ab+q*n+kk)
                        tl.store(A_ptr+Ab+p*n+kk,c*apk+s*aqk); tl.store(A_ptr+Ab+q*n+kk,-s*apk+c*aqk); kk+=1
                    kk=0
                    while kk<n:
                        akp=tl.load(A_ptr+Ab+kk*n+p); akq=tl.load(A_ptr+Ab+kk*n+q)
                        tl.store(A_ptr+Ab+kk*n+p,c*akp+s*akq); tl.store(A_ptr+Ab+kk*n+q,-s*akp+c*akq); kk+=1
                    kk=0
                    while kk<n:
                        qkp=tl.load(Q_ptr+Qb+kk*n+p); qkq=tl.load(Q_ptr+Qb+kk*n+q)
                        tl.store(Q_ptr+Qb+kk*n+p,c*qkp+s*qkq); tl.store(Q_ptr+Qb+kk*n+q,-s*qkp+c*qkq); kk+=1
                    brow=brow-1+b+1; bcol=brow-1
            i-=1
def bulge_q(A,b):
    B,n,_=A.shape; Aw=A.contiguous().clone(); Q=torch.eye(n,device=dev).expand(B,n,n).contiguous().clone()
    _bulgeq_kernel[(B,)](Aw,Q,B,n,b,num_warps=4)
    return Aw,Q
import time
B,n,b=640,512,32
A=torch.randn(B,n,n,device=dev);A=0.5*(A+A.transpose(-1,-2))
idx=torch.arange(n,device=dev);A=A*((idx[:,None]-idx[None,:]).abs()<=b)
torch.cuda.synchronize();t0=time.perf_counter()
bulge_q(A,b)
torch.cuda.synchronize()
print(f"bulge_q b={b} n={n} B={B}: {(time.perf_counter()-t0)*1e6:.0f}us (single run incl compile)")
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(3): bulge_q(A,b)
torch.cuda.synchronize()
print(f"  warm avg: {(time.perf_counter()-t0)/3*1e6:.0f}us")
