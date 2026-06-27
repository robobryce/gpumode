import torch, time
import triton, triton.language as tl
dev="cuda"; torch.manual_seed(0)
def timeit(fn,it=6,w=2):
    for _ in range(w): fn()
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize();return (time.perf_counter()-t0)/it*1e6

# compact-band bulge-chase, EIGENVALUES ONLY (no Q). Band stored lower-compact:
# bl[B, n, b+1], bl[i,t]=A[i,i-t]. Rotations touch only band cells.
@triton.jit
def _bnoq(bl_ptr, B, n, b, BW: tl.constexpr):
    pid=tl.program_id(0)
    if pid>=B: return
    base=pid*n*BW
    # loader: A[r,c] for r>=c, r-c<=b : bl[r, r-c]; for r<c use symmetry A[c,r]
    for j in range(0, n-2):
        hi = tl.minimum(j+b, n-1)
        i = hi
        while i >= j+2:
            brow=i; bcol=j; cont=1
            while cont==1:
                if (brow>=n) or (brow<=bcol+1):
                    cont=0
                else:
                    p=brow-1; q=brow
                    a=tl.load(bl_ptr+base+p*BW+(p-bcol), mask=(p-bcol)<BW, other=0.0)
                    bb=tl.load(bl_ptr+base+q*BW+(q-bcol), mask=(q-bcol)<BW, other=0.0)
                    r=tl.sqrt(a*a+bb*bb); rs=tl.where(r==0.0,1.0,r)
                    c=tl.where(r>0.0,a/rs,1.0); s=tl.where(r>0.0,bb/rs,0.0)
                    # affected columns k in [p-b .. q+b]
                    kk=0
                    while kk < (2*BW+2):
                        k=(p-b)+kk
                        if (k>=0) and (k<=n-1):
                            # A[p,k]
                            apk=tl.where(p>=k, tl.load(bl_ptr+base+p*BW+(p-k),mask=((p-k)>=0)&((p-k)<BW),other=0.0),
                                              tl.load(bl_ptr+base+k*BW+(k-p),mask=((k-p)>=0)&((k-p)<BW),other=0.0))
                            aqk=tl.where(q>=k, tl.load(bl_ptr+base+q*BW+(q-k),mask=((q-k)>=0)&((q-k)<BW),other=0.0),
                                              tl.load(bl_ptr+base+k*BW+(k-q),mask=((k-q)>=0)&((k-q)<BW),other=0.0))
                            npk=c*apk+s*aqk; nqk=-s*apk+c*aqk
                            if p>=k:
                                tl.store(bl_ptr+base+p*BW+(p-k),npk,mask=((p-k)>=0)&((p-k)<BW))
                            else:
                                tl.store(bl_ptr+base+k*BW+(k-p),npk,mask=((k-p)>=0)&((k-p)<BW))
                            if q>=k:
                                tl.store(bl_ptr+base+q*BW+(q-k),nqk,mask=((q-k)>=0)&((q-k)<BW))
                            else:
                                tl.store(bl_ptr+base+k*BW+(k-q),nqk,mask=((k-q)>=0)&((k-q)<BW))
                        kk+=1
                    brow=brow-1+b+1; bcol=brow-1-(b+1)+1  # = old brow-1
            i-=1

def to_compact(A,b):
    B,n,_=A.shape; bl=torch.zeros(B,n,b+1,device=dev)
    for t in range(b+1):
        # bl[:,i,t]=A[i,i-t] for i>=t
        idx=torch.arange(t,n,device=dev)
        bl[:,idx,t]=A[:,idx,idx-t]
    return bl
def from_compact(bl,n,b):
    d=bl[:,:,0].contiguous(); e=bl[:,1:,1].contiguous()  # A[i,i-1]
    return d,e
def bulge_noq(A,b):
    B,n,_=A.shape; bl=to_compact(A,b).contiguous()
    BW=b+1
    _bnoq[(B,)](bl,B,n,b,BW,num_warps=2)
    return from_compact(bl,n,b)

# correctness small
for (B,n,b) in [(2,40,8),(2,64,16)]:
    A=torch.randn(B,n,n,device=dev);A=0.5*(A+A.transpose(-1,-2))
    idx=torch.arange(n,device=dev);A=A*((idx[:,None]-idx[None,:]).abs()<=b)
    d,e=bulge_noq(A,b)
    T=torch.diag_embed(d)+torch.diag_embed(e,-1)+torch.diag_embed(e,1)
    everr=(torch.linalg.eigvalsh(A)-torch.linalg.eigvalsh(T)).abs().max().item()
    print(f"correctness B={B} n={n} b={b}: everr={everr:.2e}")
# speed
for (B,n,b) in [(640,512,64),(640,512,16),(60,1024,16)]:
    A=torch.randn(B,n,n,device=dev);A=0.5*(A+A.transpose(-1,-2))
    idx=torch.arange(n,device=dev);A=A*((idx[:,None]-idx[None,:]).abs()<=b)
    t=timeit(lambda:bulge_noq(A,b))
    print(f"SPEED B={B} n={n} b={b}: bulge_noq={t:.0f}us")
