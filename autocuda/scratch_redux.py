import torch, time
dev="cuda"; torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=True
def timeit(fn,it=8,w=3):
    for _ in range(w): fn()
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize();return (time.perf_counter()-t0)/it*1e6

# worker-1 full->band with qr complete
def band_qrcomplete(a,b):
    batch,n,_=a.shape; t=a.clone()
    q1=torch.eye(n,device=dev).expand(batch,n,n).contiguous(); c0=0
    while c0+b<n:
        r0=c0+b
        qp,_=torch.linalg.qr(t[:,r0:,c0:c0+b],mode="complete")
        t[:,r0:,:]=qp.transpose(-1,-2)@t[:,r0:,:]; t[:,:,r0:]=t[:,:,r0:]@qp; q1[:,:,r0:]=q1[:,:,r0:]@qp
        c0+=b
    return 0.5*(t+t.transpose(-1,-2)),q1

# full->band via Householder-WY panel (reduced QR via householder on the panel, apply via WY GEMM)
# Use torch.linalg.qr REDUCED to get reflectors? reduced gives Q (m x b) + R. To apply two-sided
# we need the full reflector action. Build compact WY from the panel's householder vectors.
def band_householder_wy(a,b):
    batch,n,_=a.shape; t=a.clone()
    q1=torch.eye(n,device=dev).expand(batch,n,n).contiguous(); c0=0
    while c0+b<n:
        r0=c0+b; m=n-r0
        panel=t[:,r0:,c0:c0+b]  # (batch,m,b)
        # householder QR of panel -> get V (m x b) reflectors and tau via geqrf? torch has no geqrf exposed cheaply.
        # Use qr reduced then form WY? Not directly. Instead: CholeskyQR-based reflector? 
        # Simplest: qr reduced gives Qp (m x b); the transform that zeros below is H with H panel=[R;0].
        # Apply H = I - Qp Qp^T + (something)... not orthogonal-completing. Skip.
        c0+=b
    return None

# CholeskyQR-based band: panel = Qp R (Qp m x b orthonormal via cholqr); the Householder-equivalent
# two-sided update needs the FULL orthogonal completion -> not available from cholqr. 
# Instead measure: just the qr_complete cost breakdown vs eigh.
for (B,n) in [(640,512),(60,1024),(8,2048)]:
    A=torch.randn(B,n,n,device=dev);A=0.5*(A+A.transpose(-1,-2))
    tb=timeit(lambda:torch.linalg.eigh(A))
    try:
        tq=timeit(lambda:band_qrcomplete(A,64))
    except Exception as e:
        tq=float('nan'); print("qr err",e)
    # eigh on the band result
    Tb,_=band_qrcomplete(A,64)
    te=timeit(lambda:torch.linalg.eigh(Tb))
    print(f"B={B} n={n}: eigh(full)={tb:.0f}  band_qrcomplete={tq:.0f}  eigh(band)={te:.0f}  band+eigh={tq+te:.0f}")
