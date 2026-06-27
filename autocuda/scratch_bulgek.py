import torch, time
import triton
import triton.language as tl
dev="cuda"; torch.manual_seed(0)
def timeit(fn,it=8,w=3):
    for _ in range(w): fn()
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize();return (time.perf_counter()-t0)/it*1e6

# Bulge-chase band->tridiagonal, EIGENVALUES ONLY (no Q accumulation).
# Band stored compactly: bandl[B, n, b+1] where bandl[:, i, t] = A[i, i-t] (t=0..b),
# the lower band (incl diagonal). We rotate within this compact storage.
# Givens zeroing A[brow,bcol] with partner A[brow-1,bcol]; rows/cols (brow-1,brow).
# In compact lower storage A[i,j] = bandl[i, i-j] for i>=j, |i-j|<=b.
@triton.jit
def _bulge_eig_kernel(bl_ptr, d_ptr, e_ptr, B, n, b, BW: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B: return
    base = pid*n*(BW)
    # helper offsets
    # We implement the symmetric Givens by reading/writing the needed band cells.
    # Because random access into a compact band in Triton scalar style is verbose,
    # we operate on a DENSE per-row representation is too big. Use scalar loads.
    # Access A[i,j] via lower storage: if i>=j: bl[i, i-j]; else bl[j, j-i].
    # We'll just do scalar loads/stores with a python-style index helper inlined.
    for j in range(0, n-2):
        # annihilate from i = min(j+b, n-1) down to j+2
        for it_i in range(0, BW):  # it_i enumerates potential i; we map i = (j+b) - it_i
            i = (j + b) - it_i
            if (i >= j+2) and (i <= n-1):
                # chase starting at (i, j)
                brow = i; bcol = j
                # bounded chase loop
                for _chase in range(0, 4096):
                    if (brow >= n) or (bcol >= n) or (brow <= bcol+1):
                        pass
                    else:
                        # load A[brow-1,bcol], A[brow,bcol]
                        # both have row>=col so lower storage
                        a = tl.load(bl_ptr + base + (brow-1)*BW + (brow-1-bcol))
                        bb = tl.load(bl_ptr + base + (brow)*BW + (brow-bcol))
                        r = tl.sqrt(a*a+bb*bb)
                        rs = tl.where(r==0.0, 1.0, r)
                        c = tl.where(r>0.0, a/rs, 1.0)
                        s = tl.where(r>0.0, bb/rs, 0.0)
                        # apply symmetric rotation rows/cols (p=brow-1, q=brow)
                        # affected cells: for k in [bcol .. min(n-1, brow+b)]: A[p,k],A[q,k] and symmetric
                        # We update the band cells in columns/rows within band of p,q.
                        p = brow-1; q = brow
                        # rotate rows p,q over columns k where A[p,k] or A[q,k] in band
                        for kk in range(0, BW*2+2):
                            k = (q - b - 1) + kk  # span columns near the band
                            if (k >= 0) and (k <= n-1):
                                # A[p,k]: row p col k
                                # fetch with symmetry
                                # define loader inline
                                # A[p,k]
                                apk = tl.where(p>=k, 
                                    tl.load(bl_ptr+base+p*BW+(p-k), mask=(p-k)<BW, other=0.0),
                                    tl.load(bl_ptr+base+k*BW+(k-p), mask=(k-p)<BW, other=0.0))
                                aqk = tl.where(q>=k,
                                    tl.load(bl_ptr+base+q*BW+(q-k), mask=(q-k)<BW, other=0.0),
                                    tl.load(bl_ptr+base+k*BW+(k-q), mask=(k-q)<BW, other=0.0))
                                npk = c*apk + s*aqk
                                nqk = -s*apk + c*aqk
                                # store back (only if within band, i.e. |row-col|<=b)
                                if (p>=k):
                                    m1 = (p-k)<BW
                                    tl.store(bl_ptr+base+p*BW+(p-k), npk, mask=m1)
                                else:
                                    m1 = (k-p)<BW
                                    tl.store(bl_ptr+base+k*BW+(k-p), npk, mask=m1)
                                if (q>=k):
                                    m2 = (q-k)<BW
                                    tl.store(bl_ptr+base+q*BW+(q-k), nqk, mask=m2)
                                else:
                                    m2 = (k-q)<BW
                                    tl.store(bl_ptr+base+k*BW+(k-q), nqk, mask=m2)
                        # next bulge
                        nbrow = brow - 1 + b + 1
                        nbcol = brow - 1
                        brow = nbrow; bcol = nbcol
    # extract d, e
    for i in range(0, BW):
        pass
