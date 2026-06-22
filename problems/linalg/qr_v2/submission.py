import torch
import triton
import triton.language as tl
from task import input_t, output_t


# =============================================================================
# Batch-MAJOR right-looking blocked Householder QR (geqrf-compatible).
#
# For a panel of width b starting at column j:
#   1. panel_factor : one program per matrix factors A[j:, j:j+b] in place
#      (b unblocked Householder steps, panel resident in registers/SMEM),
#      producing reflectors V (unit lower-trapezoidal) and tau.
#   2. build_T      : per matrix, form the b x b WY T factor from V and tau.
#   3. trailing     : A[j:, j+b:] := (I - V T^T V^T) A[j:, j+b:]  (BLAS-3).
#
# H upper-tri = R, below-diag = reflectors, tau separate. FP32 throughout.
# Verified bit-equivalent to torch.geqrf in a PyTorch prototype.
# =============================================================================


@triton.jit
def _panel_factor_kernel(
    A_ptr, tau_ptr, Vbuf_ptr, Tbuf_ptr,
    B, N, j, pheight, b,
    stride_ab, stride_an,
    stride_vb, stride_vk, stride_vn,
    stride_tb, stride_tk, stride_tn,
    BLK: tl.constexpr, MAXH: tl.constexpr,
):
    bid = tl.program_id(0)
    if bid >= B:
        return

    rows = tl.arange(0, MAXH)                 # relative to panel top (global row j)
    cols = tl.arange(0, BLK)
    row_valid = rows < pheight
    col_valid = cols < b

    a_base = A_ptr + bid * stride_ab + j * stride_an + j
    aptr = a_base + rows[:, None] * stride_an + cols[None, :]
    mask = row_valid[:, None] & col_valid[None, :]
    panel = tl.load(aptr, mask=mask, other=0.0)          # (MAXH, BLK)

    tau_panel = tl.zeros((BLK,), dtype=tl.float32)
    # Build the WY T factor (BLK x BLK upper-tri) incrementally here (LAPACK
    # dlarft forward): at step k, T[0:k,k] = -tau_k * T[0:k,0:k] * (V[:,0:k]^T v_k).
    # This fuses build_T into panel_factor, eliminating a whole low-occupancy
    # kernel launch + a re-read of V from HBM.
    Tmat = tl.zeros((BLK, BLK), dtype=tl.float32)

    for k in range(0, BLK):
        do_k = k < b
        col_is_k = cols == k
        xk = tl.sum(tl.where(col_is_k[None, :], panel, 0.0), axis=1)   # (MAXH,)
        active = (rows >= k) & row_valid
        xk = tl.where(active, xk, 0.0)

        alpha = tl.sum(tl.where(rows == k, xk, 0.0))
        tailv = tl.where(rows > k, xk, 0.0)
        tail_n2 = tl.sum(tailv * tailv)
        normx = tl.sqrt(alpha * alpha + tail_n2)
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * normx
        has_refl = tail_n2 > 0.0
        beta_safe = tl.where(beta == 0.0, 1.0, beta)
        tau_k = tl.where(has_refl, (beta - alpha) / beta_safe, 0.0)

        denom = alpha - beta
        denom = tl.where(denom == 0.0, 1.0, denom)
        v = tl.where(rows > k, xk / denom, 0.0)
        v = tl.where(rows == k, 1.0, v)
        v = tl.where(active, v, 0.0)
        v = tl.where(has_refl, v, tl.where(rows == k, 1.0, 0.0))

        # w[c] = v_k . panel[:,c] -- used both for the trailing-panel update AND
        # the incremental T factor. Because v_k is supported only on rows >= k,
        # for c < k we have z[c] = V[:,c].v_k == w[c] exactly (the diagonal/above
        # terms vanish), so the WY recurrence needs no separate (MAXH,BLK) Vmat
        # tensor or extra reduction -- a big register/occupancy win on the panel.
        w = tl.sum(v[:, None] * panel, axis=0)            # (BLK,)

        # --- incremental T column k:  T[a<k,k] = -tau_k * (T @ w[c<k]) ---
        z = tl.where(cols < k, w, 0.0)
        Tcol = -tau_k * tl.sum(Tmat * z[None, :], axis=1)  # (BLK,)
        Tcol = tl.where(cols < k, Tcol, 0.0)
        Tcol = tl.where(cols == k, tau_k, Tcol)
        Tmat = tl.where(col_is_k[None, :], Tcol[:, None], Tmat)

        # apply H_k to trailing panel columns (c > k)
        upd = tau_k * v[:, None] * w[None, :]
        col_gt_k = cols > k
        panel = tl.where(col_gt_k[None, :], panel - upd, panel)

        # store this column: R[k,k] on diag, reflector below; leave rows<k
        # (already-finalized R entries from earlier steps) untouched.
        diagval = tl.where(has_refl, beta, alpha)
        new_colk = tl.where(rows == k, diagval, v)        # row==k -> beta, row>k -> v
        panel = tl.where(col_is_k[None, :] & (rows[:, None] >= k), new_colk[:, None], panel)

        tau_panel = tl.where(col_is_k & do_k, tau_k, tau_panel)

    tl.store(aptr, panel, mask=mask)

    tptr = tau_ptr + bid * N + j + cols
    tl.store(tptr, tau_panel, mask=col_valid)

    # V transposed: Vt[c, r] = 1 if r==c, panel[r,c] if r>c, else 0
    panelT = tl.trans(panel)                               # (BLK, MAXH)
    cc = cols[:, None]
    rr = rows[None, :]
    Vt = tl.where(rr == cc, 1.0, tl.where(rr > cc, panelT, 0.0))
    Vt = tl.where((rr < pheight) & (cc < b), Vt, 0.0)
    vbase = Vbuf_ptr + bid * stride_vb
    vptr = vbase + cc * stride_vk + rr * stride_vn
    tl.store(vptr, Vt, mask=(rr < pheight) & (cc < b))

    # store T
    tbase = Tbuf_ptr + bid * stride_tb
    tptr2 = tbase + cols[:, None] * stride_tk + cols[None, :] * stride_tn
    tl.store(tptr2, Tmat)


# Trailing update is split into TWO kernels to be race-free: the first reads
# A_trail and produces YT = T^T V^T A_trail (read-only consumer of A_trail); the
# second reads YT (read-only) and applies A_trail -= V @ YT (each program owns a
# disjoint output tile). Fusing them (one kernel that both reads all rows of a
# column and writes its own row slice) races: program for row-tile r0 reads rows
# owned by row-tile r1 while r1 writes them.
@triton.jit
def _trailing_YT_kernel(
    A_ptr, Vbuf_ptr, Tbuf_ptr, YT_ptr,
    B, N, j, pheight, ncols, jb,
    stride_ab, stride_an,
    stride_vb, stride_vk, stride_vn,
    stride_tb, stride_tk, stride_tn,
    stride_yb, stride_yk, stride_yn,
    BLK: tl.constexpr, BM: tl.constexpr, BNc: tl.constexpr,
):
    col_tile = tl.program_id(0)
    bid = tl.program_id(1)
    if bid >= B:
        return
    c0 = col_tile * BNc
    ccols = c0 + tl.arange(0, BNc)
    cmask = ccols < ncols
    krange = tl.arange(0, BLK)

    a_trail_base = A_ptr + bid * stride_ab + j * stride_an + jb
    v_base = Vbuf_ptr + bid * stride_vb

    # W = V^T @ A_trail over all panel rows, in chunks of BM
    W = tl.zeros((BLK, BNc), dtype=tl.float32)
    nchunks = tl.cdiv(pheight, BM)
    for ci in range(0, nchunks):
        rr = ci * BM + tl.arange(0, BM)
        rrmask = rr < pheight
        ap = a_trail_base + rr[:, None] * stride_an + ccols[None, :]
        achunk = tl.load(ap, mask=rrmask[:, None] & cmask[None, :], other=0.0)   # (BM,BNc)
        vp = v_base + krange[:, None] * stride_vk + rr[None, :] * stride_vn
        vchunk = tl.load(vp, mask=rrmask[None, :], other=0.0)                    # (BLK,BM)
        W += tl.dot(vchunk, achunk, input_precision="tf32x3")

    # YT = T^T @ W
    tp = Tbuf_ptr + bid * stride_tb + krange[:, None] * stride_tk + krange[None, :] * stride_tn
    Tm = tl.load(tp)
    YT = tl.dot(tl.trans(Tm), W, input_precision="tf32x3")                        # (BLK,BNc)

    yp = YT_ptr + bid * stride_yb + krange[:, None] * stride_yk + ccols[None, :] * stride_yn
    tl.store(yp, YT, mask=cmask[None, :])


@triton.jit
def _trailing_apply_kernel(
    A_ptr, Vbuf_ptr, YT_ptr,
    B, N, j, pheight, ncols, jb,
    stride_ab, stride_an,
    stride_vb, stride_vk, stride_vn,
    stride_yb, stride_yk, stride_yn,
    BLK: tl.constexpr, BM: tl.constexpr, BNc: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    if bid >= B:
        return
    num_col_tiles = tl.cdiv(ncols, BNc)
    row_tile = pid // num_col_tiles
    col_tile = pid % num_col_tiles
    rrows = row_tile * BM + tl.arange(0, BM)
    ccols = col_tile * BNc + tl.arange(0, BNc)
    rmask = rrows < pheight
    cmask = ccols < ncols
    krange = tl.arange(0, BLK)

    a_trail_base = A_ptr + bid * stride_ab + j * stride_an + jb
    v_base = Vbuf_ptr + bid * stride_vb

    # V[rrows,:] : (BM, BLK)
    vp = v_base + krange[None, :] * stride_vk + rrows[:, None] * stride_vn
    Vrow = tl.load(vp, mask=rmask[:, None], other=0.0)
    # YT[:, ccols] : (BLK, BNc)
    yp = YT_ptr + bid * stride_yb + krange[:, None] * stride_yk + ccols[None, :] * stride_yn
    YT = tl.load(yp, mask=cmask[None, :], other=0.0)
    delta = tl.dot(Vrow, YT, input_precision="tf32x3")                            # (BM,BNc)

    ap = a_trail_base + rrows[:, None] * stride_an + ccols[None, :]
    amask = rmask[:, None] & cmask[None, :]
    aorig = tl.load(ap, mask=amask, other=0.0)
    tl.store(ap, aorig - delta, mask=amask)


def custom_kernel(data: input_t) -> output_t:
    A = data
    B, N, _ = A.shape
    H = A.clone()
    tau = torch.zeros((B, N), device=A.device, dtype=torch.float32)

    if N <= 32:
        BLK = min(16, N)
    elif N >= 1536:
        # Tall panels (n>=2048): a narrow block halves the panel register
        # footprint -> much higher occupancy (~2x faster) than BLK=32.
        BLK = 16
    else:
        BLK = 32

    Vbuf = torch.empty((B, BLK, N), device=A.device, dtype=torch.float32)
    Tbuf = torch.empty((B, BLK, BLK), device=A.device, dtype=torch.float32)
    YTbuf = torch.empty((B, BLK, N), device=A.device, dtype=torch.float32)

    sab, san = H.stride(0), H.stride(1)
    svb, svk, svn = Vbuf.stride(0), Vbuf.stride(1), Vbuf.stride(2)
    stb, stk, stn = Tbuf.stride(0), Tbuf.stride(1), Tbuf.stride(2)
    syb, syk, syn = YTbuf.stride(0), YTbuf.stride(1), YTbuf.stride(2)

    # Trailing-update tiles. YT reduces V^T@A_trail over the tall panel-height
    # (wants a tall reduction chunk BM_Y); apply does a rank-BLK subtract with
    # disjoint (BM_A, BNc_A) output tiles (wants big output tiles for tensor-core
    # MMA reuse and few waves). Separate tiles + more warps lift the apply from
    # the measured 38% SM throughput / 39 waves (tiny 32x64 tiles, 2 warps).
    BM_Y, BNc_Y, NW_Y = 64, 64, 4
    BM_A, BNc_A, NW_A = 64, 128, 4
    j = 0
    while j < N:
        b = min(BLK, N - j)
        pheight = N - j
        MAXH = triton.next_power_of_2(pheight)

        # num_warps must scale with the panel height: the panel kernel holds a
        # (MAXH, BLK) register tensor; too few warps -> register spill -> huge
        # slowdown (measured 7-10x on n>=1024). Empirically-tuned per MAXH.
        if MAXH <= 512:
            nwp = 4
        elif MAXH <= 1024:
            nwp = 8
        else:
            nwp = 32

        _panel_factor_kernel[(B,)](
            H, tau, Vbuf, Tbuf,
            B, N, j, pheight, b,
            sab, san, svb, svk, svn, stb, stk, stn,
            BLK=BLK, MAXH=MAXH, num_warps=nwp,
        )

        ncols = N - (j + b)
        if ncols > 0:
            nct_y = triton.cdiv(ncols, BNc_Y)
            _trailing_YT_kernel[(nct_y, B)](
                H, Vbuf, Tbuf, YTbuf,
                B, N, j, pheight, ncols, j + b,
                sab, san, svb, svk, svn, stb, stk, stn, syb, syk, syn,
                BLK=BLK, BM=BM_Y, BNc=BNc_Y, num_warps=NW_Y,
            )
            nct_a = triton.cdiv(ncols, BNc_A)
            nrt_a = triton.cdiv(pheight, BM_A)
            _trailing_apply_kernel[(nrt_a * nct_a, B)](
                H, Vbuf, YTbuf,
                B, N, j, pheight, ncols, j + b,
                sab, san, svb, svk, svn, syb, syk, syn,
                BLK=BLK, BM=BM_A, BNc=BNc_A, num_warps=NW_A,
            )
        j += b

    return H, tau
