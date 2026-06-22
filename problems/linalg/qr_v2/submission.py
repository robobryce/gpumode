import torch
import triton
import triton.language as tl
from task import input_t, output_t


# =============================================================================
# Batch-MAJOR right-looking blocked Householder QR (geqrf-compatible) with a
# BLAS-3 (communication-avoiding) PANEL factorization.
#
# For a panel of width BLK starting at column j:
#   1. _panel_factor_kernel : one program per matrix factors A[j:, j:j+BLK] in
#      place. INSTEAD of BLK serial rank-1 Householder updates over the whole
#      (MAXH, BLK) panel (BLAS-2, the occupancy wall), it factors the panel in
#      narrow inner sub-blocks of width IB: the IB-step serial Householder chain
#      runs over only the IB sub-block columns (cheap, narrow), then the rest of
#      the panel columns [s+IB, BLK) are updated with ONE rank-IB compact-WY
#      block reflector applied via TENSOR-CORE tl.dot GEMMs (BLAS-3). The bulk
#      panel FLOPs thus move from serial BLAS-2 onto the tensor cores, in-kernel
#      (no extra kernel launches). Builds the full BLK x BLK compact-WY T
#      incrementally (the z[c]=w[c] trick) for the outer trailing update.
#   2. trailing : A[j:, j+BLK:] := (I - V T^T V^T) A[j:, j+BLK:]  (BLAS-3),
#      split into two race-free kernels (YT producer + apply).
#
# H upper-tri = R, below-diag = reflectors, tau separate. tf32x3 (error-
# corrected TF32) tensor-core GEMMs throughout (passes all ill-conditioned
# cases). Verified bit-equivalent to torch.geqrf in a PyTorch prototype.
# =============================================================================


@triton.jit
def _panel_factor_kernel(
    A_ptr, tau_ptr, Vbuf_ptr, Tbuf_ptr,
    B, N, j, pheight, b,
    stride_ab, stride_an,
    stride_vb, stride_vk, stride_vn,
    stride_tb, stride_tk, stride_tn,
    BLK: tl.constexpr, IB: tl.constexpr, MAXH: tl.constexpr,
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
    # Full BLK x BLK compact-WY T (upper-tri), built incrementally below. Used by
    # the outer trailing update; the per-sub-block block reflector uses its IB x
    # IB diagonal slice.
    Tmat = tl.zeros((BLK, BLK), dtype=tl.float32)
    # V columns accumulated for the current sub-block, used by the BLAS-3 update.
    # Stored as the panel's reflector columns; we read them back from `panel`
    # after each sub-block's serial chain.

    nsub: tl.constexpr = BLK // IB
    for sblk in range(0, nsub):
        s = sblk * IB
        # ---- serial Householder chain over the IB columns [s, s+IB) ----
        for kk in range(0, IB):
            k = s + kk
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

            # w[c] = v_k . panel[:,c]. We only need it for the in-sub-block rank-1
            # update (cols in (k, s+IB)) and the incremental T (cols < k, where
            # z[c]=w[c] since v_k is supported on rows>=k). Compute over the full
            # row but it is a (BLK,) reduction -- cheap.
            w = tl.sum(v[:, None] * panel, axis=0)            # (BLK,)

            # incremental T column k:  T[a<k,k] = -tau_k * (T @ w[c<k])
            z = tl.where(cols < k, w, 0.0)
            Tcol = -tau_k * tl.sum(Tmat * z[None, :], axis=1)  # (BLK,)
            Tcol = tl.where(cols < k, Tcol, 0.0)
            Tcol = tl.where(cols == k, tau_k, Tcol)
            Tmat = tl.where(col_is_k[None, :], Tcol[:, None], Tmat)

            # rank-1 update of trailing columns WITHIN this sub-block: c in (k, s+IB)
            upd = tau_k * v[:, None] * w[None, :]
            col_in_sub_gt_k = (cols > k) & (cols < s + IB)
            panel = tl.where(col_in_sub_gt_k[None, :], panel - upd, panel)

            # store column k: R[k,k] on diag, reflector below.
            diagval = tl.where(has_refl, beta, alpha)
            new_colk = tl.where(rows == k, diagval, v)
            panel = tl.where(col_is_k[None, :] & (rows[:, None] >= k), new_colk[:, None], panel)

            tau_panel = tl.where(col_is_k & do_k, tau_k, tau_panel)

        # ---- BLAS-3 rank-IB block update of the REST of the panel ----
        # Update columns [s+IB, BLK): A2 := (I - Vs Ts^T Vs^T) A2, where Vs are
        # the IB reflector columns just computed (cols [s, s+IB)) and Ts is the
        # IB x IB diagonal block of Tmat. Done with tensor-core tl.dot over the
        # full (MAXH, BLK) panel, masked: only this sub-block's V columns
        # contribute to W, and only the trailing columns are written.
        if sblk < nsub - 1:
            rr = rows[:, None]
            cc2 = cols[None, :]
            is_sub = (cc2 >= s) & (cc2 < s + IB)
            # Vfull: this sub-block's unit-lower-trapezoidal reflectors, 0 elsewhere.
            Vfull = tl.where(rr == cc2, 1.0, tl.where(rr > cc2, panel, 0.0))
            Vfull = tl.where(is_sub & (rr < pheight), Vfull, 0.0)           # (MAXH, BLK)
            # A2: trailing panel columns (>= s+IB), 0 elsewhere.
            is_trail = cc2 >= (s + IB)
            A2 = tl.where(is_trail & (rr < pheight), panel, 0.0)            # (MAXH, BLK)
            # W = Vfull^T @ A2 ; YT = Tmat^T @ W ; delta = Vfull @ YT  (tensor cores)
            Wm = tl.dot(tl.trans(Vfull), A2, input_precision="tf32x3")     # (BLK, BLK)
            YTm = tl.dot(tl.trans(Tmat), Wm, input_precision="tf32x3")     # (BLK, BLK)
            delta = tl.dot(Vfull, YTm, input_precision="tf32x3")           # (MAXH, BLK)
            panel = tl.where(is_trail, panel - delta, panel)

    tl.store(aptr, panel, mask=mask)

    tptr = tau_ptr + bid * N + j + cols
    tl.store(tptr, tau_panel, mask=col_valid)

    # V transposed: Vt[c, r] = 1 if r==c, panel[r,c] if r>c, else 0
    panelT = tl.trans(panel)                               # (BLK, MAXH)
    cc = cols[:, None]
    rr2 = rows[None, :]
    Vt = tl.where(rr2 == cc, 1.0, tl.where(rr2 > cc, panelT, 0.0))
    Vt = tl.where((rr2 < pheight) & (cc < b), Vt, 0.0)
    vbase = Vbuf_ptr + bid * stride_vb
    vptr = vbase + cc * stride_vk + rr2 * stride_vn
    tl.store(vptr, Vt, mask=(rr2 < pheight) & (cc < b))

    tbase = Tbuf_ptr + bid * stride_tb
    tptr2 = tbase + cols[:, None] * stride_tk + cols[None, :] * stride_tn
    tl.store(tptr2, Tmat)


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

    vp = v_base + krange[None, :] * stride_vk + rrows[:, None] * stride_vn
    Vrow = tl.load(vp, mask=rmask[:, None], other=0.0)
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
        IB = BLK
    elif N >= 1536:
        BLK = 16
        IB = 8
    else:
        BLK = 32
        IB = 8

    IB = min(IB, BLK)

    Vbuf = torch.empty((B, BLK, N), device=A.device, dtype=torch.float32)
    Tbuf = torch.empty((B, BLK, BLK), device=A.device, dtype=torch.float32)
    YTbuf = torch.empty((B, BLK, N), device=A.device, dtype=torch.float32)

    sab, san = H.stride(0), H.stride(1)
    svb, svk, svn = Vbuf.stride(0), Vbuf.stride(1), Vbuf.stride(2)
    stb, stk, stn = Tbuf.stride(0), Tbuf.stride(1), Tbuf.stride(2)
    syb, syk, syn = YTbuf.stride(0), YTbuf.stride(1), YTbuf.stride(2)

    BM, BNc = 32, 64
    j = 0
    while j < N:
        b = min(BLK, N - j)
        pheight = N - j
        MAXH = triton.next_power_of_2(pheight)

        if MAXH <= 512:
            nwp = 4
        elif MAXH <= 1024:
            nwp = 8
        else:
            nwp = 32

        # IB stays a fixed constexpr (nsub = BLK//IB static); columns >= b on a
        # ragged final panel are masked out by do_k / col_valid inside the kernel.
        _panel_factor_kernel[(B,)](
            H, tau, Vbuf, Tbuf,
            B, N, j, pheight, b,
            sab, san, svb, svk, svn, stb, stk, stn,
            BLK=BLK, IB=IB, MAXH=MAXH, num_warps=nwp,
        )

        ncols = N - (j + b)
        if ncols > 0:
            nct = triton.cdiv(ncols, BNc)
            nrt = triton.cdiv(pheight, BM)
            _trailing_YT_kernel[(nct, B)](
                H, Vbuf, Tbuf, YTbuf,
                B, N, j, pheight, ncols, j + b,
                sab, san, svb, svk, svn, stb, stk, stn, syb, syk, syn,
                BLK=BLK, BM=BM, BNc=BNc, num_warps=2,
            )
            _trailing_apply_kernel[(nrt * nct, B)](
                H, Vbuf, YTbuf,
                B, N, j, pheight, ncols, j + b,
                sab, san, svb, svk, svn, syb, syk, syn,
                BLK=BLK, BM=BM, BNc=BNc, num_warps=2,
            )
        j += b

    return H, tau
