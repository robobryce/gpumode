import torch
import triton
import triton.language as tl
from task import input_t, output_t


# =============================================================================
# Batch-MAJOR right-looking TWO-LEVEL (nested) blocked Householder QR
# (geqrf-compatible).
#
# Outer block width NB (64/128); inner sub-block width ib (8/16). For each outer
# block [jo, jo+NB):
#   1. inner factorization: ib columns at a time, _panel_factor_kernel runs the
#      unblocked Householder chain on the narrow ib sub-panel (panel resident,
#      incremental ib x ib dlarft T2 fused in) -> writes reflectors V[:, s:s+ib]
#      and tau, AND an inner BLAS-3 trailing update bounded to the REST OF THE
#      OUTER BLOCK [s+ib, NB) so the next sub-panel sees the right data.
#   2. _build_wide_T_kernel: assemble the full NB x NB compact-WY T from the
#      accumulated outer-block V + tau (dlarft forward over all NB columns).
#   3. outer trailing update against [jo+NB, N): ONE wide-K (K=NB) BLAS-3 pass
#      A_trail := (I - V T^T V^T) A_trail, split into TWO race-free kernels.
#
# Widening K from ib to NB on the OUTER trailing update means the big trailing
# matrix is swept N/NB times instead of N/ib times -> far fewer passes / less
# HBM traffic on the dominant n=512/1024 shapes. The inner serial Householder
# chain stays narrow (ib registers) so panel occupancy does not regress.
#
# H upper-tri = R, below-diag = reflectors, tau separate. tf32x3 (error-
# corrected TF32) for the trailing GEMMs (passes all ill-conditioned cases).
# Verified bit-equivalent to torch.geqrf in a PyTorch two-level prototype.
# =============================================================================


@triton.jit
def _panel_factor_kernel(
    A_ptr, tau_ptr, Vbuf_ptr, Tbuf_ptr,
    B, N, j, pheight, b, koff, colstop,
    stride_ab, stride_an,
    stride_vb, stride_vk, stride_vn,
    stride_tb, stride_tk, stride_tn,
    BLK: tl.constexpr, MAXH: tl.constexpr,
):
    # Factor the narrow ib sub-panel A[j:, j:j+b] (b <= BLK).  Reflectors land in
    # Vbuf at k-offset `koff` (so the same wide outer V buffer holds every inner
    # block).  The within-panel rank-1 updates touch columns up to `colstop`
    # (the OUTER block boundary), leaving the global trailing untouched -- that
    # is the wide-K outer trailing kernel's job.
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
    # ib x ib WY T2 built incrementally (LAPACK dlarft forward).
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

        # w[c] = v_k . panel[:,c]; reused for the within-panel update and the
        # incremental T2 (z[c]=w[c] for c<k since v_k is supported on rows>=k).
        w = tl.sum(v[:, None] * panel, axis=0)            # (BLK,)

        z = tl.where(cols < k, w, 0.0)
        Tcol = -tau_k * tl.sum(Tmat * z[None, :], axis=1)  # (BLK,)
        Tcol = tl.where(cols < k, Tcol, 0.0)
        Tcol = tl.where(cols == k, tau_k, Tcol)
        Tmat = tl.where(col_is_k[None, :], Tcol[:, None], Tmat)

        # apply H_k to trailing panel columns (c > k, within this ib sub-panel)
        upd = tau_k * v[:, None] * w[None, :]
        col_gt_k = cols > k
        panel = tl.where(col_gt_k[None, :], panel - upd, panel)

        diagval = tl.where(has_refl, beta, alpha)
        new_colk = tl.where(rows == k, diagval, v)
        panel = tl.where(col_is_k[None, :] & (rows[:, None] >= k), new_colk[:, None], panel)

        tau_panel = tl.where(col_is_k & do_k, tau_k, tau_panel)

    tl.store(aptr, panel, mask=mask)

    tptr = tau_ptr + bid * N + j + cols
    tl.store(tptr, tau_panel, mask=col_valid)

    # store reflectors V (transposed) into the wide outer V buffer at k-offset
    # koff.  The buffer row axis is the GLOBAL row index, so a reflector whose
    # local panel row is `rr` (relative to panel top j) lands at global row j+rr.
    # This single convention lets the inner (j=col) and outer/T-build (j=jo)
    # readers all index V by global row with no per-block offset bookkeeping.
    panelT = tl.trans(panel)                               # (BLK, MAXH)
    cc = cols[:, None]
    rr = rows[None, :]
    Vt = tl.where(rr == cc, 1.0, tl.where(rr > cc, panelT, 0.0))
    Vt = tl.where((rr < pheight) & (cc < b), Vt, 0.0)
    vbase = Vbuf_ptr + bid * stride_vb + koff * stride_vk
    vptr = vbase + cc * stride_vk + (j + rr) * stride_vn
    tl.store(vptr, Vt, mask=(rr < pheight) & (cc < b))

    # store the inner ib x ib T2 (Tbuf is sized ib x ib, used by the inner
    # trailing update; the wide outer T is built separately).
    tbase = Tbuf_ptr + bid * stride_tb
    tptr2 = tbase + cols[:, None] * stride_tk + cols[None, :] * stride_tn
    tl.store(tptr2, Tmat)


@triton.jit
def _build_wide_T_kernel(
    Vbuf_ptr, tau_ptr, Twide_ptr,
    B, N, jo, pheight, NBe,
    stride_vb, stride_vk, stride_vn,
    stride_tb, stride_tk, stride_tn,
    NB: tl.constexpr, MAXH: tl.constexpr,
):
    # Assemble the NB x NB compact-WY T over the full outer block from the
    # accumulated reflectors V (k in [0,NBe)) and tau (LAPACK dlarft forward):
    #   T[k,k]   = tau_k
    #   T[:k,k]  = -tau_k * T[:k,:k] @ (V[:, :k]^T v_k)
    # One program per matrix; loops k=0..NB.  V is stored transposed in Vbuf as
    # V[k, row], so V[:, :k]^T v_k = sum_row V[i,row]*V[k,row] for i<k.
    bid = tl.program_id(0)
    if bid >= B:
        return

    rows = tl.arange(0, MAXH)
    ks = tl.arange(0, NB)
    row_valid = rows < pheight
    k_valid = ks < NBe

    # load full transposed V block: Vt[k, row]  (NB, MAXH).  V is stored by
    # GLOBAL row, so the outer block's rows [0,pheight) live at global jo+rows.
    vbase = Vbuf_ptr + bid * stride_vb
    vptr = vbase + ks[:, None] * stride_vk + (jo + rows[None, :]) * stride_vn
    Vt = tl.load(vptr, mask=k_valid[:, None] & row_valid[None, :], other=0.0)  # (NB, MAXH)

    # tau over the outer block
    tptr = tau_ptr + bid * N + jo + ks
    tau_blk = tl.load(tptr, mask=k_valid, other=0.0)                            # (NB,)

    Tmat = tl.zeros((NB, NB), dtype=tl.float32)
    for k in range(0, NB):
        col_is_k = ks == k
        vk = tl.sum(tl.where(col_is_k[:, None], Vt, 0.0), axis=0)               # (MAXH,)
        tau_k = tl.sum(tl.where(col_is_k, tau_blk, 0.0))
        # z[i] = V[i,:] . v_k for i<k   (i.e. row-reduce Vt*vk over rows)
        z = tl.sum(Vt * vk[None, :], axis=1)                                    # (NB,)
        z = tl.where(ks < k, z, 0.0)
        Tcol = -tau_k * tl.sum(Tmat * z[None, :], axis=1)                       # (NB,)
        Tcol = tl.where(ks < k, Tcol, 0.0)
        Tcol = tl.where(ks == k, tau_k, Tcol)
        Tmat = tl.where(col_is_k[None, :], Tcol[:, None], Tmat)

    tbase = Twide_ptr + bid * stride_tb
    tp = tbase + ks[:, None] * stride_tk + ks[None, :] * stride_tn
    tl.store(tp, Tmat)


# Trailing update is split into TWO kernels to be race-free: the first reads
# A_trail and produces YT = T^T V^T A_trail (read-only consumer of A_trail); the
# second reads YT (read-only) and applies A_trail -= V @ YT (each program owns a
# disjoint output tile). Fusing them races. Parameterized by koff (k-offset into
# the wide V buffer) so the SAME kernels serve both the inner (K=ib, koff=s) and
# outer (K=NB, koff=0) trailing updates.
@triton.jit
def _trailing_YT_kernel(
    A_ptr, Vbuf_ptr, Tbuf_ptr, YT_ptr,
    B, N, j, pheight, ncols, jb, koff,
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
    v_base = Vbuf_ptr + bid * stride_vb + koff * stride_vk

    # W = V^T @ A_trail over all panel rows, in chunks of BM
    W = tl.zeros((BLK, BNc), dtype=tl.float32)
    nchunks = tl.cdiv(pheight, BM)
    for ci in range(0, nchunks):
        rr = ci * BM + tl.arange(0, BM)
        rrmask = rr < pheight
        ap = a_trail_base + rr[:, None] * stride_an + ccols[None, :]
        achunk = tl.load(ap, mask=rrmask[:, None] & cmask[None, :], other=0.0)   # (BM,BNc)
        vp = v_base + krange[:, None] * stride_vk + (j + rr[None, :]) * stride_vn
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
    B, N, j, pheight, ncols, jb, koff,
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
    v_base = Vbuf_ptr + bid * stride_vb + koff * stride_vk

    # V[rrows,:] : (BM, BLK).  V is stored by global row -> add panel top j.
    vp = v_base + krange[None, :] * stride_vk + (j + rrows[:, None]) * stride_vn
    Vrow = tl.load(vp, mask=rmask[:, None], other=0.0)
    # YT[:, ccols] : (BLK, BNc)
    yp = YT_ptr + bid * stride_yb + krange[:, None] * stride_yk + ccols[None, :] * stride_yn
    YT = tl.load(yp, mask=cmask[None, :], other=0.0)
    delta = tl.dot(Vrow, YT, input_precision="tf32x3")                            # (BM,BNc)

    ap = a_trail_base + rrows[:, None] * stride_an + ccols[None, :]
    amask = rmask[:, None] & cmask[None, :]
    aorig = tl.load(ap, mask=amask, other=0.0)
    tl.store(ap, aorig - delta, mask=amask)


def _trailing_update(H, Vbuf, Tbuf, YTbuf, B, N, j, pheight, jb, ncols, koff,
                     BLK, BM, BNc, sab, san, svb, svk, svn, stb, stk, stn,
                     syb, syk, syn, nw):
    nct = triton.cdiv(ncols, BNc)
    nrt = triton.cdiv(pheight, BM)
    _trailing_YT_kernel[(nct, B)](
        H, Vbuf, Tbuf, YTbuf,
        B, N, j, pheight, ncols, jb, koff,
        sab, san, svb, svk, svn, stb, stk, stn, syb, syk, syn,
        BLK=BLK, BM=BM, BNc=BNc, num_warps=nw,
    )
    _trailing_apply_kernel[(nrt * nct, B)](
        H, Vbuf, YTbuf,
        B, N, j, pheight, ncols, jb, koff,
        sab, san, svb, svk, svn, syb, syk, syn,
        BLK=BLK, BM=BM, BNc=BNc, num_warps=nw,
    )


def custom_kernel(data: input_t) -> output_t:
    A = data
    B, N, _ = A.shape
    H = A.clone()
    tau = torch.zeros((B, N), device=A.device, dtype=torch.float32)

    # Inner sub-block ib (the serial Householder chain width -> register footprint)
    # and outer block NB (the wide-K trailing-update width).
    if N <= 32:
        IB = min(16, N)
        NB = IB
    elif N >= 1536:
        # Tall panels (n>=2048): narrow ib for occupancy; modest NB.
        IB = 16
        NB = 32
    else:
        IB = 16
        NB = 64

    NB = min(NB, N)
    IB = min(IB, NB)

    # V buffer must be zeroed: a reflector at k-index kk is supported only on
    # global rows >= jo+kk, but the wide-K trailing reads ALL rows [jo, N) for
    # every k in [0,NB) (and k in [NBe,NB) on the last short block) -- the unread
    # upper-triangular dead zone must read as 0, not stale/garbage.  Zeroed once
    # per outer block below (panel_factor only writes each reflector's support).
    Vbuf = torch.zeros((B, NB, N), device=A.device, dtype=torch.float32)
    Tin = torch.empty((B, IB, IB), device=A.device, dtype=torch.float32)
    Twide = torch.empty((B, NB, NB), device=A.device, dtype=torch.float32)
    YTbuf = torch.empty((B, NB, N), device=A.device, dtype=torch.float32)

    sab, san = H.stride(0), H.stride(1)
    svb, svk, svn = Vbuf.stride(0), Vbuf.stride(1), Vbuf.stride(2)
    sib, sik, sin_ = Tin.stride(0), Tin.stride(1), Tin.stride(2)
    swb, swk, swn = Twide.stride(0), Twide.stride(1), Twide.stride(2)
    syb, syk, syn = YTbuf.stride(0), YTbuf.stride(1), YTbuf.stride(2)

    BM, BNc = 32, 64

    jo = 0
    while jo < N:
        NBe = min(NB, N - jo)
        colstop = jo + NBe
        pheight_o = N - jo
        MAXH_o = triton.next_power_of_2(pheight_o)

        # clear the V region this block will read (rows [jo, N), all NB cols) so
        # the unwritten reflector dead-zone reads as 0 (correctness, see alloc).
        Vbuf[:, :, jo:].zero_()

        # ---- inner factorization of the outer panel [jo, jo+NBe) ----
        s = 0
        while s < NBe:
            ib = min(IB, NBe - s)
            col = jo + s
            pheight = N - col
            MAXH = triton.next_power_of_2(pheight)
            if MAXH <= 512:
                nwp = 4
            elif MAXH <= 1024:
                nwp = 8
            else:
                nwp = 32

            _panel_factor_kernel[(B,)](
                H, tau, Vbuf, Tin,
                B, N, col, pheight, ib, s, colstop,
                sab, san, svb, svk, svn, sib, sik, sin_,
                BLK=IB, MAXH=MAXH, num_warps=nwp,
            )

            # inner trailing update: columns [col+ib, colstop) within outer block
            inner_ncols = colstop - (col + ib)
            if inner_ncols > 0:
                _trailing_update(
                    H, Vbuf, Tin, YTbuf, B, N, col, pheight, col + ib, inner_ncols,
                    s, IB, BM, BNc, sab, san, svb, svk, svn, sib, sik, sin_,
                    syb, syk, syn, 2,
                )
            s += ib

        # ---- build the wide NB x NB T over the whole outer block ----
        if MAXH_o <= 512:
            nwt = 4
        elif MAXH_o <= 1024:
            nwt = 8
        else:
            nwt = 16
        _build_wide_T_kernel[(B,)](
            Vbuf, tau, Twide,
            B, N, jo, pheight_o, NBe,
            svb, svk, svn, swb, swk, swn,
            NB=NB, MAXH=MAXH_o, num_warps=nwt,
        )

        # ---- outer trailing update against [jo+NBe, N): wide K=NB ----
        outer_ncols = N - colstop
        if outer_ncols > 0:
            _trailing_update(
                H, Vbuf, Twide, YTbuf, B, N, jo, pheight_o, colstop, outer_ncols,
                0, NB, BM, BNc, sab, san, svb, svk, svn, swb, swk, swn,
                syb, syk, syn, 2,
            )
        jo += NBe

    return H, tau
