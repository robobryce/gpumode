---
name: qr-megakernel-report-facts
description: "Verified primary-source facts for the GPUMODE QR megakernel report (strict \"one kernel = whole algorithm\" definition)"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa613736-6931-4e5f-b956-3dc2660e0049
---

Facts for the QR megakernels report (`autocuda/2026-06-13-to-2026-06-29-qr-consolidated-report-megakernels-v3.md`), all verified against submission code + commit bodies + harness logs (2026-07-05). Strict definition: a TRUE megakernel does the WHOLE QR of a matrix (panel + trailing) in ONE launch/matrix; panel-only + separate/library trailing is NOT one.

**Three phases & bottlenecks (the report's spine):** panel factorization = serial-latency-bound (dependent reflector chain, near-zero FLOPs/traffic; flat across batch: 356µs@B=148 ≈ 404µs@B=640); compact-WY T-build = small serial, optional (unblocked kernels skip it); trailing update = O(n³) compute-bound, embarrassingly parallel, wants all 148 SMs. Fusability: small n → all phases tiny, whole thing launch/latency-bound, one CTA suffices → fuse wins. Large n → trailing dominates & needs whole chip, but a megakernel gives one matrix to one CTA (~1/148 of compute); panel (wants 1 CTA) & trailing (wants whole chip) have CONTRADICTORY hardware needs → can't satisfy both in one kernel → de-fuse wins. Crossover ≈ n=176 (marginal: megakernel kept in qr_v2 champion B=40, but regressed on qr_py).

**True megakernels:** B200 CUDA — `blocked_qr_tiny` n=32 (whole matrix in regs, KEPT), `qr_mega_resident_kernel` n=176 (whole matrix in SMEM, unblocked, KEPT), `mega_qr_persistent_kernel` n=512 (148-CTA work-queue, correct incl ill-cond, ~935ms/call =260× slower, GATED OFF), full-matrix-resident n≥352 NEVER VIABLE. B200 Triton-steered run — `qr_mega_la_tc` n=512/1024 (CUDA/WMMA via NVRTC, whole matrix, 8.6× slower n=512 / 31× n=1024 vs batch-major, ABANDONED), `_panel_trailing_mega_kernel` n=1024 Triton (gated off, VRES OOMs 400KB>227KB). RTX — CuTe n=32/176 (1.69×), `_mega_kernel` j057n57jl n=512 Triton (RAN+validated ~9,500µs but slower than de-fused winner), full-height Triton kj7cvsimx (OOM 388KB>99KB, NEVER RAN).

**Key numbers:** B200 baseline torch.geqrf ~131,000µs (~73k launches @ n=512 B=640, >90% launch overhead). Winners all DE-FUSED: CUDA ~82×, Triton ~64×, qr_py origin ~37×. Capacity: full FP32 matrix fits to n≈241; n=352=495KB, n=512=1MB, cap 228KB/SM. RTX baseline ~117,000µs. CAVEAT: small-n CUDA speedups (4.5× n=32, ~12–41× n=176) are from the LATER 2026-06-29 rebuild, NOT ancestors of the 2026-06-23 champion — do not attribute to champion.

Exclusions (NOT megakernels, panel+separate trailing): B200 Triton `_panel_factor_kernel` (~64× workhorse), RTX Triton q1nb1oliq (42.8×), RTX CUDA ocix8g7hp (~21×), qr_py WMMA `block_qr_wmma_kernel` (per-OB-block). Related: [[eigh-run-node-facts]].
