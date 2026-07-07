# GPUMODE QR — Megakernels

## Executive summary

The task factors a large batch of matrices (a QR decomposition of each). The baseline,
`torch.geqrf`, factors them one at a time, and for the batched shapes that means a storm of tiny
GPU launches — about 73,000 of them on the n=512 batch, with **over 90% of the wall-clock spent in
launch overhead** rather than arithmetic.

A **megakernel** is the most aggressive possible answer: do the *entire* factorization of a matrix
inside **one GPU launch**, start to finish, keeping every intermediate on-chip. This report uses
that strict definition throughout — one kernel, whole algorithm. A kernel that factors only part of
the matrix and hands the rest to another kernel or a library call is *not* a megakernel here, and
those are called out where they matter, because they turn out to be the designs that actually won.

The finding, stated plainly and then explained by the algorithm's structure:

> **A true megakernel wins only for small matrices. For large matrices it is not just slower but
> structurally wrong, and every leading result abandoned it in favor of splitting the work back
> into separate kernels.** No whole-matrix megakernel ever carried a large size in any winning
> submission across the whole project — the ~82× (CUDA C++) and ~64× (Triton) winners are both
> *de-fused* designs, as were the earliest winning submissions before them.

Why that is true comes down to the three phases of the algorithm and what limits each one. The rest
of this report builds that argument, then lists every true megakernel that was tried.

## The three phases, and what limits each

Blocked Householder QR — the algorithm every kernel here uses — marches down the matrix in strips
("panels"). Each step has three phases:

| Phase | What it computes | What limits it (its bottleneck) |
|---|---|---|
| **Panel factorization** | Turns one narrow strip of columns into Householder reflectors | **Latency.** Each reflector depends on the previous one, so it's a serial chain with almost no arithmetic and almost no memory traffic. It is *inherently* slow-and-thin: measured panel time is essentially flat whether you give it 148 matrices or 640 (356 µs vs 404 µs), because the chain length, not the workload, sets the time. |
| **Compact-WY build** *(optional)* | Assembles the panel's reflectors into one small triangular factor so the next phase can be one big multiply | A short serial step. Small kernels skip it entirely and apply reflectors one at a time ("unblocked"); it only earns its keep once the trailing update below is large. |
| **Trailing update** | Applies the finished panel to *all the remaining columns* of the matrix | **Compute.** This is an O(n³) matrix-multiply and the dominant cost at large sizes. It is embarrassingly parallel and wants to be spread across the *entire* GPU — all 148 processing units — to go fast. |

The two expensive phases therefore want **opposite** things from the hardware:

- **The panel** wants to sit still in one place — one block, on-chip, minimal parallelism; spreading
  it out only adds communication to a latency-bound chain.
- **The trailing update** wants maximum spread — every processing unit busy.

To fuse both phases into one kernel, one block must be sized and shaped for both at once — and a
block has a single warp count and a single shared-memory footprint for its whole life. The panel
wants few warps and its V/T resident on-chip; the trailing GEMM wants many warps and many blocks
resident per SM so it can hide the latency of streaming the trailing columns from global memory.
There is no single shape that is good at both, so every way of building the megakernel sacrifices
one phase to serve the other. **That is the tension, and which phase gets sacrificed — and whether
the sacrifice is affordable — depends entirely on the matrix size.**

### The three ways to build it

There are essentially three constructions for a whole-algorithm megakernel, and all three were built
and measured:

1. **One matrix resident per block.** Load the whole matrix into shared memory, one block per matrix,
   do everything on-chip. Ideal for the panel; works only while the matrix fits in shared memory.
2. **One matrix spread across many blocks.** Split the matrix's rows across blocks so the trailing
   GEMM has parallelism, and coordinate the phases with a device-wide `grid.sync` each column.
3. **One matrix per block, streaming (tiled) through it.** Keep only a chunk resident, stream the
   rest of the matrix through global memory — so it fits at any size, one block per matrix.

The next two sections trace what each construction does at small and large sizes.

## Why fusing wins at small sizes

When the matrix is small (n = 32, n = 176), all three phases are *cheap*, and the trailing update
has too little arithmetic to benefit from the whole chip anyway. What dominates instead is the
launch overhead — the thousands of tiny kernel starts. Here the megakernel's one-launch design is
exactly right: it deletes the overhead, and because the work is latency-bound rather than
compute-bound, giving each matrix a single block costs nothing. The proof that it's overhead and not
arithmetic: the winning n=176 megakernel wins *while leaving most of the GPU idle*, and when someone
added a separate matrix-multiply step back into that path, it got **slower**, not faster.

At these sizes the megakernel also skips the compact-WY phase — with so few columns, applying
reflectors one at a time (unblocked) is simpler and just as fast.

## Why fusing loses at large sizes

Each of the three constructions hits a different wall at large n — which is the real evidence that
the problem is structural, not a tuning knob:

1. **One matrix resident per block — runs out of shared memory.** The whole matrix must fit in the
   228 KB scratchpad, and a full FP32 matrix fits only up to about **n = 241**. At n = 352 it needs
   495 KB; at n = 512, a full megabyte. Past ~241 this construction simply cannot be built, and every
   attempt to force it was reverted.

2. **One matrix spread across many blocks — pays a crippling synchronization cost.** Splitting a
   matrix across blocks gives the trailing GEMM its parallelism, but the panel is a serial chain, so
   the blocks must now coordinate with a device-wide `grid.sync` every column. That barrier traffic
   dominated: the cooperative version was correct but ran **~16× slower** than a plain library panel.
   The latency-bound panel gets nothing from the extra blocks; it just pays to synchronize them.

3. **One matrix per block, streaming — the in-kernel GEMM is simply worse than the library, and
   residency starves it.** This is the interesting case, because at the large batch (640 matrices)
   one-block-per-matrix *does* fill the whole GPU — every SM is busy on its own matrices, so this is
   not about leaving hardware idle. It loses for two on-SM reasons instead. First, fusing forbids
   calling the vendor GEMM (cuBLAS) from inside the kernel, so the trailing multiply must be
   hand-written with tensor-core intrinsics — and the hand-written version measured about **7× slower
   than cuBLAS** on the same work. Second, the trailing columns (the bulk of the matrix) do not fit
   on-chip and are streamed from global memory; hiding that memory latency needs several blocks
   resident per SM to switch between, but the panel's resident footprint (~210 KB) leaves room for
   only one block per SM, so the tensor cores stall on the stream with nothing to switch to. The tell
   that this is a genuine bind rather than a parameter: forcing a second block per SM was **worse**
   (98,000 µs vs 54,800 µs), because two matrices' residency fight even harder for shared memory. End
   to end this construction ran **8.6× slower at n = 512 and 31× slower at n = 1024** than the
   library-based split.

The escape from all three is the same: **de-fuse.** Write the panel's V and T out to global memory,
then apply the trailing update as a separate **batched cuBLAS call**. That one move fixes every wall
at once — the matrix no longer needs to be resident (no capacity limit), the phases no longer need a
grid barrier (no sync cost), and the trailing update is now the vendor's GEMM at full throughput
instead of a hand-rolled one throttled by low occupancy. The cost is the very thing fusion set out to
avoid — an extra launch and a round-trip through global memory — but at large n the O(n³) trailing
GEMM is so dominant that getting it done by cuBLAS dwarfs that overhead. This de-fused design is what
every winning submission uses for its large sizes. The crossover sits around n = 176: the largest
size where fusing still wins, and even there it is marginal (it won on one batch and regressed on
another).

## Every true megakernel that was tried

Only whole-algorithm kernels appear below (one launch, panel *and* trailing on-chip). The
**Construction** column keys each kernel to the three designs above — **(1)** one matrix resident per
block, **(2)** one matrix spread across blocks, **(3)** one matrix per block streaming through global
memory. The de-fused panel kernels that actually won the large sizes are **not** megakernels and are
omitted here by design — they are the winning alternative described above. Speedups are against the
relevant `torch.geqrf` baseline (B200 ≈ 131,000 µs; RTX Pro 6000 ≈ 117,000 µs) and are **not
comparable across GPUs**. Commits point to the best/representative instance of each kernel; all touch
`problems/linalg/qr_v2/submission.py`.

### B200 CUDA C++ — the main effort (~82×; the winning submission is de-fused)

| Megakernel | Constr. | n | What it does & how it's implemented | Correct? | Outcome | Commit |
|---|---|---|---|---|---|---|
| `blocked_qr_tiny` | 1 | 32 | One warp per matrix does the whole 32×32 factorization in registers using warp shuffles — no shared memory, no barriers; it loops the columns, forming each reflector and applying it to the trailing columns in-register, one launch for the whole batch | Yes | **Kept** — the small-*n* path of the winning submission. Fusing is unambiguously right here, but n=32 is a single panel with no real trailing update to fuse, so it is the easy case and adds little to the headline speedup | `4f9fb73d0` |
| `qr_mega_resident_kernel` | 1 | 176 | One block per matrix loads the whole 176×176 matrix into ~124 KB of shared memory and runs a right-looking *unblocked* Householder there — every reflector and every rank-1 trailing update on-chip — then writes the result back, one launch | Yes | **Kept, but marginal** — the crossover size. This kernel is live in a winning submission, yet closely related whole-matrix variants (a blocked/compact-WY form and a plain unblocked form, holding the matrix in ~192 KB) *regressed* — 652 µs and 524 µs versus ~305 µs for the de-fused split. The win for fusing is thin here and flips with batch size | `4f9fb73d0` |
| `mega_qr_persistent_kernel` | 1 | 512 | One block factors an entire matrix resident in shared memory (reflectors applied to the trailing columns one at a time in plain SIMT, no library call); a fixed grid of 148 persistent blocks pulls matrices from a shared atomic queue so the resident design reaches n=512 | Yes (incl. ill-conditioned) | **Abandoned** — correct but ~935 ms/call (~260× slower); left disabled. One block per matrix starves the trailing update | `42ab334d0` |
| `hh_coop_panel_kernel` | 2 | 512 | One matrix's rows split across all SMs via `cudaLaunchCooperativeKernel`, with a device-wide `grid.sync` every column to coordinate the phases; emits the flat `(V, τ)` directly | Yes | **Abandoned** — correct but **~16× slower** (~898,000 µs vs 54,600 for a library panel); the per-column grid barriers and per-column block-reductions dominate | `c58ddfe45` |
| `qr_mega_la_tc` (and `qr_mega_la`) | 3 | 512, 1024 | One block per matrix runs the full right-looking blocked factorization — panel, compact-WY, and the trailing update — in a single launch, doing the trailing matrix-multiply in-kernel on tensor cores (TF32×3 WMMA) and streaming the trailing columns through global memory rather than holding them resident (`_la` = look-ahead, `_tc` = tensor-core trailing) | Yes | **Abandoned** — 8.6× (n=512) to 31× (n=1024) slower than the batched-library split. The definitive large-*n* measurement: the hand-written in-kernel GEMM is ~7× slower than cuBLAS and residency caps the SM at one block, so it cannot hide the streamed trailing columns | `0dab5261` |

This CUDA C++ effort spans the project's full history — the technique and the first megakernels
appeared in its earliest experiments — and even those first winning submissions were **already de-fused**:
the trailing update went to a separate batched library GEMM, and every attempt to pull it on-chip
lost. A whole-matrix megakernel never carried the large sizes, from the first submission to the last.

### RTX Pro 6000 — controlled language experiments (24 short experiments; separate baseline)

| Megakernel | Constr. | n | Language | What it does & how it's implemented | Correct? | Outcome |
|---|---|---|---|---|---|---|
| `_mega_kernel` | 3 | 512 | Triton | One Triton program per matrix (`program_id` = matrix index) does the whole factorization on-chip — panel in registers, trailing applied as `(I − V·T·Vᵀ)` in the same kernel — with zero inter-kernel launches | Yes | **Ran and validated** at ~9,500 µs — but still lost to that experiment's de-fused panel+trailing design, so it was not the winning route |
| `@cute.kernel` per-matrix QR (n=32/176) | 1 | 32, 176 | CuTe DSL | One `@cute.kernel` per matrix; the n=176 version loops all 176 columns forming and applying reflectors in-kernel (whole matrix), using block-index-per-matrix and warp-level reductions. *(The exact symbol names are not on a retained branch — the preserved CuTe branches keep only marker/no-op/probe kernels like `_cute_marker_kernel`.)* | Yes (small *n*) | **Small-only** — whole-matrix for n=32/176 (1.69× best); n≥512 fell back to the library |
| `_megakernel_qr` | 1 | 512 | Triton | Attempt at one Triton program holding the full-height n=512 matrix resident and doing all phases; Triton materializes every live tile, so the footprint blew past the scratchpad | — | **Never ran** — needs 388 KB of shared memory against a ~99 KB limit on this GPU; `OutOfResources` at launch |

Across these experiments the *winning* designs were, again, de-fused: the best Triton result (42.8×) and
the best CUDA result (~21×) both factor a panel in one kernel and apply the trailing update in a
*separate* kernel or batched library GEMM — not megakernels.

## Takeaways

- **The bottleneck flips with size, and that flip is the whole result.** Small matrices are
  launch/latency-bound, which is exactly what one-launch fusion fixes; large matrices are dominated
  by the O(n³) trailing GEMM, which fusion is bad at — so the win crosses over from fused to de-fused
  as n grows.

- **A megakernel forces two phases with opposite hardware needs into one block shape.** The panel
  wants few warps and its data resident; the trailing GEMM wants many resident blocks per SM to hide
  the memory latency of streaming the trailing columns. One block has a single shape, so it serves
  one phase and throttles the other. At large n the only fix is to split them — which is, by
  definition, not a megakernel.

- **The large-batch loss is not idle hardware — it is a worse GEMM.** At 640 matrices, one block per
  matrix already fills every SM. The fused kernel still loses because fusion bars a cuBLAS call (its
  hand-written in-kernel GEMM measured ~7× slower) and the panel's resident footprint caps the SM at
  one block, so it cannot hide the trailing columns streamed from memory. De-fusing restores the
  vendor GEMM at full occupancy.

- **No whole-matrix megakernel ever won a large size — anywhere in the project.** Every leading
  submission (~82× CUDA C++, ~64× Triton, and the earliest winners before them) is de-fused. True
  megakernels earned their place only at n ≤ 176.

- **Three constructions, three different walls — so it is structural, not a tuning knob.** Resident
  (1) runs out of shared memory past n ≈ 241; cooperative (2) pays ~16× for its grid barriers;
  streaming (3) is throttled by the hand-written GEMM and one-block occupancy (and forcing a second
  block per SM was *worse*, 98k vs 54.8k µs). No parameter closes any of the three.

- **"Correct" and "fast" were separate bars, and fast was the one that failed.** The persistent
  work-queue megakernel was correct on every case including ill-conditioned inputs, yet ~260× too
  slow. At large n, making a megakernel *work* was rarely the obstacle; making it competitive was
  impossible.

## What the model struggled with

All B200 work was one model (Opus 4.8). It could write a correct megakernel, but it repeatedly
tripped on specific low-level details of the hardware and the tools. The concrete failure modes, by
layer of the stack:

**Tensor-core instructions: it defaulted to the older API and couldn't drive the new one.** Every
working fused trailing update used the legacy `nvcuda::wmma` (HMMA) path. The model *tried* the
Blackwell-native 5th-generation tensor-core instructions (`tcgen05`) many times, but they failed at
the assembler — `ptxas ... error : Illegal modifier '.scale_vec::1X' for instruction
'tcgen05.mma'` — and it correctly diagnosed but never fixed the cause ("my instruction syntax was
slightly off"). It also mis-estimated the payoff of new low-precision modes until it measured them:
"legacy `mma.sync` FP8 = FP16 throughput on B200 (microbenchmarked 0.96×, NOT 2–4×)."

**Tensor-core memory layout: silent wrong answers from fragment alignment.** The sharpest bug was
not a crash but a silently corrupt result: "WMMA accumulator `load`/`store_matrix_sync` need an
8-aligned leading dim; the block's odd `LDS = ob|1` (a bank-conflict pad) silently corrupted the
in-place `C := C + VY` round-trip → a deterministic, precision-INDEPENDENT orthogonality defect."
The model first chased this as a numerical-precision problem (trying 1×/3×TF32) before realizing the
constant error across every precision was the signature of a layout bug, not arithmetic.

**Synchronization: races, wrong barrier scope, and cooperative-launch cost.** A missing barrier
produced NaNs ("Fixed missing `__syncthreads` after smem load"); a partial-warp shuffle under
divergence silently zeroed data ("`__shfl_sync`-under-divergence bug was zeroing strict-lower V"); a
hand-written cross-warp reduction via a named `bar.sync` was "unreliable on sm_100 EVEN WITH a memory
clobber on the inline asm." When it reached for cooperative-groups `grid.sync` to fuse across blocks,
the synchronization cost dwarfed the work — a per-column grid barrier ran "~16× slower" — and a
software grid barrier "DEADLOCKS non-deterministically."

**Registers and occupancy: the fused kernel spilled, and forcing occupancy made it worse.** A
one-CTA-per-matrix kernel hit "255 registers/thread (the max) + 39.4M local loads + 32.7M local
stores" — i.e. massive spilling to local memory — and sat at "1 block/SM, 12.5% occupancy." Stacking
tensor-core fragments overflowed the file outright: "32 warps × the 3×TF32 fragment set overflows the
64K-reg/block budget (`cudaErrorLaunchOutOfResources`)." And the obvious lever backfired: capping
registers with `__launch_bounds__` to force more blocks/SM "forces register SPILLS whose latency
outweighs the extra occupancy."

**Correctness under fusion, and a benchmark that hid it.** Packing every phase into one kernel meant
many small ways to be wrong: an under-allocated shared-memory scratch from an odd stride gave a
`cudaErrorIllegalAddress`; the low-precision path overflowed FP16's range to NaN ("`M = T⁻¹` entries
exceed FP16 max 65504 → NaN"). Worse, a quick benchmark once "gave a FALSE 24% signal by timing a NaN
kernel" — a kernel that isn't computing is fast — which is why the experiment later insisted on always
correctness-gating a timing measurement.

**Build cost: the template zoo hit the compile-time limit.** The fused design fanned out into many
template instantiations, and "the 7625-line file compiles near the remote 240 s limit, and ~1/3 of
recent submits TIMED OUT," with "pure DEAD ptxas instantiations sitting on the critical
single-TU compile path" — a failure mode that has nothing to do with the kernel's speed and
everything to do with how it was generated.

**Things it did *not* get wrong are as telling.** It knew the dynamic-shared-memory opt-in and its
size ("BLOCK with opt-in is ~227KB; the default cap is 48KB. 37KB < 48KB so no opt-in needed"),
tracked bank conflicts explicitly, and did build `cp.async` and double-buffered TMA pipelines — which
then bought little, because the fused panel is barrier-bound, not memory-latency-bound ("the dominant
stall is CTA-BARRIERS at 38.6% ... `cp.async` C-prefetch has no memory latency to hide"). The gap was
not ignorance of modern GPU APIs; it was getting the fiddly details (fragment alignment, barrier
scope, register budget, the newest instruction encodings) right under the pressure of fusing
everything into one kernel.

**In the higher-level languages, the wall was expressiveness, not detail.** Triton could not build a
fully-resident megakernel because it "materializes every live tile in SMEM" with no way to free or
alias scratch by hand ("388KB >> 99KB limit ... needs raw CUDA `__shared__` with dynamic-SMEM
opt-in"), it lacks the grid-wide barrier a cross-block fusion needs, and `tl.dot` refuses the narrow
tiles a panel produces ("`tl.dot` requires K ≥ 16"). It also hid a data race behind a clean API — a
loop-carried in-place dependency was "a silent race in Triton; a per-step `tl.debug_barrier()` fixes
it at zero speed cost" — and had a compiler footgun where "`tl.range(num_stages=1)` compiles
DIFFERENTLY from plain `range`." (The CuTe-DSL experiments hit their own toolchain walls — `redux.f32`
not supported on `sm_120a`, an IR-verifier internal error on values used outside an `scf.for` region —
but those experiments used a weaker model, so they are not evidence about the language's ceiling.)

## Scope & sources

A thematic slice through the QR project, restricted to true megakernels (one launch, whole
algorithm). It draws on the two B200 lines of work — a longer CUDA C++ effort (~82×), which spans
the project's whole history and is where the technique and the first megakernels appeared, and a
shorter Triton-steered effort (~64×) — plus the RTX Pro 6000 language experiments (24 in all: 10
Triton, 10 CUDA C++, 4 CuTe DSL). All B200 work used one model, Opus 4.8.

Every classification here was checked against the actual kernel source (does one launch do both
panel and trailing?), and every speedup against the measured number in the commit that introduced
it; speedups are stated against each line's own baseline and never cross-compared between GPUs. Two
cautions apply: the small-*n* CUDA speedups (about 4.5× at n=32 and 12–41× at n=176) come from a
later rebuild of those kernels, not from the winning submission's own history, so they are described
qualitatively rather than pinned to the winner; and the large-*n* fused kernel, though written in
CUDA/WMMA, was produced within the Triton-steered line of work.
