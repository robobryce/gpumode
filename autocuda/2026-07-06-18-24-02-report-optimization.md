# linalg/qr_v2 Optimization Report

## Executive Summary

This run explored several distinct implementations of a specialized `B640 x 512` QR megakernel and found a few large structural improvements followed by increasingly small pipeline refinements. Those improvements never overcame the strong starting implementation on the full 12-case suite: the lowest recorded geomean was 1,419.742 us versus a 1,234.320 us baseline, or **0.8694x** overall.

| Metric | Value |
|---|---:|
| Overall best speedup vs baseline | **0.8694x** |
| Baseline | 1,234.320 us (`0b3d6910`) |
| Best recorded candidate | 1,419.742 us (`8a107892`), 15.02% slower by runtime |
| Distinct successful technique clusters | 7 |
| Successful / attempted kernel trials | 167 / 202 |
| Search breadth | 18 briefs, 221 tree nodes, 17 leaves |
| Most prevalent inherited technique | NB16/compact-WY core, carried by 13 of 18 descendant heads |

**Key takeaways**

- NB16 panel blocking and a column-major compact-WY workspace produced the largest measured marginal suite gain, **1.2468x** at `bd94579f`; the targeted dense n512 case fell from 804,274 to 59,649 us.
- Concurrency and ownership choices mattered almost as much as math: wave size, CTA ownership, and warp allocation repeatedly changed dense n512 latency by 10-50%.
- Native tcgen05/TMEM became useful only as a stationary last-128-column slab stacked on the proven SIMT panel core; attempts to tensorize the whole update were slower or incorrect.
- TMA swizzling, direct waits, and tail lookahead bought the final few percent on the target path, but sub-0.3% full-suite movements were dominated by untouched-route variance.
- The CSV's prevalence rows include ancestor commits repeated on descendant branches. They show inheritance, not independent rediscovery; originating-brief counts below are the stronger prevalence signal.

## Most Impactful Optimization Techniques

| Technique | Peak contribution | Prevalence | Representative commits |
|---|---:|---|---|
| NB16 panel blocking with compact-WY workspace | **1.2468x** | 2 originating briefs; core ancestors in 13/18 descendant heads | `bd94579f`, `37feb988`, `a6f4aa79` |
| CTA wave, thread, and warp ownership tuning | **1.1161x** | 5 originating briefs; many ownership brackets | `6d5b7310`, `2198a249`, `79220c49` |
| Balanced P8/C24 lookahead pipeline | **1.0709x** | 1 originating brief; retained by later descendants | `1a87c538` |
| DSM replication and wider update tiles | **1.0360x** | 2 originating briefs, 3 principal successful trials | `a826631b`, `f8bd588e`, `d87c9574` |
| Persistent stationary tcgen05/TMEM slab | **1.0145x** | 3 originating briefs; core ancestor in 12/18 descendant heads | `03814c2d`, `0ffeee75`, `522b6a89` |
| Shared-memory tiled input/output transpose | **1.0055x** | 1 originating brief; inherited by 13/18 descendant heads | `1b0630d1` |
| TMA swizzling, direct waits, and tail lookahead | **1.0030x** | 1 TMA origin plus 6 tail follow-up briefs | `dcf112f6`, `8cfbce69`, `2061b37e` |

**NB16 panel blocking with compact-WY workspace.** `bd94579f` replaced per-reflector global synchronization with 16-column panels and a column-major V/T workspace; `37feb988` then formed and applied compact WY explicitly. The large gain came from amortizing barriers and turning many serial reflector updates into blocked matrix work. Brief 2 independently confirmed NB16 against NB32 at `a6f4aa79`, although most later branch prevalence is inheritance from brief 1 rather than rediscovery.

**CTA wave, thread, and warp ownership tuning.** `2198a249` raised the resident wave from 32 to 128 matrices and cut the targeted dense case from 93,068 to 40,973 us. Single-CTA variants then bracketed 512, 576, 640, 768, and 1,024 threads and different panel/update splits; `6d5b7310`, `79220c49`, and `d6055f33` show that keeping enough independent update warps resident was more important than simply maximizing threads. The CSV's 1.1161x peak is a useful ranking signal but exceeds the directly reported 0.19% suite delta at `6d5b7310`, so it should not be treated as a precise decomposition.

**Balanced P8/C24 lookahead pipeline.** `1a87c538` restored a serialized dependency frontier with eight panel warps and 24 consumer warps, removed globaltimer instrumentation, and reduced dense n512 from 19,298 to 17,192 us. It stacks on the NB16/compact-WY line and worked by overlapping panel formation with disjoint far-column updates without extending the critical dependency frontier.

**DSM replication and wider update tiles.** `a826631b` copied root-CTA V/T slabs out of remote DSM once, trading a small staging cost for local shared-memory reuse. `f8bd588e` moved from four CTAs/128 columns to two CTAs/256 columns, while `d87c9574` doubled resident update tiles to 256 columns. All three reduced repeated DSM loads and task/barrier count; the technique was explored from both cluster-tiled and resident-grid designs.

**Persistent stationary tcgen05/TMEM slab.** `03814c2d` kept the final 128 columns resident in TMEM across the first 24 panels while SIMT warps continued the correctness-critical panel work. `0ffeee75` made that slab persistent across 296 CTAs and reused one TMEM allocation per CTA. This selective tensor-core use stacked successfully; broader tcgen05 replacements did not.

**Shared-memory transpose and late TMA refinements.** `1b0630d1` changed strided global input/output copies into padded warp-private 32x32 transpose tiles, improving the target by 5.51%. Later, `dcf112f6` patched Householder triangles directly in 128-byte-swizzled TMA stages and `8cfbce69` removed a named publish barrier in favor of direct 128-thread acquire waits. `2061b37e` combined direct waits with the P16/C16 tail. The table's 1.0030x peak comes from a six-phase boundary whose target latency actually regressed 0.21%; it is cross-route geomean noise, while the direct target-path measurements are the reliable evidence.

## Failed Optimization Techniques

**Whole-update tcgen05/TMEM tensorization.** Several descriptor, layout, and MMA-algebra attempts failed differential validation (`896d751f`, `b2e91662`, `46d3ab70`, `040e34cf`), while wider in-place variants hit illegal memory accesses (`91737e4f`, `ee324bf5`, `ad2677a0`). Correct TF32x3 variants were still slower because commit/wait synchronization and staging exceeded the saved arithmetic; plain TF32 at `1dfdddc5` recovered 11.26% locally but remained slower than the SIMT update.

**Occupancy-first thread reductions and dynamic register control.** Reducing the stationary kernel to 576 threads repeatedly failed dense differential checks, while 640/768-thread variants that enabled a second resident CTA lost to the one-CTA schedule through spills, barriers, or insufficient eligible warps. Brief 9's `setmaxnreg`/`noinline` role isolation remained correct but increased local-memory traffic and regressed.

**Over-aggressive stationary/lookahead overlap.** The P8/C20 stationary-era overlap at `37aaf441` passed the small official cases but failed all eight dense B640 guards and two mixed guards. A later attempt to extend lookahead into stationary epochs also failed validation in brief 16. The dependency frontier was too tight to move without publishing partially updated panels.

**Direct native TMA panel layouts.** `c8adb6b4` produced roughly 1.31 million mismatches because the tcgen B operand did not match the assumed shared-memory layout. The successful path required explicit 128-byte-swizzled TMA staging and a verified triangle patch instead of direct native consumption.

**Fully unrolled tail schedules.** `b3bf9d2d` expanded the compact phase loop into a large compile-time body, regressed about 2.1%, and generated roughly 23.9 MB of local spill requests. Smaller pair-unroll variants remained slower; compact runtime loops preserved instruction-cache behavior and register lifetime better.

## Unexplored Areas

**The other eleven benchmark routes.** Nearly all run-specific work targeted the dense `B640 x 512` path. The n32, n176, n352, n1024, n2048, n4096, mixed, rank-deficient, clustered, and near-rank routes remained inherited from the baseline, so the run had no way to recover the 15.0% full-suite deficit once the new n512 path plateaued.

**Production-batch validation as a first-class test.** Official n512 tests use batch 16, while the optimized kernel is gated on `(640, 512, 512)`. Workers added randomized B640 differential, invariance, and off-grid guards, and benchmark mode rechecked outputs, but the task's fixed test suite never directly activates the new path. Adding a reduced-cost production-grid test would make future failures cheaper and more reproducible.

**A reusable CUTLASS/CuTe implementation.** The run used bespoke inline PTX, hand-built TMA descriptors, and CUDA C++ kernels. It did not establish a CUTLASS/CuTe baseline for the same panel/update geometry; such a comparison could expose a better tensor-core schedule without repeating the descriptor and swizzle errors seen here.

**Robust overlap beyond panel 384.** Only one shallow stationary-era overlap attempt was made before it failed validation, and the final search concentrated on six/eight-phase post-flush boundaries. A formally staged double-buffer protocol with explicit per-panel ownership, rather than barrier deletion, remains an open way to overlap the stationary epochs without consuming incomplete Householder data.
