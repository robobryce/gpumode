# GPUMODE QR — Languages & DSLs

## Executive Summary

The runs authored these languages, DSLs, and library paths to attack the QR kernel, using the GPU techniques noted:

| Language / DSL / library | Role | Outcome |
|---|---|---|
| **Python / PyTorch** | host language of every submission; `torch.geqrf` baseline + `torch.bmm`/`torch.linalg` prototypes | baseline everywhere; kept in champions only as the correctness fallback |
| **CUDA C++** (`load_inline`) | the winning B200 lineage (best submission + earlier `qr_py` run) + 10 steered runs | **won** — the top-scoring B200 submission (~82×) is pure CUDA C++ |
| **Triton** (`@triton.jit`, `tl.dot`) | a full parallel B200 lineage (4 tags, ~3,000-line submissions) + 10 steered runs | reached ~64× on B200 at roughly a third of the winning lineage's code; most token-efficient; 42.8× peak (RTX Pro 6K) |
| **CUTE DSL** (CUTLASS Python: `@cute.kernel`) | 4 steered runs | worst — 1.69× ceiling, real compute only at small `n` |
| **Inline PTX** | `cp.async` staging, `mma.sync`, `tcgen05.mma`, FP8 `e4m3` | `cp.async` shipped; tensor-core PTX written but mostly gated off (regressed on B200) |
| **cuSOLVER / cuBLAS** | cuSOLVER under `torch.geqrf`; cuBLAS `GemmStridedBatchedEx`/TRSM in-kernel | cuBLAS is the workhorse GEMM inside every CUDA C++ champion |

GPU techniques exercised within them: **WMMA** (`nvcuda::wmma` FP16→FP32 fragments), **cuBLAS strided-batched GEMM**, **3×FP16 / 3×TF32 Kahan precision splitting** (≈FP32 accuracy at tensor-core rate), and — written but shelved for regressing on B200 — **5th-gen `tcgen05`** and **FP8 e4m3 `mma.sync`**.

**Key takeaways**

- **Two parallel B200 lineages — one CUDA C++, one Triton — are the core languages finding.** The B200 CUDA C++ lineage reached ~82× (1,600.9 µs) with a ~9,000-line submission; a distinct B200 Triton-steered lineage on the *same* hardware reached ~64× (2,040.9 µs) with a ~3,000-line one. Higher score at 3× the code, vs. slightly lower score at far cleaner code — that trade-off, not a lone winner, is the story. (CUTE, tried only in the separate RTX Pro 6K evals, never got a real kernel onto the shape that matters.)
- **Precision splitting is the cross-language lever** — 3×TF32 diagonal + 3×FP16 trailing (Kahan hi/lo) was reinvented independently in CUDA C++, Triton, and cuBLAS compute modes.
- **The frontier PTX (`tcgen05`, FP8) was reachable but not yet worth it** — real code exists in commits, but FP8 runs at FP16 throughput on sm_100 and the 2–4× FP8 peak needs `tcgen05`.
- **The dominant failure mode is toolchain, not algorithm** — CUDA C++'s non-portable cubin trap, Triton's ~100 KB/SM shared-memory wall, CUTE's install/PTX/JIT walls.
- **Language and model are confounded** — every 17–43× result is Opus 4.8; Codex/GPT-5.5 capped at 1.1–5× on *every* abstraction — but Opus-only, Triton still beats CUDA C++ per token.

---

## The language landscape

These reports cover automated optimization runs: an agent iteratively rewrites the QR kernel, keeps every version that is both faster and still correct, and the fastest kept version is that run's *winning submission*. This report asks which kernel-authoring language each run used and how far each got. The work splits into **three lineages**:

1. **The B200 CUDA C++ lineage.** The longer, best-scoring line of work — convoluted CUDA C++, ~9,000-line submissions — plus an earlier run on the related `qr_py` problem in the same technique family. Its best submission (`292432ed7166`) hits **1,600.9 µs, ~82×** over the 131,025 µs `torch.geqrf` baseline: the top score.
2. **The B200 Triton-steered lineage (B200).** A distinct, deliberately Triton-dominant line of work on the *same* leaderboard hardware — ~3,000-line submissions across 4 tags — reaching **2,040.9 µs, ~64×** (tag `2026-06-22-09-10-03-qr_v2`). This is a primary B200 lineage in its own right, not a stray Triton detour inside the CUDA C++ run: it got to within striking distance of the winner on a third of the code.
3. **The kernel-language experiment (RTX Pro 6K).** A separate controlled study that steered each of 24 short runs to exactly one language (10 Triton / 10 CUDA C++ / 4 CUTE DSL) on a *different* GPU with its own ~116,000 µs baseline, isolating the language from everything else. Best: **2,744.5 µs, 42.8×** vs. its own baseline (not comparable to the B200 × figures).

The two B200 lineages are the heart of a languages report: same hardware, same problem, same precision recipe — but one reached a higher score through sheer volume of CUDA C++, the other nearly matched it with far less, far cleaner Triton.

The counts below are verified against actual submission code (`git show <commit>:problems/linalg/qr_v2/submission.py`), not raw keyword frequency — an earlier keyword scan's "cute ~1103" hits turned out to be noise (`execute`, comments), so every CUTE/CUTLASS claim here was confirmed against real imports.

The B200 CUDA C++ lineage's best submission (`champion_qr_v2_submission.py`, 9,061 lines, commit `292432ed7166`) is **pure CUDA C++**:

| Marker | Count | Meaning |
|---|---|---|
| `import triton` / `@triton.jit` / `tl.dot` | **0** | no Triton in the winner |
| `__global__` | 54 | custom CUDA kernels |
| `nvcuda::wmma` fragment ops | 84 | WMMA tensor-core GEMM |
| WMMA `mma_sync` | 8 | these are WMMA `mma_sync`, *not* raw PTX `mma.sync` |
| `cp.async` | 14 | async shared-memory staging PTX |
| cuBLAS refs | 162 | strided-batched GEMM / TRSM |
| `load_inline` | 5 | 5 CUDA translation units compiled at import |
| raw `mma.sync` / `tcgen05` PTX | **0** | none in the *shipped* winner |
| `cutlass` | 1 | a profiling comment ("cutlass tf32 wide GEMM at ~20% throughput") — **not** CUTLASS/CUTE usage |

That one submission is not the whole picture. Aggregated across all 1,866 B200 `qr_v2` branches that carry a submission — spanning *both* B200 lineages — the language mix is:

| Language mix | Branches | Share |
|---|---|---|
| CUDA C++ only | 1,544 | 82.7% |
| Triton only | 230 | 12.3% |
| Triton + CUDA C++ | 69 | 3.7% |
| Pure torch (no custom kernel) | 23 | 1.2% |
| **Any Triton** | **299** | **16.0%** |
| **Any CUDA C++** | **1,613** | **86.4%** |

"CUDA C++, zero Triton" is precisely true of the *top-scoring submission*, but Triton is not marginal across the B200 work: 299 branches (16.0%) carry Triton, and the 230 Triton-only branches are largely the distinct B200 Triton-steered lineage — the 4 tags whose best submission hit ~64× on ~3,000 lines (many using the same `input_precision="tf32x3"` knob the RTX Pro 6K Triton runs later used) — not scattered exploration inside the CUDA C++ line. CUTE DSL never appears in any B200 branch.

The earlier `qr_py` run — the origin of this same CUDA C++ lineage, on a related B200 problem — tells the same story. Its winning submission (`94d3f325c466`, 12,566 lines) is pure CUDA C++ — 0 Triton, 83 `__global__`, 82 `wmma`, cuBLAS `GemmStridedBatchedEx` with `CUDA_R_16F`, built for `compute_100,code=sm_100` (plain — no `a`-suffix, because it never shipped `tcgen05`). Across all 434 `qr_py` branches: **98.2% CUDA C++, 0 Triton, 0 CUTE** (the lone `cutlass`(2)/`tcgen05`(1) hits are comments). This is where the whole CUDA C++ technique family — WMMA trailing updates, cuBLAS GEMM, CholeskyQR — was first built, then carried into `qr_v2`.

The controlled RTX Pro 6K evals was steered by a single directive appended to an otherwise-bare optimization prompt — verbatim, e.g. `Write custom kernels with CUTE DSL.` The 24 runs split **10 Triton / 10 CUDA C++ / 4 CUTE DSL** (`kl_abstraction_map.tsv`), and each winning submission genuinely uses its assigned language: the CUTE one imports `cutlass.cute` and uses `@cute.kernel`/`@cute.jit`/`cute.arch.*`/`make_ptr`; the Triton one uses `import triton` with 7 `@triton.jit` and 17 `tl.dot`; the CUDA one uses `load_inline` + `__global__` + cuBLAS.

The baseline everywhere is **`torch.geqrf`** — a serial per-matrix cuSOLVER blocked-Householder loop (~73,000 kernel launches for one n=512, batch=640 shape; SIMT-only, no tensor cores on either GPU). Every language's job was to replace it. It survives in all winning submissions only as the correctness fallback for off-grid, ill-conditioned, or small-batch inputs, never as a scored path.

---

## Head-to-head: one language per run (RTX Pro 6K)

The cleanest comparison is the RTX Pro 6K evals, where each run used exactly one language against the same ~116,000 µs `torch.geqrf` baseline (sm_120; small 3-worker trees, ~97–251 trials each). Speedups here are each run's fastest kept trial ÷ its own measured baseline µs, recovered from the raw `autocuda/<tag>-optimize-tree-worker-*-log.csv` files because the consolidated CSV's `best_speedup` column is blank (a known join bug); they reconcile exactly with the consolidated cost and optimization reports. Model per run is from `message.model` in the logs: 12 runs are Opus 4.8 (`claude-opus-4-8`), 12 are GPT-5.5 (`openai/openai/gpt-5.5`, "Codex"); **all 4 CUTE runs are Codex.**

| Run (short) | Language | Model | Baseline µs | Best µs | **Best ×** | Batch |
|---|---|---|---|---|---|---|
| q1nb1oliq | Triton | Opus | 117,536 | 2,744.5 | **42.83** | long (06-25) |
| kurboui6f | Triton | Opus | 115,820 | 3,347.1 | **34.60** | long |
| g7v577r9x | Triton | Opus | 115,731 | 4,194.5 | **27.59** | long |
| j057n57jl | Triton | Opus | 116,122 | 4,764.0 | **24.37** | long |
| tszacp309 | Triton | Opus | 116,915 | 5,118.3 | **22.84** | long |
| ocix8g7hp | CUDA C++ | Opus | 112,394 | 5,234.1 | **21.47** | long |
| kj7cvsimx | Triton | Opus | 117,106 | 5,500.5 | **21.29** | long |
| 3otwejh6b | CUDA C++ | Opus | 116,323 | 6,633.7 | **17.54** | long |
| w7jxbr46z | CUDA C++ | Opus | 117,514 | 6,854.2 | **17.14** | long |
| kpnikqu5d | CUDA C++ | Opus | 111,805 | 6,563.6 | **17.03** | long |
| m7w1ravpo | Triton | Codex | 116,501 | 23,156.6 | **5.03** | short (06-24) |
| vcp095s3n | Triton | Codex | 115,764 | 64,204.3 | 1.80 | short |
| ask3necgb | CUTE DSL | Codex | 117,112 | 69,228.8 | **1.69** | short |
| pcevwrpue | CUTE DSL | Codex | 115,824 | 75,864.2 | 1.53 | short |
| 5hrpmgplj | CUDA C++ | Codex | 116,350 | 78,388.1 | 1.48 | short |
| ptv6z9eog | CUDA C++ | Codex | 117,450 | 83,901.9 | 1.40 | short |
| 96364acw1 | CUTE DSL | Codex | 115,659 | 91,134.4 | 1.27 | short |
| ohzemi9pv | CUTE DSL | Codex | 117,381 | 94,558.8 | 1.24 | short |
| p9y2rr1gw | Triton | Codex | 115,508 | 94,915.3 | 1.22 | short |
| yxiwb80jg | CUDA C++ | Codex | 116,406 | 102,473.9 | 1.14 | short |
| q651pl8jg | Triton | Codex | 117,001 | 108,226.9 | 1.08 | short |
| t5sr96or9 | CUDA C++ | Codex | *(blank)* | — | n/a | short |

`t5sr96or9` is excluded: its worker logs carry a blank/bogus baseline metric (flagged by the cost report as a timeout-inflated baseline), so no honest ratio can be computed. Two further CUDA C++ runs (`rlhgzw2ql`, `zgqpd77zz`) are export-only — JSONL logs but no optimize branches — giving the 24-run / 22-optimize-tag split.

The pattern is structural: **the top 10 are all Opus, the bottom 12 all Codex.** The language did not set the ceiling — the model did.

### Per language

| Language | Runs | Model(s) | **Best ×** | Mean × (of run-bests) | Token eff (Mtok/×) | $/× | Trial pass rate | Ceiling reason |
|---|---|---|---|---|---|---|---|---|
| **Triton** | 10 | 6 Opus + 4 Codex | **42.8×** | 18.3× | **157** | $147 | 86.6% (614/709) | model, not language — Opus hit the tensor-core ceiling |
| **CUDA C++** | 10 | 6 Opus + 4 Codex | 21.5× | 11.0× (7 valid) | 322 | $296 | 88.5% (554/626) | 2× the tokens/× of Triton; verbose kernels + arch/compile friction |
| **CUTE DSL** | 4 | 4 Codex | 1.69× | 1.4× | 429 | $303 | 81.8% (261/319) | DSL not installed; never reached tensor-core MMA; dominant n=512 punted to plain-SIMT CUDA C++ |

Token-efficiency and per-language totals are recomputed from the cost CSV and reconcile with the consolidated cost report (Triton 157 / CUDA C++ 322 / CUTE 429 Mtok/×). Pass rate = `succeeded / (succeeded + validation_error + build_error + runtime_error)` over each language's worker logs. The error-mode fingerprints differ:

| Language | build-error % | validation-error % | runtime-error % |
|---|---|---|---|
| Triton | **0.7%** (lowest) | 10.9% | 1.8% |
| CUDA C++ | **2.6%** (highest) | 7.8% | 1.1% |
| CUTE DSL | 3.1% | **12.2%** (highest) | 2.8% |

Triton almost never fails to build (its JIT compiles cleanly); CUDA C++ pays the most build failures (nvcc); CUTE DSL fails validation most often — immature kernels that compile but produce wrong factors, plus the install/JIT friction below.

### Removing the model confound

Comparing Opus-only isolates the language from the model:

| Language | Opus runs | Best × | Mean × | Codex runs | Best × | Mean × |
|---|---|---|---|---|---|---|
| Triton | 6 | **42.8×** | 28.9× | 4 | 5.03× | 2.28× |
| CUDA C++ | 4 | 21.5× | 18.3× | 3 valid | 1.48× | 1.34× |
| CUTE DSL | 0 | — | — | 4 | 1.69× | 1.43× |

Even with the model fixed to Opus and near-identical spend (~$954/run Triton vs ~$953 CUDA C++), Triton bought more speedup per token. The verdict survives the confound for Triton-vs-CUDA C++; CUTE simply has no same-model datapoint. Overall: Opus was most productive in Triton (42.8× peak, 28.9× mean) and strong in CUDA C++ (21.5× / 18.3×); it has no CUTE run (all CUTE runs predate the Opus batch). Codex capped low on *every* language — best 5.03× (Triton), 1.69× (CUTE), 1.48× (CUDA C++) — and looked *relatively* better in CUTE and Triton only because its CUDA C++ runs mostly never wired cuBLAS tensor cores. Codex is the whole reason CUTE looks bad in aggregate.

---

## CUDA C++ — the workhorse

The B200 CUDA C++ lineage's submissions — the top-scoring `qr_v2` one (~82×) and its earlier `qr_py` predecessor — are both pure CUDA C++, and in the controlled RTX Pro 6K evals CUDA C++ reached a strong 21.5× (`ocix8g7hp`, Opus) — but at ~2× Triton's token cost per unit of speedup, the verbosity of hand-writing kernels plus build/link plumbing showing up as tokens. The trade-off against the parallel B200 Triton lineage is the sharper one: ~82× on ~9,000 lines vs. ~64× on ~3,000.

**Tensor cores on the B200 — two live FP16→FP32 dialects.** The winning `qr_v2` submission hits tensor cores through:

1. **WMMA (`nvcuda::wmma`).** 84 fragment declarations, all `16,16,16` with `half` inputs and `float` accumulator (e.g. `fragment<matrix_a,16,16,16,half,col_major> a; … mma_sync(acc, a, b, acc);`), inside the resident megakernels. Instruction mix: 18 `load_matrix_sync`, 8 `mma_sync`, 7 `fill_fragment`, 11 `store_matrix_sync`.
2. **cuBLAS `cublasGemmStridedBatchedEx`** with `CUDA_R_16F` operands and `CUBLAS_COMPUTE_32F` — a 3-term FP16 Kahan hi/lo split (`Bh`/`Bl`, winner lines ~6764–6771) giving ~FP32-accurate trailing GEMM at ~2× the TF32 rate, plus `CUBLAS_TF32_TENSOR_OP_MATH` (`cublasSetMathMode`) for the diagonal solve and TRSM. `cp.async` (`cp.async.ca.shared.global`, `cp.async.commit_group`/`wait_group`) stages BF16 tiles into shared memory.

It builds for `arch=compute_100a,code=sm_100a` — the **`a`-suffix**.

**Frontier dialects, built but shelved.** 5th-generation `tcgen05` and FP8 `mma.sync` were written and validated in isolation, but gated off and never shipped:

- Commit `32e1c1d1f` integrated a real `tcgen05` substrate (41 PTX/marker lines: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`, `tcgen05.mma.cta_group::1.kind::f16`, `tcgen05.ld.sync.aligned.32x32b.x8.b32`, `tcgen05.commit`/`wait`/`dealloc`/`relinquish_alloc_permit`, `mbarrier`, `_smem_desc`), citing the canonical gau-nernst `tcgen05` example and PTX ISA 9.7.17. Commit `95aa08480` describes the design: `tcgen05.mma.cta_group::1.kind::f16` produces garbage at M=64/M=32 and is correct only at M=128, TMEM-accumulating to escape the 255-register / 12.5%-occupancy wall. It stayed dead code ("not wired into the live path → path byte-unchanged").
- Commit `50f5935c5` built a real 4-slice, error-compensated FP8 GEMM (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`). Verdict, verbatim: "even on real FP8 tensor cores fp8s2 regresses, because legacy mma.sync FP8=FP16 throughput on B200 (microbenchmarked 0.96x, NOT 2-4x) … The 2-4x FP8 peak needs tcgen05/MXFP8 (5th-gen) which requires sm_100a (harness builds sm_100, unbuildable remotely)."

This is what the `a`-suffix requirement means concretely: WMMA and `mma.sync` build fine under plain `sm_100`, but the 5th-gen `tcgen05`/TMEM/thread-block-cluster path requires `compute_100a,code=sm_100a` (`TORCH_CUDA_ARCH_LIST="10.0a"`). The `qr_v2` winner ships `sm_100a` for its WMMA/`cp.async` path; the `qr_py` winner, never attempting `tcgen05`, ships plain `compute_100,code=sm_100`.

**On the RTX Pro 6K, cuBLAS only.** Every CUDA C++ run there reached tensor cores exclusively through cuBLAS: `sgemm_sb()` dispatching `CUBLAS_COMPUTE_32F` / `..._FAST_TF32` / `..._32F_EMULATED_16BFX9` (BF16×9 ~FP32-accurate emulation) via `cublasGemmStridedBatchedEx` — no raw WMMA, `mma.sync`, or `tcgen05` (different GPU, sm_120). The one Codex run that wrote no cuBLAS at all (`ptv6z9eog`) stayed pure-SIMT and capped at 1.40×.

**Failure modes.**

- **The `a`-suffix / cubin-portability trap (the standout finding).** `ocix8g7hp` hit "no kernel image is available for execution on the device" **40 times**, then ran a controlled probe: case A (`code=sm_90` only, no PTX, on an sm_120 device — mimicking the sm_120-on-B200 bug) failed as predicted; case B (`code=compute_90` PTX only → JIT forward-compat to sm_120) ran OK. Root cause: hardcoding `code=sm_120` (the dev GPU) emits SASS with no PTX, so the B200/sm_100 leaderboard box has no runnable image. Fix (committed with a PORTABILITY comment): drop the hardcode and let torch inject default arch flags (running GPU + PTX for forward-compat JIT), so the module "compiles+runs on sm_100 (B200) and sm_120 alike."
- **nvcc compile volume.** Compile-error mentions scale with kernel complexity — `kpnikqu5d` ~2,018, `zgqpd77zz` ~1,097, `ocix8g7hp` ~987 — from redefinitions and name clashes across concatenated `load_inline` snippets, missing `cublas_v2.h`, and template/type errors. CUDA C++ carried the highest build-error rate (2.6%).

---

## Triton — the token-efficient challenger

Triton shows up twice, and both are primary results. On the **B200 leaderboard hardware**, a distinct B200 Triton-steered lineage (4 tags, best `2026-06-22-09-10-03-qr_v2`) reached **2,040.9 µs, ~64×** with ~3,000-line submissions — within striking distance of the B200 CUDA C++ lineage's ~82× at roughly a third of the code, and reinventing the same precision recipe independently. On the separate **RTX Pro 6K evals**, Triton posted that study's highest peak (`q1nb1oliq`, 2,744.5 µs, 42.8× vs. its own baseline) and the fewest tokens per unit of speedup (157 Mtok/×). The two are on different GPUs and baselines, so their × figures are not directly comparable — but together they are the report's clearest evidence that Triton buys most of the speedup at a fraction of the code.

The precision and launch details below are drawn from both the B200 Triton lineage and the RTX Pro 6K Triton runs, which converged on the same recipe.

**Precision dialects via `tl.dot`,** chosen per sub-step: `input_precision="ieee"` (bit-exact FP32) for the orthogonality-critical Gram and T-factor dots that must stay orthonormal (`S += tl.dot(tl.trans(Vblk), Vblk)`), and either `input_precision="tf32x3"` (3-pass TF32 ≈ FP32) or a hand-written bf16 hi/lo 3-term split (`_bgemm3`: `a.to(tl.bfloat16)` + residual, `Ah@Bh + Ah@Bl + Al@Bh`, FP32 accumulate, used for n≥2048) for the trailing-update GEMMs that carry the FLOPs. The Opus runs kept a documented `_TRAIL_IP` tunable for exactly this choice ("Tensor-core precision for the trailing-update GEMMs (\"ieee\" | \"tf32\" | \"tf32x3\")"); `kurboui6f` leaned on `tf32x3` (28 occurrences). `torch.backends.cuda.matmul.allow_tf32 = False` is set globally to keep the cuBLAS trailing GEMMs exact where needed.

**Manual launch config.** `num_warps` (`_PANEL_WARPS`/`_BUILDT_WARPS`) and `num_stages` (1 or 2) were set per kernel rather than paying the `@triton.autotune` first-launch sweep cost the environment warned about.

**The one wall: shared memory.** On sm_120 (~100 KB/SM), `q1nb1oliq` logged `W=128 fp32: FAILED out of resource: shared memory, Required: 131072, Hardware limit: 101376`; W=64 tiles fit (n=512 = 8,630 µs at fp16), forcing narrow, re-read panels. Otherwise Triton's runtime was very clean — only a handful of runtime/API-error hits across all 6 Opus runs, and a 0.7% build-error rate (lowest of the three) because the JIT rarely fails to compile.

---

## CUTE DSL — not yet viable here

CUTE DSL topped out at 1.69× (`ask3necgb`), for five compounding, evidenced reasons:

1. **The DSL wasn't installed.** A probe in `ask3necgb` printed `cutlass ERR ModuleNotFoundError("No module named 'cutlass'") … triton OK`; the model had to `pip install nvidia-cutlass-dsl` (71 MB of libs, plus `cuda-core`/`cuda-bindings`) before its first `import cutlass.cute` — a bootstrapping tax Triton and CUDA C++ never paid (Triton 3.6.0 and `torch.utils.cpp_extension` were already present).
2. **An sm_120a PTX gap.** CuTe's `redux_sync` lowers to `redux.f32`, which ptxas rejects on this GPU: `redux.f32' not supported on .target 'sm_120a'` (surfaced as an MLIRError through the pass pipeline). `pcevwrpue` worked around it by swapping in a `__shfl`-based warp reduction (commit `c8dccb524`, "use shuffle parallel cute qr32") — but only after burning trials on the failure. This is the CUTE counterpart to CUDA C++'s arch trap: the DSL emitted a PTX op the sm_120a target does not support.
3. **JIT instability under the validator.** The eval harness runs `custom_kernel` in a spawned subprocess pool, and CuTe's compile exceptions are not picklable across it (`Can't get local object '_site_initialize.<locals>.MLIRError'`), so one compile error corrupts the whole validation run rather than failing a single trial cleanly. `pcevwrpue` logged ~16 hard CUTE failures against ~42 "succeeded" rows (mostly re-tunes of ~5 working kernel families), and one run's own conclusion explicitly recommended to "switch away from CUTE DSL toward torch/CUDA extension approaches for QR."
4. **It never authored the shape that matters.** In no CUTE run did a `@cute.kernel` carry the geomean-dominating dense n=512, batch=640 workload — that stayed on `torch.geqrf` (± cosmetic CUTE wrappers, or plain CUDA) in all four. Real CUTE compute reached only small n (32/176/352), via serial per-column Householder kernels that blew the runtime budget at n≥512. A steering directive to workers even reads "Implement a true CUTE DSL n352 specialization … no marker shortcuts, no n512 combine work, and avoid n4096" — the search consciously left the dominant n512/n4096 shapes on non-CUTE paths.
5. **No tensor-core MMA.** A reference CUTLASS example on disk (`modal_nvfp4_dual_gemm/submission.py`, using `tcgen05`, `cute.TiledMma`, `cute.gemm`, `cute.make_copy_atom`) could not be adapted to QR; the CUTE kernels stayed scalar/warp-level and never reached the TF32/FP16 tensor-core trailing GEMM that got Triton and CUDA C++ to 17–43×.

**What actually produced each CUTE run's speedup (from a deep re-audit):**

| Run | Best × | Where the gain came from | Real CUTE compute? |
|---|---|---|---|
| `ask3necgb` | 1.69× | Hand-written `load_inline` CUDA (`update512_kernel`, `block_sum512`, n=512 tiled) + `torch.geqrf`; CUTE code present but partly cosmetic (no-op `norm_hint*0.0` epilogues) | n=32, n=352 only |
| `pcevwrpue` | 1.53× | **No custom CUDA at all** — genuine CUTE Householder (n=32 QR32 after the shuffle workaround, n=176, n=352 multi-launch panel) + `torch.geqrf` tau-pad/tail tricks for n≥512 | **most real CUTE of the four** |
| `96364acw1` | 1.27× | CUTE never imported/installed; pure `load_inline` CUDA (n=32, n=176) + `torch.geqrf` | none |
| `ohzemi9pv` | 1.24× | Only real CUTE was the n=32 QR32 warp kernel; larger n used CUTE only as detector/probe/output-assembly kernels routing to `torch.geqrf` | n=32 only |

So the run with the *most* genuine CUTE compute (`pcevwrpue`, 1.53×) is **not** the fastest — the winner `ask3necgb` (1.69×) got there mostly on hand-written CUDA C++ with CUTE as a thin, partly cosmetic wrapper. That inversion is the single strongest sign that CUTE was not the productive lever here.

Whether CUTE's ceiling is *fundamentally* ~1.7× or partly a Codex + short-budget artifact is **undetermined** — all 4 CUTE runs were Codex on the short 06-24 budget, with no Opus or long-budget CUTE run to disambiguate. The honest read: partly the Codex confound and substantially real, since three walls bite independently of model — the `nvidia-cutlass-dsl` install gap, the sm_120a `redux.f32` ptxas rejection, and the CuTe/MLIR JIT errors that are unpicklable across the eval subprocess pool.

---

## The library path (cuSOLVER / cuBLAS / torch)

- **cuSOLVER `geqrf` = the baseline** on both GPUs (`torch.geqrf`, serial per-matrix, ~73K launches, SIMT-only). Every winning submission replaces it and keeps it only as the gold-standard fallback for off-grid n, ill-conditioned, small-batch, or secret-test inputs.
- **cuBLAS strided-batched** (`cublasGemmStridedBatchedEx`, `cublasSgemmStridedBatched`, batched TRSM) is the primary library GEMM inside the custom CUDA kernels — the trailing-update FLOP sink at large n, and the device-filling path for the tiny-batch n=4096 CholeskyQR.
- **Custom WMMA / `tl.dot` megakernels** handle the launch-bound small/mid-n regime where even batched cuBLAS is launch-dominated.
- `torch.bmm` appears only in early prototypes.

---

## Scope & Methodology

*A thematic slice over the same corpus as the five consolidated qr reports (41 tags / 3,089 experiments): which GPU-kernel authoring languages the runs used — Triton, CUDA C++ (via torch `load_inline`; incl. WMMA / `mma.sync` / `tcgen05` PTX / TMA), the CUTLASS Python **CUTE DSL**, and the torch/library path (cuSOLVER / cuBLAS / `torch.geqrf` / `bmm`) — and how far each got, per model and per GPU.*

- **Scope — three lineages, 41 tags / 3,089 experiments:** (1) the B200 CUDA C++ lineage on B200 — 14 `qr_v2` tags plus 1 earlier `qr_py` tag in the same technique family; (2) the B200 Triton-steered lineage on B200 — 4 `qr_v2` tags (`2026-06-22-09-10-03`, `2026-06-24-21-01-44-simplify`, `2026-06-25-02-02-25`, `2026-06-25-23-41-18`); (3) the kernel-language experiment on RTX Pro 6K — 24 runs. (The two B200 lineages share the `qr_v2` problem, so branch counts below are pooled across all 18 B200 `qr_v2` tags.) Two kernel-language runs (`rlhgzw2ql`, `zgqpd77zz`) are export-only (JSONLs, no optimize branches), giving the 24-run / 22-optimize-tag split.
- **Language inventory verified against submission code**, not keyword counts: `git show <commit>:problems/linalg/qr_v2/submission.py` for each winner, plus a full-branch scan (1,866 B200 `qr_v2` + 434 `qr_py` branches, threaded `git show`) classifying real `import triton` (top-of-file) vs `load_inline`+`__global__` vs `import cutlass.cute`. This corrects the earlier keyword scan's noisy "cute" count (the top submission's single "cutlass" hit is a comment; the CUTE runs' matches are genuine `cutlass.cute` imports).
- **Speedups recomputed from raw logs:** min `status=succeeded` µs ÷ that run's `status=baseline` µs, per `autocuda/<tag>-optimize-tree-worker-*-log.csv`. Verified to reconcile with the consolidated cost report (per-language: Triton 42.8× @157 Mtok/×, CUDA C++ 21.5× @322, CUTE 1.69× @429) and optimization report (Triton best, CUDA C++ close, CUTE worst).
- **Model attribution** from `message.model` in every kernel-language run's JSONLs: 12 runs on `claude-opus-4-8` (their tags carry a `_claude_` marker) + 12 on `openai/openai/gpt-5.5` (Codex, tags without that marker); all 4 CUTE DSL runs are Codex.
- **Failure modes** mined from the harness JSONLs under `/tmp/qrwork/harness/<tag>/autocuda/harness-records/` (Claude-schema `message.content` and Codex-schema `payload`), plus git commit messages for the B200 `tcgen05`/FP8 dialect story (`32e1c1d1f`, `95aa08480`, `50f5935c5`).
- **GPU segmentation observed throughout:** B200 (sm_100a, ~131K µs `qr_v2` baseline / 44.7K µs `qr_py`) and RTX Pro 6K (sm_120, ~116K µs baseline) are never conflated — their speedups are against different baselines on different hardware.

**Undetermined / flagged:**

- **CUTE DSL's true ceiling is unknown, but substantially real.** No Opus CUTE run and no long-budget CUTE run exist, so its 1.69× cannot be *fully* separated into "Codex confound" vs "language limit." But three of its walls are model-independent and would bite any model: the `nvidia-cutlass-dsl` install gap, the sm_120a `redux.f32` ptxas rejection, and the CuTe/MLIR JIT errors unpicklable across the eval subprocess pool. A matched Opus/long-budget CUTE run is the missing experiment, but the evidence says the language — not just the model — capped this one.
- **`t5sr96or9`** (CUDA C++, Codex) has a blank baseline metric in its worker logs and is excluded from all ratios.
- The RTX Pro 6K kernel-language numbers are **local geomean** signals against a different (~116K µs) baseline; the leaderboard ranks on B200, so those × values are not directly comparable to *either* B200 lineage's end-to-end figures — the B200 CUDA C++ lineage's ~82× (`qr_v2`) / ~37× (`qr_py`) or the B200 Triton-steered lineage's ~64×.
