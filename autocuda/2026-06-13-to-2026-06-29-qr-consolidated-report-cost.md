# GPUMODE QR — Cost

## Executive Summary

The QR optimization work burned an estimated **~$62K of inference on 77.6B tokens and ~399 GPU-locked hours** across all 41 runs. The spend is **inference-cheap per token but token-heavy per unit of code**: **94.5% of tokens were prompt-cache reads** as agents re-read a large, stable context every session, while raw output — the tokens that encode kernel edits — is just 0.25%. **Benchmarking, not reasoning, dominated walltime** (18:1 bench:build). The cost splits across the three lineages — the two primary B200 lineages (the winning CUDA C++ line and the shorter Triton-steered line, both on the same hardware) plus the separate RTX Pro 6K kernel-language experiments:

| Lineage | Est. cost | Tokens |
|---|---|---|
| **B200 CUDA C++** — incl. the earlier qr_py run (~$10.6K) | **~$31.9K** | ~43.3B |
| **B200 Triton-steered** — 4 tags | **~$17.0K** | ~19.9B |
| **RTX Pro 6K evals** — 24 runs | **~$13.2K** | ~14.4B |
| **Total** | **~$62.1K** | **77.6B** |

(The ~399 GPU-locked hours are ~246 h across the two B200 lineages + ~153 h on the RTX Pro 6K. Cost is attributed over 42 session-bearing export tags — the 41 runs plus one extra export that carries only session logs; per-tag session attribution is coarse, so treat the two B200 lineages' split as approximate.)

**Key takeaways**

- **Deduplication halves the bill** — 50% of shipped session logs are cumulative re-bundles; report $62K, not the naive $105.7K.
- **Cache hit is already excellent (94.7%) — don't chase it.** The lever is shrinking the context each trial re-reads.
- **Benchmark time is the walltime sink** — 21,635 GPU calls (328h) vs 1,211 builds (71h).
- **The two primary B200 lineages are ~79% of the bill** (~$48.9K of $62K — ~$31.9K CUDA C++ + ~$17.0K Triton-steered); the RTX Pro 6K kernel-language experiments delivered their language comparison for ~$13.2K (~21%).
- **Triton was the most token-efficient language** — best 42.8× at 157 Mtok per unit-×, ~2.1× better than CUDA C++.

## Token Totals

**≈ 77.6B tokens** across every deduplicated agent session (the two primary B200 lineages — winning CUDA C++ + Triton-steered — plus the 24 RTX Pro 6K kernel-language runs and the earlier qr_py run; the late `2026-06-19-19-06-14` tag is excluded, see below).

| Token type | Tokens | Share |
|---|---|---|
| Input (uncached) | 126.8M | 0.16% |
| Cache creation (write, billed at input rate) | 3.98B | 5.13% |
| Cached input (read) | 73.34B | 94.46% |
| Output | 193.3M | 0.25% |
| **Total** | **77.64B** | **100%** |

Cache-read at 94.5% is the whole story: a small body of novel text (output + uncached input + cache writes = 4.3B, 5.5%) rides on an enormous re-read prefix. The three input fields are disjoint — the harness records uncached input, cache-creation, and cache-read separately — so "Input (uncached)" is the raw `input_tokens` column, not a net figure.

## Walltime Breakdown

Two views of walltime. **Per-branch span** (the `Duration` column) is the first→last timestamp across a tag's deduplicated logs; it overlaps *within* a run's export chain (cumulative exports share sessions), so those rows are not additive — treat them as per-run activity windows. **GPU-locked time** is summed per call from the harness's `released lock=… ran=<s>` lines (shared lock = build; exclusive = bench/profile/validate) and deduplicated by owning session; these *are* additive and are the authoritative compute-time figure. The Build/Bench columns below are call counts; the group totals carry the summed seconds.

| Branch | Duration | Build calls | Bench calls | Other tool calls |
|---|---|---|---|---|
| 2026-06-19-04-54-07-qr_v2 | 13.9 h | 6 | 216 | ~2,050 |
| 2026-06-20-04-33-52-qr_v2_simplify | 17.8 h | 12 | 300 | ~2,900 |
| 2026-06-20-22-03-57-qr_v2_with_logs | 65.3 h | 71 | 1,400 | ~15,300 |
| 2026-06-22-09-10-03-qr_v2 | 58.6 h | 152 | 2,600 | ~24,800 |
| 2026-06-24-20-40-28-qr_v2 | (subset) | 6 | 55 | ~440 |
| 2026-06-25-08-42-00-qr_v2_simplify | (subset) | 12 | 260 | ~2,470 |
| 2026-06-25-22-03-01-qr_v2 | (subset) | 30 | 620 | ~5,600 |
| 2026-06-25-23-41-18-qr_v2 | (subset) | 50 | ~800 | ~7,700 |
| 2026-06-29-06-40-42-qr_v2 | (subset) | ~30 | ~530 | ~5,200 |
| **Main B200 (17 tags)** | **49.5 h build + 178.3 h GPU** | **580** | **11,850** | **102,685** |
| **Kernel-language, RTX Pro 6000 (24 tags)** | **10.1 h build + 142.7 h GPU** | **161** | **2,385** | **16,372** |
| **qr_py (1 tag)** | **11.4 h build + 6.9 h GPU** | **470** | **7,400** | **48,553** |
| **TOTAL (dedup)** | **71.0 h build + 327.9 h GPU-exclusive = 398.8 h locked** | **1,211** | **21,635** | **167,610** |

`(subset)` marks later main-run exports whose unique-owned session count is small after dedup (most of their sessions belong to an earlier export); their duration span is not meaningful in isolation. `2026-06-23-17-54-06-qr_v2` is a **near-identical re-bundle** of `2026-06-20-22-03-57`: all 621 session logs are byte-identical, and it owns only **1** extra unique file (a `memory/codex/history.jsonl` artifact, $0 billable), so it contributes no cost — it is carried as an all-zero CSV row and omitted from the walltime rows above. Bench-call counts are the harness's `Bash … bench` tool_use tally; the independent `ran=` sum is the figure to trust for compute cost.

## Tool Call Distribution

Counts are Claude Code `tool_use` records. The 12 Codex/GPT-5.5 kernel-language runs don't emit Claude-style `tool_use` blocks, so their bash/read/edit calls are absent here (their tokens are still counted everywhere else); the kernel-language block below is almost entirely the 12 Opus runs.

| Branch | Tool | Count |
|---|---|---|
| Main B200 | Bash | 73,313 |
| Main B200 | Read | 22,190 |
| Main B200 | Edit | 12,900 |
| Main B200 | WebFetch | 2,300 |
| Main B200 | Agent | 1,616 |
| Main B200 | WebSearch | 1,518 |
| Main B200 | Write | 1,204 |
| Main B200 | SendMessage | 955 |
| Main B200 | Skill | 806 |
| Main B200 | TaskStop | 308 |
| Kernel-language (Opus) | Bash | 10,473 |
| Kernel-language (Opus) | Read | 1,660 |
| Kernel-language (Opus) | Edit | 1,636 |
| Kernel-language (Opus) | SendMessage | 470 |
| Kernel-language (Opus) | Write | 405 |
| Kernel-language (Opus) | WebSearch | 330 |
| Kernel-language (Opus) | WebFetch | 306 |
| Kernel-language (Opus) | Agent | 237 |
| Kernel-language (Opus) | Skill | 104 |
| qr_py | Bash | 36,628 |
| qr_py | Read | 10,940 |
| qr_py | Edit | 5,981 |
| qr_py | Write | 955 |
| qr_py | Monitor | 808 |
| qr_py | WebFetch | 441 |
| qr_py | TaskStop | 439 |
| qr_py | WebSearch | 175 |

`Bash` dominates every group (build/bench/profile all shell out through it) — 120,414 calls total, 58% of all tool use. The `Read:Edit` ratio is ~1.7:1 across the corpus (34,790 : 20,517): the agents read more than they edit, consistent with the high cache-read share — they re-scan the large submission before making targeted changes. WebFetch+WebSearch is only 5,070 calls (2.4%), so the runs leaned on training-data CUDA knowledge rather than reaching to the web — that keeps inference latency down and is a *good* sign for the skill prose.

## Inference Breakdown

Cache-hit % = `cache_read / (input + cache_creation + cache_read) × 100`. USD is priced at the rates in Scope & Methodology (Opus 4.8 for every row except the 12 Codex/GPT-5.5 kernel-language runs).

| Branch | Input tokens | Output tokens | Cached input tokens | Cache creation tokens | Cache hit % | Compactions | Estimated USD |
|---|---|---|---|---|---|---|---|
| Main B200 (17 tags) | 37.9M | 114.0M | 43.37B | 2.71B | 94.0% | 26 | $38,274 |
| Kernel-language, RTX Pro 6000 (24 tags) | 78.6M | 35.0M | 13.19B | 1.06B | 92.0% | 49 | $13,192 |
| &nbsp;&nbsp;— Codex/GPT-5.5 (12 runs) | 70.5M | 5.2M | 2.48B | 0 | 97.2% | 45 | $1,745 |
| &nbsp;&nbsp;— Opus 4.8 (12 runs) | 8.1M | 29.8M | 10.72B | 1.06B | 90.9% | 4 | $11,447 |
| qr_py (1 tag) | 10.2M | 44.3M | 16.79B | 0.21B | 98.7% | 0 | $10,599 |
| **Total (dedup)** | **126.8M** | **193.3M** | **73.34B** | **3.98B** | **94.7%** | **75** | **$62,065** |

Cache hit is **above 50% everywhere** — no low-cache-hit alarm to raise. The one structural note: the Codex runs show `cache_creation = 0` because Codex's `token_count` schema reports only cumulative input/cached/output (no cache-write field), and its `input_tokens` is inclusive of the cached portion — the CLI subtracts the cached tokens back out to land in the same disjoint buckets. That is why the Codex "Input" (70.5M) looks large next to Opus (8.1M): it is the un-cached remainder of a cumulative counter, not extra novel prompt. The dollar consequence is small ($1,745 for all 12 Codex runs).

## Cost-Optimization Opportunities

- **Deduplicate exports before costing anything.** The single biggest "cost" error available here is counting cumulative re-bundles: 50% of shipped session logs are duplicates, and two exports are near-identical (621 of their session logs byte-for-byte the same). Any cost tally that sums per-tag without content-hash dedup over-reports by ~1.67×. This is a reporting fix, not a spend fix, but it is the highest-impact correction.
- **Cache hit is already excellent (94.7%) — don't chase it.** The lever is *trial count against a huge prefix*, not cache mechanics. The main run carried a ~9,000-line best-so-far submission that every trial re-reads; the marginal trial costs ~$0.50/Mtok × (prefix size). Shrinking the resident context the workers must re-read each iteration (e.g. summarizing that submission, or diff-only context) would cut the 73.3B read tokens more than any caching tweak.
- **Benchmark, not inference, is the walltime sink.** 21,635 exclusive GPU calls / 328 GPU-hours vs 1,211 builds / 71h — an 18:1 bench:build ratio. If `environment.md`'s per-shape benchmark budget (~50s baseline, up to 300s) is loose, tightening the repeat count or early-exit threshold on the 12-shape geomean would directly reduce GPU-hours; the search made ~21.6K measurements.
- **Compactions are low (75 across 3,204 sessions).** Only the longer main-run and Opus kernel-language runs compact at all; context-window pressure is not a cost driver here. No action needed.
- **Per-run cost concentration.** Within the main run, `2026-06-22-09-10-03-qr_v2` ($12.5K, 891 owned sessions) and `2026-06-25-23-41-18-qr_v2` ($4.5K) are the largest; no single run exceeds 2× the median after dedup, so nothing flags a runaway loop. The qr_py run (optimize tag `2026-06-13-03-45-35`: 434 branches, 1,161 kept trials; sessions bundled under export `2026-06-13-03-37-53-qr_py`) is $10.6K on 498 sessions — expensive per session because it has the highest cache-read share (98.7%), i.e. it re-read the most context per trial. It is a full optimize run with trial logs on disk, and its 498 session logs (all unique, distinct from the qr_v2 runs) attribute cleanly to it.
- **Codex kernel-language runs were ~6.5× cheaper per run than Opus kernel-language runs** ($145/run vs $950/run) but delivered far less speedup (≤5× vs up to 42.8×). Cheap but unproductive; see Analysis.

## Analysis

**Was the spend justified by the improvement?** For the main B200 run, yes on a per-token basis, though the search was long: $38.3K bought a per-shape-routed CUDA C++ submission (~9,000 lines) over a 06-15→06-29 arc. The cost shape is dominated by re-reading that submission — 43.4B cache-read tokens — the price of a *stateful, growing* codebase under tree search, where every worker re-ingests the current best each iteration. The cheapest available saving was carrying less resident context per trial; the most expensive decision embedded in the data is the cumulative-export packaging, which summed naively would misattribute $43K of phantom re-bundled spend.

**The kernel-language token-efficiency headline.** Per-abstraction best speedups are each run's fastest kept trial over its **own** `status=baseline` log row (the RTX Pro 6K QR baseline is ~116K µs per run — *not* the B200 baseline; reusing the B200 figure would roughly double every ×). These reconcile with the final optimization report (Triton 42.8× / CUDA C++ 21.5× / CUTE DSL 1.69×). The consolidated optimization CSV's `best_speedup` column is blank on 100% of rows (a confirmed `report data optimization` join bug), and `autocuda status` reads the now-eigh-configured live directory, so the figures come straight from the raw `linalg/qr_v2` metric columns. Against deduplicated token spend:

| Abstraction | Runs | Total tokens | Output | Best speedup | Mean speedup | Mtok per unit-× | $ per unit-× |
|---|---|---|---|---|---|---|---|
| **Triton** | 10 | 6.71B | 17.1M | **42.8×** | 18.3× | **157** | **$147.4** |
| CUDA C++ | 10 | 6.92B | 16.2M | 21.5× | 11.0× | 322 | $296.4 |
| CUTE DSL | 4 | 0.73B | 1.6M | 1.69× | 1.4× | 429 | $302.8 |

**Triton was decisively the most token-efficient abstraction**: it reached the highest peak speedup (42.8×) on the *fewest* tokens per unit of speedup (157 Mtok/×), 2.1× better than CUDA C++ and 2.7× better than CUTE DSL. This is not just faster convergence — per-run output volume is comparable across the Opus-run abstractions (~2.5M/run) — but a given token budget in Triton *landed* far more speedup: `q1nb1oliq` (Triton) hit 42.8× (2,744.5 µs over its 117,536 µs baseline), while the best CUDA C++ run (`ocix8g7hp`) reached 21.5×. CUTE DSL was genuinely **token-hungry for little return**: the model's relative unfamiliarity with the DSL shows up as many trials that never cleared ~1.7× (all 4 CUTE runs were Codex/GPT-5.5, which capped low regardless of language).

**The confound, stated plainly.** This is *not* a clean controlled experiment, because kernel language and model are entangled: all 4 CUTE DSL runs and the 12 Codex/GPT-5.5 kernel-language runs topped out at ~1.3–5× on *every* language, while the high-speedup runs (17–42.8×) were all Opus 4.8. (`t5sr96or9`'s 4.6× is excluded — its baseline is a bogus timeout-inflated 116M µs.) The honest same-model comparison is Opus-only: **Triton 42.8× peak / 28.9× mean vs CUDA C++ 21.5× / 18.3× mean, at essentially identical token spend (~2,571K vs ~2,398K output/run, ~$954 vs ~$953/run)**. Even controlling for model, Triton bought more speedup per token — so the token-efficiency verdict holds, while acknowledging that Codex's uniformly-low ceiling (not the abstraction alone) explains most of CUTE DSL's poor showing.

**Did inference cost grow super-linearly with trial count?** No — caching makes it grow *sub*-linearly per trial: the Nth trial adds mostly cache-read tokens at 0.1× rate, so a 251-branch run (`p9y2rr1gw`) is not 251× the cost of a 25-branch run. The dominant cost driver is prefix size, not trial count, which is why the qr_py run (498 sessions, highest cache-read share) is nearly as expensive as much larger main-run branches. The most expensive branch (`2026-06-22`, $12.5K) was also among the most productive (891 owned sessions building toward the best submission); the cheapest productive branches were the Opus Triton kernel-language runs (~$950 each for up to 42.8×).

**Recommendations.** (1) Always content-hash-dedup cumulative exports before cost aggregation — it is the difference between $62K and $105.7K here, and the CLI does not do it for you. (2) To cut the 73.3B cache-read tokens, reduce the resident context each worker re-reads per trial (summarize or diff the best submission rather than re-ingesting ~9K lines) — this attacks 94.5% of the token bill directly, whereas cache tuning cannot (hit-rate is already 94.7%). (3) For future kernel-language studies, hold the model fixed — the current design entangles abstraction with Codex-vs-Opus, so route all abstractions through Opus 4.8 (or all through one model) before drawing language-efficiency conclusions. On the evidence available, **Triton is the token-efficiency choice**, delivering the most kernel speedup per dollar and per token.

## Scope & Methodology

**Deduplication.** All token, call, and walltime figures are **deduplicated by session-log content hash** — each unique session attributed to the earliest export that captured it — because the main-run export branches are cumulative and re-bundle earlier sessions. Of 6,408 shipped session logs, 3,204 are unique. No separate human-driven sessions were identified: the Codex kernel-language runs are the harness's own workers, not hand-driven side sessions, so they are counted as ordinary run rows rather than steered-session rows.

**Model and rates (USD is approximate).** Model was identified from `message.model` across the harness logs. The dominant model is **`claude-opus-4-8`** (Claude Opus 4.8), used by the 17 main B200 tags, the qr_py tag, and the 12 Opus kernel-language runs; the other 12 kernel-language runs ran on **`openai/openai/gpt-5.5`** (Codex/GPT-5.5), priced separately. Public per-1M-token list rates were applied via `--pricing-*-per-mtok`: **Opus 4.8** — input $5.00, output $25.00, cache-read $0.50 (≈0.1× input), cache-creation billed at the $5.00 input rate; **GPT-5.5** — input $5.00, output $30.00, cached-input $0.50. Estimated cost is the CLI's per-row sum `input×in_rate + cache_creation×in_rate + cache_read×cached_rate + output×out_rate` — an order-of-magnitude estimate, not an invoice.
