# GPUMODE QR — Resource Usage

## Executive Summary

The June QR work covers **43 tagged attempts**: 19 on B200 systems and 24 on RTX PRO 6000 systems. Available lock records measure **at least 217.47 GPU-hours**, plus **49.36 hours under the shared build lock**. The GPU total comprises about **45.2 benchmark GPU-hours**, **22.5 profile GPU-hours**, and **149.7 GPU-hours of validation, custom probes, submissions, and other exclusive work**. Three early B200 tags do not have enough lock evidence for an accurate compute breakdown, so the measured total is a lower bound.

Inference used **44.79B tokens**. Cache reads account for **95.2%** of that volume, while output accounts for **0.34%**. Tool activity is now counted across both harness formats: **185,184 Claude calls** and **36,097 Codex calls**, or **221,281 underlying calls** before compound shell operations are split into their executed build, benchmark, profile, and edit entries.

**Key takeaways**

- **Other exclusive GPU work is the largest measured category.** Validation, custom probes, and submissions used about 149.7 GPU-hours, versus 45.2 for benchmarking and 22.5 for profiling.
- **The 2×B200 runs are accounted per device.** Durations from `gpu-0` and `gpu-1` are added, with no second multiplier.
- **Build, benchmark, and profile operations are no longer hidden inside Bash.** Their counts appear as separate rows in the all-harness tool table.
- **Prompt context dominates inference volume.** Cache reads are 42.65B of the 44.79B tokens.
- **The July QR activity on this node is outside the report scope.**

## Token Usage

| Harness | Input | Cached input | Cache creation | Output | Total |
|---|---:|---:|---:|---:|---:|
| Claude | 20.15M | 38.55B | 1.87B | 143.94M | 40.58B |
| Codex | 101.24M | 4.10B | 0 | 7.48M | 4.21B |
| **Total** | **121.39M** | **42.65B** | **1.87B** | **151.42M** | **44.79B** |

| Token type | Tokens | Share |
|---|---:|---:|
| Input | 121,388,173 | 0.27% |
| Cache creation | 1,868,563,110 | 4.17% |
| Cached input | 42,646,429,160 | 95.22% |
| Output | 151,420,003 | 0.34% |
| **Total** | **44,787,800,446** | **100%** |

Cache hit rate is **95.54%**, calculated as cached input divided by input plus cache creation plus cached input. The dominant resource lever is therefore the amount of resident context re-read on each turn, not cache configuration.

## Walltime by Experiment Tag

There is one row per canonical experiment tag. **Activity span** is the first-to-last experiment-log timestamp and is not additive because experiments overlap. Build cells report shared build-lock calls and hours. Benchmark, profile, and other-exclusive cells report calls and per-device GPU-hours. On 2×B200 nodes, simultaneous `gpu-0` and `gpu-1` time contributes two GPU-seconds per wall-clock second.

| Experiment tag | Activity span | Hardware | Build calls / h | Benchmark calls / GPU-h | Profile calls / GPU-h | Other exclusive calls / GPU-h | Total GPU-h |
|---|---:|---|---:|---:|---:|---:|---:|
| `2026-06-13-03-45-35` | 48.30 h | 2× B200 | 1,787 / 10.79 | 416 / 0.99 | 1,192 / 1.73 | 2,411 / 4.14 | 6.86 |
| `2026-06-15-05-53-17-qr_v2` | 35.26 h | 2× B200 | —† | —† | —† | —† | —† |
| `2026-06-18-00-40-34-qr_v2` | ~0.78 h† | 2× B200 | —† | —† | —† | —† | —† |
| `2026-06-18-01-38-06-qr_v2` | 20.47 h | 2× B200 | —† | —† | —† | —† | —† |
| `2026-06-19-00-34-18-qr_v2` | 2.61 h | 2× B200 | 79 / 0.50 | 21 / 0.34 | 60 / 0.19 | 89 / 1.19 | 1.73 |
| `2026-06-19-04-54-07-qr_v2` | 13.19 h | 2× B200 | 334 / 2.58 | 100 / 0.83 | 159 / 1.02 | 213 / 4.46 | 6.31 |
| `2026-06-19-19-06-14` | 3.71 h | 2× B200 | 134 / 0.84 | 33 / 0.53 | 82 / 0.11 | 120 / 0.30 | 0.94 |
| `2026-06-20-04-33-52-qr_v2_simplify` | 16.46 h | 2× B200 | 517 / 3.06 | 195 / 2.07 | 157 / 0.26 | 427 / 8.93 | 11.25 |
| `2026-06-20-20-26-49-qr_v2` | 33.46 h | 1× B200 | 582 / 3.50 | 544 / 5.19 | 130 / 1.42 | 596 / 22.51 | 29.12 |
| `2026-06-20-22-03-57-qr_v2` | 65.17 h | 2× B200 | 1,372 / 11.58 | 246 / 2.22 | 766 / 3.36 | 1,717 / 22.43 | 28.01 |
| `2026-06-22-09-10-03-qr_v2` | 56.38 h | 1× B200 | 684 / 0.63 | 133 / 1.62 | 208 / 3.35 | 818 / 10.43 | 15.41 |
| `2026-06-23-17-54-06-qr_v2` | 22.22 h | 2× B200 | 239 / 1.83 | 74 / 0.86 | 96 / 0.44 | 409 / 6.31 | 7.61 |
| `2026-06-24-03-03-06-qrv2_kernel_languages_96364acw1` | 5.18 h | 1× RTX PRO 6000 | 80 / 0.35 | 64 / 1.01 | 16 / 0.12 | 73 / 3.05 | 4.17 |
| `2026-06-24-03-03-08-qrv2_kernel_languages_q651pl8jg` | 5.19 h | 1× RTX PRO 6000 | 84 / 0.02 | 80 / 1.09 | 15 / 0.12 | 91 / 3.37 | 4.58 |
| `2026-06-24-03-03-44-qrv2_kernel_languages_vcp095s3n` | 5.18 h | 1× RTX PRO 6000 | 69 / 0.02 | 51 / 0.72 | 22 / 0.47 | 75 / 3.35 | 4.53 |
| `2026-06-24-03-04-02-qrv2_kernel_languages_yxiwb80jg` | 5.19 h | 1× RTX PRO 6000 | 76 / 0.36 | 56 / 1.00 | 19 / 0.08 | 69 / 2.84 | 3.93 |
| `2026-06-24-03-43-38-qrv2_kernel_languages_ohzemi9pv` | 4.52 h | 1× RTX PRO 6000 | 82 / 0.03 | 65 / 0.87 | 35 / 0.53 | 82 / 3.16 | 4.56 |
| `2026-06-24-03-46-49-qrv2_kernel_languages_t5sr96or9` | 4.51 h | 1× RTX PRO 6000 | 112 / 0.00 | 100 / 2.30 | 27 / 0.88 | 119 / 0.97 | 4.15 |
| `2026-06-24-09-37-38-qrv2_kernel_languages_5hrpmgplj` | 7.99 h | 1× RTX PRO 6000 | 122 / 0.63 | 86 / 2.14 | 40 / 0.97 | 107 / 4.44 | 7.54 |
| `2026-06-24-09-37-40-qrv2_kernel_languages_p9y2rr1gw` | 8.13 h | 1× RTX PRO 6000 | 140 / 0.04 | 118 / 1.61 | 56 / 0.89 | 149 / 5.49 | 7.99 |
| `2026-06-24-09-37-42-qrv2_kernel_languages_pcevwrpue` | 7.98 h | 1× RTX PRO 6000 | 75 / 0.03 | 55 / 1.69 | 32 / 0.98 | 84 / 5.13 | 7.81 |
| `2026-06-24-09-38-11-qrv2_kernel_languages_ask3necgb` | 8.08 h | 1× RTX PRO 6000 | 110 / 0.44 | 81 / 2.87 | 27 / 0.31 | 93 / 3.74 | 6.92 |
| `2026-06-24-09-38-12-qrv2_kernel_languages_ptv6z9eog` | 8.04 h | 1× RTX PRO 6000 | 145 / 0.82 | 109 / 1.72 | 28 / 0.13 | 134 / 5.08 | 6.94 |
| `2026-06-24-09-38-31-qrv2_kernel_languages_m7w1ravpo` | 8.12 h | 1× RTX PRO 6000 | 95 / 0.03 | 69 / 1.18 | 36 / 1.09 | 95 / 5.27 | 7.53 |
| `2026-06-24-20-40-28-qr_v2` | 10.44 h | 2× B200 | 102 / 1.53 | 66 / 1.10 | 35 / 1.36 | 53 / 1.83 | 4.29 |
| `2026-06-24-21-01-44-qr_v2_simplify` | 3.87 h | 2× B200 | 25 / 0.01 | 20 / 3.14 | 21 / 0.15 | 21 / 0.77 | 4.06 |
| `2026-06-25-00-41-50-qrv2_kernel_languages_claude_kurboui6f` | 7.83 h | 1× RTX PRO 6000 | 80 / 0.02 | 5 / 0.31 | 1 / 0.01 | 107 / 0.82 | 1.14 |
| `2026-06-25-00-44-08-qrv2_kernel_languages_claude_kj7cvsimx` | 8.06 h | 1× RTX PRO 6000 | 85 / 0.02 | 14 / 0.31 | 21 / 0.12 | 74 / 1.36 | 1.78 |
| `2026-06-25-00-51-32-qrv2_kernel_languages_claude_q1nb1oliq` | 7.72 h | 1× RTX PRO 6000 | 95 / 0.03 | 29 / 0.68 | 2 / 0.00 | 119 / 1.25 | 1.93 |
| `2026-06-25-00-59-02-qrv2_kernel_languages_claude_ocix8g7hp` | 7.67 h | 1× RTX PRO 6000 | 121 / 1.50 | 23 / 0.50 | 5 / 0.01 | 75 / 1.22 | 1.73 |
| `2026-06-25-01-02-17-qrv2_kernel_languages_claude_rlhgzw2ql` | 7.77 h‡ | 1× RTX PRO 6000 | 104 / 0.70 | 10 / 0.15 | 8 / 0.11 | 162 / 0.73 | 0.99 |
| `2026-06-25-01-25-17-qrv2_kernel_languages_claude_3otwejh6b` | 7.12 h | 1× RTX PRO 6000 | 106 / 0.54 | 22 / 0.39 | 32 / 0.05 | 133 / 0.86 | 1.30 |
| `2026-06-25-02-02-25-qr_v2` | 7.23 h | 2× B200 | 117 / 0.04 | 7 / 0.61 | 44 / 0.13 | 228 / 1.91 | 2.65 |
| `2026-06-25-08-42-00-qr_v2_simplify` | 10.71 h | 2× B200 | 151 / 1.09 | 13 / 0.16 | 37 / 0.70 | 91 / 1.05 | 1.90 |
| `2026-06-25-11-58-38-qrv2_kernel_languages_claude_tszacp309` | 7.85 h | 1× RTX PRO 6000 | 81 / 0.10 | 27 / 0.47 | 7 / 0.02 | 106 / 1.33 | 1.81 |
| `2026-06-25-12-01-35-qrv2_kernel_languages_claude_w7jxbr46z` | 7.83 h | 1× RTX PRO 6000 | 106 / 0.62 | 15 / 0.21 | 6 / 0.02 | 160 / 1.00 | 1.22 |
| `2026-06-25-12-03-48-qrv2_kernel_languages_claude_zgqpd77zz` | 7.76 h‡ | 1× RTX PRO 6000 | 128 / 0.72 | 39 / 1.08 | 14 / 0.07 | 145 / 1.69 | 2.84 |
| `2026-06-25-12-08-24-qrv2_kernel_languages_claude_g7v577r9x` | 7.81 h | 1× RTX PRO 6000 | 54 / 0.02 | 18 / 0.38 | 19 / 0.09 | 177 / 0.86 | 1.34 |
| `2026-06-25-12-21-21-qrv2_kernel_languages_claude_kpnikqu5d` | 7.57 h | 1× RTX PRO 6000 | 71 / 0.36 | 10 / 0.64 | 14 / 0.32 | 72 / 0.70 | 1.65 |
| `2026-06-25-12-48-50-qrv2_kernel_languages_claude_j057n57jl` | 7.14 h | 1× RTX PRO 6000 | 50 / 0.01 | 20 / 0.66 | 17 / 0.12 | 162 / 1.77 | 2.54 |
| `2026-06-25-22-03-01-qr_v2` | 23.45 h | 1× B200 | 190 / 2.11 | 9 / 0.12 | 20 / 0.40 | 54 / 1.20 | 1.73 |
| `2026-06-25-23-41-18-qr_v2` | 16.14 h | 2× B200 | 153 / 0.05 | 15 / 1.07 | 18 / 0.29 | 203 / 1.75 | 3.10 |
| `2026-06-29-06-40-42-qr_v2` | 8.68 h | 2× B200 | 249 / 1.81 | 24 / 0.41 | 18 / 0.11 | 57 / 1.05 | 1.57 |
| **Measured total (40 tags)** | — | mixed | **8,986 / 49.36** | **3,082 / 45.22** | **3,542 / 22.52** | **10,170 / 149.75** | **217.47** |

† The activity span is available, but the harness corpus does not contain enough lock records for an accurate build or GPU breakdown. The `2026-06-18-00-40-34-qr_v2` span runs from tag creation to its last recorded commit.

‡ These two RTX tags have session and lock records but no optimization log; their activity spans come from the harness timestamps.

## Tool Calls Across Claude and Codex

Claude `tool_use` and Codex function-call records are normalized into the same categories. Build, benchmark, and profile operations are removed from generic shell activity and counted below as their own entries.

| Tool or operation | Claude | Codex | Total |
|---|---:|---:|---:|
| Bash / shell, other | 96,292 | 26,181 | 122,473 |
| Read | 34,931 | 0 | 34,931 |
| Edit | 19,866 | 1,446 | 21,312 |
| **Build** | **9,188** | **1,797** | **10,985** |
| **Benchmark** | **7,928** | **1,565** | **9,493** |
| **Profile** | **4,604** | **723** | **5,327** |
| Task update | 508 | 3,429 | 3,937 |
| Write | 2,627 | 0 | 2,627 |
| Web fetch | 2,362 | 0 | 2,362 |
| Agent | 1,762 | 449 | 2,211 |
| Send message | 1,553 | 326 | 1,879 |
| Monitor | 1,147 | 711 | 1,858 |
| Web search | 1,538 | 9 | 1,547 |
| Task stop | 761 | 211 | 972 |
| Skill | 819 | 0 | 819 |
| Remaining named tools | 926 | 1 | 927 |
| **Normalized entries** | **186,812** | **36,848** | **223,660** |

The remaining named tools are task output/create/list/get, structured output, cron create/delete/list, push notification, schedule wakeup, workflow, and leaderboard ranking.

The normalized total is 2,379 above the 221,281 underlying harness calls because a compound shell call can execute more than one build, benchmark, or profile operation, and a resource command can also contain an edit. Each executed operation is represented in its corresponding row; it is not retained in the generic shell row.

The tool table counts issued operations, while the walltime table counts completed lock releases that carry measured durations. Their build, benchmark, and profile call counts therefore answer different questions and are not expected to match.

## Resource Opportunities

- **Reduce repeated validation and exploratory GPU probes first.** Other-exclusive work is about 69% of measured GPU time, substantially more than benchmarking or profiling.
- **Tighten benchmark budgets where confidence permits.** Benchmarking still accounts for about 45.2 GPU-hours across 3,082 measured calls.
- **Use profiling selectively.** Profiling accounts for about 22.5 GPU-hours and 3,542 measured calls; profiles that do not change the next experiment can be avoided.
- **Reduce resident context size.** With 42.65B cached-input tokens, smaller summaries or diff-focused context would target the dominant inference category directly.
- **Preserve build artifacts when safe.** Shared build-lock activity totals 49.36 hours across 8,986 measured build calls.

## Scope and Methodology

**Scope.** The report covers 43 June tags: 19 B200 attempts and 24 RTX PRO 6000 attempts. Of the RTX attempts, 22 have optimization logs and two have session-only evidence. The recent July QR activity on this node is excluded.

**GPU topology.** Sixteen B200 tags ran on 2×B200 nodes, three ran on 1×B200 nodes, and all RTX PRO 6000 tags used one GPU. GPU-hours are the sum of unique per-device exclusive lock durations. A 2× node therefore contributes up to two GPU-hours per wall-clock hour when both devices are active; the lock total is not multiplied again.

**Walltime categories.** Build time comes from shared build-lock releases. Benchmark and profile time come from exclusive lock releases classified using the originating command and result markers. Other exclusive time includes validation, custom GPU probes, submissions, and GPU work that is neither a benchmark nor a profile. Three early B200 tags have incomplete lock evidence and are excluded from measured compute totals rather than estimated.

**Tool calls.** Claude and Codex records are parsed in their native schemas and matched by stable call ID. `autocuda run slice/shared` operations are classified as builds; benchmark scripts and `eval.py benchmark` as benchmarks; profiler scripts, `nsys profile`, and capture-mode `ncu` as profiles. Remaining shell calls stay in the generic shell category.

**Tokens.** Claude usage uses the final values for each response ID. Codex usage uses the final per-session token snapshot, with cached input separated from the inclusive input field. The four token fields in this report are disjoint.
