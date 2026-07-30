# Cost Report: linalg/cholesky_py (2026-07-19-01-44-06-cholesky-resumed-local-unmerged)

## Executive Summary

As of the 2026-07-20 21:07:18 UTC CSV snapshot, this ongoing single experiment is expensive in absolute terms: 258.1 overlapping agent-hours, 6.9B accounted tokens, and an API-rate-equivalent estimate of $4.1K. Inference volume drives dollars—cached reads are 98.8% of tokens and $3.4K of spend—while recorded run wrappers occupy 148.0 hours of active time. Because only one experiment tag is in scope, branch rows describe within-run allocation rather than cross-run efficiency.

**Key takeaways**

- All 93 fleet briefs and four steered workstreams are covered by 258 unique session files; no known harness segment is missing.
- Briefs 78 and 81 cost $306 and $261, respectively, versus a $23 fleet median; 26 briefs exceeded twice that median.
- Cache reuse is strong at 98.82%; preserving it matters because cache reads still account for 83.0% of estimated dollars at this scale.
- The wrapper median of 84.0 s fits the 90 s validation/benchmark allowance, but its 294.6 s p95 and 48.4 min maximum do not; measured lock wait was zero.

**Snapshot total: 258.1 h · ~$4.1K · 81.7M uncached input / 6.9B cached input / 9.9M output · 63 compactions; the runtime tail is not in line with the environment timeout estimates.**

## Token Totals

**6.9B tokens**

| Token type | Tokens | Share |
|---|---:|---:|
| Input | 81.7M | 1.175% |
| Cached input | 6.9B | 98.683% |
| Output | 9.9M | 0.142% |
| **Total** | **6.9B** | **100.000%** |

Input is uncached input as reported by the Codex rollout counters; it is disjoint from cache reads. These totals include the four steered rows. Codex did not expose a separate cache-creation counter, so the zero cache-creation values below are a schema limitation, not evidence that cache population was free.

## Walltime Breakdown

| Branch | Duration | Build calls | Bench calls | Other tool calls |
|---|---:|---:|---:|---:|
| brief-0 | 1.11 h | 43 | 0 | 254 |
| brief-1 | 1.53 h | 94 | 6 | 345 |
| brief-2 | 1.09 h | 63 | 8 | 243 |
| brief-3 | 1.10 h | 80 | 6 | 279 |
| brief-4 | 2.26 h | 83 | 1 | 563 |
| brief-5 | 0.37 h | 15 | 0 | 105 |
| brief-6 | 0.41 h | 28 | 4 | 88 |
| brief-7 | 0.65 h | 35 | 0 | 166 |
| brief-8 | 3.87 h | 159 | 10 | 712 |
| brief-9 | 3.77 h | 129 | 6 | 801 |
| brief-10 | 4.18 h | 163 | 16 | 1,210 |
| brief-11 | 2.06 h | 83 | 26 | 567 |
| brief-12 | 0.80 h | 41 | 1 | 186 |
| brief-13 | 0.22 h | 15 | 4 | 65 |
| brief-14 | 0.92 h | 40 | 14 | 193 |
| brief-15 | 1.00 h | 34 | 12 | 194 |
| brief-16 | 1.47 h | 49 | 11 | 303 |
| brief-17 | 1.06 h | 39 | 12 | 215 |
| brief-18 | 1.00 h | 35 | 13 | 232 |
| brief-19 | 0.59 h | 27 | 6 | 111 |
| brief-20 | 0.36 h | 15 | 1 | 102 |
| brief-21 | 0.12 h | 6 | 3 | 66 |
| brief-22 | 2.83 h | 89 | 18 | 512 |
| brief-23 | 1.37 h | 46 | 6 | 240 |
| brief-24 | 2.94 h | 91 | 9 | 642 |
| brief-25 | 5.03 h | 144 | 4 | 1,652 |
| brief-26 | 0.86 h | 32 | 11 | 196 |
| brief-27 | 0.43 h | 18 | 7 | 156 |
| brief-28 | 1.37 h | 45 | 6 | 352 |
| brief-29 | 3.38 h | 81 | 2 | 803 |
| brief-30 | 0.78 h | 30 | 0 | 220 |
| brief-31 | 0.90 h | 21 | 8 | 240 |
| brief-32 | 1.23 h | 33 | 5 | 311 |
| brief-33 | 0.93 h | 29 | 3 | 228 |
| brief-34 | 0.64 h | 21 | 10 | 202 |
| brief-35 | 0.00 h | 3 | 1 | 36 |
| brief-36 | 1.56 h | 38 | 5 | 353 |
| brief-37 | 0.00 h | 3 | 2 | 38 |
| brief-38 | 0.89 h | 30 | 4 | 259 |
| brief-39 | 0.95 h | 30 | 5 | 233 |
| brief-40 | 0.48 h | 15 | 3 | 138 |
| brief-41 | 0.64 h | 24 | 0 | 194 |
| brief-42 | 0.56 h | 21 | 2 | 158 |
| brief-43 | 4.13 h | 111 | 6 | 897 |
| brief-44 | 1.05 h | 21 | 2 | 265 |
| brief-45 | 0.98 h | 23 | 3 | 218 |
| brief-46 | 0.52 h | 22 | 3 | 141 |
| brief-47 | 0.48 h | 17 | 4 | 124 |
| brief-48 | 0.44 h | 18 | 0 | 140 |
| brief-49 | 6.76 h | 105 | 15 | 989 |
| brief-50 | 3.29 h | 58 | 11 | 708 |
| brief-51 | 0.53 h | 18 | 3 | 167 |
| brief-52 | 1.61 h | 40 | 8 | 384 |
| brief-53 | 3.84 h | 65 | 12 | 596 |
| brief-54 | 0.96 h | 27 | 2 | 246 |
| brief-55 | 3.16 h | 76 | 14 | 615 |
| brief-56 | 1.95 h | 56 | 2 | 461 |
| brief-57 | 0.98 h | 29 | 1 | 270 |
| brief-58 | 9.79 h | 94 | 20 | 608 |
| brief-59 | 1.00 h | 25 | 3 | 209 |
| brief-60 | 1.33 h | 31 | 8 | 294 |
| brief-61 | 2.07 h | 52 | 3 | 427 |
| brief-62 | 1.72 h | 37 | 3 | 445 |
| brief-63 | 1.48 h | 22 | 2 | 385 |
| brief-64 | 2.04 h | 49 | 6 | 417 |
| brief-65 | 0.87 h | 18 | 3 | 210 |
| brief-66 | 0.60 h | 16 | 0 | 177 |
| brief-67 | 0.51 h | 16 | 2 | 152 |
| brief-68 | 1.45 h | 42 | 3 | 330 |
| brief-69 | 1.13 h | 32 | 4 | 293 |
| brief-70 | 0.00 h | 4 | 1 | 69 |
| brief-71 | 8.34 h | 114 | 7 | 1,291 |
| brief-72 | 1.32 h | 35 | 1 | 195 |
| brief-73 | 0.76 h | 11 | 1 | 121 |
| brief-74 | 2.52 h | 58 | 6 | 379 |
| brief-75 | 0.00 h | 4 | 1 | 86 |
| brief-76 | 1.32 h | 41 | 4 | 273 |
| brief-77 | 4.87 h | 112 | 5 | 905 |
| brief-78 | 13.79 h | 292 | 10 | 2,466 |
| brief-79 | 0.72 h | 25 | 3 | 207 |
| brief-80 | 0.16 h | 7 | 3 | 111 |
| brief-81 | 12.61 h | 253 | 10 | 2,234 |
| brief-82 | 1.57 h | 23 | 2 | 186 |
| brief-83 | 3.93 h | 73 | 7 | 795 |
| brief-84 | 2.97 h | 56 | 7 | 406 |
| brief-85 | 4.26 h | 85 | 6 | 887 |
| brief-86 | 3.07 h | 67 | 3 | 467 |
| brief-87 | 2.17 h | 52 | 8 | 247 |
| brief-88 | 4.51 h | 96 | 8 | 777 |
| brief-89 | 3.79 h | 75 | 8 | 956 |
| brief-90 | 1.93 h | 43 | 11 | 468 |
| brief-91 | 0.74 h | 23 | 4 | 168 |
| brief-92 | 0.00 h | 3 | 0 | 37 |
| steered/dashboard-fix | 2.80 h | 1 | 18 | 864 |
| steered/manager | 54.67 h | 94 | 49 | 2,517 |
| steered/reward-audit | 7.10 h | 13 | 34 | 739 |
| steered/run-recovery | 5.85 h | 0 | 2 | 327 |

Durations sum segment-active time and overlap across concurrent agents; they are not elapsed clock time. The fleet contributes 187.73 branch-hours, while the four steered workstreams contribute 70.41 segment-hours. From 4,780 unique release records at or before the CSV cutoff, run wrappers consumed 532,778.4 s (147.99 h): median 84.0 s, mean 111.5 s, p95 294.6 s, and maximum 2,906.6 s. All records use `lock=build mode=shared` because this project runs build, validation, and remote benchmark work through `autocuda run slice`; therefore lock metadata cannot reliably split those 147.99 hours by phase. The separate command classifier supplies the build/bench counts above. All 4,838 acquired-lock records at or before the cutoff reported 0.000 s wait, ruling out local lock contention as the long-tail cause.

## Tool Call Distribution

| Branch | Tool | Count |
|---|---|---:|
| brief-0 | exec | 194 |
| brief-0 | wait | 100 |
| brief-0 | send_message | 2 |
| brief-0 | spawn_agent | 1 |
| brief-1 | exec | 320 |
| brief-1 | wait | 123 |
| brief-1 | send_message | 2 |
| brief-2 | exec | 218 |
| brief-2 | wait | 89 |
| brief-2 | send_message | 7 |
| brief-3 | exec | 273 |
| brief-3 | wait | 83 |
| brief-3 | send_message | 3 |
| brief-3 | wait_agent | 3 |
| brief-3 | list_agents | 2 |
| brief-3 | spawn_agent | 1 |
| brief-4 | exec | 452 |
| brief-4 | wait | 195 |
| brief-5 | exec | 73 |
| brief-5 | wait | 47 |
| brief-6 | exec | 94 |
| brief-6 | wait | 25 |
| brief-6 | send_message | 1 |
| brief-7 | exec | 130 |
| brief-7 | wait | 70 |
| brief-7 | send_message | 1 |
| brief-8 | exec | 613 |
| brief-8 | wait | 267 |
| brief-8 | send_message | 1 |
| brief-9 | exec | 598 |
| brief-9 | wait | 331 |
| brief-9 | send_message | 4 |
| brief-9 | spawn_agent | 2 |
| brief-9 | list_agents | 1 |
| brief-10 | exec | 1,025 |
| brief-10 | wait | 302 |
| brief-10 | send_message | 30 |
| brief-10 | spawn_agent | 13 |
| brief-10 | wait_agent | 12 |
| brief-10 | list_agents | 6 |
| brief-10 | followup_task | 1 |
| brief-11 | exec | 459 |
| brief-11 | wait | 204 |
| brief-11 | send_message | 10 |
| brief-11 | list_agents | 2 |
| brief-11 | spawn_agent | 1 |
| brief-12 | exec | 142 |
| brief-12 | wait | 86 |
| brief-13 | exec | 56 |
| brief-13 | wait | 28 |
| brief-14 | exec | 162 |
| brief-14 | wait | 85 |
| brief-15 | exec | 146 |
| brief-15 | wait | 94 |
| brief-16 | exec | 235 |
| brief-16 | wait | 128 |
| brief-17 | exec | 166 |
| brief-17 | wait | 100 |
| brief-18 | exec | 201 |
| brief-18 | wait | 79 |
| brief-19 | exec | 82 |
| brief-19 | wait | 62 |
| brief-20 | exec | 80 |
| brief-20 | wait | 38 |
| brief-21 | exec | 52 |
| brief-21 | wait | 23 |
| brief-22 | exec | 376 |
| brief-22 | wait | 243 |
| brief-23 | exec | 166 |
| brief-23 | wait | 126 |
| brief-24 | exec | 464 |
| brief-24 | wait | 278 |
| brief-25 | exec | 1,054 |
| brief-25 | wait | 682 |
| brief-25 | send_message | 36 |
| brief-25 | spawn_agent | 11 |
| brief-25 | list_agents | 8 |
| brief-25 | wait_agent | 8 |
| brief-25 | followup_task | 1 |
| brief-26 | exec | 155 |
| brief-26 | wait | 84 |
| brief-27 | exec | 138 |
| brief-27 | wait | 43 |
| brief-28 | exec | 273 |
| brief-28 | wait | 130 |
| brief-29 | exec | 627 |
| brief-29 | wait | 243 |
| brief-29 | send_message | 6 |
| brief-29 | spawn_agent | 6 |
| brief-29 | list_agents | 3 |
| brief-29 | wait_agent | 1 |
| brief-30 | exec | 166 |
| brief-30 | wait | 84 |
| brief-31 | exec | 171 |
| brief-31 | wait | 98 |
| brief-32 | exec | 213 |
| brief-32 | wait | 136 |
| brief-33 | exec | 159 |
| brief-33 | wait | 101 |
| brief-34 | exec | 160 |
| brief-34 | wait | 73 |
| brief-35 | exec | 28 |
| brief-35 | wait | 12 |
| brief-36 | exec | 274 |
| brief-36 | wait | 122 |
| brief-37 | exec | 30 |
| brief-37 | wait | 13 |
| brief-38 | exec | 202 |
| brief-38 | wait | 91 |
| brief-39 | exec | 177 |
| brief-39 | wait | 91 |
| brief-40 | exec | 98 |
| brief-40 | wait | 58 |
| brief-41 | exec | 150 |
| brief-41 | wait | 68 |
| brief-42 | exec | 121 |
| brief-42 | wait | 60 |
| brief-43 | exec | 677 |
| brief-43 | wait | 321 |
| brief-43 | send_message | 12 |
| brief-43 | wait_agent | 2 |
| brief-43 | list_agents | 1 |
| brief-43 | spawn_agent | 1 |
| brief-44 | exec | 182 |
| brief-44 | wait | 106 |
| brief-45 | exec | 158 |
| brief-45 | wait | 86 |
| brief-46 | exec | 114 |
| brief-46 | wait | 52 |
| brief-47 | exec | 91 |
| brief-47 | wait | 54 |
| brief-48 | exec | 105 |
| brief-48 | wait | 53 |
| brief-49 | exec | 739 |
| brief-49 | wait | 337 |
| brief-49 | send_message | 24 |
| brief-49 | wait_agent | 4 |
| brief-49 | list_agents | 3 |
| brief-49 | interrupt_agent | 1 |
| brief-49 | spawn_agent | 1 |
| brief-50 | exec | 471 |
| brief-50 | wait | 269 |
| brief-50 | send_message | 22 |
| brief-50 | wait_agent | 13 |
| brief-50 | list_agents | 2 |
| brief-51 | exec | 128 |
| brief-51 | wait | 60 |
| brief-52 | exec | 306 |
| brief-52 | wait | 121 |
| brief-52 | send_message | 3 |
| brief-52 | wait_agent | 2 |
| brief-53 | exec | 470 |
| brief-53 | wait | 180 |
| brief-53 | send_message | 18 |
| brief-53 | wait_agent | 3 |
| brief-53 | list_agents | 1 |
| brief-53 | spawn_agent | 1 |
| brief-54 | exec | 177 |
| brief-54 | wait | 97 |
| brief-54 | send_message | 1 |
| brief-55 | exec | 476 |
| brief-55 | wait | 213 |
| brief-55 | send_message | 12 |
| brief-55 | list_agents | 2 |
| brief-55 | wait_agent | 2 |
| brief-56 | exec | 351 |
| brief-56 | wait | 159 |
| brief-56 | send_message | 7 |
| brief-56 | spawn_agent | 2 |
| brief-57 | exec | 187 |
| brief-57 | wait | 109 |
| brief-57 | send_message | 4 |
| brief-58 | exec | 457 |
| brief-58 | wait | 216 |
| brief-58 | send_message | 29 |
| brief-58 | wait_agent | 11 |
| brief-58 | list_agents | 6 |
| brief-58 | spawn_agent | 3 |
| brief-59 | exec | 169 |
| brief-59 | wait | 66 |
| brief-59 | send_message | 1 |
| brief-59 | wait_agent | 1 |
| brief-60 | exec | 219 |
| brief-60 | wait | 114 |
| brief-61 | exec | 312 |
| brief-61 | wait | 163 |
| brief-61 | send_message | 5 |
| brief-61 | wait_agent | 2 |
| brief-62 | exec | 305 |
| brief-62 | wait | 177 |
| brief-62 | send_message | 3 |
| brief-63 | exec | 257 |
| brief-63 | wait | 139 |
| brief-63 | wait_agent | 8 |
| brief-63 | list_agents | 5 |
| brief-64 | exec | 288 |
| brief-64 | wait | 184 |
| brief-65 | exec | 141 |
| brief-65 | wait | 90 |
| brief-66 | exec | 130 |
| brief-66 | wait | 63 |
| brief-67 | exec | 108 |
| brief-67 | wait | 62 |
| brief-68 | exec | 239 |
| brief-68 | wait | 131 |
| brief-68 | send_message | 4 |
| brief-68 | list_agents | 1 |
| brief-69 | exec | 214 |
| brief-69 | wait | 106 |
| brief-69 | list_agents | 3 |
| brief-69 | send_message | 3 |
| brief-69 | wait_agent | 3 |
| brief-70 | exec | 48 |
| brief-70 | wait | 26 |
| brief-71 | exec | 814 |
| brief-71 | wait | 551 |
| brief-71 | send_message | 29 |
| brief-71 | list_agents | 6 |
| brief-71 | wait_agent | 6 |
| brief-71 | followup_task | 3 |
| brief-71 | interrupt_agent | 2 |
| brief-71 | spawn_agent | 1 |
| brief-72 | exec | 171 |
| brief-72 | wait | 56 |
| brief-72 | send_message | 2 |
| brief-72 | wait_agent | 2 |
| brief-73 | exec | 86 |
| brief-73 | wait | 45 |
| brief-73 | send_message | 2 |
| brief-74 | exec | 313 |
| brief-74 | wait | 106 |
| brief-74 | send_message | 13 |
| brief-74 | list_agents | 6 |
| brief-74 | wait_agent | 3 |
| brief-74 | spawn_agent | 2 |
| brief-75 | exec | 52 |
| brief-75 | wait | 32 |
| brief-75 | wait_agent | 4 |
| brief-75 | send_message | 2 |
| brief-75 | list_agents | 1 |
| brief-76 | exec | 222 |
| brief-76 | wait | 92 |
| brief-76 | send_message | 3 |
| brief-76 | list_agents | 1 |
| brief-77 | exec | 665 |
| brief-77 | wait | 337 |
| brief-77 | send_message | 18 |
| brief-77 | wait_agent | 2 |
| brief-78 | exec | 1,830 |
| brief-78 | wait | 914 |
| brief-78 | send_message | 16 |
| brief-78 | list_agents | 5 |
| brief-78 | interrupt_agent | 1 |
| brief-78 | spawn_agent | 1 |
| brief-78 | wait_agent | 1 |
| brief-79 | exec | 150 |
| brief-79 | wait | 79 |
| brief-79 | send_message | 5 |
| brief-79 | wait_agent | 1 |
| brief-80 | exec | 73 |
| brief-80 | wait | 46 |
| brief-80 | send_message | 2 |
| brief-81 | exec | 1,547 |
| brief-81 | wait | 914 |
| brief-81 | send_message | 22 |
| brief-81 | wait_agent | 7 |
| brief-81 | list_agents | 4 |
| brief-81 | spawn_agent | 2 |
| brief-81 | interrupt_agent | 1 |
| brief-82 | exec | 149 |
| brief-82 | wait | 61 |
| brief-82 | send_message | 1 |
| brief-83 | exec | 593 |
| brief-83 | wait | 270 |
| brief-83 | send_message | 7 |
| brief-83 | spawn_agent | 4 |
| brief-83 | list_agents | 1 |
| brief-84 | exec | 388 |
| brief-84 | wait | 62 |
| brief-84 | send_message | 13 |
| brief-84 | spawn_agent | 5 |
| brief-84 | wait_agent | 1 |
| brief-85 | exec | 680 |
| brief-85 | wait | 249 |
| brief-85 | send_message | 24 |
| brief-85 | spawn_agent | 10 |
| brief-85 | wait_agent | 10 |
| brief-85 | list_agents | 5 |
| brief-86 | exec | 364 |
| brief-86 | wait | 146 |
| brief-86 | send_message | 11 |
| brief-86 | spawn_agent | 7 |
| brief-86 | wait_agent | 7 |
| brief-86 | list_agents | 2 |
| brief-87 | exec | 213 |
| brief-87 | wait | 90 |
| brief-87 | send_message | 4 |
| brief-88 | exec | 604 |
| brief-88 | wait | 261 |
| brief-88 | send_message | 7 |
| brief-88 | spawn_agent | 4 |
| brief-88 | wait_agent | 3 |
| brief-88 | list_agents | 2 |
| brief-89 | exec | 792 |
| brief-89 | wait | 178 |
| brief-89 | send_message | 22 |
| brief-89 | wait_agent | 19 |
| brief-89 | spawn_agent | 18 |
| brief-89 | list_agents | 9 |
| brief-89 | followup_task | 1 |
| brief-90 | exec | 357 |
| brief-90 | wait | 153 |
| brief-90 | send_message | 8 |
| brief-90 | spawn_agent | 3 |
| brief-90 | wait_agent | 1 |
| brief-91 | exec | 129 |
| brief-91 | wait | 66 |
| brief-92 | exec | 26 |
| brief-92 | wait | 14 |
| steered/dashboard-fix | exec | 778 |
| steered/dashboard-fix | wait | 35 |
| steered/dashboard-fix | send_message | 31 |
| steered/dashboard-fix | spawn_agent | 14 |
| steered/dashboard-fix | wait_agent | 13 |
| steered/dashboard-fix | list_agents | 12 |
| steered/manager | exec | 1,461 |
| steered/manager | wait | 603 |
| steered/manager | wait_agent | 298 |
| steered/manager | spawn_agent | 129 |
| steered/manager | send_message | 109 |
| steered/manager | followup_task | 32 |
| steered/manager | list_agents | 23 |
| steered/manager | interrupt_agent | 5 |
| steered/reward-audit | exec | 648 |
| steered/reward-audit | send_message | 58 |
| steered/reward-audit | spawn_agent | 25 |
| steered/reward-audit | wait_agent | 24 |
| steered/reward-audit | wait | 17 |
| steered/reward-audit | list_agents | 12 |
| steered/reward-audit | followup_task | 2 |
| steered/run-recovery | exec | 292 |
| steered/run-recovery | send_message | 14 |
| steered/run-recovery | spawn_agent | 9 |
| steered/run-recovery | list_agents | 6 |
| steered/run-recovery | wait | 4 |
| steered/run-recovery | wait_agent | 3 |
| steered/run-recovery | interrupt_agent | 1 |

Across all rows, 48,042 tool calls were classified. `exec` accounts for 31,889 (66.4%) and `wait` for 14,530 (30.2%); no WebFetch or WebSearch calls were recorded. Build-classified calls outnumber benchmark-classified calls 4,900 to 633 (7.7:1), so reducing unpromising candidates before full validation/build cycles has more leverage than trimming benchmark count alone.

## Inference Breakdown

| Branch | Input tokens | Output tokens | Cached input tokens | Cache creation tokens | Cache hit % | Compactions | Estimated USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| brief-0 | 501,094 | 70,055 | 28,182,784 | 0 | 98.25% | 0 | $18.70 |
| brief-1 | 760,798 | 78,433 | 45,766,912 | 0 | 98.36% | 1 | $29.04 |
| brief-2 | 537,548 | 57,885 | 44,219,648 | 0 | 98.80% | 1 | $26.53 |
| brief-3 | 652,650 | 76,379 | 46,314,752 | 0 | 98.61% | 1 | $28.71 |
| brief-4 | 793,870 | 111,396 | 67,318,528 | 0 | 98.83% | 1 | $40.97 |
| brief-5 | 175,216 | 17,103 | 8,739,584 | 0 | 98.03% | 0 | $5.76 |
| brief-6 | 178,348 | 27,373 | 10,220,288 | 0 | 98.28% | 0 | $6.82 |
| brief-7 | 260,737 | 30,603 | 18,327,040 | 0 | 98.60% | 0 | $11.39 |
| brief-8 | 1,349,654 | 173,832 | 103,840,000 | 0 | 98.72% | 3 | $63.88 |
| brief-9 | 1,747,334 | 172,198 | 114,962,176 | 0 | 98.50% | 2 | $71.38 |
| brief-10 | 3,183,188 | 364,468 | 158,374,656 | 0 | 98.03% | 4 | $106.04 |
| brief-11 | 887,659 | 116,384 | 74,351,872 | 0 | 98.82% | 1 | $45.11 |
| brief-12 | 298,365 | 34,766 | 23,995,136 | 0 | 98.77% | 0 | $14.53 |
| brief-13 | 217,760 | 16,477 | 6,588,160 | 0 | 96.80% | 0 | $4.88 |
| brief-14 | 307,991 | 35,966 | 27,507,712 | 0 | 98.89% | 0 | $16.37 |
| brief-15 | 322,445 | 42,867 | 28,959,488 | 0 | 98.90% | 0 | $17.38 |
| brief-16 | 799,175 | 59,782 | 49,875,712 | 0 | 98.42% | 0 | $30.73 |
| brief-17 | 600,197 | 44,646 | 35,535,104 | 0 | 98.34% | 0 | $22.11 |
| brief-18 | 534,278 | 57,753 | 37,974,784 | 0 | 98.61% | 0 | $23.39 |
| brief-19 | 435,514 | 28,123 | 14,378,752 | 0 | 97.06% | 0 | $10.21 |
| brief-20 | 318,586 | 24,430 | 9,852,160 | 0 | 96.87% | 0 | $7.25 |
| brief-21 | 190,433 | 13,386 | 5,161,984 | 0 | 96.44% | 0 | $3.93 |
| brief-22 | 1,200,591 | 86,882 | 81,913,856 | 0 | 98.56% | 1 | $49.57 |
| brief-23 | 664,127 | 53,236 | 41,278,208 | 0 | 98.42% | 0 | $25.56 |
| brief-24 | 1,332,556 | 104,249 | 103,364,864 | 0 | 98.73% | 1 | $61.47 |
| brief-25 | 4,075,133 | 464,325 | 231,512,832 | 0 | 98.27% | 6 | $150.06 |
| brief-26 | 622,890 | 44,082 | 27,052,032 | 0 | 97.75% | 0 | $17.96 |
| brief-27 | 238,822 | 33,826 | 17,356,288 | 0 | 98.64% | 0 | $10.89 |
| brief-28 | 483,810 | 69,147 | 63,251,456 | 0 | 99.24% | 0 | $36.12 |
| brief-29 | 1,521,450 | 184,860 | 105,532,416 | 0 | 98.58% | 1 | $65.92 |
| brief-30 | 309,937 | 37,782 | 25,833,728 | 0 | 98.81% | 0 | $15.60 |
| brief-31 | 318,282 | 35,847 | 31,625,728 | 0 | 99.00% | 0 | $18.48 |
| brief-32 | 504,909 | 40,883 | 38,205,696 | 0 | 98.70% | 0 | $22.85 |
| brief-33 | 305,009 | 31,367 | 26,433,280 | 0 | 98.86% | 0 | $15.68 |
| brief-34 | 303,234 | 36,371 | 27,201,024 | 0 | 98.90% | 0 | $16.21 |
| brief-35 | 90,332 | 8,066 | 2,037,504 | 0 | 95.75% | 0 | $1.71 |
| brief-36 | 520,680 | 51,809 | 58,133,248 | 0 | 99.11% | 0 | $33.22 |
| brief-37 | 71,882 | 7,411 | 2,230,272 | 0 | 96.88% | 0 | $1.70 |
| brief-38 | 400,706 | 45,645 | 45,724,160 | 0 | 99.13% | 0 | $26.23 |
| brief-39 | 387,166 | 56,849 | 42,269,440 | 0 | 99.09% | 0 | $24.78 |
| brief-40 | 206,960 | 20,847 | 15,094,528 | 0 | 98.65% | 0 | $9.21 |
| brief-41 | 274,659 | 32,528 | 22,573,312 | 0 | 98.80% | 0 | $13.64 |
| brief-42 | 246,405 | 32,300 | 16,818,688 | 0 | 98.56% | 0 | $10.61 |
| brief-43 | 1,273,657 | 185,071 | 158,771,863 | 0 | 99.20% | 1 | $91.31 |
| brief-44 | 325,093 | 49,511 | 32,022,272 | 0 | 98.99% | 0 | $19.12 |
| brief-45 | 373,985 | 42,960 | 33,955,584 | 0 | 98.91% | 0 | $20.14 |
| brief-46 | 199,505 | 26,344 | 13,702,400 | 0 | 98.56% | 0 | $8.64 |
| brief-47 | 220,806 | 23,585 | 13,687,296 | 0 | 98.41% | 0 | $8.66 |
| brief-48 | 220,227 | 22,701 | 15,662,336 | 0 | 98.61% | 0 | $9.61 |
| brief-49 | 1,402,402 | 210,327 | 175,682,217 | 0 | 99.21% | 1 | $101.16 |
| brief-50 | 1,104,395 | 134,017 | 108,373,099 | 0 | 98.99% | 1 | $63.73 |
| brief-51 | 243,077 | 25,527 | 19,013,376 | 0 | 98.74% | 0 | $11.49 |
| brief-52 | 492,644 | 103,092 | 57,121,214 | 0 | 99.14% | 0 | $34.12 |
| brief-53 | 1,017,047 | 136,964 | 98,931,819 | 0 | 98.98% | 1 | $58.66 |
| brief-54 | 198,926 | 45,857 | 35,777,539 | 0 | 99.45% | 0 | $20.26 |
| brief-55 | 815,964 | 145,191 | 125,016,906 | 0 | 99.35% | 1 | $70.94 |
| brief-56 | 558,804 | 93,515 | 80,614,458 | 0 | 99.31% | 0 | $45.91 |
| brief-57 | 218,334 | 48,148 | 41,251,602 | 0 | 99.47% | 0 | $23.16 |
| brief-58 | 1,394,949 | 186,353 | 117,521,845 | 0 | 98.83% | 1 | $71.33 |
| brief-59 | 310,134 | 45,001 | 25,802,541 | 0 | 98.81% | 0 | $15.80 |
| brief-60 | 301,098 | 60,708 | 37,917,841 | 0 | 99.21% | 0 | $22.29 |
| brief-61 | 304,125 | 76,641 | 99,085,600 | 0 | 99.69% | 0 | $53.36 |
| brief-62 | 283,934 | 88,521 | 84,489,543 | 0 | 99.67% | 0 | $46.32 |
| brief-63 | 891,752 | 70,599 | 94,935,283 | 0 | 99.07% | 1 | $54.04 |
| brief-64 | 280,115 | 64,775 | 82,844,909 | 0 | 99.66% | 0 | $44.77 |
| brief-65 | 151,111 | 35,433 | 26,317,598 | 0 | 99.43% | 0 | $14.98 |
| brief-66 | 155,367 | 27,276 | 21,645,114 | 0 | 99.29% | 0 | $12.42 |
| brief-67 | 191,071 | 32,977 | 24,633,729 | 0 | 99.23% | 0 | $14.26 |
| brief-68 | 243,309 | 52,818 | 57,045,904 | 0 | 99.58% | 0 | $31.32 |
| brief-69 | 277,601 | 70,953 | 53,748,527 | 0 | 99.49% | 0 | $30.39 |
| brief-70 | 87,100 | 14,499 | 4,572,612 | 0 | 98.13% | 0 | $3.16 |
| brief-71 | 1,510,553 | 232,256 | 239,237,004 | 0 | 99.37% | 2 | $134.14 |
| brief-72 | 230,631 | 56,173 | 32,255,060 | 0 | 99.29% | 0 | $18.97 |
| brief-73 | 165,417 | 28,318 | 14,340,397 | 0 | 98.86% | 0 | $8.85 |
| brief-74 | 1,175,836 | 121,342 | 71,858,172 | 0 | 98.39% | 1 | $45.45 |
| brief-75 | 119,915 | 16,989 | 7,323,506 | 0 | 98.39% | 0 | $4.77 |
| brief-76 | 291,419 | 65,597 | 59,360,971 | 0 | 99.51% | 0 | $33.11 |
| brief-77 | 892,834 | 157,648 | 176,463,038 | 0 | 99.50% | 1 | $97.43 |
| brief-78 | 3,668,965 | 457,928 | 547,675,495 | 0 | 99.33% | 5 | $305.92 |
| brief-79 | 170,103 | 35,407 | 26,259,043 | 0 | 99.36% | 0 | $15.04 |
| brief-80 | 131,766 | 19,009 | 11,216,045 | 0 | 98.84% | 0 | $6.84 |
| brief-81 | 2,934,802 | 391,523 | 468,298,444 | 0 | 99.38% | 4 | $260.57 |
| brief-82 | 220,970 | 56,348 | 31,211,913 | 0 | 99.30% | 0 | $18.40 |
| brief-83 | 1,444,208 | 189,314 | 159,810,456 | 0 | 99.10% | 1 | $92.81 |
| brief-84 | 1,269,522 | 136,506 | 81,670,814 | 0 | 98.47% | 1 | $51.28 |
| brief-85 | 1,830,386 | 295,124 | 160,051,629 | 0 | 98.87% | 1 | $98.03 |
| brief-86 | 1,198,831 | 193,934 | 83,821,371 | 0 | 98.59% | 1 | $53.72 |
| brief-87 | 320,025 | 63,765 | 61,489,796 | 0 | 99.48% | 0 | $34.26 |
| brief-88 | 1,208,949 | 196,331 | 136,258,936 | 0 | 99.12% | 1 | $80.06 |
| brief-89 | 2,916,016 | 458,387 | 144,501,578 | 0 | 98.02% | 2 | $100.58 |
| brief-90 | 1,116,219 | 108,127 | 96,392,744 | 0 | 98.86% | 1 | $57.02 |
| brief-91 | 366,655 | 38,971 | 25,952,975 | 0 | 98.61% | 0 | $15.98 |
| brief-92 | 69,293 | 7,155 | 2,247,831 | 0 | 97.01% | 0 | $1.69 |
| steered/dashboard-fix | 3,107,810 | 302,218 | 113,325,628 | 0 | 97.33% | 5 | $81.27 |
| steered/manager | 7,623,491 | 512,600 | 269,376,361 | 0 | 97.25% | 6 | $188.18 |
| steered/reward-audit | 3,151,752 | 337,961 | 70,907,136 | 0 | 95.74% | 1 | $61.35 |
| steered/run-recovery | 1,867,186 | 149,287 | 40,481,869 | 0 | 95.59% | 0 | $34.06 |

The fleet totals $3,768.82 across 93 briefs; steered manager, reward-audit, dashboard-fix, and run-recovery work adds $364.86. At the stated rates, uncached input contributes $408.36, cached input $3,428.91, and output $296.41. Inference latency is already part of branch/session duration; the USD estimate is not added to walltime or treated as a separate elapsed interval.

## Cost-Optimization Opportunities

- Retire or hand off long-lived briefs earlier. Briefs 78 and 81 were 13.1× and 11.1× the $23.39 fleet median, with five and four compactions; 26 briefs exceeded the skill's 2×-median runaway signal.
- Tighten wrapper deadlines and surface remote hangs. The 294.6 s p95 and 2,906.6 s maximum far exceed the environment's 90 s validation/benchmark and 120 s fresh-build allowances, while lock wait is exactly zero.
- Reduce polling chatter. `wait` is 30.2% of all tool calls; longer event-driven waits can cut inference turns without reducing remote GPU utilization.
- Add cheap rejection gates before full build/validation. Build-classified calls exceed benchmark calls 7.7:1 across 1,584 recorded trials.
- Preserve prompt caching. The 98.82% hit rate is excellent; making the same cache-read volume uncached would increase its input-rate equivalent from about $3.4K to $34.3K.

## Analysis

The same-tag optimization logs show 1,584 trials, including 1,407 successful measurements. The best observed result, brief 88 trial 19 at commit `6d5995fe`, reduced the baseline from 2,153.690 µs to 743.614 µs: 2.896× speedup, or 65.47% lower latency. That is a substantial result, but the $4.1K figure remains a public-API-rate comparison proxy rather than an invoice.

Spend and productivity are related but not proportional. Brief 78 is the most expensive at $305.92 over 97 trials and reached 777.498 µs; brief 81 cost $260.57 over 83 trials and reached 757.720 µs. The winning brief 88 cost $80.06 over 32 trials, roughly one quarter of brief 78's spend. Brief 92's $1.69 row reached 745.132 µs on one combine trial, but it was still active and primarily inherited prior work, so it is not evidence that the cheapest independent search won.

Context growth is controlled better by caching than by conversation turnover. Cached reads dominate volume, while 63 compactions cluster in the manager (6), brief 25 (6), brief 78 (5), dashboard-fix (5), and briefs 10/81 (4 each). The longest briefs cost about $3.1 per trial versus a $2.38 fleet average, suggesting moderate context accumulation rather than explosive super-linear growth.

**Recommendations:** cap or checkpoint briefs near 40–50 trials, enforce phase-specific 90/120 s deadlines with explicit remote-timeout telemetry, and replace short polling loops with event-driven waits. Keep the current cache-friendly prompt structure and fork-boundary accounting.

## Scope & Methodology

This is single-experiment mode for tag `2026-07-19-01-44-06-cholesky-resumed-local-unmerged`, target `linalg/cholesky_py`. Per-branch tables intentionally omit totals rows. The CSV snapshot was written at 2026-07-20 21:07:18 UTC while briefs 78, 81, 89, 91, and 92 lacked `brief_stop`; active files and logs continued to grow after the cutoff, so every figure is provisional.

Coverage comprises 184 fleet mappings and 74 steered mappings: manager (27), reward audit (22), dashboard fix (15), and run recovery (10). All 258 content hashes are unique and parsed successfully; the CLI proved child boundaries for 237 forked rollouts and treated 21 as non-fork sessions. Every brief 0–92 is covered. Four local Claude Code files were excluded because they were automated model probes that ended before the run, and the export ref contains memory/history but no rollout JSONLs. No known session-coverage gap remains.

The identified model is GPT-5.6-sol: 829 turn contexts use `gpt-5.6-sol` and 618 use the equivalent fully qualified `openai/openai/gpt-5.6-sol`. Pricing assumes public standard rates of $5.00/M uncached input, $0.50/M cached input, and $30.00/M output.[^1] Token and tool summaries come from the mapped JSONLs; fleet duration comes from experiment logs, steered duration is the sum of segment spans, and exact duplicate content would be counted once by the CLI. Wrapper durations are deduplicated by release timestamp and filtered to the CSV cutoff. Estimates are approximate, and subscription, priority-tier, negotiated, or infrastructure charges may differ.

[^1]: [OpenAI GPT-5.6 Sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol), accessed 2026-07-20. The displayed rates are per 1M tokens and are used only for this approximate API-rate-equivalent estimate.
