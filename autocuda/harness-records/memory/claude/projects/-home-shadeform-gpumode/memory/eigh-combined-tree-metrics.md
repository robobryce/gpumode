---
name: eigh-combined-tree-metrics
description: "How to compute autocuda tree-diversity (Shao-Sokal B2) for the COMBINED two-session Opus eigh run (05-33-43 + 18-52-36), and the resulting metrics vs the Codex run"
metadata: 
  node_type: memory
  type: reference
  originSessionId: dc1df5f2-71f9-4f60-a53b-d1ecac08ce91
---

Task: compare the Opus 4.8 eigh experiment (two sessions [[eigh-tree-run-2026-06-27]]-lineage: run1 `2026-06-30-05-33-43-eigh` + run2 `2026-06-30-18-52-36-eigh`, see [[eigh-tree-run-2026-06-30]] / [[eigh-tree-run-2026-06-30-resume]]) against the ongoing Codex run `2026-07-03-13-39-25` (harness=`codex-aab-real`, model gpt-5.5 per running proc / attributed gpt-5.6-sol in cost reports).

**The two Opus sessions are ONE logical search tree.** run1 rooted at true `main` (`08291948`, 56238µs) and produced `d6553505f` (the three-win stack) as brief-1's trial-4 (47850µs). run2's *baseline* IS `d6553505f` (47824µs) and every run2 round-1 brief forks off it. So combined diversity = reconstruct the single tree spanning both and run the shipped `tree_diversity` on it.

**Why `autocuda status` alone can't do it:** run1 uses the OLD worker-log schema (`agent_id`,`iteration`, 3 duplicate baseline rows, files named `-worker-N-log.csv`); the current CLI pipeline (`_dashboard.load_log`→`parse_rows`→`resolve_commits`→`build_tree`→`tree_diversity`) only reads `-brief-K-log.csv` + `-manager-log.csv` + `-reference-log.csv`, so run1 returns `diversity: null` and `trial_count: 0`.

**Method that worked (validated):** synthesize unified current-schema logs under a temp canonical tag in a scratch dir, then run the REAL `autocuda status --data-dir <scratch> --tag <ts>`:
- reference-log: single `baseline` row = true main (56238.201, commit 08291948).
- manager-log: union of both runs' brief rows (cols timestamp,brief_id,parent_commits,brief_kind,description). Renamespace run1 brief-ids +1000 to avoid collision with run2's 0..121 (safe: graft is by commit-SHA via `resolve_commits`, not brief-id).
- brief-logs: run1 worker rows mapped iteration→trial, drop baseline rows (ref owns baseline), brief_id+1000; run2 brief logs copied verbatim.
- tag MUST be canonical `%Y-%m-%d-%H-%M-%S[-suffix]` or status rejects it (used `2026-06-30-05-33-43-combined`).
VALIDATION: feeding run2-alone through the same synth path reproduced the official `autocuda status` diversity for 18-52-36 EXACTLY (nodes 907/leaves 96/eff 6.76/balance 0.4186). GRAFT VERIFIED: the 5 run2 briefs that sit on the baseline root in run2-alone re-attach to the `d6553505f` trial node (`trial:1004:181`) in the combined tree; 0 run2 briefs leaked onto the true-main root (only run1's own 5 round-1 briefs sit there). `tree_diversity` ignores scores — pure topology — so faithful structure is all that's needed.

**Diversity is Shao & Sokal's B2 index** (`_dashboard.py:tree_diversity`): equal-split probability walk root→leaves, entropy H over leaf probs; effective_directions=2^H, balance=H/log2(leaves). nodes counts every node incl baseline+briefs+trials.

RESULTS (metrics for the comparison table):
| metric | Opus run1 05-33-43 | Opus run2 18-52-36 | Opus COMBINED | Codex 13-39-25 (ongoing) |
|---|---|---|---|---|
| tree nodes | (n/a schema) | 907 | **1021** | 565 |
| tree leaves | " | 96 | **119** | 11 |
| effective_directions | " | 6.76 | **11.72** | 5.47 |
| balance | " | 0.4186 | **0.515** | 0.7086 |
| briefs | 15 (36 mgr rows) | 122 | 137 | 15 |
| trials (excl start/stop/baseline) | 78 | 784 | 862 | ~548 (live) |
| errors | 7 | 8 | 15 (~1.7%) | 169 (~31%, live) |
| walltime active | 10.9h | 53.1h | ~64h | ~12.9h & counting |
| best local geomean | 47850 | 25458.938 (=304e5f61f, 1.878× vs its 47824 resume base) | 25459 = **2.209× vs true main 56238** | 40614.977 = **1.386× vs main 56277** |
| best LOC (submission.py) | | 7236 total / 4880 code (12 __global__ + Triton) | | 1398 total / 1267 code (9 __global__, no Triton) |
| total tokens | 1.87B | 5.84B | **7.71B** | **~2.89B** (live) |

TOKENS method: sum per-session token usage across harness logs. Opus=Claude Code JSONL under `~/.claude/projects/-home-shadeform-gpumode/`: manager `0435e030` (run1, 15 subagents) + `94ecbd36` (run2, 134 subagents), summing per-message input+cache_creation+cache_read+output (cache_read dominates: 7.35B of 7.71B). Codex=`~/.codex/sessions/2026/07/0{3,4}/rollout-*.jsonl`, 17 sessions >= 13:37 UTC, summing each file's LAST `event_msg.payload.type=token_count → info.total_token_usage` (cumulative per session; total=input+output, cached_input is a SUBSET of input; cached 2.86B of 2.89B). Both count cache reads → comparable. CAVEAT: cross-checking the Codex method against the published `2026-07-03-05-14-36-report-cost.md` (finished run, said 1.64B) overshot ~35% (I got 2.22B) — I include resumed-session re-reads the report's mapping drops; treat Codex absolute as upper-ish bound, ratio is method-consistent.

DERIVED efficiency (FINAL formula, operator-directed 2026-07-04, supersedes both raw-ratio and (s-1)/cost): **metric = speedup ^ (1 / cost)** — geometric/CAGR-style per-unit-cost multiplicative rate. cost = tokens in billions, or code lines in thousands. Answers "per unit cost, by what factor did perf multiply?" Satisfies the operator's required boundary conditions that BOTH prior forms failed: (1) baseline (speedup 1.0, 0 cost) → metric = 1.0 exactly (1^(1/0)=1, and limit as cost→0 with s=1 is 1.0); (2) any real gain (s>1) → metric ALWAYS >1; (3) monotonic ↑ in speedup at fixed cost, decays →1.0 as cost→∞ at fixed speedup. Prior forms (s/cost and (s-1)/cost) DIVERGE to ∞ as cost→0, so can't hit 1.0 at baseline — that boundary condition is the tell the metric must be a rate of a multiplicative quantity, not a ratio. Labels in files: "Speedup ^ (1 / billion tokens)" and "Speedup ^ (1 / 1k code lines)".

Values (speedup, ^1/Btok, ^1/kLOC): 12h Opus 1.175/1.086/1.256, Sol 1.382/1.133/1.290 (Sol wins both); 29h Opus 1.569/1.129/1.383, Sol 2.012/1.070/1.179 (Opus wins both); FULL(live~29.7h) Opus 2.209/1.108/1.176, Sol 2.062/1.071/1.187 (Opus wins tokens, Sol wins LOC). NOTE Sol still running — FULL Sol is a moving snapshot (was 1.39× at first full-run table, now 2.06×).

SUPERSEDED history: (a) raw speedup/cost; (b) (speedup-1)/cost [operator rejected: fails baseline=1.0 condition]. Rationale: a do-nothing run still scores speedup 1.0×, so raw-speedup÷cost credits a run for merely existing; `speedup−1` = the actual gain over baseline the tokens/lines bought (ROI not revenue). This is not cosmetic — it FLIPS rankings because Opus's speedups sit far above 1 while Codex's early ones are near 1 (the −1 hits Codex's numerator proportionally harder). Corrected values (gain/Btok, gain/1k-code-lines):
- FULL run: Opus 0.157 / 0.248 ; Codex 0.133 / 0.304 (Opus wins tokens, Codex wins LOC — raw had Codex winning BOTH).
- 12h: Opus 0.090 / 0.247 ; Codex 0.148 / 0.300 (Codex wins both — raw had Opus winning both).
- 28h: Opus 0.152 / 0.392 ; Codex 0.094 / 0.229 (Opus wins both).
Raw (÷ speedup) values, superseded: FULL Opus 0.287/0.453 Codex 0.479/1.094; 12h Opus 0.603/1.658 Codex 0.535/1.088; 28h Opus 0.431/1.112 Codex 0.195/0.474. Row labels in the 3 md files now read "(Speedup − 1) / billion tokens" and "(Speedup − 1) / 1k code lines".

**NCU/NSYS profiler tool-call counts** (full run / first-12h): Opus ncu=132/3, nsys=546/28; Codex ncu=22/18, nsys=200/174. CRITICAL METHOD NOTE: the eigh harness mandates profiling THROUGH WRAPPERS `harness/profile_ncu.sh` / `harness/profile_nsys.sh` (layout.md: "Profile through the harness scripts"), NOT raw `ncu`/`nsys`. Counting bare binary names misses ~99% (gives Opus nsys=4 not 546). Also, Opus workers write "ncu"/"nsys" constantly in git commit messages + `--description` log text ("nsys shows the Jacobi kernel...", "ncu --set full") — a substring/leading-token scan of the tool-call text counts that PROSE as calls (inflates Opus nsys to 334). Correct method: per tool-call, match `profile_(ncu|nsys)\.sh` OR a shell-segment whose leading token (after env-assign/sudo, path-stripped) is the raw binary followed by a real subcommand/flag; scan the ACTUAL command string (Codex: `cmd:"..."` inside custom_tool_call name=exec; Claude: Bash tool_use input.command). Both harnesses lean nsys-heavy (cheap per-kernel summary) over ncu (expensive full-section) per layout guidance. Codex 12h≈full (192 vs 222) — most profiling was early.

**Codex run trajectory (NOT plateaued):** by-hour best speedup 1.12(h1)→1.38(h12)→1.55→1.93× at 28h (29,112µs, commit 4ee2a0a6, 51 briefs/1154 trials, 9.9B tokens). At 28h Codex OVERTOOK Opus's same-window mark (1.93 vs 1.55×) and its best is now the LARGER submission (4076 code/22 kernels vs Opus 28h 3adba24b 1389/3). Still running past 29h as of 2026-07-04 19:10. Windowed comparison files: `autocuda/2026-07-04-03-30-31-opus-4.8-vs-gpt-5.6-sol{,-first-12h}.md` + `autocuda/2026-07-04-19-10-50-...-first-28h.md`. 28h tree: Opus nodes318/leaves43/eff11.72/bal0.654; Codex nodes1206/leaves31/eff5.53/bal0.498.

Opus best `304e5f61f` was leaderboard-verified 25,470µs public rank #4/58. Opus report's "1.49× / 32,182µs (e23884fa48)" is a MID-RUN snapshot, NOT the final best — don't quote it as the run result. Codex best `ef5491d6` not yet submitted.

CORRECTION worth remembering: Opus best is NOT "pure PyTorch/cuSOLVER orchestration" — it has 12 custom `__global__` CUDA kernels (mega_eigh family, warp/mw jacobi, fused_gram_chol) + Triton (4 @triton, 57 tl.*) + cuSOLVER only as residual-gated floor.
