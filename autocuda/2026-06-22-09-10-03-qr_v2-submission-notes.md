# Leaderboard submission log — qr_v2 / B200

Leaderboard name: qr_v2   GPU: B200   user: badelsteinlelbach

## Standings snapshot at run start (2026-06-22 09:18)
- #1  apeirontic        422.355 µs   candidate_166.py   <-- TARGET to beat for #1
- #2  10billiontokens 1,418.072 µs   (3.4x cliff from #1 => #1 has a structurally better approach)
- #5  badelsteinlelbach 1,767.456 µs <-- our account (other agent) MUST BEAT
- baseline geqrf local geomean: 131,465 µs (serial cuSOLVER loop; ranks ~last)

## Submissions
- 2026-06-22 09:20 | commit 2b40ba3c (BASELINE geqrf) | id 827292 | ACCEPTED 22/22 | local 131465 us

## Re-root 2026-06-22 09:28
- Pulled origin/main: 1 new commit ceb573f5 "Update submit.sh to report failures properly" (FAILURE-FIRST verdict parsing).
- Verified measured-path files (submission/eval/reference/task/guards/build/validate/benchmark) byte-identical 2b40ba3c vs ceb573f5 -> baseline metric 131465 us valid at new HEAD.
- Re-logged baseline at ceb573f5; re-synced 3 worker worktrees to ceb573f5. RUN BASELINE = ceb573f5.
- NOTE: ranked B=640 benchmark can REJECT a kernel that passes the 22 small test shapes ("R-Q.T@A too large"). Workers must validate on large shapes, not just test shapes.

## Submissions (run, after baseline)
- 12:20 | W2 29a660d9 (fused? no — iter3) local 4114 | id 827599 ACCEPTED 22/22 | board geomean 4184us
- 12:31 | W2 28c45c2f (two-kernel split, tf32x3) local 3627 | id ~827xxx ACCEPTED 22/22 | run's leaderboard-VALID best
- NOTE: board still shows account #3=1761us = OTHER AGENTS' kernel (operator confirmed). My run best 3627us has NOT beaten it yet.
- CONSTRAINT: W2 fused-trailing 1cc13350 (local 3452us) leaderboard-REJECTED "work on another stream" -> two-kernel split required.
- TARGET: board geomean < 1418us to win #1 (gap 2.4-2.95x). Panel factorization is the universal wall across all 3 workers.

## Continued submissions (n=512/1024 focus phase)
- W0 e721f880 3317us local / 3239us remote (qr_tiny+batch-major) ACCEPTED id ~828xxx
- W2 0a9b49da 3178us local (on-chip-W fused trailing, no YT round-trip) ACCEPTED 22/22 — confirms on-chip-W trailing LEGAL
- STRATEGY: geomean is 63% n512+n1024. Workers partitioned: W0=n1024 (B=60 under-sat), W1=n512 panel, W2=n512 trailing.
- W0 PROVED named-barrier fused/multi-CTA is leaderboard-LEGAL (id 828414).

## W2 brief-16 COMBINE (n=1024 BM=64 graft onto efdfb81c)
- 01:58 | W2 88088f40 (by-N fused-trailing: n=1024 BM=64, n=512 BM=32; grafted onto efdfb81c base) | id 829055 | ACCEPTED 22/22 | local geomean 3040.5us (A/B 3-round mean: base 3059.7 vs graft 3040.2, -0.63%) | board rank #4 @ 1740.075us
- PROVENANCE: _trailing_fused_kernel + _panel_factor_kernel byte-identical 397dbf82 vs efdfb81c -> BM=64 win transfers; W0 n2048(split)/n4096(2level) wins untouched (BLK==32-gated dispatch). num_stages left at Triton DEFAULT everywhere.
- gap to #1 (10billiontokens s14.py 1418us): ~1.23x (22.7%).

## W1 brief-11 COMBINE (panel ALU trims grafted onto W0 best 07aed12f)
- 03:14 | W1 39f6328d (graft: drop dead v=where(active,v,0) + dead has_refl=False fallback from BOTH panels, onto W0 07aed12f clone-fusion+n4096/n2048/n1024 1-pass-tf32 base) | id 829218 | ACCEPTED 22/22 | local geomean 2839us
- A/B interleaved by commit (G1 2838.6, P1 2927.3, G2 2839.5, P2 2930.4): graft 2839.1us vs W0 parent 2928.8us = -3.06% geomean; EVERY shape improved (n176 +5.3%, n2048 +4.1%, n512 +3.5%, n4096 +2.45%, n32 +0.08%), ZERO regressions.
- PROVENANCE: trims verified DISJOINT — still present at L969/970 & L1423/1424 in 07aed12f (W0's base never applied them). FP-EXACT (xk pre-masked by active; tail_n2==0 => v==e_k bit-exact). Validate: 22/22 incl ill-cond rankdef/clustered/nearrank + test.18 n=4096 upper scaled_residual=0 (W0 1-pass-tf32 n4096 path CORRECT). Invariance guard CLEAN. kernelguard CLEAN.
- Beats prior account-best W2 88088f40 (local 3040us) by -6.6% local. Board re-rank pending at log time (was #4 @ 1740us; projected ~1625us board via 0.572 local->board ratio -> likely #3, gap to #1 narrows ~1.23x->~1.15x).

## W1 brief-12 (dead-ALU extension, more panel selects removed)
- 04:09 | W1 1410c063 (5 FP-exact dead-ALU removals stacked on 39f6328d) | id 829324 | ACCEPTED 22/22 | local geomean ~2776us (down from 2839)
- Removals (all PROVABLY FP-exact, re-validated 22+ill-cond+invariance after each):
  1. _panel_factor_kernel: drop xk active-mask `where((rows>=k)&row_valid, xk, 0)` — xk is col k of panel, already 0 at rows>=pheight (masked load + invariant), rows<k never read. -1.29% (A/B confirmed: parent 2839.97 vs 2803.27).
  2. _panel_factor2_kernel: same xk-mask removal (n=4096 sub-panels). shape6 36201->35211us (-2.7%).
  3. _panel_factor_kernel: drop z-mask `where(cols<k,w,0)` AND Tcol-mask `where(cols<k,Tcol,0)` in WY-T column build — Tmat strict-lower=0 + cols>=k still initial 0 => both redundant, use w directly. -0.67%.
  4. _panel_factor2_kernel: same z/Tcol T-column removal. neutral (n=4096 diluted).
  5. both panels: drop beta_safe `where(beta==0,1,beta)` — beta==0 iff has_refl==False where the divide is discarded. scalar, neutral.
- NET: parent 2839.5 -> ~2776us local (-2.2%), the two structural wins (#1 panel xk-mask, #3 T-column masks) carry it.
- FINDINGS for manager: trailing kernels (_trailing_fused, _trailing_YT/apply, YT2 split, cross_T) have NO removable FP-exact dead tl.where — they are clean tensor-core dots; the fused kernel's double-V-read is structurally required by chunking (resident V/A spills, rejected). qr_tiny is hand-CUDA, clean. The panel was the only kernel class with removable dead ALU, and it is now exhausted. Board still ~1.2x from #1 (10billiontokens 1418us); incremental, not structural — next lever must be a faster PANEL algorithm, not more ALU trims.

## Late-stage progression (precision ladder + dead-code + combines)
- 39f6328d 2839us (W1 combine: W0 clone-fusion+1pass-tf32 + W1 panel-ALU-trims) ACCEPTED id 829218
- 7b685a1c 2741us (W0 brief-15: 2-pass tf32x2 for n=1024 trailing; 1-pass REJECTED by gate, correct) ACCEPTED
- Local progression: 3138->3059->2926->2839->2776->2741us (~48x baseline, 1.93x from #1's 1418us)
- KEY METHOD: empirically-gated precision ladder (1-pass n>=2048, 2-pass tf32x2 n=1024, testing 2-pass n=512) + FP-exact dead-code removal + cross-branch combines.
- Standings 2026-06-23 04:18: #1 10billiontokens 1418, #2 dhu.randhar 1505, #3 nikhilbarhate 1729, #4 Olek 1730, #5 badelsteinlelbach 1740 (account best across all submissions; competitive race, others improving).

## W1 brief-12 FINAL A/B (HEAD d12b4537 vs parent 39f6328d), interleaved by commit
- HEAD G1 2775.42, P1 2839.91, HEAD G2 2776.85, P2 2840.13 => HEAD mean 2776.13 vs parent mean 2840.02 = -2.25% geomean (HEAD spread 1.4us, parent 0.2us; tight).
- Final HEAD d12b4537 (7 FP-exact removals) is perf-identical to the SUBMITTED 1410c063 (~2776, id 829324 ACCEPTED 22/22): the iter5 (YT2 Tm-mask) + iter6 (cross_T T-masks) removals are n=4096-only and benchmark-neutral, so the accepted board score applies to d12b4537 too. The two STRUCTURAL wins (iter0 panel xk-mask -1.29% A/B-confirmed; iter2 panel z/Tcol-mask -0.67%) carry the gain; iters 1/3/4/5/6 are FP-exact correctness-honest removals that are diluted/scalar/n4096-neutral.

## W1 brief-13 COMBINE (7 dead-ALU removals grafted onto W0's 2-pass-tf32x2 base)
- 04:52 | W1 6957b600 (graft: my 7 FP-exact dead-ALU removals from d12b4537 re-applied onto W0's 2-pass-tf32x2-for-n1024 base 7b685a1c) | id 829410 | ACCEPTED 22/22 | local geomean 2679.65us (NEW account-best, down from 2741us, -2.2%)
- METHOD: `git cherry-pick 5c4a8db1 57835da0 47ce086d 71deca91 1410c063 4afc8840 d12b4537` onto 7b685a1c applied ALL 7 with NO conflict — per-commit 3-way merge auto-resolved the "1 overlap region" the manager flagged (cherry-pick applies each small commit's local context independently, unlike a flat branch merge).
- PROVED exact union (not approximate): (combined-vs-7b685a1c diff) byte-identical to my pure 7-removal diff 39f6328d..d12b4537; (combined-vs-d12b4537 diff) byte-identical to W0's pure 2-pass diff 39f6328d..7b685a1c. No `+tl.where` re-added (no removed select came back); no `-(tf32|pass)` (no W0 precision undone).
- VALIDATE: 22/22 + kernelguard CLEAN + diff-correctness CLEAN + invariance CLEAN. Critically W0's precision branches survived the graft: n=1024 rankdef (t13/14/15) PASS, n=2048 rankdef (t17) PASS, n=4096 upper (t18 scaled_residual=0) PASS — confirms 2-pass(n1024) + 1-pass(n>=2048) paths intact.
- A/B interleaved by commit (util=0% throughout, 2 runs each): combined 2682.9/2676.4 (mean 2679.65) vs W0base 7b685a1c 2740.0/2739.8 (mean 2739.88) = -2.20% geomean; best-of-2 geomean -2.29%. ZERO shapes regressed — every shape faster or within noise: n176 -4.04%, n352 -3.46%, n4096 -4.12%, n512 -1.91to-2.27%, n1024 -1.92to-2.27%, n2048 -1.17%, n32 -0.05% (noise).
- PROVENANCE of stacking: my removals are all in panel/helper kernels (xk active-mask both panels = the -1.96% load-bearing pair; WY-T z/Tcol masks both panels; beta_safe both panels; YT2 Tm kvalid-mask both inner-trailing; cross_T T0/T1/T01 real-masks) — DISJOINT from W0's n=1024-trailing precision change. Both off common base 39f6328d (verified merge-base).
- Board rank #5/140 @ 1740.075us (gap +22.7% to #1 10billiontokens s14.py 1418us) at submit time — reflects PRIOR best 7b685a1c; id 829410's lower score (2679 local) pending board re-rank, projects ~1573-1700us board via local->board ratio (likely holds #5, possibly nudges #4 Olek/nikhilbarhate @ ~1725us).

## Structural-win phase (two-level panel + combines + ILP)
- W1 brief-15: n=512 TWO-LEVEL IB16/NB32 panel decoupling (gate passed: wide-apply un-spilled 154r/6blk) -> 2646us ACCEPTED id 829776. Broke the 2677 plateau (structural, not micro).
- W2 brief-23: recovered STRANDED n1024 BM=64 onto two-level -> 067983d6=2625.8us ACCEPTED. (was missing from best since W1 forked pre-BM64.)
- W0 brief-21: co-batch FALSIFIED (W-reduction per-warp loop-carried-dep-bound, even WMMA fails) BUT intra-warp ILP (2 accum + num_stages=3 + wider BNc) won af9917fd=2629us (-4/-5% W-reduction), DISJOINT -> W0 brief-22 stacking it onto leader.
- Progression: 3138->...->2646->2625.8us (~50.2x baseline). Gap to #1 (1418) = 1.85x STRUCTURAL.
- Standings 2026-06-23 ~08:00: #1 10billiontokens 1418, #2 dhu.randhar 1439 (qr_zsample.py, improving!), #5 badelsteinlelbach 1736 (account best). Field compressing.
