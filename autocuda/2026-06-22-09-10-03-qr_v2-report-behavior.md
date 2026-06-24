# Behavior Report — `2026-06-22-09-10-03-qr_v2`

*Single experiment, 4-worker `optimize-tree` run on one B200. 172 brief-branches
across workers W0/W1/W2/W3, 2026-06-22 09:16 → 2026-06-24 17:38 (≈56.4 h wall).*

## Executive Summary

This was a long, productive, and — given the conditions — strikingly reliable run.
Across 56.4 h of walltime the fleet logged 332 trials over 172 brief-branches
(83 of which produced ≥1 trial; the other 89 were write-only or instantly-abandoned
briefs). Sustained throughput was modest at ≈4.5 trials per active worker-hour, but
that number is set almost entirely by the cost of a single CUDA iteration on this
problem (build + validate-on-22-ranked-shapes + benchmark + nsys/ncu profile), not by
agent stalling. The headline reliability story is that the run survived **≈71 API-drop
worker deaths and ≈33 manager respawns** with zero lost work and **zero genuine
early-termination** — the manager recovered every crashed worker losslessly from its
branch + log, and a mid-run torch 2.11→2.12.1+cu130 venv upgrade plus an end-of-run
spin-down were both executed as clean, operator-directed pause/resume windows.

**Key takeaways**

- **The fleet is remarkably balanced, not bottlenecked on one worker.** Per-worker
  throughput is 4.67 / 4.47 / 4.55 / 3.89 trials/active-h for W0/W1/W2/W3 — every
  worker is within **0.86–1.04× of the fleet median**. There is *no* 2× outlier; the
  large per-*brief* variance (CoV 3.49×) comes from brief *length*, not worker skill.
- **The 4th worker arrived 39 h late.** W3 did not exist until 2026-06-24 00:27 — the
  run was a **3-worker fleet for its first ~39 h** and only reached 4 workers for the
  final ~17 h. Judged against the *intended* size per era, the fleet was at full
  strength **82.6 % of the 3-worker era and 79.2 % of the 4-worker era**.
- **Lock contention on the single GPU is a real, double-digit tax.** Only one of four
  workers can hold the exclusive `gpu-0` lock (bench/profile/validate) at a time;
  captured lock-wait totals **≈9.5 h ≈ 17 % of fleet walltime** (a conservative floor —
  truncated logs imply the true figure is higher), with the worst single wait
  **12.8 min**. This, not agent idleness, is the dominant structural drag on a 4-on-1
  fleet.
- **Worker deaths were frequent but invisible to the result.** ≈71 distinct
  API-mid-response deaths (W0:27, W1:21, W2:21, W3:2), heaviest on 06-22/06-23,
  each recovered by the manager re-reading the branch and re-issuing the in-flight
  brief. Several "deaths" were stale duplicate notifications the manager correctly
  declined to act on.
- **No agent tried to quit the run.** 34 messages are workers *correctly* declaring one
  brief exhausted and handing back to the manager (normal). The only halt-and-wait was
  W2 on brief-69 correctly honoring a *direct user freeze* over a conflicting
  coordinator instruction — the right call, not a give-up.

**One-line:** 172 branches · 332 trials · 56.4 h wall · mean ≈4.5 trials/active-worker-h
(worst sustained worker 3.89) · reliability verdict: **high** — frequent infrastructure
failures, zero work lost, zero spurious termination.

## Fleet Behavior

### Worker Concurrency Over Time

Active intervals were built by pairing each worker's `brief_start` with its next
`brief_stop`. Bucketed into 5-minute windows over the 56.4 h run:

| Active workers | Windows | Walltime | Share |
|---|---|---|---|
| 0 (fully idle) | 13 | 1.08 h | 1.9 % |
| 1 | 9 | 0.75 h | 1.3 % |
| 2 | 68 | 5.67 h | 10.0 % |
| 3 | 423 | 35.25 h | 62.5 % |
| 4 (peak) | 164 | 13.67 h | 24.2 % |

Peak concurrency was **4**, but reached only in the final ~17 h: **worker W3's first
activity is 2026-06-24 00:27**, ~39 h into the run. So the raw "73.9 % under strength
vs. 4" figure is misleading — for most of the run the *target* was 3, not 4. Splitting
by era against the intended size:

- **3-worker era (start → 06-24 00:27, 39.25 h):** at full strength (≥3) **82.6 %**,
  under-strength **14.6 %**, fully idle **2.8 %**.
- **4-worker era (06-24 00:27 → end, 17.25 h):** at full strength (≥4) **79.2 %**,
  under-strength **20.8 %**, fully idle **0 %**.

The longest fully-idle stretch is small (the idle windows are scattered single buckets,
mostly clustered around the late-run maintenance/spin-down). The under-strength time is
the aggregate cost of the death/respawn churn — each death drops the fleet by one worker
until the manager notices and respawns (handoff median 82 s, below). The fleet never
collapsed; it ran one-worker-down briefly and often, not many-down for long.

### Brief-Cycle Timing (manager → worker handoff)

For each of the 206 manager-log brief rows, the handoff latency is the gap from the
manager writing the brief to the matching worker emitting `brief_start` for that
`brief_id`. Over the 170 briefs with a clean same-id match:

- **median 82 s, p95 616 s (~10 min), max 3103 s (51.7 min).**
- **16 briefs exceeded 5 min.** The 36 briefs without a clean same-id match are exactly
  the respawned/duplicated briefs (a crashed instance's brief_start does not line up
  with the re-issued row).

The slowest handoffs cluster on 06-23 when death churn peaked — e.g. W1 brief-23
(51.7 min, 06-23 10:10), W0 brief-19 (39.0 min, 06-23 05:24), W2 brief-69 (29.1 min,
06-24 14:50, during the late-run freeze). These long handoffs are the manager waiting on
a dead-worker respawn or on a busy GPU, not the worker ignoring a brief. *(Note: the
gap from `brief_start` to the worker's first logged trial is a separate ~20 min median —
that is one build+validate+benchmark cycle, i.e. the cost of doing the work, not a
handoff problem.)*

### Lock Contention

`autocuda run slice` (build, **shared** `build` lock) and `autocuda run exclusive`
(bench/profile/validate, **exclusive** `gpu-0` lock) emit paired acquire/release lines
in the session logs. De-duplicated across replayed sessions (871 unique acquisitions:
231 build/shared, 640 gpu-0/exclusive — all GPU locks are `gpu-0`, a single B200):

| Lock kind | Acquisitions | Total wait | Median wait | Max wait | Total hold (`ran`) |
|---|---|---|---|---|---|
| `build` / shared (builds) | 231 | 2.61 h | 8.3 s | 447.5 s | 0.63 h |
| `gpu-0` / exclusive (bench/profile/validate) | 640 | 6.86 h | 0.0 s | **768.0 s** | 15.41 h |
| **Total** | 871 | **9.47 h** | — | 768.0 s | 16.0 h |

The GPU was held under exclusive lock **≈15.4 h ≈ 27 % of walltime** — that is the
structural source of contention: with 4 workers and 1 GPU, three workers are blocked
whenever one benchmarks or profiles. Most exclusive acquisitions got in instantly
(median 0 s), so the **9.47 h of wait is a long-tail phenomenon** — when several workers
finished editing at once they queued, the worst single wait being 12.8 min (06-23
18:50). Aggregate captured wait is **≈17 % of the 56.4 h fleet walltime**, and because
the acquire line is the one most often truncated by "Output too large" the true figure
is higher (release lines, which survive, are ~2× more numerous). **Lock contention is a
meaningful fraction of fleet time and the clearest scaling limit of running 4 workers on
1 GPU.** Per-worker attribution was not possible — the candidate session logs are
manager/shared and all reference all four workers.

### Per-Worker Reliability

| Worker | Briefs (non-empty) | Trials | Active h | Trials/active-h | vs median | Deaths |
|---|---|---|---|---|---|---|
| W0 | 47 (27) | 117 | 25.0 | 4.67 | 1.04× | 27 |
| W1 | 50 (24) | 100 | 22.4 | 4.47 | 0.99× | 21 |
| W2 | 54 (24) | 88 | 19.3 | 4.55 | 1.01× | 21 |
| W3 | 21 (8) | 27 | 6.9 | 3.89 | 0.86× | 2 |

**No worker is a 2× outlier in either direction.** W3's slightly lower 3.89 is the
late-spawn settling in over a shorter window, not a defect. W0 took the most deaths (27)
yet still posted the *highest* throughput — strong evidence the respawn recovery is
genuinely lossless. The fleet is well-load-balanced; differences between workers are
noise, not skill or reliability gaps.

## Throughput

| Branch | Trials | Duration | Trials/hour | Improvement % |
|---|---|---|---|---|
| `worker-0-brief-37` | 3 | 25s | 293.39 | +0.00 |
| `worker-2-brief-11` | 2 | 1m36s | 37.39 | +0.00 |
| `worker-0-brief-51` | 2 | 2m26s | 24.67 | +0.00 |
| `worker-1-brief-29` | 2 | 2m55s | 20.62 | +0.00 |
| `worker-1-brief-14` | 2 | 3m34s | 16.83 | +0.00 |
| `worker-2-brief-63` | 2 | 4m15s | 14.10 | +0.00 |
| `worker-1-brief-25` | 2 | 4m26s | 13.53 | +0.00 |
| `worker-0-brief-49` | 2 | 4m48s | 12.49 | +0.00 |
| `worker-2-brief-19` | 2 | 4m52s | 12.33 | +0.00 |
| `worker-1-brief-27` | 2 | 5m22s | 11.18 | +0.00 |
| `worker-3-brief-14` | 2 | 6m0s | 10.01 | +0.00 |
| `worker-0-brief-12` | 5 | 26m17s | 9.13 | +0.00 |
| `worker-0-brief-48` | 5 | 26m25s | 9.08 | +0.00 |
| `worker-2-brief-68` | 3 | 13m33s | 8.86 | +0.00 |
| `worker-2-brief-3` | 5 | 27m26s | 8.75 | +0.00 |
| `worker-2-brief-22` | 2 | 7m40s | 7.83 | +0.00 |
| `worker-0-brief-50` | 5 | 32m34s | 7.37 | +0.00 |
| `worker-3-brief-17` | 3 | 17m3s | 7.04 | +0.00 |
| `worker-0-brief-32` | 2 | 8m37s | 6.97 | +0.00 |
| `worker-2-brief-37` | 4 | 26m1s | 6.92 | +0.00 |
| `worker-1-brief-12` | 7 | 52m57s | 6.80 | +0.00 |
| `worker-3-brief-19` | 5 | 36m5s | 6.65 | +0.00 |
| `worker-2-brief-23` | 2 | 9m25s | 6.37 | +0.00 |
| `worker-0-brief-21` | 2 | 10m0s | 6.00 | +0.00 |
| `worker-2-brief-5` | 4 | 32m12s | 5.59 | +0.00 |
| `worker-3-brief-11` | 2 | 10m51s | 5.53 | +0.00 |
| `worker-0-brief-45` | 2 | 11m8s | 5.39 | +0.00 |
| `worker-0-brief-22` | 3 | 22m30s | 5.33 | +0.00 |
| `worker-2-brief-15` | 6 | 56m46s | 5.29 | +0.00 |
| `worker-2-brief-4` | 5 | 45m33s | 5.27 | +0.00 |
| `worker-0-brief-4` | 11 | 2.0h | 4.93 | +0.00 |
| `worker-2-brief-9` | 4 | 37m5s | 4.85 | +0.00 |
| `worker-2-brief-41` | 5 | 50m25s | 4.76 | +0.00 |
| `worker-0-brief-0` | 7 | 1.3h | 4.71 | +0.00 |
| `worker-1-brief-42` | 4 | 38m39s | 4.66 | +0.00 |
| `worker-0-brief-53` | 2 | 13m10s | 4.56 | +0.00 |
| `worker-1-brief-5` | 5 | 54m41s | 4.39 | +0.00 |
| `worker-2-brief-40` | 2 | 14m2s | 4.28 | +0.00 |
| `worker-3-brief-1` | 2 | 14m13s | 4.22 | +0.00 |
| `worker-0-brief-11` | 5 | 57m1s | 4.21 | +0.00 |
| `worker-1-brief-0` | 11 | 2.4h | 4.09 | +0.00 |
| `worker-1-brief-2` | 9 | 2.0h | 3.95 | +0.00 |
| `worker-0-brief-7` | 5 | 1.0h | 3.94 | +0.00 |
| `worker-1-brief-39` | 3 | 30m53s | 3.89 | +0.00 |
| `worker-0-brief-56` | 8 | 1.8h | 3.83 | +0.00 |
| `worker-1-brief-15` | 6 | 1.4h | 3.69 | +0.00 |
| `worker-1-brief-38` | 4 | 48m56s | 3.68 | +0.00 |
| `worker-2-brief-27` | 2 | 17m5s | 3.51 | +0.00 |
| `worker-1-brief-10` | 3 | 34m37s | 3.47 | +0.00 |
| `worker-0-brief-14` | 5 | 1.2h | 3.46 | +0.00 |
| `worker-1-brief-17` | 4 | 54m19s | 3.31 | +0.00 |
| `worker-1-brief-40` | 2 | 18m6s | 3.31 | +0.00 |
| `worker-0-brief-1` | 8 | 2.2h | 3.19 | +0.00 |
| `worker-0-brief-43` | 2 | 18m47s | 3.19 | +0.00 |
| `worker-1-brief-3` | 3 | 37m53s | 3.17 | +0.00 |
| `worker-1-brief-45` | 2 | 18m56s | 3.17 | +0.00 |
| `worker-1-brief-1` | 9 | 2.6h | 3.10 | +0.00 |
| `worker-1-brief-31` | 5 | 1.3h | 3.10 | +0.00 |
| `worker-1-brief-52` | 2 | 20m12s | 2.97 | +0.00 |
| `worker-2-brief-0` | 7 | 2.0h | 2.96 | +0.00 |
| `worker-2-brief-10` | 4 | 1.0h | 2.88 | +0.00 |
| `worker-2-brief-18` | 6 | 1.7h | 2.88 | +0.00 |
| `worker-0-brief-3` | 4 | 1.1h | 2.77 | +0.00 |
| `worker-0-brief-15` | 2 | 22m47s | 2.63 | +0.00 |
| `worker-0-brief-17` | 4 | 1.2h | 2.59 | +0.00 |
| `worker-1-brief-43` | 2 | 23m36s | 2.54 | +0.00 |
| `worker-2-brief-1` | 4 | 1.3h | 2.31 | +0.00 |
| `worker-1-brief-37` | 4 | 1.4h | 2.21 | +0.00 |
| `worker-3-brief-20` | 2 | 27m7s | 2.21 | +0.00 |
| `worker-0-brief-54` | 2 | 28m18s | 2.12 | +0.00 |
| `worker-0-brief-44` | 13 | 5.7h | 2.10 | +0.00 |
| `worker-2-brief-52` | 8 | 3.4h | 2.06 | +0.00 |
| `worker-0-brief-9` | 3 | 1.0h | 1.99 | +0.00 |
| `worker-0-brief-46` | 3 | 1.1h | 1.82 | +0.00 |
| `worker-2-brief-69` | 2 | 32m53s | 1.82 | +0.00 |
| `worker-2-brief-45` | 2 | 33m22s | 1.80 | +0.00 |
| `worker-3-brief-5` | 3 | 1.1h | 1.80 | +0.00 |
| `worker-3-brief-0` | 8 | 4.0h | 1.76 | +0.00 |
| `worker-0-brief-34` | 2 | 43m56s | 1.37 | +0.00 |
| `worker-1-brief-26` | 5 | 3.1h | 1.31 | +0.00 |
| `worker-1-brief-4` | 2 | 46m32s | 1.29 | +0.00 |
| `worker-2-brief-21` | 2 | 54m4s | 1.11 | +0.00 |
| `worker-2-brief-39` | 3 | 2.0h | 1.00 | +0.00 |

*83 branches with ≥1 trial shown (sorted by trials/hour). The remaining 89 branches
logged 0 trials — briefs the manager wrote but a worker either never picked up
(write-only, often superseded by a redirect) or abandoned before the first build.
`Improvement %` is uniformly +0.00: tree-worker logs do not carry `pct_change`, so this
column is structurally blank for every tree branch (the actual speedups — geomean
131465 µs → 2040.9 µs, ~64× — live in the optimization report, not here).*

Single-experiment run, so there is no cross-*experiment* mean/stddev row. The
cross-*branch* spread is the relevant variance signal: over the 83 active branches,
trials/hour is **mean 9.13, median 3.95, stddev 31.86 (CoV 3.49×)**. That spread is an
artifact of brief duration — a 25 s probe (brief-37) and a 5.7 h benchmark marathon
(brief-44, 13 trials) both appear, and short briefs inflate the per-hour rate without
doing more work. Aggregated to the worker level the variance nearly vanishes (4.67 /
4.47 / 4.55 / 3.89), which is the honest read on fleet consistency.

## Gaps And Stalls

66 of 172 branches recorded a gap (a >10 min quiet transition between trials), 154 gaps
total. These are overwhelmingly **benchmark/profile duration and GPU-lock queueing**, not
agent silence — the long-running shapes (n=2048/4096 and the B=640 ranked validation
sweep) take tens of minutes per benchmark, and only one worker can hold `gpu-0` at a
time. The notable cases:

- **`worker-0-brief-44` — 7 gaps, longest 142 min (8524 s), 13 trials over 5.7 h.** The
  single largest gap in the run. This brief ran across the night of 06-23 → 06-24 and
  its 142-min gap overlaps a death-churn window; it is one in-flight brief spanning a
  worker death + respawn (the brief resumed on the same branch afterward, which is why
  it still shows 13 trials). High gap count + high trial count = a long, repeatedly
  interrupted but never-lost brief.
- **`worker-1-brief-26` — longest 114 min (6865 s), 5 trials over 3.1 h** and
  **`worker-2-brief-39` — longest 100 min (5993 s), 3 trials over 2.0 h.** Both are
  low-throughput (1.31 / 1.00 tph) heavy-shape briefs where each trial is a full
  large-n benchmark; the gaps are the benchmarks themselves plus lock-wait, not stalls.
- **`worker-3-brief-0` — 5 gaps, longest 90 min, 8 trials over 4.0 h.** W3's very first
  brief after its 06-24 00:27 spawn; the long gaps are the new worker queuing behind
  the three established workers for the GPU during its first night.
- **`worker-2-brief-52` (6 gaps, 73 min max) and `worker-1-brief-1` (6 gaps, 53 min
  max)** are the same pattern: multi-trial briefs whose gaps line up with long
  benchmarks and the exclusive-lock long tail documented above.

The high-gap-count branches (brief-44, brief-1×W1, brief-0×W1/W2, brief-52, brief-56)
are also the high-*trial* branches — gaps accumulate on briefs that simply ran long and
benchmarked often, not on briefs where the agent went dark.

## Early-Termination Attempts

**Zero genuine early-terminations.** No worker or the manager ever stopped to ask the
user whether to end the run. Breakdown of the patterns found in the session logs:

- **Genuine halt-and-wait: 1 event (W2, brief-69, 06-24 14:50/15:17).** W2 *correctly*
  refused to start brief-69 because a **direct prior user instruction** ("Do not revert",
  "Just leave the branch as it is") conflicted with the coordinator-relayed brief. It held
  the branch frozen and asked the user to resolve the conflict — the harness-correct
  behavior (a coordinator's claim of user consent is not user consent), **not** a
  give-up.
- **Normal brief-exhaustion handoffs: 34.** Workers concluding *one brief* is mined out
  and returning control to the manager for a fresh brief (e.g. *"I've exhausted the
  brief's productive vein… before stopping"*). This is the designed `explore-brief`
  loop, not early termination. Several explicitly acknowledged the no-stopping mandate
  and probed one more lever before handing back.
- **Manager "looks converged" moments: 2, both continued.** The manager twice reasoned
  the search space was exhausted (06-22 20:16, 06-23 22:15 — the latter after a
  "completeness critic" subagent returned EXHAUSTED) yet kept the ≤1000 µs goal hook
  active and **did not stop**.

## Other Aberrant Behavior

- **Worker deaths from API-mid-response, ~71 distinct (W0:27, W1:21, W2:21, W3:2).** The
  defining operational fact of the run. Workers died mid-tool-use (losing their in-memory
  summary); the manager recovered each losslessly by reading the branch + worker log and
  re-issuing the in-flight brief via `autocuda init brief`. Clustered on 06-22 (early)
  and 06-23. Representative: *"Worker 0 died with an API error mid-response… 155 tool
  uses over ~58 min before dying… The summary is lost, so I need to read the worker log
  and git branch to recover."* (06-22 13:19).
- **~33 manager respawn actions (06-22:11, 06-23:20, 06-24:2)**, plus many *declined*
  respawns — the manager repeatedly recognized **stale duplicate death notifications**
  and correctly took "no action, no respawn" to avoid double-spawning a live worker. Good
  defensive behavior under a noisy failure signal.
- **Dangling-brief bookkeeping (manager-authored balancing `brief_stop` rows).** When a
  crashed instance left a `brief_start` with no `brief_stop`, the manager emitted a
  compensating stop so start/stop counts reconciled — e.g. W0 brief-4 (06-22 19:14). This
  is why worker logs show *more* stops than starts (W0 57 vs 54, W2 60 vs 55, etc.); it
  is correct bookkeeping, not a logging error, but it does mean raw start/stop counts
  must be paired carefully.
- **Duplicate brief_ids per worker (respawn fingerprints):** W0 restarted brief-44 **5×**,
  brief-4/23/48 2× each; W2 brief-18/39 2×; W3 brief-0/19 2×. Each duplicate is a death
  + respawn of the same in-flight brief, consistent with the death log.
- **Late fleet expansion.** W3 was added ~39 h in (first activity 06-24 00:27) rather
  than at launch — the run effectively changed its target worker count mid-stream. Worth
  flagging for any apples-to-apples throughput comparison with other runs.
- **89 zero-trial briefs.** Slightly over half of all briefs produced no trial. Most are
  manager briefs superseded by a redirect before a worker engaged, or picked up and
  abandoned at build time; a few coincide with a death between brief-write and first
  build. Not a correctness problem, but it inflates the branch count well above the
  number of *substantive* explorations (83).

## Analysis

The run's behavior is best summarized as **a slow-but-steady, infrastructure-stressed
fleet that never actually lost ground.** Sustained throughput sits at ≈4.5 trials per
active worker-hour and is essentially flat across W0/W1/W2 — the differences are noise.
That rate is governed by the per-iteration cost of this specific problem: every kept or
discarded trial pays for a build, a validate over 22 ranked/ill-conditioned shapes, a
benchmark, and frequently an nsys/ncu profile, and the heavy shapes (n=2048/4096, B=640)
make a single benchmark a tens-of-minutes affair. The per-*brief* trials/hour CoV of
3.49× looks alarming until you see it is entirely a function of brief *length* —
25-second probes and 5.7-hour marathons share one column — and collapses to near-zero
once aggregated per worker. The variance is *not* driven by an outlier worker.

**Where did the time actually go?** Two places, both structural rather than behavioral.
First, the **exclusive GPU lock**: the single B200 was held under exclusive lock ≈15.4 h
(≈27 % of walltime) and the fleet paid ≈9.5 h (≈17 %, a floor) waiting to acquire it,
with a 12.8-min worst case. With four workers and one GPU this is unavoidable — three
workers idle whenever one benchmarks — and it is the clearest reason the gaps in
`Gaps And Stalls` line up with the heavy-shape, high-trial briefs (brief-44, brief-26,
brief-39, brief-52). Second, the **death/respawn churn**: ≈71 deaths cost the fleet its
under-strength time (14.6 % of the 3-worker era, 20.8 % of the 4-worker era) and stretched
the slowest manager→worker handoffs to 30–52 min on 06-23 when the manager was waiting on
respawns. Crucially, neither bottleneck is the agent stalling — the median handoff is
82 s and the fully-idle time is <2 % of the run.

The reliability engineering is the standout. A 56-hour run that absorbs ≈71 API-drop
deaths, a torch major-version venv swap, and a deliberate spin-down — all without losing
a single brief's work and without one spurious "should I stop?" — is the behavior you
want from an autonomous harness. The manager's lossless recovery (read branch + log,
re-issue brief), its restraint on stale death notifications, and its correct handling of
the venv window (drain → snapshot for rollback → upgrade on idle GPU → verify 22/22 →
resume) all held up. The one halt (W2/brief-69) was the agent *correctly* prioritizing a
direct user instruction over a conflicting relay — a feature, not a fault.

**Recommendations.**
1. **Attack the GPU lock, not the agent.** ≈17–27 % of walltime is GPU serialization on
   a 4-on-1 fleet. The highest-leverage next change is either fewer workers per GPU
   (3-on-1 showed 82.6 % full-strength vs. the 4-on-1 era's 79.2 %, at lower contention)
   or shorter benchmark/profile holds (cache profiles, profile fewer shapes per trial,
   or gate ncu behind a kept-result filter) so the exclusive lock is released sooner.
2. **Harden against API-drop deaths.** ≈71 deaths and ~33 respawns dominated the
   operational noise and pushed worst-case handoffs past 50 min. A faster death detector
   (tighter respawn polling) or automatic mid-response retry would reclaim most of the
   under-strength time without changing the optimization logic.
3. **Launch the full worker count from the start.** W3 arriving 39 h late means ~39 h of
   3-worker throughput where 4 were intended; spawning all workers at launch (and keeping
   the no-op-brief rate down — 89 of 172 briefs produced nothing) would raise effective
   fleet utilization on the next run.
