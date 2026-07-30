# Behavior Report: autocuda optimize-tree 2026-07-19-01-44-06-cholesky-resumed-local-unmerged

## Executive Summary

At the 2026-07-20 21:01:15 UTC snapshot, 93 branches had produced 1,581 structured trials at 8.43 trials per summed branch-active hour (36.77 trials per fleet wall-clock hour). The run was productive but behavior was uneven: 88.8% of trials succeeded, while 134 inter-trial gaps exceeded ten minutes and branch rates ranged from 0.00 to 23.62 trials/hour. This is one still-running experiment, so branch spread is descriptive and cannot establish cross-run variance.

**Key takeaways**

- Five briefs occupied 466/517 five-minute windows (90.1%); 36 windows were under-strength, while two stale open briefs made 15 windows appear to have seven workers.
- Context-role drift was the top issue: briefs 58 and 63 stopped optimizing after compaction, and briefs 71 and 81 temporarily entered manager-style wait loops.
- No unauthorized early-termination request or convergence claim was found; normal completion messages followed logged stops or were explicit compaction checkpoints.
- All 4,697 paired lock calls were shared `build` calls with 0.000 seconds of wait; no exclusive per-GPU event was recorded, so GPU contention is not measurable from this run.

**Snapshot totals:** 93 branches (92 with trials), 1,581 trials, 43.00 fleet hours, 187.50 summed measured branch hours, 8.43 aggregate trials/hour, 0.00 worst reported branch rate, and a mixed reliability verdict driven by compaction-role failures rather than lock contention.

## Fleet Behavior (Optimize-Tree Runs Only)

### Concurrency Over Time

The configured target was five workers. At five-minute midpoints across 517 windows, exactly five briefs were open in 466 windows (38h45m, 90.1%), fewer than five in 36 windows (3h00m, 7.0%), and the logged peak of seven in 15 windows (1h15m, 2.9%). The longest under-strength run covered two buckets (10 minutes); exact lifecycle edges place it at 11m32s, from 2026-07-20 10:22:48 to 10:34:20 UTC.

The seven-brief peak was not seven productive workers. Briefs 58 and 63 remained lifecycle-open after losing their worker role, overlapping their replacements until both were administratively stopped at 07:45:37 UTC. Exact interval integration gives 91.4% at five, 5.8% below five, and 2.8% above five; therefore the desired five productive briefs were not in flight continuously, even though the open-interval average was 5.00.

### Brief-Cycle Timing

For 92 briefs with a non-lifecycle result, the manager-row-to-first-result median was 9m23s and nearest-rank p95 was 16m25s. Brief 92 had only a start row at the cutoff. Eighty-seven handoffs exceeded five minutes; the only completed briefs at or below five minutes were 1 (3.4m), 2 (5.0m), 6 (4.7m), 14 (4.4m), and 16 (4.4m), so every other completed brief individually exceeded the threshold.

The largest delays were brief 58 (69m24s), brief 62 (32m30s), brief 11 (25m32s), brief 71 (19m52s), and brief 70 (16m25s). This definition intentionally includes the first edit/build/validation/benchmark cycle before a result row; it is not a pure process-spawn latency.

### Lock Contention

| Lock / mode | Paired calls | Total wait | Total run time | Worst wait | Highest-wait brief |
|---|---:|---:|---:|---:|---|
| `build` / shared | 4,697 | 0.000s | 144h26m24s | 0.000s | Tie: every brief paid 0.000s |
| `gpu-*` / exclusive | 0 | Not observed | Not observed | Not observed | Not observable |

The 184 mapped worker/helper JSONLs contained 4,755 in-snapshot acquisitions. Time-order pairing by source, lock, and mode yielded 4,697 complete calls and 58 unmatched acquisitions from interrupted or killed commands; no unmatched release was found. Aggregate measured lock wait was 0% of both the 43.00-hour fleet window and the 187.50 summed measured branch hours. Because every recorded call used shared `build` locking and none emitted `gpu-*` exclusive events, this proves there was no shared-lock queueing but does not measure remote Modal GPU contention or per-GPU load.

### Per-Brief Reliability

The fleet median was 8.26 trials/hour. More than 2x above it were briefs 3 (23.62), 6 (19.45), 2 (19.34), 1 (18.98), and 13 (18.00). Below half the median were briefs 58 (2.76), 63 (2.70), and 73 (2.64), plus one-trial briefs 35, 37, 70, and 75 whose zero-duration rows mechanically report 0.00; open brief 92 had no trial and is excluded as a reliability outlier. Briefs 58 and 63 are genuine low-throughput outliers tied to role loss, while the one- and three-trial cases are small-denominator artifacts.

## Throughput

| Branch | Trials | Duration | Trials/hour | Improvement % |
|---|---:|---:|---:|---:|
| brief-3 | 27 | 1:06:03 | 23.62 | +30.73% |
| brief-6 | 9 | 0:24:41 | 19.45 | +50.71% |
| brief-2 | 22 | 1:05:09 | 19.34 | +34.81% |
| brief-1 | 30 | 1:31:42 | 18.98 | +5.59% |
| brief-13 | 5 | 0:13:20 | 18.00 | +84.15% |
| brief-12 | 14 | 0:48:07 | 16.21 | +66.74% |
| brief-7 | 11 | 0:39:03 | 15.37 | +44.97% |
| brief-8 | 60 | 3:52:13 | 15.24 | +67.06% |
| brief-11 | 31 | 2:03:51 | 14.53 | +66.94% |
| brief-14 | 14 | 0:55:01 | 14.18 | +90.76% |
| brief-19 | 9 | 0:35:33 | 13.50 | +109.63% |
| brief-46 | 8 | 0:31:18 | 13.42 | +134.67% |
| brief-10 | 56 | 4:10:42 | 13.16 | +103.17% |
| brief-4 | 29 | 2:15:30 | 12.40 | +18.57% |
| brief-27 | 6 | 0:25:34 | 11.73 | +124.19% |
| brief-0 | 14 | 1:06:41 | 11.70 | +11.74% |
| brief-30 | 10 | 0:46:34 | 11.60 | +130.79% |
| brief-26 | 11 | 0:51:45 | 11.59 | +117.33% |
| brief-48 | 6 | 0:26:17 | 11.41 | +136.07% |
| brief-17 | 13 | 1:03:42 | 11.30 | +109.24% |
| brief-9 | 43 | 3:46:27 | 11.13 | +98.23% |
| brief-20 | 5 | 0:21:45 | 11.03 | +115.26% |
| brief-41 | 8 | 0:38:08 | 11.02 | +133.33% |
| brief-18 | 12 | 1:00:09 | 10.97 | +104.88% |
| brief-16 | 17 | 1:28:18 | 10.87 | +103.64% |
| brief-42 | 7 | 0:33:30 | 10.74 | +134.82% |
| brief-5 | 5 | 0:22:23 | 10.72 | +47.81% |
| brief-33 | 11 | 0:56:00 | 10.71 | +131.69% |
| brief-47 | 6 | 0:28:43 | 10.44 | +134.90% |
| brief-23 | 15 | 1:22:04 | 10.24 | +112.55% |
| brief-22 | 30 | 2:50:02 | 10.23 | +127.69% |
| brief-28 | 15 | 1:22:12 | 10.22 | +125.98% |
| brief-38 | 10 | 0:53:30 | 10.09 | +133.27% |
| brief-15 | 11 | 0:59:57 | 10.01 | +103.38% |
| brief-24 | 30 | 2:56:11 | 9.88 | +117.62% |
| brief-79 | 8 | 0:42:56 | 9.78 | +180.56% |
| brief-39 | 10 | 0:56:47 | 9.51 | +133.13% |
| brief-51 | 6 | 0:31:39 | 9.48 | +136.78% |
| brief-34 | 7 | 0:38:15 | 9.41 | +133.00% |
| brief-25 | 48 | 5:01:51 | 9.34 | +116.40% |
| brief-56 | 18 | 1:57:01 | 8.72 | +144.73% |
| brief-43 | 36 | 4:07:33 | 8.48 | +135.36% |
| brief-21 | 2 | 0:07:10 | 8.38 | +115.59% |
| brief-40 | 5 | 0:28:42 | 8.36 | +131.90% |
| brief-54 | 9 | 0:57:36 | 8.33 | +143.28% |
| brief-76 | 12 | 1:19:26 | 8.31 | +176.79% |
| brief-68 | 13 | 1:27:12 | 8.26 | +172.86% |
| brief-91 | 6 | 0:36:36 | 8.20 | +181.12% |
| brief-57 | 9 | 0:58:44 | 8.17 | +153.89% |
| brief-32 | 11 | 1:13:53 | 8.12 | +128.46% |
| brief-69 | 10 | 1:07:49 | 7.96 | +171.88% |
| brief-67 | 5 | 0:30:35 | 7.85 | +172.99% |
| brief-64 | 17 | 2:02:23 | 7.84 | +170.93% |
| brief-61 | 17 | 2:04:09 | 7.73 | +162.18% |
| brief-36 | 13 | 1:33:33 | 7.70 | +132.32% |
| brief-29 | 27 | 3:22:52 | 7.69 | +129.55% |
| brief-55 | 25 | 3:09:26 | 7.60 | +146.79% |
| brief-72 | 11 | 1:19:19 | 7.57 | +171.64% |
| brief-52 | 13 | 1:36:52 | 7.43 | +139.78% |
| brief-77 | 37 | 4:52:00 | 7.40 | +179.00% |
| brief-87 | 17 | 2:10:19 | 7.37 | +184.42% |
| brief-45 | 8 | 0:58:31 | 7.18 | +132.87% |
| brief-74 | 19 | 2:31:22 | 7.13 | +177.19% |
| brief-59 | 8 | 1:00:16 | 6.97 | +154.74% |
| brief-78 | 97 | 13:47:27 | 6.96 | +177.00% |
| brief-88 | 32 | 4:30:28 | 6.88 | +189.62% |
| brief-86 | 22 | 3:04:22 | 6.83 | +184.50% |
| brief-60 | 10 | 1:19:37 | 6.78 | +155.12% |
| brief-90 | 14 | 1:56:04 | 6.72 | +187.54% |
| brief-66 | 5 | 0:35:55 | 6.68 | +170.04% |
| brief-31 | 7 | 0:54:03 | 6.66 | +127.97% |
| brief-81 | 82 | 12:30:56 | 6.47 | +184.23% |
| brief-62 | 12 | 1:42:59 | 6.41 | +148.18% |
| brief-85 | 28 | 4:15:28 | 6.34 | +182.37% |
| brief-80 | 2 | 0:09:35 | 6.26 | +183.12% |
| brief-83 | 24 | 3:55:33 | 5.86 | +184.23% |
| brief-89 | 23 | 3:47:40 | 5.80 | +185.08% |
| brief-65 | 6 | 0:52:09 | 5.75 | +170.27% |
| brief-84 | 18 | 2:57:59 | 5.73 | +178.65% |
| brief-44 | 7 | 1:03:13 | 5.69 | +135.26% |
| brief-53 | 21 | 3:50:16 | 5.21 | +138.48% |
| brief-50 | 17 | 3:17:09 | 4.87 | +144.10% |
| brief-49 | 32 | 6:45:54 | 4.58 | +144.26% |
| brief-82 | 8 | 1:34:22 | 4.45 | +183.59% |
| brief-71 | 37 | 8:20:27 | 4.32 | +181.52% |
| brief-58 | 28 | 9:47:20 | 2.76 | +154.24% |
| brief-63 | 5 | 1:28:50 | 2.70 | +158.52% |
| brief-73 | 3 | 0:45:25 | 2.64 | +169.23% |
| brief-35 | 1 | 0:00:00 | 0.00 | +131.57% |
| brief-37 | 1 | 0:00:00 | 0.00 | +130.41% |
| brief-70 | 1 | 0:00:00 | 0.00 | +168.15% |
| brief-75 | 1 | 0:00:00 | 0.00 | +169.64% |
| brief-92 | 0 | 0:00:00 | 0.00 | +0.00% |

Durations are first-to-last trial spans, not lifecycle-open time. Their sum is 187.50 hours because workers overlap. Single-experiment policy omits a mean/stddev row; the unweighted productive-branch rate is 8.96 trials/hour (median 8.29, sample CV 0.47), but that is branch heterogeneity rather than replicate-run variance.

## Gaps And Stalls

The 134 gaps total 52.74 overlapping branch-hours; most contain builds, validation, full-grid benchmarks, or hosted-profile work rather than agent silence. Every gap-bearing branch is accounted for below.

- **brief-8:** 2 gaps, longest 32m17s; both were explicit remote no-progress failures, one in a 640-thread benchmark and one in literal-unroll JIT compilation.
- **brief-9:** 1 gap, 13m01s; hosted validation stopped progressing after extension startup and was terminated.
- **brief-16:** 1 gap, 15m05s; wrapped Modal validation emitted no output well beyond the configured allowance.
- **brief-22:** 1 gap, 17m50s; a 256-thread persistent candidate exceeded the benchmark timeout before the runtime-error row.
- **brief-24:** 2 gaps, longest 10m28s; both were successful full build/validation/benchmark sweeps of large factor-output variants.
- **brief-29:** 1 gap, 13m48s; the successful CuTe rank-k integration required a complete validation and benchmark cycle.
- **brief-31:** 2 gaps, longest 11m11s; one benchmark hung after initialization and the next interval restored and revalidated the safe path.
- **brief-36:** 1 gap, 24m08s; remote compilation/validation failed after the local extension build.
- **brief-43:** 5 gaps, longest 18m12s; profile-directed n=2048 variants repeatedly ran full validation and grid measurements, with no long harness silence.
- **brief-44:** 1 gap, 31m03s; the packed-sidecar benchmark made no progress and was logged as a runtime failure.
- **brief-45:** 1 gap, 26m49s; Modal validation exceeded its timeout after a cleanup-fusion edit.
- **brief-49:** 5 gaps, longest 1h35m35s; hosted Nsight scheduling/artifact work, manager-style waits, and long full-grid cycles dominated, with one later validation failure.
- **brief-50:** 3 gaps, longest 1h06m33s; no-output Modal runs and manager-style wait calls surrounded two runtime failures and a recovery trial.
- **brief-52:** 2 gaps, longest 13m53s; profile inspection plus compile/full-grid measurements explain both successful transitions.
- **brief-53:** 4 gaps, longest 1h04m16s; hosted-profile coordination, full-grid work, and a 1,542-second remote compilation timeout account for the pauses.
- **brief-54:** 1 gap, 12m29s; the worker consumed and analyzed its authorized hosted profile before the successful trial.
- **brief-55:** 4 gaps, longest 23m24s; two software-barrier benchmark failures and two complete profile-guided trials account for the intervals.
- **brief-56:** 1 gap, 14m42s; exact hosted-profile analysis led to a full validation that exposed NaN/Inf.
- **brief-57:** 1 gap, 13m20s; the higher-thread residency benchmark reached a runtime failure after validation.
- **brief-58:** 4 gaps, longest 6h06m19s; three were long remote runs/failures, while the largest was role loss after Trial 26 followed by manager-loop waits until administrative recovery logged Trial 27.
- **brief-59:** 1 gap, 22m50s; hosted-profile work and a byte-identical restore/remeasurement filled the interval.
- **brief-60:** 2 gaps, longest 20m56s; a heterogeneous software-barrier run timed out, followed by interrupted-state cleanup and a successful recovery.
- **brief-61:** 1 gap, 21m46s; the worker analyzed an exact hosted profile and ran a complete successful candidate.
- **brief-62:** 2 gaps, longest 15m31s; hosted profiling preceded one successful trial, then the next storage rewrite failed validation.
- **brief-63:** 3 gaps, longest 53m28s; direct swizzle experiments caused validation failures and a 2,907-second benchmark hang. The later role-loss interval after Trial 4 is not counted because Trial 5 was never logged.
- **brief-64:** 1 gap, 33m52s; the worker recovered from an illegal-address first trial and completed hosted-profile-backed validation.
- **brief-65:** 2 gaps, longest 17m42s; one contains hosted profile work, the other a Modal validation timeout on a restored source.
- **brief-66:** 1 gap, 10m16s; exact hosted-profile analysis and the following successful full-grid run account for it.
- **brief-68:** 1 gap, 16m43s; the authorized hosted profile drove a successful register-cap experiment.
- **brief-71:** 6 gaps, longest 2h40m46s; after Trial 29 the worker entered manager-style waits, including a one-hour wait, before resuming; later profile-guided validation/runtime failures add the smaller gaps.
- **brief-72:** 1 gap, 19m24s; hosted-profile work and noisy full-grid sampling preceded the successful trial.
- **brief-73:** 2 gaps, longest 23m22s; both contain hosted-profile capture/artifact analysis and successful remeasurement.
- **brief-74:** 1 gap, 34m34s; an all-thread staging benchmark hung until termination.
- **brief-76:** 1 gap, 12m02s; authorized hosted profiling and artifact canonicalization account for the successful transition.
- **brief-77:** 5 gaps, longest 33m37s; hosted-profile analysis and long, successful full-grid graph experiments dominate; no lock wait was present.
- **brief-78:** 9 gaps, longest 1h12m41s; the largest includes a 61m37s task-complete-to-restart interruption after Trial 17, while smaller gaps map to remote failures, profiling, and long full-grid runs.
- **brief-81:** 13 gaps, longest 1h15m19s; the largest contains dashboard/fleet inspection and a one-hour manager-style wait after Trial 36; other gaps are profile work and a 1,567-second benchmark timeout.
- **brief-82:** 2 gaps, longest 38m52s; hosted-profile analysis drove one successful transition and one later Modal runtime failure.
- **brief-83:** 6 gaps, longest 52m50s; a compaction/resumption and multiple complete benchmark iterations occurred between rows, without measurable lock wait.
- **brief-84:** 2 gaps, longest 47m52s; exact hosted-profile work and a later runtime timeout explain them.
- **brief-85:** 5 gaps, longest 28m03s; profile analysis, a validation failure, and long full-grid iterations account for the delays.
- **brief-86:** 4 gaps, longest 11m40s; all four were near-threshold successful build/validation/full-grid cycles for packed small-matrix variants.
- **brief-87:** 2 gaps, longest 27m30s; hosted-profile analysis and complete successful measurements account for both.
- **brief-88:** 6 gaps, longest 29m04s; profile-guided large-shape trials and one runtime timeout dominated.
- **brief-89:** 7 gaps, longest 29m16s; the authorized profile and long/outlier-heavy Modal full-grid trials explain the intervals.
- **brief-90:** 4 gaps, longest 18m47s; profile-guided pipeline variants repeatedly ran noisy full grids, including a globally slow repeat.
- **brief-91:** 1 gap, 10m53s; the 14-stage full-grid benchmark remained live past its nominal allowance and eventually returned a successful result.

## Early-Termination Attempts

No unauthorized early-termination attempt was found across the mapped harness records. No worker asked whether it should continue or declared the experiment converged. Normal `complete`, `stopped`, or `exhausted` language followed a `brief_stop`; active-brief final-answer records on briefs 55, 58, 85, 89, and 90 were explicit compaction handoffs/checkpoints and those workers were continued, not operator-confirmation gates.

## Other Aberrant Behavior

- **brief-58 after Trial 26:** compaction erased the worker role; the session spent about 4h42m in manager-style `wait_agent` calls and produced no optimization state until recovery logged Trial 27 and stopped the brief.
- **brief-63 after Trial 4:** the same role error abandoned a built and validated Trial 5 benchmark session; no Trial 5 row, commit, or metric was recovered before the administrative stop.
- **brief-71 after Trial 29 and brief-81 after Trial 36:** both temporarily performed manager/dashboard duties and issued one-hour waits. They recovered and resumed, but caused the run's 2h40m46s and 1h15m19s branch gaps.
- **brief-19 after Trial 8:** the worker invoked `autocuda log optimize-tree stop` twice to correct a commit SHA, creating one orphan duplicate `brief_stop`. Trial numbering is otherwise contiguous and all timestamps are monotonic.
- **Lock coverage:** 58 acquisitions lack release lines because their commands were killed or interrupted, and the run emitted no exclusive `gpu-*` records. Neither condition fabricated a trial row, but both limit contention attribution.

## Analysis

The fleet delivered useful optimization work, but trial throughput slowed as the search moved from early library/dispatch changes into graph-captured and profile-driven kernels. That complexity explains much of the decline from the 18–24 trials/hour early outliers to the 4–8 trials/hour later branches: builds, remote compilation, full 15-cell grids, and hosted profiles became longer. The one-run branch-rate CV of 0.47 is therefore primarily workload mix and branch age, not evidence of unstable repeated experiments.

The 134 gaps are not 52.74 hours of fleet idleness because workers overlap and most gaps contain external work. The bottleneck worth fixing is the distinct harness failure: context compaction caused workers to adopt manager behavior. Briefs 58 and 63 permanently lost productive time, while briefs 71 and 81 recovered after long waits. Shared build locking contributed no delay, and the absence of exclusive lock events means the report cannot blame or clear remote GPU contention.

Logging was otherwise reliable: all 1,581 trial numbers are contiguous within their branches, timestamps are ordered, and 88.8% of rows succeeded. The lone lifecycle defect is brief 19's corrective duplicate stop. No operator-confirmation gate or convergence claim interrupted the autonomous loop.

**Recommendations:** make worker identity and brief ID immutable across compaction and have the 30-minute health audit check trial progress, not merely live task count; record a distinct worker-start/first-command timestamp so handoff latency is not conflated with the first benchmark; and route validation/benchmark/profile work through instrumented exclusive locking (or emit equivalent remote-GPU events) while enforcing bounded no-output timeouts.

## Scope & Methodology

This is a live single-experiment snapshot of tag `2026-07-19-01-44-06-cholesky-resumed-local-unmerged` through 2026-07-20 21:01:15 UTC. It covers manager rows and brief logs 0–92; briefs 78, 81, 89, 91, and 92 were lifecycle-open at the cutoff, and brief 92 had not produced a trial. Later rows from the ongoing run are intentionally excluded. The behavior CSV was regenerated with `autocuda report data behavior --tag 2026-07-19-01-44-06-cholesky-resumed-local-unmerged --output 2026-07-19-01-44-06-cholesky-resumed-local-unmerged` and schema-checked before analysis.

Concurrency pairs every `brief_start` with the next `brief_stop`, ignores brief 19's orphan second stop, caps open intervals at the cutoff, and samples active sets at five-minute window midpoints; exact edge integration is reported separately. Handoff latency follows the report contract: manager assignment to the first later non-lifecycle row. Throughput and gap counts come from the generated CSV; gap causes were correlated with per-trial descriptions and harness timelines.

First-line agent-path mapping found 184 Codex worker/helper JSONLs covering all 93 briefs after excluding two read-only status helpers and correcting the nested brief-58 and brief-49 ownership exceptions. A session began strictly inside the log span for 24 briefs; for the other 69, the associated top-level worker session began 9–107 seconds before its first `brief_start`, so the nearest preceding session was used rather than claiming missing coverage. All mapped sessions were read for lock and aberration evidence. Lock events were admitted only when their embedded timestamp followed that source's first timestamp and preceded the cutoff, removing fork-replayed parent history; acquisitions/releases were paired by source, lock, mode, and time order.

The run records only shared `build` locks, even for commands that performed remote validation/benchmarking, so exclusive per-GPU utilization and remote queueing are coverage gaps. Branch-rate spread is descriptive only: with one experiment tag there is no cross-run mean/stddev row and no basis for a replicated stability estimate.
