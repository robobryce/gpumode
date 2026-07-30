# Autocuda Status Metrics Backtest: 2026-07-19-01-44-06-cholesky-resumed-local-unmerged

## Executive Summary

A timestamp-prefix replay of all 1,960 logged events shows that the exact progress record is reliable, but the current recent-improvement signal does not distinguish sustained plateaus from ordinary gaps between records. The run improved from 2,153.690 µs to 743.614 µs (2.896247×, a 65.47% latency reduction), while `stalled` was true for 1,437 of 1,454 successful-trial snapshots (98.83%). Even the narrower condition “recent rate equals zero” spent 8.47 off-target hours alerting.

The two requested targets are qualitatively different. Region A is an exact 1.55-hour record drought from hour 17.761 to 19.309. Region B runs from hour 30.215 to 39.620 and contains several micro-records, but they improve the starting record by only 0.488% before a +2.01% breakout. The most selective in-sample candidate tested is a persistence-filtered wall-clock slope: alert after the trailing one-hour record gain remains below 0.25% for 30 continuous minutes. It covered 41.9% of A and 84.1% of B with only 0.283 off-target alert-hours, giving 96.8% duration precision when the right-censored terminal drought is excluded.

**Key takeaways**

- `global_best` correctly tracked 119 record advances and the final brief 88 trial 19 winner.
- The narrow zero-rate proxy covered only 57.0% of A and 72.9% of B while repeatedly firing elsewhere.
- The proposed persistent-slope detector produced two requested alerts, two short off-target alerts totaling 17 minutes, and one unscored right-censored terminal alert.
- Material-record age, effort age, and an adaptive historical drought score are useful diagnostics but less selective as booleans.
- The intended tree replay declined from 1.0 to 0.4291, ending with 6.99 effective directions across 93 historical leaves.
- The shell-installed CLI is version-skewed: its strict SHA parser returns `balance: null` after the first abbreviated commit, while the configured plugin parser computes the tree.

## Results At A Glance

| Signal | Backtest result | Verdict |
|---|---:|---|
| Exact global record | 1.000000× → 2.896247×; 119 advances | Trustworthy and directly interpretable |
| Recent improvement | Median 0.0083265% per successful trial; 657/1,454 observations were exactly zero | Useful as a continuous rate |
| `stalled` at 1%/trial | 1,437/1,454 snapshots; 44.675 of 44.907 hours after first success | Saturated; poor discriminator |
| Persistent one-hour slope | Under 0.25% gain for 30 continuous minutes | Most selective in-sample candidate tested |
| Requested plateau coverage | A: 41.9% after 54 min; B: 84.1% after 90 min | Delayed but selective |
| Detector selectivity | 0.283 off-target hours; 96.8% duration precision excluding terminal drought | Large improvement over zero-rate |
| Look-forward from stalled snapshots | 782/1,427 had a new record within the next 10 successes; 263 gained at least 1% | Not a convergence signal |
| Intended tree balance | 1.0 → 0.4291; 6.99 effective directions, 93 leaves, 1,737 nodes | Descriptive, but needs its components |
| Shell CLI tree balance | Unavailable after 2026-07-19 02:08:09 UTC | Version-skew defect |
| Fleet width | Exactly five open for 40.701/44.943 hours (90.56%); peak seven | Per-brief state adds useful context |

## Plateau Detection Objective

![Target plateau regions](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-plateau-targets.png){width=100%}

Region A is the exact-record plateau from 17.761 to 19.309 hours. It lasted 1.548 hours and contained 27 successful trials that did not set a record; the 28th success ended it with a +0.912% record gain.

Region B needs a material-progress definition rather than exact record age. It begins at the 2.831240× record at hour 30.215. Six later records reached only 2.845048×, a cumulative +0.488%, so the progress line remained visually flat. The regime ended at hour 39.620 when brief 88 trial 8 jumped to 2.888188×, +2.01% from the starting record. Treating every micro-record as a reset fragments B into several gaps and misses the behavior of interest.

The final exact-record plateau from hour 41.118 through the 45.029-hour run end is shaded separately. An alert there is plausible but unscored because the interval is right-censored: the experiment ended before a recovery could confirm its full duration. It is excluded from detector precision rather than being counted as either a true or false positive.

## Alternative Plateau Detectors

![Detector alert comparison](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-plateau-detector-comparison.png){width=100%}

Five online alternatives were replayed in addition to a narrow `recent rate == 0` zero-growth baseline. This proxy is distinct from the current `stalled` rule, which uses `recent rate < 1%`:

1. **Exact record age:** hours since any new exact record; alert at one hour. It is simple, but B’s micro-records repeatedly reset it.
2. **Material record age:** maintain checkpoint `C`, updating it only when the global record improves by at least 0.5% from `C`; alert after one hour without such an improvement.
3. **Material effort age:** from the same 0.5% checkpoint, count all completed attempts—including build, validation, and runtime failures—and divide by five workers; alert at six attempts per worker.
4. **Adaptive drought ratio:** material age divided by the nearest-rank online 95th percentile of prior completed 0.5%-material gaps; alert at one after at least 20 completed gaps.
5. **Persistent wall-clock slope:** compute trailing one-hour record gain and alert only after gain below 0.25% persists for 30 minutes. This is the proposed primary detector.

![Normalized alternative metric traces](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-plateau-metric-traces.png){width=100%}

The table evaluates one-minute online snapshots from hour 5 onward. “Off-target” means alert time outside A, B, and the right-censored terminal interval; it can include plausible secondary slowdowns, so it is a selectivity measure rather than proof that every extra alert is wrong. Duration precision is target alert-hours divided by target plus off-target alert-hours.

| Detector | A delay | A coverage | B delay | B coverage | Off-target alert time | Duration precision |
|---|---:|---:|---:|---:|---:|---:|
| Zero-rate proxy | 40 min | 57.0% | 26 min | 72.9% | 8.467 h | 47.8% |
| Exact record age ≥ 1 h | 60 min | 35.5% | 112 min | 44.8% | 1.333 h | 78.1% |
| 0.5%-material age ≥ 1 h | 24 min | 74.2% | 60 min | 89.4% | 4.150 h | 69.7% |
| Material attempts/worker ≥ 6 | 24 min | 74.2% | 65 min | 88.5% | 5.250 h | 64.4% |
| Material age ≥ online p95 | 35 min | 62.4% | 103 min | 81.8% | 3.033 h | 74.1% |
| **Persistent 1 h slope (proposed)** | **54 min** | **41.9%** | **90 min** | **84.1%** | **0.283 h** | **96.8%** |

### Proposed Status Metric

Let `R(t)` be the forward-filled exact global-record speedup at time `t`: the last record set at or before `t`. Compute:

`gain_1h(t) = 100 × (R(t) / R(t - 1 hour) - 1)`.

Track how long `gain_1h` has remained below 0.25%, resetting the timer whenever it reaches the threshold. Set `plateau: true` after that low-gain condition lasts 30 continuous minutes.

Implement the dwell statelessly from record history, not by incrementing a timer only when status is polled. Since `gain_1h` can change both when a record arrives and when that record crosses the one-hour lookback boundary, evaluate those record timestamps and timestamp-plus-one-hour breakpoints to recover the most recent continuous low-gain interval at any status call.

On this run the detector alerted at 18.667–19.317 hours and 31.717–39.633 hours for the requested regions. Its only earlier off-target episodes were 12.750–12.950 hours and 29.200–29.283 hours, totaling 17 minutes. It also alerted at 42.633 hours and remained active through run end, producing an unscored alert in the right-censored terminal drought.

This is the most selective in-sample candidate because it operates in the same wall-clock domain as the visual plateaus, retains small improvements without letting them reset the state, and requires persistence before changing the boolean. Its 1-hour, 0.25%, and 30-minute constants were selected on this experiment and require validation across other problems before adoption. Material age and attempts-per-worker should be exposed alongside it as continuous diagnostic context rather than combined into the primary boolean.

## Exact Progress

![Exact progress timeline](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-progress.png){width=100%}

The baseline was 2,153.690 µs at 2026-07-19 01:56:27 UTC. The record crossed 2.0× after 4.7 hours, 2.5× after 19.6 hours, and 2.8× after 27.7 hours. Brief 88 trial 19 set the final 743.614 µs / 2.896247× record after 41.12 hours; the remaining 3.91 hours produced no higher record.

The gray points in the graph are all 1,454 complete successful measurements, while the green line is the exact reference-anchored record used by `global_best`. The record is monotone by construction, uses the schema’s lower-is-better direction, and agrees with the final status payload. This is the strongest status signal in the backtest.

The 1,638 attempts comprised 1,454 successes (88.77%), 81 runtime errors, 80 validation errors, and 23 build errors. Failures do not affect either the record or the recent-improvement window.

## Recent Improvement And `stalled`

![Recent improvement and stalled timeline](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-recent-improvement.png){width=100%}

With five configured workers, the current trailing window is `max(2 × workers, 10) = 10` complete successful trials. At each successful trial the replay computes

`100 × ((record_after / record_before)^(1 / window_size) - 1)`.

The window uses all successes so far until ten exist. Consequently the first complete success can immediately declare a stall: the run’s first successful result was slower than baseline, producing a 0% rate and `stalled: true` with only one observation.

The state changed only five times:

| Successful trial index | UTC timestamp | Recent rate | State | Global record |
|---:|---|---:|---|---:|
| 1 | 2026-07-19 02:03:45 | 0.000000% | stalled | 1.000000× |
| 2 | 2026-07-19 02:05:33 | 8.225839% | not stalled | 1.171283× |
| 12 | 2026-07-19 02:12:12 | 0.000000% | stalled | 1.171283× |
| 100 | 2026-07-19 03:19:34 | 1.120966% | not stalled | 1.507083× |
| 107 | 2026-07-19 03:26:51 | 0.942514% | stalled through run end | 1.507083× |

After trial 107, the record still rose from 1.507083× to 2.896247×. This does not contradict the formula—the flag asks whether the *trailing ten successes* compounded at 1% per trial—but it does show that the boolean is too coarse to summarize the search’s long-run potential.

Among the 1,427 stalled snapshots with ten later successes available, 782 (54.80%) were followed by a new record within those next ten successes, and 263 (18.43%) were followed by at least a 1% record gain. These overlapping look-forward windows are descriptive rather than independent statistical trials, but they reinforce that `stalled` must not be read as “no improvement ahead.”

### Threshold Sensitivity

| Hypothetical threshold (% per successful trial) | Successful snapshots classified stalled |
|---:|---:|
| 0.010 | 51.93% |
| 0.025 | 61.49% |
| 0.050 | 70.98% |
| 0.100 | 81.16% |
| 0.250 | 92.64% |
| 0.500 | 97.46% |
| 1.000 (current) | 98.83% |

One experiment is not enough to select a replacement threshold, but it is enough to reject 1%/trial as a discriminating policy for this run. A full ten-success warm-up should precede any verdict based on this signal. Keep the continuous per-success rate as descriptive context; use the persistence-filtered wall-clock slope above for plateau detection, subject to cross-run calibration.

The live experiment used an older stall rule early in the run and the current 1% rule later. This report deliberately applies the current formula to every prefix so the timeline is comparable; it is not a reconstruction of the changing historical boolean.

## Tree Balance

![Tree balance timeline](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-tree-balance.png){width=100%}

Balance is the normalized Shao–Sokal B2 entropy of the tree’s leaf-walk probabilities. It is 1.0 when the probability is evenly distributed across leaves and trends toward zero as ancestry concentrates the search. The intended abbreviated-SHA-compatible replay was 1.0 across the initial five-way fan-out, then declined to 0.4291. At run end, the tree contained 1,737 nodes and 93 historical leaves but only 6.99 effective directions.

The effective-direction count reached 6.99 about 4.6 hours into the run and then remained flat to two-decimal precision while historical leaves continued to accumulate. At that point the performance record was 1.981489×; it later improved another 46.17%. Balance therefore describes accumulated topology, not expected future improvement. Reporting `effective_directions`, leaf count, and whether the metric is computed over historical or active tips would make the scalar substantially easier to interpret.

### Parser Version Skew

The run contains valid historical seven-character commit identities. The currently configured plugin cache accepts 7–40 lowercase hexadecimal characters, resolves compatible prefixes, and produces the purple timeline. The shell-installed `autocuda` package still requires complete 40-character SHAs. It fails when brief 2 trial 1 introduces `9586dd7` at 2026-07-19 02:08:09 UTC, so the present shell command reports `progress.balance: null` for the finished run.

This is not evidence that the search tree is absent. It is an installation mismatch between two dashboard implementations shipped as autocuda 0.4.0. The CLI and configured plugin should be updated atomically, and status should expose a balance error or data-quality field instead of collapsing parser failure into an unexplained null.

## Per-Brief Fleet Health

![Fleet health timeline](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-fleet-health.png){width=100%}

From the first moment all five workers were open through the last stop, exactly five briefs were lifecycle-open for 40.701 of 44.943 hours (90.56%). Handoffs caused many short dips below five. The peak of seven came from stale lifecycle-open briefs overlapping replacements, not seven productive workers.

The replay also derives an aggregate from status’s per-brief last-activity fields. Up to three open briefs were simultaneously quiet for at least 30 minutes. Long builds, validation, benchmarks, and profiles can legitimately cross that threshold, so quietness is a health-check trigger rather than proof of failure. Still, an explicit `active_briefs` and `stale_active_briefs` summary would save the manager from reconstructing fleet health from the entire `per_brief` array.

## Recommendations

1. Keep the reference-anchored `global_best` calculation unchanged.
2. Make a persistence-filtered wall-clock slope the primary plateau signal: expose `gain_1h`, `low_gain_duration`, and `plateau`, with 0.25% gain and 30 minutes of persistence as provisional defaults. Validate all three constants across representative runs before treating them as policy.
3. Expose 0.5%-material record age and material attempts per worker as continuous diagnostics. They detect both requested regions earlier than the primary signal, but their extra alerts make them better explanatory context than standalone booleans.
4. Retain the current per-success recent-improvement rate as cadence-sensitive search-yield telemetry, not as the plateau or convergence signal. Deprecate or rename its ambiguous `stalled` boolean; if retained for compatibility, require a complete trailing window and never use it alone as a termination condition.
5. Return the recent window size and record gain across that window alongside its continuous rate so a manager can distinguish no progress from slow progress.
6. Add `effective_directions`, leaf count, and balance computation status/error to the progress payload; consider a second balance over active frontier tips.
7. Synchronize the shell package and configured plugin parser, retaining abbreviated-SHA compatibility for historical logs.
8. Add compact fleet aggregates (`active`, `stale`, target width) while keeping the detailed per-brief records.

## Scope & Methodology

This report covers tag `2026-07-19-01-44-06-cholesky-resumed-local-unmerged` from the 2026-07-19 01:56:27 UTC baseline through the final 2026-07-20 22:58:12 UTC log row, a 45.03-hour run. It reads the reference log, 98 manager brief rows, and every brief log, including 1,638 attempt rows and 223 lifecycle rows.

The replay evaluates every timestamped merged-log prefix in chronological order. This is necessary because `autocuda status --as-of <timestamp>` changes only staleness math; it does **not** hide rows written after that timestamp. Looping `--as-of` over the complete data directory would therefore leak the final record and tree into every historical point.

Progress and recent improvement use the current autocuda 0.4.0 manager formulas: one `min` benchmark (`linalg/cholesky_py`), a 2,153.690 µs baseline, a ten-success trailing window, failures excluded, and `stalled` when the rounded recent rate is below 1%. Tree replay uses the latest configured 0.4.0 dashboard parser (SHA-256 `05d0e5063408…`), while the orange availability trace uses the shell-installed strict parser (`9b64353d0e6f…`). The two manager implementations are identical; only their dashboard/tree dependency differs.

The plateau-detector backtest samples the online-recomputable state on a baseline-anchored one-minute grid from hour 5 through the final complete minute, 44 seconds before the final log row. Target A is `[17.761, 19.309)` hours, the exact-record drought immediately before hour 20. Target B is `[30.215, 39.620)` hours, the materially flat period that permits less than 0.5% cumulative record growth before its breakout. The terminal interval `[41.118, 45.029]` hours is right-censored: alerts there are recognized but omitted from both the precision numerator and denominator.

The delays and coverage therefore describe a hypothetical one-minute status observer. The metric can be recomputed online at any timestamp, but the notification-driven manager sees a transition only on its next status call; its operational alert can lag the table by up to that polling gap.

Detection delay is the elapsed time from a target's start to its first alert. Coverage is the fraction of target snapshot-minutes under alert. Off-target time is alert duration outside A, B, and the terminal interval during the evaluation period. Duration precision is alert time inside A and B divided by that target time plus off-target time; it measures temporal selectivity, not independent-event classification accuracy. The detector thresholds were selected after inspecting this single run, so these are in-sample backtest results rather than evidence of cross-run generalization.

The companion [event-level replay CSV](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics.csv) contains the original graph inputs: event identity, cumulative attempt/success counts, record and recent-rate fields, intended and shell-CLI balance, tree dimensions, and per-event fleet health. The [one-minute plateau-detector CSV](2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-status-metrics-plateau-detectors.csv) contains every alternative metric, threshold, alert state, and target label used in the detector comparison. The PNG timelines were rendered with matplotlib. Historical live status calls are not substituted for prefix replay because the status implementation changed during the experiment; the report measures one consistent current policy against the complete historical data.
