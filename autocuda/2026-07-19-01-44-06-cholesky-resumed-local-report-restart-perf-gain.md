# Why Reallocating the Whole Fleet Broke the Performance Plateau

**Benchmark:** `linalg/cholesky_py`  
**Report tag:** `2026-07-19-01-44-06-cholesky-resumed-local-unmerged`  
**Runs studied:** the original run, `cholesky_r2`, and `cholesky_r3`

## Short Answer

The original run made excellent early progress, but eventually spent hundreds of trials refining and merging the same family of solutions. The two follow-on runs kept the best code and replaced all five active assignments with fresh ones. That directed work toward opportunities the original run had missed.

The workers did not run trials faster. Their new assignments simply produced better ideas.

Autocuda should reproduce that result only when the search has genuinely stalled:

```text
SEARCH NORMALLY
       ↓
DETECT A REAL STALL
       ↓
STOP THE WHOLE FLEET AFTER ITS CURRENT TRIALS
       ↓
START A FRESH NON-COMBINE FLEET
       ↓
SEARCH NORMALLY
```

A four-hour cooldown prevents repeated reallocations when the search remains difficult.

## What the Numbers Show

| Run | Time | Trials | Successful trials | Best local time | Best public time |
|---|---:|---:|---:|---:|---:|
| Original run | 45:14 | 1,638 | 1,454 | 743.614 µs | 735.196 µs |
| r2 | 5:32 | 156 | 130 | 715.900 µs | 711.991 µs |
| r3 | 3:19 | 70 | 55 | 670.920 µs | 656.445 µs |

The two follow-on runs improved the retained public result from `735.196` to `656.445 µs`, a `10.712%` reduction, in 226 trials.

The original run's productivity fell sharply:

| Part of the search | Trials | Improvement per 100 trials |
|---|---:|---:|
| First 30 hours | 1,175 | 118.184 µs |
| Final 15.24 hours | 463 | 4.626 µs |

The final part was about 25.5 times less productive per trial. After its last local record, the run performed another 120 trials for almost four hours without improving it.

The follow-on runs had roughly the same trials per hour and lower success rates. Their advantage was better trial selection, not more trial throughput.

Timing varies between runs, so small score differences are not strong evidence by themselves. The clearest evidence comes from large improvements on specific benchmark cases and the close final public and secret scores.

## Why a Full Reallocation Helped

### The long run remained active but its returns had collapsed

The original run began with very different approaches. Over time, more work concentrated on the current leading code:

- 58 of 98 assignments were about merging earlier work;
- 72 of 93 assignments after the initial batch started from the current best version;
- many later assignments had to preserve most of the existing design.

The manager still chose every brief's starting commit explicitly, and the skill already required diversity. The missing action was not another diversity rule. It was a moment when **all five assignments were reconsidered together** after measured progress had stopped.

### Fresh assignments exposed neglected work

The clearest example was `batch16/n512`. The original run spent substantial effort on `batch640/n512` but never gave the low-batch n512 case its own clear assignment. A fresh r3 worker built a different method and cut:

`508.062 → 175.995 µs`

That benchmark case became `65.36%` faster and explains a large part of the final result.

### Fresh assignments reused successful code more broadly

In r2, a change of only 38 lines routed existing fast code to four additional matrix shapes. In r3, a method written for `batch2/n4096` was adapted to `batch8/n2048`, cutting:

`1797.576 → 977.081 µs`

This suggests one useful question whenever the fleet is reallocated:

> Which successful methods have not yet been tried on other compatible benchmark cases?

### Starting without combines restored exploration

The initial assignments in the follow-on runs were not combine briefs. Workers first established separate improvements. r3's final merge then worked because those improvements mostly affected different benchmark cases.

The final merges in the original run and r2 did not improve their best results. The lesson is not to combine more; it is to combine only after fresh exploration has produced clear, compatible wins.

## Plan

### 1. Replace the current stall calculation with a meaningful one

A trial counts as meaningful progress when it does either of the following:

- improves the trusted global best by more than normal timing variation; or
- makes a similarly clear improvement within its own brief, showing that the worker is still advancing a potentially useful branch.

Important gains should be remeasured before they reset the stall window.

Declare a stall only when:

1. enough trials have run since the last fleet allocation;
2. no worker has produced meaningful global or branch progress over the trailing trial window;
3. the same conclusion appears in two consecutive checks;
4. the four-hour reallocation cooldown has expired.

This avoids both errors: reallocating a productive fleet too early and allowing noise-sized wins to keep a weak search alive indefinitely.

### 2. Keep `autocuda status` simple

Status needs only three progress fields:

```json
{
  "stalled": true,
  "trials_since_improvement": 37,
  "reallocate": false
}
```

- `stalled` says whether recent trials show meaningful progress.
- `trials_since_improvement` explains the decision.
- `reallocate` is the action signal. It is true only when the fleet is stalled and the cooldown has expired.

Noise estimates, timestamps, and cooldown bookkeeping should remain internal. The existing best-trial and per-brief status output already provides the rest of the useful detail.

### 3. Stop every active worker at a safe boundary

When `reallocate` becomes true, tell every worker:

1. finish the current build, validation, benchmark, and trial log;
2. commit the measured code;
3. write `brief_stop`;
4. do not start another trial;
5. return the best result and remaining ideas.

Do not interrupt a trial halfway through. Start no replacement worker until the old fleet has returned, so no brief has two writers.

### 4. Start a fresh fleet using round-zero rules

Create all replacement assignments together.

- Create exactly the configured number of briefs.
- Use no `combine` briefs.
- Require exactly one brief to start from the original baseline.
- Select the other parents through the existing parent-selection rules.
- Apply the existing diversity rules to the fresh fleet as a whole.
- Give every brief a fresh worker.

The baseline brief is intentionally expensive insurance. It gives one worker a clean path away from assumptions embedded in the current best code. Its assignment should try a substantially different approach, not recreate known work.

After this initial fresh batch is launched, normal brief selection resumes as workers finish.

### 5. Enforce a four-hour cooldown

Record the time of each fleet allocation internally. Another full reallocation is forbidden for four hours.

During the cooldown:

- status may report `stalled: true`;
- status must report `reallocate: false`;
- the manager continues normal per-worker steering and replacement.

Progress calculations for the new fleet must ignore trials from before the allocation. Otherwise the old stall could immediately trigger another reallocation.

A meaningful improvement resets `trials_since_improvement`, but the four-hour cooldown remains anchored to the last allocation.

## Minimal Implementation Changes

1. Add an internal fleet-allocation number to manager and brief rows.
2. Track the last meaningful improvement within that allocation.
3. Teach workers to honor a “finish this trial and stop” request.
4. Add one manager action that stops the fleet and writes a new non-combine batch.
5. Keep the last allocation time internally to enforce the four-hour cooldown.

No new diversity system is required. Reallocation should reuse the diversity and parent-selection rules that already exist.

## Tests

The change should include these regression tests:

1. The original run's 120-trial flat tail triggers `reallocate`.
2. r3's early productive search does not trigger a false stall.
3. A worker making meaningful branch progress prevents a fleet-wide stall.
4. Same-version timing variation does not count as meaningful progress.
5. Old trials cannot immediately retrigger a stall after reallocation.
6. Two reallocations cannot happen within four hours.
7. Every active worker stops before replacement workers start.
8. Every replacement brief is non-combine.
9. Exactly one replacement brief starts from the baseline.
10. Normal parent selection and diversity checks apply to the other briefs.

## Performance Evaluation

Compare the current skill and the reallocation design with equal GPU time, inference budget, and trial count across many benchmarks.

Measure:

- improvement per 100 trials;
- time and trials needed to reach the best trusted result;
- number of reallocations;
- useful improvements found after reallocation;
- consistency across repeated runs.

The main question is simple:

> When progress genuinely stops, does replacing the whole fleet find better results than continuing to replace workers one at a time?

## Scope and Limits

This report studies 112 assignments and 1,864 trials across the three runs. The follow-on runs reused code and knowledge from earlier work, so this was not a controlled experiment. Timing also varies between runs.

Even with those limits, the result strongly supports one targeted change: continue the current search while meaningful progress exists, then perform a cooldown-limited, round-zero-style fleet reallocation when robust stall tracking says progress has stopped.

## Evidence Files

- [Original optimization report](./2026-07-19-01-44-06-cholesky-resumed-local-unmerged-report-optimization.md)
- [Original manager log](./2026-07-19-01-44-06-cholesky-resumed-local-unmerged-optimize-tree-manager-log.csv)
- [r2 manager log](./2026-07-20-23-00-46-cholesky-unmerged-optimize-tree-manager-log.csv)
- [r3 manager log](./2026-07-21-04-34-16-cholesky-unmerged-optimize-tree-manager-log.csv)
- [Original public submissions](./2026-07-19-01-44-06-cholesky-resumed-local-unmerged-leaderboard-submissions.md)
- [r2 public submissions](./2026-07-20-23-00-46-cholesky-unmerged-leaderboard-submissions.md)
- [r3 public submissions](./2026-07-21-04-34-16-cholesky-unmerged-leaderboard-submissions.md)
