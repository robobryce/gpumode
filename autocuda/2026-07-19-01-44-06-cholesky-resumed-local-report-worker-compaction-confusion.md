# Optimize-Tree Worker Compaction Confusion Incident

## Executive Summary

One high-severity worker-role contract defect affected two `autocuda:explore-brief` workers in run `2026-07-19-01-44-06-cholesky-resumed-local-unmerged`, and one medium-severity manager health-check gap delayed detection. Both workers adopted the manager role after context compaction and entered its indefinite wait loop instead of continuing their assigned briefs; direct harness evidence is complete for the two affected branches.

**Key takeaways**

- Briefs 58 and 63 stopped producing trials because compaction recovery did not preserve their worker identity.
- Neither incident was caused by a tool, API, GPU, or remote infrastructure failure.
- Brief 63 abandoned a validated in-flight Trial 5 benchmark session; Brief 58 stopped after a successful Trial 26.
- A task-list-only health check reported both sessions as running and could not detect zero trial throughput.
- The fix makes worker identity permanent and adds a rate-limited `autocuda status` health audit every 30 minutes.

## Skill Issue Candidates

| Branch | Skill | Severity | Category | Evidence | Recommendation |
|---|---|---|---|---|---|
| `brief-58` | `autocuda:explore-brief` | High | role_recovery | At 06:31 UTC the worker reported that, after compaction, it misidentified itself as manager and waited for worker completions instead of resuming after Trial 26. | Make worker role and `brief-id` permanent across compaction; resume only the assigned brief and return if identity is unavailable. |
| `brief-63` | `autocuda:explore-brief` | High | role_recovery | At 06:31 UTC the worker reported that it entered the manager wait loop and ignored the handoff to collect benchmark session `42908` for validated Trial 5. | Apply the same permanent-role guard in the worker skill and every host-specific worker definition. |
| Manager | `autocuda:optimize-tree` | Medium | health_check | The 30-minute check counted live tasks only; Briefs 58 and 63 remained marked running for 290 and 338 minutes without new trial rows. | Audit `autocuda status` at most once per 30 minutes, inspect per-brief activity, and checkpoint or replace unresponsive workers. |
| Briefs 0–57, 59–62, 64–78 | `autocuda:optimize-tree` | Info | coverage_gap | The targeted data build mapped only the two affected worker JSONLs; 77 other brief branches have missing-harness rows in the companion CSV. | Treat this report as an incident audit, not a complete run-wide skill audit. |

The Brief 58 harness record at `/home/ubuntu/.codex/sessions/2026/07/19/rollout-2026-07-19T22-42-00-019f7c8b-1f8f-7d30-ae2a-2b40049a0eb4.jsonl:4084` contains the worker's direct postmortem. Its last trial row was successful at `2026-07-20T01:39:18.623972`, after which no additional Brief 58 state was written.

The Brief 63 harness record at `/home/ubuntu/.codex/sessions/2026/07/19/rollout-2026-07-19T23-11-58-019f7ca6-8f68-7a02-8844-040b14cb1d0f.jsonl:3614` contains the second direct postmortem. Its last logged row was Trial 4 at `2026-07-20T00:50:56.378366`; the worker later confirmed that Trial 5 had built and validated but was never collected, committed, or logged.

## Root Causes

- `plugins/autocuda/skills/explore-brief/SKILL.md` defined the worker loop but did not state that worker identity and assignment survive compaction. Inherited manager conversation therefore became an ambiguous recovery signal.
- The Claude Code, Codex, and Pi `optimize-tree-worker` definitions did not contain a role-recovery invariant or a safe return path when launch arguments are missing.
- `plugins/autocuda/skills/optimize-tree/SKILL.md` told the manager to confirm only the live brief count every 30 minutes and otherwise treat an unnotified quiet worker as alive indefinitely. That detects missing tasks but not a running worker with zero trial progress.

## Fix Plan

1. Add a concise worker-identity section to `explore-brief`: recover `brief-id`, `tag`, and `data-dir`, re-enter the same brief after compaction, never enter the manager loop, and return if the assignment is unavailable. Validate with `pytest plugins/autocuda/tests/test_autocuda_cli.py::test_optimize_tree_worker_agent_is_a_thin_wrapper_over_explore_brief`.
2. Mirror the invariant in all three host definitions: `agents/optimize-tree-worker.md`, `agents/optimize-tree-worker.toml`, and `agents/pi/optimize-tree-worker.md`. Keep a regression assertion over every format.
3. Require every manager launch and continuation prompt to reassert permanent worker role and brief assignment.
4. Replace the count-only timed check with a rate-limited health audit: run `autocuda status` no more than once every 30 minutes, inspect activity/trial/status fields, and checkpoint a running worker with no progress before interrupting it.
5. Run the full plugin test suite and repository hooks, then upstream the complete fix in one pull request.

## Verification

- `autocuda report data skill` generated the companion CSV from the two affected worker harness records and the selected run tag.
- `autocuda schema check report-skill-csv --output 2026-07-19-01-44-06-cholesky-resumed-local-unmerged-worker-compaction` passed.
- Direct harness review confirmed both role-recovery failures and excluded tool/API failures.
- The focused worker-role regression tests passed; the full autocuda suite passed with 665 tests and 10 skips.
- All repository pre-commit hooks passed: whitespace, EOF, line endings, YAML/JSON, conflict markers, large files, Ruff, ShellCheck, and shfmt.
- Upstream fix: [autocuda PR #381](https://github.com/brycelelbach-private/autocuda/pull/381).
- Harness coverage outside Briefs 58 and 63 was not mapped for this targeted incident report.

## Scope & Methodology

The report covers optimize-tree run `2026-07-19-01-44-06-cholesky-resumed-local-unmerged`, specifically branches `autocuda/optimize/2026-07-19-01-44-06-cholesky-resumed-local-unmerged/brief-58` and `brief-63`. Evidence combines the tagged skill-report CSV, both per-brief trial logs, the 06:29 UTC `autocuda status` health audit, and direct Codex harness records including each worker's read-only postmortem. Candidate extraction did not classify the self-reported role failures automatically, so they were added through the report skill's required direct harness review. The other 77 brief branches remain coverage gaps in this focused report.
