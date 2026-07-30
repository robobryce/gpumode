# Autocuda Skill Reliability Report: linalg/cholesky_py

## Executive Summary

The audit found seven distinct skill or CLI-contract defects across autocuda:report-optimization, autocuda:report-cost, autocuda:explore-brief, autocuda:optimize-tree, and the schema CLI help. Four defects are resolved in PRs #373 and #381; three remain open: missing packaged references, unenforced remote timeouts, and the schema-help mismatch. Coverage is complete for this snapshot: 258 parseable Codex segments cover all 93 brief branches and every mapped manager-side segment, with no missing or unreadable harness record.

**Key takeaways**

- Two high-severity report parsers silently rejected or omitted valid run data; both now have regression coverage.
- One high-severity compaction-role defect stopped briefs 58 and 63, while a manager health-check gap delayed detection by 290 and 338 minutes.
- The installed explore-brief package still loses its skill-relative reference links, and timeout limits remain advisory rather than enforced.
- The run was still active at the 2026-07-20 21:12:01 UTC snapshot, so later-created records are outside this report.

## Skill Issue Candidates

| Branch | Skill | Severity | Category | Evidence | Recommendation |
|---|---|---|---|---|---|
| All branches | autocuda:report-cost | High | harness_schema | The first cost build populated only steered/manager; all 27 autonomous rows were blank because the fork parser did not recognize Codex response_item/agent_message handoffs. Resolved in PR #373. | Retain both fork-boundary encodings and the inherited-token subtraction regression test from issue #375. |
| All branches | autocuda:report-optimization | High | schema_contract | Optimization data aborted on valid abbreviated commit 9586dd7 even though the run logs and dashboard accepted abbreviated Git identities. Resolved in PR #373. | Retain unique 7–40 character Git-prefix resolution and the real-repository regression from issue #374. |
| brief-58 | autocuda:explore-brief | High | role_recovery | After compaction the worker adopted the manager role and waited instead of resuming after Trial 26. Resolved in PR #381. | Keep worker identity and brief assignment permanent across compaction; return rather than entering the manager loop if assignment recovery fails. |
| brief-63 | autocuda:explore-brief | High | role_recovery | The same role drift abandoned a built and validated Trial 5 benchmark session before collection, commit, or logging. Resolved in PR #381. | Apply the permanent-role guard in the skill and every host-specific worker definition. |
| briefs 8, 9, 16 | autocuda:explore-brief | Medium | timeout_contract | Four remote jobs produced 773–1937 second trial gaps; no-progress periods reached 757–1846 seconds despite the recorded 90-second allowance. Open. | Enforce the environment timeout at the invocation boundary and log the resolved timeout with every failure. |
| brief-49 | autocuda:explore-brief | Medium | packaging_contract | The installed skill references skill-relative references/profilers.md and references/libraries.md, but its installed references directory is empty; the source checkout contains symlinks that packaging omitted. Open. | Materialize or preserve the referenced files during packaging and add an installed-artifact regression test. |
| manager | autocuda CLI | Medium | cli_help_contract | Top-level help says schema can “define or inspect,” but autocuda schema --help implements only define and check; schema inspect was rejected. Open. | Either implement inspect or change the top-level help to describe define/check exactly, then test both help surfaces together. |
| manager | autocuda:optimize-tree | Medium | health_check | The original 30-minute check counted live tasks and prohibited status inspection, so running-but-unproductive briefs 58 and 63 remained undetected for hours. Resolved in PR #381. | Retain the rate-limited status audit of last activity, trial count, and last status, followed by checkpoint and replacement of stale workers. |

The report-parser evidence is preserved in the manager harness at /home/ubuntu/.codex/sessions/2026/07/19/rollout-2026-07-19T01-42-43-019f780a-39d5-7013-94e3-36648902f851.jsonl:6579 (2026-07-20T06:38:09.496Z), together with the issue and regression results. Current source contains test_canonical_report_commit_resolves_abbreviated_sha and test_fork_scope_accepts_response_item_agent_message_trigger.

The two role failures are direct worker postmortems, not inferred inactivity: /home/ubuntu/.codex/sessions/2026/07/19/rollout-2026-07-19T22-42-00-019f7c8b-1f8f-7d30-ae2a-2b40049a0eb4.jsonl:4084 at 2026-07-20T06:31:14.001Z and /home/ubuntu/.codex/sessions/2026/07/19/rollout-2026-07-19T23-11-58-019f7ca6-8f68-7a02-8844-040b14cb1d0f.jsonl:3614 at 2026-07-20T06:31:20.515Z. Both explicitly exclude command, API, and infrastructure failure.

The manager then confirmed the health-check contract mismatch and measured 290.1 and 338.4 minutes since activity in the manager harness at lines 6300–6312 (2026-07-20T06:29:17.511Z through 06:29:33.233Z). PR #381 added the permanent-role guard and the 30-minute per-brief activity audit now present in the installed skills.

The packaging failure was reported by the brief-49 worker at /home/ubuntu/.codex/sessions/2026/07/19/rollout-2026-07-19T16-52-10-019f7b4a-d91a-7001-a806-195c84e2c63f.jsonl:1648 (2026-07-19T16:52:36.163Z). Inspection confirms that the source skill has two symlinks under skills/explore-brief/references while the installed cache has an empty directory; the shared root reference files happen to exist, but the documented skill-relative paths do not.

The schema-help contradiction is directly reproducible and appears in the manager harness at line 436 (2026-07-19T01:54:46.523Z). The timeout risk is also direct: for example, the brief-16 harness at /home/ubuntu/.codex/sessions/2026/07/19/rollout-2026-07-19T06-38-20-019f7918-def5-7413-8203-5aebab119f1b.jsonl records a 756.856-second wrapped validation ending at 2026-07-19T08:04:23.586Z after the worker manually noticed that the 90-second allowance had been exceeded.

## Root Causes

- plugins/autocuda/lib/_report.py had two producer-consumer contract gaps: commit canonicalization demanded a full SHA at input, and fork scoping recognized only the older handoff record shape. PR #373 moved canonicalization to output and accepted current Codex agent-message boundaries.

- plugins/autocuda/skills/explore-brief/SKILL.md and the three optimize-tree worker definitions originally lacked an invariant preserving worker role and brief identity across compaction. The contemporaneous optimize-tree skill compounded this with a task-count-only health check that could not distinguish a productive worker from a live session in the wrong loop.

- plugins/autocuda/skills/explore-brief/references uses source-tree symlinks to shared reference pages. The plugin cache installation created the directory but did not preserve or dereference those links, leaving valid source paths absent from the installed artifact.

- plugins/autocuda/lib/autocuda_cli.py advertises “inspect” in the top-level schema help while the schema parser exposes define and check. The manager followed the advertised command and hit a deterministic rejection.

- The worker skill tells agents to use environment.md timeouts but its build, validate, and benchmark examples do not enforce them, and autocuda run has no timeout argument. Enforcement therefore depends on each worker remembering to wrap the nested command and manually noticing a hang.

## Fix Plan

1. Keep the PR #373 report fixes and regressions in plugins/autocuda/tests/test_autocuda_reports.py and test_autocuda_report_cost_dedup.py. Validate with pytest plugins/autocuda/tests/test_autocuda_reports.py::test_canonical_report_commit_resolves_abbreviated_sha plugins/autocuda/tests/test_autocuda_report_cost_dedup.py::test_fork_scope_accepts_response_item_agent_message_trigger.

2. Keep the PR #381 permanent-role clauses in explore-brief, optimize-tree, and all three worker definitions, plus the 30-minute status activity audit. Validate with pytest plugins/autocuda/tests/test_autocuda_cli.py::test_optimize_tree_worker_agent_is_a_thin_wrapper_over_explore_brief and a fixture that compacts a worker handoff without changing its brief identity.

3. Replace the skill-relative reference symlinks with packaged files, or teach the plugin packager to dereference internal links. Add assertions that every references/*.md path named by each installed SKILL.md is a readable regular file; validate with pytest plugins/autocuda/tests/test_packaging.py.

4. Make the schema command description and implemented verbs agree. Validate with autocuda --help, autocuda schema --help, and a CLI test that asserts the parent help names only accepted subcommands.

5. Add an explicit timeout option to autocuda run or require a concrete timeout wrapper in every skill command example, resolving the value from environment.md. Validate with a short hanging-child test that exits at the configured limit, releases its lock, and produces a runtime_error row.

## Verification

- autocuda report data skill regenerated the companion CSV from all 258 mapped Codex JSONLs; autocuda schema check report-skill-csv --output 2026-07-19-01-44-06-cholesky-resumed-local-unmerged passed.

- Direct review covered every mapped source: 184 fleet segments spanning all 93 briefs plus 74 steered segments (27 manager, 22 reward-audit, 15 dashboard-fix, and 10 run-recovery). Every JSONL parsed; no source was missing or unreadable.

- The extractor produced five candidates. One survived as the schema-help mismatch. Brief 43's bad run-slice flags were an agent retry error; two candidates were quoted test-fixture search output; and the traceback came from invoking a development checkout console script outside its import path and succeeded on retry.

- Direct harness review added the report-parser defects, two role incidents, health-check gap, packaging defect, and timeout-enforcement gap that pattern extraction could not identify reliably. Ordinary kernel failures and the active run's temporary leaderboard lag were excluded.

- autocuda schema check report-skill-markdown --output 2026-07-19-01-44-06-cholesky-resumed-local-unmerged passed, and autocuda report html rendered the HTML artifact.

- Evidence gaps at the snapshot are zero. The run remained active, so this is a complete snapshot rather than a terminal audit.

## Scope & Methodology

This report covers optimize-tree run 2026-07-19-01-44-06-cholesky-resumed-local-unmerged through the skill CSV write at 2026-07-20 21:12:01 UTC. The mapping uses first-line Codex agent paths, with two ownership corrections: /root/brief49_resume/brief58 belongs to brief 58, and /root/brief71/brief49_wrap belongs to brief 49. Four local Claude JSONLs were excluded as pre-run sdk-cli model probes, and the export ref contains Codex history memory but no rollout JSONLs.

The audit combined automatic candidate extraction, candidate-context triage, direct reading of all mapped harness records, installed-versus-source skill inspection, current CLI help, experiment logs, behavior-gap evidence, and the fixes/tests already created during this run. The run was ongoing at cutoff: briefs 78, 81, 89, 91, and 92 had no brief_stop near the snapshot, and any later record or incident belongs to a subsequent report refresh.
