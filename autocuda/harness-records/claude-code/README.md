# Claude Code harness records

The historical top-level records described below belong to qr_v2 run
`2026-06-22-09-10-03-qr_v2`.

Additional run-scoped records live under `by-run/<experiment-tag>/`. In
particular, `by-run/2026-07-03-13-39-25/` contains the scrubbed Claude Code
session that made the final optimization-report Markdown/HTML edits for that
experiment, together with a provenance manifest and run-specific README.

This directory holds the **Claude Code session logs** for the `linalg/qr_v2`
`autocuda:optimize-tree` run (batched square Householder QR on a single B200),
which produced the data under `autocuda/` and the board-accepted winning commit
`c508b154` (2040.9 µs, 64.2× over the `torch.geqrf` baseline).

## Layout

One subdirectory + top-level `.jsonl` per Claude Code session:

- **`cff4df73-32b1-46cb-8e5a-58ad5f4046ff.jsonl`** — the **manager** session
  (the `optimize-tree` orchestrator). Spans 2026-06-22T09:05Z → 2026-06-24T18:09Z
  (the full run). ~20 MB.
  - `cff4df73-.../subagents/agent-*.jsonl` — **444 worker sub-agent transcripts**
    (every `optimize-tree-worker` / `explore-brief` instance the manager spawned
    or resumed across the run, plus the report sub-agents). This is the bulk of
    the record — the per-brief edit→build→validate→benchmark→profile→commit loops.
  - `cff4df73-.../workflows/` — workflow orchestration scripts + run journals
    (`wf_*.json`, `scripts/*.js`) for the deep-research / multi-agent workflows.
  - `cff4df73-.../tool-results/*.txt` — persisted large tool outputs that were
    spilled to disk during the session (text only).
- **`4986b236-320c-4afc-820b-6c478a027cc1.jsonl`** — a short auxiliary qr_v2
  session on 2026-06-24 (same run; 84 qr_v2 references).

## What was excluded and why

- **`tool-results/*.jpg` and `tool-results/*.pdf`** (~147 MB of screenshots and
  research PDFs) — binary artefacts, not session logs. The `autocuda export`
  binary-exclude filter drops these anyway; omitting them keeps the snapshot to
  the actual conversation/transcript record.
- The older `00dc36b3-...` session (2026-06-12/13) belongs to a **different,
  earlier** run and is not part of this tag.

## Provenance

Copied verbatim from `~/.claude/projects/-home-shadeform-gpumode/` on the run
host. Scanned for live secrets (GitHub/Anthropic/AWS/Slack tokens, SSH private
keys, popcorn `cli_id`) before staging — zero hits; the only `cli_id` string
matches are the literal word in prose, not a credential value.
