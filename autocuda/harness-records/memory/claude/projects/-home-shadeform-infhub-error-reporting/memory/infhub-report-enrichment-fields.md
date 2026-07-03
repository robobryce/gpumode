---
name: infhub-report-enrichment-fields
description: "InfHub team's required per-failure fields + the analyzer/report scripts and key root-cause findings"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4455881f-eb61-436a-a890-0a983ff959ae
---

The inference-hub team wants, per socket/API failure: node IP + date/time,
seconds-since-last-request, context-window size, and failed-request token count.
Plus exhaustive coverage ("every single error, no truncation").

Scripts in `infhub-error-reporting/scripts/` (all stdlib + matplotlib + gh):
- `audit_socket_errors.py` — aggregate report (out/report.md + CSV/JSON/charts).
  Fixed 2026-06-24 to capture Azure-era families (see [[azure-switch-error-vocab]]):
  classes model_api_socket_disconnect / capacity_overload / server_error_5xx /
  model_api_other / input_rejected. Socket headline stays socket-only;
  api_failures = socket+capacity+5xx+other.
- `analyze_api_failures.py` — EXHAUSTIVE per-failure instrument. Scans every
  local Claude session JSONL (~/.claude/projects) AND every cached remote-export
  session (.cache/raw, harness-records/claude-code + raw-sessions), parses every
  line, dedups export copies by (session,ts,line,err-prefix), and CROSS-CHECKS
  the count with an independent regex recount (must match raw pre-dedup count).
  Derives token/timing fields from the preceding real assistant turn (the failed
  turn itself is synthetic/all-zero usage). Outputs out/api_failures.{csv,json}
  + api_failures_coverage.csv + api_failures_report.md.
- `build_failure_windows.py` — emits out/failure_windows/<family>.json: compact
  pre-error context windows (tool calls, result sizes, stream state) for agent
  analysis without touching the 5 GB of raw logs.

KEY FINDINGS (run 2026-06-24, 898 deduped failures across 6,579 files, x-check OK):
- **Socket disconnects cluster at 240–305s since last request** (88% Bedrock-4.8,
  85% Azure, both ~250s median) = ~5-min gateway/LB response-timeout signature,
  PERSISTENT across the Bedrock→Azure switch → shared request-layer/gateway, not
  the model backend. capacity_overload does NOT cluster there (0% in-band,
  median 187s) — a distinct load-driven mode.
- Failed requests skew to large context (network/server p90≈616k vs baseline
  p90≈337k tokens) → long requests on big contexts are likeliest to be cut.
- Family mix: capacity_overload 312, input_rejected 277 (huge ctx, client-side,
  excluded from network metric), socket_disconnect 221, server_5xx 61, other 27.
