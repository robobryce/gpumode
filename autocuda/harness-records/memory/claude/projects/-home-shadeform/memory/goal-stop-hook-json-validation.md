---
name: goal-stop-hook-json-validation
description: "Why Claude Code's /goal shows \"Stop hook error: JSON validation failed\" — the prompt hook fails OPEN, which silently kills the goal loop (NOT cosmetic)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: d0666fc5-aba3-41ef-9a45-ce5c8da60bb7
---

`/goal <condition>` is a built-in Claude Code feature (NOT a plugin, NOT in settings.json): it registers a prompt-based Stop hook whose model-backed evaluator reads the transcript on each Stop and must reply with bare JSON (`{ok, reason}`). When the evaluator replies in prose/markdown instead, Claude Code prints `Stop hook error: JSON validation failed` (transcript: `stop_hook_summary` with `hookErrors:["JSON validation failed"]`).

**CORRECTION (do not call this cosmetic).** The hook's JOB is to BLOCK the stop until the goal holds. On a JSON parse failure it FAILS OPEN: `preventedContinuation=false` → the stop is PERMITTED → the goal loop halts with the condition UNMET. For a keep-going feature, fail-open is the wrong-direction failure. There is no auto-retry: a fail-open stop returns the session to idle, and a Stop hook only re-fires on the next turn's stop attempt — so the loop is DEAD until a human re-prompts. Confirmed in session 1173e1df: after the 23:29:03 fail-open, the next activity was the user typing a new prompt ~4 min later. Severity: interactive = annoying/recoverable; UNATTENDED autonomous run (the autocuda fleet, where these fired) = materially broken. Upstream defect anthropics/claude-code #11947 — a Stop-blocking evaluator that fails open on parse error is safety-backwards.

The earlier 2026-06-09 (session 56b28751) and first 2026-06-29 pass both concluded "cosmetic/harmless" — that judgment was WRONG; the user pushed back correctly. The direction ("fails open") was right; the value call was inverted.

REAL FIX (the user rejected "switch to optimize-hill/loop" as a non-fix — it abandons the LLM-judged-condition capability instead of fixing it). The built-in /goal CANNOT be fixed locally: the fail-open branch is inside a stripped native ELF at ~/.local/share/claude/versions/<ver> (the `claude` on PATH is an AAB bash wrapper around ~/.local/bin/claude → that ELF); auto-update clobbers any binary patch; the only stop-hook env knobs (CLAUDE_CODE_STOP_HOOK_BLOCK_CAP, CLAUDE_CODE_EXIT_AFTER_STOP_DELAY) don't change the evaluator or fail direction. So the two real fixes are: (A) UPSTREAM (#11947) — not local; can draft a precise report (fail-open on parse error + unbounded evaluator prompt). (B) SELF-HOST the capability as a Stop hook in settings.json (update-proof, in user control) that FAILS CLOSED (re-block + retry, cap ~3 then surface loudly) and BOUNDS the evaluator input to last N msgs. Two flavors: LLM-judge hook (calls `claude -p` on bounded transcript — keeps "model decides") OR promise-tag hook (loop continues until main agent emits `<promise>...</promise>`, exact-string match — ZERO LLM-JSON/prompt-length exposure; the installed-but-unenabled ralph-loop plugin already implements this). Recommend promise-tag for unattended fleets.

Likely trigger = evaluator PROMPT OVERFLOW: binary has an `evaluator prompt too long; retrying` code path; the one ground-truth failure (1173e1df 23:29:03) took durationMs=8261 (~8.3s), consistent with oversized prompt on a long run. NOT proven causal from logs (the "prompt too long" string hits were mostly investigation artifacts, not stop_hook_summary telemetry — verify before asserting). Crisp single-clause goals reduce frequency but do NOT make fail-open safe; the offending goal was compound+unverifiable: "There must be 4 worker agents running. You can never run out of ideas. Re-read optimize-tree regularly."

The `ralph-loop` plugin also ships a Stop hook, but it's a different command-type mechanism and was NOT enabled (only agitentic + autocuda are) — not the cause.
