---
name: aab-deepseek-double-1m-suffix
description: "~/.aab/.env DEEPSEEK_MODEL carries a [1m] the wrapper doubles; benign for Claude Code, breaks raw consumers"
metadata: 
  node_type: memory
  type: project
  originSessionId: bba6c7f7-dfe5-4105-aee3-9ee27c28bae0
---

In `~/.aab/.env`, `AAB_CLAUDE_CODE_THIRD_PARTY_DEEPSEEK_MODEL` is set to
`nvidia/deepseek-ai/deepseek-v4-pro[1m]` — but the `claude-third-party-deepseek`
wrapper appends its own `[1m]` (bootstrap.bash ~line 2146), yielding
`...deepseek-v4-pro[1m][1m]`. The contract is that the env value must NOT carry
`[1m]` (the `third-party-anthropic` arm does not append; the deepseek/nemotron
arms do).

**Why:** Claude Code strips ALL trailing `[1m]` before the request (verified with
a logging proxy: it sent the clean `nvidia/deepseek-ai/deepseek-v4-pro`), so the
doubling is harmless *for Claude Code*. But any tool that uses the raw env value
sends `...deepseek-v4-pro[1m]`, which the gateway rejects with HTTP 401 "key not
allowed to access model ... Tried to access ...deepseek-v4-pro[1m]" (that id is
not in the catalog; the clean id is).

**How to apply:** If raw-config consumers 401, fix the env value to
`nvidia/deepseek-ai/deepseek-v4-pro` (no suffix) — re-run bootstrap or hand-edit.
Don't "fix" it by editing the wrapper's append; that would break the Claude Code
context-window/[1m] convention. Real DeepSeek V4 Pro id on the hub:
`nvidia/deepseek-ai/deepseek-v4-pro` (also `azure/` and `baseten/` variants).
Related: [[autocuda-pr-fanout-export-overhaul]].
