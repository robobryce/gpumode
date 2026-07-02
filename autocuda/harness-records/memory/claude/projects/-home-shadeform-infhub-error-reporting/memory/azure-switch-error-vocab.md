---
name: azure-switch-error-vocab
description: InfHub switched Bedrock→Azure on 2026-06-22; socket-disconnect error strings changed and audit script under-counts
metadata: 
  node_type: memory
  type: project
  originSessionId: 4455881f-eb61-436a-a890-0a983ff959ae
---

The autocuda/InfHub fleet switched model routing from AWS Bedrock to Azure on
**2026-06-22T11:09Z** (first `azure/anthropic/claude-opus-4-8` turn; Bedrock's
last turn was 2026-06-20). Model-id eras in `~/.claude/projects/**/*.jsonl`:
`claude-opus-4-7` (→05-28), `claude-opus-4-8` / `aws/anthropic/bedrock-claude-opus-4-8`
(05-29→06-20), `azure/anthropic/claude-opus-4-8` (06-22→).

The socket-disconnect error message **changed wording** with the switch:
- Bedrock: `API Error: the socket connection was closed unexpectedly. ...`
- Azure: `API Error: Connection closed mid-response. The response above may be incomplete.`

The renamed disconnect is still caught by `audit_socket_errors.py`'s generic
`"api error"+"connection"` rule. BUT measuring against `isApiErrorMessage:true`
turns (Claude Code's authoritative "turn died" flag), the script only catches
~18-20% of API-error turns in BOTH eras. The dominant Azure failures it MISSES:
`API Error: Request rejected (529) · status code (no body)` (capacity/overload,
~63% of Azure API-error turns), `repeated N Overloaded errors`, `internal server
error`, `502 Bad gateway`. These are distinct from socket disconnects (capacity/
5xx) and should be classified separately, not folded into the socket-disconnect
headline metric. See [[infhub-report-enrichment-fields]].

**Why:** The 2026-06-15 report predates the Azure switch entirely (0 Azure data).
**How to apply:** Any "is the report accurate" question must check era coverage
and that Azure error families are captured + classified separately.
