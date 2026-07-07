# Codex `wait_agent` Short-Wait Regression in AutoCUDA `optimize-tree`

Date: 2026-07-07 UTC

## Executive Summary

The recent AutoCUDA manager behavior—issuing a `wait_agent` call roughly every
30 seconds instead of waiting for worker completion with long timeouts—is real,
but it is **not caused by a new timeout restriction in the open-source
`wait_agent` implementation or by a recent AutoCUDA skill change**.

The primary cause is a new model-specific base-instruction bundle delivered to
GPT-5.6 models through the Codex backend model catalog. It contains two rules
that, together, strongly bias the manager toward short waits:

- do not leave the user without a commentary update for more than 60 seconds;
- avoid blocking sleep or wait calls longer than 60 seconds.

Those rules are present for GPT-5.6-Sol, GPT-5.6-Terra, and GPT-5.6-Luna in the
current model catalog. They are absent from the current GPT-5.5, GPT-5.4,
GPT-5.4-Mini, GPT-5.3-Codex-Spark, GPT-OSS, and Codex Auto Review instruction
bundles.

Local session evidence places the rollout on this machine on 2026-07-05. The
same Codex CLI version (`0.142.3`) and model (`gpt-5.6-sol`) used long waits in
the older instruction environment and 30-second waits in the new environment.
The rollout appears to have been staggered or cache-dependent, so the exact
global backend deployment time cannot be recovered from the public Codex Git
history.

The current tool still supports a maximum wait of one hour. A single
`wait_agent` call was never literally infinite; the manager achieved
indefinite monitoring by issuing long or repeated event-driven waits until a
worker reached a terminal state. AutoCUDA's `optimize-tree` skill explicitly
requires this notification-driven behavior and allows a 30-minute fleet health
check, so the new GPT-5.6 prompt conflicts with the workflow it is supposed to
execute.

## User-Visible Impact

The regression changes the manager from an event-driven coordinator into a
frequent timeout loop:

- Current run: 530 recorded `wait_agent` calls; 527 explicitly requested
  `30000` ms, with the remaining three requesting 1, 10, and 20 seconds.
- Earlier run, on 2026-07-03: all 62 waits issued that day requested between
  60 seconds and 30 minutes; 31 requested 30 minutes and 10 requested 20
  minutes.
- A 30-second timeout can produce up to 120 manager turns per hour while no
  worker completes. A 30-minute wait needs at most two timeout turns per hour,
  and either duration may return early when relevant activity arrives.

Consequences include additional inference cost, context growth and compaction,
UI noise, repeated tool calls, and a direct mismatch with AutoCUDA's instruction
to wait for completion notifications rather than poll the fleet.

## Findings

| Finding | Conclusion | Confidence |
|---|---|---:|
| Did `wait_agent` recently acquire a 60-second cap? | No. Current maximum is 3,600,000 ms (one hour). | High |
| Did its default recently fall to 30 seconds? | No. The default has been 30 seconds since the first implementation. | High |
| Did AutoCUDA recently tell the manager to wait for only 30 seconds? | No. Its skill says to wait for completion and permits a 30-minute health check. | High |
| Is the manager explicitly choosing 30 seconds? | Yes. The affected rollout records `{"timeout_ms":30000}` on nearly every call. | High |
| What changed? | GPT-5.6's backend-supplied base instructions added explicit 60-second communication/wait guidance. | High |
| When did it change? | First local capture: 2026-07-05 14:49 UTC; same-provider capture by 20:48 UTC. | High locally; global time unknown |
| Do other models receive different instructions? | Yes. Only the GPT-5.6 family in the current catalog contains both 60-second rules. | High |

## Local Session Evidence

Codex persists the resolved base instructions in the first `session_meta`
record of each rollout JSONL. It also records each model-generated function
call, including the explicit `timeout_ms` supplied to `wait_agent`.

### Instruction rollout

| Session start (UTC) | Model/provider | Prompt state | Notes |
|---|---|---|---|
| 2026-07-03 13:37:43 | `gpt-5.6-sol` / `openai` | Old; neither 60-second rule | CLI `0.142.3`; decoded base prompt was 21,348 bytes. |
| 2026-07-05 14:49:12 | `gpt-5.6-sol` / `third-party-openai` | New; both rules present | Earliest local session containing the new bundle; CLI still `0.142.3`. |
| 2026-07-05 19:25:10 | `gpt-5.6-sol` / `openai` | Old bundle recorded | Evidence of staged rollout, process cache, or session/model prompt stickiness. |
| 2026-07-05 20:48:02 | `gpt-5.6-sol` / `openai` | New; both rules present | Confirms the new bundle on the built-in OpenAI provider. |
| 2026-07-06 18:18:35 | `gpt-5.6-sol` / `openai` | New; both rules present | Affected AutoCUDA manager session. |

Relevant rollout files:

- Old bundle:
  `/home/shadeform/.codex/sessions/2026/07/03/rollout-2026-07-03T13-37-30-019f2832-e134-7e31-9969-1d56b2f65234.jsonl`
- First local new-bundle capture:
  `/home/shadeform/.codex/sessions/2026/07/05/rollout-2026-07-05T14-49-06-019f32c1-26a1-7dc0-b13e-a3e483817986.jsonl`
- Intervening old-bundle OpenAI-provider session:
  `/home/shadeform/.codex/sessions/2026/07/05/rollout-2026-07-05T19-25-03-019f33bd-c762-7892-96da-99c54cca25f3.jsonl`
- First new-bundle OpenAI-provider capture:
  `/home/shadeform/.codex/sessions/2026/07/05/rollout-2026-07-05T20-47-55-019f3409-a744-7d61-97c4-d3aee4449fcc.jsonl`
- Affected manager:
  `/home/shadeform/.codex/sessions/2026/07/06/rollout-2026-07-06T18-18-28-019f38a7-30b3-7bf3-ac20-1a4c4db7be75.jsonl`

### Wait-call comparison

The beginning of the older manager session used waits such as:

```json
{"timeout_ms":1800000}
{"timeout_ms":1200000}
{"timeout_ms":480000}
{"timeout_ms":1800000}
```

The affected manager settled into:

```json
{"timeout_ms":30000}
{"timeout_ms":30000}
{"timeout_ms":30000}
```

The current manager is not merely accepting the tool's 30-second default. It
is explicitly supplying `30000`; therefore changing only the configured
default timeout will not change this behavior.

The following command reproduces the timeout distribution for a rollout:

```bash
jq -r '
  select(.type == "response_item"
         and .payload.type == "function_call"
         and (.payload.name | endswith("wait_agent")))
  | .payload.arguments | fromjson | .timeout_ms
' ROLLOUT.jsonl | sort -n | uniq -c
```

## Open-Source `wait_agent` History

The public history contradicts the hypothesis that Codex recently shortened
the tool's maximum wait.

| Date | Commit | Change |
|---|---|---|
| 2026-01-09 | [`568b938`](https://github.com/openai/codex/commit/568b938c80a3454a3aa091b4ba20636662dea86b) | First collaboration-tool skeleton: 30-second default, 5-minute maximum; wait not yet implemented. |
| 2026-01-12 | [`623707a`](https://github.com/openai/codex/commit/623707ab586e233247b267d8d6151c51313c3a22) | Implemented waiting on subagent terminal status with the same default and maximum. |
| 2026-01-26 | [`375a5ef`](https://github.com/openai/codex/commit/375a5ef05116a3d289dc548402387f86e28851cc) | Added a 10-second minimum to prevent busy polling. |
| 2026-02-22 | [`4666a6e`](https://github.com/openai/codex/commit/4666a6e631567f37366f9ce19905d16ce62b36f2) | Raised the maximum from 5 minutes to one hour. |
| 2026-03-13 | [`cfd97b3`](https://github.com/openai/codex/commit/cfd97b36da76a17db407b2d9653ed993636e0a30) | Renamed the tool from `wait` to `wait_agent`. |
| 2026-03-23 | [`450dc28`](https://github.com/openai/codex/commit/450dc289c3305bf9d94d862d6d30c4916aa2497a) | Multi-agent v2 retained 10-second minimum, 30-second default, and one-hour maximum. |
| 2026-04-28 | [`34d71d4`](https://github.com/openai/codex/commit/34d71d43eb87e16429a3945ec3de5799ea2153c0) | Made the v2 minimum configurable. |
| 2026-05-13 | [`7c57a59`](https://github.com/openai/codex/commit/7c57a59f51b605b82f49e71833aa6e675b9ec54c) | Made v2 default and maximum configurable; retained a hard one-hour ceiling. |
| 2026-06-15 | [`ee40ddd`](https://github.com/openai/codex/commit/ee40dddbf6a50a6f0641180ca299be7a3a03fd22) | Allowed new user steering to interrupt v2 waits; did not shorten the timeout. |

In Codex `0.142.3`, the defaults are defined as:

```rust
DEFAULT_MULTI_AGENT_V2_MIN_WAIT_TIMEOUT_MS = 10_000;
DEFAULT_MULTI_AGENT_V2_MAX_WAIT_TIMEOUT_MS = 3600 * 1000;
DEFAULT_MULTI_AGENT_V2_DEFAULT_WAIT_TIMEOUT_MS = 30_000;
```

The v2 handler accepts an explicit timeout between the configured minimum and
maximum and creates its deadline directly from that value. See:

- [`config/mod.rs`](https://github.com/openai/codex/blob/rust-v0.142.3/codex-rs/core/src/config/mod.rs#L207-L209)
- [`multi_agents_v2/wait.rs`](https://github.com/openai/codex/blob/rust-v0.142.3/codex-rs/core/src/tools/handlers/multi_agents_v2/wait.rs#L51-L67)

The only sense in which prior behavior was "forever" was at the manager level:
it continued issuing event-driven waits until a completion notification
arrived. Every individual tool invocation still had a finite deadline.

## Where the New Instructions Come From

The two 60-second rules are not present in any commit in the public Codex
repository. A full `git log -S` search for both sentences returns no commits.
They are backend model metadata.

The delivery path in Codex `0.142.3` is:

1. Codex requests `GET /models?client_version=0.142.3`.
2. Each returned `ModelInfo` contains a model-specific `base_instructions`
   string.
3. Codex caches the response in `~/.codex/models_cache.json` with a five-minute
   default TTL and an HTTP ETag.
4. At session creation, Codex resolves base instructions in this order:
   configured override, instructions stored in resumed session history, then
   current model instructions.
5. The resolved text is sent as the top-level Responses API `instructions`
   field and persisted in the rollout's `session_meta` record.

Relevant source locations:

- [`codex-api/src/endpoint/models.rs`](https://github.com/openai/codex/blob/rust-v0.142.3/codex-rs/codex-api/src/endpoint/models.rs#L31-L64)
- [`protocol/src/openai_models.rs`](https://github.com/openai/codex/blob/rust-v0.142.3/codex-rs/protocol/src/openai_models.rs#L349-L370)
- [`models-manager/src/manager.rs`](https://github.com/openai/codex/blob/rust-v0.142.3/codex-rs/models-manager/src/manager.rs#L298-L340)
- [`core/src/session/mod.rs`](https://github.com/openai/codex/blob/rust-v0.142.3/codex-rs/core/src/session/mod.rs#L586-L602)
- [`core/src/client.rs`](https://github.com/openai/codex/blob/rust-v0.142.3/codex-rs/core/src/client.rs#L790-L826)

For ChatGPT authentication, the built-in provider selects
`https://chatgpt.com/backend-api/codex` as its base URL. Thus, for this auth
mode, the model catalog and its base instructions are server-controlled even
though the client code that consumes them is open source.

Current local cache metadata at the time of this report:

```text
fetched_at:    2026-07-07T00:20:22.568662803Z
client_version: 0.142.3
etag:          W/"e579022bc1fe0d0ffcf42b3ad20003de"
```

## Model-Specific Differences

The current `~/.codex/models_cache.json` contains the following result:

| Model | Catalog multi-agent version | Commentary ≤60s | Avoid waits >60s |
|---|---:|---:|---:|
| `gpt-5.6-sol` | v2 | Yes | Yes |
| `gpt-5.6-terra` | v2 | Yes | Yes |
| `gpt-5.6-luna` | v1 | Yes | Yes |
| `gpt-5.5` | Not specified | No | No |
| `gpt-5.4` | Not specified | No | No |
| `gpt-5.4-mini` | Not specified | No | No |
| `gpt-5.3-codex-spark` | Not specified | No | No |
| `gpt-oss-120b` | Not specified | No | No |
| `gpt-oss-20b` | Not specified | No | No |
| `codex-auto-review` | Not specified | No | No |

Codex has supported model-specific prompts since commit
[`916fdc2`](https://github.com/openai/codex/commit/916fdc2a37b40e4cc44f4512eea7159eb09cb252)
on 2025-09-14. Model-specific instruction infrastructure is therefore not new;
the material change is the GPT-5.6 instruction content returned by the backend.

Switching to a non-GPT-5.6 model would avoid these exact sentences, but the
current catalog does not advertise a multi-agent version for those models.
That makes model switching a behavior-changing workaround rather than a safe
fix for `optimize-tree`.

## AutoCUDA Skill History and Conflict

The installed AutoCUDA skill points in the opposite direction from the new
GPT-5.6 prompt:

- Step 7 establishes a 30-minute brief-count reminder.
- Step 8 says to wait for a worker completion notification.
- The waiting section says that notification is the only signal on which the
  manager acts.
- It forbids polling logs, processes, or `autocuda status` between worker
  returns.
- The management loop is explicitly indefinite until operator interruption.

Installed file:

```text
/home/shadeform/.codex/plugins/cache/brycelelbach-autocuda/autocuda/0.4.0/skills/optimize-tree/SKILL.md
```

Relevant AutoCUDA history:

| Date | Commit | Change |
|---|---|---|
| 2026-06-08 | `92028e15f1f4ce9c...` | Replaced fleet watching with waiting for harness completion notifications. |
| 2026-06-19 | `4a0b5de45a70610d...` | Defined a 30-minute worker-count health check through the harness task-listing facility. |
| 2026-06-29 | `a59edbfcf47eee9d...` | Retained completion-driven waiting and the 30-minute health check in the ephemeral-worker rewrite. |

The installed skill matches the June 29 lineage. There was no corresponding
AutoCUDA change on July 5 that would explain the sudden 30-second calls.

This creates a direct instruction conflict:

| AutoCUDA workflow requirement | GPT-5.6 base guidance |
|---|---|
| Wait for the completion notification; do not poll. | Avoid a blocking wait longer than 60 seconds. |
| Check fleet/brief health every 30 minutes. | Communicate with the user within 60 seconds. |
| Continue indefinitely until interrupted. | Re-enter the model after each short wait to remain communicative. |

The model currently resolves that conflict in favor of short `wait_agent`
timeouts, despite the explicitly invoked skill's event-driven design.

## Root Cause

### Primary cause

A backend update to GPT-5.6 model `base_instructions` introduced an explicit
prohibition against blocking wait calls longer than 60 seconds. The model now
translates that general communication policy into explicit 30-second
`wait_agent` arguments.

### Contributing factors

1. The tool's default is 30 seconds, providing a salient value even though the
   manager supplies it explicitly.
2. The instruction does not distinguish an event-driven collaboration wait
   from a blocking shell sleep. `wait_agent` can wake early on mailbox activity,
   completion, user steering, or its deadline, but the prompt treats it as a
   generic blocking call.
3. Base instructions are session-sticky. Resumed sessions keep the prompt
   stored in `session_meta`, producing temporarily inconsistent behavior during
   a staged rollout or across long-running sessions.
4. The model catalog is cached, so separate processes can observe a backend
   prompt update at different times.

### Ruled-out causes

- No user-created `AGENTS.md` instruction introduced the 60-second rules.
- No local `developer_instructions` setting introduced them.
- The sentences are not compiled into the Codex binary or present in its Git
  history.
- `wait_agent` did not gain a 60-second maximum.
- The AutoCUDA skill did not change to request 30-second waits.

## Remediation Options

### 1. Preferred upstream prompt correction

Narrow the GPT-5.6 rule so that event-driven monitoring tools are exempt. For
example:

> Avoid long blocking shell sleeps or polling loops. Event-driven monitoring
> tools such as `wait_agent` may use the workflow-defined timeout because they
> wake on completion, mailbox activity, or user steering. While delegated work
> is running, honor the active skill's monitoring cadence.

This preserves the user-communication goal without forcing repeated model
turns when the workflow intentionally waits for an external event.

### 2. Improve the tool/harness contract

Codex could make the distinction structural instead of prompt-dependent:

- mark `wait_agent` as communication-safe/event-driven metadata;
- allow a manager to wait "until notification" without choosing a timeout;
- keep a one-hour internal safety ceiling while automatically renewing the wait
  without another inference turn;
- allow user steering to interrupt the wait, as v2 already does;
- expose the active base-instruction source, model-catalog ETag, and instruction
  hash in `/status` or diagnostics.

### 3. Local base-instruction override

Codex officially supports `model_instructions_file`, which replaces the
built-in/model-provided base instructions with a file. The durable local
workaround is to copy the current desired GPT-5.6 prompt, revise the two
60-second rules with an event-driven-wait exemption, and configure:

```toml
model_instructions_file = "/absolute/path/to/codex-instructions.md"
```

This is a **complete base-prompt replacement**, not a patch. It must therefore
be maintained as the upstream prompt evolves. A project-scoped
`.codex/config.toml` can use a relative path resolved from its containing
`.codex/` directory. See the official
[Codex configuration reference](https://developers.openai.com/codex/config-reference#configtoml).

An additive `developer_instructions` entry can state that `wait_agent` is
exempt, but it does not remove the conflicting base rule and is consequently
less deterministic than replacing the base instructions.

### 4. Timeout configuration as a secondary measure

Codex `0.142.3` supports:

```toml
[features.multi_agent_v2]
min_wait_timeout_ms = 10000
default_wait_timeout_ms = 1800000
max_wait_timeout_ms = 3600000
```

This makes omitted timeouts default to 30 minutes and retains the one-hour
maximum. It is useful after correcting the prompt, but it does **not** fix the
observed run by itself because the model explicitly supplies `30000`.

### 5. Do not patch the cache

Editing `~/.codex/models_cache.json` is not durable. The cache has a five-minute
TTL and is rewritten from the backend model catalog. It is useful as diagnostic
evidence, not as a configuration surface.

## Recommended Fix Order

1. Add an event-driven `wait_agent` exemption to the GPT-5.6 base prompt.
2. Start a fresh session so the old `session_meta.base_instructions` is not
   reused.
3. Set the v2 default timeout to 30 minutes as a defensive default.
4. Run a controlled `optimize-tree` test with workers lasting more than two
   minutes.
5. Confirm the manager uses a long explicit timeout or omits the timeout, wakes
   immediately on completion, and performs only the skill's 30-minute health
   check when no completion arrives.
6. Upstream a Codex change that treats collaboration waits as event-driven so
   future model-prompt updates cannot recreate the regression.

## Verification Plan

A successful fix should satisfy all of the following:

- Fresh rollout `session_meta.base_instructions` contains the intended
  event-driven-wait exception.
- A worker running for at least five minutes does not cause ten 30-second
  manager timeout turns.
- The manager requests 1,800,000–3,600,000 ms, or uses a true
  wait-until-notification mechanism.
- Worker completion wakes the manager before the deadline.
- User steering interrupts the wait promptly.
- The manager does not poll logs, processes, or `autocuda status`.
- The manager performs at most the workflow-defined 30-minute brief-count
  health check.
- The optimize-tree loop continues after context compaction and does not return
  a terminal summary unless the operator interrupts it.

## Limitations

- The exact backend deployment timestamp is not available in the public Codex
  repository because model-catalog payload history is not public.
- Local evidence establishes when this machine observed the prompt, not when
  every account or deployment received it.
- The July 5 mixture of old and new prompt bundles could reflect staged backend
  rollout, per-process cache timing, session prompt stickiness, or a combination
  of those mechanisms.
- Model behavior is not determined by one sentence alone. The observed
  before/after tool arguments, unchanged CLI version, current model catalog,
  and absence of a matching source/skill change together make the prompt
  rollout the strongest supported explanation.

## Conclusion

The manager's short waiting cadence is a cross-layer prompt regression. The
open-source tool supports one-hour waits, and AutoCUDA explicitly asks for
notification-driven waiting with a 30-minute health cadence. GPT-5.6's newly
delivered general communication rule collapses that design into repeated
30-second calls.

The correct fix is not to increase the tool's hard limit—it is already one
hour. The fix is to distinguish event-driven collaboration waits from blocking
sleep/polling in the GPT-5.6 instructions, then use the existing configurable
long timeout as a defensive default.
