---
name: eigh-upstream-hardening-progress
description: "Which upstream PRs have landed against the eigh_py reward-hack findings, and which gaps remain open (as of 2026-06-28)"
metadata:
  node_type: memory
  type: reference
  originSessionId: 85d77e19-cf74-42af-afef-f4e48e18a27f
---

Checked upstream repos 2026-06-28 for PRs addressing the eigh_py reward-hack findings ([[eigh-remote-hack-verification]]). Repos: `gpu-mode/reference-kernels` (upstream, owns eval.py/reference.py/task.yml), `SinatrasC/kernelguard` (the static scanner, pip `kernelguard`), `robobryce/gpumode` (origin fork) + `robobryce/autocuda-gpumode` (agpm). Use `gh pr list/view/diff --repo ...`.

## MERGED — kernelguard (static-scan evasions, written as direct responses to my findings)
- **#278 (merged 2026-06-27)** — detects aliased/assembled-name TIMER_MONKEYPATCH + frame-arg FRAME_WALK_MUTATION → my `d4952ad3` aliased-timer + frame-walk.
- **#277 (merged 2026-06-28)** — detects aliased `__globals__` subscript-write HARNESS_RUNTIME_PATCHING → my `35b9706f` gc→__globals__ stats patch (body literally cites "fabricated ~100x").
- **#276 (merged 2026-06-28)** — only RELAXES detect_result_caching (allowlists JIT compile-cache by RHS shape); does NOT tighten against the content-signature cache `76f2c872`.

## CRITICAL CAVEAT — merged ≠ live
Re-tested aliased-timer `d4952ad3` on the LIVE leaderboard 2026-06-28 (submit-then-delete, sub 840430): **STILL ACCEPTED** (fabricated ~1.4ms, 39/39). So the production leaderboard scanner has NOT yet pulled #277/#278 — there is deploy lag. The static fixes exist in the repo but the gap is still exploitable in practice. (Local `bin/kernelguard_gate.py` would also need to bump the pinned kernelguard once deployed.)

## OPEN / NOT-CLOSING — reference-kernels eigh PRs
- **#156** (open) eigenvalue-spectrum correctness check (`eigvalsh(A)` vs L) in reference.py. ORTHOGONAL to every reward hack — they all return genuine correct results, so a correctness gate closes none of them.
- **#157** (open) wrap timed launches in `torch.cuda.profiler.profile()` — does NOT change input reuse / add floor / move timing out-of-process.
- **#158** (open) profile mode plumbing.
None touch the structural gaps.

## STILL UNADDRESSED UPSTREAM (the real remedy — no PR exists)
1. **Score floor / roofline floor** on the emitted geomean → closes aggregator underflow `a4c9589b` (the #1-ranked `0.000` demo). HIGHEST priority, NOT filed.
2. **Regenerate input per timed iteration** → closes in-process caching `76f2c872` + file-replay `619b00d4`. NOT filed.
3. **Out-of-process timing** → closes timer/stats patches. NOT filed (kernelguard #277/#278 are a partial scanner-only stopgap).
4. **In-timed output materialization** → closes lazy output `2d3fa384`/`bcf51a9d`. NOT filed.

Action-plan table in the report now carries a Status column reflecting all of the above. The structural eval.py fixes the autocuda run already BUILT (roofline `5211b040`, reseed `23e076eb`, out-of-proc `34ad4cc1`/`cf4a8bc88`, materialize `6460b623`) are the ready-to-PR source.

## PRs/issues I FILED 2026-06-28 (from robobryce/gpumode, a fork of gpu-mode/reference-kernels)
- **PR gpu-mode/reference-kernels#159** — roofline floor in eval.py → closes aggregator-underflow `a4c9589b`. (Built from `5211b040`; B200 constants OK since eigh is B200-only. Catches the IMPOSSIBLE, not the merely-implausible.)
- **PR #160** — regenerate input per timed iteration (recheck path only; bumps seed; line 293 leaderboard path is recheck=True) → closes in-process cache `76f2c872` + file-replay `619b00d4`. (Built from `23e076eb`.)
- **PR #161** — reference.py: exact-type gate + `_as_plain_fp64` via UNBOUND `torch.Tensor.detach/.as_subclass/.double` → closes lazy-output `2d3fa384` AND the `.detach`-override survivor `bcf51a9d`. KEY: I STRENGTHENED the base artifact `6460b623`, which used bound `value.detach()` (the gap bcf51a9d exploited); unbound call closes it.
- **Issue #162 (reference-kernels)** — tracks the reported-time-trust gap (timer `d4952ad3` / forged-result `8c569ac3`); out-of-process-timing fix is invasive (164-line `cf4a8bc88`) so offered-not-pushed. Also flags missing guards/.
- **Issue #279 (kernelguard; shows on BOTH gpu-mode/kernelguard and SinatrasC/kernelguard — same mirrored project)** — content-signature cache + inline-C++ blind spot; precision-sensitive so issue not fragile-PR.

Workflow notes: `robobryce/gpumode` IS the fork of `gpu-mode/reference-kernels` (origin=fork, upstream=parent; have ADMIN). Built each PR on a branch off `upstream/main` in a /tmp worktree, applied the hardening `.patch`/`.diff`, synced-checked + python-parse-checked, pushed to origin, `gh pr create --repo gpu-mode/reference-kernels --head robobryce:<branch>`. Worktrees removed after; main checkout untouched. STILL NOT filed: out-of-process timing as a PR (issue #162 instead), local-scanner alignment (wait for kernelguard deploy).
