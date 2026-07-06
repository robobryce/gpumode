---
name: dead-ssh-auth-sock-not-broken-signing
description: dead SSH_AUTH_SOCK errors agents report are a red herring — signing/push work fine; do NOT touch global git config
metadata: 
  node_type: memory
  type: project
  originSessionId: e5529a2b-6069-495a-9353-a0177780a78e
---

On aab-bootstrapped boxes, agents keep reporting "dead SSH socket" / `ssh-add -l`
"communication with agent failed" and misdiagnosing it as broken commit signing.
It is a **red herring**. Nothing on these boxes uses the SSH agent:

- Commit signing uses `gpg.format=ssh` with an **unencrypted on-disk key**
  (`~/.ssh/id_aab_signing`); `ssh-keygen -Y sign` reads it directly, no agent.
- GitHub auth is an **HTTPS token via `gh`** (`gh auth git-credential`), not SSH.

The dead socket comes from SSH **agent-forwarding** left by a disconnected login
(socket owner is `sshd`, not `ssh-agent`), and tmux's `update-environment`
re-injects it into every new pane. Probing it can *hang*, not just error.

**Why it matters (not cosmetic):** an agent that thinks signing is broken may edit
the global git config to "fix" it — which [[goal-stop-hook-json-validation]]-style
unattended fleets can't recover from, and which the aab identity/signing pre-commit
hook (`~/.aab/git-hooks/aab-git-hook`) then *rejects*. CLAUDE.md forbids changing
`user.*`, `commit.gpgsign`, `user.signingkey`, `core.hooksPath`.

**How to apply:** If you see dead-socket errors, confirm signing actually works
(`git commit` in a throwaway repo → `git cat-file commit HEAD | grep gpgsig`) and
move on. Do NOT touch global git config. The verify-side "No signature /
allowedSignersFile" message is also expected (no verify config), not a signing
failure. Root-cause fix upstreamed: brycelelbach/autonomous-agent-bootstrap PR #93
adds a dead-`SSH_AUTH_SOCK` guard to the managed `~/.bashrc` block (unsets it unless
a live agent answers within 1s). Local tmux fix: drop `SSH_AUTH_SOCK` from
`update-environment` + `set-environment -gu SSH_AUTH_SOCK`.
