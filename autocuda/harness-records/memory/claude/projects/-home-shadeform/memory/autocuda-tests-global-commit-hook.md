---
name: autocuda-tests-global-commit-hook
description: Why 8 autocuda init-brief/init-manager pytest fixtures fail locally but pass in CI
metadata: 
  node_type: memory
  type: reference
  originSessionId: eeaa3861-17b6-4b2c-a59c-05b95e39e920
---

Running autocuda's `plugins/autocuda/tests/test_autocuda_optimize_tree.py` locally on this machine fails ~8 tests (the `init_manager` / `init_brief` fixtures), all with the same error: the autonomous-agent-bootstrap pre-commit hook at `~/.aab/git-hooks` (wired via global `core.hooksPath`) rejects the fixtures' `git commit` because they author as `t <t@t>`, not the configured global identity.

This is environmental, not a code defect. CI (GitHub Actions) checks out fresh with no global hook, so the fixtures commit fine and pytest is green.

To run the suite cleanly locally without touching any global config: `GIT_CONFIG_GLOBAL=$(mktemp) PYTHONPATH=plugins/autocuda/lib python3 -m pytest plugins/autocuda/tests/` — gives the fixture subprocesses an empty global config (no `core.hooksPath`), so all tests pass (443 passed, 5 skipped as of 2026-06-24). Do NOT bypass the hook for real commits — only for these throwaway-repo test fixtures.
