# >>> autonomous-agent-bootstrap >>>
## Always use the configured git identity

This machine is set up by autonomous-agent-bootstrap with a global git author,
email, and (optionally) commit-signing key. Always commit and tag with that
configured identity.

- Commit with a plain `git commit`. Do not override the identity with
  `git -c user.name=... -c user.email=...`, `git commit --author=...`, or the
  `GIT_AUTHOR_NAME` / `GIT_AUTHOR_EMAIL` / `GIT_COMMITTER_NAME` /
  `GIT_COMMITTER_EMAIL` environment variables, and do not run
  `git config user.name` / `git config user.email` inside a repository to set a
  different identity.
- Do not change `user.name`, `user.email`, `user.signingkey`, `commit.gpgsign`,
  or `core.hooksPath` in the global git config.
- Do not disable commit signing when it is configured (`-c commit.gpgsign=false`
  or `--no-gpg-sign`), and do not bypass hooks with `--no-verify`.
- Run `git config --global --get user.name` and `git config --global --get
  user.email` if you need to know who you are committing as.

A global pre-commit hook enforces this and rejects commits whose identity does
not match the global git config.
# <<< autonomous-agent-bootstrap <<<
