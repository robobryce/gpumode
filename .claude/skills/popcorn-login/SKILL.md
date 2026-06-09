---
name: popcorn-login
description: >-
  Authenticate popcorn-cli the safe way. `popcorn-cli register` prints a one-time
  OAuth URL whose base64 `state` breaks if retyped/wrapped (the "Invalid state
  parameter" failure), and it returns before the browser flow finishes so there's
  no "am I done?" signal. This skill extracts the URL verbatim, saves it to a temp
  file, and polls until login actually completes. Use when the user needs to log
  in / authenticate / register popcorn-cli, or when CLI commands fail with 401 /
  unauthorized. Keywords - login, authenticate, register, oauth, 401, unauthorized,
  popcorn-cli auth.
allowed-tools: Bash
---

# popcorn-login

`popcorn-cli register <provider>` trips people — and especially agents — up two
ways:

1. **The URL gets mangled.** The one-time OAuth URL carries a base64-JSON `state`
   parameter. Retype it, or let it line-wrap and copy it back one character off,
   and login fails with "Invalid state parameter".
2. **There's no "done" signal.** `register` returns the instant it saves a local
   `cli_id` — *before* the browser flow has linked that id to your account. So
   the id looks saved while still being unauthorized, and nothing tells you when
   you've actually finished.

This skill wraps `register` to remove both: it extracts the printed URL
**verbatim** (so nobody retypes it), saves it to a temp file it prints, and then
**polls** until the `cli_id` is actually linked — printing a definite
`authenticated ✓` or timing out.

## When to use this skill

Use it when the user wants to:

- "Log in / authenticate / register popcorn-cli" (first-time or re-auth).
- Fix CLI commands failing with `401` / "Invalid or unauthorized auth header".
- Check whether they're currently authenticated.

Do **not** use it to read standings (that's the `leaderboard-rankings` skill,
which needs no auth) or to submit a kernel (`popcorn-cli submit …`).

## How to invoke

Run the bundled script:

```
scripts/login.sh [discord|github] [options]
```

- provider — `discord` (default) or `github`.
- `--check` — report whether you're authenticated, then exit (0 = yes). Use this
  for a quick auth probe without starting a login.
- `--no-wait` — print + save the URL and exit without polling (hand the URL to
  the user, check back later with `--check`).
- `--force` — re-authenticate even if already logged in.
- `--timeout SECS` — how long to wait for the browser flow (default 300).
- `--leaderboard L` — leaderboard used to verify auth (default `histogram_v2`;
  any name works — auth is validated before the name is).

Idempotent: with no flags it no-ops if you're already authenticated.

### Driving it as an agent

The interactive step is the human opening the URL in a browser. A good pattern:

1. Run `scripts/login.sh --check`. If it exits 0, you're done — say so.
2. Otherwise run `scripts/login.sh <provider>`. It prints the OAuth URL (also
   saved to a temp file it names) and then blocks, polling.
3. **Relay the URL to the user verbatim — copy it from the printed temp file, do
   not retype it** — and ask them to open it and authorize.
4. The script returns `authenticated ✓` on its own once they finish, or times
   out (exit 1) — bump `--timeout` and re-run, or use `--no-wait` + `--check`
   if you'd rather not hold a long-running process.

## Example

User: "Log me into popcorn-cli."

```
scripts/login.sh --check || scripts/login.sh discord
```

Then give the user the printed URL (from the temp file path it names) to open,
and wait for the `authenticated ✓` line.

## Requirements

- `popcorn-cli` — found on `PATH` or at `~/.local/bin/popcorn-cli` (where the
  repo's `bin/install.sh` drops it).
- A browser to open the printed URL (on a headless host, relay it to the user).

## Tests

`tests/login.sh` covers every path offline against a mock `popcorn-cli` — under
an isolated `HOME` and `TMPDIR`, with no network and no real OAuth:

```
bash tests/login.sh
```
