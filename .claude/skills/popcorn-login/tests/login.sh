#!/usr/bin/env bash
# Test the popcorn-login skill's scripts/login.sh against a mock popcorn-cli. No
# network, no real OAuth: a fake popcorn-cli on PATH plays the role of the CLI,
# and a state file lets the mock flip from "not authed" to "authed" so we can
# exercise the poll loop.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

BIN="$TMP_DIR/bin"
mkdir -p "$BIN"
STATE="$TMP_DIR/authed"          # exists => mock reports authenticated
URL="https://discord.com/oauth2/authorize?client_id=123&response_type=code&redirect_uri=https%3A%2F%2Fhost%2Fauth%2Fcli%2Fdiscord&scope=identify&state=eyJhIjoiYiJ9.SOME-base64_state~chars"

# --- mock popcorn-cli --------------------------------------------------------
cat > "$BIN/popcorn-cli" <<PY
#!/usr/bin/env bash
case "\$1" in
  register)
    # ANSI-wrapped URL, just like the real CLI prints it.
    printf '  \033[4m%s\033[0m\n' "$URL"
    echo "CLI ID saved"
    exit 0 ;;
  --no-tui)
    # 'submissions list ...' — 401 until the state file appears.
    if [ -f "$STATE" ]; then echo "No submissions found."; else
      echo "Application error: Server returned status 401 Unauthorized: Invalid or unauthorized auth header"
    fi
    exit 0 ;;
esac
PY
chmod +x "$BIN/popcorn-cli"

# isolate HOME so the real ~/.popcorn.yaml and config are untouched
export HOME="$TMP_DIR/home"
mkdir -p "$HOME"
# Sandbox the URL temp file into our own dir so we can find and clean it.
export TMPDIR="$TMP_DIR/tmp"
mkdir -p "$TMPDIR"
run() { PATH="$BIN:$PATH" BROWSER=/bin/true bash "$SKILL_DIR/scripts/login.sh" "$@"; }

fail() { echo "FAIL: $1" >&2; exit 1; }
# Path the script saved the URL to, parsed from its "also saved to <path>" line
# (strip ANSI first; the path is a single non-space token).
saved_url_file() {
  printf '%s' "$1" | sed -E 's/\x1b\[[0-9;]*[mK]//g' \
    | sed -nE 's/.*also saved to ([^ ]+).*/\1/p' | head -n1
}

# 1. --check when not authed -> exit 1
if run --check >/dev/null 2>&1; then fail "--check should be nonzero when unauthed"; fi

# 2. --check when authed -> exit 0
: > "$STATE"
run --check >/dev/null 2>&1 || fail "--check should be 0 when authed"

# 3. already-authed no-op
out="$(run 2>&1)" || fail "no-op run should exit 0 when authed"
grep -qi "already authenticated" <<<"$out" || fail "expected 'already authenticated':\n$out"

# 4. --force --no-wait saves the URL verbatim to a temp file (state intact)
rm -f "$STATE"
out="$(run discord --force --no-wait 2>&1)" || fail "--no-wait should exit 0"
URL_FILE="$(saved_url_file "$out")"
[ -n "$URL_FILE" ] || fail "no 'also saved to' path in output:\n$out"
case "$URL_FILE" in "$TMPDIR"/*) ;; *) fail "URL file not under TMPDIR: $URL_FILE" ;; esac
[ -f "$URL_FILE" ] || fail "URL file not written: $URL_FILE"
saved="$(cat "$URL_FILE")"
[ "$saved" = "$URL" ] || fail "saved URL not verbatim:\n  want: $URL\n  got:  $saved"
[ "$(wc -l <"$URL_FILE")" -eq 1 ] || fail "URL file should be a single line"
grep -qF "$URL" <<<"$out" || fail "URL not printed to stdout"
rm -f "$URL_FILE"

# 5. poll loop: starts unauthed, becomes authed mid-wait -> exit 0
rm -f "$STATE"
( sleep 3; : > "$STATE" ) &
helper=$!
if run discord --force --timeout 20 >"$TMP_DIR/poll.out" 2>&1; then
  grep -qi "authenticated ✓" "$TMP_DIR/poll.out" || fail "poll success missing confirmation"
  uf="$(saved_url_file "$(cat "$TMP_DIR/poll.out")")"
  [ -n "$uf" ] && [ ! -e "$uf" ] || fail "URL temp file should be removed on success: $uf"
else
  wait "$helper" 2>/dev/null || true
  fail "poll loop should have succeeded once state flipped:\n$(cat "$TMP_DIR/poll.out")"
fi
wait "$helper" 2>/dev/null || true

# 6. timeout path -> exit 1, URL preserved
rm -f "$STATE"
if run discord --force --timeout 2 >"$TMP_DIR/to.out" 2>&1; then
  fail "timeout path should exit nonzero"
fi
grep -qi "timed out" "$TMP_DIR/to.out" || fail "timeout message missing"

# 7. github provider URL is also recognized
cat > "$BIN/popcorn-cli" <<'PY'
#!/usr/bin/env bash
case "$1" in
  register) printf '  https://github.com/login/oauth/authorize?client_id=abc&state=xyz\n'; exit 0 ;;
  --no-tui) echo "401 Unauthorized: auth header"; exit 0 ;;
esac
PY
chmod +x "$BIN/popcorn-cli"
out="$(run github --force --no-wait 2>&1)" || fail "github --no-wait should exit 0"
grep -qF "github.com/login/oauth/authorize" <<<"$out" || fail "github URL not extracted:\n$out"

# 8. register that prints no URL -> friendly error, exit 1
cat > "$BIN/popcorn-cli" <<'PY'
#!/usr/bin/env bash
case "$1" in
  register) echo "boom, no url"; exit 0 ;;
  --no-tui) echo "401 Unauthorized: auth header"; exit 0 ;;
esac
PY
chmod +x "$BIN/popcorn-cli"
if run discord --force --timeout 2 >"$TMP_DIR/nourl.out" 2>&1; then
  fail "no-URL path should exit nonzero"
fi
grep -qi "could not find an OAuth URL" "$TMP_DIR/nourl.out" || fail "no-URL error missing"

echo "PASS: popcorn-login"
