#!/usr/bin/env bash
# Authenticate popcorn-cli, the safe way.
#
# `popcorn-cli register <provider>` prints a one-time OAuth URL whose `state` is
# base64-encoded JSON. Retyping or line-wrapping that URL corrupts the state and
# the login fails with "Invalid state parameter" — the single most common way
# this step goes wrong (especially for agents relaying the URL by hand). And
# `register` returns the instant it has saved a local cli_id, *before* you have
# actually completed the browser flow, so there's no signal for "am I done yet?"
#
# This wrapper fixes both:
#   1. it extracts the URL verbatim (no human ever retypes it), prints it alone
#      on one line, and writes it to a file you can open/copy exactly;
#   2. it then polls until the cli_id is actually linked to your account, so you
#      get a definite "authenticated" instead of a guess.
#
# Usage:
#   login.sh [discord|github] [options]
#
#   --check           report whether you're authenticated, then exit (0 = yes)
#   --force           re-authenticate even if already logged in
#   --no-wait         print the URL and exit without polling for completion
#   --timeout SECS    how long to wait for you to finish in the browser (default 300)
#   --leaderboard L   leaderboard used to verify auth (default histogram_v2; any
#                     name works — auth is checked before the name is)
#
# Idempotent: with no flags it no-ops if you're already authenticated.
set -euo pipefail

PROVIDER=""
FORCE=0
WAIT=1
CHECK_ONLY=0
TIMEOUT=300
VERIFY_LB="${GPUMODE_VERIFY_LB:-histogram_v2}"
URL_FILE=""   # a temp file, created once we have a URL to save (see below)

log()  { printf '\033[1;36m[login]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[login]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[login]\033[0m %s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
    case "$1" in
        discord|github) PROVIDER="$1" ;;
        --force)        FORCE=1 ;;
        --no-wait)      WAIT=0 ;;
        --check)        CHECK_ONLY=1 ;;
        --timeout)      TIMEOUT="${2:?--timeout needs a value}"; shift ;;
        --leaderboard)  VERIFY_LB="${2:?--leaderboard needs a value}"; shift ;;
        -h|--help)      awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit 0 ;;
        *)              err "unknown argument: $1"; exit 2 ;;
    esac
    shift
done
PROVIDER="${PROVIDER:-discord}"

# --- locate popcorn-cli (PATH, else the spot install.sh drops it) ------------
if command -v popcorn-cli >/dev/null 2>&1; then
    POPCORN="$(command -v popcorn-cli)"
elif [ -x "$HOME/.local/bin/popcorn-cli" ]; then
    POPCORN="$HOME/.local/bin/popcorn-cli"
else
    err "popcorn-cli not found. Run bin/install.sh first."
    exit 1
fi

# --- auth probe --------------------------------------------------------------
# A cli_id that isn't yet linked to an account yields HTTP 401 ("Invalid or
# unauthorized auth header"); once linked, even an unknown leaderboard returns
# "No submissions found" (auth is validated before the leaderboard name). So the
# presence of a 401 / auth-header error is the unambiguous "not authed" signal.
is_authed() {
    local out
    out="$("$POPCORN" --no-tui submissions list --leaderboard "$VERIFY_LB" --limit 1 2>&1 || true)"
    if grep -qiE '\b401\b|unauthorized|auth header' <<<"$out"; then
        return 1
    fi
    return 0
}

if [ "$CHECK_ONLY" -eq 1 ]; then
    if is_authed; then log "authenticated ✓"; exit 0; else log "not authenticated"; exit 1; fi
fi

if [ "$FORCE" -eq 0 ] && is_authed; then
    log "already authenticated ✓  (re-run with --force to authenticate again)"
    exit 0
fi

# --- run register and capture its output -------------------------------------
log "starting $PROVIDER authentication ..."
REG_OUT="$("$POPCORN" register "$PROVIDER" 2>&1 || true)"

# Extract the OAuth URL verbatim: strip ANSI, then take the authorize URL.
# Matches both discord (oauth2/authorize) and github (login/oauth/authorize).
# `|| true`: a no-match grep must fall through to the friendly error below,
# not trip `set -e` (which would abort before we can print anything useful).
URL="$(printf '%s\n' "$REG_OUT" \
    | sed -E 's/\x1b\[[0-9;]*[mK]//g' \
    | grep -oE 'https://[^[:space:]]*(oauth2/authorize|login/oauth/authorize)[^[:space:]]*' \
    | head -n1 || true)"

if [ -z "$URL" ]; then
    err "could not find an OAuth URL in 'popcorn-cli register $PROVIDER' output:"
    printf '%s\n' "$REG_OUT" | sed 's/^/    /' >&2
    exit 1
fi

# Save the URL to a temp file (honoring $TMPDIR) so it can be copied exactly —
# it's one-time, ephemeral state, so it doesn't belong in the config dir. We
# remove it on a successful login below.
URL_FILE="$(mktemp "${TMPDIR:-/tmp}/popcorn-auth-url.XXXXXX")"
printf '%s\n' "$URL" > "$URL_FILE"

cat <<EOF

  $(printf '\033[1m▸ Open this EXACT URL in a browser to finish logging in:\033[0m')

$URL

  $(printf '\033[2m(also saved to %s — copy from there to avoid mangling the state)\033[0m' "$URL_FILE")

EOF

if [ "$WAIT" -eq 0 ]; then
    log "not waiting (--no-wait). Re-run '$0 --check' once you've authorized."
    exit 0
fi

# --- poll until the browser flow completes -----------------------------------
log "waiting up to ${TIMEOUT}s for you to authorize (Ctrl-C is safe — the URL is saved) ..."
INTERVAL=3
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
    if is_authed; then
        rm -f "$URL_FILE"   # consumed — don't leave the one-time URL lying around
        log "authenticated ✓  (cli_id saved to $HOME/.popcorn.yaml)"
        log "verify any time with:  $POPCORN --no-tui submissions list --leaderboard $VERIFY_LB"
        exit 0
    fi
    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
done

err "timed out after ${TIMEOUT}s without a completed login."
err "The URL is still in $URL_FILE — open it, then run: $0 --check"
exit 1
