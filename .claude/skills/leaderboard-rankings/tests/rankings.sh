#!/usr/bin/env bash
# Test the leaderboard-rankings skill's scripts/rankings.py against a local mock
# of the kernelboard JSON API. No network, no creds: a tiny stdlib HTTP server
# serves canned fixtures and we point rankings.py at it with --site.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
SERVER_PID=""

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

PORT=$(( (RANDOM % 20000) + 20000 ))

# --- mock server: serves /api/leaderboard-summaries and /api/leaderboard/{id} -
cat > "$TMP_DIR/server.py" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SUMMARIES = {"code": 0, "data": {"leaderboards": [
    {"id": 1, "name": "demo_v2",   "deadline": "2100-12-31T00:00:00+00:00",
     "gpu_types": ["H100", "A100"], "visibility": "public"},
    {"id": 2, "name": "closed_v2", "deadline": "2000-01-01T00:00:00+00:00",
     "gpu_types": ["H100"],         "visibility": "public"},
    {"id": 3, "name": "other_v2",  "deadline": "2100-12-31T00:00:00+00:00",
     "gpu_types": ["A100"],         "visibility": "public"},
]}}

RANK = {
    1: {"H100": [
            {"rank": 1, "score": 1.0e-05, "user_name": "alice", "file_name": "a.py",
             "submission_time": "2026-06-01T00:00:00Z"},
            {"rank": 2, "score": 1.2e-05, "user_name": "me",    "file_name": "m.py",
             "submission_time": "2026-06-02T00:00:00Z"},
        ],
        "A100": [
            {"rank": 1, "score": 2.0e-05, "user_name": "me",    "file_name": "m.py",
             "submission_time": "2026-06-03T00:00:00Z"},
        ]},
    2: {"H100": [
            {"rank": 1, "score": 9.0e-05, "user_name": "bob",   "file_name": "b.py",
             "submission_time": "2025-01-01T00:00:00Z"},
        ]},
    3: {"A100": []},
}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/api/leaderboard-summaries":
            self._send(SUMMARIES)
        elif self.path.startswith("/api/leaderboard/"):
            lb = int(self.path.rsplit("/", 1)[1])
            self._send({"code": 0, "data": {"rankings": RANK.get(lb, {})}})
        else:
            self.send_response(404); self.end_headers()

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

python3 "$TMP_DIR/server.py" "$PORT" &
SERVER_PID=$!

# wait for the server to accept connections
for _ in $(seq 1 50); do
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

SITE="http://127.0.0.1:$PORT"
RANK_PY=("python3" "$SKILL_DIR/scripts/rankings.py" "--site" "$SITE")

fail() { echo "FAIL: $1" >&2; exit 1; }
assert_has()  { grep -qF -- "$2" <<<"$1" || fail "expected '$2' in output:\n$1"; }
assert_lacks() { grep -qF -- "$2" <<<"$1" && fail "did not expect '$2' in output:\n$1" || true; }

# 1. active-only summary on H100 hides the closed problem and the A100-only one
out="$("${RANK_PY[@]}" --gpu H100)"
assert_has  "$out" "demo_v2"
assert_lacks "$out" "closed_v2"
assert_lacks "$out" "other_v2"

# 2. --include-closed surfaces the closed problem
out="$("${RANK_PY[@]}" --gpu H100 --include-closed)"
assert_has "$out" "closed_v2"

# 3. --user reports rank, gap, and total entries
out="$("${RANK_PY[@]}" --gpu H100 --user me)"
assert_has "$out" "#2/2"      # rank 2 of 2 on demo_v2
assert_has "$out" "+20.0%"    # 1.2e-5 / 1.0e-5 - 1 = 20%
assert_has "$out" "alice"     # leader shown

# 4. single-problem standings, highlight the user
out="$("${RANK_PY[@]}" --gpu H100 --problem demo_v2 --user me)"
assert_has "$out" "<- you"
assert_has "$out" "10.000"    # alice 1.0e-5 -> 10 us
assert_has "$out" "12.000"    # me 1.2e-5 -> 12 us

# 5. JSON mode is valid JSON with the computed gap
out="$("${RANK_PY[@]}" --gpu H100 --user me --json)"
python3 -c "import json,sys; d=json.load(sys.stdin); \
  r=[p for p in d['problems'] if p['problem']=='demo_v2'][0]; \
  assert r['my_rank']==2 and abs(r['gap_pct']-20.0)<1e-6, r" <<<"$out" \
  || fail "JSON gap/rank wrong:\n$out"

# 6. empty ranking for a GPU renders without crashing
out="$("${RANK_PY[@]}" --gpu A100 --problem other_v2)"
assert_has "$out" "No ranked submissions"

# 7. unknown problem exits non-zero with a hint
if "${RANK_PY[@]}" --problem demo >/dev/null 2>&1; then
  fail "expected nonzero exit for unknown problem 'demo'"
fi

echo "PASS: leaderboard-rankings"
