#!/usr/bin/env bash
# BENCHMARK step — autocuda <-> GPU MODE bridge.
#
# Runs the official GPU MODE eval harness in `benchmark` mode over the problem's
# benchmark shapes, then reduces the per-shape mean runtimes to a single scalar:
# the GEOMETRIC MEAN of the per-shape means, in microseconds (lower is better).
# That is exactly how the GPU MODE leaderboard ranks, and
# geomean(baseline)/geomean(trial) == geomean(baseline_i/trial_i), so reporting
# this one scalar to autocuda with direction=min makes the baseline/trial
# speedup the ratio-space aggregate the worker skill prescribes.
#
# eval.py runs an obligatory correctness check before timing each shape and
# reports an error (not a Stats block) on mismatch; this script treats a missing
# mean for any shape (or any error / absent `check: pass`) as a runtime_error
# (non-zero exit), so a wrong or crashing kernel can never be scored as fast.
#
# stdout: a single fenced result block, emitted LAST:
#     ===GPUMODE_RESULT_BEGIN===
#     <set>/<problem>=<geomean us>
#     ===GPUMODE_RESULT_END===
# The metric NAME is the <set>/<problem> token itself — the same string you pass
# here and scope the optimize-tree run with (`benchmark=<set>/<problem>`) — so
# the emitted key matches the autocuda schema column with no extra lookup. (The
# GPU MODE *leaderboard* name, used only for `popcorn-cli submit`, is a separate
# thing resolved on demand via `gen_specs.py --leaderboard`.)
# stderr: the raw eval.py output + a per-shape microsecond breakdown.
#
# Usage:  bash harness/benchmark.sh <set>/<problem>   # e.g. pmpp_v2/histogram_py
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "$@"

METRIC_NAME="$PROBLEM"
SPECS="$("$PYTHON" "$REPO_DIR/bin/gen_specs.py" "$PROBLEM_DIR/task.yml" --emit benchmarks)"
SPECFILE="$(mktemp)"; trap 'rm -f "$SPECFILE" "$OUT"' EXIT
printf '%s' "$SPECS" > "$SPECFILE"

cd "$PROBLEM_DIR" || exit 1
OUT="$(mktemp)"
POPCORN_FD=3 "$PYTHON" "$SET_DIR/eval.py" benchmark "$SPECFILE" 3>"$OUT"
rc=$?
{ echo "----- eval.py benchmark output -----"; cat "$OUT"; echo "------------------------------------"; } >&2
[ $rc -eq 0 ] || { echo "benchmark: FAILED (eval.py exited $rc)" >&2; exit $rc; }

METRIC_NAME="$METRIC_NAME" "$PYTHON" - "$OUT" <<'PY'
import math, sys, re, os
metric = os.environ.get("METRIC_NAME", "score")
out = open(sys.argv[1]).read().splitlines()
means, specs, errors, count = {}, {}, {}, None
for ln in out:
    m = re.match(r"benchmark-count:\s*(\d+)", ln)
    if m: count = int(m.group(1))
    m = re.match(r"benchmark\.(\d+)\.mean:\s*([0-9.eE+-]+)", ln)
    if m: means[int(m.group(1))] = float(m.group(2))
    m = re.match(r"benchmark\.(\d+)\.spec:\s*(.+)", ln)
    if m: specs[int(m.group(1))] = m.group(2).strip()
    m = re.match(r"benchmark\.(\d+)\.error:\s*(.+)", ln)
    if m: errors[int(m.group(1))] = m.group(2).strip()
check_pass = any(ln.strip() == "check: pass" for ln in out)
if count is None:
    print("ERROR: no benchmark-count in eval output", file=sys.stderr); sys.exit(2)
if errors or not check_pass:
    for i, e in sorted(errors.items()):
        print(f"ERROR shape {i} ({specs.get(i,'?')}): {e}", file=sys.stderr)
    print("ERROR: benchmark did not report 'check: pass' for all shapes", file=sys.stderr); sys.exit(3)
missing = [i for i in range(count) if i not in means]
if missing:
    print(f"ERROR: missing mean for shapes {missing}", file=sys.stderr); sys.exit(3)
us = {i: means[i] / 1000.0 for i in range(count)}  # eval means are ns -> us
for i in range(count):
    print(f"  shape {i:2d} [{specs.get(i,'?')}]: {us[i]:.3f} us", file=sys.stderr)
geo = math.exp(sum(math.log(us[i]) for i in range(count)) / count)
print("===GPUMODE_RESULT_BEGIN===")
print(f"{metric}={geo:.6f}")
print("===GPUMODE_RESULT_END===")
PY
rc=$?
[ $rc -eq 0 ] || { echo "benchmark: FAILED (parse rc=$rc)" >&2; exit $rc; }
