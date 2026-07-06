#!/usr/bin/env python3
"""Manager ranking helper. autocuda status' global_best/top_best returns None
under our multi-benchmark schema with per-tag N/A scoping (the ranker requires
ALL schema benchmarks non-N/A in the baseline row). Rank manually instead:
for one tag, find its active benchmark column, read the baseline, and rank every
succeeded row by baseline/metric (min direction) speedup.

Usage: rank.py <tag> [topK]
"""
import csv, sys, glob, os

DD = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1]
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 8

BENCHES = ["pmpp_v2/sort_py", "pmpp_v2/histogram_py", "pmpp_v2/conv2d_py"]

rows = []
for f in sorted(glob.glob(os.path.join(DD, f"{tag}-optimize-tree-worker-*-log.csv"))):
    wid = f.split("worker-")[-1].split("-log")[0]
    with open(f) as fh:
        for r in csv.DictReader(fh):
            r["_wid"] = wid
            rows.append(r)

# active benchmark = the one with a numeric baseline cell
baseline = next((r for r in rows if r.get("status") == "baseline"), None)
if not baseline:
    print("no baseline row"); sys.exit(1)
active = None
for b in BENCHES:
    v = (baseline.get(b) or "").strip()
    if v and v.upper() != "N/A":
        active = b; base_val = float(v); break
print(f"tag={tag}  active={active}  baseline={base_val}")

succ = []
for r in rows:
    if r.get("status") != "succeeded":
        continue
    v = (r.get(active) or "").strip()
    if not v or v.upper() == "N/A":
        continue
    try: val = float(v)
    except ValueError: continue
    sp = base_val / val  # min direction
    succ.append((sp, val, r.get("_wid"), r.get("brief_id"), r.get("iteration"),
                 (r.get("commit") or "")[:12], (r.get("description") or "")[:70]))

succ.sort(key=lambda x: -x[0])
print(f"{'speedup':>8} {'metric':>11}  w/brief/iter  commit        desc")
for sp, val, wid, bid, it, c, d in succ[:topk]:
    print(f"{sp:8.4f} {val:11.3f}  w{wid}/b{bid}/i{it:<3} {c}  {d}")
if not succ:
    print("(no succeeded rows with the active metric yet)")
