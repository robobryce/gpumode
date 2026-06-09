#!/usr/bin/env python3
"""Query GPU MODE leaderboard standings (rankings) from the command line.

popcorn-cli has no "rankings" command — `submissions list`/`show` print every
score as `-`, and the bot API only exposes a ranked score on your own
`leaderboard`-mode runs. The *public* per-GPU standings live on the kernelboard
website's JSON API, same-origin and unauthenticated:

    GET {site}/api/leaderboard-summaries   -> [{id,name,deadline,gpu_types,...}]
    GET {site}/api/leaderboard/{id}        -> {rankings: {GPU: [{rank,score,
                                               user_name,file_name,...}, ...]}}

`score` is the geomean runtime in **seconds** (lower is better); we show it in
µs. This script turns those two endpoints into the two queries you actually want:

  * a leaderboard across every active problem for one GPU (default H100), with
    the current leader and — if you pass --user — your own rank and gap to #1;
  * the full standings for a single problem on one GPU (--problem).

Examples:
    rankings.py                              # active problems on H100
    rankings.py --user badelsteinlelbach     # ... with my rank + gap
    rankings.py --gpu B200 --user me          # same, different GPU
    rankings.py --problem histogram_v2 --top 10
    rankings.py --all-gpus --user me          # my rank on every GPU
    rankings.py --json                        # machine-readable

Stdlib only — no venv, no torch, no network creds. Override the site with
--site or GPUMODE_SITE; default user via GPUMODE_USER; default GPU via
GPUMODE_GPU.
"""
import argparse
import datetime as _dt
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_SITE = os.environ.get("GPUMODE_SITE", "https://www.gpumode.com").rstrip("/")
DEFAULT_GPU = os.environ.get("GPUMODE_GPU", "H100")
DEFAULT_USER = os.environ.get("GPUMODE_USER") or None
TIMEOUT = 30


# --- HTTP -------------------------------------------------------------------
def _get_json(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "gpumode-rankings/1.0",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"error: {url} -> HTTP {e.code} {e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"error: cannot reach {url}: {e.reason}")
    try:
        doc = json.loads(body)
    except json.JSONDecodeError:
        sys.exit(f"error: {url} did not return JSON (is --site correct? got "
                 f"{body[:60]!r}...)")
    # The kernelboard API wraps payloads as {code, data, ...}; unwrap when present.
    if isinstance(doc, dict) and "data" in doc and set(doc) <= {"code", "data", "message"}:
        return doc["data"]
    return doc


def fetch_summaries(site):
    """All leaderboards: list of {id, name, deadline, gpu_types, visibility}."""
    data = _get_json(f"{site}/api/leaderboard-summaries")
    return data["leaderboards"] if isinstance(data, dict) else data


def fetch_rankings(site, lb_id, gpu):
    """Sorted ranking list for one leaderboard on one GPU (may be empty)."""
    data = _get_json(f"{site}/api/leaderboard/{lb_id}")
    return (data.get("rankings") or {}).get(gpu, [])


# --- selection --------------------------------------------------------------
def _is_active(lb, now):
    dl = lb.get("deadline")
    if not dl:
        return True
    try:
        return _dt.datetime.fromisoformat(dl.replace("Z", "+00:00")) > now
    except ValueError:
        return True  # unparseable deadline -> don't hide it


def select_problems(summaries, gpu, include_closed):
    now = _dt.datetime.now(_dt.timezone.utc)
    out = []
    for lb in summaries:
        if gpu not in (lb.get("gpu_types") or []):
            continue
        if not include_closed and not _is_active(lb, now):
            continue
        out.append(lb)
    out.sort(key=lambda lb: lb["name"])
    return out


def find_user(rankings, user):
    if not user:
        return None
    lo = user.lower()
    for e in rankings:
        if (e.get("user_name") or "").lower() == lo:
            return e
    return None


def us(score):
    """Seconds -> microseconds string."""
    return "-" if score is None else f"{score * 1e6:,.3f}"


# --- rendering --------------------------------------------------------------
def cmd_summary(args, summaries):
    probs = select_problems(summaries, args.gpu, args.include_closed)
    if not probs:
        scope = "" if args.include_closed else "active "
        sys.exit(f"No {scope}problems support {args.gpu}. "
                 f"Try --gpu <one of the supported types> or --include-closed.")

    rows = []
    for lb in probs:
        ranking = fetch_rankings(args.site, lb["id"], args.gpu)
        leader = ranking[0] if ranking else None
        mine = find_user(ranking, args.user)
        rows.append({
            "problem": lb["name"],
            "entries": len(ranking),
            "leader_user": leader["user_name"] if leader else None,
            "leader_score": leader["score"] if leader else None,
            "my_rank": mine["rank"] if mine else None,
            "my_score": mine["score"] if mine else None,
            "gap_pct": (None if not (mine and leader and leader["score"])
                        else (mine["score"] / leader["score"] - 1.0) * 100.0),
        })

    if args.json:
        print(json.dumps({"gpu": args.gpu, "user": args.user, "problems": rows},
                         indent=2))
        return

    print(f"GPU MODE standings — {args.gpu} "
          f"({'all' if args.include_closed else 'active'} problems)\n")
    if args.user:
        hdr = f"{'problem':16} {'entries':>7}  {'my rank':>9}  {'my µs':>12}  {'gap→#1':>8}  {'#1 holder':<22} {'#1 µs':>12}"
    else:
        hdr = f"{'problem':16} {'entries':>7}  {'#1 holder':<22} {'#1 µs':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        leader = f"{(r['leader_user'] or '-'):<22} {us(r['leader_score']):>12}"
        if args.user:
            if r["my_rank"]:
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r["my_rank"], "")
                rank = f"#{r['my_rank']}/{r['entries']}{medal}"
                gap = "—" if r["my_rank"] == 1 else f"+{r['gap_pct']:.1f}%"
                myus = us(r["my_score"])
            else:
                rank, gap, myus = f"unranked/{r['entries']}", "", "-"
            print(f"{r['problem']:16} {r['entries']:>7}  {rank:>9}  {myus:>12}  {gap:>8}  {leader}")
        else:
            print(f"{r['problem']:16} {r['entries']:>7}  {leader}")

    if args.user and not any(r["my_rank"] for r in rows):
        print(f"\n(no ranked entry for '{args.user}' on {args.gpu} — check the name "
              f"with --problem <name>, or submit with --mode leaderboard)")


def cmd_problem(args, summaries):
    match = [lb for lb in summaries if lb["name"] == args.problem]
    if not match:
        near = [lb["name"] for lb in summaries if args.problem in lb["name"]]
        hint = f" Did you mean: {', '.join(sorted(near))}?" if near else ""
        sys.exit(f"No leaderboard named '{args.problem}'.{hint}")
    lb = match[0]
    if args.gpu not in (lb.get("gpu_types") or []):
        sys.exit(f"'{args.problem}' does not run on {args.gpu}. "
                 f"Supported: {', '.join(lb.get('gpu_types') or [])}.")

    ranking = fetch_rankings(args.site, lb["id"], args.gpu)
    if args.json:
        print(json.dumps({"problem": args.problem, "gpu": args.gpu,
                          "rankings": ranking}, indent=2))
        return
    if not ranking:
        print(f"No ranked submissions for {args.problem} on {args.gpu} yet.")
        return

    shown = ranking if args.top <= 0 else ranking[:args.top]
    print(f"{args.problem} — {args.gpu}  ({len(ranking)} ranked"
          f"{', showing top %d' % len(shown) if len(shown) < len(ranking) else ''})\n")
    hdr = f"{'rank':>4}  {'µs':>12}  {'user':<24} {'file':<28} submitted"
    print(hdr)
    print("-" * len(hdr))
    for e in shown:
        mark = " <- you" if args.user and (e.get("user_name") or "").lower() == args.user.lower() else ""
        when = (e.get("submission_time") or "")[:10]
        fn = (e.get("file_name") or "")[:28]
        print(f"{e['rank']:>4}  {us(e['score']):>12}  {(e.get('user_name') or '-'):<24} {fn:<28} {when}{mark}")


def main():
    ap = argparse.ArgumentParser(
        description="Query GPU MODE leaderboard standings (per-GPU rankings).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  rankings.py --user badelsteinlelbach\n"
               "  rankings.py --problem histogram_v2 --top 10\n"
               "  rankings.py --all-gpus --user me --json")
    ap.add_argument("--gpu", default=DEFAULT_GPU,
                    help=f"GPU type (default {DEFAULT_GPU}; e.g. H100/A100/B200/L4)")
    ap.add_argument("--user", default=DEFAULT_USER,
                    help="leaderboard user_name to highlight / report your rank "
                         "(default $GPUMODE_USER)")
    ap.add_argument("--problem",
                    help="show full standings for one problem instead of the "
                         "cross-problem summary")
    ap.add_argument("--all-gpus", action="store_true",
                    help="repeat the summary for every GPU the problems support")
    ap.add_argument("--include-closed", action="store_true",
                    help="include problems whose deadline has passed (default: "
                         "active only)")
    ap.add_argument("--top", type=int, default=20,
                    help="rows to show in --problem mode (0 = all; default 20)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--site", default=DEFAULT_SITE,
                    help=f"kernelboard base URL (default {DEFAULT_SITE})")
    args = ap.parse_args()
    args.site = args.site.rstrip("/")

    summaries = fetch_summaries(args.site)

    if args.problem:
        cmd_problem(args, summaries)
        return

    if args.all_gpus:
        gpus = sorted({g for lb in summaries for g in (lb.get("gpu_types") or [])})
        for i, gpu in enumerate(gpus):
            if i:
                print()
            args.gpu = gpu
            try:
                cmd_summary(args, summaries)
            except SystemExit as e:  # "no problems for this GPU" — keep going
                print(e)
        return

    cmd_summary(args, summaries)


if __name__ == "__main__":
    main()
