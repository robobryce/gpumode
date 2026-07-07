#!/usr/bin/env python3
"""Time-series comparison of the Opus 4.8 and GPT 5.6 Sol eigh_py autocuda runs.

Three stacked panels (shared x = elapsed active hours), two series each:
  1. Speedup vs time  — running-best (baseline_metric / best_metric), step
  2. Tokens vs time    — cumulative tokens processed (input incl. cache + output)
  3. LOC vs time        — code-only LOC of the current best submission, step

Extraction mirrors autocuda's own dashboard/_plot.py conventions:
  - KEPT_STATUSES = baseline/improved/succeeded
  - elapsed = timestamp - first_timestamp
  - running-best via cummin (eigh metric is lower-is-better)
Opus is the two chained sessions (run1 05-33-43 rooted at main + run2 18-52-36);
its elapsed axis is ACTIVE time (run1 span then run2 appended, idle gap removed),
matching how the comparison tables define Opus wallclock.
"""
import csv, glob, os, json, subprocess
from datetime import datetime, timezone

REPO = "/home/shadeform/gpumode"
CC_DIR = "/home/shadeform/.claude/projects/-home-shadeform-gpumode"
KEPT = {"baseline", "improved", "succeeded"}


def pts(s):
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def loc_code(commit):
    """code-only LOC of submission.py at a commit (blank + pure-comment lines removed)."""
    if not commit or commit == "N/A":
        return None
    s = subprocess.run(
        ["git", "-C", REPO, "show", f"{commit}:problems/linalg/eigh_py/submission.py"],
        capture_output=True, text=True,
    ).stdout
    if not s:
        return None
    return sum(1 for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("#"))


# ---- metric time-series (speedup + LOC) from the brief/worker logs ----
def metric_series(patterns, baseline, baseline_commit, t_start):
    """Return list of (dt, best_metric_so_far, best_commit_so_far).

    Seeded with the baseline anchor at t_start, and the running best is clamped
    at the baseline: the baseline submission is always available, so "best so
    far" never does worse than it (speedup never drops below 1.0×). A first
    trial slower than baseline therefore holds the line at baseline, rather than
    dragging the curve below 1.0.
    """
    rows = []
    files = [f for p in patterns for f in glob.glob(f"{REPO}/autocuda/{p}")
             if "reference" not in f and "manager" not in f]
    for fp in files:
        for r in csv.DictReader(open(fp, newline="")):
            st = (r.get("status") or "").strip()
            if st not in KEPT:
                continue
            t = pts(r.get("timestamp", ""))
            try:
                v = float(r.get("linalg/eigh_py", ""))
            except Exception:
                v = None
            if t and v and st != "baseline":   # baseline is the seed, added below
                rows.append((t, v, (r.get("commit") or "").strip()))
    rows.sort(key=lambda x: x[0])
    # seed: baseline submission at the run's start
    best = baseline; bestc = baseline_commit
    out = [(t_start, best, bestc)]
    for t, v, c in rows:
        if v < best:                            # clamp: never worse than baseline
            best = v; bestc = c
        out.append((t, best, bestc))
    return out


# ---- cumulative tokens over time ----
def opus_token_events(managers):
    """(dt, tokens_this_message) for every assistant message across manager+subagents."""
    ev = []
    for mgr in managers:
        files = [f"{CC_DIR}/{mgr}.jsonl"] + glob.glob(f"{CC_DIR}/{mgr}/subagents/*.jsonl")
        for fp in files:
            if not os.path.exists(fp):
                continue
            seen = set()
            for line in open(fp):
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t = pts(o.get("timestamp", ""))
                m = o.get("message", {})
                if not (t and isinstance(m, dict)):
                    continue
                u = m.get("usage")
                if not u:
                    continue
                mid = m.get("id"); key = (mid, u.get("output_tokens"))
                if mid and key in seen:
                    continue
                if mid:
                    seen.add(key)
                tok = (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                       + u.get("cache_read_input_tokens", 0) + u.get("output_tokens", 0))
                ev.append((t, tok))
    ev.sort(key=lambda x: x[0])
    return ev


def sol_token_events(run_start="2026-07-03T13:37"):
    """(dt, delta_tokens) from Codex rollups: diff the per-session cumulative total_tokens."""
    ev = []
    for fp in sorted(glob.glob("/home/shadeform/.codex/sessions/2026/07/0[345]/rollout-*.jsonl")):
        base = os.path.basename(fp); stamp = base.split("rollout-")[1][:19]
        iso = stamp[:10] + "T" + stamp[11:].replace("-", ":")
        if iso < run_start:
            continue
        prev = 0
        for line in open(fp):
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = pts(o.get("timestamp", ""))
            p = o.get("payload", {})
            if t and isinstance(p, dict) and p.get("type") == "token_count":
                info = p.get("info") or {}
                tot = (info.get("total_token_usage") or {}).get("total_tokens")
                if tot is not None:
                    ev.append((t, max(0, tot - prev)))
                    prev = tot
    ev.sort(key=lambda x: x[0])
    return ev


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OPUS_C = "#4C78A8"   # blue
    SOL_C  = "#F58518"   # orange

    OPUS_BASE, SOL_BASE = 56238.201, 56277.006
    R1_FIRST = pts("2026-06-30T05:38:43.724548+00:00")
    R1_LAST  = pts("2026-06-30T16:33:23.712919+00:00")
    R2_FIRST = pts("2026-06-30T19:06:04.093220+00:00")
    R1_ACTIVE_H = (R1_LAST - R1_FIRST).total_seconds() / 3600.0

    def opus_elapsed_h(t):
        """Active-time hours: run1 measured from its start; run2 appended after run1's span."""
        if t <= R1_LAST:
            return (t - R1_FIRST).total_seconds() / 3600.0
        return R1_ACTIVE_H + (t - R2_FIRST).total_seconds() / 3600.0

    # both runs branch from `main`; that commit is the baseline submission
    MAIN = "08291948124036edc19e0409887fa3fe8765a229"

    # Sol's run start = its first log timestamp (used as the baseline-seed time)
    def first_ts(patterns):
        ts = []
        for p in patterns:
            for fp in glob.glob(f"{REPO}/autocuda/{p}"):
                if "reference" in fp or "manager" in fp:
                    continue
                for r in csv.DictReader(open(fp, newline="")):
                    t = pts(r.get("timestamp", ""))
                    if t:
                        ts.append(t)
        return min(ts)

    sol_first = first_ts(["2026-07-03-13-39-25-optimize-tree-brief-*-log.csv"])

    def sol_elapsed_h(t):
        return (t - sol_first).total_seconds() / 3600.0

    # ---- speedup + LOC series (seeded at baseline = 1.0×, clamped never worse) ----
    opus_m = metric_series(
        ["2026-06-30-05-33-43-eigh-optimize-tree-worker-*-log.csv",
         "2026-06-30-18-52-36-eigh-optimize-tree-brief-*-log.csv"], OPUS_BASE, MAIN, R1_FIRST)
    sol_m = metric_series(
        ["2026-07-03-13-39-25-optimize-tree-brief-*-log.csv"], SOL_BASE, MAIN, sol_first)

    # speedup step series
    opus_sx = [opus_elapsed_h(t) for t, _, _ in opus_m]
    opus_sy = [OPUS_BASE / b for _, b, _ in opus_m]
    sol_sx = [sol_elapsed_h(t) for t, _, _ in sol_m]
    sol_sy = [SOL_BASE / b for _, b, _ in sol_m]

    # LOC step series (LOC of current best commit; cache lookups)
    loc_cache = {}
    def cached_loc(c):
        if c not in loc_cache:
            loc_cache[c] = loc_code(c)
        return loc_cache[c]

    def loc_series(m, elapsed_fn):
        xs, ys = [], []
        for t, _, c in m:
            l = cached_loc(c)
            if l is not None:
                xs.append(elapsed_fn(t)); ys.append(l)
        return xs, ys

    opus_lx, opus_ly = loc_series(opus_m, opus_elapsed_h)
    sol_lx, sol_ly = loc_series(sol_m, sol_elapsed_h)

    # ---- token cumulative series ----
    # Clip to each run's last *trial* time so all three panels span the same
    # active window (manager sessions keep emitting tokens after the final
    # committed trial; those trailing tokens would stretch the x-axis alone).
    opus_last = max(t for t, _, _ in opus_m)
    sol_last = max(t for t, _, _ in sol_m)
    opus_tok = opus_token_events(["0435e030-2232-415d-8dba-1bc8695e4b21",
                                   "94ecbd36-5548-428d-ad8f-ac69e1c4c1f6"])
    sol_tok = sol_token_events()

    def cum(ev, elapsed_fn, last_t):
        xs, ys = [], []; run = 0
        for t, d in ev:
            if t > last_t:
                break
            run += d
            xs.append(elapsed_fn(t)); ys.append(run / 1e9)  # billions
        return xs, ys

    opus_tx, opus_ty = cum(opus_tok, opus_elapsed_h, opus_last)
    sol_tx, sol_ty = cum(sol_tok, sol_elapsed_h, sol_last)

    # ---- plot ----
    import numpy as np
    from matplotlib.lines import Line2D
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    def linfit(xs, ys):
        """Least-squares line + fit-quality stats (R², RMSE, and prediction-band pieces)."""
        n = len(xs)
        mx = sum(xs) / n; my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        m = sxy / sxx; b = my - m * mx
        resid = [y - (m * x + b) for x, y in zip(xs, ys)]
        ss_res = sum(r * r for r in resid)
        ss_tot = sum((y - my) ** 2 for y in ys)
        s_err = (ss_res / max(n - 2, 1)) ** 0.5          # regression standard error (for band)
        rmse = (ss_res / n) ** 0.5                        # fit error, panel units
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        return dict(m=m, b=b, mx=mx, sxx=sxx, s_err=s_err, n=n, rmse=rmse, r2=r2)

    opus_end = opus_sx[-1]      # Opus run length in active hours (~64h)

    def add_trends(ax, ox, oy, sx, sy, fmt, box_xy):
        """Draw linear trends for both series (dashed), project Sol to opus_end
        (dotted) with a 95% prediction band, and annotate with the projection
        value + linear-fit error (R², RMSE) over each series' observed data."""
        of = linfit(ox, oy); sf = linfit(sx, sy)
        sol_end = sx[-1]
        ax.plot([ox[0], opus_end], [of["m"] * ox[0] + of["b"], of["m"] * opus_end + of["b"]],
                color=OPUS_C, ls="--", lw=1.4, alpha=0.9)
        ax.plot([sx[0], sol_end], [sf["m"] * sx[0] + sf["b"], sf["m"] * sol_end + sf["b"]],
                color=SOL_C, ls="--", lw=1.4, alpha=0.9)
        ax.plot([sol_end, opus_end], [sf["m"] * sol_end + sf["b"], sf["m"] * opus_end + sf["b"]],
                color=SOL_C, ls=":", lw=1.6, alpha=0.9)
        tt = np.linspace(sx[0], opus_end, 200)
        fit = sf["m"] * tt + sf["b"]
        half = 1.96 * sf["s_err"] * np.sqrt(1.0 + 1.0 / sf["n"] + (tt - sf["mx"]) ** 2 / sf["sxx"])
        ax.fill_between(tt, fit - half, fit + half, color=SOL_C, alpha=0.12, lw=0)
        proj = sf["m"] * opus_end + sf["b"]
        ph = 1.96 * sf["s_err"] * np.sqrt(1.0 + 1.0 / sf["n"] + (opus_end - sf["mx"]) ** 2 / sf["sxx"])
        ax.plot([opus_end], [proj], marker="o", ms=5, color=SOL_C)
        txt = (f"Sol proj @ {opus_end:.0f}h ≈ {fmt(proj)}  ({fmt(proj - ph)}–{fmt(proj + ph)}, 95%)\n"
               f"linear fit vs observed data:\n"
               f"  Opus  R²={of['r2']:.3f}, RMSE={fmt(of['rmse'])}\n"
               f"  Sol   R²={sf['r2']:.3f}, RMSE={fmt(sf['rmse'])}")
        ax.annotate(txt, xy=(opus_end, proj), xycoords="data",
                    xytext=box_xy, textcoords="axes fraction",
                    ha="left", va="top", fontsize=7.8, color="black",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=SOL_C, alpha=0.92),
                    arrowprops=dict(arrowstyle="->", color=SOL_C, lw=1.0))
        return proj

    # panel 1 — speedup
    ax1.step(opus_sx, opus_sy, where="post", color=OPUS_C, lw=2.2)
    ax1.step(sol_sx, sol_sy, where="post", color=SOL_C, lw=2.2)
    ax1.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
    add_trends(ax1, opus_sx, opus_sy, sol_sx, sol_sy, lambda v: f"{v:.2f}×", (0.43, 0.72))
    ax1.set_ylabel("Speedup vs main\n(baseline / best)")
    ax1.set_title("Speedup over time   (dashed = linear trend · dotted = Sol projected to Opus run length · shaded = 95% band)",
                  loc="left", fontweight="bold", fontsize=9.6)
    ax1.grid(alpha=0.25)

    # panel 2 — tokens (no trend/projection; raw cumulative series only)
    ax2.step(opus_tx, opus_ty, where="post", color=OPUS_C, lw=2.2)
    ax2.step(sol_tx, sol_ty, where="post", color=SOL_C, lw=2.2)
    ax2.set_ylabel("Cumulative tokens\n(billions)")
    ax2.set_title("Tokens processed over time", loc="left", fontweight="bold")
    ax2.grid(alpha=0.25)

    # panel 3 — LOC (no trend/projection; raw series only)
    ax3.step(opus_lx, opus_ly, where="post", color=OPUS_C, lw=2.2)
    ax3.step(sol_lx, sol_ly, where="post", color=SOL_C, lw=2.2)
    ax3.set_ylabel("Best submission\ncode LOC")
    ax3.set_title("Lines of code (of current best) over time", loc="left", fontweight="bold")
    ax3.set_xlabel("Elapsed active time (hours)")
    ax3.grid(alpha=0.25)

    fig.suptitle("Opus 4.8 vs GPT 5.6 Sol — eigh_py autocuda run\n"
                 "(Opus finished ~64h; Sol still running ~29.7h)",
                 fontweight="bold", y=0.995)
    # single shared legend (the two data series) below all three panels,
    # via explicit proxies so the per-panel trend lines never leak into it.
    proxies = [Line2D([], [], color=OPUS_C, lw=2.2, label="Opus 4.8"),
               Line2D([], [], color=SOL_C, lw=2.2, label="GPT 5.6 Sol")]
    fig.legend(handles=proxies, loc="lower center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    out = f"{REPO}/autocuda/opus-vs-sol-timeseries.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
    # quick sanity dump
    print(f"Opus: speedup pts={len(opus_sx)} last={opus_sy[-1]:.3f}@{opus_sx[-1]:.1f}h; "
          f"tokens last={opus_ty[-1]:.2f}B@{opus_tx[-1]:.1f}h; LOC last={opus_ly[-1]}@{opus_lx[-1]:.1f}h")
    print(f"Sol : speedup pts={len(sol_sx)} last={sol_sy[-1]:.3f}@{sol_sx[-1]:.1f}h; "
          f"tokens last={sol_ty[-1]:.2f}B@{sol_tx[-1]:.1f}h; LOC last={sol_ly[-1]}@{sol_lx[-1]:.1f}h")


if __name__ == "__main__":
    main()
