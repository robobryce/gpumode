---
name: leaderboard-rankings
description: >-
  Read GPU MODE leaderboard standings (per-GPU rankings) from the command line —
  popcorn-cli can't. Shows, for one GPU, each active problem's #1 holder and (with
  --user) your own rank and gap to #1; or the full standings for a single problem.
  Use when the user asks "what's my rank / where do I stand / who's #1 / show the
  leaderboard" for a GPU MODE problem. Keywords - rank, ranking, standings,
  leaderboard, score, gap to first, who is winning.
allowed-tools: Bash
---

# leaderboard-rankings

`popcorn-cli` has no way to read a rank: `submissions list` / `submissions show`
print every score as `-`, and the bot API only attaches a score to your own
`leaderboard`-mode runs — never a *position*. The public per-GPU standings live
on the kernelboard website's JSON API instead (same-origin, unauthenticated):

- `GET {site}/api/leaderboard-summaries` → `[{id, name, deadline, gpu_types, …}]`
- `GET {site}/api/leaderboard/{id}` → `{rankings: {GPU: [{rank, score, user_name,
  file_name, submission_time, …}]}}`, where `score` is the geomean runtime in
  **seconds** (lower is better; this script prints µs).

This skill wraps those two endpoints into the queries you actually want.

## When to use this skill

Use it when the user asks any of:

- "What's my rank on the active H100 problems?" / "Where do I stand?"
- "Who's #1 on histogram_v2?" / "Show the leaderboard for <problem>."
- "How far am I behind first place?"

Do **not** use it to *submit* (that's `popcorn-cli submit … --mode leaderboard`)
or to authenticate (use the `popcorn-login` skill). Reading standings needs no
auth at all.

## How to invoke

Run the bundled script (stdlib Python 3 only — no venv, no torch, no creds):

```
scripts/rankings.py [--gpu GPU] [--user NAME] [--problem NAME]
                    [--all-gpus] [--include-closed] [--top N] [--json] [--site URL]
```

Common forms:

```
scripts/rankings.py --user <name>                 # rank + gap to #1 on every active GPU=H100 problem
scripts/rankings.py --user <name> --gpu B200      # ... a different GPU
scripts/rankings.py --problem histogram_v2 --top 10   # full standings for one problem
scripts/rankings.py --all-gpus --user <name>      # the summary repeated for every GPU
scripts/rankings.py --json                        # machine-readable (pipe to jq)
```

- `--gpu` defaults to `H100` (or `$GPUMODE_GPU`). Accepts `H100/A100/B200/L4`.
- `--user` is the leaderboard `user_name` to highlight and report your rank for
  (defaults to `$GPUMODE_USER`). Without it, only the #1 holder is shown.
- Active problems only by default (deadline in the future); `--include-closed`
  adds the retired v1 boards.
- `--problem` switches from the cross-problem summary to one problem's full table.
- `--site` (or `$GPUMODE_SITE`) overrides the kernelboard host
  (default `https://www.gpumode.com`).

The script exits non-zero on a network/HTTP error, an unknown `--problem` (with a
fuzzy hint), or a GPU no active problem supports.

## Output

Summary mode with `--user` (one row per active problem on the GPU):

```
problem          entries    my rank         my µs    gap→#1  #1 holder                     #1 µs
------------------------------------------------------------------------------------------------
conv2d_v2             16     #1/16🥇    19,828.077         —  badelsteinlelbach        19,828.077
grayscale_v2          20      #5/20     1,368.085    +17.3%  ağaç.mp4                  1,166.016
histogram_v2          16     #1/16🥇        11.659         —  badelsteinlelbach            11.659
vectorsum_v2          30     #2/30🥈        76.973    +11.8%  QiSun                        68.835
```

Scores are the geomean the leaderboard ranks by, in µs (lower is better);
`gap→#1` is how much slower you are than first place.

## Example

User: "Where do I stand on the H100 boards? I'm badelsteinlelbach."

```
scripts/rankings.py --user badelsteinlelbach
```

User: "Show me the top of the histogram_v2 H100 leaderboard."

```
scripts/rankings.py --problem histogram_v2 --top 10 --user badelsteinlelbach
```

Tip: set `GPUMODE_USER` (and optionally `GPUMODE_GPU`) once so the flags can be
omitted.

## Tests

`tests/rankings.sh` exercises every mode offline against a local stdlib HTTP mock
of the API (no network, no creds):

```
bash tests/rankings.sh
```
