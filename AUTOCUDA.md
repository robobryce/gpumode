# Optimizing these kernels with autocuda

This fork of [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) adds a thin layer for optimizing the leaderboard kernels with the [autocuda](https://github.com/brycelelbach/autocuda) plugin's `optimize-tree` / `optimize-simple` / `optimize-hill` skills. The problems themselves are upstream's, unchanged; optimization happens **in place** on each problem's own `submission.py` — nothing is copied or scaffolded.

```
bin/install.sh              one-time machine setup (popcorn-cli, torch venv, gpumode.env)
bin/gen_specs.py            task.yml -> eval.py spec files + leaderboard/gpu/meta lookup
harness/                    build/validate/benchmark/profile/submit bridge (takes <set>/<problem> as an arg)
autocuda/layout.md          committed, machine-agnostic project description (the ground truth)
                            (autocuda/environment.md, the per-machine half, is written by /autocuda:discover)
.claude/skills/             repo-level agent skills (also exposed as .agents/skills):
                            popcorn-login, leaderboard-rankings
```

## Quick start

```bash
./bin/install.sh                                    # once per machine
bash .claude/skills/popcorn-login/scripts/login.sh  # authenticate popcorn-cli (once)
/autocuda:discover                                  # once per machine -> autocuda/environment.md

# optimize a problem IN PLACE (the <set>/<problem> path is the one token):
/autocuda:optimize-tree workers=4 benchmark=pmpp_v2/histogram_py tag-suffix=histogram_py
```

Leaderboard submissions are mandatory for autocuda runs. The baseline must be submitted before workers start, every meaningful safe improvement must be submitted before it is treated as real, and the final candidate must have a successful leaderboard submission. A run without the baseline submission, or without regular submissions for improvements, is invalid for leaderboard comparison because local timings alone do not prove remote acceptance, rule compliance, or actual public performance. Use `bash harness/submit.sh <set>/<problem>` from the commit being evaluated; it resolves the leaderboard/GPU metadata, submits with `popcorn-cli --mode leaderboard`, and prints recent submissions.

`benchmark=<set>/<problem>` is the one knob that selects the target. The same `<set>/<problem>` path is passed as the first argument to every `harness/` script, which `harness/env.sh` uses to find the editable `submission.py` and to put the problem dir + set root on `PYTHONPATH` (so the frozen `eval.py`/`utils.py` resolve where they already live). No environment variable is exported — the path travels as a command argument, not as shell state an agent harness wouldn't preserve between invocations. There is a single autocuda data dir at the repo root (`autocuda/`), so one `.gpu.lock` serializes the whole fleet; `tag-suffix=<problem>` keeps each problem's tag, logs, and `autocuda/optimize/<tag>/...` branches legible. See `autocuda/layout.md` for the full contract.

### Pick problems with real headroom

The best targets are problems whose reference is *not* already one optimal library call. `histogram` (reference `torch.bincount` — collapses under atomic contention) and `grayscale` (materializes a temporary then reduces) have large structural headroom. `matmul` (cuBLAS), `sort` (CUB radix), `conv2d` (cuDNN) are near-optimal already — weak demos.

## Skills

Reusable helpers ship as agent skills under `.claude/skills/<name>/`: a `SKILL.md` plus, for the script-backed ones (`popcorn-login`, `leaderboard-rankings`), a `scripts/` and an offline `tests/`. `.agents/skills` is a symlink to `.claude/skills`, so harnesses that look under either path find the same skills. Each skill's test runs offline: `bash .claude/skills/<name>/tests/*.sh`.
