#!/usr/bin/env python3
"""Local runner for the GPU MODE / KernelBot `linalg/qr_v2` problem.

This reproduces, on a single local GPU, what the KernelBot leaderboard
infrastructure does for you remotely — WITHOUT submitting anything to the
leaderboard. It stages the files named in ``task.yml`` into a flat working
directory, generates the ``test`` / ``benchmark`` spec files from the
``tests:`` / ``benchmarks:`` blocks of ``task.yml``, then runs the read-only
``eval.py`` harness over them via the popcorn ``POPCORN_FD`` protocol and
prints the structured results.

It is deliberately self-contained (only PyYAML + torch) and never touches the
network. Pick the submission to score with ``--submission`` (default:
``submission.py`` in this directory).

Usage:
    python run_local.py test                 # correctness gate (22 cases)
    python run_local.py benchmark            # timed ranking set (12 cases)
    python run_local.py leaderboard          # local-only ranked-style rerun
    python run_local.py test --submission my_kernel.py
    python run_local.py benchmark --seed 7   # perturb the case seeds

Exit code is the harness exit code: 0 = all pass, non-zero = a failure
(112 = a case failed, 113 = bad spec file, 111 = POPCORN_FD unset).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TASK_YML = HERE / "task.yml"

# Keys that get serialized into a spec line, in a stable order. `case` is
# optional (omitted when absent, matching how KernelBot generates specs).
_SPEC_KEYS = ("batch", "n", "cond", "seed", "case")


def _spec_line(case: dict) -> str:
    """Serialize one test/benchmark dict to the `k: v; k: v` spec format that
    eval.py's get_test_cases() parses."""
    parts = []
    for key in _SPEC_KEYS:
        if key in case:
            parts.append(f"{key}: {case[key]}")
    # Preserve any extra keys we did not anticipate, so the runner never
    # silently drops a spec field a future task.yml adds.
    for key, val in case.items():
        if key not in _SPEC_KEYS:
            parts.append(f"{key}: {val}")
    return "; ".join(parts)


def _write_specs(cases: list[dict], path: Path) -> None:
    path.write_text("\n".join(_spec_line(c) for c in cases) + "\n")


def _resolve_source(name: str, source: str, submission: Path) -> Path:
    """Map a task.yml `files:` entry to a concrete source path."""
    if source == "@SUBMISSION@":
        return submission
    return (HERE / source).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["test", "benchmark", "leaderboard"],
                    help="eval.py mode to run")
    ap.add_argument("--submission", default=str(HERE / "submission.py"),
                    help="submission file to score (default: ./submission.py)")
    ap.add_argument("--seed", type=int, default=None,
                    help="optional POPCORN_SEED to perturb every case's seed")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to run eval.py with")
    ap.add_argument("--keep", action="store_true",
                    help="keep the staged working directory for inspection")
    args = ap.parse_args()

    submission = Path(args.submission).resolve()
    if not submission.is_file():
        print(f"run_local: submission not found: {submission}", file=sys.stderr)
        return 2

    spec = yaml.safe_load(TASK_YML.read_text())
    main_entry = spec.get("config", {}).get("main", "eval.py")

    work = Path(tempfile.mkdtemp(prefix="qr_v2_local_"))
    try:
        # Stage every file named in the task.yml manifest into one flat dir,
        # exactly as the KernelBot runner would, so intra-package imports
        # (submission/task/reference/utils/eval) all resolve flatly.
        for entry in spec["files"]:
            src = _resolve_source(entry["name"], entry["source"], submission)
            if not src.is_file():
                print(f"run_local: missing source for {entry['name']}: {src}",
                      file=sys.stderr)
                return 2
            shutil.copy2(src, work / entry["name"])

        # Generate the spec file for this mode. `leaderboard` reuses the
        # benchmark set, matching task.yml (benchmarks drive the ranking).
        cases = spec["tests"] if args.mode == "test" else spec["benchmarks"]
        spec_file = work / ("tests.txt" if args.mode == "test" else "benchmarks.txt")
        _write_specs(cases, spec_file)

        # Drive eval.py via the popcorn protocol: results are written to the
        # fd named by POPCORN_FD. We hand it a pipe and stream it to stdout.
        read_fd, write_fd = os.pipe()
        env = dict(os.environ)
        env["POPCORN_FD"] = str(write_fd)
        if args.seed is not None:
            env["POPCORN_SEED"] = str(args.seed)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(work), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

        proc = subprocess.Popen(
            [args.python, main_entry, args.mode, spec_file.name],
            cwd=str(work), env=env, pass_fds=(write_fd,))
        os.close(write_fd)
        with os.fdopen(read_fd, "r") as r:
            for line in r:
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = proc.wait()
        return rc
    finally:
        if args.keep:
            print(f"run_local: staged dir kept at {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
