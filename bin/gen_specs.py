#!/usr/bin/env python3
"""Read a GPU MODE problem's task.yml and emit what the autocuda harness needs.

eval.py (the official KernelBot harness) reads a spec file whose every line is
`key: value; key: value; ...` (e.g. `size: 1310720; seed: 6252`). The backend
generates these from task.yml; we do the same locally, on the fly, so nothing
is copied or scaffolded — the problem's own task.yml is the single source.

Usage:
    gen_specs.py <task.yml> --emit tests          # spec lines on stdout
    gen_specs.py <task.yml> --emit benchmarks
    gen_specs.py <task.yml> --leaderboard         # leaderboard/metric name
    gen_specs.py <task.yml> --gpus                # supported GPU tokens, CSV
    gen_specs.py <task.yml> --meta                # TEST_TIMEOUT=… etc. (KEY=VALUE)

The leaderboard name and supported GPUs come from the set-level yaml
(e.g. problems/pmpp_v2.yaml) that maps each problem `directory:` to its
`name:` / `gpus:`; we locate the entry whose directory matches the task.yml's
problem dir. Falls back to the problem dir's basename if no set-yaml lists it.
"""
import argparse
import glob
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not available; run inside the gpumode venv "
             "(GPUMODE_VENV_PYTHON) which install.sh provisions with pyyaml.")


def _emit(items):
    # Order within a line does not matter: eval.py parses each part into a dict
    # and passes it as **kwargs to generate_input.
    return "".join("; ".join(f"{k}: {v}" for k, v in it.items()) + "\n" for it in items)


def _resolve_meta(task_path):
    """Return (leaderboard_name, gpus_list) by matching this problem's dir
    against the set-level yaml registry one level up."""
    problem_dir = os.path.dirname(os.path.abspath(task_path))
    base = os.path.basename(problem_dir)
    set_dir = os.path.dirname(problem_dir)            # problems/<set>
    problems_root = os.path.dirname(set_dir)          # problems/
    set_name = os.path.basename(set_dir)
    full = f"{set_name}/{base}"

    ymls = sorted(glob.glob(os.path.join(problems_root, "*.yaml")) +
                  glob.glob(os.path.join(problems_root, "*.yml")))

    def search(pred):
        for yml in ymls:
            try:
                doc = yaml.safe_load(open(yml)) or {}
            except Exception:
                continue
            for prob in (doc.get("problems") or []):
                d = (prob.get("directory") or "").strip("/")
                if pred(d, set_name):
                    return prob.get("name") or base, prob.get("gpus") or []
        return None

    # Prefer a full <set>/<problem> match (several sets reuse a trailing dir
    # name); fall back to a trailing-basename match.
    hit = (search(lambda d, s: d == full or d == base and s == set_name)
           or search(lambda d, _s: os.path.basename(d) == base))
    return hit if hit else (base, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_yml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--emit", choices=("tests", "benchmarks"))
    g.add_argument("--leaderboard", action="store_true")
    g.add_argument("--gpus", action="store_true")
    g.add_argument("--meta", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.task_yml)) or {}

    if args.emit:
        sys.stdout.write(_emit(spec.get(args.emit, []) or []))
        return

    if args.leaderboard:
        print(_resolve_meta(args.task_yml)[0])
        return

    if args.gpus:
        print(",".join(_resolve_meta(args.task_yml)[1]))
        return

    if args.meta:
        name, gpus = _resolve_meta(args.task_yml)
        print(f"LEADERBOARD={name}")
        print(f"GPUS={','.join(gpus)}")
        for key in ("test_timeout", "benchmark_timeout", "ranked_timeout"):
            if key in spec:
                print(f"{key.upper()}={spec[key]}")
        print(f"N_TESTS={len(spec.get('tests', []) or [])}")
        print(f"N_BENCHMARKS={len(spec.get('benchmarks', []) or [])}")


if __name__ == "__main__":
    main()
