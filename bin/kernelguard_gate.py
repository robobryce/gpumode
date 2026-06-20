#!/usr/bin/env python3
"""kernelguard gate — static reward-hack scan for a single submission.

Thin first-party wrapper around the kernelguard module (MIT,
SinatrasC/kernelguard), installed as a venv dependency via bin/install.sh
(`pip install kernelguard`). It reads ONE submission file, runs kernelguard's
rule-based detectors over the source, prints a human-readable report, and turns
the verdict into an EXIT CODE so the harness can gate on it:

    0  CLEAN     — kernelguard returns should_filter=False (valid / low-signal).
    2  FLAGGED   — should_filter=True (classification "hacked"): an auto-filter
                   rule fired (e.g. config-keyed result caching, hardcoded
                   output). This is the leaderboard-disqualifying class.
    3  ERROR     — kernelguard is not importable (run bin/install.sh) or the
                   submission file cannot be read.

We gate on kernelguard's own `should_filter` flag rather than re-deriving a
threshold: kernelguard already encodes a precision-first policy where only
auto-filter-grade rules set should_filter (support/telemetry-only signals do
NOT). Lower-severity matches are printed as warnings but do not fail the gate,
so honest kernels that merely *look* unusual are not blocked.

This is a STATIC scan (no GPU, no execution of the submission) — it complements,
does not replace, the runtime correctness guards. harness/validate.sh runs it
for EVERY problem, before spending GPU time, alongside the banned-stream scan.

Usage:  kernelguard_gate.py <submission.py>
        kernelguard_gate.py --self-test     # smoke-test the installed module
"""
from __future__ import annotations

import json
import sys

try:
    import kernelguard  # installed via bin/install.sh (pip install kernelguard)
except Exception as exc:  # pragma: no cover - missing dependency
    print("kernelguard gate: ERROR — cannot import kernelguard "
          f"({exc}); install it with bin/install.sh or `pip install kernelguard`",
          file=sys.stderr)
    sys.exit(3)


def scan(code: str) -> dict:
    # compute_structural_hash is only useful for cross-submission dedup; skip the
    # expensive identifier normalization for a single-file gate.
    return kernelguard.analyze_code(code, compute_structural_hash=False)


def report(path: str, result: dict) -> int:
    classification = result.get("classification", "unknown")
    should_filter = bool(result.get("should_filter"))
    patterns = result.get("matched_patterns", []) or []

    print(f"kernelguard gate: {path}")
    print(f"  classification: {classification}    should_filter: {should_filter}")
    if patterns:
        print(f"  matched patterns ({len(patterns)}):")
        for p in patterns:
            name = p.get("pattern") or p.get("rule") or "?"
            severity = p.get("severity", "")
            evidence = (p.get("evidence", "") or "").replace("\n", " ")[:120]
            field = p.get("field", "")
            print(f"    - [{severity:>6}] {name} ({field}): {evidence}")
    else:
        print("  matched patterns: none")

    if should_filter:
        reason = result.get("filter_reason") or "auto-filter rule fired"
        print(f"kernelguard gate: FLAGGED — {reason}")
        return 2
    print("kernelguard gate: CLEAN")
    return 0


def self_test() -> int:
    """Smoke test: a known cache hack must FLAG, a trivial kernel must be CLEAN."""
    hack = (
        "import torch\n_C={}\n"
        "def custom_kernel(data):\n"
        "    k=data.shape\n"
        "    if k in _C: return _C[k]\n"
        "    r=torch.linalg.qr(data); _C[k]=r; return r\n"
    )
    clean = (
        "import torch\n"
        "def custom_kernel(data):\n"
        "    a,b=data\n"
        "    return a@b\n"
    )
    hr = scan(hack)
    cr = scan(clean)
    ok = bool(hr.get("should_filter")) and not bool(cr.get("should_filter"))
    print(json.dumps({
        "hack_should_filter": hr.get("should_filter"),
        "hack_classification": hr.get("classification"),
        "clean_should_filter": cr.get("should_filter"),
        "clean_classification": cr.get("classification"),
        "self_test": "PASS" if ok else "FAIL",
    }, indent=2))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 3
    path = argv[1]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            code = fh.read()
    except OSError as exc:
        print(f"kernelguard gate: ERROR — cannot read {path}: {exc}", file=sys.stderr)
        return 3
    return report(path, scan(code))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
