#!/usr/bin/env python3
"""Fail fast when the local GPU MODE runtime drifts from production."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.specifiers import SpecifierSet


REQUIRED_PACKAGES = {
    "torch": "==2.12.0",
    "ninja": "~=1.11",
    "wheel": "~=0.45",
    "requests": "~=2.32.4",
    "packaging": "~=25.0",
    "numpy": "~=2.3",
    "tinygrad": "~=0.10",
    "nvidia-cutlass-dsl": "==4.5.2",
    "cuda-python": "==13.0",
    "cuda-tile": "==1.4.0",
    "nvmath-python": "==0.9.0",
    "nvidia-libmathdx-cu13": "==0.3.2.6",
    "cuda-toolkit": "==13.0.2",
    "tilelang": "==0.1.12",
}
REQUIRED_IMPORTS = ("torch", "cuda.tile", "nvmath", "cutlass.cute", "triton", "tilelang")


def fail(message: str) -> None:
    print(f"GPU MODE environment mismatch: {message}", file=sys.stderr)
    raise SystemExit(1)


if sys.version_info[:2] != (3, 13):
    fail(f"Python 3.13 required, found {sys.version.split()[0]} at {sys.executable}")

for package, spec in REQUIRED_PACKAGES.items():
    try:
        installed = version(package)
    except PackageNotFoundError:
        fail(f"missing package {package}; rerun bin/install.sh")
    if installed not in SpecifierSet(spec):
        fail(f"{package} {spec} required, found {installed}; rerun bin/install.sh")

for module in REQUIRED_IMPORTS:
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report the import's real failure
        fail(f"cannot import {module}: {exc}")

import torch  # noqa: E402

if torch.version.cuda != "13.0":
    fail(f"Torch CUDA runtime 13.0 required, found {torch.version.cuda}")

cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
nvcc = cuda_home / "bin" / "nvcc"
if not nvcc.is_file():
    fail(f"nvcc not found at {nvcc}")
nvcc_text = subprocess.run(
    [str(nvcc), "--version"], check=True, text=True, capture_output=True
).stdout
match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc_text)
if not match or match.group(1) != "13.3":
    fail(f"CUDA toolkit 13.3 required at {cuda_home}; nvcc reported {match.group(1) if match else 'unknown'}")
build_match = re.search(r"V([0-9]+\.[0-9]+\.[0-9]+)", nvcc_text)
if not build_match or build_match.group(1) != "13.3.33":
    fail(
        "CUDA nvcc 13.3.33 required for leaderboard parity; "
        f"found {build_match.group(1) if build_match else 'unknown'}"
    )

cutlass = Path(os.environ.get("CUTLASS_PATH", "/opt/cutlass"))
mathdx = Path(os.environ.get("MATHDX_HOME", "/opt/mathdx"))
if not (cutlass / "include/cutlass/cutlass.h").is_file():
    fail(f"CUTLASS headers missing under {cutlass}; rerun bin/install.sh")
cutlass_commit = subprocess.run(
    ["git", "-c", f"safe.directory={cutlass}", "-C", str(cutlass), "rev-parse", "HEAD"],
    check=False,
    text=True,
    capture_output=True,
).stdout.strip()
if cutlass_commit != "db1c288993354c88e551c40c19a8fb93a774a241":
    fail(f"CUTLASS v4.5.2 required under {cutlass}; rerun bin/install.sh")
if not (mathdx / "include/cublasdx.hpp").is_file():
    fail(f"MathDx headers missing under {mathdx}; rerun bin/install.sh")
mathdx_marker = mathdx / (
    ".gpumode-mathdx-26.06.0-"
    "042b7c57a636c271cca32dffcc0a822ed6b2abc0b8ef5703ab2445d58563a1e6"
)
if not mathdx_marker.is_file():
    fail(f"verified MathDx 26.06.0 install required under {mathdx}; rerun bin/install.sh")

cutile_rs = Path(os.environ.get("CUTILE_RS_PATH", "/opt/cutile-rs"))
cuda_oxide = Path(os.environ.get("CUDA_OXIDE_PATH", "/opt/cuda-oxide"))
if not (cutile_rs / ".gpumode-cutile-rs-0.2.0-d89788bca7de8a9cbeabc5ded63740520a96c223").is_file():
    fail(f"cuTile Rust v0.2.0 missing under {cutile_rs}; rerun bin/install.sh")
if not (cuda_oxide / ".gpumode-cuda-oxide-0.2.1-4514af2ca8a21a9f8feb187567f61fe67090f881").is_file():
    fail(f"CUDA Oxide v0.2.1 missing under {cuda_oxide}; rerun bin/install.sh")
for executable in (Path.home() / ".cargo/bin/cargo", Path.home() / ".cargo/bin/cargo-oxide", Path("/usr/bin/llc-21")):
    if not executable.is_file():
        fail(f"required programming-model tool missing: {executable}; rerun bin/install.sh")

print(
    f"GPU MODE environment OK: Python {sys.version.split()[0]}, "
    f"Torch {torch.__version__}, CUDA toolkit 13.3, runtime {torch.version.cuda}"
)
