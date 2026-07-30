"""Run this checkout with the production GPU MODE Modal dependencies.

The image below intentionally mirrors gpu-mode/kernelbot's production
``src/runners/modal_runner.py`` at KERNELBOT_IMAGE_COMMIT.  Keep the package
list exact: adding convenient local-only packages can make a submission pass
here and fail on the leaderboard (CuPy was one such false positive).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/workspace/gpumode"
GPU = os.environ.get("MODAL_GPU", "B200")

# Source of truth used when this mirror was last synchronized:
# https://github.com/gpu-mode/kernelbot/blob/4bdd839ad76eaac0cdab986d30f8d77f989a289b/src/runners/modal_runner.py
KERNELBOT_IMAGE_COMMIT = "4bdd839ad76eaac0cdab986d30f8d77f989a289b"
CUDA_TAG = "13.3.0-devel-ubuntu24.04"
MATHDX_VERSION = "26.06.0"
MATHDX_ARCHIVE = f"nvidia-mathdx-{MATHDX_VERSION}-cuda13.tar.gz"
MATHDX_URL = (
    "https://developer.download.nvidia.com/compute/cublasdx/redist/"
    f"cublasdx/cuda13/{MATHDX_ARCHIVE}"
)
MATHDX_SHA256 = "042b7c57a636c271cca32dffcc0a822ed6b2abc0b8ef5703ab2445d58563a1e6"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install("git", "curl", "gcc-13", "g++-13", "clang-18")
    .uv_pip_install(
        "ninja~=1.11",
        "wheel~=0.45",
        "requests~=2.32.4",
        "packaging~=25.0",
        "numpy~=2.3",
        "pytest",
        "PyYAML",
    )
    .uv_pip_install("tinygrad~=0.10", "helion")
    .uv_pip_install(
        "nvidia-cutlass-dsl==4.5.2",
        "cuda-core[cu13]",
        "cuda-python[all]==13.0",
        "cuda-tile==1.4.0",
        "nvmath-python[cu13-dx]==0.9.0",
        "nvidia-libmathdx-cu13==0.3.2.6",
        "cuda-toolkit[cccl,nvrtc]==13.0.2",
    )
    # Production installs torch last so its CUDA/NCCL dependency set wins.
    .uv_pip_install("torch==2.12.0")
    .run_commands(
        "git clone --depth 1 --branch v4.5.2 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
        (
            f"curl -fsSL {MATHDX_URL} -o /tmp/{MATHDX_ARCHIVE} && "
            f"echo '{MATHDX_SHA256}  /tmp/{MATHDX_ARCHIVE}' | sha256sum -c - && "
            "mkdir -p /opt/mathdx && "
            f"tar -xzf /tmp/{MATHDX_ARCHIVE} --strip-components=4 -C /opt/mathdx && "
            f"rm /tmp/{MATHDX_ARCHIVE} && "
            f"touch /opt/mathdx/.gpumode-mathdx-{MATHDX_VERSION}-{MATHDX_SHA256}"
        ),
    )
    .env(
        {
            "CUTLASS_PATH": "/opt/cutlass",
            "MATHDX_HOME": "/opt/mathdx",
            "CPLUS_INCLUDE_PATH": (
                "/opt/mathdx/include:/opt/mathdx/external/cutlass/include:"
                "/opt/cutlass/include:/opt/cutlass/tools/util/include"
            ),
        }
    )
    .run_commands(
        "python -m pip check",
        'python -c "import cuda.tile, nvmath"',
        "tileiras --version",
        (
            "printf '#include <cublasdx.hpp>\\n' | "
            "nvcc -std=c++17 -x cu -c - -o /tmp/cublasdx-smoke.o && "
            "rm /tmp/cublasdx-smoke.o"
        ),
    )
    .add_local_dir(
        LOCAL_REPO,
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[
            ".git/**",
            "autocuda/optimize/**",
            "**/.torch_ext/**",
            "**/__pycache__/**",
        ],
    )
)
app = modal.App("gpumode-autocuda", image=image)


@app.function(gpu=GPU, timeout=60 * 60)
def run_harness(action: str, problem: str) -> tuple[int, str, str]:
    import subprocess

    if action not in {"validate", "benchmark"}:
        return 2, "", f"unsupported action: {action}\n"

    env = os.environ.copy()
    env.update(
        GPUMODE_VENV_PYTHON=sys.executable,
        CUDA_HOME="/usr/local/cuda",
        CUDA_VISIBLE_DEVICES="0",
        GPUMODE_SKIP_STATIC_CHECKS="1",
    )
    result = subprocess.run(
        ["bash", f"harness/{action}.sh", problem],
        cwd=REMOTE_REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode, result.stdout, result.stderr


@app.local_entrypoint()
def main(action: str, problem: str) -> None:
    """Run ACTION (validate or benchmark) for SET/PROBLEM on Modal."""
    if action not in {"validate", "benchmark"}:
        raise SystemExit("--action must be 'validate' or 'benchmark'")

    returncode, stdout, stderr = run_harness.remote(action, problem)
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    if returncode:
        raise SystemExit(returncode)
