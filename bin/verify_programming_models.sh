#!/usr/bin/env bash
# Compile and execute one small GPU kernel in every supported programming model.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GPUMODE_VENV_PYTHON:-$REPO_DIR/.venv/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUTLASS_PATH="${CUTLASS_PATH:-/opt/cutlass}"
CUTILE_RS_PATH="${CUTILE_RS_PATH:-/opt/cutile-rs}"
CUDA_OXIDE_PATH="${CUDA_OXIDE_PATH:-/opt/cuda-oxide}"
export CUDA_HOME CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-$CUDA_HOME}"
export CUDA_OXIDE_LLC="${CUDA_OXIDE_LLC:-/usr/bin/llc-21}"
export PATH="$(dirname "$PYTHON"):$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$CUDA_HOME/bin:$PATH"

log() { printf '\033[1;35m[programming-models]\033[0m %s\n' "$*"; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

CUDA_ARCH="${GPUMODE_CUDA_ARCH:-$("$PYTHON" - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
print(f"sm_{major}{minor}")
PY
)}"
[[ "$CUDA_ARCH" =~ ^sm_[0-9]+$ ]] || {
    echo "invalid CUDA architecture: $CUDA_ARCH" >&2
    exit 1
}

log "CUDA C++ ($CUDA_ARCH)"
"$CUDA_HOME/bin/nvcc" -std=c++17 "-arch=$CUDA_ARCH" "$REPO_DIR/bin/smoke/cuda_cpp.cu" -o "$tmp/cuda-cpp"
"$tmp/cuda-cpp"

log "CuTe DSL"
"$PYTHON" "$CUTLASS_PATH/examples/python/CuTeDSL/dsl_tutorials/dataclass_immutable.py"

log "cuTile Python"
"$PYTHON" "$REPO_DIR/bin/smoke/cutile_python.py"

log "Triton"
"$PYTHON" "$REPO_DIR/bin/smoke/triton_smoke.py"

log "TileLang"
"$PYTHON" "$REPO_DIR/bin/smoke/tilelang_smoke.py"

log "cuTile Rust"
cargo run --manifest-path "$CUTILE_RS_PATH/Cargo.toml" \
    --release -p cutile-examples --example saxpy

log "CUDA Oxide"
(cd "$CUDA_OXIDE_PATH" && cargo oxide doctor && cargo oxide run vecadd)

log "all programming models passed"
