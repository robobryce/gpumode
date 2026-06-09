#!/usr/bin/env bash
# One-time machine setup for optimizing GPU MODE leaderboard problems with
# autocuda, IN PLACE inside this reference-kernels fork.
#
# Idempotent. Provisions:
#   1. popcorn-cli            (GPU MODE submission CLI) -> ~/.local/bin
#   2. a Python venv + PyTorch (CUDA build) + numpy/pyyaml/ninja
#   3. ~/.config/gpumode/gpumode.env  (toolchain paths the harness sources)
#
# Unlike a separate harness repo, this does NOT clone reference-kernels — this
# repo IS it. After install, run /autocuda:discover once to write
# autocuda/environment.md, then optimize a problem in place (see README).
#
# Tunables via environment:
#   GPUMODE_ROOT      where the venv lives                (default ~/gpumode)
#   GPUMODE_PY        python to build the venv from        (default python3.12 or python3)
#   TORCH_INDEX_URL   PyTorch wheel index (match your CUDA/GPU)
#                     default https://download.pytorch.org/whl/cu128 (H100/sm_90)
#   CUDA_HOME         CUDA toolkit root  (default /usr/local/cuda)
set -euo pipefail

GPUMODE_ROOT="${GPUMODE_ROOT:-$HOME/gpumode}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CFG_DIR="$HOME/.config/gpumode"
CFG="$CFG_DIR/gpumode.env"
mkdir -p "$GPUMODE_ROOT" "$CFG_DIR" "$HOME/.local/bin"

log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }

# --- 0. sanity: GPU + nvcc ---------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found — need an NVIDIA GPU host"; exit 1; }
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "nvcc not found at $CUDA_HOME/bin/nvcc — set CUDA_HOME to your toolkit root"; exit 1
fi
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); nvcc: $("$CUDA_HOME/bin/nvcc" --version | grep -oE 'release [0-9.]+' | head -1)"

# --- 1. popcorn-cli ----------------------------------------------------------
if command -v popcorn-cli >/dev/null 2>&1 || [ -x "$HOME/.local/bin/popcorn-cli" ]; then
    log "popcorn-cli already installed ($(command -v popcorn-cli || echo "$HOME/.local/bin/popcorn-cli"))"
else
    log "installing popcorn-cli ..."
    curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
fi

# --- 2. venv + torch ---------------------------------------------------------
VENV="$GPUMODE_ROOT/.venv"
PYBIN="$VENV/bin/python"
if [ -x "$PYBIN" ] && "$PYBIN" -c "import torch" 2>/dev/null; then
    log "venv + torch already present at $VENV ($("$PYBIN" -c 'import torch;print(torch.__version__)'))"
else
    GPUMODE_PY="${GPUMODE_PY:-$(command -v python3.12 || command -v python3)}"
    log "creating venv at $VENV from $GPUMODE_PY ..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$GPUMODE_PY" "$VENV"
        VIRTUAL_ENV="$VENV" uv pip install --python "$PYBIN" numpy pyyaml ninja
        log "installing torch from $TORCH_INDEX_URL (large download) ..."
        VIRTUAL_ENV="$VENV" uv pip install --python "$PYBIN" torch --index-url "$TORCH_INDEX_URL"
    else
        "$GPUMODE_PY" -m venv "$VENV"
        "$PYBIN" -m pip install --upgrade pip
        "$PYBIN" -m pip install numpy pyyaml ninja
        log "installing torch from $TORCH_INDEX_URL (large download) ..."
        "$PYBIN" -m pip install torch --index-url "$TORCH_INDEX_URL"
    fi
fi
log "torch CUDA check: $("$PYBIN" -c 'import torch;print("avail",torch.cuda.is_available(),"dev",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")')"

# --- 3. write machine config -------------------------------------------------
cat > "$CFG" <<EOF
# GPU MODE machine config — written by bin/install.sh. Sourced by harness/env.sh.
GPUMODE_VENV_PYTHON="$PYBIN"
CUDA_HOME="$CUDA_HOME"
EOF
log "wrote $CFG:"; sed 's/^/    /' "$CFG"

cat <<EOF

$(printf '\033[1;32m✓ install complete\033[0m')

Next steps:
  1. Authenticate popcorn-cli (one-time) with the popcorn-login skill:
       bash .claude/skills/popcorn-login/scripts/login.sh
  2. Set up this machine:   /autocuda:discover   (writes autocuda/environment.md)
  3. Pick a problem and optimize it in place — its <set>/<problem> path is the
     one token (passed as benchmark=), e.g.:
       /autocuda:optimize-tree workers=4 benchmark=pmpp_v2/histogram_py tag-suffix=histogram_py
EOF
