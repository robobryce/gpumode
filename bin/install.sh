#!/usr/bin/env bash
# Reproducible GPU MODE host setup.
#
# Installs/selects CUDA 13.3, creates a fresh Python 3.13 venv in this checkout,
# mirrors the production KernelBot/Modal dependency set, installs the CUTLASS
# and MathDx header trees used by that image, and writes the machine config
# consumed by harness/env.sh.
#
# Safe to rerun: the venv is rebuilt from scratch and versioned assets are
# reused when already correct.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_SERIES="${GPUMODE_CUDA_SERIES:-13.3}"
CUDA_PACKAGE_SERIES="${CUDA_SERIES/./-}"
CUDA_PREFIX="${GPUMODE_CUDA_PREFIX:-/usr/local/cuda-$CUDA_SERIES}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VENV="${GPUMODE_RUNTIME_VENV:-$REPO_DIR/.venv}"
PYBIN="$VENV/bin/python"
CFG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/gpumode"
CFG="$CFG_DIR/gpumode.env"
CUTLASS_VERSION="4.5.2"
CUTLASS_PATH="${CUTLASS_PATH:-/opt/cutlass}"
CUTILE_RS_VERSION="0.2.0"
CUTILE_RS_COMMIT="d89788bca7de8a9cbeabc5ded63740520a96c223"
CUTILE_RS_PATH="${CUTILE_RS_PATH:-/opt/cutile-rs}"
CUDA_OXIDE_VERSION="0.2.1"
CUDA_OXIDE_COMMIT="4514af2ca8a21a9f8feb187567f61fe67090f881"
CUDA_OXIDE_PATH="${CUDA_OXIDE_PATH:-/opt/cuda-oxide}"
CUDA_OXIDE_TOOLCHAIN="nightly-2026-04-03"
TILELANG_VERSION="0.1.12"
MATHDX_VERSION="26.06.0"
MATHDX_HOME="${MATHDX_HOME:-/opt/mathdx}"
MATHDX_ARCHIVE="nvidia-mathdx-${MATHDX_VERSION}-cuda13.tar.gz"
MATHDX_URL="https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/cuda13/$MATHDX_ARCHIVE"
MATHDX_SHA256="042b7c57a636c271cca32dffcc0a822ed6b2abc0b8ef5703ab2445d58563a1e6"
MATHDX_MARKER="$MATHDX_HOME/.gpumode-mathdx-$MATHDX_VERSION-$MATHDX_SHA256"

log() { printf '\033[1;36m[gpumode-setup]\033[0m %s\n' "$*"; }
die() { echo "gpumode setup: $*" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo -n)
else
    die "root or passwordless sudo is required to install CUDA and /opt assets"
fi

nvcc_release() {
    "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1
}

install_host_tools() {
    command -v apt-get >/dev/null 2>&1 || {
        command -v curl >/dev/null 2>&1 || die "curl is required"
        command -v git >/dev/null 2>&1 || die "git is required"
        return 0
    }
    log "installing host build tools"
    "${SUDO[@]}" apt-get update
    DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential gcc-13 g++-13 clang-18 pkg-config \
        cmake libedit-dev libxml2-dev zlib1g-dev lsb-release wget gnupg software-properties-common
}

install_cuda() {
    if [ -x "$CUDA_PREFIX/bin/nvcc" ] && [ "$(nvcc_release "$CUDA_PREFIX")" = "$CUDA_SERIES" ]; then
        log "CUDA $CUDA_SERIES already present at $CUDA_PREFIX"
    else
        command -v apt-get >/dev/null 2>&1 \
            || die "CUDA $CUDA_SERIES is missing and automatic installation requires apt-get"
        . /etc/os-release
        [ "${ID:-}" = ubuntu ] || die "automatic CUDA installation currently supports Ubuntu only"
        case "${VERSION_ID:-}" in
            24.04) cuda_repo=ubuntu2404 ;;
            22.04) cuda_repo=ubuntu2204 ;;
            *) die "unsupported Ubuntu release ${VERSION_ID:-unknown} for automatic CUDA installation" ;;
        esac
        arch="$(dpkg --print-architecture)"
        [ "$arch" = amd64 ] || die "automatic CUDA installation currently supports amd64 only"
        keyring="$(mktemp --suffix=.deb)"
        trap 'rm -f "${keyring:-}" "${mathdx_tmp:-}"' EXIT
        log "installing CUDA toolkit $CUDA_SERIES (driver packages are not installed)"
        curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/$cuda_repo/x86_64/cuda-keyring_1.1-1_all.deb" -o "$keyring"
        "${SUDO[@]}" dpkg -i "$keyring"
        "${SUDO[@]}" apt-get update
        DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y --no-install-recommends \
            "cuda-toolkit-$CUDA_PACKAGE_SERIES=13.3.0-1" \
            'cuda-compiler-13-3=13.3.0-1' \
            'cuda-nvcc-13-3=13.3.33-1' 'cuda-crt-13-3=13.3.33-1' \
            'libnvvm-13-3=13.3.33-1' 'libnvptxcompiler-13-3=13.3.33-1'
        "${SUDO[@]}" apt-mark hold \
            "cuda-toolkit-$CUDA_PACKAGE_SERIES" cuda-compiler-13-3 \
            cuda-nvcc-13-3 cuda-crt-13-3 libnvvm-13-3 libnvptxcompiler-13-3
    fi

    if command -v update-alternatives >/dev/null 2>&1; then
        "${SUDO[@]}" update-alternatives --install /usr/local/cuda cuda "$CUDA_PREFIX" 133
        "${SUDO[@]}" update-alternatives --set cuda "$CUDA_PREFIX"
    else
        "${SUDO[@]}" ln -sfn "$CUDA_PREFIX" /usr/local/cuda
    fi
    CUDA_HOME=/usr/local/cuda
    [ "$(nvcc_release "$CUDA_HOME")" = "$CUDA_SERIES" ] \
        || die "$CUDA_HOME does not resolve to CUDA $CUDA_SERIES"
    nvcc_build="$($CUDA_HOME/bin/nvcc --version | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | head -n1)"
    [ "$nvcc_build" = 13.3.33 ] \
        || die "CUDA nvcc 13.3.33 required for leaderboard parity, found ${nvcc_build:-unknown}"
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then return; fi
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installation failed"
}

install_llvm21() {
    if command -v llc-21 >/dev/null 2>&1 && command -v clang-21 >/dev/null 2>&1; then
        return
    fi
    command -v apt-get >/dev/null 2>&1 || die "LLVM/Clang 21 installation requires apt-get"
    log "installing LLVM and Clang 21 for CUDA Oxide"
    llvm_script="$(mktemp)"
    trap 'rm -f "${keyring:-}" "${mathdx_tmp:-}" "${llvm_script:-}"' EXIT
    curl -fsSL https://apt.llvm.org/llvm.sh -o "$llvm_script"
    "${SUDO[@]}" bash "$llvm_script" 21
    DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y --no-install-recommends \
        llvm-21 llvm-21-dev clang-21 libclang-21-dev
    "${SUDO[@]}" update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 210
    "${SUDO[@]}" update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 210
}

install_rust() {
    if ! command -v rustup >/dev/null 2>&1; then
        log "installing rustup"
        rustup_script="$(mktemp)"
        trap 'rm -f "${keyring:-}" "${mathdx_tmp:-}" "${llvm_script:-}" "${rustup_script:-}"' EXIT
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$rustup_script"
        sh "$rustup_script" -y --profile minimal
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
    rustup toolchain install stable --profile minimal
    rustup toolchain install "$CUDA_OXIDE_TOOLCHAIN" --profile minimal \
        --component rust-src --component rustc-dev --component llvm-tools
}

install_sdk_headers() {
    if [ ! -f "$CUTLASS_PATH/include/cutlass/cutlass.h" ]; then
        log "installing CUTLASS v$CUTLASS_VERSION headers at $CUTLASS_PATH"
        "${SUDO[@]}" rm -rf "$CUTLASS_PATH"
        "${SUDO[@]}" git clone --depth 1 --branch "v$CUTLASS_VERSION" \
            https://github.com/NVIDIA/cutlass.git "$CUTLASS_PATH"
    fi
    cutlass_commit="$("${SUDO[@]}" git -c safe.directory="$CUTLASS_PATH" -C "$CUTLASS_PATH" rev-parse HEAD 2>/dev/null || true)"
    [ "$cutlass_commit" = db1c288993354c88e551c40c19a8fb93a774a241 ] \
        || die "$CUTLASS_PATH is not CUTLASS v$CUTLASS_VERSION; remove it and rerun bin/install.sh"

    if [ ! -f "$MATHDX_MARKER" ]; then
        log "installing MathDx $MATHDX_VERSION headers at $MATHDX_HOME"
        mathdx_tmp="$(mktemp --suffix=.tar.gz)"
        curl -fsSL "$MATHDX_URL" -o "$mathdx_tmp"
        echo "$MATHDX_SHA256  $mathdx_tmp" | sha256sum -c -
        "${SUDO[@]}" rm -rf "$MATHDX_HOME"
        "${SUDO[@]}" mkdir -p "$MATHDX_HOME"
        "${SUDO[@]}" tar -xzf "$mathdx_tmp" --strip-components=4 -C "$MATHDX_HOME"
        "${SUDO[@]}" touch "$MATHDX_MARKER"
    fi

    CPLUS_INCLUDE_PATH="$MATHDX_HOME/include:$MATHDX_HOME/external/cutlass/include:$CUTLASS_PATH/include:$CUTLASS_PATH/tools/util/include"
    export CPLUS_INCLUDE_PATH
    printf '#include <cublasdx.hpp>\n' | "$CUDA_HOME/bin/nvcc" \
        -std=c++17 -x cu -c - -o /tmp/gpumode-cublasdx-smoke.o
    "${SUDO[@]}" rm -f /tmp/gpumode-cublasdx-smoke.o
}

install_rust_dsls() {
    if [ ! -f "$CUTILE_RS_PATH/.gpumode-cutile-rs-$CUTILE_RS_VERSION-$CUTILE_RS_COMMIT" ]; then
        log "installing cuTile Rust v$CUTILE_RS_VERSION at $CUTILE_RS_PATH"
        "${SUDO[@]}" rm -rf "$CUTILE_RS_PATH"
        "${SUDO[@]}" git clone --depth 1 --branch "v$CUTILE_RS_VERSION" \
            https://github.com/NVlabs/cutile-rs.git "$CUTILE_RS_PATH"
        "${SUDO[@]}" touch "$CUTILE_RS_PATH/.gpumode-cutile-rs-$CUTILE_RS_VERSION-$CUTILE_RS_COMMIT"
    fi
    "${SUDO[@]}" chown -R "$(id -u):$(id -g)" "$CUTILE_RS_PATH"
    if [ ! -f "$CUDA_OXIDE_PATH/.gpumode-cuda-oxide-$CUDA_OXIDE_VERSION-$CUDA_OXIDE_COMMIT" ]; then
        log "installing CUDA Oxide v$CUDA_OXIDE_VERSION at $CUDA_OXIDE_PATH"
        "${SUDO[@]}" rm -rf "$CUDA_OXIDE_PATH"
        "${SUDO[@]}" git clone --depth 1 --branch "v$CUDA_OXIDE_VERSION" \
            https://github.com/NVlabs/cuda-oxide.git "$CUDA_OXIDE_PATH"
        "${SUDO[@]}" touch "$CUDA_OXIDE_PATH/.gpumode-cuda-oxide-$CUDA_OXIDE_VERSION-$CUDA_OXIDE_COMMIT"
    fi
    "${SUDO[@]}" chown -R "$(id -u):$(id -g)" "$CUDA_OXIDE_PATH"
    log "installing cargo-oxide v$CUDA_OXIDE_VERSION"
    CARGO_TARGET_DIR="$HOME/.cache/gpumode/cargo-oxide-install" \
        cargo "+$CUDA_OXIDE_TOOLCHAIN" install --locked --force \
        --path "$CUDA_OXIDE_PATH/crates/cargo-oxide"
}

install_runtime() {
    log "recreating production-matched Python 3.13 runtime at $VENV"
    uv venv --clear --python 3.13 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install --python "$PYBIN" \
        'ninja~=1.11' 'wheel~=0.45' 'requests~=2.32.4' 'packaging~=25.0' \
        'numpy~=2.3' pytest PyYAML 'tinygrad~=0.10' helion \
        'nvidia-cutlass-dsl==4.5.2' 'cuda-core[cu13]' \
        'cuda-python[all]==13.0' 'cuda-tile==1.4.0' \
        'nvmath-python[cu13-dx]==0.9.0' 'nvidia-libmathdx-cu13==0.3.2.6' \
        'cuda-toolkit[cccl,nvrtc]==13.0.2'
    # Match production ordering: Torch's CUDA/NCCL dependency set wins.
    VIRTUAL_ENV="$VENV" uv pip install --python "$PYBIN" 'torch==2.12.0'
    # Local control-plane tools are intentionally absent from the remote image.
    VIRTUAL_ENV="$VENV" uv pip install --python "$PYBIN" kernelguard 'modal>=1.1'
    VIRTUAL_ENV="$VENV" uv pip install --python "$PYBIN" "tilelang==$TILELANG_VERSION"
    uv pip check --python "$PYBIN"
}

write_config() {
    mkdir -p "$CFG_DIR"
    umask 022
    {
        echo '# GPU MODE machine config — written by bin/install.sh. Sourced by harness/env.sh.'
        printf 'GPUMODE_VENV_PYTHON=%q\n' "$PYBIN"
        printf 'CUDA_HOME=%q\n' "$CUDA_HOME"
        printf 'CUTLASS_PATH=%q\n' "$CUTLASS_PATH"
        printf 'MATHDX_HOME=%q\n' "$MATHDX_HOME"
        printf 'CUTILE_RS_PATH=%q\n' "$CUTILE_RS_PATH"
        printf 'CUDA_OXIDE_PATH=%q\n' "$CUDA_OXIDE_PATH"
        printf 'CUDA_TOOLKIT_PATH=%q\n' "$CUDA_HOME"
        printf 'CUDA_OXIDE_LLC=%q\n' /usr/bin/llc-21
    } > "$CFG"
    log "wrote $CFG"
}

install_host_tools
install_cuda
install_uv
install_llvm21
install_rust
install_sdk_headers
install_rust_dsls
install_runtime
write_config

export GPUMODE_VENV_PYTHON="$PYBIN" CUDA_HOME CUTLASS_PATH MATHDX_HOME
export CUTILE_RS_PATH CUDA_OXIDE_PATH CUDA_TOOLKIT_PATH="$CUDA_HOME"
export CUDA_OXIDE_LLC=/usr/bin/llc-21
export PATH="$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$PATH"
export CPLUS_INCLUDE_PATH="$MATHDX_HOME/include:$MATHDX_HOME/external/cutlass/include:$CUTLASS_PATH/include:$CUTLASS_PATH/tools/util/include"
"$PYBIN" "$REPO_DIR/bin/verify_environment.py"
"$PYBIN" "$REPO_DIR/bin/kernelguard_gate.py" --self-test
"$REPO_DIR/bin/verify_programming_models.sh"

if [ "${GPUMODE_INSTALL_POPCORN:-1}" = 1 ]; then
    if command -v popcorn-cli >/dev/null 2>&1 || [ -x "$HOME/.local/bin/popcorn-cli" ]; then
        log "popcorn-cli already installed"
    else
        log "installing popcorn-cli"
        curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
    fi
fi

log "setup complete: CUDA $CUDA_SERIES, $PYBIN"
echo "Authenticate optional services separately: popcorn-cli via the popcorn-login skill; Modal via $VENV/bin/modal setup."
