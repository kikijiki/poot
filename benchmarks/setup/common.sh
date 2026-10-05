#!/usr/bin/env bash
# Common pod prep for the poot benchmark suite (spec 112). Run once on a fresh CUDA-12.4 *-devel pod
# (candle + llama.cpp compile CUDA kernels with nvcc, so use a -devel image, not runtime/base).
# Safe to re-run. See ../README.md.
set -euo pipefail

echo "== apt basics =="
export DEBIAN_FRONTEND=noninteractive
apt-get update
# ninja-build is needed by vLLM's engine (it shells out to `ninja`); patchelf for the scp'd poot binary.
apt-get install -y build-essential pkg-config curl git cmake libssl-dev python3 python3-venv \
  ninja-build patchelf

echo "== uv (harness venv manager) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== harness venv =="
cd "$(dirname "$0")/.."
uv sync   # pynvml + psutil for the observer; report/compare need no deps

echo "== sanity =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
nvcc --version | grep -i release || echo "WARN: nvcc not found - candle/llama.cpp builds need a -devel image"
echo "common setup done. Now run the per-framework setup scripts you need (setup/<fw>.sh)."
