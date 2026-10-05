#!/usr/bin/env bash
# candle runner: needs nvcc (compiles .cu kernels at build time) + rust. Run on the pod (-devel image).
# Set CUDA_COMPUTE_CAP to the pod GPU (Ampere 80/86, Ada 89, Hopper 90) or the build guesses wrong.
# Pinned to candle 0.10.2 (see ../runners/candle/Cargo.toml + NOTES.md).
set -euo pipefail
: "${CUDA_COMPUTE_CAP:?set CUDA_COMPUTE_CAP to the pod GPU arch, e.g. 86 (3090) / 89 (4090) / 90 (H100)}"

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc required (use a CUDA -devel image)"; exit 1; }

cd "$(dirname "$0")/../runners/candle"
echo "== build bench-candle-runner (CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP) =="
cargo build --release --features cuda

BIN="$(pwd)/target/release/bench-candle-runner"
echo "built: $BIN"
echo "export POOT_BENCH_CANDLE_BIN=$BIN   # the harness picks it up"
