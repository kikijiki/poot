#!/usr/bin/env bash
# On-pod decode-curve sweep: run the decode-curve scenario for all engines on this pod's GPU and write one
# combined snapshot. Assumes the pre-baked image (engines under /opt/*-venv, /opt/llama.cpp, the candle
# binary) or that setup/*.sh has been run, and that the fresh poot binary and the model are present.
#
# Each baseline lives in its own venv (vLLM pins its own torch), so the harness routes each engine via its
# bin_env (runners.toml). This script points those at the image's baked paths; a single `bench run` then
# collects all 5 engines into one results.jsonl. poot runs on the backend POOT_BENCH_BACKEND names (default ptx,
# the pod's CUDA/PTX backend); the runner reports the backend and the commit it was built from on every row.
#
# Usage (on the pod, from /opt/benchmarks or a refreshed suite checkout):
#   MODELS_DIR=/root/models POOT_BENCH_POOT_BIN=/root/poot-bench-runner ./run-sweep.sh --model qwen2.5-0.5b
# Extra args after the model pass through to `bench run` (e.g. --run-id, --results-dir, --skip-unsupported,
# --repeats N: N run directories of the same matrix, the input `bench compare` wants five of per side).
set -euo pipefail

export MODELS_DIR="${MODELS_DIR:-/root/models}"
# xl-tier models (8B+) can push one decode-curve cell (all ISL points, one runner process) well past the
# observer's 5400s default; poot's naive prefill is the long pole (README, "poot prefill is naive and
# serial"). A kill loses every ISL point in the cell, since the observer only parses the runner's final
# stdout JSON, so raise the per-cell timeout.
export POOT_BENCH_CELL_TIMEOUT_S="${POOT_BENCH_CELL_TIMEOUT_S:-10800}"
export POOT_BENCH_POOT_BIN="${POOT_BENCH_POOT_BIN:-/root/poot-bench-runner}"
export POOT_BENCH_CANDLE_BIN="${POOT_BENCH_CANDLE_BIN:-/opt/benchmarks/runners/candle/target/release/bench-candle-runner}"
export LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-/opt/llama.cpp/build/bin/llama-bench}"
export POOT_BENCH_TORCH_PYTHON="${POOT_BENCH_TORCH_PYTHON:-/opt/tf-venv/bin/python}"
export POOT_BENCH_VLLM_PYTHON="${POOT_BENCH_VLLM_PYTHON:-/opt/vllm-venv/bin/python}"
export POOT_BENCH_LLAMACPP_PYTHON="${POOT_BENCH_LLAMACPP_PYTHON:-/opt/tf-venv/bin/python}"

cd "$(dirname "$0")"
echo "== decode-curve sweep: models=$MODELS_DIR poot_bin=$POOT_BENCH_POOT_BIN =="
nvidia-smi --query-gpu=name,uuid,power.limit --format=csv,noheader || true
exec uv run bench run --scenario decode-curve --models-dir "$MODELS_DIR" "$@"
