#!/usr/bin/env bash
# Image smoke check (run at the end of the Dockerfile build, no GPU): the CUDA 13 toolkit works and the
# binaries built against the 12.6 base still resolve every shared library. libcuda/libnvidia come from the
# pod's driver at runtime, so they are the only libraries allowed to be unresolved here.
set -euo pipefail
nvcc --version | tail -2
[ "$(nvcc --version | grep -o 'release [0-9]*' | head -1)" = "release 13" ] || { echo "nvcc is not CUDA 13" >&2; exit 1; }
for bin in "$LLAMA_BENCH_BIN" "$LLAMA_SERVER_BIN" "$POOT_BENCH_CANDLE_BIN"; do
  echo "== ldd $bin"
  out=$(ldd "$bin")
  echo "$out"
  missing=$(echo "$out" | grep 'not found' | grep -v -E 'libcuda\.so|libnvidia' || true)
  [ -z "$missing" ] || { echo "unresolved libraries in $bin:" >&2; echo "$missing" >&2; exit 1; }
done
