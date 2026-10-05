#!/usr/bin/env bash
# llama.cpp baseline: clone a pinned tag + build with CUDA (needs nvcc -> -devel image). Run on the pod.
# Set CUDA arch list to the pod GPU. GGUF-only: convert/download GGUFs separately (see NOTES.md). Pinned
# to tag b9601 (date-stamped rolling tags, not semver).
set -euo pipefail
LLAMACPP_TAG="${LLAMACPP_TAG:-b9601}"
ARCHES="${CMAKE_CUDA_ARCHITECTURES:-80;86;89;90}"
DEST="${LLAMACPP_DIR:-/opt/llama.cpp}"

command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc required (use a CUDA -devel image)"; exit 1; }

if [ ! -d "$DEST" ]; then
  git clone --depth 1 --branch "$LLAMACPP_TAG" https://github.com/ggml-org/llama.cpp.git "$DEST"
fi
cd "$DEST"
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHES"
cmake --build build --config Release -j"$(nproc)"

echo "built: $DEST/build/bin/{llama-cli,llama-bench,llama-mtmd-cli,llama-quantize}"
echo "export LLAMA_BENCH_BIN=$DEST/build/bin/llama-bench   # the runner uses llama-bench (NOT llama-cli)"
echo "Convert a GGUF:  python $DEST/convert_hf_to_gguf.py <hf_dir> --outfile m-f16.gguf --outtype f16"
echo "Quantize 4-bit:  $DEST/build/bin/llama-quantize m-f16.gguf m-Q4_K_M.gguf Q4_K_M"
