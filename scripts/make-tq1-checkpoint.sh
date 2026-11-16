#!/usr/bin/env bash
# Produce the checkpoints the TQ1_0 llama.cpp rows compare on (crates/poot-llm/src/driver/llamacpp_tq1_tests.rs): the
# TQ1_0 GGUF of TriLM 1.1B (a ternary-trained model, so TQ1_0 holds its weights without a lossy re-quantization),
# and an f16 copy of it (llama.cpp's own `to_float` widens every tensor, so the copy is llama.cpp's dequantization
# of the TQ1_0 model and the rows can tell a decode difference from llama.cpp's activation quantization). The
# Q4_0 file of the same model and its f16 copy are the rows' control: a different quantization of the weights.
#
#   scripts/make-tq1-checkpoint.sh
#
# Writes $POOT_MODELS_DIR/tq1/TriLM_1.1B_Unpacked-{tq1_0,q4_0}.gguf (316 and 619 MB) and
# $POOT_MODELS_DIR/tq1/f16/TriLM_1.1B_Unpacked-{tq1_0,q4_0}-as-F16.gguf (2.2 GB each) unless they exist.
# Needs curl, sha256sum and llama-quantize on PATH. The downloads are untrusted data: their sha256 is checked
# against the one the rows were measured on.
set -euo pipefail

models=${POOT_MODELS_DIR:-$HOME/models}
base=https://huggingface.co/Mungert/TriLM_1.1B_Unpacked-GGUF/resolve/main
mkdir -p "$models/tq1/f16"

fetch() {
  local quant=$1 sha=$2
  local name=TriLM_1.1B_Unpacked-$quant
  local file=$models/tq1/$name.gguf
  [ -e "$file" ] || curl -fL -o "$file" "$base/$name.gguf"
  echo "$sha  $file" | sha256sum -c -
  local copy=$models/tq1/f16/$name-as-F16.gguf
  [ -e "$copy" ] || llama-quantize --allow-requantize "$file" "$copy" F16
}

fetch tq1_0 33e968bc4a632547694798eb2fb3a68c645d5e9969f26531d91a7b9628dc7ec8
fetch q4_0 90c64e240b42847c71b325433f86a42f4aecae2273a64c7a05f98c2a49a36342
