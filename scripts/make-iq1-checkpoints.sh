#!/usr/bin/env bash
# Produce the checkpoints the IQ1 llama.cpp rows compare on (crates/poot-llm/src/driver/llamacpp_iq1_tests.rs): the
# IQ1_S and IQ1_M GGUFs of Qwen2.5-0.5B-Instruct, and an f16 copy of each (llama.cpp's own `to_float` widens
# every tensor, so the copy is llama.cpp's dequantization of the IQ1 model and the rows can tell a decode
# difference from llama.cpp's activation quantization).
#
#   scripts/make-iq1-checkpoints.sh
#
# Writes $POOT_MODELS_DIR/iq1/Qwen2.5-0.5B-Instruct-{IQ1_S,IQ1_M}.gguf (about 316 MB each) and
# $POOT_MODELS_DIR/iq1/f16/Qwen2.5-0.5B-Instruct-{IQ1_S,IQ1_M}-as-F16.gguf (about 1 GB each) unless they exist.
# Needs curl, sha256sum and llama-quantize on PATH. The downloads are untrusted data: their sha256 is checked
# against the one the rows were measured on.
set -euo pipefail

models=${POOT_MODELS_DIR:-$HOME/models}
base=https://huggingface.co/ThomasBaruzier/Qwen2.5-0.5B-Instruct-GGUF/resolve/main
mkdir -p "$models/iq1/f16"

fetch() {
  local quant=$1 sha=$2
  local file=$models/iq1/Qwen2.5-0.5B-Instruct-$quant.gguf
  [ -e "$file" ] || curl -fL -o "$file" "$base/Qwen2.5-0.5B-Instruct-$quant.gguf"
  echo "$sha  $file" | sha256sum -c -
  local copy=$models/iq1/f16/Qwen2.5-0.5B-Instruct-$quant-as-F16.gguf
  [ -e "$copy" ] || llama-quantize --allow-requantize "$file" "$copy" F16
}

fetch IQ1_S 8a365459e5d2631aaa0add52fe7557002cb68c7b9902cb626d5f2cec0f799091
fetch IQ1_M 4b7f1938adcdd2c3ac3995f3b5d099d8a93924cc24a332dddf01b8dbde1016bd
