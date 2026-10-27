#!/usr/bin/env bash
# Produce the F16 GGUFs the llama.cpp LongRoPE logits rows compare against
# (crates/poot-llm/src/driver/llamacpp_longrope_tests.rs): llama.cpp's own converter over the same safetensors,
# so the reference is an unquantized f16 path and the rows test the rope tables, not quantization noise.
#
#   scripts/make-f16-ggufs.sh
#
# Reads $POOT_MODELS_DIR/phi-4-mini and $POOT_MODELS_DIR/phi-3.5-mini (HF checkpoints) and writes
#   $POOT_MODELS_DIR/phi-4-mini-gguf/Phi-4-mini-instruct-F16.gguf
#   $POOT_MODELS_DIR/phi-3.5-mini-gguf/Phi-3.5-mini-instruct-F16.gguf
# unless they exist. About 7.7 GB each.
#
# Environment:
#   POOT_MODELS_DIR   the models directory (default $HOME/models)
#   LLAMA_CPP_DIR     a llama.cpp checkout that holds convert_hf_to_gguf.py and gguf-py (default $HOME/refs/llama.cpp)
#   CONVERT_PYTHON    a python with torch (cpu is enough), numpy, transformers, sentencepiece, safetensors,
#                     protobuf and gguf installed (`pip install -e $LLAMA_CPP_DIR/gguf-py`); default python3.
#                     On NixOS put libstdc++ on LD_LIBRARY_PATH and unset PYTHONPATH for a pip-installed torch.
set -euo pipefail

models=${POOT_MODELS_DIR:-$HOME/models}
llama=${LLAMA_CPP_DIR:-$HOME/refs/llama.cpp}
python=${CONVERT_PYTHON:-python3}

convert() {
  local checkpoint=$1 outdir=$2 name=$3
  local out=$models/$outdir/$name
  if [ -e "$out" ]; then
    echo "have $out"
    return
  fi
  [ -d "$models/$checkpoint" ] || { echo "missing checkpoint $models/$checkpoint" >&2; exit 1; }
  mkdir -p "$models/$outdir"
  HF_HUB_OFFLINE=1 "$python" "$llama/convert_hf_to_gguf.py" "$models/$checkpoint" --outfile "$out" --outtype f16
}

convert phi-4-mini phi-4-mini-gguf Phi-4-mini-instruct-F16.gguf
convert phi-3.5-mini phi-3.5-mini-gguf Phi-3.5-mini-instruct-F16.gguf
