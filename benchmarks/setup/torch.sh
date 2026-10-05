#!/usr/bin/env bash
# transformers v5 baseline env (+ a separate AWQ venv, since autoawq pins transformers 4.47.1).
# Run on the pod. Pins from ../runners/torch/NOTES.md. Do not reinstall torch if the base image already
# ships a CUDA-12.x build; that breaks fresh pods.
set -euo pipefail
PY="${POOT_BENCH_PYTHON:-python3}"

echo "== main env (transformers v5) =="
$PY -m pip install --upgrade pip
# Pin torch only if the base image's torch is not CUDA-12.x compatible (uncomment to force):
# $PY -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
$PY -m pip install \
  "transformers==5.0.0" "accelerate==1.10.1" "optimum==1.24.0" \
  "safetensors==0.4.5" "tokenizers==0.21.0" "pillow==11.0.0" "huggingface-hub==0.27.0" \
  "bitsandbytes==0.46.1"
$PY -m pip install "gptqmodel==5.8.0" --no-build-isolation || \
  echo "WARN: gptqmodel install failed (GPTQ cells will error); continuing"
$PY -m pip check || true
$PY -m pip freeze > "$(dirname "$0")/../runners/torch/freeze-main.txt"
echo "froze main env -> runners/torch/freeze-main.txt"

echo "== AWQ venv (isolated; autoawq downgrades transformers to 4.47.1) =="
$PY -m venv /opt/awq-venv
/opt/awq-venv/bin/pip install --upgrade pip
/opt/awq-venv/bin/pip install "autoawq==0.2.9" "transformers==4.47.1" "accelerate==1.2.1" safetensors pillow
/opt/awq-venv/bin/pip freeze > "$(dirname "$0")/../runners/torch/freeze-awq.txt"
echo "AWQ venv at /opt/awq-venv (run AWQ cells with POOT_BENCH_PYTHON=/opt/awq-venv/bin/python)"
