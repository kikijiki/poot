#!/usr/bin/env bash
# vLLM baseline env. Pinned to 0.22.1 (V1 engine). Run on the pod. No CUDA-12.4 wheel exists; the
# cu128/cu129 wheel runs on a 12.4 driver via minor-version compat (drop to cu128 if the driver is too
# old). vLLM brings its own torch - don't pre-install a conflicting one. See ../runners/vllm/NOTES.md.
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv pip install --system "vllm==0.22.1" --torch-backend=auto || \
    uv pip install --system "vllm==0.22.1" --extra-index-url https://download.pytorch.org/whl/cu128
else
  python3 -m pip install "vllm==0.22.1" --extra-index-url https://download.pytorch.org/whl/cu129 || \
    python3 -m pip install "vllm==0.22.1" --extra-index-url https://download.pytorch.org/whl/cu128
fi

# accelerate is needed if this venv also runs the transformers runner (device_map). pillow for VLM.
if command -v uv >/dev/null 2>&1; then uv pip install --system accelerate pillow; else python3 -m pip install accelerate pillow; fi

python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -m pip freeze | grep -E "vllm|torch|flashinfer|xformers" > "$(dirname "$0")/../runners/vllm/freeze.txt" || true
echo "vllm env ready -> runners/vllm/freeze.txt"
echo "NOTE: gpt-oss MXFP4 needs Hopper/Blackwell for the native kernel; on Ampere/Ada it is not accelerated."
echo "NOTE: without nvcc (e.g. a runtime-only CUDA image), the FlashInfer sampling kernel's runtime JIT dies"
echo "with 'nvcc: not found' on the first engine start. runners/vllm/run.py sets VLLM_USE_FLASHINFER_SAMPLER=0;"
echo "set it yourself if you invoke vllm directly. See ../runners/vllm/NOTES.md."
