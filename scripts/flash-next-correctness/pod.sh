#!/usr/bin/env bash
# POOT-1333, on the pod (poot-orchestrator exec): fetch Flash-Next UD-Q2_K_XL, build llama.cpp for the pod's GPU,
# run run.sh (llama.cpp CPU and CUDA, poot on PTX), leave /root/out.tgz. Uploaded: fn.tar (this directory),
# poot-score (via --bin), hf-token. Everything here is execution; the logic is in run.sh and compare.py.
set -uo pipefail
export PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
work=/root/work out=/root/out
mkdir -p "$work" "$out" /root/.cache/huggingface
exec > >(tee "$out/pod.log") 2>&1
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

stamp "pod: $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader)"
free -g | head -2
nproc
df -h /root | tail -1
cp /root/hf-token /root/.cache/huggingface/token

tar -xf /root/fn.tar -C "$work"
repo=unsloth/Qwen3.8-Flash-Next-GGUF
models=/root/models/qwen3.8-flash-next-gguf
(
    command -v hf >/dev/null || uv tool install --quiet huggingface_hub
    hf download "$repo" --include 'UD-Q2_K_XL/*.gguf' --local-dir "$models"
    uv venv --quiet --python 3.12 /root/py
) >"$out/download.log" 2>&1 &
download=$!

arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
stamp "building llama.cpp v0.4.1 for sm_$arch"
(
    set -e
    git clone --depth 1 --branch v0.4.1 https://github.com/ggml-org/llama.cpp.git "$work/llama.cpp"
    cd "$work/llama.cpp"
    git rev-parse HEAD
    cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' -DLLAMA_CURL=OFF
    cmake --build build -j"$(nproc)" --target llama-perplexity llama-server
) >"$out/llama-build.log" 2>&1 || { stamp "llama.cpp build failed"; tail -30 "$out/llama-build.log"; exit 1; }
stamp "llama.cpp built: $(head -1 "$out/llama-build.log")"

wait "$download" || { stamp "download failed"; tail -20 "$out/download.log"; exit 1; }
stamp "model downloaded: $(du -sh "$models" | cut -f1)"
stamp "free memory: $(free -g | awk '/Mem/ {print $7 " GiB available"}')"

export MODEL=$models/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf
export OUT=$out/cases LLAMA_BIN=$work/llama.cpp/build/bin POOT_SCORE=/root/poot-score POOT_BACKEND=ptx
export LLAMA_GPU_FLAGS="-ngl 99" PYTHON=/root/py/bin/python POOT_LLC=/usr/lib/llvm-22/bin/llc
# Run 1 hung in llama.cpp v0.4.1's CPU decode of this model (a 48-token prompt, then 19 minutes at 1500% CPU with
# no token): the CPU greedy continuation is skipped here, and every step has a limit.
export CPU_GREEDY=0 STEP_TIMEOUT_PERPLEXITY=1500 STEP_TIMEOUT_POOT=1500 STEP_TIMEOUT_GREEDY=300
status=0
# The short window end to end first, its verdict on stdout before the long window starts: the acceptance (mean KL
# within llama.cpp's own CPU-vs-GPU noise, top-1 near llama.cpp's) is decided there.
bash "$work/flash-next-correctness/run.sh" all short || status=$?
stamp "short window: run.sh exit $status"
for f in report.json poot.json; do
    echo "=== short/$f"; cat "$OUT/short/$f" 2>/dev/null; echo
done
bash "$work/flash-next-correctness/run.sh" all long || status=$?
stamp "long window: run.sh exit $status"
for f in report.json; do
    echo "=== long/$f"; cat "$OUT/long/$f" 2>/dev/null; echo
done

# The reports and the small artifacts; the logits files (hundreds of MB) stay on the pod.
(cd "$out" && find cases \( -name '*.json' -o -name 'ids.txt' -o -name '*.log' \) -print0 | tar --null -czf /root/out.tgz -T - pod.log llama-build.log download.log)
exit $status
