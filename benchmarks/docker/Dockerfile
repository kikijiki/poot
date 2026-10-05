# Pre-baked poot benchmark image (spec 112/115/117). Multi-stage to stay lean (18.2 GB on disk):
#   1. compiles llama.cpp + the candle runner in a throwaway `-devel` builder stage (nvcc lives only here),
#   2. ships only the built artifacts into a slim `nvidia/cuda:...-cudnn-runtime` final base (2.14 GB),
#   3. reimplements the minimal PUBLIC_KEY -> sshd `/start.sh` the orchestrator depends on.
# The torch venvs (vLLM / transformers) bundle their own CUDA libs; only candle + llama.cpp link the
# system cublas/cudart, which the cudnn-runtime base carries. Pushed to GHCR, pulled by `create-pod` via
# imageName.
#
# Not baked: the model weights (fetched on the pod, or read from an attached network volume) and the poot
# binary (scp'd fresh per run). Build context = the `benchmarks/` dir.

# --- builder: full CUDA toolkit, discarded after the compiles
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH \
    CUDA_COMPUTE_CAP=86

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config curl git cmake libssl-dev ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# rust toolchain (candle); stays in the builder.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

# llama.cpp (CUDA, sm_86). ggml-cuda links the driver stub, so libggml-cuda.so gets NEEDED=libcuda.so.1,
# but the toolkit's stubs/ dir only has libcuda.so. Symlink it to the .1 soname so the link resolves; at
# runtime the pod injects the real driver.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN git clone --depth 1 --branch b9601 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CUDA_ARCHITECTURES=86 \
       -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build -j"$(nproc)" --target llama-bench llama-server llama-cli

# candle runner (compiles CUDA kernels for sm_86). Needs the suite source.
COPY . /opt/benchmarks
RUN cd /opt/benchmarks/runners/candle && cargo build --release --features cuda

# --- runtime: slim cuda runtime base
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:/root/.local/bin:$PATH \
    CUDA_COMPUTE_CAP=86

# openssh-server: the orchestrator drives the pod over ssh/scp. patchelf: the poot binary is patched at
# runtime. tmux: the sweep detaches into a tmux session. libvulkan1/vulkan-tools: RunPod mounts the NVIDIA
# ICD + driver libraries, but wgpu needs the userspace Vulkan loader and a diagnostic binary in the image.
# libxext6/libx11-6: the NVIDIA ICD (libGLX_nvidia.so.0) needs libXext.so.6, which needs libX11.so.6;
# without them vkCreateInstance fails with ERROR_INCOMPATIBLE_DRIVER (update 1090). gcc/g++/ninja-build:
# vLLM's torch.compile (inductor + triton) JIT-compiles a C/CUDA helper at engine init, and the EngineCore
# dies with "Failed to find C compiler" without them. Python 3.11 is not installed system-wide; uv fetches
# an interpreter for the harness venv and each engine venv.
RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server patchelf tmux libvulkan1 vulkan-tools libxext6 libx11-6 \
      gcc g++ ninja-build curl git ca-certificates gnupg \
    && mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && rm -rf /var/lib/apt/lists/*

# LLVM 22 llc (NVPTX): poot's runtime path emits LLVM IR and shells `llc` to JIT each kernel to PTX (the
# llc version the engine is built against). The bench warmup amortizes the one-time JIT. From apt.llvm.org.
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" \
       > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends llvm-22 \
    && ln -sf /usr/lib/llvm-22/bin/llc /usr/local/bin/llc \
    && rm -rf /var/lib/apt/lists/*
ENV POOT_LLC=/usr/lib/llvm-22/bin/llc

# uv (harness + per-engine venvs) + the `hf` CLI for model downloads (the slim base has no system pip;
# `uv tool` installs hf into /root/.local/bin).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN /root/.local/bin/uv tool install huggingface_hub

# Start script: inject the create-pod PUBLIC_KEY env into authorized_keys, then run sshd in the foreground.
RUN printf '%s\n' \
      '#!/bin/bash' \
      'set -e' \
      'if [ -n "$PUBLIC_KEY" ]; then' \
      '  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys' \
      '  chmod 600 /root/.ssh/authorized_keys' \
      'fi' \
      'ssh-keygen -A' \
      'exec /usr/sbin/sshd -D -e' \
    > /start.sh && chmod +x /start.sh

# Built artifacts only. Copy llama.cpp's whole build/bin (binaries and libggml*.so; found via $ORIGIN
# rpath) plus the HF->GGUF converter the llamacpp runner invokes on demand. b9601 split the converter into a
# `conversion/` package that convert_hf_to_gguf.py imports; without it the converter dies with
# ModuleNotFoundError: No module named 'conversion'.
COPY --from=builder /opt/llama.cpp/build/bin/ /opt/llama.cpp/build/bin/
COPY --from=builder /opt/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=builder /opt/llama.cpp/conversion/ /opt/llama.cpp/conversion/
COPY --from=builder /opt/llama.cpp/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=builder /opt/benchmarks/runners/candle/target/release/bench-candle-runner \
     /opt/benchmarks/runners/candle/target/release/bench-candle-runner

# The benchmark suite (setup scripts, harness, the candle runner source tree minus its build dir).
COPY . /opt/benchmarks
WORKDIR /opt/benchmarks

# Harness venv (pynvml, psutil).
RUN uv sync

# vLLM 0.22.1 in its own venv (brings its own torch). --torch-backend=cu130 matches its compiled _C ext
# (built for CUDA 13) and avoids the Intel-XPU wheel uv picks on a GPU-less host; runs on the pod's 580 driver.
RUN uv venv /opt/vllm-venv --python 3.11 \
    && uv pip install --python /opt/vllm-venv/bin/python "vllm==0.22.1" --torch-backend=cu130 \
    && uv pip install --python /opt/vllm-venv/bin/python accelerate pillow

# vLLM's V1 EngineCore auto-selects FlashInfer for top-k/top-p sampling when importable, and FlashInfer
# JIT-compiles its CUDA kernel with nvcc during profile_run, so it fires on every vLLM cell. This runtime
# stage has no nvcc, so the JIT died with "nvcc: not found", surfaced only as a generic "Engine core
# initialization failed" (seen 2026-08-20 on an RTX 3090 pod). Force the native PyTorch sampler; the suite
# samples greedily. runners/vllm/run.py also sets this for on-pod installs (setup/vllm.sh).
ENV VLLM_USE_FLASHINFER_SAMPLER=0

# transformers v5 in its own venv (separate from vLLM's pinned deps; also runs the GGUF converter).
# --torch-backend=cu124 is required: without it uv resolves the CPU wheel on the GPU-less build host and the
# transformers cell errors with "no CUDA device". sentencepiece/protobuf (some tokenizers) and pyyaml/tqdm
# (gguf-py) let this venv run convert_hf_to_gguf.py for the llama.cpp runner.
RUN uv venv /opt/tf-venv --python 3.11 \
    && uv pip install --python /opt/tf-venv/bin/python --torch-backend=cu124 "transformers==5.0.0" accelerate safetensors pillow torch sentencepiece protobuf pyyaml tqdm

# Binary paths the harness reads via env (the poot binary is set per run).
ENV POOT_BENCH_CANDLE_BIN=/opt/benchmarks/runners/candle/target/release/bench-candle-runner \
    LLAMA_BENCH_BIN=/opt/llama.cpp/build/bin/llama-bench \
    LLAMA_SERVER_BIN=/opt/llama.cpp/build/bin/llama-server

# The nvidia base ENTRYPOINT execs CMD.
CMD ["/start.sh"]
