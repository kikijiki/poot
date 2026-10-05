# candle runner notes (research 2026-06-12, candle 0.10.2)

> Status: snapshot. Pins as of 2026-06; spot-checked 2026-09-19 against `benchmarks/docker/Dockerfile`
> (llama.cpp b9601, vLLM 0.22.1, transformers 5.0.0, candle 0.10.2 match). The image is the source of truth and
> is now Ubuntu 24.04 / CUDA 12.6.3, so CUDA 12.4 / Ubuntu 22.04 mentions below are stale.

Rust peer. Pin published crates `=0.10.2` (candle-core / candle-nn / candle-transformers all on the SAME
version) + `tokenizers =0.22.0` (onig) + `hf-hub =0.5.0` + `safetensors =0.7.0` + `half =2.5.0`. Enable GPU
with the `cuda` feature (`candle-core/cuda` + candle-nn/cuda + candle-transformers/cuda). CUDA device is a
runtime choice (`Device::new_cuda(0)`), not a feature.

## Build needs nvcc (compiles .cu kernels) - like llama.cpp, unlike poot/vLLM

candle's `cuda` feature compiles CUDA C++ kernels at build time (bindgen_cuda on 0.10.x). A driver-only pod
cannot build it; install the CUDA 12.4 toolkit (`--toolkit` only, driver already present). Set
`CUDA_COMPUTE_CAP` to the pod GPU (Ampere 80/86, Ada 89, Hopper 90) or the build guesses wrong. A pre-built
binary can run on a driver-only pod. Build on the pod, or build on a toolkit host + scp the binary.

## Generation loop

`model.forward(&input, start_pos) -> [b,1,vocab]` (already narrows to last pos + applies lm_head). KV cache
internal; pass full prompt at step 0, 1 token after, advance via `start_pos`. `model.clear_kv_cache()`
between prompts. Sampler: `LogitsProcessor::from_sampling(seed, Sampling::ArgMax)` for greedy,
`Sampling::TopK{k,temperature}` for top-k. Load: `VarBuilder::from_mmaped_safetensors(&paths, dtype,
&device)` (unsafe; handles sharded via the index.json file list).

## Timing / VRAM

TTFT = Instant after `sample()` at index 0 minus Instant before first `forward()`. Per-token = consecutive
post-sample Instants for index>=1. `sample()` reads logits to host (forces a sync) so the post-sample
Instant is accurate; still call `device.synchronize()` at start + end. candle has no peak-VRAM API; poll
NVML (`nvml-wrapper` crate, `device.memory_info().used`, ~50-100ms, keep max). Run one engine per pod.

## Precision / arch gaps (confirmed from source)

- Precision: f32 / f16 / bf16 (DType, bf16 is the GPU default). GGUF k-quants supported
  (`quantized::gguf_file` + `ModelWeights::from_gguf`). **NO GPTQ, NO AWQ** (grep finds nothing).
- Arches SHIPPED: qwen2, qwen3, llama, mistral, mixtral, qwen2*moe, qwen3_moe, deepseek2,
  granitemoehybrid (+ quantized*\* variants).
- **NOT shipped (mark unsupported in the manifest):**
  - **SmolVLM / Idefics3** - candle's `smol` dir is SmolLM3 (text), not SmolVLM. VLM options that DO exist:
    LLaVA, Moondream, PaliGemma, Pixtral; Qwen3-VL only on git-main (not 0.10.2).
  - **OLMoE** - not in the MoE list (only mixtral/qwen\*\_moe/deepseek2/granitemoehybrid).
  - **gpt-oss** - not present.
- VLMs do NOT share the generic text `forward(input, offset)` loop; each needs its own image-preprocess
  runner. Out of scope for the text runner.

Runner: a small Rust bin in this dir (the loop above) emitting the suite JSON line; VRAM is left to the
harness external sampler. The quant baseline for candle is GGUF only, matched on bit-width with a caveat (as
for llama.cpp).
