---
id: feature-matrix
title: Feature matrix
sidebar_position: 1
---

# Feature matrix

The `poot-llm` library selects a GPU runtime with `--backend`. **Three production backends** run the
same graph IR today: **wgpu** (Vulkan/SPIR-V, default), **rocm** (AMD raw HSA), and **ptx** (NVIDIA).
They are not at parity. A fourth backend, raw Vulkan, has its own section below and is not a column in
these tables.
This page answers "does X work on backend Y?". Each row is backed by a test or a code path checked when
the page was written.

NPU offload was removed; it is not listed as a column here. See
[NPU offload](../architecture/npu.mdx).

_Last reviewed: September 2026. Synced with backend capability claims on this site._

For lower-level kernel/dispatch capability (elementwise, GEMV, flash attention, capture/replay, ...), see
the [Backends](../architecture/backends.mdx#capability-comparison) architecture page, this page is the
serving-level view: what `poot-serve` and the `poot-llm` library provide, with generation serving marked
as being restored.

Legend: colored badges use the same vocabulary on this page and on
[Backends: capability comparison](../architecture/backends.mdx#capability-comparison).
**Yes** = the indicated path has test coverage; footnotes distinguish format tests from model receipts. **Partial** = works, with scope limits noted.
**No** = not implemented for that backend. **N/A** = the concept does not apply to that backend.

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today `poot-serve` serves encoder embeddings and reranking. `/v1/completions` and `/v1/chat/completions` return a typed `503`, and `/health` reports `"engine_loaded": false`. See [Serving](../serve/index.md).

## Models

|                                                                                                  | wgpu                                                                             | ROCm                                                                                                                      | PTX                                                                                                                       |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Dense decoders (Qwen2/3, Llama, Mistral, Gemma 2/3, Granite, OLMo 2, Phi-3/4, SmolLM3, BLOOM, MPT)[^dense-driver] | <span class="matrix-status matrix-status--yes">Yes</span>                        | <span class="matrix-status matrix-status--yes">Yes</span>                                                                 | <span class="matrix-status matrix-status--yes">Yes</span>                                                                 |
| Mixture-of-experts (Qwen3-MoE, GraniteMoE, Mixtral, OLMoE, gpt-oss)                              | <span class="matrix-status matrix-status--yes">Yes</span>                        | <span class="matrix-status matrix-status--yes">Yes</span>[^moe-rocm-ptx]                                                  | <span class="matrix-status matrix-status--yes">Yes</span>[^moe-rocm-ptx]                                                  |
| DeepSeek-V2/V3 (multi-head latent attention)                                                     | <span class="matrix-status matrix-status--yes">Yes</span>                        | <span class="matrix-status matrix-status--yes">Yes</span>                                                                 | <span class="matrix-status matrix-status--yes">Yes</span>                                                                 |
| Qwen3-Next (hybrid linear-attention + MoE)                                                       | <span class="matrix-status matrix-status--no">No</span> (being restored)[^not-loadable]           | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   |
| Gemma 4 (dense and MoE)                                                                          | <span class="matrix-status matrix-status--no">No</span> (being restored)[^not-loadable]           | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   |
| BERT-class embeddings and cross-encoder reranking                                                | <span class="matrix-status matrix-status--yes">Yes</span>[^encoder-cpu]          | <span class="matrix-status matrix-status--yes">Yes</span>[^encoder-cpu]                                                   | <span class="matrix-status matrix-status--yes">Yes</span>[^encoder-cpu]                                                   |
| SmolVLM / idefics3 (vision-language)                                                             | <span class="matrix-status matrix-status--yes">Yes</span>                        | <span class="matrix-status matrix-status--partial">Partial</span> (wgpu-verified; ROCm/PTX hardware verification pending) | <span class="matrix-status matrix-status--partial">Partial</span> (wgpu-verified; ROCm/PTX hardware verification pending) |
| DeepSeek-V3.2 Sparse Attention (DSA)                                                             | <span class="matrix-status matrix-status--partial">Partial</span>[^synth-family] | <span class="matrix-status matrix-status--no">No</span>                                                                   | <span class="matrix-status matrix-status--no">No</span>                                                                   |
| Nemotron-H (hybrid Mamba2 + attention)                                                           | <span class="matrix-status matrix-status--partial">Partial</span>[^synth-family] | <span class="matrix-status matrix-status--no">No</span>                                                                   | <span class="matrix-status matrix-status--no">No</span>                                                                   |
| Qwen3.8 QSA                                                                                      | <span class="matrix-status matrix-status--no">No</span> (being restored)[^not-loadable]           | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   |
| DeepSeek-V4-Flash-0731 (routed MoE)                                                              | <span class="matrix-status matrix-status--no">No</span> (being restored)[^not-loadable]           | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   |
| MiniMax-M2.5 (packed-expert MoE)                                                                 | <span class="matrix-status matrix-status--no">No</span> (being restored)[^not-loadable]           | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   |
| GLM-5.3-Flash                                                                                    | <span class="matrix-status matrix-status--no">No</span> (being restored)[^not-loadable]           | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   |
| Qwen3.8-27B-FP8 (`qwen3_5` text tower)                                                           | <span class="matrix-status matrix-status--no">No</span> (being restored)[^not-loadable]           | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   | <span class="matrix-status matrix-status--no">No</span> (being restored)                                                                   |

[^dense-driver]:
    The dense families load through the model registry (`ModelHandle::load`) and generate through the
    driver; the other families in this table load through `Runner::load`. A Mixtral GGUF names the dense
    `llama` architecture and does not load for now.

[^encoder-cpu]:
    Embedding and cross-encoder reranking requests always run on the CPU evaluator, not on the selected
    GPU backend. The per-backend cells mean the `poot-serve` route is available regardless of backend,
    not that the encoder runs on that backend.

[^moe-rocm-ptx]:
    Mixtral and Qwen3-MoE have ROCm and PTX library decode tests. gpt-oss and OLMoE
    are traced, loaded, and CPU-oracle verified on all backends, but their GPU decode coverage is thinner than
    Mixtral/Qwen3-MoE's, test the specific architecture + backend combination you plan to run. Serving is
    being restored (see the caution above).

[^synth-family]:
    Bounded synthetic safetensors fixtures load through `Runner::load` and complete cached
    single-sequence wgpu generation against the CPU oracle (logits, recurrent/cache tensors, greedy tokens).
    This is not a real public-checkpoint claim and not HTTP/`poot-serve` continuous-batching support.

[^not-loadable]:
    `Runner::load` refuses this family with an unsupported-model error naming its architecture string. It is
    not available yet (poot is being refactored; planned to return); until then no backend runs it.

## Expert pooling

Expert pooling (a fixed-capacity expert pool per MoE layer with a runtime residency manager) is not available yet (poot is being refactored; planned to return).
The MoE models above run through their ordinary, unpooled decode paths, with every expert's
weights resident.

The former pooled-decode rows for DeepSeek-V3 and Gemma 4 MoE are removed from the Models table. DeepSeek-V3
is covered by the DeepSeek-V2/V3 row, which is the unpooled decode path. Gemma 4 and Qwen3-Next have a row that
says they are being restored.

[^serve-removed]:
    Generation serving is not available yet (poot is being refactored; planned to return). `/v1/completions` and `/v1/chat/completions` return a typed `503`
    naming the missing engine. See [Serving](../serve/index.md).

## Quantization

|                                                 | wgpu                                                                                                 | ROCm                                                                                              | PTX                                                                                                         |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| GGUF Q4_0/Q4_1/Q5_0/Q5_1/Q8_0                   | <span class="matrix-status matrix-status--yes">Yes</span>                                            | <span class="matrix-status matrix-status--yes">Yes</span>                                         | <span class="matrix-status matrix-status--yes">Yes</span>                                                   |
| GGUF K-quants (Q2_K-Q6_K, IQ4_NL, IQ4_XS)       | <span class="matrix-status matrix-status--yes">Yes</span>                                            | <span class="matrix-status matrix-status--yes">Yes</span>                                         | <span class="matrix-status matrix-status--yes">Yes</span>                                                   |
| GGUF MXFP4 (gpt-oss routed experts)[^mxfp4]     | <span class="matrix-status matrix-status--yes">Yes</span>                                            | <span class="matrix-status matrix-status--yes">Yes</span>                                         | <span class="matrix-status matrix-status--partial">Partial</span> (masked decode; synthetic packed prefill) |
| Safetensors GPTQ (4-bit, symmetric)             | <span class="matrix-status matrix-status--yes">Yes</span>                                            | <span class="matrix-status matrix-status--yes">Yes</span>                                         | <span class="matrix-status matrix-status--yes">Yes</span>                                                   |
| Safetensors AWQ (4-bit, GEMM)                   | <span class="matrix-status matrix-status--yes">Yes</span>                                            | <span class="matrix-status matrix-status--yes">Yes</span>                                         | <span class="matrix-status matrix-status--yes">Yes</span>                                                   |
| Safetensors FP8 (`compressed-tensors`)          | <span class="matrix-status matrix-status--partial">Partial</span> (packed Qwen projections[^fp8])    | <span class="matrix-status matrix-status--partial">Partial</span> (packed Qwen projections[^fp8]) | <span class="matrix-status matrix-status--partial">Partial</span> (packed Qwen projections[^fp8])           |
| Packed projection storage[^packed-memory] | <span class="matrix-status matrix-status--yes">Yes</span> (dense packed projections) | <span class="matrix-status matrix-status--yes">Yes</span>                                         | <span class="matrix-status matrix-status--yes">Yes</span>                                                   |

[^mxfp4]:
    MXFP4 weights remain packed at load time. The current routed-expert graph decodes and stacks expert
    tables before `IndexedMatMul`; it is not a direct packed indexed contraction. Format-level kernel
    coverage does not imply real-checkpoint coverage for every gpt-oss execution path.

[^fp8]:
    The supported Qwen `compressed-tensors` per-channel E4M3 projection path loads packed and uses the
    shared generated lowering. DeepSeek-V3.2 block-FP8 loading is also covered by synthetic fixtures;
    it is not a real-checkpoint generation claim. See [Using quantized checkpoints](../serve/quantized-checkpoints.md).

[^packed-memory]:
    Dense packed projections and packed embedding gathers avoid full weight materialization. Packed MoE
    paths currently materialize expert tables during execution. Dense BF16/F16 weights are kept as
    stored words with no host f32 copy. Packed checkpoint size is not a bound on total peak memory.

## Serving features

|                                                                                                      | wgpu                                                                    | ROCm                                                                                  | PTX                                                                     |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| OpenAI-compatible HTTP server (`poot-serve`)                                                         | <span class="matrix-status matrix-status--yes">Yes</span>               | <span class="matrix-status matrix-status--yes">Yes</span>                             | <span class="matrix-status matrix-status--yes">Yes</span>               |
| Continuous (in-flight) batching                                                                      | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| Paged KV cache                                                                                       | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| Prefix caching (reuse across requests)                                                               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| Fast one-shot prefill for the batched path                                                           | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| KV preemption / swap on eviction                                                                     | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| Speculative decoding, single-sequence (prompt-lookup + draft-model, greedy + sampled)                | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| Speculative decoding, inside continuous batching (prompt-lookup and draft-model, greedy and sampled) | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| On-device sampling: greedy                                                                           | <span class="matrix-status matrix-status--yes">Yes</span>[^device-sample] | <span class="matrix-status matrix-status--yes">Yes</span>[^device-sample]             | <span class="matrix-status matrix-status--yes">Yes</span>[^device-sample] |
| On-device sampling: temperature/top-k/top-p/min-p                                                    | <span class="matrix-status matrix-status--yes">Yes</span>[^device-sample] | <span class="matrix-status matrix-status--yes">Yes</span>[^device-sample]             | <span class="matrix-status matrix-status--yes">Yes</span>[^device-sample] |
| LoRA adapter serving (hot-load, per-request selection)                                               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^lora-scope]    | <span class="matrix-status matrix-status--no">No</span> (being restored)[^lora-scope]                  | <span class="matrix-status matrix-status--no">No</span> (being restored)[^lora-scope]    |
| Tensor parallelism (multi-GPU)                                                                       | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| Structured/guided decoding (regex, JSON schema, grammar, choice, forced tool calls)                  | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed]               | <span class="matrix-status matrix-status--no">No</span> (being restored)[^serve-removed] |
| Embeddings (`/v1/embeddings`) and reranking (`/v1/rerank`)                                           | <span class="matrix-status matrix-status--yes">Yes</span>[^encoder-cpu] | <span class="matrix-status matrix-status--yes">Yes</span>[^encoder-cpu]               | <span class="matrix-status matrix-status--yes">Yes</span>[^encoder-cpu] |
| Prometheus metrics (`/metrics/prometheus`)                                                           | <span class="matrix-status matrix-status--yes">Yes</span>               | <span class="matrix-status matrix-status--yes">Yes</span>                             | <span class="matrix-status matrix-status--yes">Yes</span>               |

[^device-sample]:
    A `poot-llm` library capability (the driver, and the Runner for the families still on it), verified by
    test on real hardware for every backend: a sampling step is appended to the decode graph itself, and the host reads back only the chosen
    token id (and a non-finite flag), not the full logits, for greedy and plain-temperature (including
    top-k/top-p/min-p) requests. A request with a logit-bias/penalty, logprob recording, or a
    guided-decoding constraint still reads the full logits back and picks on the host, since those
    transforms are host-side. `poot-serve` does not call this path today (see the caution above); the capability lives in the library.

[^lora-scope]:
    LoRA adapter serving is not available yet (poot is being refactored; planned to return). Registration is refused. The adapter scope, when it returns, is
    dense qwen2/llama-shaped models and the seven attention and MLP projections (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`). See
    [Using LoRA adapters](../serve/lora-adapters.md).

## Precision and execution model

|                                                       | wgpu                                                                                  | ROCm                                                                                          | PTX                                                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Compute dtype                                         | f32                                                                                   | f32, opt-in bf16 tensor-core matmul                                                           | f32, opt-in bf16 tensor-core matmul                                                  |
| Tensor-core / matrix-unit matmul                      | <span class="matrix-status matrix-status--no">No</span>[^wgpu-coopmat]                | <span class="matrix-status matrix-status--yes">Yes</span> (RDNA WMMA, verified gfx1151)       | <span class="matrix-status matrix-status--yes">Yes</span> (NVIDIA WMMA)              |
| Execution model per token                             | Cached command-buffer re-encode + submit                                              | Raw HSA AQL packet record/replay                                                              | CUDA graph capture/replay                                                            |
| Megakernel (on-device token loop, no host round-trip) | <span class="matrix-status matrix-status--no">No</span> (no grid-wide sync primitive) | <span class="matrix-status matrix-status--no">No</span> (removed)[^megakernel-scope]          | <span class="matrix-status matrix-status--no">No</span> (removed)[^megakernel-scope] |
| Requires a vendor SDK / driver beyond Vulkan          | <span class="matrix-status matrix-status--no">No</span>                               | <span class="matrix-status matrix-status--yes">Yes</span> (ROCm/HSA userspace)                | <span class="matrix-status matrix-status--yes">Yes</span> (CUDA)                     |
| On by default                                         | <span class="matrix-status matrix-status--yes">Yes</span>                             | <span class="matrix-status matrix-status--no">No</span> (`--features rocm`, `--backend rocm`) | <span class="matrix-status matrix-status--no">No</span> (`--backend ptx`)            |

[^wgpu-coopmat]:
    A narrowly-scoped SPIR-V cooperative-matrix primitive exists and is hardware-verified
    (`K == 16` only, f32 output, weight operand only), but no model's matmul feeds it yet, so it is not on
    the hot path for any model today.

[^megakernel-scope]:
    The implementation was removed. The strategy is planned as a recorder behind the one compiled
    program on PTX and ROCm; see [The Megakernel](../architecture/megakernel.mdx).

## Raw Vulkan

Raw Vulkan (`--backend vulkan`) runs the same SPIR-V kernels as wgpu through `ash`, with no wgpu. It implements the
same executor contract as the other backends: a program is recorded once as native Vulkan command buffers and
replayed, so a step costs a submit and a fence wait. It is lower priority than PTX and ROCm and is verified on
one device family.

| Item                                    | Status                                                                                               |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Executor parity table                   | <span class="matrix-status matrix-status--yes">Yes</span> (the parity table and the backend-neutral coverage fixtures) |
| Dense decoder, greedy decode            | <span class="matrix-status matrix-status--yes">Yes</span> (the registry's qwen2 fixture generates the same tokens as wgpu) |
| Tensor-core matmul (cooperative matrix) | <span class="matrix-status matrix-status--yes">Yes</span> (needs `VK_KHR_cooperative_matrix` and subgroup-size control; verified RDNA3.5) |
| Device-time profiling                   | <span class="matrix-status matrix-status--partial">Partial</span> (library only: `VulkanDevice::new_with_timing`; the device span runs from the first dispatch's start to the last dispatch's end, so it includes the fence waits between submissions, and a replay of more than 4096 dispatches reports no time) |
| Multi-GPU / collectives                 | <span class="matrix-status matrix-status--no">No</span>                                            |
| Serving                                 | <span class="matrix-status matrix-status--no">No</span> (being restored, see the caution above) |
| Hardware verified                       | AMD Strix Halo iGPU under RADV only. NVIDIA and Intel Vulkan drivers are untested.                 |

The device reports its own limits (largest buffer, grid, workgroup memory, subgroup sizes and matrix hardware)
from the physical device, and the planner refuses what they rule out with a typed error instead of failing in the
driver.

## How to read this if you are choosing hardware

- **NVIDIA GPU:** PTX has the most mature single-GPU library path. Sampling picks on the host, same as
  every other backend. Serving is being restored (see the caution above).
- **AMD GPU:** ROCm is the primary backend on AMD hardware for single-sequence library decode. wgpu also
  runs on the same AMD GPU as a fully portable fallback.
- **Anything else (Intel iGPU/dGPU, other Vulkan-capable hardware):** wgpu is the supported choice. Raw Vulkan
  runs there in principle but is verified on AMD (RADV) only. Pick either through the `poot-llm` API;
  the `poot-serve` backend option is being restored.
- **NPU:** removed. See
  [Architecture: NPU Offload](../architecture/npu.mdx).

See [Picking a backend](../serve/choosing-a-backend.md) for the decision walkthrough.
