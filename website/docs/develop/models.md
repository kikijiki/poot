---
id: models
title: Supported models
sidebar_position: 7
---

# Supported models

poot traces model forwards into a primitive graph and runs them on its own kernels. This page lists
families by architecture shape. It does not duplicate the full backend tables; use the [feature
matrix](../reference/feature-matrix.md) for verified wgpu, ROCm, PTX, and NPU coverage.

Checkpoint formats: **safetensors** (including GPTQ, AWQ, and FP8 layouts) and **GGUF** (single file with
embedded config and tokenizer). Both work from the library (`poot-llm`). A dense decoder loads as a
`driver::ModelHandle` and generates through the driver; the mixture-of-experts and hybrid families still load as
a `Runner`. The checkpoint's config decides which: the dense families are the ones the model registry knows.
Generation serving in `poot-serve` is not available yet (poot is being refactored; planned to return). Gemma 4 and Qwen3-Next are also not available yet (poot is being refactored; planned to return). Both loaders refuse them with an unsupported-model error naming the architecture.

## Dense decoders

Qwen2/3, Llama, Mistral, Gemma 2/3, Granite, OLMo 2, Phi-3/4, SmolLM3, BLOOM, MPT.

Standard causal LM stack: RMSNorm (or equivalent), RoPE, SwiGLU or GELU MLP, multi-head attention. GGUF and
safetensors are both in routine use, including GPTQ, AWQ and FP8 safetensors, whose quantized projections stay
packed as stored. These load with `ModelHandle::load` and generate with `Driver` on wgpu, ROCm, or PTX; see
the matrix for backend coverage. Gemma 2's sliding window covers the even layers, as in the HF reference.

## Mixture-of-experts (MoE)

Qwen3-MoE, GraniteMoE, Mixtral, OLMoE, gpt-oss.

Router plus sparse expert MLPs, with every expert's weights resident (expert pooling is not available yet (poot is being refactored; planned to return)). These load through `Runner::load` and `Runner::load_gguf`. A Mixtral GGUF does not load for now: its file names the `llama` architecture, which loads as the dense Llama family, and that family does
not carry experts yet. Mixtral and Qwen3-MoE have the strongest cross-backend library coverage; thinner paths
exist for some MoE families; check the matrix for your architecture and backend.

## MLA and hybrid attention

DeepSeek-V2/V3 (multi-head latent attention).

DeepSeek-V3 adds high-expert-count MoE. Qwen3-Next (hybrid linear-attention plus MoE) is not available yet (poot is being refactored; planned to return). Backend wiring differs by model; the matrix is authoritative.

## Embeddings and reranking

BERT-class encoder models for `/v1/embeddings` and cross-encoder reranking.

Safetensors is the usual format for HF-style encoder checkpoints. In `poot-serve`, encoders and cross-encoders
run on the CPU evaluator, not a GPU backend.

## Vision-language

SmolVLM, idefics3.

Multimodal chat: vision tower plus text decoder, library path only (the `poot-caption` binary). Vision-language serving is not available yet (poot is being refactored; planned to return). ROCm and
PTX GPU verification is thinner than wgpu for some modalities; see the matrix before relying on it.

## Synthetic and partial families

These load through the same `Runner::load` path and pass CPU-oracle and bounded synthetic tests, but they
are not verified for production HTTP serving:

- **Nemotron-H** (hybrid Mamba2 + attention)
- **DeepSeek-V3.2 DSA** (sparse attention)

:::note
Synthetic fixtures prove tracing, load, and single-sequence wgpu generation against the CPU oracle. They
do not mean a public checkpoint is verified on every backend or in continuous-batching serve. Read the
[feature matrix](../reference/feature-matrix.md) rows for these families before relying on them.
:::

## Families that do not load yet

`Runner::load` refuses these families with an unsupported-model error that names the architecture string
(`deepseek_v4`, `minimax_m2`, `glm5_next`, `qwen3_5`, `qwen4_exp`):

- **DeepSeek-V4-Flash-0731**
- **MiniMax-M2.5**
- **GLM-5.3-Flash**
- **Qwen3.8-27B-FP8** (`qwen3_5` text tower)
- **Qwen3.8 QSA** (`qwen4_exp`)

These families are being restored (poot is being refactored; planned to return). Until then no backend runs them, and the cells for them in the
[feature matrix](../reference/feature-matrix.md) are **No (being restored)**.

## Next steps

- Library generation: [Run a model (library)](./run-a-model.md)
- HTTP serving: [Serving your first model](../serve/serving-your-first-model.md)
- Backend choice: [Picking a backend](../serve/choosing-a-backend.md)
- Full capability tables: [Feature matrix](../reference/feature-matrix.md)
