---
id: index
title: Serving
sidebar_position: 1
---

# Serving

`poot-serve` is the OpenAI-compatible HTTP server. These pages cover operating it: API shape,
checkpoints, first-time bring-up, and the features that are available. For embedding poot as a library,
see [Develop](../develop/index.md).

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today the server starts, parses and bounds requests, and serves encoder embeddings and reranking. `/v1/completions` and `/v1/chat/completions` return `503`, and `/health` reports `"engine_loaded": false`. Continuous batching, speculative decoding, tensor parallelism, pooled MoE serving, vision-language serving, LoRA adapters, Gemma 4 and Qwen3-Next are not available yet.

Design and paper provenance for continuous batching, speculative decoding, LoRA, quantization,
and tensor parallelism live under [Serving design](../architecture/serving-design.mdx).

## Bring-up

- **[Serving your first model](./serving-your-first-model.md)**: First HTTP responses from a GGUF or
  safetensors checkpoint.
- **[Picking a backend](./choosing-a-backend.md)**: Backends in the libraries; the server has no backend option.
- **[Using quantized checkpoints](./quantized-checkpoints.md)**: GGUF K-quants, GPTQ, AWQ, FP8.
- **[Configuration reference](./configuration.md)**: CLI flags and environment variables in one place.

## Reference

- **[Feature matrix](../reference/feature-matrix.md)**: Backend capabilities (models, quant, serving
  features).
- **[Performance](../reference/performance.md)**: Single-request decode benchmarks across backends.
- **[FAQ](../reference/faq.md)**: Short answers to common operator questions.

## API and modalities

- **[OpenAI-compatible API](./api.md)**: Endpoints, the `503` refusal, bounds and metrics.
- **[Guided decoding](./guided-decoding.md)**: Constrained output (regex, grammar, JSON schema, tools).
- **[Embeddings and reranking](./embeddings-and-reranking.md)**: `/v1/embeddings` and `/v1/rerank`.
- **[Multimodal](./multimodal.md)**: Not available yet (being restored).

## Serving features

- **[Continuous batching and concurrency](./continuous-batching.md)**: Not available yet (being restored).
- **[Speculative decoding](./speculative-decoding.md)**: Not available yet (being restored).
- **[Using LoRA adapters](./lora-adapters.md)**: Not available yet (being restored).
- **[Tensor parallelism](./tensor-parallelism.md)**: Not available yet (being restored).

## Library users

- **[Develop: Build and test](../develop/build-and-test.md)**: Nix shell, `just` recipes, release builds.
- **[Develop: Run a model](../develop/run-a-model.md)**: Load checkpoints and generate from Rust.
- **[Authoring kernels](../develop/authoring-kernels.md)**: Write `#[kernel]` functions through `pootc`.
