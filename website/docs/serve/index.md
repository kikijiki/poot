---
id: index
title: Serving
sidebar_position: 1
---

# Serving

`poot-serve` is the OpenAI-compatible HTTP server. These pages cover operating it: API shape,
checkpoints, first-time bring-up, and the features that are available. For embedding poot as a library,
see [Develop](../develop/index.md).

The server loads a registered model, fits it to the device and serves `/v1/completions` and `/v1/chat/completions` from one scheduling loop over the driver ([Continuous batching](./continuous-batching.md)). `/health` answers `200` `serving` once the engine thread is running, `503` while it starts, drains, or after it dies (`"status": "dead"` with the cause and reason). The embeddings and reranking routes and LoRA administration run on the same engine thread. Tensor parallelism and pooled MoE serving are not implemented. A SmolVLM checkpoint also takes images on `/v1/chat/completions` ([Multimodal](./multimodal.md)). Speculative decoding is served through the same loop ([Speculative decoding](./speculative-decoding.md)).

Design and paper provenance for continuous batching, speculative decoding, LoRA, quantization,
and tensor parallelism live under [Serving design](../architecture/serving-design.mdx).

## Bring-up

- **[Serving your first model](./serving-your-first-model.md)**: First HTTP responses from a GGUF or
  safetensors checkpoint.
- **[Picking a backend](./choosing-a-backend.md)**: The `--backend` option and how `auto` chooses.
- **[Using quantized checkpoints](./quantized-checkpoints.md)**: GGUF K-quants, GPTQ, AWQ, FP8.
- **[Configuration reference](./configuration.md)**: CLI flags and environment variables in one place.

## Reference

- **[Feature matrix](../reference/feature-matrix.md)**: Backend capabilities (models, quant, serving
  features).
- **[Performance](../reference/performance.md)**: Single-request decode benchmarks across backends.
- **[FAQ](../reference/faq.md)**: Short answers to common operator questions.

## API and modalities

- **[OpenAI-compatible API](./api.md)**: Endpoints, error classes, bounds and metrics.
- **[Guided decoding](./guided-decoding.md)**: Constrained output (regex, grammar, JSON schema, tools).
- **[Embeddings and reranking](./embeddings-and-reranking.md)**: `/v1/embeddings` and `/v1/rerank`.
- **[Multimodal](./multimodal.md)**: Image input for a SmolVLM checkpoint.

## Serving features

- **[Continuous batching and concurrency](./continuous-batching.md)**: The scheduling loop, rows and bounds.
- **[Speculative decoding](./speculative-decoding.md)**: Prompt lookup and draft-model policies, served through the one loop.
- **[Using LoRA adapters](./lora-adapters.md)**: Hot-load, select and unload adapters.
- **[Tensor parallelism](./tensor-parallelism.md)**: Not implemented; what the server does instead.

## Library users

- **[Develop: Build and test](../develop/build-and-test.md)**: Nix shell, `just` recipes, release builds.
- **[Develop: Run a model](../develop/run-a-model.md)**: Load checkpoints and generate from Rust.
- **[Authoring kernels](../develop/authoring-kernels.md)**: Write `#[kernel]` functions through `pootc`.
