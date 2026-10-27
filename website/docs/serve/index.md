---
id: index
title: Serving
sidebar_position: 1
---

# Serving

`poot-serve` is the OpenAI-compatible HTTP server. It loads a registered model, fits it to the device and serves `/v1/completions` and `/v1/chat/completions` from one scheduling loop. Embedding and reranking checkpoints serve `/v1/embeddings` and `/v1/rerank`, and so can a decoder. LoRA administration runs on the same engine thread.

Related pages:

- Embedding poot as a library: [Develop](../develop/index.md).
- The design behind these pages: [Serving design](../architecture/serving-design.mdx).
- Which families and weight formats plan on which backend: [Supported models and formats](../develop/supported-models.md).
- What each backend's lane runs: [Serving capabilities](./capabilities.md).
- Measured speed: [Performance](../performance/index.mdx).

## Getting started

- [Serving your first model](./serving-your-first-model.md): first HTTP responses from a checkpoint.
- [Picking a backend](./choosing-a-backend.md): the `--backend` option and how `auto` chooses.
- [Using quantized checkpoints](./quantized-checkpoints.md): packed GGUF and safetensors formats.
- [Configuration reference](./configuration.md): every command-line flag.

## API

- [OpenAI-compatible API](./api.md): endpoints, error classes, bounds and metrics.
- [Guided decoding](./guided-decoding.md): constrained output (regex, grammar, JSON schema, tools).
- [Embeddings and reranking](./embeddings-and-reranking.md): `/v1/embeddings` and `/v1/rerank`.
- [Multimodal](./multimodal.md): image input for a checkpoint with a vision front end.
- [Using LoRA adapters](./lora-adapters.md): load, select and unload adapters.

## Scheduling and placement

- [Continuous batching and concurrency](./continuous-batching.md): the scheduling loop, rows and bounds.
- [Speculative decoding](./speculative-decoding.md): prompt lookup, draft model and the model's own MTP layer.
- [Tensor parallelism](./tensor-parallelism.md): weights and the paged KV pool divided over GPU ranks.
- [Serving capabilities by backend](./capabilities.md): what each backend's lane runs, and expert pooling.
