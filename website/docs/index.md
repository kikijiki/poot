---
id: index
title: Documentation
slug: /
sidebar_position: 1
---

# Documentation

poot traces decoder models into a backend-neutral primitive graph, fuses it into GPU kernels, and runs
inference with async capture/replay.

## Pick a path

- **[Develop](./develop/index.md)**: build the workspace, run models as a library, author `#[kernel]` functions,
  embed crates.
- **[Serve](./serve/index.md)**: OpenAI-compatible HTTP server. It starts, bounds requests and serves encoder
  embeddings and reranking; generation returns when the new scheduling loop lands.
- **[Architecture](./architecture/index.mdx)**: graph IR, fusion, backends, capture/replay, serving design.

## Develop

- [Build and test](./develop/build-and-test.md)
- [Compute engine](./develop/compute-engine.md)
- [Authoring kernels](./develop/authoring-kernels.md)
- [Run a model](./develop/run-a-model.md)
- [Embedding poot (LLM)](./develop/embedding.md)
- [Supported models](./develop/models.md)

## Serve

- [Serving your first model](./serve/serving-your-first-model.md)
- [Picking a backend](./serve/choosing-a-backend.md)
- [OpenAI-compatible API](./serve/api.md)
- [Configuration reference](./serve/configuration.md)
- [Continuous batching](./serve/continuous-batching.md)
- [Speculative decoding](./serve/speculative-decoding.md)
- [LoRA adapters](./serve/lora-adapters.md)
- [Quantized checkpoints](./serve/quantized-checkpoints.md)
- [Guided decoding](./serve/guided-decoding.md)
- [Embeddings and reranking](./serve/embeddings-and-reranking.md)
- [Multimodal](./serve/multimodal.md)
- [Tensor parallelism](./serve/tensor-parallelism.md)

## Architecture

- [Architecture overview](./architecture/index.mdx)
- [Graph IR](./architecture/graph-ir.mdx)
- [Fusion and planning](./architecture/fusion.mdx)
- [Backends](./architecture/backends.mdx)
- [Tracing](./architecture/tracing.mdx)
- [Kernel import](./architecture/kernel-import.mdx)
- [Execution](./architecture/execution.mdx)
- [Megakernel](./architecture/megakernel.mdx)
- [Attention](./architecture/attention.mdx)
- [Serving design](./architecture/serving-design.mdx)
- [Implementation](./architecture/implementation.mdx)
- [NPU offload](./architecture/npu.mdx)
- [Glossary](./architecture/glossary.mdx)

## Reference

- [Feature matrix](./reference/feature-matrix.md)
- [Performance](./reference/performance.md)
- [FAQ](./reference/faq.md)

Production backends: **wgpu** (default), **rocm**, **ptx**. Raw Vulkan (**vulkan**) implements the same
executor contract; it is verified on AMD (RADV) only, and NVIDIA and Intel Vulkan drivers are untested. NPU
offload was removed. See
[Backends](./architecture/backends.mdx).
