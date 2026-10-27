---
id: index
title: Develop
sidebar_position: 1
---

# Develop

This section is for embedding poot as a library and contributing kernels and engine code. For the HTTP server, start with [Serving](../serve/index.md); for design rationale, see [Architecture](../architecture/index.mdx).

## In this section

- **[Build](./build.md)**: Enter the Nix dev shell, build poot, kernel cache and backends.
- **[Compute engine](./compute-engine.md)**: Build graphs, fuse, run on CPU/GPU, or dispatch `#[kernel]`s
  without an LLM.
- **[Authoring kernels](./authoring-kernels.md)**: Write `#[kernel]` functions and dispatch them through
  `pootc`. [Compute engine](./compute-engine.md) covers how that fits next to the graph IR path.
- **[Embedding poot (LLM)](./embedding.md)**: `ModelHandle` load and `Driver` generate API for checkpoints.
- **[Run a model (library)](./run-a-model.md)**: Quick library generation examples.
- **[Models](./models.md)**: Checkpoint formats, kinds of model and how to add a family.
- **[Supported models and formats](./supported-models.md)**: Which families and weight formats plan on which
  backend, and what a loader refuses.
- **[Test](./test.md)**: Run the test suite, the device and checkpoint lanes.

## Backends

| Backend | Runs on | Select with |
| --- | --- | --- |
| wgpu | Any Vulkan-capable GPU | `--backend wgpu` |
| ROCm | AMD GPUs | `--backend rocm` |
| PTX | NVIDIA GPUs | `--backend ptx` |
| Vulkan (raw) | Any Vulkan-capable GPU | `--backend vulkan` |
| NPUs (AMD XDNA2, Intel NPU4) | In development; not a backend | n/a |

`--backend auto` is the default. [Picking a backend](../serve/choosing-a-backend.md#supported-backends) has the
tradeoffs and what is verified where; [Backends](../architecture/backends.mdx) has codegen and runtime details.
