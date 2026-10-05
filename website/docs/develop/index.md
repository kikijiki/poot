---
id: index
title: Develop
sidebar_position: 1
---

# Develop

This section is for embedding poot as a library and for contributing kernels and engine code. If you
only need the HTTP server, start with [Serving](../serve/index.md). For design rationale and IR details,
see [Architecture](../architecture/index.mdx).

## In this section

- **[Build and test](./build-and-test.md)**: Enter the Nix dev shell, sanity-check the toolchain, run
  `just` recipes.
- **[Compute engine](./compute-engine.md)**: Build graphs, fuse, run on CPU/GPU, or dispatch `#[kernel]`s
  without an LLM.
- **[Authoring kernels](./authoring-kernels.md)**: Write `#[kernel]` functions and dispatch them through
  `pootc`. [Compute engine](./compute-engine.md) covers how that fits next to the graph IR path.
- **[Embedding poot (LLM)](./embedding.md)**: `ModelHandle` load and `Driver` generate API for checkpoints.
- **[Run a model (library)](./run-a-model.md)**: Quick library generation examples.
- **[Supported models](./models.md)**: Model families, checkpoint formats, and where to check backend
  coverage.

## Backends

poot has **three production backends** and a fourth, lower-priority one (`--backend wgpu|rocm|ptx|vulkan`):

- **wgpu** (default): Vulkan/SPIR-V; any modern GPU; most complete continuous-batching library path
- **rocm**: AMD via raw HSA (`--features rocm`)
- **ptx**: NVIDIA via cudarc
- **vulkan**: the same SPIR-V kernels as wgpu through raw Vulkan; verified on AMD (RADV) only, NVIDIA and Intel Vulkan drivers are untested

**Removed:**

- **NPU offload**: removed; not a `--backend` and not in the library. See
  [NPU](../architecture/npu.mdx).

See [Backends](../architecture/backends.mdx) for codegen and runtime details, and [Picking a
backend](../serve/choosing-a-backend.md) for serve-time tradeoffs.
