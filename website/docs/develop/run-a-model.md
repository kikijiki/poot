---
id: run-a-model
title: Run a model (library)
sidebar_position: 6
---

# Run a model (library)

`poot-llm` loads checkpoints and runs generation from Rust. This is the library path; for the
OpenAI-compatible HTTP API, see [Serving](../serve/index.md).

## Checkpoint formats

poot loads **safetensors** (dense plus GPTQ, AWQ, and FP8 quant layouts) and **GGUF** (single-file
checkpoints with embedded config and tokenizer). Point either at a local path on disk.

## Example: GGUF generation

```bash
cargo run -p poot-llm --release --example gguf_generate -- /path/to/model.gguf
```

Optional prompt as a second argument. The example streams greedy tokens through the CPU oracle (good for
load-path smoke tests without a GPU).

A runnable, end-to-end GPU decode example is not available yet (poot is being refactored; planned to return). The GPU decode code paths are exercised only by `poot-llm`'s own test suite. Other
examples under `crates/poot-llm/examples/` remain: graph tracing and fusion (`trace_and_fuse`,
`flash_attention_fusion`), CPU-executor building blocks (`build_and_eval`), encoder embeddings and
reranking (`embeddings`), and sampling (`sampling`); none of them dispatch on a GPU.

## Backends

poot has **three production backends** and a fourth, lower-priority one (`--backend wgpu|rocm|ptx|vulkan`):

- **wgpu** (default): Vulkan/SPIR-V; any modern GPU; most complete continuous-batching library path
- **rocm**: AMD via raw HSA (`--features rocm`)
- **ptx**: NVIDIA via cudarc
- **vulkan**: the same SPIR-V kernels as wgpu through raw Vulkan (`ash`), with no wgpu; verified on AMD (RADV) only, NVIDIA and Intel Vulkan drivers are untested

CLI binaries (benchmark runners and examples) take `--backend rocm`, `--backend ptx` or `--backend vulkan`; build ROCm with
`--features rocm`. The `poot-serve` `--backend` option is not available yet (poot is being refactored; planned to return). In
library code you select the backend by wiring the matching executor crate (`poot-gpu`,
`poot-rocm-gpu`, `poot-ptx-gpu`, or `poot-vulkan-device`). See [Picking a backend](../serve/choosing-a-backend.md) for
tradeoffs.

:::note
NPU offload was removed. See [Develop: Backends](./index.md#backends).
:::

## What runs where

Model families and per-backend verification live in [Supported models](./models.md) and the
[feature matrix](../reference/feature-matrix.md). Use the matrix before assuming a family works on your
target backend. Generation serving is not available yet (poot is being refactored; planned to return).
