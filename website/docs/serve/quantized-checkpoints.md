---
id: quantized-checkpoints
title: Using quantized checkpoints
sidebar_position: 4
---

# Using quantized checkpoints

The library loads supported GGUF and safetensors weights in their packed format. Generated kernels
decode dense projections and embedding rows as they consume them. Loading a format does not imply
that every model architecture or execution path supports it; see the [feature matrix](../reference/feature-matrix.md).
Generation serving is not available yet (poot is being refactored; planned to return) and returns `503`.

## GGUF

`ModelHandle::load` loads a single GGUF file containing weights, model metadata and tokenizer data. This
example generates through the driver on the first device that opens; it is not a performance benchmark:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf \
  --local-dir ~/models/qwen2.5-0.5b-gguf

cargo run -p poot-llm --release --example gguf_generate -- \
  ~/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf
```

The reader accepts F32, F16 and BF16, legacy quants `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
K-quants `Q2_K` through `Q6_K`, `IQ4_NL`, `IQ4_XS`, and MXFP4. Quantized weights stay packed at
load time. Q8_0 and Q4_K_M Qwen library decode paths have real-checkpoint receipts on wgpu, ROCm
and PTX; those receipts do not establish every family/format combination.

`Q8_K` and the IQ2/IQ3 grid formats are unsupported and refused by name.

A mixed format such as `Q4_K_M` uses a higher bit width for some tensors. The loader preserves each
tensor's stored format. Packed MoE expert paths currently materialize the expert table on the device
before indexed matmul, so their peak device memory can exceed the packed checkpoint size. Dense
packed projections and packed embedding gathers avoid that full-table materialization.

## GPTQ and AWQ (safetensors, 4-bit)

`ModelHandle::load` packs quantized projections directly from their stored tensors, for every dense family
the registry knows; GPTQ/AWQ projections need no opt-in loader to avoid a dense f32 copy. Supported configuration variants are
GPTQ 4-bit symmetric with `desc_act: false`, and AWQ 4-bit GEMM. Unsupported variants are refused.

There is one loader. It keeps the checkpoint's other BF16/F16 weights as stored words and holds no
widened f32 copy of them. Loading is therefore not a promise that all temporary device allocations stay
at packed checkpoint size.

## FP8 safetensors

The `compressed-tensors` per-channel E4M3 projection path, for any dense family the registry knows (not only
Qwen), uses the same packed descriptors and
compiler lowering as other quantized projections. It no longer expands those projections to f32 at
load time or requires a separate resident-E4M3 executor route. Generated kernels decode the stored
values and apply their scales; this does not require native FP8 tensor-core instructions.

The DeepSeek-V3.2 loader also preserves block-FP8 expert payloads and scales. That loader coverage is
not a real-checkpoint generation claim for DeepSeek-V3.2 or other FP8 families. Its bounded synthetic
family coverage and current backend limits are listed in the feature matrix. Packed MoE table
materialization also applies here.

## Memory expectations

- Packed loading reduces weight storage; activations, KV capacity, upload buffers and temporary expert
  tables still consume memory.
- Dense BF16/F16 weights stay BF16/F16 in host memory after loading. The CPU reference reads them
  through explicit casts in the graph each time it runs a step, which costs time rather than resident
  memory. Keeping them stored is not a claim of full BF16/F16 arithmetic on every backend.
- Choose a supported model architecture and execution path as well as a format. A format-level kernel
  test alone is not a complete model-support receipt.

Design notes: [Serving design](../architecture/serving-design.mdx).
