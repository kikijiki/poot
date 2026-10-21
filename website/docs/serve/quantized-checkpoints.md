---
id: quantized-checkpoints
title: Using quantized checkpoints
sidebar_position: 4
---

# Using quantized checkpoints

The library loads supported GGUF and safetensors weights in their packed format. Generated kernels
decode dense projections and embedding rows as they consume them. Loading a format does not imply
that every model architecture or execution path supports it; see the [feature matrix](../reference/feature-matrix.md).
`poot-serve` serves the checkpoints its engine plan prepares; a refusal names the equation and the missing capability.

## GGUF

`ModelHandle::load` loads a single GGUF file containing weights, model metadata and tokenizer data. This
example generates through the driver on the first device that opens; it is not a performance benchmark:

```bash
hf download <org>/<model>-GGUF <file>.gguf --local-dir ~/models/<model>-gguf

cargo run -p poot-llm --release --example gguf_generate -- \
  ~/models/<model>-gguf/<file>.gguf
```

The reader accepts F32, F16 and BF16, legacy quants `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
K-quants `Q2_K` through `Q6_K`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, and MXFP4. Quantized weights stay packed at
load time. Q8_0 and Q4_K_M library decode paths have real-checkpoint receipts on wgpu, ROCm
and PTX. Raw Vulkan has a real-checkpoint receipt for a bf16 safetensors decode and none for a GGUF quant. These
receipts do not establish every family/format combination.

`Q8_K` and the grid formats not listed above (the IQ1 family and the other IQ2 formats) are unsupported and refused by name.

A mixed format such as `Q4_K_M` uses a higher bit width for some tensors. The loader preserves each
tensor's stored format. Packed MoE experts stay packed too: the routed projection is one indexed packed
contraction that reads only the blocks of the expert each row selects, so no expert table is ever
decoded on the device. Packed projections and packed embedding gathers avoid full-table
materialization the same way. Each indexed launch covers one projection's rows at a time; a very large
row count (the experts of a long prompt chunk) is refused with a typed grid-limit error naming the
equation, so long prompts run in smaller chunks.

## GPTQ and AWQ (safetensors, 4-bit)

`ModelHandle::load` packs quantized projections directly from their stored tensors, for every dense family
the registry knows; GPTQ/AWQ projections need no opt-in loader to avoid a dense f32 copy. Supported configuration variants are
GPTQ 4-bit symmetric with `desc_act: false` in the v1 checkpoint format, and AWQ 4-bit GEMM with zero points.
Unsupported variants are refused by name: a GPTQModel `checkpoint_format: gptq_v2` checkpoint (its zeros would decode
one unit off) and AWQ with `zero_point: false` are refused at load, naming the config field and value. A LoRA adapter whose
`adapter_config.json` sets a non-empty `rank_pattern` or `alpha_pattern` (per-module rank or alpha) is refused by name,
since poot applies the one `r` and `lora_alpha` to every target module.

There is one loader. It keeps the checkpoint's other BF16/F16 weights as stored words and holds no
widened f32 copy of them. Loading is therefore not a promise that all temporary device allocations stay
at packed checkpoint size.

## FP8 safetensors

The `compressed-tensors` per-channel E4M3 projection path, for any dense family the registry knows, uses the same packed descriptors and
compiler lowering as other quantized projections. Projections stay packed at load time (no f32
expansion) and need no separate resident-E4M3 executor route. Generated kernels decode the stored
values and apply their scales; this does not require native FP8 tensor-core instructions.

## Memory expectations

- Packed loading reduces weight storage; activations, KV capacity and upload buffers still consume
  memory. A routed expert table is uploaded once, one buffer per source role, with each expert padded
  to a whole number of 32-bit words.
- Dense BF16/F16 weights stay BF16/F16 in host memory after loading. The CPU reference reads them
  through explicit casts in the graph each time it runs a step, which costs time rather than resident
  memory. Keeping them stored is not a claim of full BF16/F16 arithmetic on every backend.
- Choose a supported model architecture and execution path as well as a format. A format-level kernel
  test alone is not a complete model-support receipt.

Design: [Serving design](../architecture/serving-design.mdx).
