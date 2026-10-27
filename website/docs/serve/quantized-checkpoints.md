---
id: quantized-checkpoints
title: Using quantized checkpoints
sidebar_position: 4
---

# Using quantized checkpoints

The library loads supported GGUF and safetensors weights in their packed format. Generated kernels decode projections and embedding rows as they consume them. A loadable format does not imply every architecture or execution path supports it; see [Supported models and formats](../develop/supported-models.md). A refusal names the equation and the missing capability.

## GGUF

`ModelHandle::load` loads a single GGUF file with weights, metadata and tokenizer. This example generates through the driver on the first device that opens; it is not a benchmark:

```bash
hf download <org>/<model>-GGUF <file>.gguf --local-dir ~/models/<model>-gguf

cargo run -p poot-llm --release --example gguf_generate -- \
  ~/models/<model>-gguf/<file>.gguf
```

Accepted formats, all kept packed at load time:

- F32, F16 and BF16;
- legacy quants `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`;
- K-quants `Q2_K` through `Q6_K`;
- `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, and MXFP4.

`Q8_K` and the other IQ1 and IQ2 grid formats are refused by name.

Real-checkpoint receipts: `Q8_0` and `Q4_K_M` library decode on wgpu, ROCm and PTX. Raw Vulkan has a receipt for a bf16 safetensors decode and none for a GGUF quant. Receipts do not establish every family/format combination.

A mixed format such as `Q4_K_M` uses a higher bit width for some tensors; the loader preserves each tensor's stored format.

- **Packed MoE experts** stay packed. The routed projection reads only the blocks of the expert each row selects, so no expert table is decoded on the device.
- **Row limit:** each indexed launch covers one projection's rows. A very large row count (the experts of a long prompt chunk) is refused with a typed grid-limit error naming the equation, so long prompts run in smaller chunks.

## GPTQ and AWQ (safetensors, 4-bit)

`ModelHandle::load` packs quantized projections directly from their stored tensors, for every dense family the registry knows, with no dense f32 copy.

Supported: GPTQ 4-bit symmetric with `desc_act: false` in the v1 checkpoint format, and AWQ 4-bit GEMM with zero points.

Refused by name at load, with the config field and value:

- a GPTQModel `checkpoint_format: gptq_v2` checkpoint (its zeros would decode one unit off);
- AWQ with `zero_point: false`;
- a LoRA adapter whose `adapter_config.json` sets a non-empty `rank_pattern` or `alpha_pattern`, since poot applies one `r` and `lora_alpha` to every target module.

## FP8 safetensors

The `compressed-tensors` per-channel E4M3 projection path works for any dense family the registry knows. Projections stay packed at load time; generated kernels decode the stored values and apply their scales. Native FP8 tensor-core instructions are not required.

## Memory

- Packed loading reduces weight storage. Activations, KV capacity and upload buffers still use memory, and temporary device allocations can exceed packed checkpoint size.
- A routed expert table is uploaded once, one buffer per source role, with each expert padded to whole 32-bit words.
- A checkpoint's size is not its peak memory. A model that fits on another stack can still run out here once activations, KV capacity, staging and temporaries are added.
- `poot-serve` fits rows and positions to the device through the engine plan; a library caller states the shape in `DriverOptions`.
- Other BF16/F16 weights stay BF16/F16 words in host memory, never widened to f32. The CPU reference reads them through casts in the graph at each step, which costs time, not resident memory.

Design: [Serving design](../architecture/serving-design.mdx).
