---
id: faq
title: FAQ
sidebar_position: 3
---

# FAQ

Gotchas that are easy to miss. Capability tables live in the [feature matrix](./feature-matrix.md);
flags in [Configuration](../serve/configuration.md); backend choice in
[Picking a backend](../serve/choosing-a-backend.md).

## Why is debug so slow?

Inference is **10-50x slower** in debug. Always use `--release` for model runs and `poot-serve`.

## Out of memory on a model that fits elsewhere

Packed GGUF, GPTQ/AWQ and supported FP8 projections remain packed at load time. Memory also includes
activations, KV capacity, staging and temporary tensors; a packed checkpoint's size is not its peak VRAM.

- Packed MoE experts are read by an indexed packed contraction; no expert table is materialized.
- Dense BF16/F16 weights are kept as stored words after loading; there is no host f32 copy. The CPU
  reference reads them through explicit casts in the graph on each step.
- The server does not read batch settings: the engine plan fits the row count and positions to the device.
  Library callers set the batch shape directly in `DriverOptions`.

FP8 projection storage and model support are separate questions. See
[Using quantized checkpoints](../serve/quantized-checkpoints.md) for the supported layouts and current
memory limits.

## Why was a graph, kernel or prepared entry refused with a limit error?

Compilation and the driver's prepared-entry set are bounded, and every bound is a typed error rather
than an out-of-memory failure. A pass that would grow a graph past `CompileLimits` returns
`CompileError::Expansion`, a generated kernel over the instruction or local limit is a planner refusal
(`BodyLimit`), and a compiled artifact over `max_artifact_bytes` is refused before it is loaded or
cached. A driver also refuses a new prepared entry past its entry count or retained-byte limit. Tracing
is bounded the same way by the caller's `TraceLimits`: a step above `max_tokens` is refused before the
model traces it (a longer prompt runs as legal chunks), and a trace that would build more values,
equations, scalar-literal bytes or owned records than the limits allow is refused with the exact resource
and total (`TraceError::Budget`), before the excess is allocated or appended. The
retained-byte count is host metadata the driver can measure: it is not device memory, and the
artifact limit does not bound the external compiler's own memory or time.

## Can I use continuous batching and tensor parallelism together?

Yes, on the `ptx` backend. Continuous batching is served ([Continuous batching](../serve/continuous-batching.md)),
and a tensor group divides the paged KV pool by KV heads. A tensor group served requests on a two-GPU NVLink A100
pair, greedy and sampled matching single-GPU, with each rank admitting against its own pool. A four-GPU group
decoded a tiny model to the single-GPU logits. Serving a group of four through the server, a real checkpoint, and GPUs
that reach each other only across CPU sockets are not covered. Other backends do not support it.

## Where is prefix caching available?

Prefix reuse is driver code, so it applies on every backend ([Continuous batching](../serve/continuous-batching.md)
describes it). Its device tests run on wgpu and ROCm; PTX serving is verified for the dense fixture only, and no
raw Vulkan device test covers prefix reuse. The driver's chunked prefill is the only prefill path, on every backend.

## Why is sampled decoding slower on ROCm?

It is not backend-specific. Every request picks the next token on-device, on every backend: a
sampling step is appended to the decode graph itself, and the host reads back only the chosen token id,
a non-finite flag and any top-n logprob alternatives. Logit bias, penalties, logprobs and guided masks
run in that step too, so no request reads the full logits back unless the caller asks for them.

## What bounds a hung GPU job?

The ROCm and raw Vulkan runtimes wait for a submission with a 600 s timeout. On a timeout, or when a submitted
command buffer is lost, the runtime reports a typed error and poisons the device so no later dispatch reuses it; the
server then answers `/health` with `503` and `"status": "dead"` ([Continuous batching](../serve/continuous-batching.md)).

## Are raw Vulkan and the NPUs production backends?

Production is `wgpu`, `rocm` and `ptx`. Raw Vulkan is selectable with `--backend vulkan` and described in the [feature matrix](./feature-matrix.md#raw-vulkan). The AMD XDNA2 and Intel NPU4 NPUs are experimental library devices, not backends you can serve with ([Architecture: NPU Offload](../architecture/npu.mdx)).

## Tests skipped. Did verification pass?

No. GPU and model tests **skip cleanly** when the device or checkpoint is missing. Skips are not
hardware proof. Use `just test-device-wgpu` (alias `just test-gpu`) when a missing wgpu device should
**fail loudly**.

## Do I need `nix develop`?

For normal work in this repo, yes. It pins nightly Rust, LLVM with SPIR-V/NVPTX, Vulkan, `just`, and
`nextest`. Without it you will usually hit the wrong `rustc` or a missing `llc` target.
