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

- Packed MoE paths currently materialize expert tables before indexed matmul.
- Dense BF16/F16 weights are kept as stored words after loading; there is no host f32 copy. The CPU
  reference reads them through explicit casts in the graph on each step.
- The serving settings that tuned the wgpu batch (`POOT_SLOTS` / `POOT_CAP`) and stored int8 KV (`POOT_KV_QUANT`) are not available yet (poot is being refactored; planned to return). Library callers
  set the batch shape directly.

FP8 projection storage and model support are separate questions. See
[Using quantized checkpoints](../serve/quantized-checkpoints.md) for the supported layouts and current
materialization limits.

## Why was a graph, kernel or prepared entry refused with a limit error?

Compilation and the driver's prepared-entry set are bounded, and every bound is a typed error rather
than an out-of-memory failure. A pass that would grow a graph past `CompileLimits` returns
`CompileError::Expansion`, a generated kernel over the instruction or local limit is a planner refusal
(`BodyLimit`), and a compiled artifact over `max_artifact_bytes` is refused before it is loaded or
cached. A driver also refuses a new prepared entry past its entry count or retained-byte limit, and a
step above `max_trace_tokens` before the model traces it; a longer prompt runs as legal chunks. The
retained-byte count is host metadata the driver can measure: it is not device memory, and the
artifact limit does not bound the external compiler's own memory or time.

## Can I use continuous batching and tensor parallelism together?

Neither is available yet (poot is being refactored; planned to return). See [Serving](../serve/index.md).

## Where are prefix caching and fast batched prefill available?

They are library-level features, not served yet (being restored). wgpu has the broadest library coverage. PTX and
ROCm generic decoders both reuse completed prompt prefixes. ROCm also has a hardware-verified one-shot path
for fresh, f32-KV Qwen2-family requests, both unadapted and with one selected LoRA adapter. Other ROCm cells
and all PTX prompts retain their existing prefill paths; PTX has no fast one-shot prefill yet.

## Why is sampled decoding slower on ROCm?

Today it is not backend-specific. Greedy and plain-temperature requests (including top-k/top-p/
min-p) pick the next token on-device, on every backend: a sampling step is appended to the decode
graph itself, and the host reads back only the chosen token id and a non-finite flag. A request with
a logit-bias/penalty, logprob recording, or a guided-decoding constraint still reads the full logits
back and picks on the host, since those transforms are host-side - and that readback/host-sampling
cost is the same on ROCm, wgpu and PTX.

## What bounds a hung ROCm GPU job?

The ROCm runtime waits for a submission or DMA with a generous timeout (600 s by default). Set
`POOT_GPU_WAIT_TIMEOUT_SECS` to lower it. On a timeout the runtime reports a typed error and poisons the
context so no later dispatch reuses the queue.

## Is raw Vulkan or NPU a fourth production backend?

Raw Vulkan is selectable (`--backend vulkan`) but is the lowest-priority backend; production is `wgpu` | `rocm` | `ptx`. NPU offload was removed.

- Raw Vulkan implements the same executor contract as the others. It is verified on AMD (RADV) only; NVIDIA and Intel Vulkan drivers are untested.
- NPU offload was removed. See [Architecture: NPU
  Offload](../architecture/npu.mdx).

## Tests skipped. Did verification pass?

No. GPU and model tests **skip cleanly** when the device or checkpoint is missing. Skips are not
hardware proof. Use `just test-device-wgpu` (alias `just test-gpu`) when a missing wgpu device should
**fail loudly**.

## Do I need `nix develop`?

For normal work in this repo, yes. It pins nightly Rust, LLVM with SPIR-V/NVPTX, Vulkan, `just`, and
`nextest`. Without it you will usually hit the wrong `rustc` or a missing `llc` target.
