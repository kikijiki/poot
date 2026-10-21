---
id: performance
title: Performance
sidebar_position: 2
description: Single-request decode and prefill throughput of poot on the wgpu and ROCm backends, with the identity of every measured run.
---

# Performance

Every number on this page comes from a run of poot's own benchmark harness (`benchmarks/`), recorded with the
identity the harness stores in each result row: the build, the adapter and driver, the dtype the model
computes in and the checkpoint digest. The run directories are in `benchmarks/results`, indexed in
`benchmarks/results/INDEX.md`.

:::info Scope
These are **single-request decode and prefill** measurements, not continuous-batching throughput. Which
families, formats and backends are supported is in the [feature matrix](./feature-matrix.md).
:::

## What was measured

| Field         | Value                                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Build         | poot `775975d9`, release profile                                                                                            |
| Hardware      | AMD Ryzen AI Max+ 395 "Strix Halo" with its Radeon 8060S iGPU (gfx1151)                                                     |
| wgpu adapter  | AMD Radeon 8060S Graphics (RADV STRIX_HALO), driver `radv Mesa 26.1.3`                                                      |
| ROCm adapter  | AMD Radeon 8060S Graphics, driver `hsa 1.18 amdgcn-amd-amdhsa--gfx1151`                                                     |
| Model         | Qwen2.5-0.5B, bf16 safetensors checkpoint, digest `blake3:ed28ccbfe8235cc8b9f74c7779287d543faa71671816cd8fe09d0068d7ea35fb` |
| Compute dtype | `f32`                                                                                                                       |

The decode-128 cell is measured on wgpu and ROCm, the prefill-1k cell on ROCm only; decode against context length is not measured on this page. PTX and raw Vulkan are not measured on this page.

Each cell is five repeats of the harness's own warmup and timed iterations; the tables give the median over the
five repeats with the minimum and maximum in brackets. The iGPU is shared with the desktop, so the spread is
part of the result.

## What poot computes

The recorded identity of this run says `compute_dtype: f32`: the dense bf16 checkpoint is read as bf16 and its
tensors are computed in f32. That is a property of the dense safetensors path measured here. Packed GGUF,
GPTQ/AWQ and supported FP8 projections stay packed in memory and are decoded by the generated kernels as they
consume them; see [Using quantized checkpoints](../serve/quantized-checkpoints.md). A run on a quantized
checkpoint carries its own identity and is not comparable to the rows below.

## Decode and prefill

| Scenario                                        | Backend | Decode tok/s           | TPOT ms | TTFT ms          |
| ----------------------------------------------- | ------- | ---------------------- | ------- | ---------------- |
| `decode-128` (5 prompt tokens, 128 generated)   | wgpu    | 165.9 (164.6 to 167.3) | 6.0     | 10.2             |
| `decode-128` (5 prompt tokens, 128 generated)   | ROCm    | 160.8 (158.8 to 162.0) | 6.2     | 9.6              |
| `prefill-1k` (1041 prompt tokens, 32 generated) | ROCm    | 150.0 (148.2 to 151.8) | 6.7     | 449 (439 to 450) |

`decode-128` isolates decode: the decode rate is the steady rate once the short prompt is primed. `prefill-1k`
isolates the prefill pass: its time to first token is the whole 1041-token prompt in one pass, about 2,300
prompt tokens per second on ROCm.

### Prefill attention

The attention of the prefill plan is the same generated streaming kernel as decode's, with the causal mask
stated by position: one workgroup per (query row, head) keeps the online-softmax state in workgroup memory, so
the `[query x key]` score matrix is never written. The plan summary of the Qwen2 prefill graph (build `775975d9`, committed as `benchmarks/performance-bank/plan-summary/qwen2-attention-775975d9.tsv`) lists every attention equation as the `generated:attention` choice with `schedule=streaming` on the AMD GCN, Vulkan, SPIR-V Vulkan and NVPTX targets. A tiled prefill schedule is planned; the
[attention page](../architecture/attention.mdx#the-streaming-schedule) describes the schedule.

## Low-head decode attention at the kernel level

One decode attention step with 2 query heads of width 128 over 1 key/value head, f32, as the performance bank
measures it through the executor contract: wall step plus readback, nine timed repeats, same build and
adapters as above. The values are the median with the minimum and maximum in brackets, in microseconds.

| Cache | Live keys              | wgpu              | ROCm                |
| ----- | ---------------------- | ----------------- | ------------------- |
| 4096  | 4096 (the cache bound) | 236 (105 to 470)  | 47.6 (46.7 to 49.0) |
| 32768 | 32768                  | 323 (266 to 403)  | 158 (146 to 165)    |
| 32768 | 1024                   | 641 (209 to 2135) | 79.0 (77.1 to 84.7) |

The planner splits the keys of a low-head decode across workgroups: only for a single-row decode whose rows do
not already fill the device's compute units, at most 16 splits, never a chunk under 128 keys (see
[Split-KV decode](../architecture/attention.mdx#split-kv-decode)). The plans of all three rows are the two-launch
partials-and-combine form. A two-head decode on a 40-compute-unit device therefore runs up to thirty-two
workgroups instead of two. The ROCm spread is narrow; the wgpu rows have outlier repeats (the 1024-live-key row
spans 209 to 2135 us), so read them as an order of magnitude, not a precise figure. Every row's result agrees
with the bank's independent f64 reference within its tolerance. The raw receipts are under
`benchmarks/performance-bank/candidates`. PTX and raw Vulkan rows are not recorded.

## Reproducing

```bash
# 5 repeats of one cell on the local backend, then regenerate the index
just bench-record rocm qwen2.5-0.5b prefill-1k
python3 benchmarks/harness/bench.py index
```

`just bench-record <backend> <model> <scenario>` takes the scenarios `decode-128`, `prefill-1k` and
`decode-curve`. `just bench-compare` judges a new set of repeats against the committed one with the harness's
noise-aware comparator.

## Planned comparisons

Comparisons against llama.cpp and vLLM on NVIDIA hardware, and poot's PTX backend on that hardware, are
planned. No pooled MoE decode throughput is measured; see [expert pooling](./feature-matrix.md#expert-pooling).

## Request timing

The benchmark runner records request start, every delivered token, and completion. TTFT ends at the first
token. TPOT is the mean of all inter-token gaps, including the first gap; finish is the time from the last
token to completion. E2E includes all three windows. Preparing wgpu decode before delivering the first token
includes submitting and completing pending uploads after a cache build or request-state reset. That cost
belongs to preparation, TTFT and E2E; it can increase TTFT and reduce the first gap without reducing E2E. Raw
cold and repeated requests are retained separately. For zero tokens, first/last token times are absent and the
whole request is finish; zero/one-token requests have no token-to-token gap, so their TPOT and decode rate
are undefined (null in the per-request record).

Unprofiled wgpu execution groups dispatches within each existing submission into one compute pass. Profiling
preserves per-dispatch passes and adds measurement overhead; use unprofiled wall time for latency comparisons.
