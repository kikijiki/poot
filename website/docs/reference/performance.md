---
id: performance
title: Performance
sidebar_position: 2
description: Single-request decode and prefill throughput across backends and frameworks.
---

# Performance

This page preserves benchmark snapshots from July and August 2026. They describe those binaries and
execution paths, not current master after the September architecture changes. Quantized loading,
kernel selection and executor code have changed since these runs; no current performance claim should
be inferred from the old tables.

:::info Scope
These are **single-request decode/prefill** snapshots, not continuous-batching throughput. The recorded
poot runs used the f32 path; comparison frameworks often used BF16/F16. For current format and model
limitations see the [feature matrix](./feature-matrix.md).
:::

All numbers below come from poot's own benchmark harness, which runs each `(model, framework, scenario)`
cell and records the raw result.

## Method

- **Hardware:** an NVIDIA RTX 3090 (rented, dedicated) for the cross-framework comparison; a local AMD Ryzen
  AI Max+ 395 "Strix Halo" iGPU (RADV/ROCm) for poot's cross-backend and the comparable-hardware view.
- **Model:** Qwen2.5-0.5B-Instruct (poot's most-verified model), bf16 checkpoint. The measured poot binary loaded bf16/f16 and
  upcast to f32; candle/transformers/vLLM run bf16; llama.cpp runs f16 GGUF.
- **Scenarios:** `decode-128` (short prompt, 128 generated tokens, decode throughput), `prefill-1k` (~1k
  prompt, 32 out, prefill/TTFT), `decode-curve` (decode throughput vs growing context).
- Warmup then 5 timed iterations per cell; steady-state decode tok/s reported.

## Decode throughput (RTX 3090, decode-128)

Decode tok/s (short prompt, 128 generated tokens). The Qwen2.5-0.5B column is the 2026-07-28 snapshot
(poot commit `78ca82f2`); the Qwen3-0.6B column is the 2026-07-03 snapshot (`a39b146b`), the latest
decode-128 run for that model. Compare rows within a column only.

| Framework            | Qwen2.5-0.5B (07-28) | Qwen3-0.6B (07-03) |
| -------------------- | -------------------- | ------------------ |
| llama.cpp (f16 GGUF) | 424                  | 386                |
| vLLM (bf16)          | (init error)         | 398                |
| candle (bf16)        | 133                  | 124                |
| poot PTX (f32)       | 101                  | 45                 |
| transformers (bf16)  | 50                   | 32                 |

poot sits above transformers and below candle here. llama.cpp and vLLM are far ahead; they use
f16/quantized paths, while these poot snapshots used the f32 path. vLLM's
engine core failed to initialize for Qwen2.5-0.5B on this pod but ran for Qwen3-0.6B, so its cell is flaky
(a setup/environment issue, not a model result).

## Prefill (RTX 3090, Qwen2.5-0.5B, prefill-1k)

| Framework    | tok/s |
| ------------ | ----- |
| llama.cpp    | 462   |
| candle       | 95    |
| transformers | 45    |
| poot (PTX)   | 49    |

poot's prefill is a naive O(L^2) attention pass with no flash-prefill on this path, a known weakness at
long prompts (poot has a fused flash-prefill kernel that this decode-oriented runner does not use yet). This number is the steady decode rate once the 1k-token prompt is primed, not the prefill pass
itself; TTFT for that prefill pass is still highly variable run to run (1.7-17.7 s across two same-day
snapshots), a separate, unresolved cost from the naive-attention pass above.

## Decode vs context length (RTX 3090, Qwen2.5-0.5B, poot PTX)

| Context (ISL) | Decode tok/s | Per-token (TPOT) |
| ------------- | ------------ | ---------------- |
| 128           | 74.0         | 13.5 ms          |
| 512           | 64.8         | 15.4 ms          |
| 2048          | 43.1         | 23.2 ms          |
| 8192          | 16.5         | 60.6 ms          |

Per-token cost grows with context because attention over the KV cache is O(context) per token on this
path (no flash-decode kernel wired here yet). This describes the measured binary.

## Cross-backend: one engine, three backends (Qwen2.5-0.5B, decode)

| Backend   | Hardware              | Decode tok/s |
| --------- | --------------------- | ------------ |
| poot PTX  | RTX 3090 (NVIDIA)     | 101          |
| poot ROCm | Strix Halo iGPU (AMD) | 27           |
| poot wgpu | Strix Halo iGPU (AMD) | 18           |

The PTX row is the 2026-07-28 decode-128 run; the ROCm and wgpu rows are the 2026-07-01 Strix Halo run
(27.1 and 18.2 tok/s, prompt 5, 40 generated), so the hardware and snapshots differ. The same Qwen2.5-0.5B
graph runs on all three of poot's backends. The RTX 3090 (a discrete GPU) leads the Strix Halo integrated
GPU; in that short decode run poot's native ROCm/HSA path is about 1.5x its portable Vulkan
(wgpu) path. In the later decode-curve run (2026-07-12) the two are level at ISL 128 (25.9 vs 25.7 tok/s), and
wgpu fails at ISL 2048 and above on a dispatch-grid cap. All three backends lower the same primitive graph, with no vendor
inference library underneath.

## Broader model coverage (RTX 3090 / L40S / RTX A6000, 2026-08-19 to 2026-08-20)

A wider decode-curve sweep across other supported models shows roughly the same gap as Qwen2.5-0.5B, at
short context (ISL 128):

| Model        | GPU      | poot tok/s | llama.cpp tok/s | poot vs llama.cpp |
| ------------ | -------- | ---------- | --------------- | ----------------- |
| OLMo 2 1B    | RTX 3090 | 39.2       | 279.4           | 0.14x             |
| Phi-3.5-mini | RTX 3090 | 25.8       | 100.4           | 0.26x             |
| SmolLM3 3B   | RTX 3090 | 16.5       | 121.0           | 0.14x             |

At larger sizes the sweep hits a different problem before it gets to a speed comparison: poot's cell
errored (rc=-9, consistent with an out-of-memory kill) on Qwen3-14B on an L40S (46 GiB), and did not
produce a result at all for Phi-4 on an RTX A6000 (48 GiB) in this run. Both are consistent with poot's
f32-everywhere memory footprint scaling worse than the bf16/f16 baselines at this parameter count; llama.cpp
completed both cells cleanly (25.8 and 24.0 tok/s). This is separate from the raw-speed gap above and is not root-caused.

## Notes and caveats

- **The measured path used f32.** Current packed GGUF, GPTQ/AWQ and supported FP8 projections stay
  packed at load time. Dense host mirrors and packed MoE materialization remain separate limitations;
  see [Using quantized checkpoints](../serve/quantized-checkpoints.md).
- **vLLM** is flaky on the test pods: its engine core failed to initialize for Qwen2.5-0.5B but ran for
  Qwen3-0.6B. That is a setup/environment issue, not a model result; failed cells are recorded as errors
  rather than dropped.
- **Run-to-run variance** on rented pods is ~10-15% (e.g. candle measured 131 then 111 tok/s on two
  different RTX 3090 rentals; poot's own prefill-1k TTFT varied 1.7-17.7 s across two same-day snapshots).
  The decode-128 and decode-vs-context-length tables above are each a single snapshot so their own rows are
  directly comparable within the table; the decode-128 table is from earlier snapshots (2026-07-28 and 2026-07-03) than the
  prefill and decode-vs-context-length tables (2026-08-20), so do not compare tok/s figures across tables.
- **`decode-128` and `decode-curve` are different scenarios, not the same measurement at ISL 128.**
  `decode-128` amortizes prefill into the reported rate; `decode-curve` (used in the decode-vs-context-length
  table and the broader-coverage table above) excludes prefill and reports steady-state TPOT only. That is
  why poot's ISL-128 decode-curve number (74.0 tok/s) does not match its decode-128 number (101 tok/s) above, they are not comparable to each other.
- **These numbers are from early builds.** poot's kernels and kernel selection are still being tuned (for example,
  `matmul_batched` beats the imported GEMV on Ampere for these projection shapes, which lifted PTX decode
  in the decode-128 scenario from 45 to 57 tok/s on 2026-07-03; the 2026-07-28 run measured 101 tok/s).
- **Pooled MoE decode is not available yet (poot is being refactored; planned to return).** There are no pooled throughput numbers; the numbers above cover the dense and
  standard-MoE decode path only. See the [feature matrix](./feature-matrix.md#expert-pooling). 

## Request timing in the current runner

The benchmark runner records request start, every delivered token, and completion. TTFT ends
at the first token. TPOT is the mean of all inter-token gaps, including the first gap; finish
is the time from the last token to completion. E2E includes all three windows. Preparing wgpu decode
before delivering the first token includes submitting and completing pending uploads after a
cache build or request-state reset. That cost belongs to preparation, TTFT and E2E; it can
increase TTFT and reduce the first gap without reducing E2E. Raw cold and repeated requests are
retained separately. For zero tokens, first/last token
times are absent and the whole request is finish; zero/one-token requests report zero TPOT and
decode rate. These conventions do not revise the historical tables above.

Unprofiled wgpu execution groups dispatches within each existing submission into one compute
pass. Profiling preserves per-dispatch passes and adds measurement overhead; use unprofiled
wall time for latency comparisons.
