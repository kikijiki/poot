# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: 53b9dfb8
- captured: 2026-07-03T10:25:19.068749+00:00

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 9.4 | 7.4 | 135.7 | 8.66 | 1.69 |  |
| llamacpp | f16 | ok | 7.7 | 2.6 | 380.5 | 3.06 | 1.70 | llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16 |
| poot | bf16 | ok | 169.7 | 36.4 | 27.5 | 15.61 | 9.72 |  |
| transformers | bf16 | ok | 29.7 | 29.0 | 34.4 | 5.85 | 2.17 |  |
| vllm | bf16 | ok | 11.9 | 2.5 | 396.3 | 8.12 | 3.58 | vLLM KV pool at gpu_memory_utilization=0.3 inflates external VRAM; prefix caching disabled; per-token ITL from AsyncLLM streaming |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 135.7 | 142.6 | 114.7 | 43.0 |
| llamacpp | 380.5 | 351.9 | 320.1 | 253.6 |
| poot | 27.5 | 14.6 | 3.2 | 0.9 |
| transformers | 34.4 | 33.5 | 34.1 | 34.0 |
| vllm | 396.3 | 393.7 | 356.0 | 264.7 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 7.4 | 7.0 | 8.7 | 23.2 |
| llamacpp | 2.6 | 2.8 | 3.1 | 3.9 |
| poot | 36.4 | 68.5 | 314.6 | 1154.6 |
| transformers | 29.0 | 29.8 | 29.3 | 29.4 |
| vllm | 2.5 | 2.5 | 2.8 | 3.8 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._