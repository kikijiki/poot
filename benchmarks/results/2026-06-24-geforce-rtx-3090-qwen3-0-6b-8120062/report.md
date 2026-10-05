# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.65.06, 24576 MiB)
- poot commit: 8120062
- captured: 2026-06-24T02:50:53.216358+00:00

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 10.2 | 8.0 | 125.6 | 7.41 | 1.67 |  |
| llamacpp | f16 | ok | 6.8 | 2.7 | 376.2 | 3.06 | 1.70 | GGUF f16 vs bf16 - closest full-precision GGUF; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); precision f16 != matched target bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot | bf16 | ok | 189.5 | 17.6 | 56.7 | 15.61 | 6.89 |  |
| transformers | bf16 | ok | 29.9 | 28.3 | 35.4 | 5.85 | 2.08 |  |
| vllm | bf16 | ok | 12.0 | 2.5 | 404.7 | 8.12 | 3.49 | prefix caching disabled; per-token ITL from AsyncLLM streaming; vLLM KV pool at gpu_memory_utilization=0.3 inflates external VRAM |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 125.6 | 127.4 | 115.6 | 43.8 |
| llamacpp | 376.2 | 367.6 | 321.9 | 252.2 |
| poot | 56.7 | 51.2 | 39.4 | 19.9 |
| transformers | 35.4 | 35.5 | 35.5 | 35.4 |
| vllm | 404.7 | 401.0 | 358.9 | 263.5 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 8.0 | 7.9 | 8.7 | 22.8 |
| llamacpp | 2.7 | 2.7 | 3.1 | 4.0 |
| poot | 17.6 | 19.5 | 25.4 | 50.3 |
| transformers | 28.3 | 28.2 | 28.2 | 28.2 |
| vllm | 2.5 | 2.5 | 2.8 | 3.8 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._