# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: d28d5fbe
- captured: 2026-09-10T20:39:48.726508+00:00

## qwen2.5-0.5b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 8.6 | 7.2 | 139.8 | 9.54 | 1.53 | 339.0 | 4.323 |  |
| llamacpp | f16 | ok | 6.1 | 2.2 | 452.9 | 2.06 | 1.22 | 340.5 | 1.325 | GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot | bf16 | ok | 165.8 | 13.8 | 72.6 | 10.40 | 6.83 | 327.0 | 15.590 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 18.4 | 17.7 | 56.5 | 4.81 | 1.57 | 254.4 | 2.986 |  |
| vllm | bf16 | ok | 7.4 | 2.1 | 476.1 | 20.98 | 3.35 | 343.1 | 2.423 | prefix caching disabled; per-token ITL from AsyncLLM streaming; vLLM KV pool at gpu_memory_utilization=0.85 inflates external VRAM |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 139.8 | 144.0 | 143.5 | 64.3 |
| llamacpp | 452.9 | 450.5 | 446.5 | 391.3 |
| poot | 72.6 | 63.5 | 42.4 | 16.2 |
| transformers | 56.5 | 57.2 | 57.5 | 56.6 |
| vllm | 476.1 | 474.8 | 473.2 | 468.2 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 7.2 | 6.9 | 7.0 | 15.6 |
| llamacpp | 2.2 | 2.2 | 2.2 | 2.6 |
| poot | 13.8 | 15.7 | 23.6 | 61.7 |
| transformers | 17.7 | 17.5 | 17.4 | 17.7 |
| vllm | 2.1 | 2.1 | 2.1 | 2.1 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._