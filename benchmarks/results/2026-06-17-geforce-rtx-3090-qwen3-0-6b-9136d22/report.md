# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 9136d22
- captured: 2026-06-17T11:38:45.741807+00:00

## qwen3-0.6b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| candle       | bf16      | ok     | 10.1    | 7.7     | 129.5        | 8.57          | 1.67         |                                                                                                                                                                                                                                                                                                  |
| llamacpp     | f16       | ok     | 7.1     | 2.6     | 382.7        | 3.06          | 1.70         | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text) |
| poot         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: Error: poot bench runner supports --mode single only; decode-curve needs a                                                                                                                                                                                                                 |
| transformers | bf16      | ok     | 29.3    | 28.0    | 35.8         | 5.85          | 2.17         |                                                                                                                                                                                                                                                                                                  |
| vllm         | bf16      | ok     | 10.0    | 2.5     | 399.2        | 8.12          | 3.57         | vLLM KV pool at gpu_memory_utilization=0.3 inflates external VRAM; prefix caching disabled; per-token ITL from AsyncLLM streaming                                                                                                                                                                |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 129.5   | 130.3   | 115.0    | 43.2     |
| llamacpp     | 382.7   | 373.7   | 344.9    | 269.3    |
| transformers | 35.8    | 36.7    | 37.0     | 37.1     |
| vllm         | 399.2   | 395.9   | 356.1    | 266.5    |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 7.7     | 7.7     | 8.7      | 23.1     |
| llamacpp     | 2.6     | 2.7     | 2.9      | 3.7      |
| transformers | 28.0    | 27.2    | 27.0     | 27.0     |
| vllm         | 2.5     | 2.5     | 2.8      | 3.8      |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
