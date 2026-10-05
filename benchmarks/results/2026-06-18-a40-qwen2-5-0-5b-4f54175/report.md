# poot benchmark report

- GPU: NVIDIA A40 (driver 570.195.03, 46068 MiB)
- poot commit: 4f54175
- captured: 2026-06-18T19:06:23.172533+00:00

## qwen2.5-0.5b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                              |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 8.0     | 6.6     | 152.6        | 8.03          | 1.51         |                                                                                                                                                                                                                                                                    |
| llamacpp     | f16       | ok     | 6.3     | 2.6     | 378.9        | 2.18          | 1.22         | GGUF f16 vs bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; precision f16 != matched target bf16 |
| poot         | bf16      | ok     | 531.5   | 22.0    | 45.4         | 11.20         | 7.56         | poot loads bf16 but computes f32 (engine is f32 end to end)                                                                                                                                                                                                        |
| transformers | bf16      | ok     | 15.1    | 14.1    | 70.7         | 4.93          | 1.61         |                                                                                                                                                                                                                                                                    |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                    | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 152.6   | 153.3   | 125.4    | 56.2     |
| llamacpp     | 378.9   | 371.6   | 363.9    | 339.4    |
| poot         | 45.4    | 47.2    | 44.8     | 26.3     |
| transformers | 70.7    | 70.2    | 70.0     | 70.2     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 6.6     | 6.5     | 8.0      | 17.8     |
| llamacpp     | 2.6     | 2.7     | 2.7      | 2.9      |
| poot         | 22.0    | 21.2    | 22.3     | 38.1     |
| transformers | 14.1    | 14.2    | 14.3     | 14.2     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
