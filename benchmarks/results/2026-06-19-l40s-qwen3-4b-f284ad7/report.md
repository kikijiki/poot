# poot benchmark report

- GPU: NVIDIA L40S (driver 570.124.06, 46068 MiB)
- poot commit: f284ad7
- captured: 2026-06-19T08:43:00.260001+00:00

## qwen3-4b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                              |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 19.9    | 14.5    | 68.8         | 19.21         | 8.05         |                                                                                                                                                                                                                                                                    |
| llamacpp     | f16       | ok     | 21.5    | 11.9    | 83.7         | 10.03         | 7.79         | GGUF f16 vs bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; precision f16 != matched target bf16 |
| poot         | bf16      | ok     | 955.2   | 43.3    | 23.1         | 38.83         | 32.89        |                                                                                                                                                                                                                                                                    |
| transformers | bf16      | ok     | 27.6    | 24.9    | 40.2         | 13.25         | 8.24         |                                                                                                                                                                                                                                                                    |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                    | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 68.8    | 65.0    | 54.4     | 26.3     |
| llamacpp     | 83.7    | 82.8    | 80.5     | 73.1     |
| poot         | 23.1    | 21.3    | 18.6     | 9.6      |
| transformers | 40.2    | 40.7    | 40.6     | 40.6     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 14.5    | 15.4    | 18.4     | 38.0     |
| llamacpp     | 11.9    | 12.1    | 12.4     | 13.7     |
| poot         | 43.3    | 46.8    | 53.9     | 103.9    |
| transformers | 24.9    | 24.6    | 24.6     | 24.6     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
