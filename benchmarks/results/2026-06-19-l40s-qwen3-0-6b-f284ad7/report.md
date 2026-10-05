# poot benchmark report

- GPU: NVIDIA L40S (driver 570.124.06, 46068 MiB)
- poot commit: f284ad7
- captured: 2026-06-19T08:35:35.317628+00:00

## qwen3-0.6b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 7.8     | 6.3     | 159.1        | 7.75          | 1.70         |                                                                                                                                                                                                                                                                                                  |
| llamacpp     | f16       | ok     | 5.1     | 2.5     | 407.1        | 3.38          | 1.70         | llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot         | bf16      | ok     | 170.2   | 12.9    | 77.4         | 15.92         | 6.91         |                                                                                                                                                                                                                                                                                                  |
| transformers | bf16      | ok     | 21.6    | 19.8    | 50.5         | 6.11          | 2.04         |                                                                                                                                                                                                                                                                                                  |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                                                  | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 159.1   | 161.7   | 160.5    | 72.4     |
| llamacpp     | 407.1   | 397.0   | 357.6    | 263.7    |
| poot         | 77.4    | 70.5    | 58.0     | 23.2     |
| transformers | 50.5    | 50.4    | 50.0     | 49.9     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 6.3     | 6.2     | 6.2      | 13.8     |
| llamacpp     | 2.5     | 2.5     | 2.8      | 3.8      |
| poot         | 12.9    | 14.2    | 17.2     | 43.0     |
| transformers | 19.8    | 19.8    | 20.0     | 20.0     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
