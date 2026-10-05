# poot benchmark report

- GPU: NVIDIA L40S (driver 570.124.06, 46068 MiB)
- poot commit: f284ad7
- captured: 2026-06-19T08:29:53.950500+00:00

## qwen2.5-0.5b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                              |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 7.5     | 6.2     | 161.1        | 8.25          | 1.55         |                                                                                                                                                                                                                                                                    |
| llamacpp     | f16       | ok     | 4.3     | 2.1     | 487.1        | 2.39          | 1.22         | ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text) |
| poot         | bf16      | ok     | 121.5   | 10.8    | 92.7         | 11.39         | 4.70         | poot loads bf16 but computes f32 (engine is f32 end to end)                                                                                                                                                                                                        |
| transformers | bf16      | ok     | 14.4    | 12.6    | 79.1         | 5.14          | 1.58         |                                                                                                                                                                                                                                                                    |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                    | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 161.1   | 162.4   | 158.6    | 94.2     |
| llamacpp     | 487.1   | 481.5   | 458.4    | 432.5    |
| poot         | 92.7    | 86.9    | 73.0     | 47.5     |
| transformers | 79.1    | 77.7    | 78.7     | 78.0     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 6.2     | 6.2     | 6.3      | 10.6     |
| llamacpp     | 2.1     | 2.1     | 2.2      | 2.3      |
| poot         | 10.8    | 11.5    | 13.7     | 21.0     |
| transformers | 12.6    | 12.9    | 12.7     | 12.8     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
