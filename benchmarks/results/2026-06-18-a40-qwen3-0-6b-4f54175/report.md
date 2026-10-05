# poot benchmark report

- GPU: NVIDIA A40 (driver 570.195.03, 46068 MiB)
- poot commit: 4f54175
- captured: 2026-06-18T19:14:51.863341+00:00

## qwen3-0.6b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 8.3     | 5.9     | 169.8        | 7.53          | 1.67         |                                                                                                                                                                                                                                                                                                  |
| llamacpp     | f16       | ok     | 7.6     | 3.2     | 316.2        | 3.17          | 1.70         | GGUF f16 vs bf16 - closest full-precision GGUF; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); precision f16 != matched target bf16 |
| poot         | bf16      | ok     | 780.5   | 26.3    | 38.1         | 15.73         | 8.96         |                                                                                                                                                                                                                                                                                                  |
| transformers | bf16      | ok     | 21.4    | 20.0    | 50.0         | 5.97          | 2.18         |                                                                                                                                                                                                                                                                                                  |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                                                  | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 169.8   | 167.0   | 96.6     | 37.5     |
| llamacpp     | 316.2   | 307.3   | 276.8    | 206.3    |
| poot         | 38.1    | 35.1    | 27.3     | 14.2     |
| transformers | 50.0    | 50.3    | 50.6     | 50.3     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 5.9     | 6.0     | 10.3     | 26.6     |
| llamacpp     | 3.2     | 3.3     | 3.6      | 4.8      |
| poot         | 26.3    | 28.5    | 36.6     | 70.6     |
| transformers | 20.0    | 19.9    | 19.8     | 19.9     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
