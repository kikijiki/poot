# poot benchmark report

- GPU: NVIDIA A40 (driver 570.195.03, 46068 MiB)
- poot commit: 4f54175
- captured: 2026-06-18T19:25:49.046995+00:00

## qwen3-4b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                              |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 26.7    | 19.7    | 50.8         | 19.13         | 8.02         |                                                                                                                                                                                                                                                                    |
| llamacpp     | f16       | ok     | 28.6    | 14.8    | 67.5         | 9.82          | 7.79         | precision f16 != matched target bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); GGUF f16 vs bf16 |
| poot         | bf16      | error  | -       | -       | -            | -             | -            | rc=-9: poot-bench: decode-curve isls=[128, 512, 2048, 8192] osl=128 warmup=2 ite                                                                                                                                                                                   |
| transformers | bf16      | ok     | 28.1    | 27.1    | 36.9         | 13.22         | 8.14         |                                                                                                                                                                                                                                                                    |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                    | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 50.8    | 46.5    | 32.6     | 15.4     |
| llamacpp     | 67.5    | 66.9    | 65.0     | 59.0     |
| transformers | 36.9    | 37.1    | 37.1     | 34.6     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 19.7    | 21.5    | 30.7     | 64.8     |
| llamacpp     | 14.8    | 14.9    | 15.4     | 17.0     |
| transformers | 27.1    | 27.0    | 26.9     | 28.9     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
