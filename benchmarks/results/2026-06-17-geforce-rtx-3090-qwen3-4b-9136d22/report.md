# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 9136d22
- captured: 2026-06-17T11:44:01.839554+00:00

## qwen3-4b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                              |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 24.6    | 14.7    | 68.2         | 22.13         | 8.07         |                                                                                                                                                                                                                                                                    |
| llamacpp     | f16       | ok     | 22.1    | 10.6    | 94.0         | 9.71          | 7.80         | GGUF f16 vs bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text) |
| poot         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: Error: poot bench runner supports --mode single only; decode-curve needs a                                                                                                                                                                                   |
| transformers | bf16      | ok     | 39.4    | 37.8    | 26.4         | 12.84         | 8.18         |                                                                                                                                                                                                                                                                    |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                    | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 68.2    | 61.6    | 41.8     | 18.5     |
| llamacpp     | 94.0    | 93.3    | 90.7     | 82.1     |
| transformers | 26.4    | 26.3    | 25.7     | 25.9     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 14.7    | 16.2    | 23.9     | 54.1     |
| llamacpp     | 10.6    | 10.7    | 11.0     | 12.2     |
| transformers | 37.8    | 38.0    | 38.8     | 38.6     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
