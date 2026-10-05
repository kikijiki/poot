# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 9136d22
- captured: 2026-06-17T11:34:26.716406+00:00

## qwen2.5-0.5b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                              |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| candle       | bf16      | ok     | 10.4    | 8.5     | 118.0        | 9.51          | 1.55         |                                                                                                                                                                                                                                                                    |
| llamacpp     | f16       | ok     | 5.6     | 2.2     | 452.0        | 2.06          | 1.22         | precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); GGUF f16 vs bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: Error: poot bench runner supports --mode single only; decode-curve needs a                                                                                                                                                                                   |
| transformers | bf16      | ok     | 19.7    | 18.6    | 53.8         | 4.81          | 1.60         |                                                                                                                                                                                                                                                                    |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                                                                    | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 118.0   | 118.1   | 117.3    | 65.5     |
| llamacpp     | 452.0   | 451.8   | 449.5    | 420.6    |
| transformers | 53.8    | 52.7    | 52.1     | 53.0     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 8.5     | 8.5     | 8.5      | 15.3     |
| llamacpp     | 2.2     | 2.2     | 2.2      | 2.4      |
| transformers | 18.6    | 19.0    | 19.2     | 18.9     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
