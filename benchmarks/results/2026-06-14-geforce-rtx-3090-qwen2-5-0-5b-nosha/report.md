# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: None
- captured: 2026-06-14T16:33:35.206776+00:00

## qwen2.5-0.5b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | -------------------------------------------------------------------------------- | ---------------------------------------- |
| candle       | bf16      | ok     | 9.5     | 7.4     | 135.5        | 9.54          | 1.52         |                                                                                  |
| llamacpp     | f16       | error  | -       | -       | -            | -             | -            | rc=1:                                                                            |
| poot         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: Error: GPU dispatch failed: CUDA launch_kernel failed: DriverError(CUDA_ER |
| transformers | bf16      | ok     | 18.2    | 17.3    | 57.7         | 4.81          | 1.56         |                                                                                  |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                  | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 135.5   | 132.3   | 132.9    | 65.4     |
| transformers | 57.7    | 58.6    | 58.3     | 58.1     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 7.4     | 7.6     | 7.5      | 15.3     |
| transformers | 17.3    | 17.1    | 17.1     | 17.2     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
