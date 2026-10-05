# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: None
- captured: 2026-06-14T16:41:55.252513+00:00

## qwen3-0.6b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | -------------------------------------------------------------------------------- | ---------------------------------------- |
| candle       | bf16      | ok     | 9.7     | 7.1     | 139.9        | 8.57          | 1.64         |                                                                                  |
| llamacpp     | f16       | error  | -       | -       | -            | -             | -            | rc=1:                                                                            |
| poot         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: Error: GPU dispatch failed: CUDA launch_kernel failed: DriverError(CUDA_ER |
| transformers | bf16      | ok     | 26.3    | 25.0    | 39.9         | 5.85          | 2.17         |                                                                                  |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                  | File "/root/.local/share/uv/python/cpyth |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 139.9   | 142.2   | 114.9    | 43.4     |
| transformers | 39.9    | 39.6    | 39.9     | 40.0     |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 7.1     | 7.0     | 8.7      | 23.1     |
| transformers | 25.0    | 25.3    | 25.1     | 25.0     |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
