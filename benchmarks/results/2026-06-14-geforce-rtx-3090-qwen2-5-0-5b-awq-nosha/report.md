# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: None
- captured: 2026-06-14T17:01:09.389835+00:00

## qwen2.5-0.5b-awq - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | -------------------------------------------------------------------------------- | ---------------------------------------- |
| poot         | awq-int4  | error  | -       | -       | -            | -             | -            | rc=1: Error: GPU dispatch failed: CUDA launch_kernel failed: DriverError(CUDA_ER |
| transformers | awq-int4  | error  | -       | -       | -            | -             | -            | rc=1: File "/opt/tf-venv/lib/python3.11/site-packages/transformers/modeling_ut   |
| vllm         | awq-int4  | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                  | File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
