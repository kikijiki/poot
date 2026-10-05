# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: None
- captured: 2026-06-14T16:52:49.945905+00:00

## qwen2.5-0.5b-gptq - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | -------------------------------------------------------------------------------- | ---------------------------------------- |
| llamacpp     | q4_k_m    | error  | -       | -       | -            | -             | -            | rc=1:                                                                            |
| poot         | gptq-int4 | error  | -       | -       | -            | -             | -            | rc=1: Error: GPU dispatch failed: CUDA launch_kernel failed: DriverError(CUDA_ER |
| transformers | gptq-int4 | error  | -       | -       | -            | -             | -            | rc=1: hf_quantizer = AutoHfQuantizer.from_config(                                | ^^^^^                                    |
| vllm         | gptq-int4 | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                  | File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
