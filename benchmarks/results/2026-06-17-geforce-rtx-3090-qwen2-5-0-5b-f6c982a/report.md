# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: f6c982a
- captured: 2026-06-17T11:08:47.185345+00:00

## qwen2.5-0.5b - prefill-1k

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                 |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| candle       | bf16      | ok     | 50.0    | 7.3     | 114.2        | 1.85          | 1.57         |                                                                                                                                                                                                                       |
| llamacpp     | f16       | ok     | 29.1    | 2.4     | 416.2        | 1.98          | 1.22         | TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); precision f16 != matched target bf16; GGUF f16 vs bf16 |
| poot         | bf16      | ok     | 13875.5 | 16.5    | 60.5         | 3.53          | 4.96         | poot loads bf16 but computes f32 (engine is f32 end to end)                                                                                                                                                           |
| transformers | bf16      | ok     | 29.5    | 20.2    | 48.7         | 1.75          | 1.58         |                                                                                                                                                                                                                       |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                       | File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
