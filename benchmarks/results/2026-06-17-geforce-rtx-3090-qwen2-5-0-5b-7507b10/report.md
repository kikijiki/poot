# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 7507b10
- captured: 2026-06-17T10:37:58.387029+00:00

## qwen2.5-0.5b - decode-128

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                 |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| candle       | bf16      | ok     | 10.8    | 8.5     | 117.5        | 1.69          | 1.60         |                                                                                                                                                                                                                       |
| llamacpp     | f16       | ok     | 3.2     | 2.2     | 454.6        | 1.74          | 1.22         | GGUF f16 vs bf16; TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); precision f16 != matched target bf16 |
| poot         | bf16      | ok     | 1031.9  | 11.3    | 88.6         | 3.25          | 4.96         | poot loads bf16 but computes f32 (engine is f32 end to end)                                                                                                                                                           |
| transformers | bf16      | ok     | 32.0    | 21.0    | 47.7         | 1.70          | 1.62         |                                                                                                                                                                                                                       |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                       | File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
