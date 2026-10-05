# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 7507b10
- captured: 2026-06-17T10:43:26.375718+00:00

## qwen3-4b - decode-128

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                 |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| candle       | bf16      | ok     | 15.2    | 14.1    | 70.9         | 8.32          | 8.07         |                                                                                                                                                                                                                       |
| llamacpp     | f16       | ok     | 12.3    | 10.5    | 95.1         | 8.34          | 7.80         | TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text) |
| poot         | bf16      | ok     | 13649.0 | 56.5    | 17.7         | 18.02         | 33.18        |                                                                                                                                                                                                                       |
| transformers | bf16      | ok     | 51.4    | 40.2    | 24.8         | 8.30          | 8.22         |                                                                                                                                                                                                                       |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                       | File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
