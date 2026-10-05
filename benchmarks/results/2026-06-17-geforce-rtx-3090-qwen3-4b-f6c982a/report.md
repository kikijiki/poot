# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: f6c982a
- captured: 2026-06-17T11:16:29.984052+00:00

## qwen3-4b - prefill-1k

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                 |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| candle       | bf16      | ok     | 260.3   | 18.4    | 38.3         | 8.79          | 8.08         |                                                                                                                                                                                                                       |
| llamacpp     | f16       | ok     | 141.2   | 10.8    | 92.2         | 8.72          | 7.80         | precision f16 != matched target bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; GGUF f16 vs bf16 |
| poot         | bf16      | ok     | 66945.5 | 103.2   | 9.7          | 19.64         | 33.19        |                                                                                                                                                                                                                       |
| transformers | bf16      | ok     | 185.8   | 38.5    | 23.2         | 8.55          | 8.24         |                                                                                                                                                                                                                       |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: with launch_core_engines(                                                                                                                                                                                       | File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
