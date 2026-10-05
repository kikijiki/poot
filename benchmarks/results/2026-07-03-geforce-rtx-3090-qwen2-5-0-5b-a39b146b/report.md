# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: a39b146b
- captured: 2026-07-03T11:21:30.480398+00:00

## qwen2.5-0.5b - decode-128

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 11.4 | 9.0 | 111.4 | 1.69 | 1.59 |  |
| llamacpp | f16 | ok | 3.2 | 2.2 | 461.5 | 1.74 | 1.22 | GGUF f16 vs bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); precision f16 != matched target bf16; TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time |
| poot | bf16 | ok | 123.2 | 17.4 | 57.5 | 3.22 | 6.86 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 32.0 | 21.3 | 46.9 | 1.70 | 1.62 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._