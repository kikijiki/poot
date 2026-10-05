# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: 1f63ed0a
- captured: 2026-08-20T05:13:53.923490+00:00

## qwen2.5-0.5b - prefill-1k

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 49.5 | 9.2 | 94.7 | 1.85 | 1.53 |  |
| llamacpp | f16 | ok | 29.8 | 2.2 | 462.0 | 1.98 | 1.22 | llama-bench uses a synthetic prompt (throughput, not the literal prompt text); TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; precision f16 != matched target bf16; GGUF f16 vs bf16 |
| poot | bf16 | ok | 1677.2 | 20.4 | 49.1 | 4.73 | 6.88 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 35.7 | 22.0 | 44.5 | 1.75 | 1.58 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._