# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 67e70970
- captured: 2026-07-03T09:20:35.090705+00:00

## qwen2.5-0.5b - prefill-1k

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 48.8 | 6.8 | 121.5 | 1.85 | 1.53 |  |
| llamacpp | f16 | ok | 28.9 | 2.2 | 458.9 | 1.98 | 1.22 | precision f16 != matched target bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); GGUF f16 vs bf16; TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time |
| poot | bf16 | ok | 86572.5 | 83.5 | 12.0 | 3.50 | 6.86 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 31.7 | 18.4 | 53.1 | 1.75 | 1.60 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._