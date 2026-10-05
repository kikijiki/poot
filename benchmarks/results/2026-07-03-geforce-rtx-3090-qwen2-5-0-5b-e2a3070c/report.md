# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: e2a3070c
- captured: 2026-07-03T09:07:31.201857+00:00

## qwen2.5-0.5b - decode-128

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 9.7 | 7.6 | 131.5 | 1.69 | 1.59 |  |
| llamacpp | f16 | ok | 3.2 | 2.3 | 440.7 | 1.74 | 1.22 | TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); GGUF f16 vs bf16; precision f16 != matched target bf16 |
| poot | bf16 | ok | 159.1 | 22.7 | 44.1 | 3.22 | 6.89 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 28.5 | 23.6 | 42.3 | 1.70 | 1.58 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._