# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: 78ca82f2
- captured: 2026-07-28T04:33:19.077381+00:00

## qwen2.5-0.5b - decode-128

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 9.9 | 7.5 | 132.7 | 1.69 | 1.57 |  |
| llamacpp | f16 | ok | 3.3 | 2.4 | 423.8 | 1.74 | 1.22 | precision f16 != matched target bf16; TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; GGUF f16 vs bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text) |
| poot | bf16 | ok | 73.1 | 9.9 | 100.8 | 3.22 | 6.87 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 26.0 | 20.1 | 49.5 | 1.70 | 1.57 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._