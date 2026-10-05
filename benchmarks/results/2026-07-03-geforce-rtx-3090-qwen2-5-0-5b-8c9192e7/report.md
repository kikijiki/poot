# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: 8c9192e7
- captured: 2026-07-03T08:33:30.230510+00:00

## qwen2.5-0.5b - decode-128

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 10.9 | 8.5 | 117.0 | 1.69 | 1.57 |  |
| llamacpp | f16 | ok | 3.2 | 2.2 | 462.6 | 1.74 | 1.22 | llama-bench uses a synthetic prompt (throughput, not the literal prompt text); precision f16 != matched target bf16; TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; GGUF f16 vs bf16 |
| poot | bf16 | error | - | - | - | - | - | rc=1: Error: generate |  | Caused by: |     0: ptx capture_decode: ptx runtime:  |
| transformers | bf16 | ok | 31.9 | 21.8 | 45.6 | 1.70 | 1.64 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._