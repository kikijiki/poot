# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: ded22c11
- captured: 2026-07-28T05:10:18.186334+00:00

## qwen3-4b - decode-128

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 15.3 | 14.1 | 70.7 | 8.32 | 8.23 |  |
| llamacpp | f16 | ok | 12.1 | 10.5 | 94.8 | 8.34 | 7.80 | GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time |
| poot | bf16 | ok | 293.6 | 42.2 | 23.7 | 17.33 | 48.07 |  |
| transformers | bf16 | ok | 52.2 | 40.1 | 24.9 | 8.30 | 8.21 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._