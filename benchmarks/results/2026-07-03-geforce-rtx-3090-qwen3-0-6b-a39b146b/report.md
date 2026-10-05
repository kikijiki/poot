# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: a39b146b
- captured: 2026-07-03T11:24:30.676467+00:00

## qwen3-0.6b - decode-128

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 10.6 | 8.0 | 124.0 | 1.91 | 1.65 |  |
| llamacpp | f16 | ok | 3.6 | 2.6 | 385.5 | 1.95 | 1.70 | precision f16 != matched target bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); GGUF f16 vs bf16 - closest full-precision GGUF; TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time |
| poot | bf16 | ok | 168.4 | 22.0 | 45.4 | 3.58 | 9.75 |  |
| transformers | bf16 | ok | 43.2 | 31.0 | 32.1 | 2.20 | 2.17 |  |
| vllm | bf16 | ok | 9.1 | 2.5 | 397.5 | 8.42 | 3.60 | TTFT is estimated (prefill-only request); vLLM V1 offline has no per-request metrics; vLLM pre-allocates a KV pool at gpu_memory_utilization=0.3; external peak VRAM reflects the reservation, not demand |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._