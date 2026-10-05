# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 67e70970
- captured: 2026-07-03T09:32:56.879838+00:00

## qwen3-0.6b - prefill-1k

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 65.9 | 7.3 | 109.2 | 2.26 | 1.66 |  |
| llamacpp | f16 | ok | 38.8 | 2.6 | 391.7 | 2.29 | 1.70 | TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; precision f16 != matched target bf16; GGUF f16 vs bf16 - closest full-precision GGUF; llama-bench uses a synthetic prompt (throughput, not the literal prompt text) |
| poot | bf16 | ok | 151882.3 | 146.5 | 6.8 | 4.14 | 9.73 |  |
| transformers | bf16 | ok | 38.8 | 33.1 | 30.0 | 2.34 | 1.96 |  |
| vllm | bf16 | ok | 9.2 | 2.5 | 370.8 | 8.42 | 3.54 | TTFT is estimated (prefill-only request); vLLM V1 offline has no per-request metrics; vLLM pre-allocates a KV pool at gpu_memory_utilization=0.3; external peak VRAM reflects the reservation, not demand |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._