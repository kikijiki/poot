# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 7507b10
- captured: 2026-06-17T10:40:29.150445+00:00

## qwen3-0.6b - decode-128

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                               |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| candle       | bf16      | ok     | 10.7    | 8.1     | 123.0        | 1.91          | 1.70         |                                                                                                                                                                                                                                                     |
| llamacpp     | f16       | ok     | 3.6     | 2.6     | 384.3        | 1.95          | 1.70         | TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; precision f16 != matched target bf16; llama-bench uses a synthetic prompt (throughput, not the literal prompt text); GGUF f16 vs bf16 - closest full-precision GGUF |
| poot         | bf16      | ok     | 1268.5  | 12.7    | 78.5         | 3.75          | 6.17         |                                                                                                                                                                                                                                                     |
| transformers | bf16      | ok     | 42.8    | 30.9    | 32.2         | 2.20          | 2.06         |                                                                                                                                                                                                                                                     |
| vllm         | bf16      | ok     | 9.5     | 2.5     | 395.4        | 8.42          | 3.65         | TTFT is estimated (prefill-only request); vLLM V1 offline has no per-request metrics; vLLM pre-allocates a KV pool at gpu_memory_utilization=0.3; external peak VRAM reflects the reservation, not demand                                           |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
