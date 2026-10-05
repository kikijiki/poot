# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: f6c982a
- captured: 2026-06-17T11:12:11.078640+00:00

## qwen3-0.6b - prefill-1k

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                               |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| candle       | bf16      | ok     | 68.1    | 6.9     | 114.3        | 2.26          | 1.66         |                                                                                                                                                                                                                                                     |
| llamacpp     | f16       | ok     | 42.1    | 2.8     | 362.3        | 2.29          | 1.70         | llama-bench uses a synthetic prompt (throughput, not the literal prompt text); GGUF f16 vs bf16 - closest full-precision GGUF; TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time; precision f16 != matched target bf16 |
| poot         | bf16      | ok     | 17327.6 | 20.2    | 49.4         | 4.71          | 6.17         |                                                                                                                                                                                                                                                     |
| transformers | bf16      | ok     | 40.1    | 28.4    | 34.8         | 2.34          | 2.18         |                                                                                                                                                                                                                                                     |
| vllm         | bf16      | ok     | 9.5     | 2.6     | 355.8        | 8.42          | 3.56         | vLLM pre-allocates a KV pool at gpu_memory_utilization=0.3; external peak VRAM reflects the reservation, not demand; TTFT is estimated (prefill-only request); vLLM V1 offline has no per-request metrics                                           |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
