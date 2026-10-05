# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 319c6ca3
- captured: 2026-08-20T19:19:03.021994+00:00

## qwen2.5-0.5b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 8.4 | 6.9 | 144.9 | 9.54 | 1.55 | 345.8 | 4.055 |  |
| llamacpp | f16 | ok | 5.3 | 2.2 | 464.6 | 2.06 | 1.22 | 366.1 | 1.176 | GGUF f16 vs bf16; precision f16 != matched target bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text) |
| poot | bf16 | ok | 140.2 | 13.5 | 74.0 | 11.08 | 6.83 | 341.4 | 14.799 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 15.7 | 14.8 | 67.8 | 4.81 | 1.61 | 257.3 | 2.654 |  |
| vllm | bf16 | error | - | - | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 144.9 | 144.5 | 145.6 | 65.4 |
| llamacpp | 464.6 | 466.3 | 462.2 | 428.4 |
| poot | 74.0 | 64.8 | 43.1 | 16.5 |
| transformers | 67.8 | 67.2 | 66.7 | 66.3 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 6.9 | 6.9 | 6.9 | 15.3 |
| llamacpp | 2.2 | 2.1 | 2.2 | 2.3 |
| poot | 13.5 | 15.4 | 23.2 | 60.6 |
| transformers | 14.8 | 14.9 | 15.0 | 15.1 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, null elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._