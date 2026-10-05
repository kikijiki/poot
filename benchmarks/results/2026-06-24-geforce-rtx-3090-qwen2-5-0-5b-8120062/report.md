# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.65.06, 24576 MiB)
- poot commit: 8120062
- captured: 2026-06-24T02:42:51.093074+00:00

## qwen2.5-0.5b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 10.8 | 8.8 | 113.6 | 7.91 | 1.54 |  |
| llamacpp | f16 | ok | 5.5 | 2.2 | 449.4 | 2.06 | 1.22 | GGUF f16 vs bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); precision f16 != matched target bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot | bf16 | ok | 165.5 | 14.1 | 71.1 | 11.08 | 4.70 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 20.7 | 19.4 | 51.5 | 4.81 | 1.55 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 113.6 | 113.6 | 112.8 | 67.1 |
| llamacpp | 449.4 | 451.8 | 444.3 | 416.2 |
| poot | 71.1 | 67.3 | 56.7 | 36.0 |
| transformers | 51.5 | 50.9 | 50.9 | 50.9 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 8.8 | 8.8 | 8.9 | 14.9 |
| llamacpp | 2.2 | 2.2 | 2.3 | 2.4 |
| poot | 14.1 | 14.8 | 17.6 | 27.8 |
| transformers | 19.4 | 19.6 | 19.7 | 19.6 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._