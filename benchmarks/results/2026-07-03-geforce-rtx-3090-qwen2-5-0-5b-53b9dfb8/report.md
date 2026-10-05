# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: 53b9dfb8
- captured: 2026-07-03T10:02:40.641742+00:00

## qwen2.5-0.5b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 9.2 | 7.3 | 136.3 | 9.44 | 1.52 |  |
| llamacpp | f16 | ok | 5.5 | 2.4 | 425.0 | 2.06 | 1.22 | GGUF f16 vs bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); precision f16 != matched target bf16 |
| poot | bf16 | ok | 119.4 | 31.8 | 31.5 | 11.19 | 6.91 | poot loads bf16 but computes f32 (engine is f32 end to end) |
| transformers | bf16 | ok | 19.7 | 17.8 | 56.2 | 4.81 | 1.53 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 136.3 | 137.3 | 126.4 | 64.9 |
| llamacpp | 425.0 | 456.0 | 450.7 | 419.5 |
| poot | 31.5 | 17.7 | 6.3 | 1.2 |
| transformers | 56.2 | 58.5 | 53.8 | 57.7 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| candle | 7.3 | 7.3 | 7.9 | 15.4 |
| llamacpp | 2.4 | 2.2 | 2.2 | 2.4 |
| poot | 31.8 | 56.5 | 159.0 | 854.0 |
| transformers | 17.8 | 17.1 | 18.6 | 17.3 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._