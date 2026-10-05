# poot benchmark report

- GPU: NVIDIA L40S (driver 580.159.03, 46068 MiB)
- poot commit: 474f03b2
- captured: 2026-08-20T15:50:19.503139+00:00

## qwen3-14b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | error | - | - | - | - | - | rc=1: candle decode-curve: osl=128 isls=[128, 512, 2048, 8192] synthetic=true vo |
| llamacpp | f16 | ok | 57.4 | 38.8 | 25.8 | 28.75 | 27.81 | precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; GGUF f16 vs bf16 |
| poot | bf16 | error | - | - | - | - | - | rc=-9:  |
| transformers | bf16 | ok | 55.4 | 43.4 | 23.0 | 33.98 | 28.28 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 25.8 | 25.7 | 25.5 | 24.5 |
| transformers | 23.0 | 22.9 | 22.5 | 20.8 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 38.8 | 38.9 | 39.3 | 40.8 |
| transformers | 43.4 | 43.7 | 44.5 | 48.2 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._