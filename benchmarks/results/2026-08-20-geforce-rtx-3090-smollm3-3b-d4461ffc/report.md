# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: d4461ffc
- captured: 2026-08-20T00:23:53.904063+00:00

## smollm3-3b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| llamacpp | f16 | ok | 19.2 | 8.3 | 121.0 | 7.32 | 6.05 | llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); precision f16 != matched target bf16; GGUF f16 vs bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot | bf16 | ok | 693.3 | 60.7 | 16.5 | 23.96 | 36.46 | isl 2048+ omitted: ptx capture_decode_from: ptx runtime: CUDA driver error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, 'out of memory') |
| transformers | bf16 | ok | 30.8 | 27.0 | 37.1 | 10.32 | 6.45 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 121.0 | 120.6 | 118.3 | 110.3 |
| poot | 16.5 | 14.9 | - | - |
| transformers | 37.1 | 36.2 | 35.9 | 37.5 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 8.3 | 8.3 | 8.5 | 9.1 |
| poot | 60.7 | 66.9 | - | - |
| transformers | 27.0 | 27.6 | 27.9 | 26.7 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._