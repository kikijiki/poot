# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 1ece8859
- captured: 2026-08-19T22:09:19.808190+00:00

## phi-3.5-mini - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| llamacpp | f16 | ok | 17.9 | 10.0 | 100.4 | 10.87 | 7.37 | GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot | bf16 | ok | 513.3 | 38.8 | 25.8 | 21.99 | 38.33 | isl 8192+ omitted: ptx batched prefill: ptx runtime: CUDA driver error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, 'out of memory') |
| transformers | bf16 | ok | 27.6 | 21.2 | 47.1 | 14.22 | 7.63 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:     with launch_core_engines( |   File "/root/.local/share/uv/python/cpyth |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 100.4 | 97.3 | 90.1 | 65.5 |
| poot | 25.8 | 21.4 | 13.2 | - |
| transformers | 47.1 | 47.5 | 48.4 | 35.6 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 10.0 | 10.3 | 11.1 | 15.3 |
| poot | 38.8 | 46.7 | 75.9 | - |
| transformers | 21.2 | 21.1 | 20.7 | 28.1 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._