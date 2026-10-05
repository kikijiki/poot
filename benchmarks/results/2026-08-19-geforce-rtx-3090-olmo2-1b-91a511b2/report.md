# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: 91a511b2
- captured: 2026-08-19T22:42:31.290620+00:00

## olmo2-1b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| candle | bf16 | error | - | - | - | - | - | rc=1: Error: candle runner: unsupported arch "Olmo2ForCausalLM" |
| llamacpp | f16 | ok | 8.2 | 3.6 | 279.4 | 4.37 | 3.05 | llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); GGUF f16 vs bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; precision f16 != matched target bf16 |
| poot | bf16 | ok | 290.6 | 25.5 | 39.2 | 11.08 | 18.63 | isl 8192+ omitted: trace-time shape error in slice ax=0 0..8192: slice 0..8192 out of range for axis of length 4096 |
| transformers | bf16 | ok | 15.5 | 14.7 | 67.9 | 7.02 | 3.41 |  |
| vllm | bf16 | error | - | - | - | - | - | rc=1:   File "/opt/vllm-venv/lib/python3.11/site-packages/vllm/engine/arg_utils. |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 279.4 | 272.8 | 255.6 | 206.4 |
| poot | 39.2 | 33.3 | 21.3 | - |
| transformers | 67.9 | 69.1 | 68.0 | 69.5 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 3.6 | 3.7 | 3.9 | 4.8 |
| poot | 25.5 | 30.0 | 46.9 | - |
| transformers | 14.7 | 14.5 | 14.7 | 14.4 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._