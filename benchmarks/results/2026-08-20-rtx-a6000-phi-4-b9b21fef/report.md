# poot benchmark report

- GPU: NVIDIA RTX A6000 (driver 580.95.05, 49140 MiB)
- poot commit: b9b21fef
- captured: 2026-08-20T16:41:18.062085+00:00

## phi-4 - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes |
|---|---|---|---|---|---|---|---|---|
| llamacpp | f16 | ok | 57.9 | 41.6 | 24.0 | 29.13 | 27.59 | llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); GGUF f16 vs bf16; precision f16 != matched target bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| transformers | bf16 | ok | 55.1 | 47.1 | 21.2 | 33.24 | 27.98 |  |
| vllm | bf16 | ok | 62.1 | 51.6 | 19.4 | 38.10 | 6.80 | vLLM KV pool at gpu_memory_utilization=0.85 inflates external VRAM; prefix caching disabled; per-token ITL from AsyncLLM streaming; re-run manually (poot-orchestrator exec, not the automated sweep) with VLLM_USE_FLASHINFER_SAMPLER=0 (update 0629 fix, not yet baked into the published bench image) and --gpu-mem-util 0.85 (runners.toml default 0.30 is too low for this model's ~28 GiB bf16 weights - update 0639 finding); same GPU class (RTX A6000 48GB) and image as the automated sweep, different pod |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 24.0 | 23.9 | 23.6 | 22.7 |
| transformers | 21.2 | 21.1 | 20.4 | 18.0 |
| vllm | 19.4 | 19.3 | 19.1 | 18.3 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 41.6 | 41.8 | 42.3 | 44.1 |
| transformers | 47.1 | 47.4 | 49.1 | 55.5 |
| vllm | 51.6 | 51.8 | 52.5 | 54.6 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
