# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T14:37:54.147209+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 39.5 | 3.4 | 293.5 | 7.98 | 1.53 | 478.8 | 1.854 |  |
| llamacpp (cuda) | f16 | ok | 18.7 | 1.7 | 599.1 | 3.65 | 1.71 | 394.6 | 0.943 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 40.4 | 3.4 | 295.0 | 4.21 | 0.62 | 373.7 | 0.926 |  |
| transformers (cuda) | bf16 | ok | 10.0 | 9.6 | 104.3 | 6.48 | 2.27 | 251.8 | 0.887 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 293.5 | 202.2 | 85.5 |
| llamacpp (cuda) | 599.1 | 553.1 | 431.6 |
| poot (ptx) | 295.0 | 179.7 | 175.7 |
| transformers (cuda) | 104.3 | 103.6 | 109.0 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.4 | 4.9 | 11.7 |
| llamacpp (cuda) | 1.7 | 1.8 | 2.3 |
| poot (ptx) | 3.4 | 5.6 | 5.7 |
| transformers (cuda) | 9.6 | 9.7 | 9.2 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._