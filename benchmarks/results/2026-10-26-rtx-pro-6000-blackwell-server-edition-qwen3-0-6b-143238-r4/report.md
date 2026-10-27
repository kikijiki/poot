# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T14:41:28.764336+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 39.9 | 3.6 | 280.0 | 9.07 | 1.34 | 455.9 | 1.464 |  |
| llamacpp (cuda) | f16 | ok | 18.8 | 1.6 | 613.4 | 3.65 | 1.71 | 336.6 | 0.518 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 43.4 | 3.4 | 292.9 | 4.21 | 0.57 | 340.7 | 0.909 |  |
| transformers (cuda) | bf16 | ok | 9.5 | 9.0 | 111.3 | 6.48 | 2.28 | 232.2 | 0.849 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 280.0 | 202.7 | 88.2 |
| llamacpp (cuda) | 613.4 | 556.2 | 434.4 |
| poot (ptx) | 292.9 | 179.1 | 176.8 |
| transformers (cuda) | 111.3 | 112.2 | 112.0 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.6 | 4.9 | 11.3 |
| llamacpp (cuda) | 1.6 | 1.8 | 2.3 |
| poot (ptx) | 3.4 | 5.6 | 5.7 |
| transformers (cuda) | 9.0 | 8.9 | 8.9 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._