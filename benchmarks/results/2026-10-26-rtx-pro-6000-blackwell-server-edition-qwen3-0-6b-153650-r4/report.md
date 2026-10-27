# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T15:47:13.166038+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 39.0 | 3.2 | 311.9 | 9.01 | 1.33 | 462.5 | 1.375 |  |
| llamacpp (cuda) | f16 | ok | 18.9 | 1.6 | 618.1 | 3.65 | 1.71 | 339.4 | 0.499 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 41.9 | 3.4 | 298.0 | 4.21 | 0.57 | 338.8 | 0.828 |  |
| transformers (cuda) | bf16 | ok | 9.6 | 9.1 | 110.3 | 6.48 | 2.15 | 236.9 | 0.801 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 311.9 | 207.0 | 88.3 |
| llamacpp (cuda) | 618.1 | 562.3 | 436.0 |
| poot (ptx) | 298.0 | 182.0 | 178.8 |
| transformers (cuda) | 110.3 | 111.4 | 110.7 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.2 | 4.8 | 11.3 |
| llamacpp (cuda) | 1.6 | 1.8 | 2.3 |
| poot (ptx) | 3.4 | 5.5 | 5.6 |
| transformers (cuda) | 9.1 | 9.0 | 9.0 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._