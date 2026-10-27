# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T15:48:12.286297+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 38.4 | 3.4 | 295.4 | 9.07 | 1.40 | 467.6 | 1.393 |  |
| llamacpp (cuda) | f16 | ok | 18.4 | 1.6 | 621.9 | 3.65 | 1.71 | 344.9 | 0.495 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 41.7 | 3.4 | 295.7 | 4.21 | 0.79 | 339.0 | 0.846 |  |
| transformers (cuda) | bf16 | ok | 9.2 | 8.7 | 114.9 | 6.48 | 2.23 | 222.3 | 0.686 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 295.4 | 206.9 | 88.3 |
| llamacpp (cuda) | 621.9 | 562.5 | 438.2 |
| poot (ptx) | 295.7 | 181.8 | 179.0 |
| transformers (cuda) | 114.9 | 116.4 | 116.8 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.4 | 4.8 | 11.3 |
| llamacpp (cuda) | 1.6 | 1.8 | 2.3 |
| poot (ptx) | 3.4 | 5.5 | 5.6 |
| transformers (cuda) | 8.7 | 8.6 | 8.6 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._