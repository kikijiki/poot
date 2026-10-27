# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T15:43:44.075547+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 40.4 | 3.6 | 280.8 | 9.10 | 1.37 | 458.5 | 1.787 |  |
| llamacpp (cuda) | f16 | ok | 19.0 | 1.6 | 606.2 | 3.65 | 1.71 | 335.7 | 0.854 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 42.1 | 3.5 | 289.6 | 4.21 | 0.59 | 357.1 | 0.950 |  |
| transformers (cuda) | bf16 | ok | 9.7 | 8.7 | 114.4 | 6.48 | 2.15 | 234.7 | 0.882 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 280.8 | 198.2 | 85.7 |
| llamacpp (cuda) | 606.2 | 550.7 | 428.5 |
| poot (ptx) | 289.6 | 175.7 | 172.9 |
| transformers (cuda) | 114.4 | 116.5 | 115.7 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.6 | 5.0 | 11.7 |
| llamacpp (cuda) | 1.6 | 1.8 | 2.3 |
| poot (ptx) | 3.5 | 5.7 | 5.8 |
| transformers (cuda) | 8.7 | 8.6 | 8.6 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._