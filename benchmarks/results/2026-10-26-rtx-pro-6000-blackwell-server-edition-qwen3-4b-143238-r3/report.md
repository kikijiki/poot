# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T14:49:00.833096+00:00

- poot build: 9b55b888

## qwen3-4b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 124.9 | 9.3 | 108.0 | 19.10 | 7.70 | 519.2 | 4.899 |  |
| llamacpp (cuda) | f16 | ok | 59.0 | 6.4 | 155.5 | 10.30 | 7.80 | 499.7 | 2.084 | GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 106.6 | 8.7 | 114.5 | 11.82 | 1.11 | 414.0 | 3.038 |  |
| transformers (cuda) | bf16 | ok | 12.5 | 11.9 | 83.9 | 13.31 | 8.48 | 361.4 | 1.584 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 108.0 | 73.5 | 35.4 |
| llamacpp (cuda) | 155.5 | 149.1 | 137.9 |
| poot (ptx) | 114.5 | 87.3 | 60.5 |
| transformers (cuda) | 83.9 | 84.8 | 82.8 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 9.3 | 13.6 | 28.2 |
| llamacpp (cuda) | 6.4 | 6.7 | 7.3 |
| poot (ptx) | 8.7 | 11.4 | 16.5 |
| transformers (cuda) | 11.9 | 11.8 | 12.1 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._