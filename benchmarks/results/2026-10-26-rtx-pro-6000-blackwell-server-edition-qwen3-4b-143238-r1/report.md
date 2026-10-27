# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T14:44:11.754134+00:00

- poot build: 9b55b888

## qwen3-4b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 127.6 | 9.3 | 107.6 | 19.38 | 7.94 | 519.1 | 5.018 |  |
| llamacpp (cuda) | f16 | ok | 58.3 | 6.4 | 155.8 | 10.30 | 7.80 | 500.3 | 2.227 | GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 110.4 | 8.8 | 113.1 | 11.82 | 1.07 | 407.8 | 3.205 |  |
| transformers (cuda) | bf16 | ok | 12.6 | 12.1 | 82.9 | 13.31 | 8.00 | 361.1 | 1.588 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 107.6 | 73.5 | 35.4 |
| llamacpp (cuda) | 155.8 | 149.0 | 137.8 |
| poot (ptx) | 113.1 | 86.1 | 60.5 |
| transformers (cuda) | 82.9 | 83.7 | 82.9 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 9.3 | 13.6 | 28.2 |
| llamacpp (cuda) | 6.4 | 6.7 | 7.3 |
| poot (ptx) | 8.8 | 11.6 | 16.5 |
| transformers (cuda) | 12.1 | 11.9 | 12.1 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._