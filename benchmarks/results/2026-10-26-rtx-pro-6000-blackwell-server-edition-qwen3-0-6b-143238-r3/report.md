# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T14:40:29.024009+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 39.7 | 3.4 | 295.0 | 9.19 | 1.39 | 456.1 | 1.447 |  |
| llamacpp (cuda) | f16 | ok | 18.7 | 1.6 | 611.7 | 3.65 | 1.71 | 336.0 | 0.534 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 40.3 | 3.4 | 294.1 | 4.21 | 0.57 | 341.4 | 0.913 |  |
| transformers (cuda) | bf16 | ok | 9.2 | 8.8 | 113.5 | 6.48 | 2.09 | 230.4 | 0.842 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 295.0 | 204.2 | 88.7 |
| llamacpp (cuda) | 611.7 | 554.4 | 434.1 |
| poot (ptx) | 294.1 | 179.2 | 176.6 |
| transformers (cuda) | 113.5 | 113.9 | 112.6 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.4 | 4.9 | 11.3 |
| llamacpp (cuda) | 1.6 | 1.8 | 2.3 |
| poot (ptx) | 3.4 | 5.6 | 5.7 |
| transformers (cuda) | 8.8 | 8.8 | 8.9 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._