# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T14:39:29.042410+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 39.7 | 3.3 | 306.8 | 9.13 | 1.26 | 455.3 | 1.479 |  |
| llamacpp (cuda) | f16 | ok | 18.7 | 1.6 | 615.7 | 3.65 | 1.71 | 335.6 | 0.514 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 40.5 | 3.4 | 292.0 | 4.21 | 0.71 | 343.2 | 0.910 |  |
| transformers (cuda) | bf16 | ok | 9.3 | 8.9 | 112.1 | 6.48 | 2.36 | 234.9 | 0.850 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 306.8 | 201.9 | 88.4 |
| llamacpp (cuda) | 615.7 | 557.3 | 432.9 |
| poot (ptx) | 292.0 | 177.5 | 174.2 |
| transformers (cuda) | 112.1 | 114.0 | 113.8 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.3 | 5.0 | 11.3 |
| llamacpp (cuda) | 1.6 | 1.8 | 2.3 |
| poot (ptx) | 3.4 | 5.6 | 5.7 |
| transformers (cuda) | 8.9 | 8.8 | 8.8 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._