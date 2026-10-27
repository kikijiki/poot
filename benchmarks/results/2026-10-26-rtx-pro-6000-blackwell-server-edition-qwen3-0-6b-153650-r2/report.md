# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T15:45:16.771829+00:00

- poot build: 9b55b888

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 40.4 | 3.2 | 310.0 | 9.07 | 1.42 | 462.2 | 1.415 |  |
| llamacpp (cuda) | f16 | ok | 18.9 | 1.6 | 622.3 | 3.65 | 1.71 | 342.0 | 0.512 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 43.8 | 3.5 | 288.3 | 4.21 | 0.69 | 336.5 | 0.895 |  |
| transformers (cuda) | bf16 | ok | 9.4 | 8.5 | 118.1 | 6.48 | 1.79 | 237.7 | 0.874 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 310.0 | 203.3 | 88.4 |
| llamacpp (cuda) | 622.3 | 559.6 | 440.5 |
| poot (ptx) | 288.3 | 176.0 | 178.0 |
| transformers (cuda) | 118.1 | 118.5 | 118.7 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.2 | 4.9 | 11.3 |
| llamacpp (cuda) | 1.6 | 1.8 | 2.3 |
| poot (ptx) | 3.5 | 5.7 | 5.6 |
| transformers (cuda) | 8.5 | 8.4 | 8.4 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._