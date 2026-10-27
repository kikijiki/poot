# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T15:07:53.040677+00:00

- poot build: 9b55b888

## qwen3.8-27b-q4 - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp (cuda) | q4_k_xl | ok | 51.7 | 14.8 | 67.4 | 17.78 | 16.69 | 538.3 | 4.212 | the same quant file as poot; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; q4_k_xl is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only |
| poot (ptx) | q5_k | ok | 533.3 | 26.3 | 38.0 | 26.38 | 1.74 | 496.9 | 13.885 | UD-Q4_K_XL stores Q5_K, IQ4_XS, Q4_K, Q6_K and smaller shares of other types; the row's precision is the dominant one; precision q5_k != matched target q4_k_xl |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 67.4 | 67.1 | 65.3 |
| poot (ptx) | 38.0 | 33.8 | 33.0 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 14.8 | 14.9 | 15.3 |
| poot (ptx) | 26.3 | 29.6 | 30.3 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._