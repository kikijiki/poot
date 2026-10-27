# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 580.82.07, 97887 MiB)
- captured: 2026-10-23T14:17:42.954864+00:00

- poot build: 3ba1c127

## qwen3.8-flash-next-q2 - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp | q2_k_xl | ok | 131.3 | 10.1 | 98.6 | 48.46 | 46.57 | 362.5 | 6.828 | the same quant files as poot; whether the image's llama.cpp tag knows qwen4exp is the cell's result; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; q2_k_xl is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only |
| poot (ptx) | iq4_nl | ok | 823.2 | 25.1 | 39.9 | 90.68 | 74.34 | 520.4 | 21.276 | UD-Q2_K_XL stores IQ4_NL, IQ2_XS, Q5_K, Q8_0 and smaller shares of other types; the row's precision is the dominant one; precision iq4_nl != matched target q2_k_xl |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 98.6 | 96.3 | 95.8 | 62.0 |
| poot (ptx) | 39.9 | 39.8 | 39.3 | 39.3 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| llamacpp | 10.1 | 10.4 | 10.4 | 16.1 |
| poot (ptx) | 25.1 | 25.1 | 25.4 | 25.5 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._