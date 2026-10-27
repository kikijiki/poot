# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-24T08:12:37.212575+00:00

- poot build: 96ddd6f9

## qwen3.8-27b-q4 - decode-curve-large

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp (cuda) | q4_k_xl | ok | 55.1 | 15.1 | 66.2 | 17.78 | 16.70 | 522.5 | 5.458 | the same quant file as poot; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; q4_k_xl is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only |
| poot (ptx) | q5_k | ok | 735.5 | 48.0 | 20.8 | 25.81 | 17.12 | 567.1 | 20.657 | UD-Q4_K_XL stores Q5_K, IQ4_XS, Q4_K, Q6_K and smaller shares of other types; the row's precision is the dominant one; precision q5_k != matched target q4_k_xl |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| llamacpp (cuda) | 66.2 | 45.4 |
| poot (ptx) | 20.8 | 20.1 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| llamacpp (cuda) | 15.1 | 22.1 |
| poot (ptx) | 48.0 | 49.8 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._