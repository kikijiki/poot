# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-24T08:23:03.035612+00:00

- poot build: d4653ef3

## qwen3-4b-q4km - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp (cuda) | q4_k_m | ok | 10.7 | 2.8 | 357.0 | 5.03 | 2.63 | 600.6 | 1.084 | the same quant file as poot; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; q4_k_m is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only |
| poot (ptx) | q4_k | ok | 188.5 | 11.7 | 85.5 | 7.81 | 2.90 | 528.7 | 13.184 | Q4_K_M stores Q4_K and Q6_K tensors (Q6_K for half of the attention-V and FFN-down weights); the row's precision is the dominant one; precision q4_k != matched target q4_k_m |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 357.0 | 329.7 | 277.2 |
| poot (ptx) | 85.5 | 72.6 | 54.4 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 2.8 | 3.0 | 3.6 |
| poot (ptx) | 11.7 | 13.8 | 18.4 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._