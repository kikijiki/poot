# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T14:59:08.505428+00:00

- poot build: 9b55b888

## qwen3-4b-q4km - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp (cuda) | q4_k_m | ok | 12.4 | 3.1 | 319.7 | 5.03 | 2.63 | 487.0 | 0.845 | the same quant file as poot; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; q4_k_m is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only |
| poot (ptx) | q4_k | ok | 135.8 | 6.1 | 163.2 | 6.68 | 0.69 | 433.2 | 2.937 | Q4_K_M stores Q4_K and Q6_K tensors (Q6_K for half of the attention-V and FFN-down weights); the row's precision is the dominant one; precision q4_k != matched target q4_k_m |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 319.7 | 295.4 | 254.0 |
| poot (ptx) | 163.2 | 113.4 | 71.8 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 3.1 | 3.4 | 3.9 |
| poot (ptx) | 6.1 | 8.8 | 13.9 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._