# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (driver None, None)
- captured: 2026-10-26T16:47:28.597569+00:00

- poot build: f2936210

## qwen3-4b-q4km - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (rocm) | q4_k | ok | 236.2 | 17.0 | 58.9 | 7.11 | 0.61 | 111.0 | 3.741 | Q4_K_M stores Q4_K and Q6_K tensors (Q6_K for half of the attention-V and FFN-down weights); the row's precision is the dominant one; precision q4_k != matched target q4_k_m |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| poot (rocm) | 58.9 | 42.9 | 28.6 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| poot (rocm) | 17.0 | 23.3 | 34.9 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._