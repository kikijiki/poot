# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (driver None, None)
- captured: 2026-10-26T17:04:50.750832+00:00

- poot build: f2936210

## qwen3.8-27b-q4 - decode-curve-large

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (rocm) | q5_k | ok | 1295.6 | 102.0 | 9.8 | 25.18 | 1.40 | 113.9 | 14.161 | UD-Q4_K_XL stores Q5_K, IQ4_XS, Q4_K, Q6_K and smaller shares of other types; the row's precision is the dominant one; precision q5_k != matched target q4_k_xl |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| poot (rocm) | 9.8 | 8.0 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| poot (rocm) | 102.0 | 125.8 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._