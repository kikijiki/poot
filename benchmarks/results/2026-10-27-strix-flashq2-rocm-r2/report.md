# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (driver None, None)
- captured: 2026-10-27T02:12:20.571069+00:00

- poot build: 310148fc

## qwen3.8-flash-next-q2 - decode-curve-large

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (rocm) | iq4_nl | ok | 1312.9 | 50.9 | 19.6 | 94.50 | 31.83 | 125.6 | 8.673 | UD-Q2_K_XL stores IQ4_NL, IQ2_XS, Q5_K, Q8_0 and smaller shares of other types; the row's precision is the dominant one; precision iq4_nl != matched target q2_k_xl |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| poot (rocm) | 19.6 | 18.0 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| poot (rocm) | 50.9 | 55.7 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._