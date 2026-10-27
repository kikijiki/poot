# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (driver None, None)
- captured: 2026-10-26T17:56:49.570668+00:00

- poot build: f2936210

## qwen3-4b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (rocm) | bf16 | ok | 249.1 | 39.8 | 25.2 | 12.43 | 1.01 | 115.2 | 4.992 |  |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| poot (rocm) | 25.2 | 21.9 | 17.2 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| poot (rocm) | 39.8 | 45.7 | 58.2 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._