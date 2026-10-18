# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (driver None, None)
- captured: 2026-10-18T07:38:48.147738+00:00

- poot build: 4c8945ee

## qwen3.8-27b-fp8 - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (wgpu) | bf16 | ok | 30654.2 | 334.8 | 3.0 | - | 23.16 | - | - | the runner computes in the activation dtype; the FP8 projections are read as stored (E4M3 payload, BF16 block scales) |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 |
|---|---|---|---|
| poot (wgpu) | 3.0 | 2.9 | 2.9 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 |
|---|---|---|---|
| poot (wgpu) | 334.8 | 339.5 | 340.8 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._