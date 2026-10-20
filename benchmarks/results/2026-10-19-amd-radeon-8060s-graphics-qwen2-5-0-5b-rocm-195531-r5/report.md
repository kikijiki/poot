# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (driver None, None)
- captured: 2026-10-19T19:55:58.171705+00:00

- poot build: 2f639c88

## qwen2.5-0.5b - decode-128

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (rocm) | bf16 | ok | 9.4 | 6.1 | 163.9 | - | 1.32 | - | - | poot loads bf16 but computes f32 (engine is f32 end to end) |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._