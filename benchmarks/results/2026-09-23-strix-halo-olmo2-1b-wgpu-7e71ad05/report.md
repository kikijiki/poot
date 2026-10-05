# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) (driver None, None)
- poot commit: 7e71ad05a
- captured: 2026-09-23T07:18:38.214009+00:00

## olmo2-1b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot | bf16 | error | - | - | - | - | - | - | - | invalid runner result: curve ISL coverage mismatch: requested [128, 512, 2048],  |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
