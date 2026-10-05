# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) (driver ?, ?)
- poot commit: 6794ff7b-dirty
- captured: 2026-09-08T07:00:00+00:00

## phi-3.5-mini - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot | bf16 | error | - | - | - | - | - | - | - | timeout: killed after ~1200s (20 min) of actual runtime with the decode-curve in |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, null elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
