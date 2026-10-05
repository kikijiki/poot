# poot cross-backend benchmark - Strix Halo (card 124)

- SoC: AMD Ryzen AI Max+ 395, GPU Radeon 8060S (gfx1151, RDNA3.5), 128 GiB unified
- poot commit: 2cf7758
- captured: 2026-07-01
- metrics: poot engine timing only (no vendor sysfs/NVML util sampling)

## qwen2.5-0.5b - decode (f32, prompt 5, gen 40, 1 warmup + 2 iters)

| backend | GPU | precision | decode tok/s | TPOT ms | TTFT ms | notes |
|---|---|---|---|---|---|---|
| rocm (raw HSA) | Radeon 8060S (gfx1151) | f32 | 27.1 | 36.8 | 331 | fastest on the Strix Halo iGPU |
| wgpu (Vulkan/RADV) | Radeon 8060S (gfx1151) | f32 | 18.2 | 55.1 | 278 | cached re-encode/submit |
| ptx (NVIDIA) | NVIDIA L40S | f32 | 94.6 | 10.6 | 82 | DIFFERENT GPU (rented pod) - cross-hardware, not same-chip |

Two separate comparisons:

- Same chip (Radeon 8060S / Strix Halo iGPU): ROCm eager decode is ~1.5x the wgpu decode rate.
- The PTX row runs on an NVIDIA L40S, a much larger GPU than the iGPU, so its 94.6 tok/s reflects the
  hardware, not backend efficiency. It confirms the PTX backend works end to end and produces the same f32
  decode; it is not a chip-for-chip comparison against ROCm/wgpu.

Absolute numbers on the Strix Halo rows are noisy (shared box, a llama.cpp server runs alongside); the
ROCm/wgpu ratio within one session is the reliable signal.
