# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) (driver None, None)
- poot commit: 6b64abcb2
- captured: 2026-09-24T23:45:01.526010+00:00

## olmo2-1b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot | bf16 | ok | 1929.4 | 62.8 | 15.9 | - | 18.55 | - | - | isl 8192+ omitted: OLMo-2-0425-1B-Instruct max_position_embeddings is 4096, so isl past 2048 exceeds the model window (not a poot bug). PTX (RunPod RTX 3090, 2026-08-20, snapshot 2026-08-19-geforce-rtx-3090-olmo2-1b-91a511b2): isl 128/512/2048 ok. On-box AMD Strix Halo gfx1151, 2026-09-25, commit 6b64abcb2, HF revision 48d788eca847d4d7548f375ad03d3c9312f6139e, both backends run serially under flock with the user's vLLM services stopped: rocm full curve ok at isl 128/512/2048 (15.9/15.1/12.3 tok/s, TPOT p50 62.8/66.0/81.3 ms, peak host RSS 18.55 GiB; snapshot 2026-09-25-strix-halo-olmo2-1b-rocm-6b64abcb2) and wgpu full curve ok (12.4/10.8/10.5 tok/s, TPOT p50 80.6/92.3/95.0 ms, peak host RSS 18.53 GiB; snapshot 2026-09-25-strix-halo-olmo2-1b-wgpu-6b64abcb2). Both are card-312 contract ok, so every iteration delivered the full 128-token budget: the ROCm 2-token EOS stop is fixed (11743c3c8 seeds the post-prefill decode capture from the prefill KV) and wgpu isl 2048 no longer trips the WebGPU 65535/dim batched-prefill grid cap (78b71acfd). Superseded: 2026-09-23 at 7e71ad05a recorded rocm ok at ~0.45 tok/s (snapshot 2026-09-23-strix-halo-olmo2-1b-rocm-7e71ad05) and a wgpu error row at isl 2048 (snapshot 2026-09-23-strix-halo-olmo2-1b-wgpu-7e71ad05); the card 298 mixed-dtype planning fix (update 1176) is confirmed on both backends. Host memory: the cell peaks near 18.5 GiB RSS plus GPU device buffers that never show up in process RSS, so with ~70G of vLLM device memory resident (30Gi available) an earlier 2026-09-25 attempt reached a global OOM at isl 2048 - it needs >=40Gi available, or >=80Gi while other lanes are building. isl 8192+ stays omitted for the model window. |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 |
|---|---|---|---|
| poot | 15.9 | 15.1 | 12.3 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 |
|---|---|---|---|
| poot | 62.8 | 66.0 | 81.3 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
