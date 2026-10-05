# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) (driver None, None)
- poot commit: 7e71ad05a
- captured: 2026-09-23T07:30:15.569405+00:00

## olmo2-1b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot | bf16 | ok | 2855.6 | 2228.4 | 0.4 | - | 18.53 | - | - | isl 8192+ omitted: OLMo-2-0425-1B-Instruct's own max_position_embeddings is 4096, so this bench's isl list exceeds the model's real context window past isl=2048 (not a poot bug); isl 128/512/2048 all pass with real decode-curve data on PTX (RunPod RTX 3090, 2026-08-20, snapshot 2026-08-19-geforce-rtx-3090-olmo2-1b-91a511b2). On-box AMD Strix Halo (gfx1151): the wgpu/rocm mixed-dtype MatMul planning failure recorded here on 2026-09-08 is FIXED (card 298, update 1176, retype_wide_bf16_consts) - olmo2-1b is confirmed to prefill and decode coherently on both wgpu and rocm (poot-llm card298_olmo2_bf16_gpu, real hardware). The wgpu decode-curve bench cell itself is now blocked by a DIFFERENT, non-engine constraint: host memory. This box's shared unified memory has the user's own ComfyUI (~27.5 GB RSS) and llama-server (~6.7 GB RSS) resident; poot-bench-runner loading olmo2-1b bf16 safetensors and running decode-curve grew to ~18.5 GB RSS before being killed to prevent a system-wide OOM (available RAM and swap both hit ~0 during the attempt, 2026-09-08, commit a7c92517). rocm was not attempted this round given the same memory ceiling. This is a host-capacity gap, not a poot correctness or planning bug - re-run when this box's other GPU workloads are stopped, or on a host with more free RAM. |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 |
|---|---|---|---|
| poot | 0.4 | 0.4 | 0.4 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 |
|---|---|---|---|
| poot | 2228.4 | 2237.5 | 2254.6 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
