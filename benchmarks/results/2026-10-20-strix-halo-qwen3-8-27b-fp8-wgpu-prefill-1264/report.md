# Qwen3.8-27B-FP8 wgpu prefill (POOT-1264) - Strix Halo

- SoC: AMD Ryzen AI Max+ 395, GPU Radeon 8060S (gfx1151, Vulkan/RADV through wgpu), 128 GiB unified memory
- model: `Qwen/Qwen3.8-27B-FP8` (`~/models/qwen3.8-27b-fp8`), block-scaled E4M3 weights read packed, bf16 precision flag
- build: `poot-bench-runner` from master `20fe8bfad` (the `build_sha` field), which already holds the register-tiled
  packed GEMM from POOT-1193 and the POOT-1263 tile-range split; no kernel change was needed
- run: `--backend wgpu --mode decode-curve --osl 4 --precision bf16 --json`, under the GPU lock and
  `systemd-run --user --scope -p MemoryMax=64G` (owner-approved guarded local run of this checkpoint);
  MemAvailable 102-106 GiB at each start; box load average 18-31 (other sessions' builds), so rates are conservative.

## Prefill (engine-produced TTFT; prefill tok/s = prompt tokens / TTFT)

| ISL | run                              | tok/s      |
| --- | -------------------------------- | ---------- |
| 128 | `--warmup 1 --iters 2`           | 48.6, 46.7 |
| 512 | `--warmup 1 --iters 2`           | 50.6, 44.0 |
| 512 | `--warmup 0 --iters 1` (TTFT 8.9 s) | 57.4    |

Baseline from the POOT-1182/1183 receipt (`2026-10-18-strix-halo-qwen3-8-27b-fp8-wgpu-4c8945ee1`): ISL 512 TTFT
130.3 s (3.9 tok/s). The acceptance floor is 20 tok/s at ISL 512: met at 44.0-57.4 tok/s (about 11-15x).

Decode TPOT is 345-469 ms in these runs against 335-340 ms in the 1183 receipt: the decode path is unchanged and the
box was loaded.

## Parity

The wgpu tier-2 rows for the register-tiled packed GEMM pass (`contraction_tiled_*_on_wgpu`,
`contraction_wide_register_tile_matches_reference_on_wgpu`, 6/6 serial under the GPU lock), against the CPU reference
within each format's tolerance (E4M3 wide tile: 1e-3 relative). The new GPU-free plan row
`a_fp8_prefill_linear_plans_the_register_tiled_gemm_on_wgpu` pins that wgpu plans this GEMM for the 27B shape.
