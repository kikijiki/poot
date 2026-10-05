# poot competitive benchmark - Strix Halo (card 177)

First committed poot-vs-llama.cpp result on the dev box (Strix Halo iGPU), all engines on the same
hardware at the same quant (Q4_K_M).

- SoC: AMD Ryzen AI Max+ 395, GPU Radeon 8060S (gfx1151, RDNA3.5), 128 GiB unified memory
- GPU: AMD Radeon 8060S Graphics (RADV STRIX_HALO)
- poot commit: edca5d2f (branch card-177-strix-halo-benchmark)
- llama.cpp: d6d8995 (build 9747), Vulkan backend
- captured: 2026-07-12
- metrics: poot rows are poot's own engine timing (no vendor sysfs/NVML); llama.cpp rows are
  llama-bench aggregate throughput.

## qwen2.5-1.5b Q4_K_M - decode (quant-resident, prompt 5, gen 128, 2 warmup + 5 iters)

All three engines run the same `qwen2.5-1.5b-instruct-q4_k_m.gguf` on the same iGPU. poot decodes
quant-resident (dequant-in-matmul, no f32 upcast), matching llama.cpp's k-quant path at bit-width parity
(k-quant is not a bit-identical format across tools; recorded as a caveat).

| engine    | backend        | precision | decode tok/s | TPOT ms | TTFT ms | vs llama.cpp | notes                             |
| --------- | -------------- | --------- | ------------ | ------- | ------- | ------------ | --------------------------------- |
| llama.cpp | Vulkan (RADV)  | Q4_K_M    | 188.5        | 5.3     | -       | 1.00x        | -ngl 99; tg128 aggregate, 3 reps  |
| poot      | rocm (raw HSA) | Q4_K      | 71.9         | 13.9    | 113     | 0.38x        | fastest poot backend on this iGPU |
| poot      | wgpu (RADV)    | Q4_K      | 42.8         | 23.4    | 134     | 0.23x        | cached re-encode/submit           |

Ratios:

- poot ROCm decode is **1.68x** the poot wgpu decode rate on the same chip (71.9 vs 42.8 tok/s), the
  reliable within-session comparison.
- llama.cpp Vulkan is **2.6x** poot's best (ROCm) and **4.4x** poot wgpu. poot's kernels are naive (no flash
  attention, host-orchestrated, no coopmat/tensor-core GEMM), so the gap is expected.

## Prefill (llama.cpp only, not comparable to poot TTFT)

llama-bench pp512 (prompt processing) on the same GGUF: **5269.8 tok/s** (stdev 19.4). poot's TTFT above is a
5-token prompt fill, not a 512-token prefill, so the two are not comparable. A matched prefill sweep (poot
batched prefill vs llama-bench pp512) is a follow-up; poot's prefill is naive and serial and would be the
slow row.

## Why this is the manual-report path, not `bench run`

`bench run` (`runners.toml` -> `bench.py`) drives the poot cell with `--precision` and defaults the runner to
`--backend ptx` (needs libcuda). There was no `--backend rocm|wgpu` plumbing in `bench.py`/`runners.toml` at the
time, so `bench run` could not select poot's ROCm/wgpu backends on this box, and `run-local-compare.sh` compares
poot backends only (no llama.cpp). To get poot-rocm, poot-wgpu, and llama.cpp in one same-hardware snapshot,
this run drove the `poot-bench-runner` binary (the one the harness uses) directly for the two poot rows and
`llama-bench` for the baseline, then assembled the snapshot by hand in the standard
`results/<run-id>/{env.json,results.jsonl,report.md}` shape.

Engines ran strictly sequentially (ROCm, wgpu, then llama.cpp), never concurrently on the iGPU, with each poot
run under `setsid timeout`. qwen2.5-0.5b Q4_K_M was tried first, but poot's quant tracer reports
`gguf_quant() is None` for both local 0.5b Q4_K_M files, so the run fell back to qwen2.5-1.5b Q4_K_M.

## Relationship to the prior Strix Halo snapshot

`results/2026-07-01-strix-halo-qwen2-5-0-5b-2cf7758` is a poot-only, f32-upcast cross-backend (rocm vs wgpu)
comparison with no baseline. This snapshot adds the first llama.cpp baseline on this box and the
quant-resident (Q4_K) decode path, and serves as a regression baseline for the poot ROCm/wgpu decode rate at
Q4_K.

---

_poot decode numbers are poot's own engine timing (median of 5 timed iters, prefill excluded from the
per-token rate). llama.cpp is llama-bench's synthetic pp512/tg128 aggregate throughput at -ngl 99. k-quant is
bit-width parity across tools, not a bit-identical format. Absolute tok/s on this shared box is noisy; the
poot ROCm/wgpu ratio within one session is the reliable signal. See benchmarks/README.md._
