# Qwen3.8-27B-FP8 ROCm prefill before/after (POOT-1193) - Strix Halo

- SoC: AMD Ryzen AI Max+ 395, GPU Radeon 8060S (gfx1151), 128 GiB unified memory
- model: `Qwen/Qwen3.8-27B-FP8` (`~/models/qwen3.8-27b-fp8`), fp8 e4m3 block-scaled weights, f32 compute
- before: `poot-bench-runner` built from master `45ff8154b` (the merge-base of `card/1193`)
- after: `poot-bench-runner` built from `card/1193` at `82d490093`
  (the `build_sha` field in all four raw files names the checkout the run was launched from, `82d490093`,
  not the binary's own commit; the binaries differ, as the prefill rates show)
- run: `--backend rocm --mode decode-curve --osl 4 --precision bf16 --json`, ISL 128 and 512 with `--warmup 1 --iters 2`,
  ISL 2048 with `--warmup 0 --iters 1` (a 2048-token master prefill takes ~2.5 minutes), base and final alternated in
  the same GPU-lock hold, under `systemd-run --user --scope -p MemoryMax=64G`; MemAvailable was 105 GiB at each start.

## Prefill (engine-produced TTFT; prefill tok/s = prompt tokens / TTFT)

| ISL  | master `45ff8154b` tok/s | `card/1193` tok/s | speedup |
| ---- | ------------------------ | ----------------- | ------- |
| 128  | 13.1, 13.0               | 55.6, 52.0        | ~4.1x   |
| 512  | 11.9, 11.1               | 57.4, 56.8        | ~5.0x   |
| 2048 | 15.6                     | 61.4              | ~3.9x   |

Earlier alternating holds on the same checkpoint, on a busier box (decode TPOT 250-330 ms), gave master 8.4-11.9 and
final 31.8-45.4 tok/s at ISL 128/512, ISL 2048 final median 38.0; the 1183 receipt baseline was 12.1-12.8 tok/s.

## Decode (unchanged path)

TPOT, ms: master 188-222, `card/1193` 198-221 at ISL 128/512; 191 vs 185 at ISL 2048.

## Coherence

`driver::device_tests::checkpoint::qwen35_27b_fp8_receipt::rocm` passes on the final kernel with the 1183 receipt's
first-token top-5 unchanged to seven digits: `(11751, -0.4532727)`, `(524, -3.3506498)`, `(264, -3.5284815)`,
`(31586, -3.920887)`, `(1076, -3.9422998)`; generated ids `[11751, 13, 198, 760, 6511, 314, 9564, 369]`.

## Open

PTX/NVIDIA prefill is not measured here: it needs a pod (POOT-1184). The new body is shared with PTX and wgpu.

## After merging master (POOT-1263 tile-range split)

`card/1193` merged with master `36cd0d307`; the register-tiled body reads `metadata[4]` (`tile_offset`) and the
planner splits it into tile-range dispatches under the device's watchdog budget (ROCm has none, so one dispatch;
wgpu splits). Paired run on the merged binary (`bench6`) against the master `45ff8154b` binary, same GPU-lock hold,
alternating, ISL 128/512, `--warmup 1 --iters 2`, on a loaded box (load average 20-40; other sessions' sims running):

| ISL | master tok/s | merged tok/s (run 1, run 3) |
| --- | ------------ | --------------------------- |
| 128 | 14.0, 14.4   | 53.6, 59.4 / 48.6, 47.7     |
| 512 | 12.1, 10.5   | 50.0, 48.3 / 46.0, 46.9     |

Raw: `raw-rebased-run{1,3}-isl128-512.json`, `raw-base-rebase-pair-isl128-512.json`. A first unpaired run of the merged
binary at load 20-40 gave 23-30 tok/s with TPOT 380-510 ms (the box, not the kernel: decode TPOT is unchanged and
the paired rerun above is the comparison).
