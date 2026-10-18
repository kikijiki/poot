# Qwen3.8-27B-FP8 on poot ROCm - Strix Halo (POOT-1183)

- SoC: AMD Ryzen AI Max+ 395, GPU Radeon 8060S (gfx1151), 128 GiB unified memory
- poot commit: `45713e6c3` plus the POOT-1183 change to the receipt test (branch `card/1183`)
- backend: rocm (raw HSA), decode by recorded replay (1536 replays, 4 recordings, 0 host picks)
- model: `Qwen/Qwen3.8-27B-FP8` (`~/models/qwen3.8-27b-fp8`, 29 GiB), fp8 e4m3 block-scaled weights, f32 compute
- run: `poot-bench-runner --mode decode-curve --isl-list 128,512,2048 --osl 128 --warmup 1 --iters 3 --precision bf16 --roofline --json`
  under `flock /tmp/poot-gpu.lock`, `systemd-run --user --scope -p MemoryMax=64G`, MemAvailable watchdog (kill < 20 GiB)

## Coherence receipt (ROCm gfx1151)

Row `driver::device_tests::checkpoint::qwen35_27b_fp8_receipt::rocm` (device half), release build, PASS in 32 s,
peak host memory `VmHWM: 28005816 kB`. Prompt `"The capital of France is"`, ids `[760, 6511, 314, 9338, 369]`, greedy.

- generated ids `[11751, 13, 198, 760, 6511, 314, 9564, 369]`, text `" Paris.\nThe capital of Germany is"`
- device first-token top-5 (token id, log-probability): `(11751, -0.4532727)`, `(524, -3.3506498)`, `(264, -3.5284815)`,
  `(31586, -3.920887)`, `(1076, -3.9422998)`; the first token is the documented answer `" Paris"`.
- Mutation: the row's `" Paris"` assertion changed to `" Rome"` fails with `the documented answer; the text was " Paris.\nThe capital of Germany is"; left: " Paris", right: " Rome"`; restored, green.
- **The oracle comparison (top-5 ids exact, log-probabilities within 1e-3 against the CPU oracle) is PENDING the
  POOT-1184 fixture.** The CPU oracle for this checkpoint needs more than 64 GB of host RAM (a run was SIGKILLed at the
  64 GB cap after about 12 minutes), so it is recorded once on the PTX pod and the ROCm and wgpu rows compare
  against that recorded oracle. Until then this card has the device half only; the top-5 above is what that
  comparison must match.

## Follow-up cards

- POOT-1193: ROCm FP8 prefill runs at ~12 tok/s (80 ms per prompt token, flat in ISL).
- POOT-1194: host memory climbs from 99 to 19 GiB MemAvailable in a cold `--profile --warmup 0` run.
- wgpu numbers for the same checkpoint come from POOT-1182; the ROCm/wgpu comparison is POOT-1187.

## Results (engine-produced; median of 3 timed iterations)

| ISL  | TTFT (ms) | prefill tok/s | TPOT (ms) | decode tok/s | decode GB/s | of 204 GB/s ceiling |
| ---- | --------- | ------------- | --------- | ------------ | ----------- | ------------------- |
| 128  | 10391     | 12.3          | 200.5     | 4.99         | 134.5       | 66%                 |
| 512  | 39970     | 12.8          | 193.0     | 5.18         | 140.8       | 69%                 |
| 2048 | 169781    | 12.1          | 192.1     | 5.21         | 135.9       | 67%                 |

- Peak host RSS of the bench process: 29059008 KiB (27.7 GiB; GNU `time -v`). Model open (load + 4 plans + recordings): 34.7 s.
- Decode is bandwidth-bound and healthy: 27.1 GB read per token, 66-69% of the gfx1151 ceiling receipt, flat in ISL.
- Prefill is the finding: ~12 tok/s at every ISL, i.e. a constant ~80 ms per prompt token. One decode step is ~195 ms and
  streams the weights once, so prefill costs about 0.4 of a weight pass per token: the batched FP8 matmuls are not
  amortizing the weight reads across the prompt (compute-bound ideal is ~40x faster). Unproven hypothesis: the FP8
  prefill matmul does not take the tiled/WMMA route (see `wmma-selection-is-dtype-driven`). Not fixed in this card (POOT-1193).
- `raw-poot-rocm-curve-profiled.json` is the same matrix with `--profile`: a PERTURBED window (per-dispatch timing changes
  submission grouping), kept only for its dispatch-time sums; do not read throughput from it.
- A `--profile --warmup 0 --iters 1 --isl-list 128 --osl 2` run climbed host MemAvailable from 99 to 19 GiB over ~10 minutes
  and tripped the watchdog; the warmed runs above did not. Not root-caused (POOT-1194).

## Per-phase breakdown (from the `--profile` window, PERTURBED)

Each window is 384 steps (3 timed iterations: one prefill step and 127 decode steps each, plus 3 for the rest) over
381 decoded tokens. Prefill is the sum of the iterations' TTFT; decode is the rest of the wall.

| ISL  | wall (s) | prefill (s) | decode (s) | host encode (ms) | device wait (s) | other (ms) | dispatch sum (ms/tok) |
| ---- | -------- | ----------- | ---------- | ---------------- | --------------- | ---------- | --------------------- |
| 128  | 105.9    | 29.9        | 76.0       | 316.9            | 105.2           | 35.4       | 267.0                 |
| 512  | 227.5    | 142.0       | 85.5       | 355.2            | 226.8           | 37.3       | 584.9                 |
| 2048 | 550.6    | 470.4       | 80.2       | 365.2            | 549.8           | 38.9       | 1433.3                |

- Host encode plus other is under 0.4% of wall at every ISL: the run is entirely device-bound; host submission is not a factor.
- Decode time per window is flat (76-86 s for 381 tokens, ~200-225 ms/token) while prefill grows linearly with ISL,
  so the whole ISL dependence of end-to-end time is prefill (POOT-1193).
- The dispatch sums include prefill dispatches, so they scale with ISL; they are per-dispatch device time with
  overlap, not throughput (the window is perturbed and per-kernel names were not retained).
