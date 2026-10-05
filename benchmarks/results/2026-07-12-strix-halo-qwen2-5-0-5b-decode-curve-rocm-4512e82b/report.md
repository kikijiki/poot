# poot ROCm context-degradation curve - Strix Halo (card 180b)

Follow-up to card 180 (`benchmarks/results/2026-07-12-strix-halo-qwen2-5-0-5b-decode-curve-c4b4a90a/`), which
got the wgpu partial curve (ISL 128/512, capped by a WebGPU dispatch-grid limit) and the llama.cpp full curve,
but poot's ROCm row crashed at every ISL (a bf16-weight naive-matmul GPU page fault in the batched-prefill bind
path). Card 181 fixed that on master (commit `2be65289`, "widen non-WMMA-eligible bf16 weight consts to f32 in
bind"); this run retries the ROCm decode-curve.

- SoC: AMD Ryzen AI Max+ 395, GPU Radeon 8060S (gfx1151, RDNA3.5), 128 GiB unified memory
- poot commit: `4512e82b` (branch `card-180b-rocm-decode-curve`, includes the card 181 fix)
- backend: rocm (raw HSA)
- model: qwen2.5-0.5b (Qwen2.5-0.5B-Instruct), bf16-native checkpoint, computed f32 end to end
- osl: 16 (this run) / 128 (card 180's rows); steady-state comparison is unaffected (see below)
- captured: 2026-07-12

## Headline: poot ROCm's curve is measured and bends up

| engine    | backend        | precision | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| --------- | -------------- | --------- | ------- | ------- | -------- | -------- |
| llama.cpp | Vulkan (RADV)  | Q4_K_M    | 357.0   | 354.3   | 333.9    | 287.5    |
| poot      | rocm (raw HSA) | f32       | 25.9    | 25.2    | 23.2     | not run  |
| poot      | wgpu (RADV)    | f32       | 25.7    | 24.2    | cap-fail | cap-fail |

Decode tok/s (steady state, decode-only, prefill excluded). poot-ROCm has numbers at 128/512/2048. ISL 8192
was not attempted (see "Why ISL 8192 was not attempted" below).

TPOT p50 (ms/token, decode only):

| engine    | backend       | ISL 128 | ISL 512 | ISL 2048 |
| --------- | ------------- | ------- | ------- | -------- |
| llama.cpp | Vulkan (RADV) | 2.80    | 2.82    | 3.00     |
| poot      | rocm (raw HSA)| 38.63   | 39.75   | 43.11    |
| poot      | wgpu (RADV)   | 38.9    | 41.3    | -        |

TTFT p50 (ms, batched prefill of ISL tokens; poot's naive O(N^2) attention shows up here because prefill
materializes the full [Hq,N,N] attention-score tensor in one forward):

| engine    | backend       | ISL 128 | ISL 512 | ISL 2048 |
| --------- | ------------- | ------- | ------- | -------- |
| llama.cpp | Vulkan (RADV) | 14.2    | 34.0    | 149.4    |
| poot      | rocm (raw HSA)| 235.6   | 1015.1  | 5769.0   |
| poot      | wgpu (RADV)   | 144.1   | 399.6   | -        |

## The gap: decode bends up modestly, TTFT sharply

**Decode tok/s ratio poot-ROCm / llama.cpp, and poot-ROCm's own degradation:**

| ISL  | poot-ROCm tok/s | llama.cpp tok/s | ratio (rocm/llama) | poot-ROCm vs its own ISL=128 |
| ---- | --------------- | --------------- | ------------------- | ----------------------------- |
| 128  | 25.89           | 357.04          | 0.0725x              | 1.00x (baseline)             |
| 512  | 25.16           | 354.32          | 0.0710x              | 0.97x (-2.8%)                 |
| 2048 | 23.19           | 333.87          | 0.0695x              | 0.90x (-10.4%)                |

poot-ROCm's decode throughput drops 10.4% from ISL 128 to ISL 2048 (naive O(context) attention adds per-token
cost as the KV cache grows). llama.cpp's curve drops 6.5% over the same range (its Vulkan flash-attention
decode has a flatter O(context) term). The poot-ROCm/llama.cpp ratio widens modestly, 0.0725x -> 0.0710x ->
0.0695x, about 4% relative on an already large gap. At OSL=16 the fixed per-token overhead (kernel dispatch,
argmax, sampling) is a larger share of decode time than at OSL=128 (card 180's wgpu/llama rows), which
compresses how much of the attention term shows through; a longer OSL would likely show a bigger split.

**TTFT shows it more clearly.** poot-ROCm's batched-prefill TTFT scales 4.3x (128->512) then 5.7x (512->2048),
approaching the O(N^2) signature of naive attention: materializing and reducing the [Hq,N,N] score tensor
scales quadratically in N while the rest of the forward pass scales linearly. llama.cpp's TTFT scales 2.4x
then 4.4x (fused/tiled flash-attention prefill, no full N x N materialization). The TTFT ratio widens from
16.5x slower at ISL=128 to 29.9x at ISL=512 and 38.6x at ISL=2048. poot's batched prefill has no fused/tiled
attention kernel on this backend, and the cost shows up almost entirely in TTFT.

## Audit finding: intermittent hang when sweeping more than one ISL in one process (ROCm)

This is a different failure from the card-180 bf16 fault, seen when reusing one `RocmGraphExecutor` across
ISLs, as `--mode decode-curve` does (`benchmarks/runners/poot/src/main.rs`).

- About half the multi-ISL invocations (`--isl-list 128,512`, `512,128`, `128,512,2048`) hung indefinitely on
  the first curve point, with no crash, no error, and no `isl=<N> done` line, until killed by `timeout` (45s,
  90s, 180s, 300s, and 1200s all hit the wall with no progress).
- Single-ISL runs (`128`, `512`, `2048` alone) were reliable across 6+ trials, 0.8-6.4s wall time each.
  `--isl-list 128,128` was also reliable.
- The same multi-ISL invocation was non-deterministic: `--isl-list 512,128` hung on attempts 1 (killed at 180s)
  and 2 (300s), then completed in seconds on attempt 3. `--isl-list 128,512,2048` hung once (killed at 1200s),
  then completed in under a minute on retry.
- After each `timeout` SIGTERM the GPU was clean: VRAM/GTT back at the ~368 MB idle baseline, no leftover or
  D-state process, no GPU reset or fault in the (permission-limited) dmesg check. This looks like a hang, not
  a crash: `RocmContext::synchronize()` (`crates/poot-rocm-runtime/src/lib.rs`, around line 1141) busy-spins on
  the HSA queue read-index with no timeout or fault check, so a delayed or dropped completion signal spins
  forever.
- Likely mechanism (from code reading only, no live repro): `RocmGraphExecutor` keeps one `const_cache` (keyed
  by weight name + logical element count, not byte width) shared across `bind` (prefill's `run_prefill`) and
  `bind_decode_slots` (`capture_decode`). The card-181 fix uploads a BF16-typed weight const either as 2 bytes
  per element (WMMA-eligible) or widened to 4-byte f32 (`crates/poot-rocm-gpu/src/lib.rs`, around lines
  620-695), but the cache lookup filter is `elem_count == numel` regardless of the stored byte width. If two
  graphs (two ISL prefill graphs, or prefill vs decode) disagree on eligibility for a same-named const, a buffer
  cached at one width could be read at the other, which fits a data hazard and the non-determinism. This is a
  hypothesis needing a live repro with tracing; it was not fixed here.
- Mitigation used: the clean 3-point curve came from a `--isl-list 128,512,2048` invocation that completed,
  cross-checked against isolated single-ISL runs (TTFT/ITL agree within noise; see `raw-poot-rocm-curve.json`).
  Future runs should budget for retries or run one ISL per process (slower: each pays model load and kernel
  JIT).

## Why ISL 8192 was not attempted

poot-ROCm's naive prefill already reached ~5.8s TTFT at ISL 2048. Extrapolating the 4.3x/5.7x-per-4x-context
growth to ISL 8192 suggests roughly 90s+ of prefill alone. With the intermittent multi-ISL hang, a 4-point
sweep risked a long GPU-serial window for one point of secondary value, since the TTFT scaling is already
clear from 128/512/2048. Left as follow-on work.

## Data provenance

- `raw-poot-rocm-curve.json`: raw per-iteration samples from
  `poot-bench-runner --model-dir ~/models/qwen2.5-0.5b --mode decode-curve --isl-list 128,512,2048
  --osl 16 --warmup 1 --iters 2 --backend rocm --json`, run inside `nix develop` under `setsid timeout 900`.
  Exit code 0, clean VRAM/GTT afterward.
- `results.jsonl`: this run's poot-rocm row (aggregated with `benchmarks/harness/bench.py`'s `curve_stats()`)
  plus the poot-wgpu and llama.cpp rows carried forward unchanged from card 180's `results.jsonl` (same box,
  model, and day).
- The single-ISL cross-check runs (`--isl-list 128`, `512`, `2048` alone) were diagnostic canaries for the hang
  and are not committed. Their TTFT/ITL matched the combined run within noise (poot rocm TTFT for isl=128:
  232-240ms across 4 measurements; isl=512: 1001-1028ms across 3; isl=2048: 5677-5825ms across 2).

## Safety / process notes

- `poot-bench-runner --release` was built fresh in the worktree (picking up the card-181 fix via the
  `poot-rocm-gpu` path dependency) at `benchmarks/runners/poot/target/release/poot-bench-runner`.
- Every invocation ran under `setsid timeout <N>`, serially (never concurrent with another GPU workload), with
  GPU state (`mem_info_vram_used`, `mem_info_gtt_used`) checked back at the ~368 MB idle baseline after every
  attempt, including the timeout-killed hangs.
- `llama-swap` (an always-on local proxy for interactive chat models) was idle throughout and not disturbed.
- No process left running, no GPU memory pinned, nothing pushed, and the worktree stayed on
  `card-180b-rocm-decode-curve`.
