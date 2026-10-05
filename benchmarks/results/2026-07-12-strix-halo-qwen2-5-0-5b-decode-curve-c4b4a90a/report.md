# poot context-degradation curve - Strix Halo (card 180)

The `decode-curve` scenario (spec 115) on the dev box: for each ISL (context length) in {128, 512, 2048, 8192},
prefill exactly ISL synthetic tokens, then decode a fixed OSL=128 window, timing every token. All engine rows
run on the same physical iGPU.

- SoC: AMD Ryzen AI Max+ 395, GPU Radeon 8060S (gfx1151, RDNA3.5), 128 GiB unified memory
- poot commit: c4b4a90a (branch card-180-decode-curve-devbox)
- llama.cpp: d6d8995 (build 9747), Vulkan backend (RADV)
- captured: 2026-07-12
- model: qwen2.5-0.5b (Qwen2.5-0.5B-Instruct); poot rows load the bf16 safetensors checkpoint and compute
  f32 end to end; the llama.cpp row uses the local Q4_K_M GGUF (this box has no bf16/f16 GGUF for the 0.5b
  model). The quant differs, but the engines share the iGPU, token counts, and OSL window.

## Headline: two gaps found

poot's decode-curve scenario (the batched-prefill entry point that fills the KV cache in one Q=N forward
before timing decode) is broken on the ROCm backend and capped at ISL<2048 on the wgpu backend. Both paths
were unexercised: `--mode decode-curve` had only run on PTX pods (the README timing table cites qwen3-0.6b
PTX only). `--mode single` (the token-by-token re-prefill path used by every prior Strix Halo report,
including card 177) is unaffected on both backends.

### poot ROCm: hard crash (GPU memory access fault), every ISL

```
poot-bench: decode-curve isls=[128] osl=8 warmup=2 iters=1 backend=rocm
Memory access fault by GPU node-1 (Agent handle: 0x55555a335180) on address 0x7fffe7802000.
Reason: Page not present or supervisor privilege.
```

Reproduced 3x (isl=16/osl=4/warmup=0, isl=128/osl=16/warmup=1, isl=128/osl=8/warmup=2), so deterministic. It
is a driver-level fault (SIGABRT, rc=134) in `generate_kv_rocm_prefilled_tokens` ->
`RocmGraphExecutor::run_prefill` (the ROCm batched-prefill entry point added 2026-06-30, commit 79860ecd).
`--mode single` (`generate_kv_rocm`, per-token re-prefill) works on the same model/backend/box (control run
below). This blocks the decode-curve scenario until fixed. Filed as a finding, not fixed here: GPU kernel
debugging needs its own investigation.

Control (same box, model, and backend, `--mode single`):

```
{"framework":"poot","precision":"bf16","prompt_tokens":5,"gen_tokens":8,"iterations":1,
 "ttft_ms":240.9,"tpot_ms":33.2,"e2e_ms":473.1,"decode_tok_s":30.1}
```

### poot wgpu: works at ISL 128/512, hits the WebGPU dispatch-grid cap at ISL 2048

```
poot-bench: isl=2048 FAILED (gpu batched prefill: runtime: dispatch grid [229376, 1, 1]
exceeds the WebGPU 65535/dim cap; put the unbounded dim on x); emitting partial curve (2/4 isls)
```

This is the runner's partial-curve behavior (main.rs): a faulting ISL stops the sweep and records a caveat
instead of discarding earlier points. The attention-score dispatch grid scales with ISL and crosses WebGPU's
per-dimension dispatch cap (65535) somewhere between 512 and 2048 tokens for this model's batched-prefill
graph, which limits how much context poot can prefill in one shot on this backend.

## Measured curve (llama.cpp full ISL range; poot wgpu ISL 128/512 only)

| engine    | backend        | precision | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| --------- | -------------- | --------- | ------- | ------- | -------- | -------- |
| llama.cpp | Vulkan (RADV)  | Q4_K_M    | 357.0   | 354.3   | 333.9    | 287.5    |
| poot      | rocm (raw HSA) | bf16      | CRASH   | CRASH   | CRASH    | CRASH    |
| poot      | wgpu (RADV)    | bf16      | 25.7    | 24.2    | cap-fail | cap-fail |

Decode tok/s (steady state, OSL=128, prefill excluded). poot-ROCm cells are "CRASH" (GPU memory access
fault, every ISL tried); poot-wgpu ISL>=2048 cells are "cap-fail" (WebGPU dispatch-grid limit, not a timeout
or OOM).

TPOT p50 (ms/token, decode only):

| engine    | backend       | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| --------- | ------------- | ------- | ------- | -------- | -------- |
| llama.cpp | Vulkan (RADV) | 2.80    | 2.82    | 3.00     | 3.48     |
| poot      | wgpu (RADV)   | 38.9    | 41.3    | -        | -        |

TTFT p50 (ms, batched prefill of ISL tokens):

| engine    | backend       | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| --------- | ------------- | ------- | ------- | -------- | -------- |
| llama.cpp | Vulkan (RADV) | 14.2    | 34.0    | 149.4    | 816.8    |
| poot      | wgpu (RADV)   | 144.1   | 399.6   | -        | -        |

## The competitive ratio does not show the expected widening yet

poot-wgpu / llama.cpp decode tok/s is 0.072x at ISL=128 and 0.068x at ISL=512, roughly flat over the two
measurable points. This is not the flash-attention-decode degradation signature; it is dominated by the
device-compute/lane-underutilization gap already characterized in card 177's single-point measurement (poot
wgpu decode ~0.23x of llama.cpp there, at a different model/quant). Two points are too little context growth
to separate the O(context) attention term from the flat per-token GEMV cost, and llama.cpp's own curve moves
little (357 -> 288 tok/s from 128 to 8192, a 1.24x slowdown). Seeing the gap widen needs the 2048/8192 points
poot cannot yet produce here (ROCm crashes; wgpu hits the dispatch cap). On either local backend, poot's naive
batched prefill cannot reach the context lengths where its O(context) attention cost would show. The
hypothesized "decode bends up with ISL" curve remains unverified on this hardware until both are fixed.

## Why the manual path, not `bench run`

`bench run --scenario decode-curve --model qwen2.5-0.5b` (via `runners.toml` + `POOT_BENCH_BACKEND`) is wired
for ROCm/wgpu since card 178/178b, but a manual/hybrid path was cleaner here:

1. `bench.py` has no CLI override for the manifest's `isl` list (only `--gen-tokens`/`--warmup`/`--iters`), so
   capping the sweep for a timeout-safety canary meant editing `manifest.toml` or driving
   `poot-bench-runner --isl-list` directly. The ROCm crash appeared on the first canary (isl=128), so the
   whole-matrix `bench run` was never reached for poot; a per-ISL canary risks fewer bytes per attempt.
2. llama.cpp's `resolve_gguf` auto-converts HF safetensors to GGUF f16 via `convert_hf_to_gguf.py` when
   `model_dir` has no GGUF, which this box's `qwen2.5-0.5b` dir (bf16 safetensors) lacks, and the converter
   needs a python env with its deps. `~/models/qwen2.5-0.5b-gguf/` already has a Q4_K_M GGUF (also the
   baseline family for card 177's 1.5b row), so driving `runners/llamacpp/run.py --mode decode-curve` against
   it avoided a conversion dependency, at the cost of the Q4_K_M-vs-bf16 caveat above.

Both invocations used the harness runners (`poot-bench-runner`, `runners/llamacpp/run.py`) with the same
`--mode decode-curve --isl-list ... --osl ...` flags `bench.py` appends (checked in `harness/bench.py:run_cell`).
Only the per-ISL cap and the `results.jsonl`/`report.md` assembly were manual, using `bench.py`'s
`curve_stats()` so the numbers match what `bench run` would compute from the same raw samples.

## Safety / process notes

- `poot-bench-runner --release` was built fresh; the `rocm` feature is unconditional in
  `benchmarks/runners/poot/Cargo.toml`, so one build covers rocm, wgpu, and ptx (unused; no NVIDIA here).
- Every invocation ran under `setsid timeout <N>` to completion or hard failure. After each crash, no
  orphaned process (`pgrep -fa poot-bench-runner`) and VRAM back at the ~368 MB idle baseline
  (`/sys/class/drm/card1/device/mem_info_vram_used`) were confirmed before relaunching, including after the
  3 ROCm crashes.
- Engines ran strictly sequentially, never concurrently on the iGPU: ROCm canaries, then wgpu (128/512, the
  2048 cap-fail canary, the final combined run), then llama.cpp.
- `llama-swap` (an always-on local proxy for interactive chat models) was idle throughout (no `llama-server`
  child, VRAM at baseline) and not disturbed.
- No process left running, no GPU memory pinned, nothing pushed, and the worktree stayed on
  `card-180-decode-curve-devbox`.
