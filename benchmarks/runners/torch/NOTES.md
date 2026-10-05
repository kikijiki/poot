# torch + transformers runner notes (research 2026-06-12, transformers v5.0.0)

> Status: snapshot. Pins as of 2026-06; spot-checked 2026-09-19 against `benchmarks/docker/Dockerfile`
> (llama.cpp b9601, vLLM 0.22.1, transformers 5.0.0, candle 0.10.2 match). The image is the source of truth and
> is now Ubuntu 24.04 / CUDA 12.6.3, so CUDA 12.4 / Ubuntu 22.04 mentions below are stale.

The universal reference. **transformers v5.0.0** (2026-01-26) changed a lot that bites a runner:

- `torch_dtype=` -> `dtype=` (default `dtype="auto"` reads config); guard on `transformers.__version__`.
- `AutoModelForVision2Seq` REMOVED -> `AutoModelForImageTextToText` (the VLM class for SmolVLM/LLaVA).
- `load_in_4bit`/`load_in_8bit` booleans REMOVED -> `quantization_config=BitsAndBytesConfig(...)`.
- AutoGPTQ dead -> **GPTQModel** (`pip install gptqmodel --no-build-isolation`).

## Pinned install (main v5 env)

Do not reinstall torch if the pod base image already ships a CUDA-12.x torch (it breaks pods). If pinning: `torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0` from the cu124 index (last
clean cu124 line; 2.7+ moved to cu126/cu128). Then:
`transformers==5.0.0 accelerate==1.10.1 optimum==1.24.0 safetensors tokenizers==0.21.0 pillow
huggingface-hub`, `bitsandbytes==0.46.1`, `gptqmodel==5.8.0 --no-build-isolation`. Freeze a lockfile on the
first good pod (`pip freeze`) for true reproducibility; validate with `pip check`.

## AWQ needs a SEPARATE venv

`pip install autoawq` force-downgrades transformers to 4.47.1, so AWQ rows run in their own venv
(`/opt/awq-venv`, `autoawq==0.2.9` + `transformers==4.47.1`), where the runner uses `torch_dtype=` (4.47
API). AutoAWQ is in maintenance; the v5-native alternative is llm-compressor / compressed-tensors (needs
re-quant, not loading existing AWQ repos). Keep AWQ isolated so the rest of the matrix stays on v5.

## Timing (CUDA is async)

- `torch.cuda.synchronize(device)` BEFORE t0 (drain prior work) and before t_end.
- Warmup one generation, THEN `reset_peak_memory_stats()` so timing + VRAM reflect steady state.
- TTFT via `TextIteratorStreamer` in a background thread: timestamp the first yielded chunk minus the
  pre-generate timestamp. The streamer is fed after a device->host token copy (a sync), so the first yield
  is a real first-token signal.
- TPOT / decode tok/s over the decode-only window `t_end - first_token_time`, divided by `n_out - 1`
  (prefill cost excluded). Streamer token counts are approximate (text re-encode); exact counts need a
  `return_dict_in_generate=True` pass.

## VRAM - report all three, NVML is the headline

`torch.cuda.max_memory_allocated` (undercounts: no CUDA context ~300-600MB, no cuBLAS workspace, no frag) <
`max_memory_reserved` (torch slab) < **NVML per-PID `usedGpuMemory`** (true process footprint). NVML is the
only fair cross-engine number (poot has no torch allocator). Poll `pynvml`
(`nvmlDeviceGetComputeRunningProcesses`, filter PID) at ~50Hz, keep max. The suite's external sampler does
this uniformly; the torch numbers are diagnostics.

## Precision / arch

- dtype: pass EXPLICIT dtype to match poot (don't trust "auto"). For GPTQ/AWQ the quant weights are fixed by
  the checkpoint; `dtype=` controls the surrounding compute (embeds/norms/KV/accumulate) - record
  `actual_dtype` + `quant_method` per row.
- GPTQ: auto-detected from config `quant_method: gptq` (gptqmodel backend), no code change, point at the dir.
- AWQ: auto-detected `quant_method: awq` (autoawq, separate venv).
- OLMoE: plain `AutoModelForCausalLM`, bf16, loads anywhere.
- **gpt-oss: MXFP4 only on sm_90+ (H100/B100); on 3090/4090/Ada it falls back to bf16 at ~4x memory (20B
  ~40GB+).** That is not a 4-bit run; record the caveat rather than comparing poot-mxfp4 to torch-bf16.
  Native MXFP4 needs a Hopper pod.
- SmolVLM: `AutoModelForImageTextToText` + `AutoProcessor`, bf16, inputs via `processor.apply_chat_template`.

Runner: `run.py` in this dir - streamer TTFT, cuda.synchronize timing, NVML peak, JSON last line on stdout.
Keep the version guard for the AWQ venv.

## First-pod validation (2026-06-12, RTX 3090)

- **`apply_chat_template(return_tensors="pt")` returns a DICT in transformers v5** - wrap with
  `return_dict=True` and move every entry to device (the runner does this). Validated: 52 tok/s bf16 decode
  on Qwen2.5-0.5B, transformers 5.11.0.
- **`device_map={"":0}` needs `accelerate`** (`pip install accelerate`). Add it to whatever env runs this.
- When sharing the vLLM venv for this runner, the venv brought transformers 5.11.0 + torch 2.11.0 - works.
