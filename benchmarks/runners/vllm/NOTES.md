# vLLM runner notes (research 2026-06-12, vLLM 0.22.1 / V1 engine)

> Status: snapshot. Pins as of 2026-06; spot-checked 2026-09-19 against `benchmarks/docker/Dockerfile`
> (llama.cpp b9601, vLLM 0.22.1, transformers 5.0.0, candle 0.10.2 match). The image is the source of truth and
> is now Ubuntu 24.04 / CUDA 12.6.3, so CUDA 12.4 / Ubuntu 22.04 mentions below are stale.

Pinned: `vllm==0.22.1` (V1 engine). Install with `uv pip install "vllm==0.22.1" --torch-backend=auto`
or `pip install "vllm==0.22.1" --extra-index-url https://download.pytorch.org/whl/cu129`. No CUDA-12.4
wheel exists; the cu128/cu129 wheel runs on a 12.4 driver via minor-version compat (RunPod 550.x is fine).
Fall back to cu128 if the driver is too old. Capture `pip freeze | grep -E "vllm|torch|flashinfer"`.

## The big caveat: offline metrics

V1 `LLM.generate()` returns `RequestOutput.metrics = None` (V0's `arrival_time`/`first_token_time` were
removed; RFC issue #26298 tracks reintroduction). So **true TTFT is not observable from offline
`generate()`**. Two options:

- Offline + external timing (what this suite does for throughput): time the whole request for e2e + decode
  tok/s; estimate TTFT with a `max_tokens=1` prefill-only request (prefill + 1 token ~= TTFT). Record it
  as `ttft_estimate`, with a caveat.
- Online server (`vllm serve`) + Prometheus `/metrics` (`vllm:time_to_first_token_seconds`,
  `vllm:time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds`) or client-side TTFT from the
  first SSE chunk - the only engine-authoritative TTFT today. Use if TTFT must be vLLM-reported.

## VRAM fairness (mandatory)

vLLM V1 pre-allocates a KV-cache pool sized by `gpu_memory_utilization` (default 0.90 of total, measured
after weights) and never releases it, so a raw `nvidia-smi` peak is the reservation, not demand, and does not
compare to transformers/candle. Required approach:

- Report two numbers: (1) weights footprint (`torch.cuda.memory_allocated()` right after `LLM()` init, or
  nvidia-smi sampled then); (2) working set at an explicit `gpu_memory_utilization` with `max_model_len`
  pinned to the benchmark context (e.g. 2048), so the pool stays bounded.
- Always emit the caveat string in the row. Optionally `enforce_eager=True` to exclude CUDA-graph buffers.
- The external sampler sees the whole pool, so the harness must record `--gpu-mem-util` in the caveat. The
  cross-engine number is the weights footprint plus the pinned-context working set, not a raw nvidia-smi
  peak.
- `gpu_memory_utilization` bounds vLLM's entire budget (weights + KV cache), so it must exceed the model's
  weight footprint or vLLM fails "No available memory for the cache blocks". The old default (0.30) was too
  low for xl-tier (~14B+) bf16 weights (~28 GiB > 0.30 x ~44 GiB usable on a 48GB-class GPU). The default is now 0.85; override per run with `--gpu-mem-util` for a tighter VRAM number
  on a small model.

## Precision / arch

- dtype: `dtype="auto"` reads config `torch_dtype`; force with `"bfloat16"`/`"float16"` to match peers.
- quant: `quantization="gptq"|"gptq_marlin"|"awq"|"awq_marlin"` (auto-detected from `quantization_config`,
  pin to force the kernel). Marlin needs Ampere+ and is much faster.
- Qwen2/2.5/3 (dense+MoE), Llama, Mistral/Mixtral, GPTQ, AWQ, OLMoE: all supported.
- **gpt-oss MXFP4: native kernel needs Hopper/Blackwell; on Ampere (3090) NOT natively accelerated -
  probe before trusting gpt-oss numbers on a 3090-class card.**
- Multimodal (SmolVLM/Idefics3, LLaVA, Qwen-VL): offline image path works via
  `llm.generate({"prompt": "...<image>...", "multi_modal_data": {"image": PIL.Image}}, sp)`. The `<image>`
  placeholder is model-specific.

Runner: see `run.py` in this dir (offline timing + prefill-subtraction TTFT + JSON line + torch VRAM
cross-check). Headline VRAM comes from the harness external sampler, not torch.

## First-pod validation (2026-06-12, RTX 3090)

- **The engine needs a C compiler at runtime.** vLLM 0.22.1 runs `torch.compile` (inductor + triton) at
  EngineCore init, which JIT-compiles a C/CUDA helper; with no `gcc` on PATH it dies during
  `determine_available_memory` with `InductorError: RuntimeError: Failed to find C compiler` (confirmed on
  an A5000, 2026-06-15). `common.sh` installs `build-essential`; the slim Docker image must install
  `gcc g++` in the final stage (only the throwaway builder had a toolchain). ninja alone does not help; gcc
  is the fix.
- vllm 0.22.1 pulled torch 2.11.0 + transformers 5.11.0 into its venv - that venv doubles as the
  transformers runner env (install `accelerate` there too). Validated: 487 tok/s bf16 decode, KV pool at
  gpu_memory_utilization=0.3.

## FlashInfer sampler needs nvcc at runtime (fixed 2026-08-20, RTX 3090)

Every cross-framework sweep on 2026-08-20 recorded `vllm` as `error` on every model (`rc=1: ... with
launch_core_engines( | ... RuntimeError: Engine core initialization failed. See root cause above.`). The
harness keeps only the last 8 stderr lines (`bench.py`'s `run_cell`), so the root cause was never captured.
Running `runners/vllm/run.py` by hand on a RunPod RTX 3090 showed that vLLM's V1 `EngineCore` picks FlashInfer
for top-k/top-p sampling whenever it is importable, and FlashInfer JIT-compiles its CUDA kernel with `nvcc` on
first use, inside `EngineCore._initialize_kv_caches -> profile_run` on every engine start. The bench image's
runtime stage (`nvidia/cuda:...-cudnn-runtime`) has no `nvcc` (only the throwaway `-devel` builder does, see
`docker/Dockerfile`). The 2026-06-15 image validation predates this; it was not root-caused which vLLM or
flashinfer version changed it.

Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` forces vLLM's PyTorch top-k/top-p sampler and skips the JIT. It is set as
an `ENV` in `docker/Dockerfile` and defensively in `runners/vllm/run.py` (`os.environ.setdefault(...)` before
`import vllm`), so it also holds for an on-pod `setup/vllm.sh` install. The runner samples greedily
(`temperature=0.0`), so the FlashInfer kernel gave no benchmark-relevant speed.

Verified on a RunPod RTX 3090 with the rebuilt `ghcr.io/kikijiki/poot-bench:latest`: `qwen2.5-0.5b-instruct`
bf16 single-shot, 469 tok/s decode (vs 487 tok/s on 2026-06-15, within noise); decode-curve mode also passes at
isl=128/512 with real per-token ITL data. Both failed before the fix.
