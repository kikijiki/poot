# llama.cpp runner notes (research 2026-06-12, tag b9601)

> Status: snapshot. Pins as of 2026-06; spot-checked 2026-09-19 against `benchmarks/docker/Dockerfile`
> (llama.cpp b9601, vLLM 0.22.1, transformers 5.0.0, candle 0.10.2 match). The image is the source of truth and
> is now Ubuntu 24.04 / CUDA 12.6.3, so CUDA 12.4 / Ubuntu 22.04 mentions below are stale.

llama.cpp ships date-stamped rolling tags (`bNNNN`), not semver. Pin by tag: **`b9601`** (2026-06-11).

## Build needs the full CUDA toolkit (nvcc)

Unlike vLLM (prebuilt kernels) and poot/cudarc (dlopen the driver), llama.cpp compiles its own ggml CUDA
kernels with nvcc. So **build ON a `*-devel` pod** (e.g. `nvidia/cuda:12.4.1-devel-ubuntu22.04`), not a
runtime/base image. cmake 3.21+, GCC 12+, driver 525+.

```bash
export LLAMACPP_TAG=b9601
git clone --depth 1 --branch "$LLAMACPP_TAG" https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"   # A100;A10/30xx;Ada/40xx;Hopper
cmake --build build --config Release -j"$(nproc)"
# build/bin/: llama-cli, llama-bench, llama-mtmd-cli, llama-quantize
```

(`-DLLAMA_CUBLAS=ON` is dead; use `-DGGML_CUDA=ON`. Always run with `-ngl 99` = all layers on GPU.)

## Metrics

- **llama-bench** (canonical, structured throughput). Separates pp (prefill) and tg (decode). `-o json`.
  A row with `n_prompt>0 && n_gen==0` -> pp tok/s = `avg_ts`; `n_gen>0 && n_prompt==0` -> tg tok/s =
  `avg_ts`. Keeps `samples_ts`/`stddev_ts`. **Excludes tokenization+sampling.** TTFT is NOT reported -
  derive `TTFT ~= n_prompt / pp_avg_ts` (a proxy, ignores tokenize + first-sample + launch).
  ```bash
  build/bin/llama-bench -m model-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 5 -o json
  ```
- **llama-cli** (real e2e incl. measured TTFT proxy). Prints `llama_perf_context_print` to STDERR:
  `prompt eval time` -> TTFT proxy + prefill rate (`/ N tokens`); `eval time` -> decode (`/ N runs` +
  tok/s); `load time` (exclude); `total time` (e2e). Regex on the metric names. Use for measured TTFT.
- Use llama-bench JSON for throughput and llama-cli stderr for measured TTFT; do not mix them for one metric.

## Models: GGUF-only

Download a pre-quant GGUF (ggml-org / bartowski / unsloth) or convert:
`python convert_hf_to_gguf.py HF_DIR --outfile m-f16.gguf --outtype f16` then
`llama-quantize m-f16.gguf m-Q4_K_M.gguf Q4_K_M`. Decoder-only causal arches only.

- Qwen2/2.5/3 (+MoE), Llama, Mistral: fully supported.
- **gpt-oss: native MXFP4 GGUF** - load the published MXFP4 as the model's full-precision form (re-quantizing
  degrades the FFN). Native on Hopper/Blackwell; Ampere upcasts to bf16 at runtime. The best MoE baseline.
- OLMoE: converts but less proven on b9601 - verify before relying; prefer gpt-oss as headline MoE.
- SmolVLM: supported via `libmtmd` / `llama-mtmd-cli`, needs model GGUF **+ an mmproj-\*.gguf** (vision
  projector). `llama-mtmd-cli -hf ggml-org/SmolVLM-Instruct-GGUF -ngl 99 --image x.jpg -p "..."`.
  **llama-bench does NOT benchmark the vision path** - VLM timing is stderr-only via llama-mtmd-cli.

## Precision matching = APPROXIMATE (k-quants, not GPTQ/AWQ)

llama.cpp quants are k-quants (`Q*_K_*`) / i-quants (`IQ*`), a different algorithm than GPTQ/AWQ. Match on
nominal bit-width, NOT algorithm:

- GPTQ-int4 -> closest `Q4_K_M` (~4.5bpw) - caveat: different quant algorithm, not bit-identical.
- AWQ-int4 -> `Q4_K_M` or `IQ4_NL`+imatrix - caveat: different algorithm.
- fp16/bf16 -> **`F16`/`BF16` GGUF directly** (no quantize step) - the fair full-precision anchor.

The suite runs a full-precision tier (F16/BF16 GGUF, comparable) and a 4-bit tier (Q4_K_M, labeled "k-quant,
algorithmically distinct from GPTQ/AWQ, matched on bit-width only"). Only bit-width parity is claimed.
gpt-oss MXFP4 is the one native match.

## VRAM

Report peak `nvidia-smi memory.used` during steady-state decode on a dedicated pod (the suite's external
sampler does this), cross-checked against llama.cpp's stderr buffer report (`CUDA0 model buffer size = ...`

- KV + compute). Always `-ngl 99`.

Runner: a `run.py` here wraps llama-bench (json) + llama-cli (stderr TTFT) and emits the suite JSON line.

## First-pod validation (2026-06-12, RTX 3090)

- **Use `llama-bench`, not `llama-cli`.** llama-cli enters interactive conversation mode for instruct
  models (prints `> ` prompts, loops on stdin EOF, hangs) even with `-no-cnv`. The runner uses
  `llama-bench -o json`; set `LLAMA_BENCH_BIN`. Validated: 456 tok/s f16 decode on Qwen2.5-0.5B.
- Build worked from the cuda12.4.1-devel image with `-DCMAKE_CUDA_ARCHITECTURES=86` (sm_86 = 3090).
- GGUF conversion: `python llama.cpp/convert_hf_to_gguf.py <hf_dir> --outfile m-f16.gguf --outtype f16`
  needs the `gguf` python pkg (ships in the vLLM venv). The runner finds the single .gguf in the model dir.
