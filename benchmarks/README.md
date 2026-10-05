# poot benchmark suite

Cross-framework LLM inference benchmarks: run the same model through poot and a set of baseline engines on
the same GPU, measure latency / throughput / VRAM / host memory with one uniform method, and store the
results so poot's performance can be tracked over time. The headline is a **context-degradation curve** -
how each engine's decode speed changes as the KV cache grows (poot's naive attention should bend up with
context; the flash/paged baselines stay flat).

The harness measures single-stream decode (see "Single-stream only"); the point is a reproducible number
that shows where poot stands and whether an optimization moved it.

Baselines: **candle** (Rust peer), **PyTorch + HF transformers** (reference), **vLLM** (production
server), **llama.cpp** (GGUF/edge). Precision is matched per model where the baseline supports it; every
gap is recorded as a caveat in the results.

## Results + tracking progress over time

Every run is a committed snapshot under `results/<run-id>/` (`env.json` + `results.jsonl` + `report.md`),
and **`results/INDEX.md`** is the progress index. The report tool renders the degradation curve
(decode tokens/s and TPOT vs context length, per engine). `bench compare --baseline <runs> --candidate
<runs>` decides whether a change moved a number: each side is 5 repeats of the same matrix (see "Compare two
builds"). The index is generated from the snapshots: regenerate it with `bench index` after each new run;
the harness tests fail when `results/INDEX.md` differs from what `bench index` writes.

Tracking restarted on 2026-06-13: the earlier `decode-128` snapshots are not compatible with the
`decode-curve` methodology (different scenario, prefill-amortized vs prefill-excluded decode metric, text
vs fixed-token input). They remain in git history (the 2026-06-12 commits) but not in the working tree.

## Layout

```
benchmarks/
  manifest.toml        the model x framework x scenario matrix (what gets benchmarked; precision per cell)
  runners.toml         how the harness invokes each runner (argv templates, env-overridable binaries)
  prompts/             fixed prompt + image inputs (short.txt, long.txt, test-image.jpg)
  schema/              result.schema.json - the one record shape every row conforms to
  pyproject.toml       the harness deps (uv); requirements are light (pynvml, psutil) and degrade gracefully
  harness/
    bench.py           the orchestrator CLI: matrix / run / report / compare / index
    observer.py        the external sampler (uniform VRAM + RSS) every runner is wrapped with
    fixtures/          a synthetic run snapshot for the hermetic `report` self-check
    tests/             the harness's own tests (`just harness-test`)
  runners/
    poot/              the poot runner: root-workspace crate `poot-bench-runner`, one measure loop over
                       the ptx, rocm and wgpu backends
    candle/  torch/  vllm/  llamacpp/    one runner + a NOTES.md (pinned versions, gotchas) each
  setup/               per-framework, pinned, on-pod install scripts (the "pull everything" part)
  docker/              the pre-baked image: Dockerfile (bakes the 5 engines, sm_86) + README
  flake.nix            a separate devshell with the image build/push tooling (podman, skopeo, gh, uv)
  justfile             image-build / image-smoke / harness commands (run in `nix develop`)
  results/             committed run snapshots: results/<date>-<gpu>-<sha>/{env.json,results.jsonl,report.md}
```

## The measurement contract (why the numbers are fair)

The metric definitions follow MLPerf Inference, vLLM `bench serve`, NVIDIA GenAI-Perf, and Anyscale LLMPerf. The older `decode-128` contract, which
amortized prefill into the decode rate, is superseded.

**Scenario `decode-curve`.** For each context length ISL in {128, 512, 2048, 8192, 16384, ...}, prefill
exactly ISL tokens, then decode a fixed small OSL (large enough for a >1s window and a stable per-token
rate). The curve of decode speed vs ISL is the result. (`decode-128` / `prefill-1k` remain as single-point scenarios.)

**Work-matched, tokenizer-neutral input.** Each engine is driven with exactly ISL _synthetic random
in-vocabulary tokens_ (no chat template), so every engine processes the identical token count. The same
text tokenizes to different counts per engine (a ~50% gap has been measured), so the suite fixes tokens, not text. Random tokens also defeat prefix caching, which is disabled anyway (it injects
order-dependent TTFT variance).

**Metrics, identical across runners, prefill separated from decode:**

- **TTFT** = request start to first generated token. The one-time prefill cost; grows with ISL.
- **ITL_i** = gap between decode token i-1 and i. Reported as p50 / p90 / p99.
- **TPOT** = (e2e - TTFT) / (OSL - 1) = mean decode ITL, prefill EXCLUDED. (MLPerf/Anyscale convention;
  the `OSL-1` denominator is stated in output, not vLLM's historical `OSL`.)
- **Decode tokens/s (steady state)** = 1000 / median TPOT_ms, the headline decode number. Not
  `gen_tokens / e2e`, which folds the one-time prefill into the per-token rate.
- Report percentiles (p50/p90/p99), not just means; windows are kept >1s (sub-second windows are
  dominated by startup / CUDA-graph warmup / clock resolution). CUDA-async engines synchronize at the same
  points (after prefill for TTFT, around each decode token for ITL, after the last for e2e) or the
  per-token timing is wrong.

**Single-stream only.** One request in flight; no concurrency / QPS / continuous-batching / goodput
(single-process load generators saturate at high QPS). vLLM's batching advantage is a caveat, not exercised.

Memory is NOT self-reported. The harness wraps each runner with `observer.py`, which samples total device
VRAM (NVML, else `nvidia-smi`) and process-tree RSS (psutil, else /proc) at a fixed interval and keeps the
peaks. Self-reported numbers do not compare across engines (torch's
allocator, vLLM's KV pool, and llama.cpp's buffer report measure different things). The
assumption is **one engine per GPU, GPU otherwise idle**, so "peak device used minus the pre-launch
baseline" is the footprint. Run one cell at a time on a dedicated pod.

Each local cell runs in its own POSIX session. If the cell timeout expires or the observer is interrupted,
the observer sends `SIGTERM` to that session's process group, escalates to `SIGKILL` after a grace period,
and reaps the runner after final group signaling. Normal completion also closes the group before reaping its
leader. Sampling runs in a separately reaped worker session, so a blocked driver/tool call cannot leave a live
sampler or tool descendant behind. The observer uses a waitable `SIGCHLD` disposition while its leaders are
pinned, then restores the caller's. Timeout rows are labeled separately from runner failures, including when a
timed-out runner reports return code zero.

Recording rules (spec 112 FR-005/FR-007/FR-009): a framework that cannot run a model is recorded as
an explicit `unsupported` row with a reason (never silently skipped); a failed run is an `error` row (the
sweep continues). For poot there is no support list: the harness runs every poot cell, and when the runner
refuses a model (a typed planning refusal, ADR 0104) it reports `"status": "unsupported"` with the refusal's
own text, which becomes the row's `reason`. The other frameworks are external tools, so their known gaps
are `support = false` entries with a checked `reason` in `manifest.toml`. Any precision/format mismatch vs the matched target (llama.cpp k-quant vs GPTQ, vLLM
KV-pool VRAM, gpt-oss MXFP4 falling back to bf16 on non-Hopper) is a `caveat` on the row, never presented
as clean apples-to-apples.

Runner JSON is not trusted because the process exits zero. Before the harness creates an `ok` row, a
scenario-specific contract checks the requested framework and precision, finite timings, positive sample
counts, and the requested token workload. A degradation curve must contain every requested ISL exactly once,
with complete per-iteration samples for the requested OSL. Invalid or partial results become `error` rows
with bounded runner and observer diagnostics; manifest and runner caveats stay visible. The contract applies
to newly run cells only; historical snapshots are not re-validated.

## Quick start

### See the matrix (no run, no deps)

```bash
cd benchmarks
python3 harness/bench.py matrix                 # every cell; support = no only where a tool's gap is known
python3 harness/bench.py matrix --model qwen2.5-0.5b
```

### Read committed results (pure stdlib, no GPU)

```bash
python3 harness/bench.py report results/<run-id>
python3 harness/bench.py index                  # regenerate results/INDEX.md
```

The report and index paths have no GPU or framework dependency - they run anywhere, including CI. The
bundled fixture exercises `report`:

```bash
python3 harness/bench.py report harness/fixtures/run-a
```

### Compare two builds

A speed change is only real if it clears the noise, so `compare` takes repeats, not single runs: at least 5
run directories per side, each a full run of the same matrix, all of one binary per side. Run the matrix 5
times per build with distinct run ids, then:

```bash
for i in 1 2 3 4 5; do uv run bench run --framework poot --model qwen2.5-0.5b --run-id base-$i; done
# ... rebuild, then the same loop with --run-id cand-$i ...
python3 harness/bench.py compare --baseline results/base-{1..5} --candidate results/cand-{1..5}
```

A cell is keyed by `(model, scenario, framework, backend, device)`. The band is the baseline median plus or
minus its range; a candidate median more than one further range below it is a regression. The exit code is 1
for a regression, a failed candidate cell or a non-finite sample, 2 when a cell has too few repeats, mixes
binaries, or nothing could be compared, and 0 otherwise. Speed is recorded, never gated.

### Run it (needs a GPU + the runners built/installed)

```bash
cd benchmarks && uv sync                        # harness venv (pynvml, psutil)
# Build/install the runners you want (see setup/ and runners/*/NOTES.md), then:
uv run bench run --model qwen2.5-0.5b --scenario decode-128 \
    --framework poot --models-dir /path/to/checkpoints   # or set POOT_MODELS_DIR
# Writes results/<date>-<gpu>-<sha>/{env.json, results.jsonl, report.md}
```

Filter with `--model`, `--framework`, `--scenario`; `--skip-unsupported` to omit the unsupported rows;
`--sample-interval-ms` for the memory sampler period. Each runner binary is found via its `bin_env`
(`POOT_BENCH_POOT_BIN`, `POOT_BENCH_CANDLE_BIN`, `POOT_BENCH_TORCH_PYTHON`, `POOT_BENCH_VLLM_PYTHON`,
`POOT_BENCH_LLAMACPP_PYTHON`, plus the llama.cpp binary env in its runner) so you don't edit `runners.toml`
to retarget a build.

### Backends and the local AMD box

The poot runner has three backends, chosen per run by `POOT_BENCH_BACKEND=ptx|rocm|wgpu` (default `ptx`,
passed to the runner as `--backend`); the result row records the backend and the device it ran on, and a
compare never merges two of them. `ptx` needs NVIDIA hardware and libcuda. `rocm` and `wgpu` run on the AMD
dev box (Strix Halo: ROCm/HSA and Vulkan), where VRAM is shared host memory, so the memory numbers read
differently from a discrete GPU. On such a box:

```bash
POOT_BENCH_BACKEND=rocm uv run bench run --framework poot --model qwen2.5-0.5b --models-dir ~/models
benchmarks/run-local-compare.sh --model-dir ~/models/qwen2.5-0.5b   # rocm vs wgpu, one table, no snapshot
```

Run one GPU job at a time on that box; a cell that does not fit in memory should be recorded as an error, not
retried next to another job.

## Running the sweep on RunPod (pre-baked public image)

The headline numbers use NVIDIA hardware (RTX 3090 or similar). Installing on a pod is slow: a community pod's PyPI/cargo mirror can throttle to
~850 KB/s even when raw bandwidth is 32 MB/s, and rustup corrupts under concurrent installs / on overlayfs
(a first attempt spent ~90 minutes on installs). So the five engines are baked into a **pre-baked public
Docker image** and a pod boots ready in ~1 minute.

RunPod always pulls the container image from a registry; the container disk and any volume are separate
storage. The image is published **public on GHCR** (`ghcr.io/<owner>/poot-bench`): public packages have no
storage or egress bill, and the standard RunPod MCP `create-pod` pulls it via `imageName` with no
credential or network volume. The source repo stays private; only the benchmark tooling and public engine
installs are in the image (no poot codegen core, no poot binary). Build and push it from the bench devshell
(`nix develop ./benchmarks`): `just image-build-push` builds, smokes, and pushes locally (or
`just sweep-image --push` via the orchestrator); `just image-build` builds only, to validate the Dockerfile. There is no CI workflow that builds the image.
The one-time public-visibility flip and details are in [`docker/`](docker/).

**Baked:** a slim NVIDIA CUDA runtime base with the reimplemented `/start.sh` SSH/`PUBLIC_KEY` machinery,
the five engines pinned to sm_86, the harness venv, Vulkan loader/diagnostic tools for NVIDIA wgpu, and the
suite source. **Not baked:** the model weights (fetched at runtime on the pod; avoids CI disk limits and
model-redistribution licensing) and the poot binary (changes every version; scp'd fresh per run, then
`patchelf`'d). The `setup/*.sh` scripts remain the source of truth for
what the image installs.

**Running a sweep.** `create-pod` (standard MCP) with `imageName: ghcr.io/<owner>/poot-bench:latest`,
`gpuTypeIds: ["NVIDIA GeForce RTX 3090"]`, `ports: ["22/tcp"]`, `env: { PUBLIC_KEY: <pubkey> }`. SSH in,
fetch the eval models (HF -> pod, fast in-cluster), scp + `patchelf` the fresh poot binary, set the runner
env vars, and run `uv run bench run ...`.

- **Default: single pod, all engines sequentially.** Every engine runs on the same physical GPU. Record GPU
  UUID + enforced power limit once.
- **Optional: multi-pod fan-out** (faster wall-clock): one 3090 pod per engine, provisioned with concurrent
  `create-pod` calls (there is no bulk endpoint). Each pod records GPU UUID + power cap + a fixed
  **calibration micro-bench** to quantify cross-machine variance; the curve shape is reliable, absolute
  cross-engine tok/s at a given context carries a "different physical 3090s" caveat.

**Operational gotchas.**

- **Detach long runs with `tmux`, not `nohup`.** A poot cell at high context can run for hours; plain
  `nohup ./run-sweep.sh &` over SSH is flaky (the process can die when the session closes, and rapid
  `pkill`/relaunch leaves an orphaned `poot-bench-runner` with no observer). Run inside tmux:
  `apt-get update && apt-get install -y tmux` (the image clears apt lists), then
  `tmux new-session -d -s poot "bash /root/run-poot.sh"` where the script sets the env and redirects to a
  log. RunPod's own advice: <https://docs.runpod.io/tips-and-tricks/tmux>.
- **Get the SSH endpoint from `list-pods`, not `get-pod`.** This MCP often leaves `publicIp` blank on
  `get-pod`; `list-pods` surfaces `portMappings` (`{"22": <port>}`) + `publicIp` once the pod is RUNNING.
- **llama.cpp needs a GGUF in the model dir.** Convert on the pod (the safetensors checkpoint is what we
  scp): `uv pip install --python /opt/tf-venv/bin/python gguf sentencepiece`, then
  `/opt/tf-venv/bin/python /opt/llama.cpp/convert_hf_to_gguf.py <model_dir> --outtype f16 --outfile <model_dir>/<id>-f16.gguf`.
  The harness passes `--gguf <model_dir>` and expects exactly one `.gguf` there.
- **poot needs `warmup >= 2`.** With `warmup 1`, one-time CUDA module-load / PTX JIT leaks into the timed
  runs and reads a fraction of the true tok/s. On a pod the runner runs its `ptx` backend (NVIDIA
  capture/replay) and needs libcuda (the pod driver) at runtime; the backend is `POOT_BENCH_BACKEND`, not
  `POOT_TARGET`. Per-call CUDA-graph capture lands in TTFT, not in decode tok/s.
- **poot prefill is naive and serial; cap the ISL if the top point is impractical.** TTFT is super-linear in
  context (qwen3-0.6b: 4 s / 20 s / 133 s at isl 128 / 512 / 2048), so isl=8192 can be a multi-hour cell.
  Raise `POOT_BENCH_CELL_TIMEOUT_S`, or cap poot at a lower ISL (edit the scenario `isl` list) and run
  baselines at full ISL; the poot line just stops early.

**Budget:** confirm spend and tear pods down immediately. The image costs nothing to store (public GHCR);
once a clean baseline curve is committed, a poot-only re-run is a single ~30-40 min pod. The orchestrator
(`crates/poot-orchestrator`) drives the pod over SSH.

## Adding a model or framework

- **A model:** add a `[[model]]` block to `manifest.toml` with `hf_repo`, `local_dir`, `size_class`,
  `match_precision`, and a `frameworks.<fw>` entry per engine (`support` + `precision`, or `support=false`
  - `reason`, for the external engines; a `frameworks.poot` entry is `precision` and an optional `caveat`,
    never `support`). The harness enumerates it automatically.
- **A scenario:** add a `[[scenario]]`. A single-point scenario takes a `prompt_file` + `gen_tokens`; a
  `decode-curve` scenario takes a context-length list (`isl`), a fixed `osl`, and the synthetic-token flag
  (see spec 115).
- **A framework:** add a runner under `runners/<fw>/` that honors the CLI + JSON contract above, a
  `[<fw>]` entry in `runners.toml`, a `setup/setup-<fw>.sh`, and the `frameworks.<fw>` entries in the
  manifest. Capture the version pins + gotchas in `runners/<fw>/NOTES.md`.

## Per-framework gotchas (full detail in each `runners/<fw>/NOTES.md`)

- **poot** - built with `cargo build --release -p poot-bench-runner` (a root-workspace member; GPU kernels are
  generated at runtime). One measure loop serves the `ptx`, `rocm` and `wgpu` backends (`--backend`); each
  backend only names the `Runner` generation entry it uses. Decode-curve contexts are prefilled in one batched
  forward and decoded with EOS ignored for a fixed token count. The runner takes only measurement flags, and
  its one output line is the result: it has no probe or receipt modes. A refused model is reported as an
  `unsupported` result carrying the refusal text.
- **candle** - no GPTQ/AWQ (GGUF k-quants only); no SmolVLM/OLMoE/gpt-oss in 0.10.2. Needs nvcc to build.
- **transformers v5** - AWQ needs a SEPARATE venv (autoawq pins transformers 4.47.1); gpt-oss MXFP4 is
  sm_90+ only (bf16 fallback elsewhere - record the caveat). NVML is the fair VRAM number.
- **vLLM 0.22.1** - offline `metrics` is `None` (TTFT via a prefill-only request); pre-allocated KV pool
  inflates VRAM - run at a low `gpu_memory_utilization` with `max_model_len` pinned, and keep the caveat.
- **llama.cpp b9601** - GGUF-only, needs nvcc; k-quants are not GPTQ/AWQ (bit-width parity only); f16/bf16
  GGUF is the fair full-precision anchor; gpt-oss native MXFP4 GGUF is the one exact match; `-ngl 99`.
