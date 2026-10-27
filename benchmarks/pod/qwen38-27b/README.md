# POOT-1184: Qwen3.8-27B-FP8 on one NVIDIA pod (poot PTX vs vLLM vs llama.cpp)

One serial pod session (POOT-ADR-0092). Nothing here creates a pod: `launch.sh start` does, and only the orchestrator
runs it, after stating the hourly price.

| file                                | where | what                                                                                                                                                                                                          |
| ----------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stage.sh`                          | local | release-builds `poot-bench-runner` + `poot-serve` (-j4, MemoryMax 20G), patchelf's them, builds the nextest archive of the oracle row and `ws.tar` into `$STAGE` (default `/tmp/claude-1000/poot-1184-stage`) |
| `launch.sh`                         | local | `print` the exact `poot-orchestrator exec` command, `start` it detached (setsid nohup), `extract LOG DEST` the returned tarball                                                                               |
| `pod-sweep.sh`                      | pod   | the whole sweep; detached on the pod; one dir of results                                                                                                                                                      |
| `summarize.py`, `oracle_fixture.py` | pod   | flatten results to `summary.tsv`; turn the oracle row's RECEIPT lines into the fixture                                                                                                                        |
| `dry-run.sh`                        | local | no GPU/pod/network proof (see below)                                                                                                                                                                          |

## Run it (orchestrator)

```
benchmarks/pod/qwen38-27b/stage.sh            # once; binaries already staged for BUILD_SHA in $STAGE/BUILD_SHA
export RUNPOD_API_KEY=...                      # then, after stating the price:
benchmarks/pod/qwen38-27b/launch.sh print      # review
benchmarks/pod/qwen38-27b/launch.sh start      # pod id: grep 'pod .* created' /tmp/claude-1000/poot-1184-exec.log
# live progress (the exec only returns at the end): ssh -i ~/.ssh/id_ed25519 -p PORT root@HOST tail -f /root/results/progress.log
benchmarks/pod/qwen38-27b/launch.sh extract /tmp/claude-1000/poot-1184-exec.log /tmp/claude-1000/poot-1184-out
# then TEAR DOWN BY ID (mcp delete-pod <id>); the launcher passes --keep and --keep-warm 120 so a dropped SSH never
# deletes the results, which also means the pod bills until you delete it.
cp /tmp/claude-1000/poot-1184-out/results/crates/poot-llm/tests/data/qwen38_27b_fp8_oracle_top5.json crates/poot-llm/tests/data/
```

GPU: `NVIDIA H100 80GB HBM3` (secure, $3.99/h, stock Low), then `NVIDIA A100-SXM4-80GB` (secure, $1.79/h). The sweep
records which card it got in `env.txt`; an A100 has no FP8 tensor cores, so its numbers are not an H100 comparison
(the note is written into `env.txt`). Host CUDA >= 13.0 is requested (`MIN_CUDA`) because the image's vLLM 0.22.1 is a
cu130 wheel. Container disk 200 GB.

## What the sweep measures (all matched: ISL 128/512/2048/8192, OSL 128, greedy, prefix cache off, one card)

- **poot (PTX), batch 1**: `poot-bench-runner --mode decode-curve` (prefill exactly ISL tokens, decode OSL, warmup 2,
  5 timed). `poot_curve` = clean timing; `poot_curve_roofline` = `--roofline` (driver phase timing, bytes/token, fraction
  of the bandwidth ceiling); `poot_curve_profile` = `--profile` (device time per phase and kernel, in `logs/*.err`;
  diagnostic, perturbs submission). Prefill tok/s = ISL / TTFT.
- **batched decode (all three engines, one client)**: `vllm bench serve` against each engine's OpenAI server
  (poot-serve `--backend ptx`, `vllm serve`, `llama-server -np 32`): conc 1 at each ISL, then conc 4/16/32 at ISL
  512/2048, 64 requests per cell, `--ignore-eos`; every row is capped at `ROW_CAP_S` (300 s; a capped row is rc=124 and
  `--save-detailed` keeps per-request errors). vLLM and llama.cpp are sized 32 sequences x 8384 tokens. **poot-serve is
  not**: it plans 4 rows x 4096 positions (`logs/serve_poot.out` of run 1), so its grid is cut to ISL 128/512 at conc 1
  and ISL 512 at conc 4 (8 prompts, OSL 32), and is NOT a matched comparison.
- **order** (cheap and decisive first): downloads, oracle fixture, vLLM, llama.cpp, poot curve + roofline + profile,
  poot-serve rows last.
- **vLLM**: baked `vllm==0.22.1` venv, FP8 as stored, `--no-enable-prefix-caching`, mm profiling off, gpu-util 0.90.
- **llama.cpp**: tag v0.4.1 built on the pod for the pod's arch (the image's CUDA 13 toolkit), `unsloth/Qwen3.8-27B-GGUF`
  `UD-Q4_K_XL` (Qwen/ publishes no GGUF); `llama-bench` pp at each ISL and tg128 at depth = ISL, `-fa 1 -b/-ub 2048`.
- **peak memory**: nvidia-smi sampled every 0.5 s per phase (`phases.tsv`, MiB) - not engine-reported.
- **oracle fixture**: the `qwen35_27b_fp8_receipt::ptx` row (CPU oracle + device, needs the large pod host RAM) ->
  `crates/poot-llm/tests/data/qwen38_27b_fp8_oracle_top5.json` (under the results dir) (prompt, ids, generated ids, oracle and device first-token top-5 ids +
  logprobs). **Top-50 / full logits are not produced**: the row prints top-5 only; a wider dump needs an edit to
  `generates_and_matches_the_oracle` in `crates/poot-llm/src/driver/device_tests.rs`, outside this card's scope.

Everything lands in `/root/results` (`results.tgz`): `env.txt`, `versions-*.txt`, `model-revisions.txt`,
`commands.tsv` (exact command of every phase), `phases.tsv` (rc, wall s, peak VRAM), `summary.tsv`, `raw/`, `logs/`.
`PHASES=vllm` (space-separated groups: `oracle vllm llamacpp poot`) reruns only that engine's phases.
A failed phase is a recorded row, not an abort; `MAX_TOTAL_S` (4 h) stops new phases from starting.

## Caveats the report must carry

- llama.cpp is now built with the image's CUDA 13.0 toolkit (earlier runs used 12.6): llama.cpp rows from this change on
  are not like-for-like with earlier ones. The toolkit version is recorded in `versions-llamacpp.txt` (`llamacpp_cuda`).
- poot's curve context is a repeated phrase cycled to exactly ISL tokens; the shared client sends random tokens
  re-tokenized by each server (counts can drift a few tokens). poot-serve has no ignore-eos switch: `summary.tsv`
  flags rows whose mean output is under 95 % of OSL.
- poot runs the FP8 checkpoint (`--precision bf16` names the compute dtype); llama.cpp runs a 4.5-bit K-quant: not
  the same precision, never present as clean apples-to-apples.
- llama.cpp v0.4.1 / vLLM 0.22.1 support for this architecture is assumed from the image pins; if a server or bench
  refuses, the failure is the recorded result (`logs/*.err`).

## Dry run (done, no GPU/pod/network)

`dry-run.sh <runner> <serve> <orchestrator>`: 43 checks. The launcher's argv is parsed by the real clap parser (stopped
at the missing API key); the real runner accepts the exact curve/roofline/profile flags and rejects unknown ones; the
real `poot-serve` takes `--backend ptx MODEL ADDR`; the whole of `pod-sweep.sh` runs against mock engines (phases,
VRAM sampling, budget skip, summarize, oracle fixture, tarball) and `launch.sh extract` round-trips the tarball. Mock
engines prove control flow and file shapes only, not any engine's real JSON (vLLM / llama.cpp output schemas are
assumed from their documented formats: the first real run is the check).

## Expected runtime and cost

| step                                        | estimate                                   |
| ------------------------------------------- | ------------------------------------------ |
| provision + image pull + SSH                | 5-30 min                                   |
| uploads (~120 MB)                           | 1-3 min                                    |
| downloads (28 GB FP8 + 17.6 GB GGUF)        | 5-15 min                                   |
| oracle row (load + CPU oracle)              | 10-40 min                                  |
| poot curve + short roofline + profile       | ~65 min (run 1 measured 44 + 19 min)       |
| poot-serve rows (cut grid, 300 s cap each)  | <= 20 min                                  |
| vLLM (load, compile, 10 cells)              | 25-40 min                                  |
| llama.cpp (apt + build, bench, server grid) | 35-60 min                                  |
| **total**                                   | **~3-4.5 h** (cap: no new phase after 4 h) |

H100 SXM $3.99/h: **~$12-18** (expect ~$14). A100 SXM $1.79/h: ~$6-9. The keep-warm/keep window bills until the pod
is deleted: delete by id as soon as `extract` succeeds. These are estimates, not measurements.

## Run 1 findings (2026-10-18, H100, poot b00cb7feb; its snapshot `2026-10-18-h100-qwen3-8-27b-fp8-ptx-b00cb7feb` was removed from `benchmarks/results/` on 2026-10-20: recorded before the identity contract, it is in git history)

- **poot PTX batch 1**: decode 8.2 tok/s (TPOT 122 ms, flat from ISL 128 to 8192), prefill ~35 tok/s (ISL 8192 = 234 s),
  peak 39.5 GiB. 122 ms per token for ~28 GB is ~230 GB/s of a ~3.35 TB/s card: a poot performance problem, not a
  hardware limit. Poot only; no baseline ran.
- **oracle row**: failed because `oracle-lane.tar.zst` was repacked with `tar -c .`, whose `./` entry nextest rejects
  ("path in archive `./` doesn't start with `target/`"). `stage.sh` now packs `tar -c target`; `dry-run.sh` extracts the
  staged archive with nextest and lists the row.
- **poot-serve rows**: the engine holds 4 rows x 4096 positions, so ISL 8192 is rejected by design (`400 ... does not
fit the engine's 4096 positions`), and the c4/c16/c32 cells at 1800 s each burned the budget. At prefill 13-35 tok/s
  and 255 ms/token a 64-request cell cannot finish. The failures at ISL 128/512/2048 left no error (vllm bench serve
  drops it without `--save-detailed`); that flag is now on, so the next run records the cause.
- **roofline `None`**: not a bug in the extraction. The runner reports `{kind: unknown, reason: "no pinned ceiling for
this device"}` (no calibrated ceiling row for the H100 in `crates/poot-profile/data/ceilings.tsv`), and the fit is
  `unknown` (needs two points with different bytes per token). `summarize.py` now prints that reason.
- `ingest.py` turns a sweep dir into a `results/<run-id>` snapshot through the harness's own result contract.

## Run 2 findings (2026-10-18, H100, same poot build; snapshot `2026-10-18-h100-qwen3-8-27b-fp8-ptx-b00cb7feb-run2`, removed likewise, is in git history)

- **oracle row passed** (3623 s): fixture committed as `crates/poot-llm/tests/data/qwen38_27b_fp8_oracle_top5.json`. ROCm and PTX device
  top-5 agree to 9e-6 and both sit a near-constant ~7e-4 below the CPU oracle (inside the 1e-3 criterion, thin margin).
- **llama.cpp** (Q4_K_XL, not the same weights): tg 79.6 tok/s, pp 2500-2800 tok/s vs poot 8.18 / ~35 (POOT-1265).
- **vLLM did not start**: DeepGEMM's JIT needs nvcc, the image has none (POOT-1269).
- poot-serve: c1 rows lose 1 of 3 requests, c4 hits the 300 s cap; llama.cpp ISL 2048 c4 also hit it.
