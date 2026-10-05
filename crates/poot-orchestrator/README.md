# poot-orchestrator

A local, resumable runner for poot benchmark sweeps on rented RunPod GPUs. It runs end to end:

1. **build** the poot bench runner from a git ref (patchelf'd for a stock-glibc pod),
2. **provision** a RunPod GPU pod (retrying across attempts / GPU types),
3. **set it up** (upload the binary, sync the current harness, `hf download` each model, convert GGUFs),
4. **sweep** the decode-curve scenario per model, in `tmux` on the pod, streaming live progress,
5. **gather** the result snapshots back into `benchmarks/results/` (the Docusaurus dashboard auto-ingests
   that dir), regenerate `results/INDEX.md`, optionally commit,
6. **tear the pod down** - always, even on failure.

It is a CLI you run locally in `nix develop`, which provides `cargo`/`pootc` to build the runner and
`ssh`/`scp`.

## Resilience

- **Durable state.** Every phase transition and per-model flag is committed to a SQLite DB
  (`$XDG_STATE_HOME/poot-orchestrator/state.db`, else `~/.local/state/poot-orchestrator/state.db`; override
  with `--state-db` / `PBO_STATE_DB`, and `status` prints the resolved path). If the orchestrator - or the
  whole machine - dies, the next `run` reads the state back and continues.
- **One state dir per machine, not per checkout.** `reap` can only terminate pods it can see, so a
  per-checkout DB would hide one checkout's pods from another checkout's reap. Progress logs sit next to
  it in `logs/`.
- **The sweep survives a crash.** It runs in `tmux` on the pod, so it keeps going while the orchestrator is
  down. Resume just reattaches to the running session and keeps streaming progress.
- **No dangling pods.** The pod id is persisted the instant the pod is created. Every `run` starts by
  reaping orphans (any pod with the `poot-bench-` name prefix that is not tied to an in-progress run), and
  the pod is always torn down at the end. `reap` cleans up manually. If the machine dies and never comes
  back, the pod lives until the orchestrator runs again (RunPod on-demand pods have no native TTL); run
  `reap` from anywhere with the API key.

## Usage

```sh
export RUNPOD_API_KEY=...                      # a RunPod API key (pod create/terminate)

# full sweep of all poot-supported models at HEAD, commit the snapshots:
cargo run -p poot-orchestrator --release -- run --commit

# specific models / GPU / ref:
cargo run -p poot-orchestrator --release -- run --models qwen2.5-0.5b,qwen3-0.6b --gpu-types "NVIDIA GeForce RTX 3090"
cargo run -p poot-orchestrator --release -- run --git-ref v0.3.0          # builds that ref from a clean worktree

# five repeats of the matrix per model (`<run-id>-r1` .. `-r5` under benchmarks/results/), the baseline `bench compare` wants:
cargo run -p poot-orchestrator --release -- run --models qwen2.5-0.5b --repeats 5

# build + validate config without touching RunPod:
cargo run -p poot-orchestrator --release -- run --dry-run

# clean up orphaned pods (safe anytime); --all force-kills every poot-bench-* pod:
cargo run -p poot-orchestrator --release -- reap
cargo run -p poot-orchestrator --release -- reap --all

# what runs exist + their phase:
cargo run -p poot-orchestrator --release -- status

# web dashboard (auto-refreshing runs + phases + per-model progress + live logs) at http://127.0.0.1:8787:
cargo run -p poot-orchestrator --release -- serve            # or: just sweep-ui
cargo run -p poot-orchestrator --release -- serve --addr 0.0.0.0:8787   # bind all interfaces

# build (+ smoke, then optionally push) the bench container image via podman, streaming progress to an observable log:
cargo run -p poot-orchestrator --release -- image --dockerfile docker/Dockerfile --tag latest --push   # or: just sweep-image ...

# tail a run/build log captured by the orchestrator (no name = list them):
cargo run -p poot-orchestrator --release -- logs image-latest.log --follow   # or: just sweep-logs ...

# one-off: rent an NVIDIA pod, scp a binary, run a command, tear down (PTX/CUDA verification, not a sweep):
cargo run -p poot-orchestrator --release -- exec --image ghcr.io/kikijiki/poot-bench:latest --bin <binary> --cmd '<args>'
```

Fire it in the background and it will keep going; re-run the same `run` command after a crash to resume.

Key flags (see `run --help` for all): `--git-ref`, `--models`, `--scenario`, `--gpu-types` (tried in
order), `--cloud` (COMMUNITY/SECURE/ALL), `--image`, `--ssh-key`, `--provision-attempts`,
`--sweep-timeout-s`, `--keep-pod` (debug; the pod is still reaped next run), `--poot-bin` (skip the build),
`--commit`, `--dry-run`.

The `exec` subcommand is a one-off pod runner that reuses the provision/teardown machinery of `run` but
skips the bench sweep: it rents an NVIDIA pod, optionally scp's a `--bin` (patchelf'd to the stock glibc
loader) plus repeatable `--upload local:remote` files, runs an optional `--setup` then the required `--cmd`
(stdout/stderr streamed back), and tears the pod down. It is the standard on-demand path for PTX/CUDA
verifications on real NVIDIA. Useful flags: `--gpu-count N` (multi-GPU, e.g. tensor-parallel), `--keep-warm <minutes>` (the next `exec` with the
same image adopts the warm pod, skipping provision + the slow image pull), `--gpu-types`, `--cloud`,
`--image`, `--min-cuda`, `--env KEY=VAL` (repeatable create-time container env; `exec` also defaults
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` on a fresh create). Warm adoption skips
create-time env. Create-time caps are required for the NVIDIA Vulkan ICD path; a later SC-003 pod
receipt is still required to prove they unlock wgpu.

`exec` resolves `--image` (default `ghcr.io/kikijiki/poot-bench:latest`; a name with no registry host is
on Docker Hub) to a digest before it records the run or creates a pod, and the pod runs the
digest reference, so a tag that moves mid-run cannot change the image. A refused or failed resolution stops
`exec` before any pod exists.

The `image` subcommand runs a no-GPU smoke check after a build, and before a push when `--push` is used. It
verifies the Vulkan loader (`libvulkan.so.1`), `vulkaninfo`, the transformer import, and the baked runner
binaries, so image failures are caught before paid pods use the tag.

## How a resume works

`run` looks for an in-progress run in the DB. If found:

- pod recorded and still alive -> **reattach** (reuse the endpoint; skip the per-model setup and sweeps already done),
- pod recorded but gone -> **abort** that run (its in-flight sweep is lost) and start fresh,
- no pod yet -> **reprovision** (continue from the pod phase).

Per-model `setup_done` / `sweep_done` / `gathered` flags make every phase idempotent, so a resume only does
the work that is left.

## Web dashboard

`serve` starts a read-only web UI (default `http://127.0.0.1:8787`) over the same SQLite state DB. It is a
React 19 + recharts single-page app (source in `ui/`, built by Vite into one inlined offline HTML) that polls
`GET /api/state` every few seconds and renders each run with its phase, pod, SSH endpoint, GPU, and
per-model `setup`/`sweep`/`gather` progress. It only reads (WAL allows concurrent readers), so it is safe to
run in a second terminal during a `run`. Endpoints: `/` (page), `/api/state` (JSON), `/healthz`.

## Design notes

- RunPod is reached via its REST API (`https://rest.runpod.io/v1`, Bearer auth); the agent-facing MCP is
  not available to a standalone program.
- The dashboard uses `tiny_http` (blocking, no tokio) to match the crate's synchronous design. The served
  page is `src/dashboard.html`, baked in with `include_str!`. It is the committed build output of the `ui/`
  React app (Vite + vite-plugin-singlefile); regenerate it after editing `ui/` with
  `cd ui && npm install && npm run build && cp dist/index.html ../src/dashboard.html`. There is no
  justfile recipe for this; the Rust crate itself has no build step.
- The decision logic (manifest parse, model selection, reap targeting, resume) is pure and unit-tested
  (`cargo test`); the I/O wrappers (`db`, `runpod`, `ssh`) are thin.
- Part of the poot workspace. Run from the repo root as
  `cargo run -p poot-orchestrator [--release] -- <subcommand>`.

## Network-volume model cache (optional; currently unused)

Not used by default: the bench models are small (qwen3-0.6b downloads in ~15s on the pod), and a volume is
datacenter-locked while SECURE sm_86 stock floats between datacenters, so pinning to one DC fails to
provision whenever that DC is out of stock. It remains an option for large models, where the per-pod
download would dominate.

The per-pod `hf download` runs on the billed pod every run. Instead, keep the prepared weights on a RunPod
network volume (mounted at `/workspace`, ~$0.07/GB/month) and attach it at pod-create. Populate it once,
pod-less, over the RunPod **S3-compatible API** (`just volume-sync <vol> <dc> <models-dir> [model...]`,
which `aws s3 sync`s to `s3://<vol>/models/<id>/`; needs `RUNPOD_S3_ACCESS_KEY` + `RUNPOD_S3_SECRET_KEY`).
Then run with `--network-volume-id <id> --data-center <dc>`: setup skips the download for any model already
present and the sweep reads `/workspace/models`. Volumes are datacenter-locked, so the pod pins to that DC
and the chosen GPU types must have SECURE stock there (drop the flags to fall back to download-on-pod).

## Backlog

- **Daemon/service mode.** The phase state machine + SQLite store already support it; a long-running mode
  could expose status + accept enqueued sweeps.

## Status

Unit tests pass (`cargo test`) and clippy is clean. Validated live against real RunPod pods: provision +
retry, SSH bring-up, binary/harness upload, model download, the tmux sweep, and teardown (no leaked pods
across many runs). S3 volume populate + attach verified. Not yet validated: one fully green sweep
(`SWEEP_EXIT=0`) plus the kill/restart reattach, on the slim image + the volume.
