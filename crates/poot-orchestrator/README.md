# poot-orchestrator

A local, resumable runner for poot benchmark sweeps on rented RunPod GPUs. It runs end to end:

1. **build** the poot bench runner from a git ref (patchelf'd for a stock-glibc pod),
2. **provision** a RunPod GPU pod (cycling the GPU types and both clouds, waiting for capacity for as long as it takes),
3. **set it up** (upload the binary, sync the current harness, `hf download` each model, convert GGUFs),
4. **sweep** the decode-curve scenario per model, in `tmux` on the pod, streaming live progress,
5. **gather** the result sets back into `benchmarks/results/`, each as soon as the harness finishes it (the
   Docusaurus dashboard auto-ingests that dir), regenerate `results/INDEX.md`, optionally commit,
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
- **A run always ends in a terminal state:** `done`, `failed` (an error waiting cannot fix), `cancelled`
  (SIGINT/SIGTERM, a panic, `--give-up-after-min`, `--max-replacements`) or `abandoned` (see "Zombie runs").
- **No dangling pods.** The pod id is persisted the instant the pod is created. Every `run` starts by
  reaping orphans (any pod with the `poot-bench-` name prefix that is not tied to an in-progress run), and
  the pod is always torn down at the end. `reap` cleans up manually. If the machine dies and never comes
  back, the pod lives until the orchestrator runs again (RunPod on-demand pods have no native TTL); run
  `reap` from anywhere with the API key.

## Who may adopt or delete which pod

| Pod                                                                              | Adopted by                                                                                                                                                                                                                                                                                                                  | Deleted by                                                                                                                 |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| A run's own pod (`poot-orch-<state id>-<run id>-aN`)                             | that run (resume re-attaches)                                                                                                                                                                                                                                                                                               | that run's teardown; `reap` once the run is terminal and its owner process is dead                                         |
| A pod of a live peer run or exec in the same state DB                            | nobody                                                                                                                                                                                                                                                                                                                      | nobody: `reap` and startup skip it (protected by its run row and live owner)                                               |
| A pod another exec kept with `--keep-warm`, inside its window                    | a later `exec` with the same image, **by design**: `--keep-warm` is the owner's opt-in to sharing the pod once its job is finished (a running job's pod is never a candidate). Candidates are tried newest first; each needs RunPod to list it with an ssh endpoint and ssh to answer (up to 4 probes over about 4 minutes) | nobody                                                                                                                     |
| A kept-warm pod RunPod lists but whose ssh never answers                         | nobody                                                                                                                                                                                                                                                                                                                      | nobody: left alone, marked unresponsive and not probed again for 10 minutes, so it costs one probe round, not one per exec |
| A kept-warm pod RunPod no longer lists, or lists running without an ssh endpoint | nobody                                                                                                                                                                                                                                                                                                                      | no delete is issued: its record is closed (a running pod without an endpoint is left for `reap`)                           |
| A kept-warm pod past its window                                                  | nobody                                                                                                                                                                                                                                                                                                                      | the next `exec` or `reap`, naming the run that kept it                                                                     |

Pods of another state DB (another owner prefix) are never touched. Reaping the orphans of a dead orchestrator is
right; a peer that is alive is not an orphan.

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
order), `--cloud` (COMMUNITY/SECURE/ALL), `--image`, `--ssh-key`, `--repeats`, `--give-up-after-min`,
`--max-replacements`, `--reuse-pod`, `--sweep-timeout-s`, `--keep-pod` (release the pod to idle; a later
`run` with the same image adopts it, otherwise it is reaped), `--poot-bin` (skip the build), `--commit`,
`--dry-run`.

- `--give-up-after-min N` ends the wait for capacity after N minutes: the run is `cancelled`, naming the GPU
  types tried. Without it provisioning never gives up.
- `--max-replacements N` ends the run as `cancelled` after N lost pods have been replaced. Without it
  replacements are unlimited.
- `--reuse-pod <ID>` runs on an existing pod of this state DB instead of provisioning. The pod must be alive and
  not driven by a live run; upload and setup still run, but models already on its disk are not downloaded again.
  Teardown then follows the normal rules.
- `--extend-set <SET_ID>` adds repeats to the sets of an earlier run (its set id, `run-<epoch>-<pid>`, the prefix
  of the sets' pod-side names): pass a larger `--repeats` and the repeats already gathered are not run again.
  Every selected model must already have a gathered set of it. Use the same `--poot-bin` as the first run.
  The pod must have the sets' hardware: once it is up the run compares its GPU name, driver and host CPU model with
  the `env.json` of every set being extended, and on a mismatch ends and terminates the pod with one line saying
  what differs. TTFT and per-point times depend on the host CPU, so repeats from another host are another cell,
  not more repeats of this one. A rented pod's host CPU cannot be chosen, so in practice only the pod the sets
  ran on (`--reuse-pod`, kept with `--keep-pod` on the first run) can extend them.
- `--skip-repeats 1,2` is a `bench.py run` flag, not an orchestrator one: the orchestrator passes it so a
  replacement pod runs only the repeats whose sets are not yet local.

The `exec` subcommand is a one-off pod runner that reuses the provision/teardown machinery of `run` but
skips the bench sweep: it rents an NVIDIA pod, optionally scp's a `--bin` (patchelf'd to the stock glibc
loader) plus repeatable `--upload local:remote` files, runs an optional `--setup` then the required `--cmd`
(stdout/stderr streamed back), and tears the pod down. It is the standard on-demand path for PTX/CUDA
verifications on real NVIDIA. Useful flags: `--gpu-count N` (multi-GPU, e.g. tensor-parallel), `--keep-warm <minutes>` (the next `exec` with the
same image adopts the warm pod, skipping provision + the slow image pull), `--gpu-types`, `--cloud`,
`--image`, `--min-cuda-version`, `--env KEY=VAL` (repeatable create-time container env; `exec` also defaults
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

## Provisioning

`run` and `exec` cycle through `--gpu-types` in order, each in COMMUNITY and then SECURE (`--cloud all`; RunPod
has no "any cloud" value, and COMMUNITY is the cheaper cloud for the same GPU, so it goes first). Every create
logs where the pod landed and what it bills (`pod X created: COMMUNITY 0.22 per hour`), and `status` and the
dashboard show each live pod's cloud and hourly price. `--max-price-per-hour USD` (optional) caps what a pod may bill per hour, for `run` and `exec` alike. Before a create the
GPU catalog (GraphQL `gpuTypes`, per-GPU price per cloud times `--gpu-count`) refuses an over-cap cloud without
creating anything, and skips a cloud that does not offer the GPU type; both are a capacity miss for that cloud, and
the wait line then reads `no price cap ... for any of ...`. The price of the created pod (`costPerHr`, else `cost`
from one read of the pod) is the backstop when the catalog is unreachable or omits the type: an over-cap pod is
deleted (the delete is retried with 2, 4, 8 s backoff, and a refused delete never aborts the run: it is logged
loudly, retried later and left for `reap`). A pod whose price RunPod never reports is kept. An idle pod or
kept-warm `exec` pod is reused only when its recorded price is within the cap, and `--reuse-pod` of a dearer one is
refused. The `--cloud all` order is fixed (COMMUNITY, then SECURE), not computed per GPU type. When a whole pass finds no capacity it waits 20 s, then 1.5x longer after every pass
up to 5 minutes, with up to 10% jitter, and tries again; the run's status reads
`provisioning (waiting for capacity, pass k)`. A refusal counts as no capacity when it is a 5xx, a transport
error, a pod that does not boot in time, or RunPod's own no-capacity message ("no instances currently
available"). Any other refusal (a bad GPU type or cloud value, auth, an unavailable image) cannot be fixed by
waiting and fails the run at once with RunPod's message, but only when every cloud refused for good: one cloud
refusing while another only lacks capacity keeps waiting.

**CUDA floor.** There is no CUDA filter by default. A pod's CUDA driver is the host's, while the CUDA toolkit is the
image's, so a floor only means "a host driver new enough for the image's CUDA". `--min-cuda-version V`
(`major.minor[.patch]`, compared as numbers, so 13.10 is newer than 13.9) is an optional explicit floor; without it
an image that declares what it needs through the label `org.poot.bench.min-cuda` (the poot bench image says 13.0)
sets the default floor for itself, and an image with no label has none. RunPod's `allowedCudaVersions` enum stops
at 13.0 and would hide every newer host, so no list is sent: the floor is checked on the `cudaVersion` the pod
reports right after the create (again once the pod has a host, when the API had none yet). A pod below the floor is
terminated at once, logged, and counts as a capacity miss (a short pause, then the next attempt). Every create logs
the pod's GPU and reported CUDA version. A pod reporting no version is kept.

A host whose driver cannot run the image's CUDA is a pod fault, not a failed cell: when the runner or the pod-side
setup fails with "CUDA driver version is insufficient for CUDA runtime version", `cudaErrorInsufficientDriver`,
a `cuInit` failure, "no CUDA-capable device is detected" or the container hook's `unsatisfied condition: cuda>=`,
the pod is replaced (counting toward `--max-replacements`) and the failure counts neither as a failed cell nor
toward the three identical failures that skip a cell.

## Run ownership and resume

Every run records its owner (host, boot id, pid namespace, process id, process start time), so liveness is
decidable: a run is `live` while its owner process runs, `dead` once it is gone, `unknown` when it cannot be told
(another host, or another or unreadable pid namespace: a sandboxed peer's pid is invisible in `/proc` here, which
proves nothing).

`run` resolves the run it is asked for before it provisions anything:

- a **live** or **unknown** owner's run is never taken over and its pod is never reaped; a second `run` makes a
  new run of its own,
- a **dead** owner's unfinished run with the same request (ref, scenario, models, GPU types, image, repeats) is
  **resumed** within 24 hours: a recorded pod that is alive is reattached (setup and sweeps already done are
  skipped), one that is gone or still booting is replaced or waited for, and no pod yet means provision,
- a **cancelled**, **abandoned** or **failed** run of the same request within 24 hours lends its result set: the
  new run keeps the name `<set>-<model>-rN` of the old one, so the repeats of both are one set and the repeats
  already gathered are not run again.

One `run` invocation is one run row, however many times it waits or replaces a pod. Per-model
`setup_done` / `sweep_done` / `gathered` flags make every phase idempotent.

## Self-healing

`run` supervises its own attempts (provision if needed, set up, sweep, gather):

- **Lost pod.** A pod RunPod reports gone, or one unreachable over ssh for more than 10 minutes, is deleted, the
  run clears its pod, a replacement is provisioned, and only the repeats not yet local run
  (`--skip-repeats`). `--max-replacements` bounds this.
- **Dead GPU.** Before a sweep starts and at every incremental gather the pod is probed (`nvidia-smi -L`, or
  `rocminfo`). A set whose `env.json` lists no GPU, or whose rows carry a CUDA host-fault marker (a CPU fallback
  is an error, never a result), is quarantined in `rejected/` and never gathered, and the pass fails with a host
  fault: the pod is replaced (the dashboard reads "host fault, replaced") and the repeat count stays honest.
- **Missing model files.** A repeat in which nothing measured and a row says the model files are absent is a setup
  fault, quarantined the same way. A replacement pod sets every model up again; supervision sets the faulting model
  up once more on the same pod, then fails that model with the reason. A run whose models lack valid sets ends
  `incomplete: <model> <valid>/<requested>` (terminal, resumable, exit 1) and not `DONE`.
- **Transient failure.** Any other failure first checks that the pod still answers (waiting with backoff within
  the same 10 minutes); if it does, the attempt is retried on that pod after a pause.
- **A cell that keeps failing.** The same failure of the same cell (model, phase, error kind) three times in a
  row skips that repeat: the orchestrator writes it as a complete set holding one `error` row, with an `env.json`
  note and a `report.md`, and the run continues.
- **Pod-side crash.** `bench.py` turns an exception in one cell into that cell's `error` row; the sweep goes on.
- **Signals.** SIGINT and SIGTERM, and a panic or early return, finalize the run as `cancelled`: the pod is
  deleted (or released with `--keep-pod`) and recorded terminated first. A refused delete still ends the run
  `cancelled`, and `reap` takes the pod later.

## Zombie runs

`run` (after resolving its own run), `status` and `serve` mark an unfinished run `abandoned` ("owner exited
without finalizing") when its owner is dead and it has no pod, its pod is gone, or it is older than the 24-hour
resume window; in the last case its live pod is terminated too. A run whose owner is live, unknown or not
recorded, a dead-owner run with a live pod inside the window (still resumable), and a run whose pod RunPod
cannot be asked about are never touched. The dashboard greys `cancelled` and `abandoned` runs and shows the reason.

## Incremental gather

While a model sweeps, the orchestrator asks the pod about every two minutes which repeats are complete
(`env.json`, `results.jsonl` and `report.md` all non-empty; the harness writes `report.md` last) and copies
each one into a staging directory under `benchmarks/.orchestrator/`. A copy is checked with the harness's own
identity rule (`result_contract.committed_row_errors`); a set that fails it is moved to
`benchmarks/.orchestrator/rejected/` and reported, never left in `results/`. A set that passes is moved into
`benchmarks/results/` under the name every committed set has, `<date>-<gpu>-<model>-<time>-r<N>`: the date and
time are when the set started, the GPU slug comes from the set's `env.json`, as `bench.py` names a run directory.
A pod lost mid-sweep therefore costs only the repeat in flight, a multi-model run has model A's sets local while
model B runs, and `bench index` accepts the sets as gathered. The perf page needs at least 5 repeats per set.

A set with no `ok` row (every cell errored) would break the harness rule that every indexed run measured
something, so it is gathered into `benchmarks/failed-results/<same name>/` instead (git-ignored, never read by
`bench index`, the perf page or `compare-committed`); a set that mixes `ok` and `error` rows stays in `results/`,
where the harness handles error rows. A failed set counts as gathered, so a resume does not run it again.

**Live results.** About every 20 s of a sweep the run process reads the rows the pod has so far (the harness
appends one per finished cell) and mirrors them, with `env.json` once it exists, into
`benchmarks/results/<committed name>.partial/`, replacing the previous copy. A `.partial` directory is not a set:
the harness tools skip it (`bench.PARTIAL_SUFFIX`), git ignores it, and it is removed when the complete set is
gathered. The run process also keeps the harness's latest `[progress]` line (`isl N done (k/4)`, ETA) in
`benchmarks/.orchestrator/progress/<run id>.txt`. `serve` only reads these files and the DB.

## Web dashboard

`serve` starts a read-only web UI (default `http://127.0.0.1:8787`) over the same SQLite state DB. It is a
React 19 + recharts single-page app (source in `ui/`, built by Vite into one inlined offline HTML) that polls
the API every few seconds. It only reads (WAL allows concurrent readers), so it is safe to run in a second
terminal during a `run`. The design, the audit of the previous UI and the screenshots are in `ui/design/`.

Views (`?view=`): **Overview** (what runs, what waits for capacity with pass, minutes waited and price cap, pods
with price/uptime/owner, the spend of the last 24 h and the burn rate, failures with their reasons and pod
replacements), **Runs** (a filterable list named by what each run ran, with a Summary, a live Log with search and
copy, Results as Table / Charts / Failures, and the Pod), **Pods**, **Compare** (decode curves of several runs
overlaid), **Images**, **Serve**. Every view has loading, empty and error states, a light and a dark theme,
keyboard navigation and a stacked layout on narrow screens. The page registry (`ui/src/pages.tsx`) is one record
per view: the scheduler views of POOT-1358 are further records, not a new shell.

Endpoints: `/` (page), `/api/state`, `/api/overview`, `/api/results?run=<id>`, `/api/logfiles`, `/api/log`,
`/api/serve-metrics`, `/healthz`; their fields are documented in the header of `src/web.rs`. `/api/state` only
ever gains fields (`scripts/status-board` reads it).

- A run's sets are found from the DB's recorded set ids, under the name they were gathered as, in `results/` or
  `failed-results/`, or as the `.partial` mirror of a set still being swept, which the Results tab shows while
  the run sweeps (marked partial, with its age and last cell). `/api/results` answers `{sets, reason}`: an exec
  run, or a run that ended before any set, says why it has none instead of an empty page, and an exec run has no
  Results tab.
- The pill of a run that ended its sweep is derived from its gathered rows: all cells ok is green `done`, some
  errors amber `done with errors (n of m cells failed)`, every cell errored red `done: all cells failed`. An
  `incomplete: <model valid/requested>` run is amber and says which models are short of sets.
- Pods are checked against RunPod's own list (read-only `GET /pods`, every 30 s on a background thread, needs
  `RUNPOD_API_KEY` in the environment of `serve`). A pod the DB calls live that RunPod does not list reads
  "missing at RunPod" and stops accruing; one RunPod could not be asked about reads "unverified" with the reason.
  Costs are estimates (pod price x lifetime): RunPod has no billing API.
- A run waiting for capacity shows the last pass its log reported (`/api/state` `wait`).
- The view is kept in the URL (`?view=&run=&tab=&sub=&image=&cmp=`), so a reload, the back/forward buttons and a
  shared link keep it; unknown values fall back to the defaults.
- `npm test` in `ui/` runs the unit and component tests (esbuild-bundled, `node --test`; no browser).
  `npm run build && CHROMIUM=<chromium> npm run screens -- <dir>` renders every view against the fixture API
  (`ui/src/fixtures.ts`, the same data the component tests use) with headless Chromium.

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
