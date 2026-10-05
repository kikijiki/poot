# Pre-baked benchmark image

A public Docker image that bakes the five baseline engines and the harness so a RunPod pod boots ready
(no ~90-minute on-pod install). It is pushed to GHCR as a **public** package (free storage and egress) and
pulled by the standard RunPod MCP `create-pod` via `imageName`, with no registry credential or network volume.

## What is / isn't in the image

Multi-stage build (18.2 GB on disk, down from 30 GB): a throwaway `nvidia/cuda:...-cudnn-devel` builder
compiles llama.cpp and the candle runner, and only the artifacts ship into a slim
`nvidia/cuda:...-cudnn-runtime` base. A reimplemented `/start.sh` provides the `PUBLIC_KEY` -> sshd contract
the orchestrator needs (the runpod base used to provide it).

- **In:** all five engines pinned to sm_86 (vLLM 0.22.1, transformers 5.0.0, candle 0.10.2 + CUDA, llama.cpp
  b9601 + CUDA), the harness venv, `hf` + `uv` + `tmux` + sshd, the Vulkan loader/diagnostic tools needed by
  NVIDIA wgpu (`libvulkan1`, `vulkan-tools`), and the benchmark suite source. The torch venvs pin an explicit
  `--torch-backend` (cu130 for vLLM, cu124 for transformers) so they are CUDA builds, not the CPU wheel uv
  picks on the GPU-less build host.
- **Out:** the model weights (fetched on the pod, or read from an attached network volume) and the poot
  binary (changes every version; scp'd fresh per run, then `patchelf`'d). `.dockerignore` also excludes
  `.orchestrator/` from the build context.

Only candle + llama.cpp link the SYSTEM cublas/cudart (the runtime base carries them); the torch venvs
bundle their own CUDA libs. The pod's injected driver provides libcuda at runtime.

## Build it

From the bench devshell (`nix develop ./benchmarks`, which provides podman / skopeo / gh) use the
`justfile`:

- `just image-build-push` - build the Dockerfile locally with rootless podman, run
  `image-smoke`, and push to GHCR (`podman login ghcr.io` first, with a token that has `write:packages`).
  `just image-build` builds only (~20 GB + a long CUDA compile, for validating the Dockerfile);
  `just image-push` pushes an already-built local image.
- `just sweep-image --tag latest --push` - the same build+push driven through the orchestrator, which streams
  progress to `image-latest.log` and the dashboard.
- `just image-smoke` - check a locally built image's Vulkan loader/diagnostic availability, env imports, and
  baked binaries. It does not need a GPU. The Vulkan diagnostic check requires `vulkaninfo --help` to print
  usage text; Ubuntu's tool can do that while returning exit code 1.

There is no CI workflow that builds the image, so use `image-build-push` or the orchestrator path. The build uses `docker/Dockerfile`
(context `benchmarks/`) and produces `ghcr.io/<owner>/poot-bench:latest`. The `setup/*.sh` scripts are the
source of truth for what goes in.

Record every pushed build (digest, source commit, changes) in [RELEASES.md](RELEASES.md). The tag is always
`latest`, so the digest is the version.

## One-time: make the package public

The first push creates the GHCR package **private**. Flip it once: GitHub -> your packages ->
`poot-bench` -> Package settings -> Danger Zone -> Change visibility -> Public. After that pods pull it
with no credential and it costs nothing. The source repo stays private; only the image is public, so only
the benchmark tooling and public engine installs are exposed (no poot codegen core, no poot binary).

## Use it in a sweep

`create-pod` with `imageName: ghcr.io/<owner>/poot-bench:latest`, `gpuTypeIds: ["NVIDIA GeForce RTX 3090"]`,
`ports: ["22/tcp"]`, `env: { PUBLIC_KEY: <pubkey> }`. SSH in, fetch the eval models, scp + `patchelf` the
fresh poot binary, set the runner env vars (`POOT_BENCH_POOT_BIN` etc.) and run `uv run bench run ...`.
Detach long runs (a poot cell can take hours) with `tmux`, not `nohup`. See "Operational gotchas" in
../README.md ("Running the sweep on RunPod") for that and the rest (endpoint via `list-pods`, GGUF
conversion, poot `warmup>=2`, ISL capping).
