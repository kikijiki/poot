# Bench image releases

Log of every build of the benchmark image `ghcr.io/kikijiki/poot-bench:latest` (see
[README.md](README.md)). The tag is always `latest`, so the digest is the version: record it here on each
rebuild and push, with the source commit and what changed. The image bakes the baseline engines and the
harness only; the poot binary is scp'd fresh per run, so a poot code change needs no image rebuild. Only a
change to the Dockerfile, the harness, a baseline engine pin, or a baseline runner does.

Digest of the live image:

```
skopeo inspect docker://ghcr.io/kikijiki/poot-bench:latest | jq -r .Digest
```

Build and push (rootless podman, inside `benchmarks/` `nix develop`; needs `podman login ghcr.io`
with a token that has `write:packages`):

```
just sweep-image --tag latest --push   # via the orchestrator: streams to image-latest.log, shows in the dashboard
just image-build-push                  # or raw: podman build -f docker/Dockerfile . ; podman push
```

There is no CI workflow that builds the image; use the paths above.

## Releases (newest first)

| Date       | Source commit  | Digest (sha256)         | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | -------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-09-06 | `e33b899b`     | `ac10161ce144`          | Add `libxext6` + `libx11-6` so the NVIDIA Vulkan ICD (`libGLX_nvidia.so.0`) resolves `libXext.so.6` / `libX11.so.6` (update 1090 miss on `6ba9af504485`). No-GPU smoke passed: `libvulkan.so.1`, `vulkaninfo --help` usage text, `tf 5.0.0`, baked vLLM/llama.cpp/candle binaries; `ldconfig` shows `libXext.so.6` and `libX11.so.6`. OCI digest `sha256:ac10161ce1445e276412d2390faed4dc96b634400b4e6f9ebc8c478cefab730d`; config digest `sha256:ed3321476b8f65a3a2a2479d7d1a4de49614456966cbd79b2ab984e6d22379be`. Card 276 SC-003 still needs a real-wgpu RunPod rerun against this digest. See update 1092.                                                                                                                                   |
| 2026-09-06 | `f7f72c89`     | `6ba9af504485`          | Rebuild of `latest` with the vLLM FlashInfer sampler fix (update 0629), `--gpu-mem-util` 0.85 (update 0653), and the Card 276 Vulkan loader/tools (`libvulkan1`/`vulkan-tools`, updates 0961/0963). No-GPU smoke passed: `libvulkan.so.1`, `vulkaninfo --help` usage text, `tf 5.0.0`, baked vLLM/llama.cpp/candle binaries. OCI digest `sha256:6ba9af50448563cc9ae2db8a7e850eeb2b13e98cfd2f53266dc29376724b45bf`; config digest `sha256:c8b164dd8e5f8532facda601a3b2c0608a68bca7da4dcc391842bab5380da3b4`. Card 276 SC-003 still needs a real-wgpu RunPod rerun. See update 1086.                                                                                                                                                                |
| 2026-06-15 | `a1ca683`      | `59e7c8e161b3`          | Multi-stage slim image, now the single `docker/Dockerfile`. 9.98 GB compressed / 18.2 GB on disk, down from 15.69 / 34 GB: a throwaway `nvidia/cuda:...-cudnn-devel` builder compiles llama.cpp + candle and only artifacts ship into a `...-cudnn-runtime` base (drops the runpod/pytorch-devel base and its unused pytorch 2.4). Reimplemented `PUBLIC_KEY`->sshd `/start.sh`; baked `hf`/`uv`/`tmux`; tf-venv pins `--torch-backend=cu124` (CPU torch caused transformers "no CUDA device"). Validated on a SECURE RTX A5000 (poot 31 / candle 156 tok/s, vLLM ran, SWEEP_EXIT=0, clean teardown). Built and pushed via the orchestrator `image` subcommand. Digest `sha256:59e7c8e161b3cb43953421b51c5d8e4f94c86e894f7a43e793f3897a24c38644`. |
| 2026-06-14 | `e9dcd6c`      | `e84b5dffcae7`          | Rebuild from master after landing the perf branch; same bench source as the 2026-06-13 image. Built locally with podman (CI was failing). Digest `sha256:e84b5dffcae75fb57375ca3212d811c4d5582e3eb66ecaaba609b0b848cff764`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-06-13 | (decode-curve) | `5ad87b75...` (partial) | First image with vLLM `--torch-backend=cu130` + CUDA triton (`--torch-backend=auto` had picked the Intel-XPU torch on the GPU-less CI builder) and the llama.cpp depth-flag fix (`-d ISL -n OSL`, was the invalid `-gp`). Harness: per-cell GPU-util% + process-tree CPU% + ~1 Hz `util_series`, live `[progress]` streamed from stdout + stderr. Built locally with podman (GH runner builds kept failing). 34.3 GB.                                                                                                                                                                                                                                                                                                                             |

> Older images (the first spec-112 builds) predate this log and are not tracked.
