---
id: choosing-a-backend
title: Picking a backend
sidebar_position: 3
---

# Picking a backend

`poot-serve --backend wgpu|rocm|ptx|vulkan|auto` (default `auto`) picks the device. The choice is made once, at
startup, before the server listens:

- `auto` opens the first device that is present, in one fixed order: **ptx** if a CUDA device opens, else
  **rocm** if an HSA GPU agent opens, else **wgpu**. It decides by device presence only. It never picks
  **vulkan**.
- A named backend that cannot open, or a model the chosen backend refuses (an operation the device lacks a
  capability for), stops startup with the refusal. `poot-serve` never tries another backend after a refusal:
  a model that does not run on the device you chose is an error you can read, not a silent move to different
  hardware.
- `--placement pipeline:ORD,ORD,...` (or `tensor:...`) runs one model over several GPUs, and only the **ptx**
  backend can: the other backends cannot copy between their devices, so a group on them is refused at startup.
  `--device-timing` is refused over a group.
- Before it serves, the server fits the workload to the device memory: it computes the exact footprint of each
  candidate (weights once, every retained arena, state) and takes the largest candidate that fits, then checks
  what the device actually allocated before `/health` reports `serving`.

`--backend vulkan` serves too: raw Vulkan reads the free device memory from `VK_EXT_memory_budget`, so the engine plan fits the workload to it like any other backend. A Vulkan driver without that extension reports no budget, and startup then stops with a typed capacity error (an unknown budget is never guessed).

The backends are libraries. **wgpu** (Vulkan/SPIR-V, the default), **rocm** (AMD raw HSA, `--features rocm`)
and **ptx** (NVIDIA CUDA) share the graph IR and model definitions, and so does **vulkan**: the same SPIR-V
kernels as wgpu, dispatched through raw Vulkan (`ash`) with no wgpu, recorded once as native command buffers
and replayed. Only codegen and dispatch differ. Pick one in `poot-serve` with `--backend
wgpu|rocm|ptx|vulkan|auto`, or in library code through the `poot-llm` API; `auto` never picks vulkan, since it runs on the same hardware as wgpu. Raw Vulkan
is the lowest-priority backend: it is verified on AMD (RADV) only. See the
[feature matrix](../reference/feature-matrix.md) for what each supports and
[Architecture: Backends](../architecture/backends.mdx) for the design.

NPU offload was removed. See [Architecture: NPU
Offload](../architecture/npu.mdx).

## Verifying the sanity check for your box

Whatever backend you use, confirm the toolchain sees your GPU before troubleshooting:

```bash
llc --version | grep -iE 'spirv|nvptx'   # both codegen targets present
vulkaninfo | grep -i 'deviceName'         # your GPU shows up to Vulkan (needed even for rocm/ptx builds)
```
