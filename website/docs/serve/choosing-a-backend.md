---
id: choosing-a-backend
title: Picking a backend
sidebar_position: 3
---

# Picking a backend

## Supported backends

| Backend | Runs on | What it is | Select with | Notes |
| --- | --- | --- | --- | --- |
| wgpu | Any Vulkan-capable GPU, integrated or discrete | Kernels compiled to SPIR-V and dispatched through wgpu; needs no vendor SDK | `--backend wgpu` | What `auto` falls back to. Verified on AMD (RADV). |
| ROCm | AMD GPUs | Kernels compiled to AMD GCN code objects and dispatched through raw HSA; the HSA library is loaded at run time | `--backend rocm` | Verified on gfx1151 and gfx1100. |
| PTX | NVIDIA GPUs | Kernels compiled to PTX and run through the CUDA driver | `--backend ptx` | The only backend that serves a model over several GPUs. Verified on NVIDIA data-center GPUs. |
| Vulkan (raw) | Any Vulkan-capable GPU | The same SPIR-V kernels as wgpu, dispatched through raw Vulkan with no wgpu | `--backend vulkan` | `auto` never picks it, since it is the same hardware as wgpu. Verified on AMD (RADV) only. Serving needs `VK_EXT_memory_budget`. |
| NPUs (AMD XDNA2, Intel NPU4) | The NPU of a Ryzen AI or Core Ultra processor | In development: library devices (`poot-npu`) that run a few ops; no model runs on one | Not a `--backend` value | See [NPU offload](../architecture/npu.mdx). |

`--backend auto` is the default. Every backend is in every build. The design of each is on [Backends](../architecture/backends.mdx).

## How the server chooses

The choice is made once, at startup, before the server listens:

- `auto` opens the first device present, in a fixed order: **ptx** if a CUDA device opens, else **rocm** if an HSA GPU agent opens, else **wgpu**. It never picks **vulkan**.
- A named backend that cannot open, or a model it refuses (an operation the device lacks a capability for), stops startup with the refusal. The server never tries another backend after a refusal.
- `--placement pipeline:ORD,ORD,...` (or `tensor:...`) runs one model over several GPUs. Only **ptx** can; a group on another backend is refused at startup, as is `--device-timing` over a group.
- `--backend vulkan` reads the free device memory from `VK_EXT_memory_budget`. A driver without it stops startup with a typed capacity error.

Before it serves, the server fits the workload to the device memory: it computes the exact footprint of each candidate (weights once, every retained arena, state), takes the largest that fits, and checks what the device allocated before `/health` reports `serving`. When none fits, startup stops with a capacity error that counts the refused candidates. A prepared engine over the budget observed later is refused with an over-budget error naming the largest allocations and the shortfall.

See also [Supported models and formats](../develop/supported-models.md) (which models and weight formats each backend plans) and [Serving capabilities](./capabilities.md) (what each backend's lane runs).

## Which backend for which hardware

- **NVIDIA GPU:** ptx, the most mature single-GPU library path and the only backend that serves a model over several GPUs.
- **AMD GPU:** rocm, the primary backend for single-sequence library decode. wgpu runs on the same GPU as a portable fallback.
- **Anything else (Intel iGPU or dGPU, other Vulkan hardware):** wgpu. Raw Vulkan is verified on AMD (RADV) only.

## Checking that the toolchain sees your GPU

Whatever backend you use, confirm the toolchain sees your GPU before troubleshooting:

```bash
# both codegen targets present
llc --version | grep -iE 'spirv|nvptx'
# your GPU shows up to Vulkan (needed even for rocm/ptx builds)
vulkaninfo | grep -i 'deviceName'
```
