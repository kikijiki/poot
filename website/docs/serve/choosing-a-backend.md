---
id: choosing-a-backend
title: Picking a backend
sidebar_position: 3
---

# Picking a backend

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today `poot-serve` has no `--backend` option and refuses it as an unknown option.

The backends are libraries. **wgpu** (Vulkan/SPIR-V, the default), **rocm** (AMD raw HSA, `--features rocm`)
and **ptx** (NVIDIA CUDA) share the graph IR and model definitions, and so does **vulkan**: the same SPIR-V
kernels as wgpu, dispatched through raw Vulkan (`ash`) with no wgpu, recorded once as native command buffers
and replayed. Only codegen and dispatch differ. Pick one through the `poot-llm` API (`--backend
wgpu|rocm|ptx|vulkan|auto`); `auto` never picks vulkan, since it runs on the same hardware as wgpu. Raw Vulkan
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
