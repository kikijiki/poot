---
id: build
title: Build
sidebar_position: 2
---

# Build

poot is a Rust workspace. The repo pins nightly Rust, LLVM (for kernel lowering) and Vulkan through Nix.

## Prerequisites

Enter the dev shell. It provides the right `rustc`, `llc` and Vulkan loader; outside it you hit the wrong `rustc` or a missing `llc` target.

```bash
nix develop
```

Check that both kernel targets and a GPU are visible.

```bash
llc --version | grep -iE 'spirv|nvptx'   # both targets present
vulkaninfo | grep -i 'deviceName'         # a GPU is visible
```

## Build poot

Build the whole workspace.

```bash
just build
```

Run `just` with no arguments to list the other recipes. These are the everyday ones:

| Recipe | What it does |
| --- | --- |
| `just check` | `cargo check` over the workspace |
| `just clippy` | Lints |
| `just fmt` | Formats every crate |

## Build a binary that runs a model

Pass `--release` for anything that runs a real model or dispatches on the GPU hot path. Debug builds are much slower.

```bash
cargo build --release -p poot-llm --features cli --bin poot
```

## Backends and features

Backends are selected at runtime with `--backend`. See [the supported backends](../serve/choosing-a-backend.md#supported-backends).

- Every backend is built into every binary.
- A backend whose device is absent is refused when it opens.
- `auto` probes devices, not features.

[Backends](../architecture/backends.mdx) covers codegen targets, dispatch models and capability notes.

## Kernel disk cache

Compiled kernels are cached on disk and reused across processes.

| Setting | Effect |
| --- | --- |
| `POOT_KERNEL_CACHE_DIR` | Cache directory, else `$XDG_CACHE_HOME/poot/kernels`, else `~/.cache/poot/kernels` |
| `POOT_KERNEL_CACHE=0` (or `off`, `false`) | Disable the shared cache and use a per-process directory |

The cache prunes itself: old entries go once they have been idle for more than a day.

## Run a binary outside the dev shell

A binary built in the dev shell links that shell's glibc. Outside the shell, for example on a pod, use one of these:

- a host glibc of at least 2.39
- the toolchain libraries bundled next to the binary

To check your work, see [Test](./test.md).
