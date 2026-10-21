---
id: authoring-kernels
title: Authoring kernels
sidebar_position: 4
---

# Authoring kernels

poot's kernels are ordinary Rust functions, fully type- and borrow-checked, that the toolchain lowers to
GPU code. There is no DSL and no separate kernel language.

## What `#[kernel]` is

The `#[kernel]` attribute (from `poot-kernel-attr`) marks a function for import by **pootc**, poot's rustc
driver. Which of the attribute's two expansions a build gets is `poot-kernel-attr`'s own `device-pass`
Cargo feature: extern-link a `poot-kernel-attr` built with
`--features device-pass` (the device pass) and the attribute renames each marked function into poot's
reserved namespace; pootc reads the functions from Stable MIR and lowers each kernel to SPIR-V and PTX, and
writes a module cache (`poot_cache.rs`) next to them. Extern-link the crate's default build (the host pass)
and the macro emits a typed host launcher that looks the compiled module up by name. There is no
cargo integration in the repository yet: you invoke `pootc` on the kernel source yourself, `--extern`-linking
whichever `poot-kernel-attr` build the pass needs.

Under `#[cfg(test)]` the same function is ordinary Rust. Call it on CPU arrays before any GPU is involved.

## Minimal example

```rust
use poot_kernel_attr::kernel;

/// Elementwise add: c = a + b, one GPU thread per element.
#[kernel]
pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = thread_index();
    if i < c.len() {
        c[i] = a[i] + b[i];
    }
}
// In a host build the macro emits `add(&ctx, &a, &b, &mut c)`, which dispatches the pootc-compiled SPIR-V.
```

The ABI is slices: inputs as `&[T]`, outputs as `&mut [T]`. Thread index intrinsics (`thread_index`,
`local_index`, `group_index`, with `_x`/`_y`/`_z` variants) lower to the correct registers on each target.

## Run the end-to-end test

```bash
cargo nextest run -p pootc --test macro_kernel
```

The test compiles a `#[kernel]` add function with `pootc` (device pass), reads back the emitted SPIR-V, and
dispatches it on a GPU. It skips the dispatch if no GPU is present.

## Standalone imported kernel

There is no standalone example that runs an imported kernel with nothing else around it. Two test suites cover
the import-and-dispatch path: `pootc`'s `import_run` test imports kernels fresh from source and runs them on a
GPU, and `poot-wgpu-device`'s `imported_kernel` tests load the committed kernel assets through the
`Device` trait and dispatch them on wgpu.

## Accepted Rust subset

The importer accepts a defined subset of Rust and rejects everything else with a named diagnostic. The subset is
listed on [Kernels from Rust](../architecture/kernel-import.mdx#accepted-rust-subset).

## Standalone kernels and generated kernels

Most kernels are never hand-authored: the planner generates a kernel body for each equation shape at compile time.
A kernel authored in Rust and committed as an asset covers what generation does not: the decode GEMVs and
prefill GEMMs over particular weight layouts, the flash-attention bodies, the on-device sampler, the KV-cache
pack and unpack pair, the packed-row gathers, and a handful of importer-capability probes. These sources live
under `crates/pootc/kernels/<family>/*.rs`, one subdirectory per kernel-choice family (contraction, attention,
movement, packed, sampling, and `probe` for kernels a backend crate loads directly).

## The one kernel asset manifest

Every standalone kernel source has exactly one committed asset (`<crate>/assets/<name>.kir.json`) and
exactly one entry in the manifest table (one Rust file per family, aggregated into one list), the single
source of truth for which source produces which asset and who ships it. A test regenerates every asset
from its source and checks it byte-for-byte against the committed one, and separately checks that each
destination `assets/` directory holds exactly its manifest's entries: a committed asset with no manifest
entry, or a manifest entry with no committed asset, fails by name. `poot-codegen` lowers each `.kir.json` asset
to SPIR-V for wgpu and raw Vulkan, PTX for NVIDIA and AMDGCN for ROCm.

After editing a kernel source under `crates/pootc/kernels/<family>/*.rs`, run `just regen-kernel-assets` to
refresh its committed `.kir.json` asset. `pootc`'s own compiler fixtures (used only by pootc's own test
suite, never shipped) stay under `crates/pootc/tests/kernels/*.rs`.

## Design depth

For the full import pipeline (MIR to kernel IR to LLVM to SPIR-V/PTX), see [Kernels from
Rust](../architecture/kernel-import.mdx). For graph IR, fusion, and how kernels fit the engine, see
[Architecture](../architecture/index.mdx).
