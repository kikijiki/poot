---
id: authoring-kernels
title: Authoring kernels
sidebar_position: 4
---

# Authoring kernels

poot's kernels are ordinary Rust functions, type- and borrow-checked, that the toolchain lowers to GPU code. There is no DSL.

## What `#[kernel]` is

The `#[kernel]` attribute (from `poot-kernel-attr`) marks a function for import by **pootc**, poot's rustc driver. Which of the attribute's two expansions a build gets is `poot-kernel-attr`'s own `device-pass` Cargo feature:

- **Device pass:** extern-link a `poot-kernel-attr` built with `--features device-pass`. The attribute renames each marked function into poot's reserved namespace. pootc reads the functions from Stable MIR, lowers each kernel to SPIR-V and PTX, and writes a module cache (`poot_cache.rs`) next to them.
- **Host pass:** extern-link the crate's default build. The macro emits a typed host launcher that looks the compiled module up by name.

There is no cargo integration yet: you invoke `pootc` on the kernel source yourself, `--extern`-linking the `poot-kernel-attr` build the pass needs.

Under `#[cfg(test)]` the same function is ordinary Rust. Call it on CPU arrays before any GPU is involved.

## Minimal example

```rust
use poot_kernel_attr::kernel;

/// Elementwise add: c = a + b, one GPU thread per element.
#[kernel]
pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = thread_index_x();
    if i < c.len() {
        c[i] = a[i] + b[i];
    }
}
```

In a host build the macro emits `add(&ctx, &a, &b, &mut c)`, which dispatches the pootc-compiled SPIR-V.

The ABI is slices: inputs as `&[T]`, outputs as `&mut [T]`. Thread index intrinsics (`thread_index_x`,
`local_index_x`, `group_index_x`, each with `_y` and `_z` variants) lower to the correct registers on each
target. A kernel that uses lane indices, the barrier or workgroup-local arrays declares its launch in the
attribute: `#[kernel(workgroup(64), lds(f32, 64))]` (see [Kernels from Rust](../architecture/kernel-import.mdx)).

## Run the end-to-end test

```bash
cargo nextest run -p pootc --test macro_kernel
```

The test compiles a `#[kernel]` add function with `pootc` (device pass), reads back the emitted SPIR-V and dispatches it on a GPU. It skips the dispatch if no GPU is present.

## Standalone imported kernel

No standalone example runs an imported kernel on its own. Two suites cover import and dispatch: `pootc`'s `import_run` imports kernels from source and runs them on a GPU, and `poot-wgpu-device`'s `imported_kernel` tests dispatch the committed kernel assets through the `Device` trait on wgpu.

## Accepted Rust subset

The importer accepts a defined subset of Rust and rejects the rest with a named diagnostic; see [Kernels from Rust](../architecture/kernel-import.mdx#accepted-rust-subset).

## Standalone kernels and generated kernels

Most kernels are never hand-authored: the planner generates a kernel body for each equation shape at compile time. Matmuls, elementwise ops, attention and data movement are always generated.

A kernel authored in Rust and committed as an asset covers what generation does not: the on-device sampler, the packed-row gathers and the state-copy kernel. Importer-capability probes are test fixtures.

These sources live under `crates/pootc/kernels/<family>/*.rs`, one subdirectory per kernel-choice family (packed, sampling, state, and `probe` for the test fixtures a backend crate's tests load directly).

## The one kernel asset manifest

The manifest table (one Rust file per family) says which source produces which asset and who ships it. Every standalone kernel source has exactly one committed asset (`<crate>/assets/<name>.kir.json`) and one manifest entry. Two tests guard it: one regenerates every asset from its source and compares byte-for-byte, and one checks each `assets/` directory holds exactly its manifest's entries (a mismatch fails by name).

`poot-codegen` lowers each `.kir.json` asset to SPIR-V for wgpu and raw Vulkan, PTX for NVIDIA and AMDGCN for ROCm.

After editing a kernel source, run `just regen-kernel-assets` to refresh its committed asset. `pootc`'s own compiler fixtures stay under `crates/pootc/tests/kernels/*.rs` and are never shipped.

## Design depth

See [Kernels from Rust](../architecture/kernel-import.mdx) for the import pipeline and [Architecture](../architecture/index.mdx) for graph IR and fusion.
