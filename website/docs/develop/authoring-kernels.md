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
Cargo feature, not an environment variable: extern-link a `poot-kernel-attr` built with
`--features device-pass` (the device pass) and the attribute renames each marked function into poot's
reserved namespace; pootc reads the functions from Stable MIR and lowers each kernel to SPIR-V and PTX, and
writes a module cache (`poot_cache.rs`) next to them. Extern-link the crate's default build (the host pass)
and the macro emits a typed host launcher that looks the compiled module up by name instead. There is no
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

Running an imported kernel with nothing else around it is not available as a standalone example (the example that demonstrated it was removed). The same import-and-dispatch path (an imported
kernel body loaded and run on the GPU, matching the engine's hand-built kernel byte-for-byte) is exercised
by `poot-gpu`'s own test suite instead.

## Accepted Rust subset

The importer accepts:

- Index and range loops (`for k in 0..n`, `0..=n`, `for x in slice`)
- Float math (`x.sqrt()`, `x.exp()`, `x.max(y)`)
- Runtime dimensions (one kernel serves any shape)
- Workgroup-parallel kernels with a barrier and workgroup-local (LDS) arrays, including a barrier inside a
  loop and a 2-D LDS tile (what a blocked GEMM needs)

Inference kernels (RMSNorm, a numerically-stable softmax, an LDS reduction, an argmax, and the GEMV
+ tiled-GEMM matmuls) are authored in plain Rust and run on the GPU byte-identically to the engine's
hand-built kernels. A sampling step built from these same primitives (a batched argmax, or a Gumbel-
noise perturbation plus top-k/top-p bisection for temperature sampling) is appended to the decode
graph itself and runs on-device, on every backend, for greedy and plain-temperature requests; the
host reads back only the chosen token id and a non-finite flag. A request with a logit-bias/penalty,
logprob recording, or a guided-decoding constraint still reads the full logits back and picks on the
host. On the wgpu path the entire f32 matmul family (every projection GEMV and tiled GEMM, decode and
prefill) are imported-from-Rust kernels dispatched every step.

## The partition: generated lowering families vs. standalone kernels

Most of poot's kernels are never hand-authored: a fusion/lowering pass generates a `Body` for an
equation shape at compile time (the elementwise, reduce, matmul and attention families kernelgen
produces on the fly). Authoring a kernel in Rust and shipping it as a committed asset is for the kernels
that family generation doesn't cover: the coalesced decode GEMVs over a `[K, N]` weight (f32 and BF16,
with and without bias), the coarsened prefill GEMMs, the M > 1 BF16 contraction over a checkpoint-order
weight and the flash-attention bodies, whose shape-generic form is simpler to hand-write than to
synthesize per shape (the decode GEMV over a checkpoint-order `[N, K]` weight is generated), the
on-device sampler bodies, the I8 KV-cache pack/unpack pair, the packed-row gathers, and a handful of
importer-capability probes. These standalone kernels live under `crates/pootc/kernels/<family>/*.rs`,
one family subdirectory per kind (matching
poot's own kernel-choice families: contraction, attention, movement, packed, sampling, plus a `probe`
family for kernels a backend crate loads directly rather than through the planner).

## The one kernel asset manifest

Every standalone kernel source has exactly one committed asset (`<crate>/assets/<name>.kir.json`) and
exactly one entry in the manifest table (one Rust file per family, aggregated into one list), the single
source of truth for which source produces which asset and who ships it. A test regenerates every asset
from its source and checks it byte-for-byte against the committed one, and separately checks that each
destination `assets/` directory holds exactly its manifest's entries - a committed asset with no manifest
entry, or a manifest entry with no committed asset, fails by name rather than drifting unnoticed.

After editing a kernel source under `crates/pootc/kernels/<family>/*.rs`, run `just regen-kernel-assets` to
refresh its committed `.kir.json` asset. `pootc`'s own compiler fixtures (used only by pootc's own test
suite, never shipped) stay under `crates/pootc/tests/kernels/*.rs`.

## Design depth

For the full import pipeline (MIR to kernel IR to LLVM to SPIR-V/PTX), see [Kernels from
Rust](../architecture/kernel-import.mdx). For graph IR, fusion, and how kernels fit the engine, see
[Architecture](../architecture/index.mdx).
