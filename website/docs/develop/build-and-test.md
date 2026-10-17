---
id: build-and-test
title: Build and test
sidebar_position: 2
---

# Build and test

poot is a Rust workspace. The repo pins nightly Rust, LLVM (for kernel lowering), and Vulkan through Nix.

## Prerequisites

```bash
nix develop
```

That drops you into a shell with the right `rustc`, `llc`, and Vulkan loader. Sanity-check it:

```bash
llc --version | grep -iE 'spirv|nvptx'   # both targets present
vulkaninfo | grep -i 'deviceName'         # a GPU is visible
```

## Common tasks

The common tasks are wrapped as `just` recipes (run `just` with no args to list them):

```bash
just build     # cargo build --workspace
just test      # dead-pub/dependency gates, self-tests, model-free test inventory and doctests
just check     # cargo check
just clippy    # lints
just fmt       # format every crate
```

`just test` never needs a GPU or a checkpoint. It runs with every device hidden and required and with no models
directory, so a test that reaches for a device or a checkpoint fails instead of skipping. Tests that need a device or
a checkpoint run in separate lanes, and each device lane fails before any test runs when its device is missing:

```bash
just test-device-wgpu      # the required migrated wgpu cases (executor contract, packed serving, launch widening, I32, cr30 failure lifecycle), wgpu device required
just test-device-kernels   # pootc and poot-kernelgen kernel suites, wgpu device required
just test-device-vulkan    # raw Vulkan add dispatch
just test-device-rocm      # poot-rocm-device resident probes and packed-serving cases
just test-rocm-feature     # the tests that exist only under --features rocm, in release
just test-device-ptx       # one PTX probe, on an NVIDIA machine
just test-real-dims        # tests named *_real_dims, in release
just test-optional         # the whole workspace; device tests skip without a device
just bench                 # both criterion benches (CPU decode, wgpu op dispatch)
```

A test that loads a checkpoint finds it only through `POOT_MODELS_DIR`, the directory that holds the models; test code
has no default. With the variable unset the test skips and prints the model's name, and with `POOT_REQUIRE_MODELS=1` an
unset variable fails the test naming it. A model missing from the directory always skips. The checkpoint lanes
(`just test-rocm-feature` and `just test-optional`) set `POOT_MODELS_DIR` to `~/models` unless you export another
directory, and `just test-rocm-feature` also sets `POOT_REQUIRE_MODELS=1`.

`scripts/test-model-free.sh` decides which packages the default lane covers. Its `gate_policy` names every
workspace package and either gates it or excludes it with a reason, and the lane fails when a package is missing
from the list. The lane runs the tests listed as `model-free` in `scripts/test-model-free-inventory.tsv`, which
the script generates from the test binaries: run `just test-model-free-inventory-regen` after adding, renaming, or
deleting a test. In most packages a test's class is derived (`model-free`, or `real-dims` for a test whose name ends
in `_real_dims`). In packages that mix host tests with device or checkpoint tests (`poot-llm`, `poot-load`,
`poot-models`, `poot-serve`, `pootc`, `poot-kernelgen` and the runtime crates) you choose `model-free`,
`checkpoint` (loads a model; `just test-optional` runs it) or `device` (dispatches on a device) for each new test
by reading it. `scripts/test-model-free.sh self-test` (also `just test-model-free-inventory`) checks the checker
and the lane against known-bad inventories.

`just test-lane-coverage` (part of `just test`) fails when a test runs in no lane: a `device` test, or any test of a
package the model-free lane excludes, must match one of the `test-device-*` selections in the `justfile`. Select a
device crate by `package(...)` there, so a test added later is picked up, and list the new test in
`scripts/required-device-cases/<lane>.txt`.

Focused runs are fine during development:

```bash
cargo nextest run -p <crate> <filter>
cargo test -p <crate> <filter>
```

## Release builds for real models

Inference tests and generation are much faster in release mode. Always pass `--release` for anything that
runs a real model or touches GPU dispatch on the hot path.

## Kernel disk cache

Compiled kernels are cached on disk and reused across processes. The cache lives under
`$POOT_KERNEL_CACHE_DIR` when set, else `$XDG_CACHE_HOME/poot/kernels`, else `~/.cache/poot/kernels`. Set
`POOT_KERNEL_CACHE=0` (also `off` or `false`) to disable the shared cache and use a per-process directory. When a process first opens a kernel root it prunes that root: it keeps the
current epoch and the 8 newest others, removes an old epoch only if it has been idle for more than 24 hours
(each cache hit refreshes the `EPOCH` sidecar's mtime, which is its last use) and removes compiler scratch
files older than one hour.

## Running a binary outside the dev shell

A binary built inside the dev shell links that shell's glibc. Running it outside the shell (for example on
a pod) needs a host glibc of at least 2.39, or the toolchain libraries bundled alongside it.

## Backends and features

Production backends are selected at runtime with `--backend wgpu|rocm|ptx|vulkan`. ROCm needs
`cargo build --features rocm` (or `--features rocm` on the specific binary you run). PTX and wgpu build
with default features in the dev shell.

See [Backends](../architecture/backends.mdx) for codegen targets, dispatch models, and capability
notes. For serve-time backend choice, see [Picking a backend](../serve/choosing-a-backend.md).

## Regenerating the diagrams

The architecture diagrams in the README (and the PNGs the website serves from `/img/architecture/`) are
generated from hand-written SVG sources in `website/diagrams/`, one per image. To change a label, edit the
SVG and run, inside the dev shell:

```bash
just diagrams                       # render every source to website/static/img/architecture/ plus index.json
just diagrams --only model-tracing,rust-kernels
just diagrams --out DIR          # render somewhere else
just diagrams --publish-dir DIR  # also copy the PNGs and index.json there
just diagrams --check            # fail if a committed PNG differs from its source
```

Rendering uses the dev shell's pinned `resvg` and one bundled font, so the output is byte-reproducible. `index.json`
holds each image's name, size, checksums, source and alt text; the README alt texts are copied from it (each SVG's
`<desc>`), and `just test` runs `--check`. The script writes only to `--out` and `--publish-dir`.
