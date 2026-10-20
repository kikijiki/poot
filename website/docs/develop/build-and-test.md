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
just gate      # the local merge gate: fmt-check, clippy with -D warnings, then just test
just test      # dead-pub/dependency gates, self-tests, model-free test inventory and doctests; includes crates-standalone (every library builds alone)
just check     # cargo check
just clippy    # lints
just fmt       # format every crate
```

`just gate` is the merge gate, and it runs on your own machine: there is no CI. It stops at the first of its three
steps that fails. `just npu-build` (part of `just test`) needs the `nix develop .#npu` shell and fails without it; on
a host that has none (a laptop), set `POOT_ALLOW_NO_NPU_SHELL=1` to say so.

`just test` never needs a GPU or a checkpoint. It runs with every device hidden and required and with no models
directory, so a test that reaches for a device or a checkpoint fails instead of skipping. Tests that need a device or
a checkpoint run in separate lanes, and each device lane fails before any test runs when its device is missing:

```bash
just test-device-wgpu      # the required migrated wgpu cases (executor contract, packed serving, launch widening, I32, cr30 failure lifecycle), wgpu device required
just test-device-kernels   # pootc and poot-kernelgen kernel suites, wgpu device required
just test-device-vulkan    # raw Vulkan: live caps receipt, executor contract, parity table and driver generation rows, under the validation layer
just test-device-rocm      # poot-rocm-device resident probes and packed-serving cases
just test-device-ptx       # one PTX probe, on an NVIDIA machine
just test-real-dims        # tests named *_real_dims, in release
just test-checkpoint       # CPU rows over the small checkpoints in ~/models; a missing model fails
just test-checkpoint-wgpu  # every real-checkpoint device row on wgpu (one per model family at least), device and models required
just test-checkpoint-rocm  # the same rows on ROCm (also -ptx on an NVIDIA pod, and -vulkan)
just test-checkpoint-large # rows that read a 20B+ checkpoint: a pod or host that holds them, never the dev box
just test-real-dims-long   # the real-dimension rows that take minutes, in release
just bench                 # the criterion benches (CPU-oracle decode, wgpu op dispatch)
```

A test that loads a checkpoint finds it only through `POOT_MODELS_DIR`, the directory that holds the models; test code
has no default. Outside the checkpoint lanes an unset variable, or a model missing from the directory, skips the test
and prints the model's name. With `POOT_REQUIRE_MODELS=1` either one fails the test, naming the variable or the model.
The checkpoint lanes (`just test-checkpoint`, `just test-checkpoint-wgpu`, `just test-checkpoint-rocm` and the other checkpoint lanes) set `POOT_MODELS_DIR` to `~/models` unless you export another directory, and set
`POOT_REQUIRE_MODELS=1`, so a lane over a models directory that lacks a model fails instead of passing on skips.

Each checkpoint lane runs its rows by exact name (an `#[ignore]`d row too): `scripts/required-checkpoint-cases.txt` for
the CPU lanes, the `checkpoint-wgpu.txt`, `checkpoint-rocm.txt`, `checkpoint-ptx.txt` and `checkpoint-vulkan.txt` lists in `scripts/required-device-cases/` for the device lanes, which also fail a row that
skipped. The lists are hand-maintained. A `checkpoint` test or an `#[ignore]`d test that is in no lane's list, and a row
that names no test, all fail `just test-lane-coverage`; an ignored test no lane runs is deleted.

`scripts/test-model-free.sh` decides which packages the default lane covers. Its `gate_policy` names every
workspace package and either gates it or excludes it with a reason, and the lane fails when a package is missing
from the list. The lane runs the tests listed as `model-free` in `scripts/test-model-free-inventory.tsv`, which
the script generates from the test binaries: run `just test-model-free-inventory-regen` after adding, renaming, or
deleting a test. In most packages a test's class is derived (`model-free`, or `real-dims` for a test whose name ends
in `_real_dims`). In packages that mix host tests with device or checkpoint tests (`poot-llm`, `poot-load`,
`poot-models`, `poot-serve`, `pootc`, `poot-kernelgen` and the runtime crates) you choose `model-free`,
`checkpoint` (loads a model; a row of `scripts/required-checkpoint-cases.txt`, run by a `just test-checkpoint*` lane) or `device` (dispatches on a device) for each new test
by reading it. `scripts/test-model-free.sh self-test` (also `just test-model-free-inventory`) checks the checker
and the lane against known-bad inventories.

`just test-lane-coverage` (part of `just test`) fails when a test runs in no lane: a `device` test, or any test of a
package the model-free lane excludes, must match one of the `test-device-*` selections in the `justfile`. Select a
device crate by `package(...)` there, so a test added later is picked up, and list the new test in
`scripts/required-device-cases/<lane>.txt`. It also fails when a test a lane selects is not held by the `gpu` test group
in `.config/nextest.toml`, because two device tests at once can hang the GPU.

Every device lane ends with the required-case gate. The lane recipe stamps the start of the run
(`scripts/check-required-cases.py begin <lane> <results.xml>`, which also deletes the old results), runs nextest under
the lane's own profile, then checks (`check <lane> <results.xml>`): the results must be newer than the stamp and written by that lane's profile, every row of
`scripts/required-device-cases/<lane>.txt` must have run and passed, and every other passing row must be named with a
reason in `<lane>.unlisted.txt`. A row that printed a skip report fails the lane whether it is listed or not.

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

Production backends are selected at runtime with `--backend wgpu|rocm|ptx|vulkan`. Every backend is built into
every binary; a backend whose device is absent is refused when it opens, and `auto` probes devices, not features.

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
