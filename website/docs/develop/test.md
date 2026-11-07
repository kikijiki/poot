---
id: test
title: Test
sidebar_position: 9
---

# Test

The default test run needs no GPU and no checkpoint. Device and checkpoint tests run in separate lanes.

## Run the test suite

Run the full local gate. It formats, lints with warnings denied, then runs the tests.

```bash
just gate
```

Run only the tests. This hides every device and runs with no models directory, so a test that reaches for either fails instead of skipping.

```bash
just test
```

Run one crate's tests while developing.

```bash
cargo nextest run -p <crate> <filter>
```

## Device lanes

Each lane needs its device. A lane fails before any test runs when the device is missing.

| Recipe | Runs on |
| --- | --- |
| `just test-device-wgpu` | wgpu (any Vulkan GPU) |
| `just test-device-vulkan` | Raw Vulkan, under the validation layer |
| `just test-device-rocm` | ROCm (AMD GPU) |
| `just test-device-ptx` | PTX (NVIDIA GPU) |
| `just test-device-kernels` | The kernel suites, on wgpu |

Run device lanes one at a time. Two device tests at once can hang the GPU.

## Checkpoint lanes

A checkpoint test finds its model through `POOT_MODELS_DIR`, the directory that holds the models.

| Situation | Result |
| --- | --- |
| Variable unset or model missing, default lane | The test skips and prints the model name |
| `POOT_REQUIRE_MODELS=1` | The test fails and names the variable or model |
| A `just test-checkpoint*` lane | Sets both for you; `POOT_MODELS_DIR` defaults to `~/models` |

The lanes:

- `just test-checkpoint` runs the CPU rows over the small checkpoints.
- `just test-checkpoint-wgpu` and `just test-checkpoint-rocm` run the device rows.
- `just test-checkpoint-large` reads 20B+ checkpoints; run it only on a host that holds them.

## The required-case gate

Every device lane ends by checking a list of required test names. Each listed test must have run and passed in that lane, and a skipped test never counts.

## Launch census

The plan digests of the model-free suite cover small fixtures, where launch geometry mostly does not bind. The launch census
covers the reference models at their real dimensions: one row per distinct contraction or attention request on each reference
target, with every field of its schedule. A change to a figure the planner reads changes a row.

```bash
just test-launch-census-regen
```

Regenerate the file after a change that moves plans on purpose, read the diff and state every changed row in the change. A change
that must not move plans leaves the diff empty. `rg dense_gemv` over the file lists every equation the dense decode Gemv serves.

## Repository guards

The model-free lane (`just test`) also runs checks on the tree itself:

| Recipe | Fails when |
| --- | --- |
| `docs-paths` | A tracked file names a `docs/` page that does not exist |
| `receipts-guard` | A tracked path sits under the top-level `receipts/` directory; per-change evidence is kept outside the repository |
| `home-paths-guard` | A tracked file contains a home-directory path; recorded data uses repo-relative paths |
| `variant-coverage` | A variant of a registered error or refusal enum names no covering test |

`scripts/check-home-paths.py --strip` rewrites the path prefix in JSON files and proves nothing else in them changed.
