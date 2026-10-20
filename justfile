# poot task runner; mirrors the commands in AGENTS.md's "Build / test" section.
# Run everything inside `nix develop` (pinned nightly + LLVM 22 + Vulkan stack).
# `just` with no recipe lists what is available.

# default: show the recipe list
default:
    @just --list

# ---- build / check ----------------------------------------------------------

# build the whole workspace
build:
    cargo build --workspace

# fast type-check the workspace (no codegen)
check:
    cargo check --workspace

# format every crate
fmt:
    cargo fmt --all

# lint the workspace; any warning fails (cargo arguments go before the `--`)
[positional-arguments]
clippy *args:
    # `poot-llm/cli` builds the `poot` binary, which needs it (anyhow, clap)
    cargo clippy --workspace --all-targets --features poot-llm/cli "$@" -- -D warnings

# ---- tests ------------------------------------------------------------------

# Every device lane selects its tests through this one command, and none may select zero: a recipe whose
# filter matches nothing (a renamed test, a moved module) fails instead of passing empty. The model-free
# lane passes the same flag in scripts/test-model-free.sh.
# `--lib --bins --tests` builds every test executable and no example: `cargo test`/nextest otherwise compile
# each example into target/debug/examples (hundreds of GiB across a target dir). `just clippy` still checks them.
# Every device lane's JUnit report lands under this directory, in `device-<lane>/results.xml` (the lane's
# profile in .config/nextest.toml); the lane recipes pass each path to scripts/check-required-cases.py.
results := "target/nextest"

nextest := "cargo nextest run --no-tests=fail --no-fail-fast --lib --bins --tests"

# The directory that holds the checkpoints model tests load. A test finds a model only through
# POOT_MODELS_DIR (there is no default in test code), so the checkpoint lanes set it here, where a developer
# sees it: override it in the environment. Outside the checkpoint lanes a model missing from it skips, with its name
# printed; the checkpoint lanes also set POOT_REQUIRE_MODELS=1, which makes a missing model a failure.
models_dir := env("POOT_MODELS_DIR", home_directory() / "models")

# The default lane covers every workspace package except those that need a device (`gate_policy` in
# scripts/test-model-free.sh names each package and, for an excluded one, why), then runs doctests. Devices
# are hidden and required in it, so a test that reaches for one fails. Tests that need a checkpoint or a
# device, and tests named `*_real_dims`, stay out; they run in the lanes below.

# The local merge gate (owner ruling Q8, 2026-10-18: no CI, every check runs on this box or the laptop): format
# check, clippy with every warning an error, then the model-free lane. Each step fails the recipe; none is skipped.
gate: fmt-check clippy test

# the gate's own self-test: it runs fmt-check, clippy -D warnings and the model-free lane in order, and fmt-check fails a
# fixture crate with a rustfmt diff (scripts/check-gate.py)
gate-check:
    python3 scripts/check-gate.py --self-test

# fail when any crate has a rustfmt diff
fmt-check:
    cargo fmt --all --check

# the one entry point for the whole gate: dead-pub and unused-dependency checks, every self-test, the model-free lane
test: test-model-free

# the model-free gate: dead-pub and unused-dependency checks, docs-path references, the inventory and lane self-tests, device-probe
# self-test, then every model-free test and doctest
test-model-free: gate-check dead-pub machete docs-paths variant-coverage mutation-sample-report diagrams-check aie-link-driver npu-build vpu-build crates-stale-names cargo-freshness crates-standalone test-model-free-inventory test-lane-coverage test-bench-runner test-bench-harness
    scripts/with-required-backend.sh --self-test
    scripts/test-model-free.sh run

# The bench runner's host tests, which the model-free lane cannot carry (a binary with no library target, which the
# lane's `--lib` selection rejects). They are rows of the model-free inventory like any
# other, so a removed or unlisted test fails the inventory check. Devices and checkpoints are hidden and required
# here exactly as in that lane. The script's POOT_MODEL_FREE_ACTIVE re-entry guard stops a recipe that
# re-enters itself.
test-bench-runner:
    scripts/test-model-free.sh run-isolated {{isolated_package}}

# The benchmark harness, all of it stdlib and GPU-free: its unit tests (result contract, row identity, the committed
# results' identity fields, the comparator, build.rs), the performance bank's claims, and the noise-aware comparator
# over the committed results (`bench-compare`). A recipe not named here runs in no gate: until POOT-1260 nothing did.
test-bench-harness: bench-compare
    python3 -m unittest discover -s benchmarks/harness/tests -p 'test_*.py'
    python3 benchmarks/harness/performance_bank.py

# The noise-aware comparator (benchmarks/harness/bench.py compare-committed) on the committed results: the baseline is
# the last committed set of each cell. With no candidate the set is compared with itself, which proves it is a
# verifiable baseline (enough repeats, one binary, a band that excludes a practical loss) and fails the gate when it
# is not; with candidate run directories (`just bench-compare benchmarks/results benchmarks/results/<run>-r1 ...`) it
# judges them against that set. `results` is the directory holding the committed set. Exit 1: a regression or a failed
# repeat; 2: a cell that cannot be verified.
bench-compare results='benchmarks/results' *candidate:
    python3 benchmarks/harness/bench.py compare-committed --results-dir {{results}} --candidate {{candidate}}

# Record the dense cell of one local backend: 5 repeats of qwen2.5-0.5b decode-128 through the release runner, one run
# directory per repeat under benchmarks/results/. Holds the GPU: run it as `flock /tmp/poot-gpu.lock just
# bench-record wgpu`, with the box otherwise idle, and commit the directories with `bench index` regenerated.
bench-record backend model='qwen2.5-0.5b' scenario='decode-128':
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release -p poot-bench-runner
    POOT_BENCH_BACKEND={{backend}} POOT_BENCH_POOT_BIN="${CARGO_TARGET_DIR:-target}/release/poot-bench-runner" \
        python3 benchmarks/harness/bench.py run --framework poot --model {{model}} --scenario {{scenario}} \
        --models-dir {{models_dir}} --repeats 5

# Fail when a test runs in no lane. A test is covered when it is model-free or real-dims (the model-free lane), a
# checkpoint test (a row of scripts/required-checkpoint-cases.txt, run by the `test-checkpoint*` recipes), or matched by one of the
# selections above; a device test or a test of a package outside the inventory that matches none is the gap this
# gate exists to close (card 1058: 155 of 239 poot-wgpu-device tests once ran nowhere).
test-lane-coverage:
    scripts/check-lane-coverage.py --self-test
    scripts/check-lane-coverage.py '{{wgpu_selection}}' '{{vulkan_selection}}' '{{rocm_selection}}' '{{ptx_selection}}' '{{ptx_multi_device_selection}}' '{{kernels_selection}}' '{{npu_selection}}'

# The add-a-model standing test (ADR-0104 rule 3), host half: a family defined inside the test (a config type, a
# checkpoint-name table, a `StandardFamily` over the shared wrapper) registers with `Registry::builtin_plus`,
# loads from a written checkpoint, compiles Prefill and Decode for the wgpu, ROCm and PTX
# reference targets, decodes the pinned oracle tokens and runs under a driver with the same typed limits as a
# built-in family. The rows are model-free, so `just test` (test-model-free) already runs them: the inventory
# classes each one. This recipe is the focused entry a stage gate names. The device half (`_wgpu`, `_rocm`,
# `_ptx`) runs in the device lanes.
test-add-a-model:
    cargo nextest run --no-tests=fail --no-fail-fast -p poot-llm --test add_a_model -E 'not test(/_(wgpu|rocm|ptx)$/)'

# render website/diagrams/*.svg to the PNGs the README and the website use (flags: --out DIR, --only a,b,
# --publish-dir DIR, --check, e.g. `just diagrams --only model-tracing`); see scripts/diagrams.py
diagrams *args:
    scripts/diagrams.py {{args}}

# the renderer's self-test (CLI paths, reference-check failures), then fail when a committed diagram PNG or index.json
# differs from its source, or the README and the website reference other images than index.json
diagrams-check:
    scripts/diagrams.py --self-test
    scripts/diagrams.py --check

# fail on a `docs/<name>.md` reference in a tracked file whose target does not exist (project notes live in
# Backstage, not in this repository's docs/)
docs-paths:
    scripts/check-docs-paths.py --self-test
    scripts/check-docs-paths.py

# every variant of the registered error and refusal enums names a covering test in scripts/variant-coverage.tsv (or is a
# listed gap), and each listed test still exists and still names its variant (POOT-1221)
variant-coverage:
    scripts/check-variant-coverage.py --self-test
    scripts/check-variant-coverage.py

# the committed cargo-mutants sample report is consistent; the sample itself is `scripts/mutation-sample.sh` (nightly or by
# hand, about 15 minutes) and fails when survivors rise above the committed count (POOT-1221)
mutation-sample-report:
    scripts/check-mutation-sample.py --self-test
    scripts/check-mutation-sample.py

# worktrees share a slot target dir, so cargo must decide freshness by source content, not mtime (POOT-1037)
cargo-freshness:
    scripts/check-cargo-freshness.sh

# rustc's dead_code lint skips every item reachable from a library crate root, so an unused `pub` item passes
# `just check` and `just clippy`. This scan reports it (name based; scripts/dead-pub/dead_pub.py explains the
# rule). There is no allow-list and no exemption: every dead `pub` item fails.

# fail on a `pub` item that nothing uses
dead-pub:
    python3 scripts/dead-pub/dead_pub.py --self-test
    python3 scripts/dead-pub/dead_pub.py

# the AIE link driver's recipe refusals (POOT-1118); stdlib only, so it needs no NPU toolchain
aie-link-driver:
    python3 scripts/poot-aie-link.py --self-test

# Build and lint poot-npu with its `xdna` feature (XRT shim, device, device rows) against the current workspace APIs.
# The default build compiles the crate to its documentation only, so without this recipe the device code can rot
# unseen (it did, 2026-10-12 to 2026-10-17, POOT-1158). The feature needs `nix develop .#npu`; where that shell
# cannot be entered the recipe FAILS, unless POOT_ALLOW_NO_NPU_SHELL=1 says that this host has none (the laptop):
# then it says so and builds nothing. It builds only: the device rows run in `test-device-npu`.
npu-build:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! nix develop .#npu -c true >/dev/null 2>&1; then
        if [ "${POOT_ALLOW_NO_NPU_SHELL:-}" = 1 ]; then
            echo "npu-build: nix develop .#npu is unavailable here; poot-npu --features xdna NOT built (POOT_ALLOW_NO_NPU_SHELL=1)" >&2
            exit 0
        fi
        echo "npu-build: nix develop .#npu cannot be entered; poot-npu --features xdna was not built (set POOT_ALLOW_NO_NPU_SHELL=1 on a host that has no NPU shell)" >&2
        exit 1
    fi
    nix develop .#npu -c cargo clippy -p poot-npu --features xdna --all-targets -- -D warnings

# Build and lint poot-npu with its `vpu` feature (the Intel NPU device over Level Zero, the blob cache, the device
# rows) against the current workspace APIs. The Level Zero loader, the NPU driver and its compiler are `dlopen`ed,
# so this builds in the default shell; only running the rows needs `nix develop .#npu-intel`. It builds only: the
# rows run in `test-device-vpu`.
vpu-build:
    cargo clippy -p poot-npu --features vpu --all-targets -- -D warnings

# each workspace library builds alone (`cargo check --workspace` unifies features and hides a gated import: POOT-1296)
crates-standalone:
    scripts/check-crates-standalone.py

# fail when a crate source or manifest cites a deleted API (Runner, GpuExecutor, run_resident_kv, run_prefill, POOT_SPEC_*)
# as if it still existed; a comment block that says the name is gone is history and passes
crates-stale-names:
    python3 scripts/crates-stale-names.py --self-test
    python3 scripts/crates-stale-names.py

# fail on a dependency a manifest declares and its crate never uses
machete:
    cargo machete crates benchmarks/runners/poot

# Inventory-only contract check: builds and lists, does not execute tests.
test-model-free-inventory:
    scripts/test-model-free.sh self-test

# Run this after adding, renaming or deleting a test; a new test in a `reviewed` package (see `gate_policy` in
# scripts/test-model-free.sh) then needs its class chosen by hand: model-free, checkpoint or device.

# rewrite scripts/test-model-free-inventory.tsv from the test binaries
test-model-free-inventory-regen:
    scripts/test-model-free.sh regen

# rewrite crates/poot-models/tests/plan_summary/digests.tsv (the plan digests); a card that changes plans on
# purpose runs this and lists every changed graph in its landing note
test-plan-summary-regen:
    POOT_PLAN_SUMMARY_REGEN=1 cargo nextest run -p poot-models --test plan_summary plan_digests_match_the_checked_in_file

# rewrite the generated region of website/docs/reference/feature-matrix.md from what `compile` says about every
# registered family's fixture graphs and every weight format on each reference target. The matrix test fails
# when the page differs from the generator, so a planner or registry change that moves a cell runs this.
feature-matrix-regen:
    POOT_FEATURE_MATRIX_REGEN=1 cargo nextest run -p poot-models --test feature_matrix the_page_is_what_the_generator_renders

# tests that build real model dimensions, in the release profile (too slow for the debug default lane)
test-real-dims:
    scripts/test-model-free.sh run-real-dims --release

# Checkpoint lanes. A lane's rows are listed by hand: scripts/required-checkpoint-cases.txt for the CPU lanes,
# scripts/required-device-cases/checkpoint-<backend>.txt for the device lanes (the required-case gate reads those
# too). Each recipe runs its lane's rows by exact name (an `#[ignore]`d row too: `--run-ignored all` is safe only
# because the filterset names every row) with POOT_REQUIRE_MODELS=1, so a model missing from the models directory
# fails its row instead of printing a SKIP. A device lane also requires its backend and runs the required-case
# gate, which fails a row that skipped. `test-lane-coverage` fails when a checkpoint test or an ignored test is
# in no lane's list and when a listed row names no test.

# checkpoint lane: CPU rows over the small checkpoints in the models directory; a missing model fails
test-checkpoint:
    POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --run-ignored all -p poot-llm -p poot-load -p poot-serve \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint)"

# rows that read a 20B+ checkpoint (never on the dev box: a pod or host that holds the checkpoints and the memory)
test-checkpoint-large:
    POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --run-ignored all -p poot-llm -p poot-load -p poot-models \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-large)"

# every real-checkpoint device row on wgpu, wgpu and the models required, then the required-case gate
test-checkpoint-wgpu:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin checkpoint-wgpu {{results}}/device-checkpoint-wgpu/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU \
        env POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --profile device-checkpoint-wgpu --run-ignored all -p poot-llm \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-wgpu)"
    scripts/check-required-cases.py check checkpoint-wgpu {{results}}/device-checkpoint-wgpu/results.xml

# every real-checkpoint device row on ROCm, ROCm and the models required, then the required-case gate
test-checkpoint-rocm:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin checkpoint-rocm {{results}}/device-checkpoint-rocm/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_ROCM \
        env POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --profile device-checkpoint-rocm --run-ignored all -p poot-llm \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-rocm)"
    scripts/check-required-cases.py check checkpoint-rocm {{results}}/device-checkpoint-rocm/results.xml

# every real-checkpoint device row on PTX (an NVIDIA pod that holds the models), then the required-case gate
test-checkpoint-ptx:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin checkpoint-ptx {{results}}/device-checkpoint-ptx/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_PTX \
        env POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --profile device-checkpoint-ptx --run-ignored all -p poot-llm \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-ptx)"
    scripts/check-required-cases.py check checkpoint-ptx {{results}}/device-checkpoint-ptx/results.xml

# the real-checkpoint row on the raw Vulkan device, then the required-case gate
test-checkpoint-vulkan:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin checkpoint-vulkan {{results}}/device-checkpoint-vulkan/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_VULKAN \
        env POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --profile device-checkpoint-vulkan --run-ignored all -p poot-llm \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-vulkan)"
    scripts/check-required-cases.py check checkpoint-vulkan {{results}}/device-checkpoint-vulkan/results.xml

# the 20B+ device rows on wgpu: NEVER on the dev box (a pod or host with the checkpoints and the memory)
test-checkpoint-large-wgpu:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin checkpoint-large-wgpu {{results}}/device-checkpoint-large-wgpu/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU \
        env POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --profile device-checkpoint-large-wgpu --run-ignored all -p poot-llm \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-large-wgpu)"
    scripts/check-required-cases.py check checkpoint-large-wgpu {{results}}/device-checkpoint-large-wgpu/results.xml

# the 20B+ device rows on ROCm: NEVER on the dev box
test-checkpoint-large-rocm:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin checkpoint-large-rocm {{results}}/device-checkpoint-large-rocm/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_ROCM \
        env POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --profile device-checkpoint-large-rocm --run-ignored all -p poot-llm \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-large-rocm)"
    scripts/check-required-cases.py check checkpoint-large-rocm {{results}}/device-checkpoint-large-rocm/results.xml

# the 20B+ device rows on PTX (an NVIDIA pod with the memory for them)
test-checkpoint-large-ptx:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin checkpoint-large-ptx {{results}}/device-checkpoint-large-ptx/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_PTX \
        env POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1 \
        {{nextest}} --profile device-checkpoint-large-ptx --run-ignored all -p poot-llm \
        -E "$(scripts/check-lane-coverage.py --lane-filter checkpoint-large-ptx)"
    scripts/check-required-cases.py check checkpoint-large-ptx {{results}}/device-checkpoint-large-ptx/results.xml

# the real-dimension rows too slow for `test-real-dims` (minutes on the CPU oracle), in release
test-real-dims-long:
    {{nextest}} --release --run-ignored all -p poot-eval -p poot-models \
        -E "$(scripts/check-lane-coverage.py --lane-filter real-dims-long)"

# the rows that need the pinned Peano llc and the IRON link driver; without them the compile is a typed error
test-npu-toolchain:
    nix develop .#npu -c {{nextest}} --run-ignored all -p poot-kernelgen \
        -E "$(scripts/check-lane-coverage.py --lane-filter npu-toolchain)"

# Device-lane selection. Each lane's nextest filterset is defined once, here, and used twice: by the lane
# recipe, and by `test-lane-coverage`, which fails when a device test (or any test of a package the
# model-free lane excludes) matches none of them. Select a device crate by `package(...)`, never by listing its
# test binaries: a binary added later is then selected without anyone remembering to list it. The kernels lane
# runs its two packages whole, so it has no filter.
# A `package(...)` selection also runs that package's model-free tests (poot-wgpu-runtime, poot-rocm-runtime), which
# `just test` runs too: the repeat is harmless and costs seconds.
wgpu_selection := 'package(poot-wgpu-runtime) | (binary_id(=poot-llm) & test(/^driver::(device_tests|step::mtp_tests)::[a-z_0-9]+_wgpu$/)) | (binary_id(=poot-llm::add_a_model) & test(/_wgpu$/)) | (binary_id(=poot-llm::pooled_deepseek3) & test(/_wgpu$/)) | (binary_id(=poot-llm::pooled_qwen3moe) & test(/_wgpu$/)) | (binary_id(=poot-llm::beam) & test(/_wgpu$/)) | (binary_id(=poot-llm::timed_executor) & test(/_wgpu$/)) | (binary_id(=poot-llm::pooled_gemma4) & test(/_wgpu$/)) | (binary_id(=poot-llm::pooled_qwen3next) & test(/_wgpu$/)) | (binary_id(=poot-llm::recurrent_snapshot_cost) & test(/_wgpu$/)) | (binary_id(=poot-llm::speculate) & test(/_wgpu$/)) | (binary_id(=poot-llm::speculate_recurrent) & test(/_wgpu$/)) | (binary_id(=poot-llm::staged_timing) & test(/_wgpu$/)) | binary_id(=poot-executor) | binary_id(=poot-executor::cr30_lifecycle) | binary_id(=poot-executor::staged_pipeline) | package(poot-wgpu-device) | binary_id(=poot-llm::wgpu_qwen2) | binary_id(=poot-llm::wgpu_gemma2) | binary_id(=poot-llm::wgpu_executor_contract) | binary_id(=poot-llm::wgpu_weight_map_binding) | binary_id(=poot-llm::wgpu_parity_models) | binary_id(=poot-llm::wgpu_staged_pipeline) | binary_id(=poot-serve::bounded_drain_production) | (binary_id(=poot-serve::served_loop) & test(/_wgpu$/)) | (binary_id(=poot-serve::bin/poot-serve) & test(/^tests::lifecycle::device::[a-z_0-9]+_wgpu$/)) | (binary_id(=poot-serve::served_groups) & test(/_wgpu$/)) | test(/^prefill_kv_shared_pool_matches_contiguous$/) | test(/^qwen2_decode_gpu_matches_cpu$/)'
vulkan_selection := 'package(poot-vulkan-runtime) | package(poot-vulkan-device) | (binary_id(=poot-llm) & test(/^driver::device_tests::(recorded_[a-z_0-9]+|attached_device_limit_refuses_an_oversized_constant)_vulkan$/)) | binary_id(=poot-llm::vulkan_generate) | binary_id(=poot-llm::vulkan_cross_backend) | binary_id(=poot-llm::vulkan_parity_models) | (binary_id(=poot-serve::served_groups) & test(/_vulkan$/)) | (binary_id(=poot-serve::served_loop) & test(/_vulkan$/))'
rocm_selection := 'package(poot-rocm-runtime) | (binary_id(=poot-llm) & test(/^driver::(device_tests|step::mtp_tests)::[a-z_0-9]+_rocm$/)) | (binary_id(=poot-llm::add_a_model) & test(/_rocm$/)) | (binary_id(=poot-llm::pooled_deepseek3) & test(/_rocm$/)) | (binary_id(=poot-llm::pooled_qwen3moe) & test(/_rocm$/)) | package(poot-rocm-device) | binary_id(=poot-llm::rocm_executor_contract) | binary_id(=poot-llm::rocm_executor_coverage) | binary_id(=poot-llm::rocm_routed_prefill_unguarded) | binary_id(=poot-llm::rocm_parity_models) | binary_id(=poot-llm::rocm_staged_pipeline) | binary_id(=poot-llm::rocm_weight_map_binding) | (binary_id(=poot-llm::beam) & test(/_rocm$/)) | (binary_id(=poot-llm::timed_executor) & test(/_rocm$/)) | (binary_id(=poot-llm::pooled_gemma4) & test(/_rocm$/)) | (binary_id(=poot-llm::pooled_qwen3next) & test(/_rocm$/)) | (binary_id(=poot-llm::recurrent_snapshot_cost) & test(/_rocm$/)) | (binary_id(=poot-llm::speculate) & test(/_rocm$/)) | (binary_id(=poot-llm::speculate_recurrent) & test(/_rocm$/)) | binary_id(=poot-llm::rocm_batched_decode_plan_classify) | (binary_id(=poot-serve::served_loop) & test(/_rocm$/)) | (binary_id(=poot-serve::bin/poot-serve) & test(/^tests::lifecycle::device::[a-z_0-9]+_rocm$/)) | (binary_id(=poot-serve::served_groups) & test(/_rocm$/))'
ptx_single_device_rows := '(binary_id(=poot-llm) & test(/^driver::(device_tests|step::mtp_tests)::[a-z_0-9]+_ptx$/)) | (binary_id(=poot-llm::add_a_model) & test(/_ptx$/)) | (binary_id(=poot-llm::beam) & test(/_ptx$/)) | (binary_id(=poot-serve::bin/poot-serve) & test(/^tests::lifecycle::device::[a-z_0-9]+_ptx$/)) | (binary_id(=poot-llm::speculate) & test(/_ptx$/)) | (binary_id(=poot-llm::speculate_recurrent) & test(/_ptx$/)) | (binary_id(=poot-llm::staged_timing) & test(/_ptx$/)) | (binary_id(=poot-llm::timed_executor) & test(/_ptx$/)) | (binary_id(=poot-llm::pooled_deepseek3) & test(/_ptx$/)) | (binary_id(=poot-llm::pooled_qwen3moe) & test(/_ptx$/)) | (binary_id(=poot-llm::pooled_gemma4) & test(/_ptx$/)) | (binary_id(=poot-llm::pooled_qwen3next) & test(/_ptx$/)) | (binary_id(=poot-serve::served_groups) & test(/_ptx$/)) | package(poot-ptx-device) | binary_id(=poot-llm::ptx_gemma2_geglu) | binary_id(=poot-llm::ptx_sc_acceptance) | binary_id(=poot-llm::ptx_routed_prefill_unguarded) | binary_id(=poot-llm::ptx_parity_models) | binary_id(=poot-llm::ptx_staged_pipeline) | binary_id(=poot-llm::ptx_weight_map_binding) | package(poot-ptx-runtime)' + ' | (' + kernels_package_selection + ' & ' + ptx_named_rows + ')'
# The multi-device rows (two NVIDIA GPUs, and the four-way tensor-parallel row) live in three test binaries, and a
# binary is the property that says how many devices a row needs: every row of one opens its devices through the
# same `open_two` / `open_devices`. `test-device-ptx` excludes them (on a one-GPU machine they fail with `invalid
# device ordinal` instead of falling outside the lane); `test-device-ptx-2gpu` and `-4gpu` select them, split by the
# four-way name. Multi-device rows live only in these binaries; a binary of one-device rows never holds one. The
# selector cannot tell a multi-device row in a single-device binary (it is selected by `package(poot-ptx-device)` and
# fails with `invalid device ordinal` on a one-GPU machine). What catches it is the classification gate: on a
# many-GPU machine it passes and is in neither the required list nor `ptx.unlisted.txt`, so
# `scripts/check-required-cases.py ptx` fails on an unclassified passing row. Listing it as required would hide it,
# so a reviewer checks that a new `ptx.txt` row is a one-device row.
ptx_multi_device_selection := 'binary_id(=poot-llm::ptx_multi_device_pipeline) | binary_id(=poot-llm::ptx_tensor_parallel) | binary_id(=poot-serve::served_multi_device)'
ptx_selection := '(' + ptx_single_device_rows + ') & not (' + ptx_multi_device_selection + ')'
# The kernels lane runs its two packages whole, less the rows that dispatch on an NVIDIA device: a row named
# `*_on_ptx` skips on any other machine, and a skip is never a pass, so those rows belong to `test-device-ptx`.
kernels_package_selection := '(package(pootc) | package(poot-kernelgen))'
ptx_named_rows := 'test(/_on_ptx(_kquant)?$/)'
kernels_selection := kernels_package_selection + ' & not ' + ptx_named_rows
npu_selection := 'package(poot-npu)'

# poot-bench-runner is a binary with no library target, so its host tests (inventory rows of the `isolated` policy,
# class model-free) run in `test-bench-runner`.
isolated_package := 'poot-bench-runner'

# Required small-device lanes. Each runs through scripts/with-required-backend.sh, which clears every
# backend requirement, sets only its own, and fails the lane before any test runs
# when no device of that backend is visible. The nextest group serializes inside this invocation only.
# Every lane also runs the required-case gate (card 670): scripts/check-required-cases.py --self-test proves
# the gate itself first, the lane runs under its own nextest profile so results land in
# target/nextest/device-<lane>/results.xml, then scripts/check-required-cases.py <lane> fails unless every
# case in scripts/required-device-cases/<lane>.txt ran and passed - a skip never counts as a pass.

# wgpu lane: the migrated executor-contract, packed serving/launch-widening, I32, state/capture, failure and
# cache-identity cases, and every poot-wgpu-device test (a test no lane selects goes red silently); wgpu device required,
# then the required-case gate
test-device-wgpu:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin wgpu {{results}}/device-wgpu/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU \
        {{nextest}} --profile device-wgpu -p poot-wgpu-device -p poot-executor -p poot-llm -p poot-wgpu-runtime -p poot-serve \
        -E '{{wgpu_selection}}'
    scripts/check-required-cases.py check wgpu {{results}}/device-wgpu/results.xml

# The poot-executor rows this lane requires (cr30_lifecycle, kernel identity) also run in `just test`;
# poot-wgpu-device is excluded from it (gate_policy: every test dispatches on a wgpu device), so this lane is
# where its required cases run.

# kernel GPU suites (pootc dispatches, poot-kernelgen runs) on wgpu, device required, then the required-case gate
test-device-kernels:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin kernels {{results}}/device-kernels/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU \
        {{nextest}} --profile device-kernels -p pootc -p poot-kernelgen --test-threads=1 -E '{{kernels_selection}}'
    scripts/check-required-cases.py check kernels {{results}}/device-kernels/results.xml

# raw Vulkan lane: the poot-vulkan-runtime add dispatch and live caps receipt, and the poot-vulkan-device executor
# contract, parity table and driver generation rows (Card 553), Vulkan device required, run under the Khronos validation
# layer (a missing layer fails the lane; any layer warning or error fails the test that caused it), then the required-case gate
test-device-vulkan:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin vulkan {{results}}/device-vulkan/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_VULKAN env POOT_VULKAN_VALIDATION=1 \
        {{nextest}} --profile device-vulkan -p poot-vulkan-runtime -p poot-vulkan-device -p poot-llm -p poot-serve \
        -E '{{vulkan_selection}}'
    scripts/check-required-cases.py check vulkan {{results}}/device-vulkan/results.xml

# ROCm lane: the migrated executor-contract, packed-serving and parity suites (card 548; the old
# resident probes went with resident.rs) and every poot-rocm-device test (a test no lane selects goes red silently),
# ROCm device required, then the required-case gate
test-device-rocm:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin rocm {{results}}/device-rocm/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_ROCM \
        {{nextest}} --profile device-rocm -p poot-rocm-device -p poot-llm -p poot-rocm-runtime -p poot-serve \
        -E '{{rocm_selection}}'
    scripts/with-required-backend.sh POOT_REQUIRE_ROCM cargo run -p poot-rocm-device --bin rocm-check -- --add
    scripts/check-required-cases.py check rocm {{results}}/device-rocm/results.xml

# NPU lane: every poot-npu device row on the XDNA2 NPU, through the real compile -> Engine -> XRT route. It
# runs in `nix develop .#npu` (XRT, Peano and the IRON driver). Each row takes `/tmp/poot-npu.lock` itself
# (flock, in the test), so two runs never share the device; the lock is not held by this recipe, which would
# deadlock the rows. POOT_REQUIRE_NPU=1 is set by the backend wrapper: an absent device or toolchain fails
# the lane, it never skips.
# Then the required-case gate. Absent the device, a bare `cargo nextest run -p poot-npu --features xdna` skips
# each row with a SKIP report, which the required-case gate never counts as a pass.
test-device-npu:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin npu {{results}}/device-npu/results.xml
    nix develop .#npu -c scripts/with-required-backend.sh POOT_REQUIRE_NPU \
        {{nextest}} --profile device-npu -p poot-npu --features xdna --test-threads=1 -E '{{npu_selection}}'
    scripts/check-required-cases.py check npu {{results}}/device-npu/results.xml

# Intel NPU lane (POOT-1164): every poot-npu `vpu` row on the Lunar Lake laptop's NPU, through the real compile ->
# Engine -> vendor graph compiler -> Level Zero route, plus the blob cache and the device's own contract. It runs
# in `nix develop .#npu-intel` (the Level Zero loader, the NPU driver and its compiler on LD_LIBRARY_PATH). Each
# row takes `/tmp/poot-vpu.lock` itself, so the recipe does not. POOT_REQUIRE_INTEL_NPU=1 is set by the backend
# wrapper: an absent device or driver stack fails the lane, it never skips; the required-case gate then checks
# every row ran.
test-device-vpu:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin vpu {{results}}/device-vpu/results.xml
    nix develop .#npu-intel -c scripts/with-required-backend.sh POOT_REQUIRE_INTEL_NPU \
        {{nextest}} --profile device-vpu -p poot-npu --features vpu --test-threads=1 -E '{{npu_selection}}'
    scripts/check-required-cases.py check vpu {{results}}/device-vpu/results.xml

# Level Zero device rows (POOT-1164) through poot-l0-runtime: the Intel NPU's graph extension (IR compiled by the
# driver's graph compiler, blob reload, recordings). There is no Level Zero GPU path. It runs in `nix develop .#npu-intel` (the Level Zero loader,
# NPU driver stack on LD_LIBRARY_PATH) on the Lunar Lake laptop. The rows are
# `#[ignore]`d, so they are run by test binary; no device means the context open fails and the recipe fails.
test-device-l0:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin l0 {{results}}/device-l0/results.xml
    nix develop .#npu-intel -c cargo nextest run --profile device-l0 -p poot-l0-runtime --run-ignored only \
        -E 'binary_id(=poot-l0-runtime::npu_graph)'
    scripts/check-required-cases.py check l0 {{results}}/device-l0/results.xml

# PTX lane: PTX implements the shared executor contract (card 549), so this mirrors the ROCm lane's own
# packed-serving theme (the whole packed_serving.rs binary: resident, capture/replay, prefill, the
# indexed packed expert contraction, and packed row-gather), the card's own SC-001/002/004/006/007/009/010/011
# acceptance rows (sc_acceptance.rs), the target-neutral matrix-fragment op's real NVPTX dispatch
# (card 530), and the rest of the migrated device suite SC-008 names: launch widening (the widened-vs-
# narrow indexed-contraction/row-gather regression guard), the cross-backend elementwise/reduce/matmul/gather
# parity suite (including its I32 index case, `gather_axis0_ptx_matches_cpu`), the Mixtral batched
# shared-pool decode/prefill receipt, the fused Gemma2 GeGLU receipt, the batched LoRA decode receipt,
# and (card 675) the Gumbel-max sampler's no-contract SC-001 - the committed production asset dispatched
# on a fixture where FMA contraction would flip the winner - run on an NVIDIA pod. Every selected test is
# an ordinary test that skips cleanly with no device (so it also runs under `just test`'s wgpu-gated lane,
# where it just skips) - this is the one
# lane that runs them with the device required. The required-case gate runs after the lane like every
# other device lane.
test-device-ptx:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin ptx {{results}}/device-ptx/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_PTX \
        {{nextest}} --profile device-ptx -p poot-ptx-device -p poot-ptx-runtime -p poot-kernelgen -p pootc -p poot-llm -p poot-serve \
        -E '{{ptx_selection}}'
    # The kernels `ptx-check` runs are staged by poot-kernelgen's emit tests into a directory made fresh for this run, so a
    # fixture left by an earlier run can never satisfy it.
    stage=$(mktemp -d) && trap 'rm -rf -- "$stage"' EXIT && \
        POOT_PTX_STAGE_DIR="$stage" {{nextest}} -p poot-kernelgen -E 'binary_id(=poot-kernelgen) | binary_id(=poot-kernelgen::emit_nvptx)' && \
        scripts/with-required-backend.sh POOT_REQUIRE_PTX cargo run -p poot-ptx-device --bin ptx-check -- "$stage"
    scripts/check-required-cases.py check ptx {{results}}/device-ptx/results.xml

# two-NVIDIA-GPU rows (the captured coordinator's pipeline and tensor-parallel decode across GPUs 0 and 1,
# POOT-748 and POOT-749, and the production server's requests over both placements, POOT-750): one device
# required and a second required too (a missing one fails instead of skipping), then the required-case gate
test-device-ptx-2gpu:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin ptx-2gpu {{results}}/device-ptx-2gpu/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_PTX env POOT_REQUIRE_PTX_2GPU=1 \
        {{nextest}} --profile device-ptx-2gpu -p poot-llm -p poot-serve -E '({{ptx_multi_device_selection}}) & not test(/four_way/)'
    scripts/check-required-cases.py check ptx-2gpu {{results}}/device-ptx-2gpu/results.xml

# four-NVIDIA-GPU rows (POOT-1102: a world-four tensor-parallel decode against single-device decode, every
# GPU pair exchanging partial sums): four devices required, then the required-case gate
test-device-ptx-4gpu:
    scripts/check-required-cases.py --self-test
    scripts/check-required-cases.py begin ptx-4gpu {{results}}/device-ptx-4gpu/results.xml
    scripts/with-required-backend.sh POOT_REQUIRE_PTX env POOT_REQUIRE_PTX_4GPU=1 \
        {{nextest}} --profile device-ptx-4gpu -p poot-llm -E '({{ptx_multi_device_selection}}) & test(/four_way/)'
    scripts/check-required-cases.py check ptx-4gpu {{results}}/device-ptx-4gpu/results.xml

# Alias for the default GPU backend.
test-gpu: test-device-wgpu

# the criterion benches: the CPU-oracle decode step (poot-models) and wgpu op dispatch (poot-wgpu-device, wgpu device
# required). They time host paths and carry no baseline a gate compares; the end-to-end model benchmarks are `bench-record`.
bench:
    cargo bench -p poot-models --bench decode
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU cargo bench -p poot-wgpu-device --bench op_dispatch

# ---- kernel assets ----------------------------------------------------------

# regenerate the committed imported-kernel assets (pootc/kernels/<family>/*.rs) from
# `poot_graph_plan::MANIFEST`, the one list (card 559; `committed_assets_match_their_kernel_sources`
# checks the same list and fails if an asset goes stale).
regen-kernel-assets:
    cargo nextest run --no-tests=fail --test import_run --run-ignored all -p pootc -E "$(scripts/check-lane-coverage.py --lane-filter regen-kernel-assets)"

# ---- docs hygiene -----------------------------------------------------------

# check docs without writing: formatting (prettier), broken links (lychee), ascii.
# Reports all even if one fails; non-zero exit if any has issues. The Docusaurus site (website/) is
# checked by `just site-check`.
docs-check:
    #!/usr/bin/env bash
    set -uo pipefail
    rc=0
    files=$(git ls-files '*.md' ':!website' ':!benchmarks/results')
    echo "-- formatting --"
    prettier --check $files || rc=1
    echo "-- links --"
    lychee --no-progress $files || rc=1
    echo '-- retired names (the generate command is `poot generate`) --'
    # The pattern is written with a bracket so this line does not match itself.
    if git grep -nI 'qwen2-gener[a]te' -- . ; then
        echo 'the old generate binary name was retired for `poot generate`: replace it where listed above'
        rc=1
    fi
    echo "-- docs/ path references --"
    just docs-paths || rc=1
    echo "-- architecture guide (layer table, stale claims, checked-against sha) --"
    # The guide and overview live in Backstage, a sibling checkout. A missing checkout fails the stage;
    # POOT_SKIP_ARCHITECTURE_DOC=1 is the explicit opt-out for a clone that has none.
    if [ -d ../backstage/projects/poot ]; then
        python3 scripts/check-architecture-doc.py --self-test || rc=1
        python3 scripts/check-architecture-doc.py || rc=1
    elif [ "${POOT_SKIP_ARCHITECTURE_DOC:-}" = 1 ]; then
        echo "../backstage not found: skipped by POOT_SKIP_ARCHITECTURE_DOC=1"
    else
        echo "../backstage not found: clone it beside this repo, or set POOT_SKIP_ARCHITECTURE_DOC=1"
        rc=1
    fi
    echo "-- ascii (no em-dashes / non-ascii) --"
    if grep -nP '[^\x00-\x7F]' $files; then
        echo "non-ASCII characters found (see above) - replace with ASCII equivalents"
        rc=1
    fi
    exit $rc

# auto-fix the docs: reformat every tracked markdown file in place (prettier). Broken links and non-ascii
# can't be auto-fixed; run `just docs-check` to see them. website/ (Docusaurus-managed) and
# benchmarks/results/ (orchestrator-generated reports; prettier mangles tables when a cell contains a
# literal `|`) are excluded.
docs-fix:
    prettier --write $(git ls-files '*.md' ':!website' ':!benchmarks/results')

# ---- website: the Docusaurus docs site in ./website -------------------------
# Runs bun via the devshell. The site is the user-facing manual only (getting started, architecture,
# usage, examples); internal project tracking lives in Backstage.

# install the site's node dependencies (first run, or after package.json changes)
site-install:
    cd website && bun install

# install deps only if missing; the dependency of the recipes below, so a fresh checkout can run
# `just site-dev` directly.
_site-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d website/node_modules ] || (cd website && bun install)

# run the live dev server with hot reload (http://localhost:3000)
site-dev: _site-deps
    cd website && bun start

# production build into website/build (also the broken-link checker)
site-build: _site-deps
    cd website && bun run build

# serve the production build locally
site-serve: _site-deps
    cd website && bun run serve

# the site's checks: TypeScript typecheck, a production build (broken links), names the code no longer has, the develop/ Rust snippets, and the ascii rule over
# tracked text sources. git grep ignores untracked build/vendor files; bun.lock is generated metadata.
site-check: _site-deps
    #!/usr/bin/env bash
    set -uo pipefail
    rc=0
    echo "-- typecheck --"
    (cd website && bun run typecheck) || rc=1
    echo "-- build (broken-link check) --"
    (cd website && bun run build) || rc=1
    echo "-- stale names (identifiers, variables, crates, paths, recipes, flags the code no longer has) --"
    scripts/site-stale-names.sh || rc=1
    echo "-- rust snippets (the develop/ pages' code compiles) --"
    scripts/site-rust-snippets.sh || rc=1
    echo "-- card numbers (the site never names an internal card or board id) --"
    scripts/site-card-numbers.sh || rc=1
    echo "-- NPU summaries (what an NPU runs is stated on architecture/npu.mdx only) --"
    scripts/site-npu-summaries.sh || rc=1
    echo "-- ascii (no em-dashes / non-ascii) --"
    ascii_rc=0
    git grep -nIP '[^\x00-\x7F]' -- website ':!website/bun.lock' || ascii_rc=$?
    if [ "$ascii_rc" -eq 0 ]; then
        echo "non-ASCII characters found (see above) - replace with ASCII equivalents"
        rc=1
    elif [ "$ascii_rc" -ne 1 ]; then
        echo "ASCII scan failed with status $ascii_rc"
        rc=1
    fi
    exit $rc
