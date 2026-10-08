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
    # `poot-llm/cli` builds the two binaries (`qwen2-generate`, `poot-caption`), which need it (anyhow)
    cargo clippy --workspace --all-targets --features poot-llm/cli "$@" -- -D warnings

# ---- tests ------------------------------------------------------------------

# Every device lane selects its tests through this one command, and none may select zero: a recipe whose
# filter matches nothing (a renamed test, a moved module) fails instead of passing empty. The model-free
# lane passes the same flag in scripts/test-model-free.sh.
# `--lib --bins --tests` builds every test executable and no example: `cargo test`/nextest otherwise compile
# each example into target/debug/examples (hundreds of GiB across a target dir). `just clippy` still checks them.
nextest := "cargo nextest run --no-tests=fail --no-fail-fast --lib --bins --tests"

# The directory that holds the checkpoints model tests load. A test finds a model only through
# POOT_MODELS_DIR (there is no default in test code), so the checkpoint lanes set it here, where a developer
# sees it: override it in the environment. A model missing from it still skips, with its name printed.
models_dir := env("POOT_MODELS_DIR", home_directory() / "models")

# The default lane covers every workspace package except those that need a device (`gate_policy` in
# scripts/test-model-free.sh names each package and, for an excluded one, why), then runs doctests. Devices
# are hidden and required in it, so a test that reaches for one fails. Tests that need a checkpoint or a
# device, and tests named `*_real_dims`, stay out; they run in the lanes below.

# the one entry point for the whole gate: dead-pub and unused-dependency checks, every self-test, the model-free lane
test: test-model-free

# the model-free gate: dead-pub and unused-dependency checks, the inventory and lane self-tests, device-probe
# self-test, then every model-free test and doctest
test-model-free: dead-pub machete cargo-freshness test-model-free-inventory test-lane-coverage test-bench-runner
    scripts/with-required-backend.sh --self-test
    scripts/test-model-free.sh run

# The bench runner's host tests, which the model-free lane cannot carry (its dependency on poot-llm turns the rocm
# feature on for every poot-llm build in the same invocation). Devices and checkpoints are hidden and required
# here exactly as in that lane. The recursion that once forked the box out of memory is removed at its root (a
# backtick in a double-quoted string of the script executed this recipe); the script's POOT_MODEL_FREE_ACTIVE
# re-entry guard is defence in depth, not the fix.
test-bench-runner:
    scripts/test-model-free.sh run-isolated {{isolated_package}}

# Fail when a test runs in no lane. A test is covered when it is model-free or real-dims (the model-free lane), a
# checkpoint test (`just test-optional` and the checkpoint recipes, with a model), or matched by one of the
# selections above; a device test or a test of a package outside the inventory that matches none is the gap this
# gate exists to close (card 1058: 155 of 239 poot-gpu tests once ran nowhere).
test-lane-coverage:
    scripts/check-lane-coverage.py --self-test
    scripts/check-lane-coverage.py '{{wgpu_selection}}' '{{vulkan_selection}}' '{{rocm_selection}}' '{{ptx_selection}}' '{{kernels_selection}}' 'package({{isolated_package}})'

# The add-a-model standing test (ADR-0104 rule 3), host half: a family defined inside the test (a config type, a
# checkpoint-name table, a tracer over `standard_stack` and `standard_layer`) registers through the public
# `Registry::register`, loads from a written checkpoint, compiles Prefill and Decode for the wgpu, ROCm and PTX
# reference targets, decodes the pinned oracle tokens and runs under a driver with the same typed limits as a
# built-in family. The rows are model-free, so `just test` (test-model-free) already runs them: the inventory
# classes each one. This recipe is the focused entry a stage gate names. The device half (`_wgpu`, `_rocm`,
# `_ptx`) runs in the device lanes.
test-add-a-model:
    cargo nextest run --no-tests=fail --no-fail-fast -p poot-llm --test add_a_model -E 'not test(/_(wgpu|rocm|ptx)$/)'

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

# rewrite crates/poot-models/tests/plan_summary/*.tsv (the plan-summary gate, card 599); a card that
# changes plans on purpose runs this and lists every changed line in its landing note
test-plan-summary-regen:
    POOT_PLAN_SUMMARY_REGEN=1 cargo nextest run -p poot-models --test plan_summary spirv_vulkan_plan_summary amd_gcn_plan_summary nvptx_plan_summary

# rewrite the generated region of website/docs/reference/feature-matrix.md from what `compile` says about every
# registered family's fixture graphs and every weight format on each reference target. The matrix test fails
# when the page differs from the generator, so a planner or registry change that moves a cell runs this.
feature-matrix-regen:
    POOT_FEATURE_MATRIX_REGEN=1 cargo nextest run -p poot-models --test feature_matrix the_page_is_what_the_generator_renders

# tests that build real model dimensions, in the release profile (too slow for the debug default lane)
test-real-dims:
    scripts/test-model-free.sh run-real-dims --release

# Optional broad coverage. Successful-test output is shown because many optional model/device tests pass
# after printing a runtime SKIP. This lane does not prove hardware coverage, and ignored tests (including
# large checkpoints) stay ignored.

# optional broad coverage: every workspace test and doctest, device tests skipping without a device
test-optional:
    POOT_MODELS_DIR={{models_dir}} cargo nextest run --workspace --lib --bins --tests --no-fail-fast --success-output final
    cargo test --workspace --doc

# Device-lane selection. Each lane's nextest filterset is defined once, here, and used twice: by the lane
# recipe, and by `test-lane-coverage`, which fails when a device test (or any test of a package the
# model-free lane excludes) matches none of them. Select a device crate by `package(...)`, never by listing its
# test binaries: a binary added later is then selected without anyone remembering to list it. The kernels lane
# runs its two packages whole, so it has no filter.
# A `package(...)` selection also runs that package's model-free tests (poot-runtime, poot-rocm-runtime), which
# `just test` runs too: the repeat is harmless and costs seconds.
wgpu_selection := 'package(poot-runtime) | (binary_id(=poot-llm) & test(/^driver::device_tests::[a-z_0-9]+_wgpu$/)) | (binary_id(=poot-llm::add_a_model) & test(/_wgpu$/)) | (binary_id(=poot-llm::speculate) & test(/_wgpu$/)) | binary_id(=poot-executor) | binary_id(=poot-executor::cr30_lifecycle) | package(poot-gpu) | binary_id(=poot-serve::bounded_drain_production) | (binary_id(=poot-serve::served_loop) & test(/_wgpu$/)) | test(/^prefill_kv_shared_pool_matches_contiguous$/) | test(/^qwen2_decode_gpu_matches_cpu$/)'
vulkan_selection := 'package(poot-vulkan-runtime) | package(poot-vulkan-device)'
rocm_selection := 'package(poot-rocm-runtime) | (binary_id(=poot-llm) & test(/^driver::device_tests::[a-z_0-9]+_rocm$/)) | (binary_id(=poot-llm::add_a_model) & test(/_rocm$/)) | package(poot-rocm-gpu) | (binary_id(=poot-llm::speculate) & test(/_rocm$/)) | binary_id(=poot-llm::card234_rocm_batched_decode_plan_classify) | (binary_id(=poot-serve::served_loop) & test(/_rocm$/))'
# `captured_executor_facade` spawns `cargo rustdoc` and a Cargo fixture, which a device pod does not carry; it is
# model-free and `just test` runs it, so the PTX lane excludes that binary by id.
ptx_selection := '(binary_id(=poot-llm) & test(/^driver::device_tests::[a-z_0-9]+_ptx$/)) | (binary_id(=poot-llm::add_a_model) & test(/_ptx$/)) | (binary_id(=poot-llm::speculate) & test(/_ptx$/)) | (package(poot-ptx-gpu) & not binary_id(=poot-ptx-gpu::captured_executor_facade)) | package(poot-ptx-runtime) | test(/^(wmma_tile_dispatches_and_matches_cpu_on_ptx|attention_leading_true_inf_mask_dispatches_and_matches_cpu_on_ptx|attention_fixture_bank_matches_cpu_on_ptx|attention_split_key_and_live_bound_match_cpu_on_ptx)$/)'
kernels_selection := 'package(pootc) | package(poot-kernelgen)'

# poot-bench-runner builds poot-llm with the rocm feature, which cannot unify into the model-free lane, so
# its host tests run in `test-bench-runner`.
isolated_package := 'poot-bench-runner'

# Required small-device lanes. Each runs through scripts/with-required-backend.sh, which clears the legacy
# global and every backend-specific requirement, sets only its own, and fails the lane before any test runs
# when no device of that backend is visible. The nextest group serializes inside this invocation only.
# Every lane also runs the required-case gate (card 670): scripts/check-required-cases.py --self-test proves
# the gate itself first, the lane runs under its own nextest profile so results land in
# target/nextest/device-<lane>/results.xml, then scripts/check-required-cases.py <lane> fails unless every
# case in scripts/required-device-cases/<lane>.txt ran and passed - a skip never counts as a pass.

# wgpu lane: the migrated executor-contract, packed serving/launch-widening, I32, state/capture, failure and
# cache-identity cases, and every poot-gpu test (a test no lane selects goes red silently); wgpu device required,
# then the required-case gate
test-device-wgpu:
    scripts/check-required-cases.py --self-test
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU \
        {{nextest}} --profile device-wgpu -p poot-gpu -p poot-executor -p poot-llm -p poot-runtime -p poot-serve \
        -E '{{wgpu_selection}}'
    scripts/check-required-cases.py wgpu

# The poot-executor rows this lane requires (cr30_lifecycle, kernel identity) also run in `just test`;
# poot-gpu is excluded from it (gate_policy: every test dispatches on a wgpu device), so this lane is
# where its required cases run.

# kernel GPU suites (pootc dispatches, poot-kernelgen runs) on wgpu, device required, then the required-case gate
test-device-kernels:
    scripts/check-required-cases.py --self-test
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU \
        {{nextest}} --profile device-kernels -p pootc -p poot-kernelgen --test-threads=1 -E '{{kernels_selection}}'
    scripts/check-required-cases.py kernels

# raw Vulkan lane: the poot-vulkan-runtime add dispatch and live caps receipt, and the poot-vulkan-device executor
# contract, parity table and driver generation rows (Card 553), Vulkan device required, run under the Khronos validation
# layer (a missing layer fails the lane; any layer warning or error fails the test that caused it), then the required-case gate
test-device-vulkan:
    scripts/check-required-cases.py --self-test
    scripts/with-required-backend.sh POOT_REQUIRE_VULKAN env POOT_VULKAN_VALIDATION=1 \
        {{nextest}} --profile device-vulkan -p poot-vulkan-runtime -p poot-vulkan-device \
        -E '{{vulkan_selection}}'
    scripts/check-required-cases.py vulkan

# ROCm lane: the migrated executor-contract, packed-serving and parity suites (card 548; the old
# resident probes went with resident.rs) and every poot-rocm-gpu test (a test no lane selects goes red silently),
# ROCm device required, then the required-case gate
test-device-rocm:
    scripts/check-required-cases.py --self-test
    scripts/with-required-backend.sh POOT_REQUIRE_ROCM \
        {{nextest}} --profile device-rocm -p poot-rocm-gpu -p poot-llm -p poot-rocm-runtime -p poot-serve --features poot-llm/rocm \
        -E '{{rocm_selection}}'
    scripts/check-required-cases.py rocm

# The poot-llm tests that exist only under `--features rocm`. Nothing else builds the feature,
# so without this recipe those tests never compile. It reports how many test binaries the feature turns from
# empty into non-empty and fails when there are none.

# rocm-feature tests (--features rocm, release) on the local ROCm device, device required
test-rocm-feature:
    #!/usr/bin/env bash
    set -euo pipefail
    # Release, as the rocm tests document: their debug run is codegen-bound (one real 3B test ran over 25 minutes).
    # Checkpoint lane: POOT_REQUIRE_MODELS=1 makes an unset models directory a failure, not a lane of skips.
    export POOT_MODELS_DIR={{models_dir}} POOT_REQUIRE_MODELS=1
    packages=(-p poot-llm --release)
    features=(--features poot-llm/rocm)
    dir=$(mktemp -d)
    trap 'rm -rf -- "$dir"' EXIT
    cargo nextest list --lib --bins --tests "${packages[@]}" --message-format json >"$dir/default.json"
    cargo nextest list --lib --bins --tests "${packages[@]}" "${features[@]}" --message-format json >"$dir/rocm.json"
    # Suites that list no test by default and some with the feature.
    non_empty=$(jq -r --slurpfile default "$dir/default.json" '
        .["rust-suites"] | to_entries[]
        | select((.value.testcases | length) > 0)
        | select((($default[0]["rust-suites"][.key].testcases // {}) | length) == 0)
        | .key
    ' "$dir/rocm.json")
    count=$(grep -c . <<<"$non_empty" || true)
    echo "rocm feature: $count test binaries are empty by default and non-empty with the feature"
    if [ "$count" -eq 0 ]; then
        echo "the rocm feature added no tests to any empty test binary: the feature is not being built" >&2
        exit 1
    fi
    # The active tests the feature adds, selected by exact identity.
    selected=$(jq -r --slurpfile default "$dir/default.json" '
        .["rust-suites"] | to_entries[]
        | .key as $binary
        | [ .value.testcases | to_entries[]
            | select(.value.ignored | not)
            | select($default[0]["rust-suites"][$binary].testcases[.key] == null)
            | "test(=" + .key + ")" ]
        | select(length > 0)
        | "(binary_id(=" + $binary + ") & (" + join(" | ") + "))"
    ' "$dir/rocm.json" | paste -sd'|' | sed 's/|/ | /g')
    scripts/with-required-backend.sh POOT_REQUIRE_ROCM \
        {{nextest}} "${packages[@]}" "${features[@]}" --test-threads=1 -E "$selected"

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
    scripts/with-required-backend.sh POOT_REQUIRE_PTX \
        {{nextest}} --profile device-ptx -p poot-ptx-gpu -p poot-ptx-runtime -p poot-kernelgen -p poot-llm \
        -E '{{ptx_selection}}'
    scripts/check-required-cases.py ptx

# Alias for the default GPU backend.
test-gpu: test-device-wgpu

# run both criterion benches: CPU decode (poot-eval) and wgpu op dispatch (poot-gpu, wgpu device required)
bench:
    cargo bench -p poot-eval --bench decode
    scripts/with-required-backend.sh POOT_REQUIRE_WGPU cargo bench -p poot-gpu --bench op_dispatch

# ---- mutation sampling (card 538, R486-015) ----------------------------------

# recurring cargo-mutants sample on the high-survival modules 486 found (matcher, planner, eval, sampler);
# reports survivors under a dated Backstage receipt, does not make them die, and exits nonzero on any
# survivor or a failed baseline. `module` is one of the names in scripts/mutation-sample.sh, or `all`.
# Expect tens of minutes per module (cargo-mutants rebuilds and re-tests the crate once per mutant); `all`
# can run well past an hour, so run it deliberately rather than as a quick check.
mutation-sample module="all":
    scripts/mutation-sample.sh {{module}}

# fast, hermetic check of the recipe's own reporting/exit-code logic (stubbed cargo-mutants output, no
# build or nix develop needed)
mutation-sample-self-test:
    scripts/mutation-sample.sh self-test

# ---- push gate --------------------------------------------------------------

# push gate: formatting, the benchmark harness's own tests, then the whole model-free gate
pre-push:
    cargo fmt --all -- --check
    just -f benchmarks/justfile harness-test
    just test-model-free

# The hook is scripts/pre-push.sh; remove it with `rm "$(git rev-parse --git-path hooks/pre-push)"`.

# opt in to running the push gate before every `git push` (also from linked worktrees)
install-pre-push-hook:
    ln -sf "$(realpath scripts/pre-push.sh)" "$(git rev-parse --git-path hooks/pre-push)"
    @echo "installed $(git rev-parse --git-path hooks/pre-push)"

# ---- kernel assets ----------------------------------------------------------

# regenerate the committed imported-kernel assets (pootc/kernels/<family>/*.rs) from
# `poot_graph_plan::MANIFEST`, the one list (card 559; `committed_assets_match_their_kernel_sources`
# checks the same list and fails if an asset goes stale).
regen-kernel-assets:
    cargo test -p pootc --test import_run regen_kernel_assets -- --ignored

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
# usage, examples); internal project tracking lives in the repo's docs/.

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

# the site's checks: TypeScript typecheck, a production build (broken links), names the code no longer has, and the ascii rule over
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
    echo "-- card numbers (the site never names an internal card or board id) --"
    card_rc=0
    git grep -nIiP '\b(cards? #?[0-9]{3,4}[a-d]?|POOT-[0-9]+|POOT-ADR-[0-9]+)\b' -- website/docs website/src || card_rc=$?
    if [ "$card_rc" -eq 0 ]; then
        echo "internal card numbers found (see above) - the site holds user-facing truth only"
        rc=1
    elif [ "$card_rc" -ne 1 ]; then
        echo "card-number scan failed with status $card_rc"
        rc=1
    fi
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
