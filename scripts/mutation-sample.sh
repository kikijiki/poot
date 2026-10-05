#!/usr/bin/env bash
# Recurring mutation sampling (card 538): cargo-mutants over a seeded sample of the modules
# where 486's hand mutation sample left a live survivor (matcher legality, planner launch geometry and
# device limits, oracle corners, the Runner's generation loops and sampler boundaries). It reports
# survivors; it does not make them die (each stage card owns its own modules' tests, R486-015 out of
# scope). A stage card that adds a seam over a high-survival cluster adds its own row to `modules` below.
#
# usage: scripts/mutation-sample.sh <module>|all|self-test [-- cargo-mutants args]
#
# Run inside `nix develop` (cargo-mutants is a devShell package). `-o` always writes into a fresh temp
# dir; nothing under mutants.out/ is committed. Exit code: 0 only when every requested module's mutants
# were all caught; 1 if any module reports a survivor, a failed baseline (a live regression, not a
# hypothetical one) or a tool error. Expect this to take tens of minutes per module (cargo-mutants builds
# and re-tests the crate once per mutant) - `all` can run well past an hour; run it deliberately, not as
# a quick check.
#
# Known cosmetic quirk (card 538): the matcher module's `-F` seed also always
# pulls in two unrelated mutants at `flash_attention_capped` and `rope_fusion` ("delete field eqns from
# struct Graph expression"), regardless of the regex - reproduced with `-F` set to a string that cannot
# match anything. This looks like a cargo-mutants matching quirk on that mutant genre, not a mis-seeded
# regex (`--exclude-re` does not remove them either); the matcher receipt may carry these two extra rows.
#
# The oracle corners (`eval`) row seeds the live successors of 486's clusters: the half-width narrowing
# (`narrow_f32`, `store_as`; the deleted `tensor.rs` `round_bf16`/`round_f16`) and, since Card 556
# deleted the flash hand loops with `flash_mask_strides`, `evaluate_composite`, the one place a
# composite's decomposition is bound, charged and evaluated.
set -euo pipefail

# name | package | space-separated file globs | -F regex seeding the sample to 486's clusters (a
# function per known guard, not every mutant cargo-mutants can generate for the file)
#
# Seeds name the function, not `in <fn>$`: a whole-body replacement mutant ("replace f -> ... with Ok(..)")
# carries no ` in <fn>` suffix, so an `in`-anchored seed drops exactly those (the eval row seeded one
# mutant of five before this was fixed, Card 1013).
#
# Every seed must name a live file and a live function: `check_seeds` refuses (exit 1) a module whose
# file glob matches no file or whose seed alternative matches no `fn` in those files, because
# cargo-mutants itself silently seeds nothing for such a row (Card 1013).
readonly modules=(
  "matcher|poot-graph-plan|crates/poot-graph-plan/src/passes/attention_match.rs|match_attention|match_pre_mask|peel_repeat_kv"
  "planner|poot-graph-plan|crates/poot-graph-plan/src/predicates.rs|is_movement_view_candidate|gemv_grid|flash_decode_width|is_tiled_gemm_in|tiled_gemm_plan"
  "eval|poot-eval|crates/poot-eval/src/ops/cast.rs crates/poot-eval/src/walk.rs|narrow_f32|store_as|evaluate_composite"
  "runner|poot-llm|crates/poot-llm/src/core/sampler.rs crates/poot-llm/src/core/generate.rs|Sampler::adjust|Sampler::apply_constraint_mask|Sampler::truncated_candidates|Sampler::pick|Runner::generate_sampled|Runner::generate_kv_masked_tokens|Runner::hit_stop"
  # Epic 493's M4 packed-weight sample (designs/mutants-m4.md). These seeds name the function, not
  # `in <fn>$`: a whole-body replacement mutant ("replace check_blocks -> ... with Ok(())") carries no
  # ` in <fn>` suffix, so an `in`-anchored seed would drop exactly the mutants M4 found alive.
  "quant|poot-quant|crates/poot-quant/src/lib.rs crates/poot-quant/src/plan.rs|PackedPayload::decode_row|check_blocks|decode_blocks|scale_and_min|decode_dense"
  "packed-kernelgen|poot-kernelgen|crates/poot-kernelgen/src/packed/kernel.rs|contraction_tiled"
  "packed-load|poot-load|crates/poot-load/src/packed_safetensors/validation.rs crates/poot-load/src/gguf/quantization.rs|validate_index_header_bijection|verify_file|update_identity_usize|f32_to_f16"
  "packed-builders|poot-graph-ir|crates/poot-graph-ir/src/ops/linear.rs|validate_indexed_inputs|require_operand_type|packed_grouped_linear"
  "packed-recognizer|poot-graph-plan|crates/poot-graph-plan/src/passes/packed.rs|recognize_|validate_packed_carriers|reject_packed_dequant_escapes"
)

module_names() {
  local m name
  for m in "${modules[@]}"; do
    IFS='|' read -r name _ <<<"$m"
    printf '%s ' "$name"
  done
}

# Module-specific fixed cargo-mutants args, kept out of the `|`-joined table above because they can
# contain spaces that must stay grouped as one argv word (a nextest filter expression). Card 538: no module may fall back to a crate-wide default test target, even one that happens to be
# safe today - every module gets an explicit, named filter, chosen with the same rationale as
# `scripts/test-model-free.sh`'s own gate_policy rather than guessed from test names.
#
# poot-graph-ir, poot-graph-plan and poot-eval are all "host" packages in that gate_policy (self-tested:
# `just test-model-free-inventory` fails if a device- or model-needing test ever lands in a "host"
# package), so no test in matcher/planner/eval can load a checkpoint or reach a device by construction.
# `-E 'all()'` (still routed through nextest, not cargo-mutants' bare `cargo test` default) makes that an
# explicit, audited choice instead of an implicit fallback, at no cost today. A name-based exclude regex
# was tried and reverted here: dozens of real, host-only, device-plan tests in these crates are named for
# the backend they plan against (`nvptx_plan_summary_matches_the_checked_in_file`,
# `spirv_vulkan_plan_summary_matches_the_checked_in_file`, `decode_gemv_chunks_on_wgpu_at_...`, etc, never
# touching a device), so excluding on substrings like "vulkan"/"ptx"/"gpu" silently dropped the committed
# plan-summary gate (card 599) along with them - caught only because a re-run showed extra survivors for
# no code reason. `eval` additionally excludes `*_real_dims` tests because poot-eval's own real-dims
# numerics suite is genuinely slow, the same exclusion `scripts/test-model-free.sh` uses for the default
# lane (that suffix is a project-wide naming CONTRACT for exactly this purpose, unlike the reverted regex).
#
# poot-llm ("runner") is NOT host-only: it is "reviewed" in gate_policy, meaning it mixes model-free tests
# with checkpoint/device tests classed by hand, and its `architectures::*_load` tests skip only when
# POOT_MODELS_DIR is unset - if it were set (this box's ~/models is populated with ~178G of real
# checkpoints), cargo-mutants' default `cargo test -p poot-llm` would load one for real, once per mutant,
# the exact incident class that has OOM'd this box twice. So `runner` uses a positive include list instead
# of an exclude list: only the three in-crate test modules verified (by reading their bodies) to exercise
# sampler.rs/generate.rs with synthetic weights and no model/device reference -
# `core::sampler::sampler_tests`, `core::generate::alibi_runner_wiring_tests`,
# `core::generate::non_finite_logits_tests` - with `alibi_runner_wiring_tests`'s one GPU-dispatching test
# (`gpu_prefill_alibi_mask_does_not_hang`) excluded by name.
#
# The M4 rows (Epic 493). `quant` is a host package; its own tests now pin every M4 survivor, so it
# needs no downstream package. `packed-builders` mutates poot-graph-ir but its numeric row
# (`packed_grouped_linear_matches_the_per_row_expert_product`) lives in poot-eval, so the extra
# `-p poot-eval` after `--` reaches `nextest run` (cargo-mutants ignores `--test-package` next to
# `-p`); both are host packages, and poot-eval keeps its `real_dims` exclusion. `packed-kernelgen` and
# `packed-load` are `reviewed` packages, so each takes a positive include list of test modules read to
# be host-only: the kernelgen interpreter rows of the tiled contraction (never the `_on_{wgpu,rocm,ptx}`
# device rows in the same module), and poot-load's packed-safetensors tests (in-memory readers and temp
# files, no checkpoint) plus the GGUF quantization unit tests. `matcher` and `packed-recognizer` mutate
# poot-graph-plan's attention matcher and packed-chain recognizers (the passes moved there from
# poot-graph-ir in Card 626); their rows drive `compile` in the same host package, so no extra `-p`.
module_fixed_args() {
  case "$1" in
    matcher | planner | quant | packed-recognizer)
      printf '%s\n' --test-tool nextest -- -E 'all()'
      ;;
    packed-builders)
      printf '%s\n' --test-tool nextest -- -p poot-eval -E 'not test(/real_dims/)'
      ;;
    packed-kernelgen)
      printf '%s\n' --test-tool nextest -- -E \
        'test(=packed::contraction_tests::contraction_tiled_matches_reference_aligned_and_ragged) | test(/^packed::planar_contraction_tests::/)'
      ;;
    packed-load)
      printf '%s\n' --test-tool nextest -- -E \
        'binary_id(poot-load::packed_safetensors) | test(/^packed_safetensors::tests::/) | test(/^gguf::quantization::tests::/)'
      ;;
    eval)
      printf '%s\n' --test-tool nextest -- -E 'not test(/real_dims/)'
      ;;
    runner)
      printf '%s\n' --test-tool nextest -- -E \
        '(test(/^core::sampler::sampler_tests::/) | test(/^core::generate::alibi_runner_wiring_tests::/) | test(/^core::generate::non_finite_logits_tests::/)) & not test(/gpu_prefill_alibi_mask_does_not_hang/)'
      ;;
    *) ;;
  esac
}

usage() {
  echo "usage: $0 <module>|all|self-test [-- cargo-mutants args]" >&2
  echo "modules: $(module_names)" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
requested=$1
shift || true
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi
extra_args=("$@")

receipts_root="${POOT_BACKSTAGE_DIR:-../backstage}/projects/poot/receipts/mutation-sample"
commit=$(git rev-parse --short HEAD)
dirty=""
git diff --quiet --ignore-submodules HEAD -- || dirty=" (dirty worktree)"

# Fails (returns 1, naming the offender) when a module's files do not exist, a seed alternative names no
# `fn` in those files, or (given a package) `cargo mutants --list` finds no mutant for it.
check_seeds() {
  local name=$1 files=$2 seed=$3 package=${4:-}
  local -a existing=()
  local f alt ident bad=0
  for f in $files; do
    if [[ -f "$f" ]]; then
      existing+=("$f")
    else
      echo "mutation-sample: $name: seed file $f does not exist" >&2
      bad=1
    fi
  done
  local -a alts
  IFS='|' read -r -a alts <<<"$seed"
  for alt in "${alts[@]}"; do
    ident=${alt#in }
    ident=${ident%\$}
    ident=${ident##*::}
    if [[ ${#existing[@]} -eq 0 ]] || ! grep -Eq "fn ${ident}" "${existing[@]}"; then
      echo "mutation-sample: $name: seed '$alt' matches no fn in: $files" >&2
      bad=1
    elif [[ -n "$package" ]]; then
      local -a list_args=(-p "$package")
      for f in $files; do list_args+=(-f "$f"); done
      if [[ -z "$(cargo mutants --list "${list_args[@]}" -F "$alt" 2>/dev/null)" ]]; then
        echo "mutation-sample: $name: seed '$alt' yields no mutant" >&2
        bad=1
      fi
    fi
  done
  return "$bad"
}

run_one() {
  local name=$1 package=$2 files=$3 seed=$4
  check_seeds "$name" "$files" "$seed" "$package" || return 1
  local file_args=() f
  for f in $files; do
    file_args+=(-f "$f")
  done

  local -a fixed_args=()
  while IFS= read -r arg; do fixed_args+=("$arg"); done < <(module_fixed_args "$name")

  echo "mutation-sample: $name ($package: $files)"
  local out
  out=$(mktemp -d)
  local log="$out/run.log"
  local status=0
  # No kernel cache: a warm cache can serve a kernel compiled before the mutation, so a mutant in an
  # emitter would look caught or survived for the cache's reason, not the test's.
  POOT_KERNEL_CACHE=0 cargo mutants -p "$package" "${file_args[@]}" -F "$seed" -o "$out" --no-times \
    "${extra_args[@]}" "${fixed_args[@]}" >"$log" 2>&1 || status=$?

  mkdir -p "$receipts_root"
  local receipt="$receipts_root/$(date -u +%Y-%m-%d)-${name}.txt"
  local outcomes="$out/mutants.out/outcomes.json"
  local run_status=0
  if {
    echo "# mutation-sample: $name"
    echo "commit: $commit$dirty"
    echo "package: $package"
    echo "files: $files"
    echo "seed: $seed"
    echo "cargo-mutants exit: $status"
    echo
    if [[ -f "$outcomes" ]] && jq -e '.outcomes[0].scenario == "Baseline" and .outcomes[0].summary == "Failure"' "$outcomes" >/dev/null 2>&1; then
      # The baseline itself fails: a real regression is present right now, not a hypothetical one that
      # cargo-mutants would have to invent. Surface it plainly instead of a bare exit code or a
      # misleading "0 survivors" (missed.txt/caught.txt exist but are empty when the baseline aborts the
      # whole run before any mutant is tested) - a broken baseline is the most severe survivor signal a
      # recurring check can report.
      echo "## BASELINE FAILED - a test fails before any mutant is applied. Every mutant in this scope"
      echo "## is unverifiable until it is fixed; treat this location as a live, uncaught survivor."
      cat "$out/mutants.out/log/baseline.log" 2>/dev/null || tail -n 200 "$log"
      run_status=1
    elif [[ -f "$outcomes" ]]; then
      jq -r '"## survivors (\(.missed | length))"' "$outcomes"
      cat "$out/mutants.out/missed.txt" 2>/dev/null
      [[ -s "$out/mutants.out/missed.txt" ]] || echo "(none)"
      echo
      jq -r '"## caught: \(.caught | length)  unviable: \(.unviable | length)  timeout: \(.timeout | length)"' "$outcomes"
      [[ -s "$out/mutants.out/unviable.txt" ]] && { echo "## unviable"; cat "$out/mutants.out/unviable.txt"; }
      [[ -s "$out/mutants.out/timeout.txt" ]] && { echo "## timeout"; cat "$out/mutants.out/timeout.txt"; }
      # card 538: a plain survivor must fail as loudly as a baseline failure - this is a
      # recurring check, and nothing downstream (cron, CI, a human skimming an exit code) should have to
      # read prose to tell a clean run from a regression. cargo-mutants' own exit code ($status above) is
      # not trusted for this: it is recorded on the receipt but the decision is made from `.missed`
      # directly, since that is what the report itself just printed.
      if [[ "$(jq -r '.missed | length' "$outcomes")" -gt 0 ]]; then
        run_status=1
      fi
    else
      # cargo-mutants failed before writing any output at all (a bad --file glob, an unbuildable
      # package): a tool error, not a survivor report.
      echo "## ERROR - cargo-mutants produced no outcomes.json; this is a tool/config failure, not a"
      echo "## survivor report."
      tail -n 200 "$log"
      run_status=1
    fi
    (exit "$run_status")
  } | tee "$receipt"
  then
    run_status=${PIPESTATUS[0]}
  else
    run_status=${PIPESTATUS[0]}
  fi
  # The block above runs in a subshell as the pipeline's first stage, so $run_status set inside it never
  # reaches this scope; PIPESTATUS[0] carries its exit code (the final `(exit ...)` in the block) instead.
  # The if/else (not `|| true`) is required: running any command after the pipe, even `true`, overwrites
  # PIPESTATUS before it can be read. The two branches are deliberately identical (both just read
  # PIPESTATUS[0]) - the `if` exists only to keep `set -e` from killing the script on a nonzero pipeline.
  echo "receipt: $receipt"
  rm -rf "$out"
  return "$run_status"
}

# Exercises run_one's reporting/exit-code logic against the three shapes cargo-mutants' own output can
# take, without running real cargo-mutants (no build, no test, no nix develop needed): a stub `cargo` on
# PATH intercepts `cargo mutants ... -o <dir> ...` and writes a canned mutants.out/outcomes.json (plus the
# text files run_one reads) for the scenario named by MUTATION_SAMPLE_SELFTEST_SCENARIO, then this
# recurses into "$0 matcher" under that stub and checks the real, unmodified exit code.
self_test() {
  local tmp
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN

  cat >"$tmp/cargo" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
scenario="${MUTATION_SAMPLE_SELFTEST_SCENARIO:?}"
for a in "$@"; do
  [[ "$a" == "--list" ]] && { echo "crates/fake.rs:1:1: replace f with g"; exit 0; }
done
out="" prev=""
for a in "$@"; do
  [[ "$prev" == "-o" ]] && out="$a"
  prev="$a"
done
[[ -n "$out" ]] || { echo "stub cargo: no -o dir found in: $*" >&2; exit 90; }
mkdir -p "$out/mutants.out/log"
case "$scenario" in
  clean)
    printf '{"outcomes":[{"scenario":"Baseline","summary":"Success"}],"missed":[],"caught":[1,2,3],"unviable":[],"timeout":[]}' \
      >"$out/mutants.out/outcomes.json"
    : >"$out/mutants.out/missed.txt"
    printf 'a\nb\nc\n' >"$out/mutants.out/caught.txt"
    ;;
  survivor)
    printf '{"outcomes":[{"scenario":"Baseline","summary":"Success"}],"missed":["m"],"caught":[1,2],"unviable":[],"timeout":[]}' \
      >"$out/mutants.out/outcomes.json"
    printf 'crates/fake.rs:1:1: replace + with - in fake\n' >"$out/mutants.out/missed.txt"
    printf 'a\nb\n' >"$out/mutants.out/caught.txt"
    ;;
  baseline_fail)
    printf '{"outcomes":[{"scenario":"Baseline","summary":"Failure"}],"missed":[],"caught":[],"unviable":[],"timeout":[]}' \
      >"$out/mutants.out/outcomes.json"
    printf 'thread panicked at fake.rs:1\n' >"$out/mutants.out/log/baseline.log"
    ;;
  *)
    echo "stub cargo: unknown scenario $scenario" >&2
    exit 91
    ;;
esac
exit 0
STUB
  chmod +x "$tmp/cargo"

  local ok=0 pair scenario expect got
  for pair in clean:0 survivor:1 baseline_fail:1; do
    scenario=${pair%%:*}
    expect=${pair##*:}
    got=0
    PATH="$tmp:$PATH" MUTATION_SAMPLE_SELFTEST_SCENARIO="$scenario" \
      POOT_BACKSTAGE_DIR="$tmp/backstage" "$0" matcher >"$tmp/out-$scenario.log" 2>&1 || got=$?
    if [[ "$got" == "$expect" ]]; then
      echo "self-test $scenario: ok (exit $got)"
    else
      echo "self-test $scenario: FAIL (expected exit $expect, got $got)" >&2
      sed 's/^/  | /' "$tmp/out-$scenario.log" >&2
      ok=1
    fi
  done
  # Seed drift must fail loudly: a missing file and a missing function are each refused, a live row passes.
  local seed_case
  for seed_case in "ok:crates/poot-quant/src/plan.rs:check_blocks:0" \
    "missing-file:crates/poot-quant/src/gone.rs:check_blocks:1" \
    "missing-fn:crates/poot-quant/src/plan.rs:in no_such_fn\$:1"; do
    IFS=: read -r scenario files seed expect <<<"$seed_case"
    got=0
    check_seeds "$scenario" "$files" "$seed" 2>"$tmp/seed-$scenario.log" || got=$?
    if [[ "$got" == "$expect" ]]; then
      echo "self-test seed-$scenario: ok (exit $got)"
    else
      echo "self-test seed-$scenario: FAIL (expected exit $expect, got $got)" >&2
      ok=1
    fi
  done
  return "$ok"
}

if [[ "$requested" == "self-test" ]]; then
  self_test
  exit $?
fi

if [[ "$requested" == "all" ]]; then
  overall=0
  for m in "${modules[@]}"; do
    IFS='|' read -r name package files seed <<<"$m"
    run_one "$name" "$package" "$files" "$seed" || overall=1
  done
  exit "$overall"
fi

for m in "${modules[@]}"; do
  IFS='|' read -r name package files seed <<<"$m"
  if [[ "$name" == "$requested" ]]; then
    run_one "$name" "$package" "$files" "$seed"
    exit $?
  fi
done

echo "unknown module: $requested" >&2
usage
