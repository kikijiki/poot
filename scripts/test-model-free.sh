#!/usr/bin/env bash
set -euo pipefail

mode=${1:-run}
case "$mode" in
  check | run | run-real-dims | regen | self-test) ;;
  *)
    echo "usage: $0 [check|run|run-real-dims|regen|self-test] [nextest args for run-real-dims]" >&2
    exit 2
    ;;
esac
shift || true

# The model-free gate covers every workspace member. `gate_policy` is the only place a package is placed in
# or out of it, and `check` fails when a workspace member is missing from it or a row names a package that is
# gone. Fields: package | policy | reason (excluded policy only).
#
#   host      every active test needs no model and no device, so the inventory derives its class: model-free.
#   reviewed  the package holds tests that need a device or a checkpoint next to host-only ones, so each test's
#             class (model-free or checkpoint-device) is chosen by hand in the inventory and a new test has no
#             default class. Nothing trusts that choice for devices: the lane runs with every device required
#             and hidden, so a device test classed model-free fails the lane (see `lane_environment`).
#   excluded  not in the default lane, for the reason given: the package's tests dispatch on a device and run
#             in the `just test-device-*` lanes, or it holds no test and would only change feature unification.
#
# `real_dims_pattern` is a naming contract, not detection: a test that builds real model dimensions is named
# `*_real_dims`, and a test so named is class real-dims, never model-free; it runs in the release-profile lane
# (`just test-real-dims`). A real-dimension test that does not carry the suffix lands model-free.
readonly gate_policy=(
  "poot-bench-runner|excluded|no tests; enables poot-llm rocm, which would unify into the lane builds"
  "poot-codegen|host|"
  "poot-eval|host|"
  "poot-executor|host|"
  "poot-executor-parity|host|"
  "poot-gpu|excluded|every test dispatches on a wgpu device"
  "poot-graph-ir|host|"
  "poot-graph-plan|host|"
  "poot-kernel-attr|host|"
  "poot-kernel-ir|host|"
  "poot-kernel-intrinsics|host|"
  "poot-kernelgen|reviewed|"
  "poot-llm|reviewed|"
  "poot-load|reviewed|"
  "poot-models|reviewed|"
  "poot-orchestrator|host|"
  "poot-profile|host|"
  "poot-ptx-check|host|"
  "poot-ptx-gpu|reviewed|"
  "poot-ptx-runtime|reviewed|"
  "poot-quant|host|"
  "poot-rocm-check|host|"
  "poot-rocm-gpu|excluded|every test dispatches on a ROCm device"
  "poot-rocm-runtime|reviewed|"
  "poot-runtime|reviewed|"
  "poot-runtime-common|host|"
  "poot-serve|reviewed|"
  "poot-target|host|"
  "poot-tensor|host|"
  "poot-test-util|host|"
  "poot-vulkan-device|excluded|every test dispatches on a raw Vulkan device"
  "poot-vulkan-runtime|reviewed|"
  "pootc|reviewed|"
)
readonly real_dims_pattern='_real_dims$'

# The TSV is generated: `regen` writes it from the test binaries, and `check` regenerates it in memory and
# fails on any difference. Only `reviewed` packages carry hand-chosen classes, kept across a regen.
readonly checked_inventory=scripts/test-model-free-inventory.tsv
readonly expected_header=$'package\ttarget\tclass\ttest'

tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
readonly discovered_inventory="$tmp_dir/discovered.tsv"
readonly discovered_targets="$tmp_dir/targets.tsv"
readonly policy_table="$tmp_dir/policy.tsv"

for row in "${gate_policy[@]}"; do
  IFS='|' read -r package policy reason <<<"$row"
  printf '%s\t%s\t%s\n' "$package" "$policy" "$reason"
done | LC_ALL=C sort >"$policy_table"

lane_packages() {
  awk -F '\t' '$2 != "excluded" { print $1 }' "$policy_table"
}

lane_package_args() {
  local package
  while IFS= read -r package; do
    printf '%s\0%s\0' -p "$package"
  done < <(lane_packages)
}

# Every workspace member needs a policy row, and every policy row a workspace member.
check_members() {
  local policy_packages=$1
  local members="$tmp_dir/members"
  local missing="$tmp_dir/members-missing"
  local stale="$tmp_dir/members-stale"
  cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | LC_ALL=C sort -u >"$members"
  LC_ALL=C comm -23 "$members" "$policy_packages" >"$missing"
  LC_ALL=C comm -13 "$members" "$policy_packages" >"$stale"
  if [[ -s "$missing" ]]; then
    echo "workspace member missing from gate_policy in $0: every package is gated, or excluded with a reason:" >&2
    sed 's/^/  + /' "$missing" >&2
  fi
  if [[ -s "$stale" ]]; then
    echo "gate_policy row names no workspace member:" >&2
    sed 's/^/  - /' "$stale" >&2
  fi
  [[ ! -s "$missing" && ! -s "$stale" ]]
}

discover_tests() {
  local json="$tmp_dir/list.json"
  local -a package_args=()
  while IFS= read -r -d '' arg; do
    package_args+=("$arg")
  done < <(lane_package_args)
  cargo nextest list "${package_args[@]}" --message-format json >"$json"
  jq -r '
    .["rust-suites"] | to_entries[]
    | [.value["package-name"], .key] | @tsv
  ' "$json" | LC_ALL=C sort -u >"$discovered_targets"
  jq -r '
    .["rust-suites"] | to_entries[]
    | .value["package-name"] as $package | .key as $target
    | .value.testcases | to_entries[]
    | [$package, $target, (if .value.ignored then "ignored" else "active" end), .key]
    | @tsv
  ' "$json" | LC_ALL=C sort >"$discovered_inventory"
}

# The inventory the test binaries and the policy imply. A `reviewed` package keeps the class already in
# `previous`; a test with none is `unclassified`, which validation rejects until someone reviews it.
derive_inventory() {
  local previous=$1
  {
    printf '%s\n' "$expected_header"
    awk -F '\t' -v OFS='\t' -v real_dims="$real_dims_pattern" \
      -v policy_file="$policy_table" -v previous_file="$previous" '
      BEGIN {
        while ((getline line < policy_file) > 0) {
          split(line, f, "\t")
          policy[f[1]] = f[2]
        }
        while ((getline line < previous_file) > 0) {
          split(line, f, "\t")
          if (f[3] == "model-free" || f[3] == "checkpoint-device") kept[f[1] "\t" f[2] "\t" f[4]] = f[3]
        }
      }
      $3 != "active" { next }
      {
        key = $1 "\t" $2 "\t" $4
        if ($4 ~ real_dims) class = "real-dims"
        else if (policy[$1] == "host") class = "model-free"
        else if (key in kept) class = kept[key]
        else class = "unclassified"
        print $1, $2, class, $4
      }
    ' "$discovered_inventory" | LC_ALL=C sort
  }
}

validate_inventory() {
  local inventory=$1
  local header
  header=$(head -n 1 "$inventory" 2>/dev/null || true)
  if [[ "$header" != "$expected_header" ]]; then
    echo "inventory schema mismatch: expected exact header: $expected_header" >&2
    echo "inventory schema mismatch: got: ${header:-<empty>}" >&2
    return 1
  fi

  local schema_errors="$tmp_dir/schema-errors"
  awk -F '\t' -v targets="$discovered_targets" -v policy_file="$policy_table" '
    BEGIN {
      while ((getline line < targets) > 0) allowed_target[line] = 1
      close(targets)
      while ((getline line < policy_file) > 0) {
        split(line, f, "\t")
        if (f[2] != "excluded") allowed_package[f[1]] = 1
      }
    }
    NR == 1 { next }
    {
      row = $0
      if (NF != 4) {
        print "inventory schema mismatch at line " NR ": expected 4 tab-separated columns: " row
        bad = 1
        next
      }
      if (!($1 in allowed_package)) {
        print "inventory package mismatch at line " NR ": " $1
        bad = 1
      }
      target_key = $1 "\t" $2
      if (!(target_key in allowed_target)) {
        print "inventory target mismatch at line " NR ": " target_key
        bad = 1
      }
      if ($3 != "model-free" && $3 != "checkpoint-device" && $3 != "real-dims") {
        print "inventory class mismatch at line " NR ": " $3 " for " $1 "\t" $2 "\t" $4
        bad = 1
      }
      if ($4 == "" || $4 !~ /^[A-Za-z0-9_:.-]+$/) {
        print "inventory test mismatch at line " NR ": " $4
        bad = 1
      }
    }
    END { exit bad }
  ' "$inventory" >"$schema_errors" || {
    cat "$schema_errors" >&2
    return 1
  }

  local rows="$tmp_dir/rows"
  local sorted_rows="$tmp_dir/sorted-rows"
  tail -n +2 "$inventory" >"$rows"
  LC_ALL=C sort "$rows" >"$sorted_rows"
  if ! cmp -s "$rows" "$sorted_rows"; then
    echo "inventory order mismatch: rows must be in raw LC_ALL=C canonical sorted order" >&2
    diff -u --label checked-order --label canonical-order "$rows" "$sorted_rows" >&2 || true
    return 1
  fi

  local duplicate_rows="$tmp_dir/duplicate-rows"
  LC_ALL=C sort "$rows" | uniq -d >"$duplicate_rows"
  if [[ -s "$duplicate_rows" ]]; then
    echo "inventory duplicate row mismatch:" >&2
    cat "$duplicate_rows" >&2
    return 1
  fi

  local identities="$tmp_dir/identities"
  local duplicate_identities="$tmp_dir/duplicate-identities"
  awk -F '\t' '{ print $1 "\t" $2 "\t" $4 }' "$rows" | LC_ALL=C sort >"$identities"
  uniq -d "$identities" >"$duplicate_identities"
  if [[ -s "$duplicate_identities" ]]; then
    echo "inventory duplicate test identity mismatch:" >&2
    cat "$duplicate_identities" >&2
    return 1
  fi
}

compare_discovery() {
  local inventory=$1
  local expected="$tmp_dir/expected-identities"
  local actual="$tmp_dir/actual-identities"
  local missing="$tmp_dir/missing-identities"
  local stale="$tmp_dir/stale-identities"
  tail -n +2 "$inventory" \
    | awk -F '\t' '{ print $1 "\t" $2 "\t" $4 }' \
    | LC_ALL=C sort >"$expected"
  awk -F '\t' '$3 == "active" { print $1 "\t" $2 "\t" $4 }' \
    "$discovered_inventory" | LC_ALL=C sort >"$actual"
  if cmp -s "$expected" "$actual"; then
    return 0
  fi
  diff -u --label checked-inventory --label discovered-tests "$expected" "$actual" || true
  LC_ALL=C comm -13 "$expected" "$actual" >"$missing"
  LC_ALL=C comm -23 "$expected" "$actual" >"$stale"
  {
    echo "inventory identity mismatch: every discovered active test needs exactly one row."
    if [[ -s "$missing" ]]; then
      echo "missing rows (discovered, not in $checked_inventory): $(wc -l <"$missing")"
      sed 's/^/  + /' "$missing"
    fi
    if [[ -s "$stale" ]]; then
      echo "stale rows (in $checked_inventory, not discovered): $(wc -l <"$stale")"
      sed 's/^/  - /' "$stale"
    fi
    echo "regenerate: scripts/test-model-free.sh regen; a test in a reviewed package then needs its class set by hand"
    echo "re-check: scripts/test-model-free.sh check"
  } >&2
  return 1
}

# The classes the policy and the real-dims rule derive must equal the checked ones. A `reviewed` row keeps
# whatever it says, so only the derived classes and `unclassified` tests can differ.
compare_classes() {
  local inventory=$1
  local derived="$tmp_dir/derived-inventory"
  local changed="$tmp_dir/changed-classes"
  derive_inventory "$inventory" >"$derived"
  if cmp -s "$inventory" "$derived"; then
    return 0
  fi
  diff -u --label checked-inventory --label derived-inventory "$inventory" "$derived" || true
  LC_ALL=C comm -13 <(LC_ALL=C sort "$inventory") <(LC_ALL=C sort "$derived") >"$changed"
  {
    echo "inventory class mismatch: the class of a test must be the one its package policy or the real-dims rule derives."
    echo "  a test that builds real dimensions is real-dims and stays out of the default lane;"
    echo "  an unclassified test in a reviewed package needs model-free or checkpoint-device."
    sed 's/^/  want: /' "$changed"
  } >&2
  return 1
}

print_summary() {
  local inventory=$1
  local package
  while IFS= read -r package; do
    local free optional slow ignored targets
    free=$(awk -F '\t' -v p="$package" '$1 == p && $3 == "model-free" { n++ } END { print n + 0 }' "$inventory")
    optional=$(awk -F '\t' -v p="$package" '$1 == p && $3 == "checkpoint-device" { n++ } END { print n + 0 }' "$inventory")
    slow=$(awk -F '\t' -v p="$package" '$1 == p && $3 == "real-dims" { n++ } END { print n + 0 }' "$inventory")
    ignored=$(awk -F '\t' -v p="$package" '$1 == p && $3 == "ignored" { n++ } END { print n + 0 }' "$discovered_inventory")
    targets=$(tail -n +2 "$inventory" | awk -F '\t' -v p="$package" '$1 == p { print $2 }' | LC_ALL=C sort -u | wc -l)
    echo "model-free inventory: $package: $free selected, $optional checkpoint/device optional, $slow real-dims, $ignored ignored, $targets targets"
  done < <(lane_packages)
  awk -F '\t' '$2 == "excluded" { print "model-free inventory: " $1 ": excluded (" $3 ")" }' "$policy_table"
}

# The default lane is hermetic, and a device is an error in it: every backend is hidden and every backend is
# required (each POOT_REQUIRE_<BACKEND> that `DeviceBackend::required` reads), so a test that reaches for a
# device panics instead of skipping. That is what keeps the hand-chosen classes of the `reviewed` packages
# honest: a device test classed model-free fails the lane. It does not depend on the machine's hardware, and
# device coverage is the `test-device-*` lanes' job, where a missing device fails. A test whose guard does not
# go through `DeviceBackend` still skips. A checkpoint is an error in it too: the models directory is unset and
# required (`poot_test_util::model_path` reads POOT_MODELS_DIR and POOT_REQUIRE_MODELS), so a checkpoint test
# classed model-free panics naming the variable instead of skipping. The lane also runs with a private TMPDIR so a
# fixture a test leaves behind cannot reach a concurrent lane. POOT_MODEL_FREE_LANE=1 marks the lane so that
# `the_model_free_lane_requires_every_backend` (poot-runtime-common) can check the list of
# POOT_REQUIRE_<BACKEND> variables here against `DeviceBackend::ALL`; the self-test checks the marker is set.
lane_environment() {
  mkdir -p "$tmp_dir/tmp"
  env -u POOT_MODELS_DIR TMPDIR="$tmp_dir/tmp" POOT_MODEL_FREE_LANE=1 POOT_REQUIRE_MODELS=1 \
    POOT_REQUIRE_WGPU=1 POOT_REQUIRE_VULKAN=1 POOT_REQUIRE_ROCM=1 POOT_REQUIRE_PTX=1 \
    ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
    VK_ICD_FILENAMES=/nonexistent VK_DRIVER_FILES=/nonexistent "$@"
}

# The default lane: every lane package's active tests except the rows that are not model-free. The inventory is
# complete (`check` ran first), so nothing unlisted runs, and a lane that selects no test fails.
run_lane_tests() {
  local inventory=$1
  local only=${2:-}
  local exclusions
  exclusions=$(awk -F '\t' '
    NR > 1 && $3 != "model-free" {
      printf "%s(binary_id(=%s) & test(=%s))", sep, $2, $4
      sep = " | "
    }
  ' "$inventory")
  local -a package_args=() filter=()
  while IFS= read -r -d '' arg; do
    package_args+=("$arg")
  done < <(lane_package_args)
  [[ -z "$exclusions" ]] || filter=(-E "not ($exclusions)")
  # `only` narrows the run to the tests the caller names, for the self-test of the gate itself.
  if [[ -n "$only" ]]; then
    filter=(-E "(${filter[1]:-all()}) & ($only)")
  fi
  lane_environment cargo nextest run --no-tests=fail --no-fail-fast "${package_args[@]}" "${filter[@]}"
}

run_model_free() {
  local inventory=$1
  local -a package_args=()
  while IFS= read -r -d '' arg; do
    package_args+=("$arg")
  done < <(lane_package_args)
  run_lane_tests "$inventory"
  lane_environment cargo test --doc "${package_args[@]}"
}

# The real-dims rows, only, with the caller's nextest arguments (the recipe passes --release).
run_real_dims() {
  local inventory=$1
  shift
  local selected
  selected=$(awk -F '\t' '
    NR > 1 && $3 == "real-dims" {
      printf "%s(binary_id(=%s) & test(=%s))", sep, $2, $4
      sep = " | "
    }
  ' "$inventory")
  if [[ -z "$selected" ]]; then
    echo "no real-dims tests in $inventory" >&2
    return 1
  fi
  local -a package_args=()
  while IFS= read -r -d '' arg; do
    package_args+=("$arg")
  done < <(lane_package_args)
  cargo nextest run --no-tests=fail --no-fail-fast "${package_args[@]}" -E "$selected" "$@"
}

expect_mutation_failure() {
  local label=$1
  local needle=$2
  local candidate=$3
  local output rc
  set +e
  output=$(validate_inventory "$candidate" 2>&1 && compare_discovery "$candidate" 2>&1 && compare_classes "$candidate" 2>&1)
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "inventory mutation unexpectedly passed: $label" >&2
    return 1
  fi
  if [[ "$output" != *"$needle"* ]]; then
    echo "inventory mutation did not name its mismatch: $label (wanted $needle)" >&2
    echo "$output" >&2
    return 1
  fi
  echo "inventory mutation rejected: $label ($needle)"
}

sorted_with_header() {
  IFS= read -r header
  printf '%s\n' "$header"
  LC_ALL=C sort
}

run_self_test() {
  local host_row real_dims_row second
  host_row=$(awk -F '\t' -v policy_file="$policy_table" '
    BEGIN {
      while ((getline line < policy_file) > 0) { split(line, f, "\t"); policy[f[1]] = f[2] }
    }
    NR > 1 && $3 == "model-free" && policy[$1] == "host" { print; exit }
  ' "$checked_inventory")
  real_dims_row=$(awk -F '\t' 'NR > 1 && $3 == "real-dims" { print; exit }' "$checked_inventory")
  second=$(sed -n '3p' "$checked_inventory")
  local host_test=${host_row##*$'\t'}
  local real_dims_test=${real_dims_row##*$'\t'}
  if [[ -z "$host_row" || -z "$real_dims_row" ]]; then
    echo "self-test needs a host and a real-dims row in $checked_inventory" >&2
    return 1
  fi

  local add="$tmp_dir/mutation-add.tsv"
  local remove="$tmp_dir/mutation-remove.tsv"
  local duplicate="$tmp_dir/mutation-duplicate.tsv"
  local rename="$tmp_dir/mutation-rename.tsv"
  local reclassify_host="$tmp_dir/mutation-reclassify-host.tsv"
  local real_dims_as_model_free="$tmp_dir/mutation-real-dims.tsv"
  local reorder="$tmp_dir/mutation-reorder.tsv"
  local first_row first_test
  first_row=$(sed -n '2p' "$checked_inventory")
  first_test=${first_row##*$'\t'}

  { cat "$checked_inventory"; printf 'poot-load\tpoot-load\tmodel-free\tmutation_added_test\n'; } | sorted_with_header >"$add"
  awk -v drop="$host_row" 'NR == 1 || $0 != drop' "$checked_inventory" >"$remove"
  { cat "$checked_inventory"; printf '%s\n' "$host_row"; } | sorted_with_header >"$duplicate"
  sed "s/${host_test}/${host_test}_renamed/" "$checked_inventory" | sorted_with_header >"$rename"
  awk -F '\t' -v OFS='\t' -v row="$host_row" '$0 == row { $3 = "checkpoint-device" } { print }' "$checked_inventory" \
    | sorted_with_header >"$reclassify_host"
  awk -F '\t' -v OFS='\t' -v row="$real_dims_row" '$0 == row { $3 = "model-free" } { print }' "$checked_inventory" \
    | sorted_with_header >"$real_dims_as_model_free"
  { head -n 1 "$checked_inventory"; printf '%s\n%s\n' "$second" "$first_row"; tail -n +4 "$checked_inventory"; } >"$reorder"

  # label | expected named mismatch | candidate
  while IFS='|' read -r label needle candidate; do
    expect_mutation_failure "$label" "$needle" "$candidate"
  done <<EOF
add|mutation_added_test|$add
remove|$host_test|$remove
duplicate|$host_test|$duplicate
rename|${host_test}_renamed|$rename
reclassify a derived class|$host_test|$reclassify_host
real-dims test listed model-free|$real_dims_test|$real_dims_as_model_free
reorder|$first_test|$reorder
EOF

  # A workspace member the policy leaves out, and a policy row for no member, are both named.
  local without_one="$tmp_dir/policy-without-one"
  local with_ghost="$tmp_dir/policy-with-ghost"
  local dropped
  dropped=$(awk -F '\t' 'NR == 1 { print $1 }' "$policy_table")
  awk -F '\t' 'NR > 1 { print $1 }' "$policy_table" >"$without_one"
  { awk -F '\t' '{ print $1 }' "$policy_table"; echo poot-no-such-package; } | LC_ALL=C sort >"$with_ghost"
  local output
  if output=$(check_members "$without_one" 2>&1); then
    echo "member check accepted a policy without $dropped" >&2
    return 1
  fi
  [[ "$output" == *"+ $dropped"* ]] || {
    echo "member check did not name the missing member $dropped" >&2
    echo "$output" >&2
    return 1
  }
  echo "member mutation rejected: $dropped missing from the policy"
  if output=$(check_members "$with_ghost" 2>&1); then
    echo "member check accepted a policy row for no member" >&2
    return 1
  fi
  [[ "$output" == *"- poot-no-such-package"* ]] || {
    echo "member check did not name the stale row" >&2
    echo "$output" >&2
    return 1
  }
  echo "member mutation rejected: stale policy row poot-no-such-package"

  # The real gate, not the checker: a test that needs a device, moved into the model-free class, must fail the
  # lane. The candidate inventory drives the same lane invocation `run` uses, narrowed to that one test.
  local gate_target=poot-runtime::run_add gate_test=add_runs_on_gpu
  local gate_row
  gate_row=$(awk -F '\t' -v target="$gate_target" -v test="$gate_test" '
    NR > 1 && $2 == target && $4 == test && $3 == "checkpoint-device" { print; exit }
  ' "$checked_inventory")
  if [[ -z "$gate_row" ]]; then
    echo "self-test needs the checkpoint-device row for $gate_target $gate_test" >&2
    return 1
  fi
  local device_as_model_free="$tmp_dir/mutation-device-as-model-free.tsv"
  awk -F '\t' -v OFS='\t' -v row="$gate_row" '$0 == row { $3 = "model-free" } { print }' "$checked_inventory" \
    | sorted_with_header >"$device_as_model_free"
  local gate_output
  if gate_output=$(run_lane_tests "$device_as_model_free" "binary_id(=$gate_target) & test(=$gate_test)" 2>&1); then
    echo "lane passed with a device test classed model-free: $gate_target $gate_test" >&2
    return 1
  fi
  [[ "$gate_output" == *"$gate_test"* ]] || {
    echo "lane failed without naming the device test $gate_test" >&2
    echo "$gate_output" >&2
    return 1
  }
  echo "gate mutation rejected: device test $gate_test classed model-free fails the lane"

  local lane_env
  lane_env=$(lane_environment env)
  grep -qx 'POOT_MODEL_FREE_LANE=1' <<<"$lane_env" || {
    echo "the lane does not set POOT_MODEL_FREE_LANE=1, so the backend-list drift test would never run" >&2
    return 1
  }
  echo "lane marker set: POOT_MODEL_FREE_LANE=1"

  assert_gpu_group_selection
}

# The nextest `gpu` group selects poot-llm and poot-serve tests by binary name or by test name, and both
# filters must list the same words. Three fixture tests of poot-llm (host-only, no body) sit on each side of
# that: one whose only device word is `hardware` in its binary name, one whose only device word is `hardware` in
# its test name, and a control with neither. The group's real membership is read from nextest, so dropping a
# word from either filter fails here.
assert_gpu_group_selection() {
  local by_binary=only_the_binary_name_places_this_test_in_the_group
  local by_test=a_hardware_named_test_joins_the_group
  local control=a_host_only_test_stays_out_of_the_group
  local groups
  groups=$(cargo nextest show-config test-groups -p poot-llm 2>&1) || {
    echo "cargo nextest show-config test-groups failed:" >&2
    echo "$groups" >&2
    return 1
  }
  local member
  for member in "poot-llm::nextest_group_hardware_binary:" "$by_binary" "$by_test"; do
    grep -qF -- "$member" <<<"$groups" || {
      echo "gpu group does not select $member (a filter lost a word: binary and test regexes in .config/nextest.toml)" >&2
      return 1
    }
  done
  if grep -qF -- "$control" <<<"$groups"; then
    echo "gpu group selects the control test $control: the filters match too much" >&2
    return 1
  fi
  echo "gpu group selects the hardware-named binary and the hardware-named test, and not the control"
}

check_gate() {
  awk -F '\t' '{ print $1 }' "$policy_table" >"$tmp_dir/policy-packages"
  check_members "$tmp_dir/policy-packages"
  discover_tests
  validate_inventory "$checked_inventory"
  compare_discovery "$checked_inventory"
  compare_classes "$checked_inventory"
  print_summary "$checked_inventory"
}

if [[ "$mode" == "regen" ]]; then
  awk -F '\t' '{ print $1 }' "$policy_table" >"$tmp_dir/policy-packages"
  check_members "$tmp_dir/policy-packages"
  discover_tests
  derive_inventory "$checked_inventory" >"$tmp_dir/regenerated.tsv"
  cp "$tmp_dir/regenerated.tsv" "$checked_inventory"
  echo "regenerated $checked_inventory: $(($(wc -l <"$checked_inventory") - 1)) rows"
  if grep -n $'\tunclassified\t' "$checked_inventory"; then
    echo "set the class of each unclassified test above (model-free or checkpoint-device) by reading it" >&2
    exit 1
  fi
  exit 0
fi

check_gate

if [[ "$mode" == "self-test" ]]; then
  run_self_test
elif [[ "$mode" == "run" ]]; then
  run_model_free "$checked_inventory"
elif [[ "$mode" == "run-real-dims" ]]; then
  run_real_dims "$checked_inventory" "$@"
fi
