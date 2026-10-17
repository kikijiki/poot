#!/usr/bin/env bash
# Run a command with exactly one backend made mandatory.
#
#   with-required-backend.sh POOT_REQUIRE_{WGPU,VULKAN,ROCM,PTX,NPU,INTEL_NPU} command [args...]
#
# Two guarantees, both independent of the tests the command runs:
#   1. Only the selected POOT_REQUIRE_* variable is set; every other one is cleared.
#   2. A device for the selected backend must be visible before the command starts. A lane whose device is
#      missing fails here instead of passing on tests that skip themselves without one. Whether an open
#      device is then used by every test is the test's own contract.
set -euo pipefail

readonly backend_vars=(
  POOT_REQUIRE_WGPU
  POOT_REQUIRE_VULKAN
  POOT_REQUIRE_ROCM
  POOT_REQUIRE_PTX
  POOT_REQUIRE_NPU
  POOT_REQUIRE_INTEL_NPU
)

# Succeeds when a device for the backend is visible. Each probe reads the tool's own view of the machine, so
# a hidden device (ROCR_VISIBLE_DEVICES=, no Vulkan ICD, no NVIDIA driver) is a missing device. Output is
# captured before matching so `pipefail` cannot turn an early `grep -q` exit into a false negative.
probe_device() {
  local selected=$1 listing
  case "$selected" in
    POOT_REQUIRE_WGPU | POOT_REQUIRE_VULKAN)
      listing=$(vulkaninfo --summary 2>/dev/null || true)
      grep -qE 'deviceType += PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED|VIRTUAL)_GPU' <<<"$listing"
      ;;
    POOT_REQUIRE_ROCM)
      listing=$(rocminfo 2>/dev/null || true)
      grep -qE 'Device Type: +GPU' <<<"$listing"
      ;;
    POOT_REQUIRE_PTX)
      listing=$(nvidia-smi -L 2>/dev/null || true)
      grep -qE '^GPU [0-9]+:' <<<"$listing"
      ;;
    POOT_REQUIRE_NPU)
      listing=$(xrt-smi examine 2>/dev/null || true)
      grep -qE '^\|\[[0-9a-fA-F:.]+\] +\|NPU ' <<<"$listing"
      ;;
    POOT_REQUIRE_INTEL_NPU)
      # The Intel NPU is a compute accelerator the `intel_vpu` kernel driver owns: the module is loaded
      # when the device is probed. Whether the user-mode stack then opens it is the test's own contract.
      listing=$(lsmod 2>/dev/null || true)
      grep -qE '^intel_vpu ' <<<"$listing"
      ;;
  esac
}

run_with_only() {
  local selected=$1
  shift
  exec env \
    -u POOT_REQUIRE_WGPU \
    -u POOT_REQUIRE_VULKAN \
    -u POOT_REQUIRE_ROCM \
    -u POOT_REQUIRE_PTX \
    -u POOT_REQUIRE_NPU \
    -u POOT_REQUIRE_INTEL_NPU \
    "$selected=1" \
    "$@"
}

# Stub tools that report a device (`present`) or only a CPU / nothing (`absent`), so the self-test checks the
# probes without depending on the machine's hardware.
make_stub_tools() {
  local dir=$1 state=$2
  mkdir -p "$dir"
  local vulkan_type=PHYSICAL_DEVICE_TYPE_CPU rocm_type=CPU nvidia_line='' nvidia_status=9
  if [[ "$state" == "present" ]]; then
    vulkan_type=PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    rocm_type=GPU
    nvidia_line='GPU 0: stub (UUID: stub)'
    nvidia_status=0
  fi
  printf '#!/bin/sh\necho "        deviceType         = %s"\n' "$vulkan_type" >"$dir/vulkaninfo"
  printf '#!/bin/sh\necho "  Device Type:             %s"\n' "$rocm_type" >"$dir/rocminfo"
  printf '#!/bin/sh\n[ -n "%s" ] && echo "%s"\nexit %s\n' "$nvidia_line" "$nvidia_line" "$nvidia_status" >"$dir/nvidia-smi"
  local xrt_line=''
  [[ "$state" == "present" ]] && xrt_line='|[0000:c7:00.1]  |NPU Strix Halo  |aie2p         |6x8       |'
  printf '#!/bin/sh\n[ -n "%s" ] && echo "%s"\nexit 0\n' "$xrt_line" "$xrt_line" >"$dir/xrt-smi"
  local lsmod_line=''
  [[ "$state" == "present" ]] && lsmod_line='intel_vpu             401408  0'
  printf '#!/bin/sh\n[ -n "%s" ] && echo "%s"\nexit 0\n' "$lsmod_line" "$lsmod_line" >"$dir/lsmod"
  chmod +x "$dir/lsmod"
  chmod +x "$dir/vulkaninfo" "$dir/rocminfo" "$dir/nvidia-smi" "$dir/xrt-smi"
}

self_test() {
  tmp=$(mktemp -d)
  trap 'rm -rf -- "$tmp"' EXIT
  make_stub_tools "$tmp/present" present
  make_stub_tools "$tmp/absent" absent

  for selected in "${backend_vars[@]}"; do
    # The lane sets itself and clears every other requirement variable.
    local inherited=(env)
    for variable in "${backend_vars[@]}"; do
      inherited+=("$variable=inherited")
    done
    local clean_env
    clean_env=$(PATH="$tmp/present:$PATH" "${inherited[@]}" "$0" "$selected" env)
    for variable in "${backend_vars[@]}"; do
      local value
      value=$(sed -n "s/^$variable=//p" <<<"$clean_env")
      if [[ "$variable" == "$selected" ]]; then
        [[ "$value" == "1" ]] || {
          echo "$selected lane did not set itself to 1" >&2
          exit 1
        }
      elif [[ -n "$value" ]]; then
        echo "$selected lane inherited unrelated $variable=$value" >&2
        exit 1
      fi
    done

    # A missing device fails the lane and the command never starts.
    local marker="$tmp/ran-$selected"
    if PATH="$tmp/absent:$PATH" "$0" "$selected" touch "$marker" 2>"$tmp/stderr"; then
      echo "$selected lane passed with no device visible" >&2
      exit 1
    fi
    [[ ! -e "$marker" ]] || {
      echo "$selected lane ran its command with no device visible" >&2
      exit 1
    }
    grep -q "no $selected device visible" "$tmp/stderr" || {
      echo "$selected lane did not name the missing device" >&2
      cat "$tmp/stderr" >&2
      exit 1
    }
    PATH="$tmp/present:$PATH" "$0" "$selected" touch "$marker"
    [[ -e "$marker" ]] || {
      echo "$selected lane did not run its command with a device visible" >&2
      exit 1
    }
  done
  echo "backend requirement environment isolation and device probes: pass"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

selected=${1:-}
shift || true
case "$selected" in
  POOT_REQUIRE_WGPU | POOT_REQUIRE_VULKAN | POOT_REQUIRE_ROCM | POOT_REQUIRE_PTX | POOT_REQUIRE_NPU | POOT_REQUIRE_INTEL_NPU) ;;
  *)
    echo "usage: $0 POOT_REQUIRE_{WGPU,VULKAN,ROCM,PTX,NPU,INTEL_NPU} command [args...]" >&2
    exit 2
    ;;
esac
if [[ "$#" -eq 0 ]]; then
  echo "missing backend command" >&2
  exit 2
fi
if ! probe_device "$selected"; then
  echo "no $selected device visible: the lane needs one and would otherwise pass on skipped tests" >&2
  exit 1
fi

run_with_only "$selected" "$@"
