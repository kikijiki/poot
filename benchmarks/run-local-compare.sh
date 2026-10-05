#!/usr/bin/env bash
# Compare poot backend performance: ROCm vs wgpu (and optionally PTX) on the same model.
#
# Usage:
#   benchmarks/run-local-compare.sh --model-dir ~/models/qwen2.5-0.5b [OPTIONS]
#
# Options:
#   --model-dir DIR     Model directory (required)
#   --gen-tokens N      Tokens to generate (default: 128)
#   --warmup N          Warmup iterations (default: 2)
#   --iters N           Timed iterations (default: 5)
#   --prompt-file FILE  Prompt file (optional, uses default prompt)
#   --mode MODE         single or decode-curve (default: single)
#   --isl-list LIST     Comma-separated ISL list for decode-curve mode
#   --osl N             Output sequence length for decode-curve mode (default: 128)
#   --ptx               Also run PTX backend (requires NVIDIA GPU)
#   --skip-rocm         Skip ROCm backend
#   --skip-wgpu         Skip wgpu backend
#   --runner PATH       Path to poot-bench-runner binary
#   --json              Also emit raw JSON files
#
# Example:
#   benchmarks/run-local-compare.sh --model-dir ~/models/qwen2.5-0.5b --gen-tokens 128 --iters 5
#   benchmarks/run-local-compare.sh --model-dir ~/models/qwen2.5-0.5b --mode decode-curve --isl-list 128,512,2048 --osl 64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
MODEL_DIR=""
GEN_TOKENS=128
WARMUP=2
ITERS=5
PROMPT_FILE=""
MODE="single"
ISL_LIST=""
OSL=128
RUN_PTX=false
SKIP_ROCM=false
SKIP_WGPU=false
RUNNER=""
EMIT_JSON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --isl-list) ISL_LIST="$2"; shift 2 ;;
        --osl) OSL="$2"; shift 2 ;;
        --ptx) RUN_PTX=true; shift ;;
        --skip-rocm) SKIP_ROCM=true; shift ;;
        --skip-wgpu) SKIP_WGPU=true; shift ;;
        --runner) RUNNER="$2"; shift 2 ;;
        --json) EMIT_JSON=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL_DIR" ]]; then
    echo "Error: --model-dir is required" >&2
    exit 1
fi

# Find or build the runner. CARGO_TARGET_DIR may point outside the checkout (card 395), so the local
# target/ can be empty after a real build.
if [[ -z "$RUNNER" ]]; then
    RUNNER="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/poot-bench-runner"
    if [[ ! -x "$RUNNER" ]]; then
        echo "Building poot-bench-runner (release)..." >&2
        cargo build --release -p poot-bench-runner 2>&1 >&2
    fi
fi

if [[ ! -x "$RUNNER" ]]; then
    echo "Error: runner not found at $RUNNER" >&2
    exit 1
fi

# Build common args
COMMON_ARGS=(
    --model-dir "$MODEL_DIR"
    --gen-tokens "$GEN_TOKENS"
    --warmup "$WARMUP"
    --iters "$ITERS"
    --mode "$MODE"
)
if [[ -n "$PROMPT_FILE" ]]; then
    COMMON_ARGS+=(--prompt-file "$PROMPT_FILE")
fi
if [[ "$MODE" == "decode-curve" ]]; then
    if [[ -z "$ISL_LIST" ]]; then
        echo "Error: --isl-list required for decode-curve mode" >&2
        exit 1
    fi
    COMMON_ARGS+=(--isl-list "$ISL_LIST" --osl "$OSL")
fi

RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

run_backend() {
    local backend=$1
    local label=$2
    local outfile="$RESULTS_DIR/${backend}.json"

    echo "=== Running $label (--backend $backend) ===" >&2
    if "$RUNNER" "${COMMON_ARGS[@]}" --backend "$backend" --json > "$outfile" 2>"$RESULTS_DIR/${backend}.log"; then
        echo "  $label completed successfully" >&2
        if $EMIT_JSON; then
            cp "$outfile" "bench-${backend}.json"
            echo "  Raw JSON saved to bench-${backend}.json" >&2
        fi
        return 0
    else
        echo "  $label FAILED (exit code $?)" >&2
        cat "$RESULTS_DIR/${backend}.log" >&2
        return 1
    fi
}

# Run backends
declare -A RAN_BACKENDS

if $SKIP_ROCM; then
    echo "Skipping ROCm backend" >&2
else
    if run_backend "rocm" "ROCm"; then
        RAN_BACKENDS[rocm]=1
    fi
fi

if $SKIP_WGPU; then
    echo "Skipping wgpu backend" >&2
else
    if run_backend "wgpu" "wgpu/Vulkan"; then
        RAN_BACKENDS[wgpu]=1
    fi
fi

if $RUN_PTX; then
    if run_backend "ptx" "PTX/NVIDIA"; then
        RAN_BACKENDS[ptx]=1
    fi
fi

# Generate comparison table
echo ""
echo "# Backend Comparison"
echo ""
echo "Model: $(basename "$MODEL_DIR")"
echo "Mode: $MODE"
echo "Gen tokens: $GEN_TOKENS | Warmup: $WARMUP | Iters: $ITERS"
echo ""

if [[ ${#RAN_BACKENDS[@]} -eq 0 ]]; then
    echo "No backends completed successfully." >&2
    exit 1
fi

if [[ "$MODE" == "single" ]]; then
    echo "| Metric | $(printf '%s |' "${!RAN_BACKENDS[@]}")"
    echo "|--------|$(printf '--------|' "${!RAN_BACKENDS[@]}")"

    extract_field() {
        local backend=$1
        local field=$2
        python3 -c "
import json, sys
try:
    d = json.load(open('$RESULTS_DIR/${backend}.json'))
    v = d.get('$field', float('nan'))
    if isinstance(v, float):
        print(f'{v:.1f}')
    else:
        print(v)
except:
    print('N/A')
"
    }

    for metric in "ttft_ms:TTFT (ms)" "tpot_ms:TPOT (ms)" "e2e_ms:E2E (ms)" "decode_tok_s:Decode tok/s"; do
        field="${metric%%:*}"
        label="${metric#*:}"
        row="| $label |"
        for backend in "${!RAN_BACKENDS[@]}"; do
            val=$(extract_field "$backend" "$field")
            row+=" $val |"
        done
        echo "$row"
    done
else
    # Decode-curve mode: per-ISL comparison
    echo "## Decode Curve Results"
    echo ""
    echo "| ISL | $(printf '%s decode tok/s |' "${!RAN_BACKENDS[@]}")"
    echo "|-----|$(printf '------------------|' "${!RAN_BACKENDS[@]}")"

    # Extract ISL list from the first available result
    first_backend=$(echo "${!RAN_BACKENDS[@]}" | awk '{print $1}')
    isls=$(python3 -c "
import json
d = json.load(open('$RESULTS_DIR/${first_backend}.json'))
for p in d.get('curve', []):
    print(p.get('isl', ''))
" 2>/dev/null)

    for isl in $isls; do
        row="| $isl |"
        for backend in "${!RAN_BACKENDS[@]}"; do
            val=$(python3 -c "
import json
d = json.load(open('$RESULTS_DIR/${backend}.json'))
for p in d.get('curve', []):
    if p.get('isl') == $isl:
        iters = p.get('iters', [])
        if iters:
            # Compute mean decode tok/s from e2e_ms and osl
            e2e_vals = [i['e2e_ms'] for i in iters]
            ttft_vals = [i['ttft_ms'] for i in iters]
            avg_e2e = sum(e2e_vals) / len(e2e_vals)
            avg_ttft = sum(ttft_vals) / len(ttft_vals)
            decode_ms = avg_e2e - avg_ttft
            if decode_ms > 0:
                print(f'{($osl - 1) / (decode_ms / 1000):.1f}')
            else:
                print('N/A')
        else:
            print('N/A')
        break
else:
    print('N/A')
" 2>/dev/null)
            row+=" $val |"
        done
        echo "$row"
    done
fi

echo ""

# Print raw logs for reference
for backend in "${!RAN_BACKENDS[@]}"; do
    if [[ -f "$RESULTS_DIR/${backend}.log" ]] && [[ -s "$RESULTS_DIR/${backend}.log" ]]; then
        echo "<details><summary>$backend runner log</summary>"
        echo ""
        echo '```'
        cat "$RESULTS_DIR/${backend}.log"
        echo '```'
        echo ""
        echo "</details>"
        echo ""
    fi
done
