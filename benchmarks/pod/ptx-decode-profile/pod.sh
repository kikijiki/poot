#!/usr/bin/env bash
# POOT-1319: per-dispatch decode profile of poot PTX on ONE NVIDIA card (runs ON the pod, as root).
#
# For each model: a clean decode-curve (TPOT) and a --profile decode-curve (device time per dispatch family, from the
# CUDA events the timed device records in the captured graph) at ISL 128 and 8192. Nothing is read back from stdout;
# everything lands under $OUT. A failing phase is a recorded row, not an abort. Models download on the pod.
set -u
OUT="${OUT:-/root/results}"; WORK="${WORK:-/root/work}"; BIN="${BIN:-/root/bin}"
ISLS="${ISLS:-128,8192}"; OSL="${OSL:-64}"
mkdir -p "$OUT/logs" "$WORK"
export PATH="/root/.local/bin:/usr/local/cuda/bin:$PATH" POOT_KERNEL_CACHE_DIR="$WORK/poot-kernel-cache" HF_HUB_DISABLE_TELEMETRY=1 HF_XET_HIGH_PERFORMANCE=1
START=$(date +%s)
say() { printf '[%s +%ss] %s\n' "$(date +%T)" "$(( $(date +%s) - START ))" "$*" | tee -a "$OUT/progress.log"; }
HF=$(command -v hf || command -v huggingface-cli)

{ nvidia-smi; nvidia-smi --query-gpu=name,clocks.max.memory,clocks.max.sm,memory.total --format=csv; nvcc --version;
  for t in nsys ncu nsys-ui; do printf '%s: %s\n' "$t" "$(command -v $t || echo none)"; done; } >"$OUT/env.txt" 2>&1
# the GPU must be alive before anything builds or runs
nvidia-smi -L >"$OUT/gpu-alive.txt" 2>&1 || { say "GPU dead: nvidia-smi -L failed"; touch "$OUT/SWEEP_DONE"; exit 1; }

run() { # NAME TIMEOUT -- CMD...
  local name=$1 tmo=$2; shift 3
  printf '%s\t%s\n' "$name" "$*" >>"$OUT/commands.tsv"
  say "BEGIN $name"
  timeout -k 30 "$tmo" "$@" >"$OUT/logs/$name.out" 2>"$OUT/logs/$name.err"; local rc=$?
  say "END   $name rc=$rc"; printf '%s\t%s\n' "$name" "$rc" >>"$OUT/phases.tsv"
}
# CELLS names the cells to run (default: all); --precision is what the store holds (the runner refuses a mismatch).
cell() { # NAME PRECISION MODELARGS... : the clean curve, then the profiled one
  local name=$1 precision=$2; shift 2
  case " ${CELLS:-$name} " in *" $name "*) ;; *) return 0 ;; esac
  local base=("$BIN/poot-bench-runner" "$@" --backend ptx --precision "$precision" --mode decode-curve --isl-list "$ISLS" --osl "$OSL" --warmup 1 --iters "${ITERS:-3}" --json)
  case " ${KINDS:-curve profile} " in *" curve "*) run "$name.curve" 3000 -- "${base[@]}" ;; esac
  case " ${KINDS:-curve profile} " in *" profile "*) run "$name.profile" 3000 -- "${base[@]}" --profile ;; esac
}

# downloads in the background while the small models run
( "$HF" download Qwen/Qwen3-4B --local-dir "$WORK/qwen3-4b" >"$OUT/logs/dl-4b.out" 2>&1
  "$HF" download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir "$WORK/qwen3-4b-gguf" >"$OUT/logs/dl-4bq.out" 2>&1
  touch "$WORK/dl-small.done"
  "$HF" download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir "$WORK/q27" >"$OUT/logs/dl-27q.out" 2>&1
  touch "$WORK/dl-27q.done"
  "$HF" download Qwen/Qwen3.8-27B-FP8 --local-dir "$WORK/fp8" >"$OUT/logs/dl-fp8.out" 2>&1
  touch "$WORK/dl-fp8.done" ) &
waitfor() {
  while [ ! -e "$WORK/$1" ]; do
    sleep 10
    if [ $(( $(date +%s) - START )) -gt 5400 ]; then return 1; fi
  done
}

waitfor dl-small.done && {
  cell qwen3-4b bf16 --model-dir "$WORK/qwen3-4b"
  cell qwen3-4b-q4km q4_k --gguf "$WORK/qwen3-4b-gguf/Qwen3-4B-Q4_K_M.gguf"
}
waitfor dl-27q.done && cell q27 q5_k --gguf "$WORK/q27/Qwen3.8-27B-UD-Q4_K_XL.gguf"
waitfor dl-fp8.done && cell fp8 e4m3block128 --model-dir "$WORK/fp8"
nvidia-smi --query-gpu=name,memory.used,temperature.gpu --format=csv >"$OUT/nvidia-smi-end.txt" 2>&1
tar -C "$OUT/.." -czf /root/results.tgz "$(basename "$OUT")"
touch "$OUT/SWEEP_DONE"
