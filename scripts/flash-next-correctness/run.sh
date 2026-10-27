#!/usr/bin/env bash
# POOT-1333: llama.cpp (CPU and GPU) and poot on the same quantized file, same token windows; one directory of
# artifacts per case plus compare.py's verdict. The same script runs on this box (a small model pair) and on the
# pod (Flash-Next Q2): only the environment differs.
#
#   MODEL=<file.gguf> OUT=<dir> POOT_SCORE=<poot-score> POOT_BACKEND=<ptx|rocm|wgpu|vulkan> \
#   LLAMA_GPU_FLAGS="<-ngl 99 [-dev X]>" [LLAMA_BIN=<dir with llama-perplexity, llama-server>] \
#   [LLAMA_CPU_BIN=<dir>, LLAMA_GPU_BIN=<dir>] scripts/flash-next-correctness/run.sh [cpu|gpu|compare|all] [case ...]
#
# Cases (default: all): `short` (kl_corpus.txt, a window of SHORT_CTX tokens) and `long` (long.txt, LONG_CTX).
# llama-perplexity needs two windows of text and scores one (`--chunks 1`); its window is the token sequence both
# engines then score, and the prompt of the greedy continuation (llama-server /completion with token ids, so no
# tokenizer difference can enter). CPU and GPU runs of llama.cpp are the reference's own noise (compare.py).
#
# Every step runs under a time limit (STEP_TIMEOUT_*): a step that hangs or fails is logged, marks the run failed
# and the pipeline goes on, so one stuck engine costs its limit, not the pod hour. The GPU phase runs first (it
# needs nothing from the CPU phase: the token ids come from its own logits file), then the CPU reference.
# CPU_GREEDY=0 skips llama.cpp's CPU greedy continuation (the CPU decode path of a large MoE model can be orders
# of magnitude slower than its CPU prefill, and the continuation is the least informative leg).
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?}" "${OUT:?}" "${POOT_SCORE:?}" "${POOT_BACKEND:?}" "${LLAMA_GPU_FLAGS:?}"
LLAMA_CPU_BIN=${LLAMA_CPU_BIN:-${LLAMA_BIN:-}}
LLAMA_GPU_BIN=${LLAMA_GPU_BIN:-${LLAMA_BIN:-}}
SHORT_CTX=${SHORT_CTX:-48}
LONG_CTX=${LONG_CTX:-2048}
GREEDY=${GREEDY:-64}
CPU_GREEDY=${CPU_GREEDY:-1}
THREADS=${THREADS:-$(nproc)}
PORT=${PORT:-18081}
PYTHON=${PYTHON:-python3}
# Seconds.
STEP_TIMEOUT_PERPLEXITY=${STEP_TIMEOUT_PERPLEXITY:-1800}
STEP_TIMEOUT_POOT=${STEP_TIMEOUT_POOT:-1800}
STEP_TIMEOUT_GREEDY=${STEP_TIMEOUT_GREEDY:-600}
STEP_TIMEOUT_COMPARE=${STEP_TIMEOUT_COMPARE:-900}

status=0
export MODEL THREADS GREEDY PORT STEP_TIMEOUT_GREEDY
step() { # <seconds> <what> <command...>
    local limit=$1 what=$2
    shift 2
    echo "-- $what" >&2
    # `timeout` runs programs, not functions: a child bash runs the function (exported below) and a timeout signals
    # the child's whole process group, so a llama-server it started goes with it.
    timeout --kill-after=30 "$limit" bash -c '"$@"' step "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "!! $what: exit $rc$([ $rc -eq 124 ] && echo " (timed out after ${limit}s)")" >&2
        status=1
    fi
    return $rc
}

perplexity() { # <bin dir> <corpus> <n_ctx> <out.bin> <device flags...>
    local bin=${1:+$1/} corpus=$2 ctx=$3 out=$4
    shift 4
    "${bin}llama-perplexity" -m "$MODEL" -f "$corpus" -c "$ctx" -b "$ctx" -t "$THREADS" --no-warmup --chunks 1 \
        --save-all-logits "$out" "$@"
}

greedy() { # <bin dir> <ids.txt> <n_ctx> <out.json> <device flags...>
    local bin=${1:+$1/} ids=$2 ctx=$3 out=$4
    shift 4
    "${bin}llama-server" -m "$MODEL" -c $((ctx + GREEDY + 16)) -t "$THREADS" --host 127.0.0.1 --port "$PORT" \
        --no-webui "$@" >"$out.server.log" 2>&1 &
    local server=$! ready=0
    for _ in $(seq 1 1800); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ready=1; break; }
        kill -0 "$server" 2>/dev/null || break
        sleep 1
    done
    local rc=1
    if [ $ready = 1 ]; then
        printf '{"prompt":[%s],"n_predict":%d,"temperature":0,"top_k":1,"return_tokens":true,"cache_prompt":false,"ignore_eos":true}' \
            "$(tr -s ' \n' ',' <"$ids" | sed 's/,$//')" "$GREEDY" |
            curl -sf --max-time "$STEP_TIMEOUT_GREEDY" "http://127.0.0.1:$PORT/completion" \
                -H 'Content-Type: application/json' -d @- >"$out"
        rc=$?
    fi
    kill "$server" 2>/dev/null
    wait "$server" 2>/dev/null
    [ $rc -eq 0 ] || rm -f "$out"
    return $rc
}

export -f perplexity greedy

phase_gpu() { # <name> <corpus> <n_ctx>
    local dir=$OUT/$1 ctx=$3
    mkdir -p "$dir"
    echo "== $1 gpu: window of $ctx tokens" >&2
    step "$STEP_TIMEOUT_PERPLEXITY" "$1: llama.cpp GPU logits" \
        perplexity "$LLAMA_GPU_BIN" "$here/corpus/$2" "$ctx" "$dir/gpu.bin" $LLAMA_GPU_FLAGS || return 0
    step 120 "$1: token ids of the window" "$PYTHON" "$here/compare.py" tokens "$dir/gpu.bin" "$dir/ids.txt" || return 0
    step "$STEP_TIMEOUT_POOT" "$1: poot-score" \
        "$POOT_SCORE" --model "$MODEL" --tokens "$dir/ids.txt" --start $((ctx / 2)) --greedy "$GREEDY" \
        --backend "$POOT_BACKEND" --out "$dir/poot.json"
    step $((STEP_TIMEOUT_GREEDY + 1800)) "$1: llama.cpp GPU greedy" \
        greedy "$LLAMA_GPU_BIN" "$dir/ids.txt" "$ctx" "$dir/greedy-gpu.json" $LLAMA_GPU_FLAGS
}

phase_cpu() { # <name> <corpus> <n_ctx>
    local dir=$OUT/$1 ctx=$3
    mkdir -p "$dir"
    echo "== $1 cpu: window of $ctx tokens" >&2
    step "$STEP_TIMEOUT_PERPLEXITY" "$1: llama.cpp CPU logits" \
        perplexity "$LLAMA_CPU_BIN" "$here/corpus/$2" "$ctx" "$dir/cpu.bin" -ngl 0 -dev none || return 0
    [ "$CPU_GREEDY" = 1 ] && [ -s "$dir/ids.txt" ] || return 0
    step $((STEP_TIMEOUT_GREEDY + 1800)) "$1: llama.cpp CPU greedy" \
        greedy "$LLAMA_CPU_BIN" "$dir/ids.txt" "$ctx" "$dir/greedy-cpu.json" -ngl 0 -dev none
}

phase_compare() { # <name>
    local dir=$OUT/$1 optional=()
    [ -s "$dir/greedy-cpu.json" ] && optional+=(--greedy-cpu "$dir/greedy-cpu.json")
    [ -s "$dir/greedy-gpu.json" ] && optional+=(--greedy-cuda "$dir/greedy-gpu.json")
    step "$STEP_TIMEOUT_COMPARE" "$1: compare" \
        "$PYTHON" "$here/compare.py" --cpu "$dir/cpu.bin" --cuda "$dir/gpu.bin" --poot "$dir/poot.json" \
        "${optional[@]}" --out "$dir/report.json" >/dev/null
    [ -s "$dir/report.json" ] || return 0
    echo "== $1: $("$PYTHON" -c "import json,sys; r=json.load(open(sys.argv[1])); print('PASS' if r['pass'] else 'FAIL', r['failures'])" "$dir/report.json")" >&2
}

phase=${1:-all}
shift || true
cases=("$@")
[ ${#cases[@]} -gt 0 ] || cases=(short long)
for phase_name in gpu cpu compare; do
    [ "$phase" = all ] || [ "$phase" = "$phase_name" ] || continue
    for name in "${cases[@]}"; do
        case $name in
        short) args=(short short.txt "$SHORT_CTX") ;;
        long) args=(long long.txt "$LONG_CTX") ;;
        *) echo "unknown case $name" >&2; exit 2 ;;
        esac
        "phase_$phase_name" "${args[@]}"
    done
done
echo "run.sh: $([ $status = 0 ] && echo 'every step finished' || echo 'a step failed or timed out')" >&2
exit $status
