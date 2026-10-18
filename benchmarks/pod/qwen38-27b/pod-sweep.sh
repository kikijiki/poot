#!/usr/bin/env bash
# POOT-1184: Qwen3.8-27B-FP8 detailed performance on ONE NVIDIA card (runs ON the pod, as root).
#
# poot (PTX) decode-curve / prefill / per-phase sweep, then vLLM (FP8) and llama.cpp CUDA (Q4_K_XL GGUF) on the
# same card at matched settings. Every result lands under $OUT; nothing is read back from stdout except the final
# tarball. launch.sh starts this detached (setsid nohup) and the launcher waits for $OUT/SWEEP_DONE.
#
# A phase failing never aborts the sweep: it is recorded in phases.tsv with its exit code and the next phase
# runs (a refusal is a result). Only the model download is a hard prerequisite for the phases that read it.
#
# Inputs (uploaded by launch.sh): $BIN/poot-bench-runner, $BIN/poot-serve (patchelf'd), $HERE/summarize.py.
# Knobs (env, all optional): OUT, WORK, MAX_TOTAL_S, ISLS, OSL, CONCS, BATCH_ISLS, LLAMACPP_TAG, VLLM_VERSION.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/root/results}"
WORK="${WORK:-/root/work}"
BIN="${BIN:-/root/bin}"
FP8_REPO="Qwen/Qwen3.8-27B-FP8"
GGUF_REPO="unsloth/Qwen3.8-27B-GGUF"
GGUF_FILE="Qwen3.8-27B-UD-Q4_K_XL.gguf"
FP8_DIR="$WORK/models/Qwen3.8-27B-FP8"
GGUF_DIR="$WORK/models/Qwen3.8-27B-GGUF"
GGUF="$GGUF_DIR/$GGUF_FILE"

ISLS="${ISLS:-128,512,2048,8192}"          # prefill / decode-curve contexts
OSL="${OSL:-128}"                          # decoded tokens per request
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-5}"
BATCH_ISLS="${BATCH_ISLS:-512 2048}"       # prompt lengths of the batched cells
CONCS="${CONCS:-4 16 32}"                  # concurrent requests of the batched cells (batch 1 is the curve)
BATCH_PROMPTS="${BATCH_PROMPTS:-64}"       # requests per batched cell
MAX_CTX=$((8192 + OSL + 64))               # per-sequence context every server is sized to
MAX_SEQS=32                                # most concurrent sequences every server is sized to
LLAMACPP_TAG="${LLAMACPP_TAG:-b9601}"      # the tag benchmarks/docker/Dockerfile pins
VLLM_VERSION="${VLLM_VERSION:-0.22.1}"     # the version benchmarks/docker/Dockerfile pins
ROW_CAP_S="${ROW_CAP_S:-300}"              # hard cap of every serve row (a capped row is rc=124 in phases.tsv, never a stall)
# poot-serve plans 4 rows x 4096 positions (see logs/serve_poot.out), so its grid is cut to what it can hold, at a
# short OSL so a row fits the cap at its measured prefill rate; vLLM / llama.cpp keep the full grid above.
POOT_SERVE_B1_ISLS="${POOT_SERVE_B1_ISLS:-128 512}"; POOT_SERVE_ISLS="${POOT_SERVE_ISLS:-512}"
POOT_SERVE_CONCS="${POOT_SERVE_CONCS:-4}"; POOT_SERVE_OSL="${POOT_SERVE_OSL:-32}"; POOT_SERVE_PROMPTS="${POOT_SERVE_PROMPTS:-8}"
MAX_TOTAL_S="${MAX_TOTAL_S:-14400}"        # no new phase starts after this many seconds (cost cap)
PORT_POOT="${PORT_POOT:-8080}"; PORT_VLLM="${PORT_VLLM:-8000}"; PORT_LLAMA="${PORT_LLAMA:-8081}"

mkdir -p "$OUT/raw" "$OUT/logs" "$WORK"
export PATH="/root/.local/bin:/usr/local/cuda/bin:$PATH"
# Kernels are generated at runtime; one cache dir so the first phase pays and the rest reuse (recorded below).
export POOT_KERNEL_CACHE_DIR="$WORK/poot-kernel-cache"
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_DISABLE_TELEMETRY=1 HF_XET_HIGH_PERFORMANCE=1
START=$(date +%s)
: >"$OUT/phases.tsv"; : >"$OUT/commands.tsv"
printf 'phase\trc\twall_s\tpeak_vram_mib\tnote\n' >"$OUT/phases.tsv"

say() { printf '[%s +%ss] %s\n' "$(date +%T)" "$(( $(date +%s) - START ))" "$*" | tee -a "$OUT/progress.log"; }
budget_ok() { [ $(( $(date +%s) - START )) -lt "$MAX_TOTAL_S" ]; }

# --- peak device memory of one phase, sampled externally (not engine-reported) -------------------------------
# One sampler per name (a server phase samples while its client cells sample on their own).
vram_start() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >"$WORK/vram.$1" 2>/dev/null & echo $! >"$WORK/vram.$1.pid"; }
vram_stop() { local pid; pid=$(cat "$WORK/vram.$1.pid" 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null; sort -n "$WORK/vram.$1" 2>/dev/null | tail -1; }

# phase NAME TIMEOUT_S -- CMD...   : run CMD under a timeout, log it, record rc / wall / peak VRAM / exact command.
phase() {
  local name="$1" tmo="$2"; shift 3
  if ! budget_ok; then say "SKIP $name (MAX_TOTAL_S=$MAX_TOTAL_S spent)"; printf '%s\t-\t0\t\tskipped: budget\n' "$name" >>"$OUT/phases.tsv"; return 1; fi
  printf '%s\t%s\n' "$name" "$*" >>"$OUT/commands.tsv"
  say "BEGIN $name (timeout ${tmo}s)"
  local t0 rc peak; t0=$(date +%s)
  vram_start "$name"
  timeout -k 30 "$tmo" "$@" >"$OUT/logs/$name.out" 2>"$OUT/logs/$name.err"; rc=$?
  peak=$(vram_stop "$name")
  printf '%s\t%s\t%s\t%s\t\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "${peak:-}" >>"$OUT/phases.tsv"
  say "END   $name rc=$rc"
  return "$rc"
}

# --- environment / versions ----------------------------------------------------------------------------------
say "pod sweep start"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt" 2>&1
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
ARCH="${CC:-90}"
{
  echo "gpu_name=$GPU_NAME"; echo "gpu_mem_mib=$GPU_MEM_MIB"; echo "compute_cap=$CC"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "host_mem_gib=$(free -g | awk '/Mem:/ {print $2}')"; echo "host_cpus=$(nproc)"
  echo "kernel=$(uname -r)"; echo "image_cuda=$(nvcc --version 2>/dev/null | tail -1)"
  echo "driver_cuda=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1)"
  echo "poot_runner=$("$BIN/poot-bench-runner" --version 2>&1 | head -1)"
  echo "llamacpp_tag=$LLAMACPP_TAG"; echo "vllm_version_pinned=$VLLM_VERSION"
  echo "isls=$ISLS osl=$OSL warmup=$WARMUP iters=$ITERS batch_isls=$BATCH_ISLS concs=$CONCS prompts=$BATCH_PROMPTS"
  echo "max_ctx=$MAX_CTX max_seqs=$MAX_SEQS"
} >"$OUT/env.txt"
case "$GPU_NAME" in
  *H100*|*H200*) FP8_NOTE="native FP8 tensor cores (sm_90)";;
  *) FP8_NOTE="NO native FP8 (sm_$CC): vLLM falls back to weight-only FP8 (Marlin); poot dequantizes in-kernel. Not an H100 comparison.";;
esac
echo "fp8_note=$FP8_NOTE" >>"$OUT/env.txt"
say "gpu=$GPU_NAME ${GPU_MEM_MIB}MiB sm_$CC; $FP8_NOTE"

# --- 1. downloads --------------------------------------------------------------------------------------------
HF=$(command -v hf || command -v huggingface-cli)
phase download_fp8 3600 -- "$HF" download "$FP8_REPO" --local-dir "$FP8_DIR"
phase download_gguf 3600 -- "$HF" download "$GGUF_REPO" "$GGUF_FILE" --local-dir "$GGUF_DIR"
for repo in "$FP8_REPO" "$GGUF_REPO"; do
  echo "$repo $(curl -fsS -m 30 "https://huggingface.co/api/models/$repo" | grep -o '"sha":"[0-9a-f]*"' | head -1)" >>"$OUT/model-revisions.txt"
done
du -sh "$FP8_DIR" "$GGUF" >>"$OUT/env.txt" 2>&1
FP8_OK=0; [ -s "$FP8_DIR/config.json" ] && [ -s "$FP8_DIR/model.safetensors.index.json" ] && FP8_OK=1
GGUF_OK=0; [ -s "$GGUF" ] && GGUF_OK=1

# --- 1b. CPU-oracle fixture (cheap, decisive: first) -------------------------------------------------------------
# The qwen35_27b_fp8_receipt row on PTX: loads the real checkpoint, runs the CPU oracle (needs a large host RAM:
# host_mem_gib in env.txt) and prints RECEIPT lines; oracle_fixture.py turns them into the fixture the ROCm / wgpu
# lanes compare against. nextest is fetched on the pod; the archive and workspace tar come from stage.sh.
ORACLE_OK=0
ORACLE_FIXTURE=crates/poot-llm/tests/data/qwen38_27b_fp8_oracle_top5.json   # repo-relative: copy $OUT/<this> into the checkout
ORACLE_ARCHIVE="${ORACLE_ARCHIVE:-/root/oracle-lane.tar.zst}"; WS_TAR="${WS_TAR:-/root/ws.tar}"; WS_DIR="${WS_DIR:-/root/ws}"; EX_DIR="${EX_DIR:-/root/ex}"
if [ "$FP8_OK" = 1 ] && [ -s "$ORACLE_ARCHIVE" ] && [ -s "$WS_TAR" ]; then
  mkdir -p "$EX_DIR" "$WS_DIR" "$WORK/oracle-models" "$OUT/oracle" "$OUT/$(dirname "$ORACLE_FIXTURE")"
  ln -sfn "$FP8_DIR" "$WORK/oracle-models/qwen3.8-27b-fp8"      # POOT_MODELS_DIR layout: <dir>/<checkpoint>
  tar -C "$WS_DIR" -xf "$WS_TAR"
  command -v cargo-nextest >/dev/null || curl -fsSL https://get.nexte.st/latest/linux | tar xz -C /usr/local/bin
  export POOT_REQUIRE_PTX=1 POOT_REQUIRE_MODELS=1 POOT_MODELS_DIR="$WORK/oracle-models" POOT_LLC="${POOT_LLC:-/usr/local/bin/llc}"
  # --run-ignored all, but the -E filter selects ONE test (the #[ignore]d PTX receipt row), so nothing else runs.
  phase oracle_receipt 5400 -- cargo-nextest nextest run --archive-file "$ORACLE_ARCHIVE" --extract-to "$EX_DIR" \
        --workspace-remap "$WS_DIR" --no-fail-fast --no-tests=fail --run-ignored all --no-capture \
        -E 'test(/qwen35_27b_fp8_receipt::ptx$/)' && ORACLE_OK=1
  cat "$OUT/logs/oracle_receipt.out" "$OUT/logs/oracle_receipt.err" >"$OUT/oracle/receipt-output.txt"
  # The fixture exists only if the row passed AND the parse succeeded: written under a temp name, moved on success.
  if [ "$ORACLE_OK" = 1 ]; then
    REV=$(grep -h 'Qwen/Qwen3.8-27B-FP8' "$OUT/model-revisions.txt" | head -1 | grep -o '[0-9a-f]\{40\}')
    if "${VENV_VLLM:-/opt/vllm-venv}/bin/python" -I "$HERE/oracle_fixture.py" "$OUT/oracle/receipt-output.txt" ptx "${REV:-unknown}" \
         >"$OUT/$ORACLE_FIXTURE.tmp" 2>"$OUT/oracle/fixture.err"; then mv "$OUT/$ORACLE_FIXTURE.tmp" "$OUT/$ORACLE_FIXTURE"
    else rm -f "$OUT/$ORACLE_FIXTURE.tmp" "$OUT/$ORACLE_FIXTURE"; say "oracle fixture NOT written: $(cat "$OUT/oracle/fixture.err")"; fi
  else
    rm -f "$OUT/$ORACLE_FIXTURE"; say "oracle row failed: no fixture written (see logs/oracle_receipt.err)"
  fi
else
  say "SKIP oracle receipt (fp8=$FP8_OK, archive/ws.tar uploaded: $([ -s "$ORACLE_ARCHIVE" ] && echo yes || echo no))"
fi

# --- shared batched client: vLLM's `bench serve`, pointed at each engine's OpenAI server --------------------
VENV_VLLM="${VENV_VLLM:-/opt/vllm-venv}"   # the image bakes vllm==0.22.1 here
if ! "$VENV_VLLM/bin/python" -c 'import vllm' 2>/dev/null; then
  VENV_VLLM="$WORK/vllm-venv"
  phase vllm_install 2400 -- bash -c "uv venv '$VENV_VLLM' --python 3.11 && uv pip install --python '$VENV_VLLM/bin/python' 'vllm==$VLLM_VERSION' --torch-backend=auto accelerate pillow"
fi
VPY="$VENV_VLLM/bin/python"; VLLM="$VENV_VLLM/bin/vllm"
{ echo "vllm=$("$VPY" -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1)"
  echo "torch=$("$VPY" -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1)"
  "$VPY" -m pip freeze 2>/dev/null | grep -Ei '^(vllm|torch|flashinfer|transformers|xformers)' ; } >"$OUT/versions-vllm.txt"
"$VLLM" bench serve --help >"$WORK/bench-serve-help.txt" 2>&1
CLIENT_FLAGS=()
for f in --temperature; do grep -q -- "$f" "$WORK/bench-serve-help.txt" && CLIENT_FLAGS+=("$f" 0); done
grep -q -- '--num-warmups' "$WORK/bench-serve-help.txt" && CLIENT_FLAGS+=(--num-warmups 2)
grep -q -- '--disable-tqdm' "$WORK/bench-serve-help.txt" && CLIENT_FLAGS+=(--disable-tqdm)
grep -q -- '--save-detailed' "$WORK/bench-serve-help.txt" && CLIENT_FLAGS+=(--save-detailed)   # per-request errors / output lengths
echo "client_flags=${CLIENT_FLAGS[*]}" >>"$OUT/env.txt"

# client ENGINE PORT MODEL TOKENIZER ISL OSL CONC NPROMPTS : one cell; JSON in $OUT/raw/<engine>_isl<ISL>_c<CONC>.json
client() {
  local engine="$1" port="$2" model="$3" tok="$4" isl="$5" osl="$6" conc="$7" n="$8"
  phase "${engine}_isl${isl}_c${conc}" "$ROW_CAP_S" -- "$VLLM" bench serve --backend openai --base-url "http://127.0.0.1:$port" \
    --endpoint /v1/completions --model "$model" --tokenizer "$tok" --dataset-name random \
    --random-input-len "$isl" --random-output-len "$osl" --num-prompts "$n" --max-concurrency "$conc" \
    --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT/raw" --result-filename "${engine}_isl${isl}_c${conc}.json" "${CLIENT_FLAGS[@]}"
}
# cells ENGINE PORT MODEL TOKENIZER : the batch-1 rows (conc 1, per ISL) then the batched grid; poot gets its own cut grid.
cells() {
  local engine="$1" port="$2" model="$3" tok="$4" isl conc b1="${ISLS//,/ }" bi="$BATCH_ISLS" bc="$CONCS" osl="$OSL" n="$BATCH_PROMPTS" it="$ITERS"
  if [ "$engine" = poot ]; then b1="$POOT_SERVE_B1_ISLS"; bi="$POOT_SERVE_ISLS"; bc="$POOT_SERVE_CONCS"; osl="$POOT_SERVE_OSL"; n="$POOT_SERVE_PROMPTS"; it=3; fi
  for isl in $b1; do client "$engine" "$port" "$model" "$tok" "$isl" "$osl" 1 "$it"; done
  for isl in $bi; do for conc in $bc; do client "$engine" "$port" "$model" "$tok" "$isl" "$osl" "$conc" "$n"; done; done
}
# serve_wait PID PORT TIMEOUT_S : poll /health (or /v1/models) until the server answers or its process dies.
serve_wait() {
  local pid="$1" port="$2" tmo="$3" t=0
  while [ "$t" -lt "$tmo" ]; do
    kill -0 "$pid" 2>/dev/null || return 2
    curl -fsS -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && return 0
    sleep 5; t=$((t + 5))
  done
  return 1
}
# with_server ENGINE PORT LOAD_TIMEOUT -- SERVER_CMD... : start detached, wait ready, run cells, stop, record peak VRAM.
with_server() {
  local engine="$1" port="$2" ltmo="$3"; shift 4
  if ! budget_ok; then say "SKIP $engine (budget)"; return 1; fi
  printf 'serve_%s\t%s\n' "$engine" "$*" >>"$OUT/commands.tsv"
  vram_start "serve_$engine"
  local t0 rc; t0=$(date +%s)
  setsid "$@" >"$OUT/logs/serve_$engine.out" 2>"$OUT/logs/serve_$engine.err" &
  local pid=$!
  serve_wait "$pid" "$port" "$ltmo"; rc=$?
  printf 'serve_%s_ready\t%s\t%s\t\t\n' "$engine" "$rc" "$(( $(date +%s) - t0 ))" >>"$OUT/phases.tsv"
  if [ "$rc" = 0 ]; then
    local model; model=$(curl -fsS "http://127.0.0.1:$port/v1/models" | "$VPY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
    echo "${engine}_model_id=$model" >>"$OUT/env.txt"
    cells "$engine" "$port" "$model" "$FP8_DIR"
  else
    say "$engine server not ready (rc=$rc); see logs/serve_$engine.err"
  fi
  kill -TERM -- "-$pid" 2>/dev/null; sleep 5; kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  printf 'serve_%s_peak\t-\t%s\t%s\t\n' "$engine" "$(( $(date +%s) - t0 ))" "$(vram_stop "serve_$engine")" >>"$OUT/phases.tsv"
  sleep 10   # let the driver release the card before the next engine
}

# --- 2. vLLM (FP8) -------------------------------------------------------------------------------------------
# --limit-mm-per-prompt zeroes the vision tower's profiling (text-only workload, like poot's). Prefix caching is
# off (random prompts defeat it anyway; matches the suite's contract). max-num-batched-tokens is vLLM's chunked
# prefill budget; recorded, not tuned.
[ "$FP8_OK" = 1 ] && with_server vllm "$PORT_VLLM" 2400 -- "$VLLM" serve "$FP8_DIR" --host 127.0.0.1 --port "$PORT_VLLM" \
    --max-model-len "$MAX_CTX" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' --dtype bfloat16

# --- 3. llama.cpp CUDA (Q4_K_XL GGUF) ------------------------------------------------------------------------
LCPP="${LCPP:-$WORK/llama.cpp}"
if [ "$GGUF_OK" = 1 ] && budget_ok; then
  # The image's baked llama.cpp is sm_86 only and the image has no nvcc: install the toolkit pieces and build the
  # pod's own arch. Same pinned tag as the image.
  phase llamacpp_build 3000 -- bash -c "set -e
    apt-get update -qq && apt-get install -y -qq cmake ninja-build cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-driver-dev-12-6 libcublas-dev-12-6 >/dev/null
    export PATH=/usr/local/cuda/bin:\$PATH
    [ -d '$LCPP' ] || git clone --depth 1 --branch '$LLAMACPP_TAG' https://github.com/ggml-org/llama.cpp.git '$LCPP'
    cd '$LCPP' && cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='$ARCH' \
      -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' -DLLAMA_CURL=OFF
    cmake --build build -j\"\$(nproc)\" --target llama-bench llama-server
    git rev-parse HEAD"
  { echo "llamacpp_tag=$LLAMACPP_TAG"; echo "llamacpp_commit=$(git -C "$LCPP" rev-parse HEAD 2>/dev/null)"
    echo "llamacpp_arch=$ARCH"; "$LCPP/build/bin/llama-bench" --version 2>&1 | head -3; } >"$OUT/versions-llamacpp.txt"
  LB="$LCPP/build/bin/llama-bench"; LS="$LCPP/build/bin/llama-server"
  if [ -x "$LB" ]; then
    # Prefill: pp at each ISL. Decode: tg128 at depth = ISL, the equivalent of "prefill ISL, then decode OSL".
    # -ub/-b 2048 so prefill is not capped by the 512 default micro-batch; -fa 1 as every modern setup runs it.
    phase llamacpp_bench_pp 2400 -- "$LB" -m "$GGUF" -ngl 99 -fa 1 -b 2048 -ub 2048 -p "$ISLS" -n 0 -r 3 -o json
    cp "$OUT/logs/llamacpp_bench_pp.out" "$OUT/raw/llamacpp_bench_pp.json"
    phase llamacpp_bench_tg 2400 -- "$LB" -m "$GGUF" -ngl 99 -fa 1 -b 2048 -ub 2048 -p 0 -n "$OSL" -d "$ISLS" -r 3 -o json
    cp "$OUT/logs/llamacpp_bench_tg.out" "$OUT/raw/llamacpp_bench_tg.json"
    # Same server-side grid as the other engines: one server, MAX_SEQS slots, MAX_CTX tokens each.
    with_server llamacpp "$PORT_LLAMA" 1200 -- "$LS" -m "$GGUF" -ngl 99 -fa on -b 2048 -ub 2048 \
        -np "$MAX_SEQS" -c "$(( MAX_SEQS * MAX_CTX ))" --host 127.0.0.1 --port "$PORT_LLAMA" --no-webui --no-context-shift
  fi
else
  say "SKIP llama.cpp (gguf=$GGUF_OK)"
fi

# --- 4. poot (PTX), single stream -------------------------------------------------------------------------------------------
# Single stream (batch 1). decode-curve prefills exactly ISL tokens, then decodes OSL with EOS ignored; TTFT of
# each point is the prefill, so prefill tok/s = ISL / ttft. --precision names the compute dtype: the FP8 weights
# are used as stored (block-scaled packed projections), there is no fp8 flag.
RUNNER="$BIN/poot-bench-runner"
CURVE=("$RUNNER" --model-dir "$FP8_DIR" --backend ptx --precision bf16 --mode decode-curve --isl-list "$ISLS"
       --osl "$OSL" --warmup "$WARMUP" --iters "$ITERS" --json)
if [ "$FP8_OK" = 1 ]; then
  phase poot_curve 5400 -- "${CURVE[@]}"                          # clean timing, first: pays the cold kernel cache
  tail -1 "$OUT/logs/poot_curve.out" >"$OUT/raw/poot_curve.json"
  # Driver phase timing + bytes read per token. Run 1 (H100): every field was `unknown` ("no pinned ceiling for this
  # device"; the fit needs >= 2 points with different bytes, and weights-only decode reads the same bytes at every
  # ISL), so the full curve again would only repeat 44 min of timing. One short point keeps the receipt.
  phase poot_curve_roofline 1800 -- "$RUNNER" --model-dir "$FP8_DIR" --backend ptx --precision bf16 --mode decode-curve \
        --isl-list 128 --osl "$OSL" --warmup 1 --iters 2 --json --roofline
  tail -1 "$OUT/logs/poot_curve_roofline.out" >"$OUT/raw/poot_curve_roofline.json"
  # Per-phase / per-kernel device time. Diagnostic: the timed executor perturbs submission, so these are
  # breakdown shares, NOT throughput evidence. Fewer iterations, same contexts.
  phase poot_curve_profile 5400 -- "$RUNNER" --model-dir "$FP8_DIR" --backend ptx --precision bf16 \
        --mode decode-curve --isl-list "$ISLS" --osl "$OSL" --warmup 1 --iters 2 --profile --json
  tail -1 "$OUT/logs/poot_curve_profile.out" >"$OUT/raw/poot_curve_profile.json"
else
  say "SKIP poot phases: FP8 checkpoint missing"
fi

# --- 5. poot batched (poot-serve, PTX), LAST: the slow, capped rows --------------------------------------------
# It has no ignore-eos switch: the client's actual output tokens are in the JSON.
[ "$FP8_OK" = 1 ] && with_server poot "$PORT_POOT" 3600 -- "$BIN/poot-serve" --backend ptx "$FP8_DIR" "127.0.0.1:$PORT_POOT"

# --- 6. collect ----------------------------------------------------------------------------------------------
say "summarize"
"$VPY" "$HERE/summarize.py" "$OUT" >"$OUT/summary.tsv" 2>"$OUT/logs/summarize.err" || say "summarize failed"
nvidia-smi --query-gpu=name,memory.used --format=csv >"$OUT/nvidia-smi-end.txt" 2>&1
say "sweep done, total $(( $(date +%s) - START ))s"
tar -C "$(dirname "$OUT")" -czf "$(dirname "$OUT")/results.tgz" "$(basename "$OUT")"
touch "$OUT/SWEEP_DONE"
