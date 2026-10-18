#!/usr/bin/env bash
# POOT-1184 local dry run: no GPU, no pod, no network, no model. Proves
#   1. every script parses (bash -n) and summarize.py compiles;
#   2. the launcher's exact `poot-orchestrator exec` argv is accepted by the REAL clap parser (the run is stopped
#      by the missing API key right after parsing: RUNPOD_API_KEY is unset for that call);
#   3. every flag pod-sweep.sh passes to poot-bench-runner / poot-serve exists (the real binaries, pointed at a
#      nonexistent model: a flag the binary does not know fails with "unknown argument", a known one gets as far
#      as the model load);
#   4. the whole of pod-sweep.sh runs end to end against mock engines (control flow, phases.tsv, budget skip,
#      summarize, tarball), and launch.sh extract round-trips the tarball.
# Usage: dry-run.sh <poot-bench-runner> <poot-serve> <poot-orchestrator>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${1:?poot-bench-runner}"; SERVE="${2:?poot-serve}"; ORCH="${3:?poot-orchestrator}"
fail=0
ok() { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }

# 1. syntax
for f in pod-sweep.sh launch.sh stage.sh dry-run.sh; do bash -n "$HERE/$f" && ok "bash -n $f" || bad "bash -n $f"; done
python3 -I -m py_compile "$HERE/summarize.py" && ok "py_compile summarize.py" || bad "py_compile summarize.py"

# 2. the orchestrator accepts the launcher's argv. Take it from `launch.sh print` so the check cannot drift.
eval "set -- $(ORCH="$ORCH" STAGE=/nonexistent "$HERE/launch.sh" print)"
out=$(env -u RUNPOD_API_KEY "$@" 2>&1); rc=$?
if [ "$rc" -ne 0 ] && echo "$out" | grep -qi 'runpod_api_key\|api key'; then ok "orchestrator exec argv parses (stopped at the API key, rc=$rc)"
else bad "orchestrator exec argv: rc=$rc: $(echo "$out" | head -3)"; fi

# 3. runner / serve flags, read from the script itself
runner_line() { grep -n -- 'poot-bench-runner\|"\$RUNNER"\|CURVE=' "$HERE/pod-sweep.sh" >/dev/null; }
check_runner() {  # NAME ARGS...
  local name="$1"; shift
  out=$("$RUNNER" "$@" 2>&1); 
  if echo "$out" | grep -q 'unknown argument\|needs a value\|must be'; then bad "runner $name: $(echo "$out" | head -2)"
  else ok "runner flags accepted: $name"; fi
}
check_runner curve --model-dir /nonexistent --backend ptx --precision bf16 --mode decode-curve --isl-list 128,512,2048,8192 --osl 128 --warmup 2 --iters 5 --json
check_runner roofline --model-dir /nonexistent --backend ptx --precision bf16 --mode decode-curve --isl-list 128 --osl 128 --warmup 2 --iters 5 --json --roofline
check_runner profile --model-dir /nonexistent --backend ptx --precision bf16 --mode decode-curve --isl-list 128 --osl 128 --warmup 1 --iters 2 --profile --json
"$RUNNER" --version >/dev/null 2>&1 && ok "runner --version" || bad "runner --version"
# the same flags must be what pod-sweep.sh passes (guards the checks above against drifting from the script)
for flag in --mode --isl-list --osl --warmup --iters --precision --backend --roofline --profile --json --model-dir; do
  grep -q -- "$flag" "$HERE/pod-sweep.sh" && ok "pod-sweep.sh passes $flag" || bad "pod-sweep.sh no longer passes $flag"
done
sout=$("$SERVE" --help 2>&1)
echo "$sout" | grep -q -- '--backend' && echo "$sout" | grep -q 'ptx' && ok "poot-serve --backend ptx" || bad "poot-serve --help lacks --backend ptx"
"$SERVE" --backend ptx /nonexistent 127.0.0.1:1 >/tmp/dry-serve.$$ 2>&1; grep -q 'unexpected argument\|unrecognized' /tmp/dry-serve.$$ && bad "poot-serve positional MODEL ADDR: $(head -2 /tmp/dry-serve.$$)" || ok "poot-serve MODEL ADDR positionals accepted"
rm -f /tmp/dry-serve.$$

# 4. whole-script run against mocks
T=$(mktemp -d); MOCK="$T/mock"; mkdir -p "$MOCK" "$T/bin" "$T/venv/bin" "$T/py"
cat >"$MOCK/nvidia-smi" <<'M'
#!/usr/bin/env bash
case "$*" in
  *-lms*) while :; do echo 61234; sleep 0.2; done ;;
  *compute_cap*) echo "9.0" ;; *memory.total*) echo 81559 ;; *name*) echo "NVIDIA H100 80GB HBM3" ;;
  *driver_version*) echo 580.0 ;; *) echo "mock nvidia-smi" ;;
esac
M
cat >"$MOCK/hf" <<'M'
#!/usr/bin/env bash
# hf download REPO [FILE] --local-dir DIR
d=""; while [ $# -gt 0 ]; do [ "$1" = --local-dir ] && d="$2"; shift; done
mkdir -p "$d"; echo '{}' >"$d/config.json"; echo '{}' >"$d/model.safetensors.index.json"; echo x >"$d/Qwen3.8-27B-UD-Q4_K_XL.gguf"
M
cat >"$MOCK/curl" <<'M'
#!/usr/bin/env bash
case "$*" in
  *huggingface.co/api*) echo '{"sha":"deadbeef0"}' ;;
  */v1/models) printf '{"object":"list","data":[{"id":"mock-model"}]}' ;;
  *) exit 22 ;;
esac
M
printf '#!/usr/bin/env bash\nexit 0\n' >"$MOCK/apt-get"; cp "$MOCK/apt-get" "$MOCK/uv"; cp "$MOCK/apt-get" "$MOCK/curl-noop"
cat >"$MOCK/cargo-nextest" <<'M'
#!/usr/bin/env bash
# prints what the real row prints on stderr (the line shapes are those of generates_and_matches_the_oracle)
echo 'RECEIPT prompt "The capital of France is" ids [760, 6511, 314, 9338, 369]' >&2
echo 'RECEIPT generated [11751, 524, 264, 31586, 1076, 13, 25, 2] text " Paris is a city. :"' >&2
echo 'RECEIPT device first-token top5 [(11751, -0.4533), (524, -3.3506), (264, -3.5284), (31586, -3.9209), (1076, -3.9423)]' >&2
echo 'RECEIPT oracle first-token top5 [(11751, -0.4533), (524, -3.3506), (264, -3.5284), (31586, -3.9209), (1076, -3.9423)]' >&2
M
mkdir -p "$T/ws-src" && echo x >"$T/ws-src/f" && tar -C "$T/ws-src" -cf "$T/ws.tar" f && echo z >"$T/oracle-lane.tar.zst"
cat >"$MOCK/git" <<'M'
#!/usr/bin/env bash
case "$1" in clone) mkdir -p "${@: -1}" ;; rev-parse) echo cafe1234 ;; esac
M
cat >"$MOCK/cmake" <<'M'
#!/usr/bin/env bash
if [ "$1" = --build ]; then mkdir -p build/bin; printf '#!/usr/bin/env bash\ncase "$*" in *--version*) echo mock-llama;; *) if echo "$*" | grep -q -- "-n 0"; then echo "[{\\"n_prompt\\":128,\\"n_gen\\":0,\\"n_depth\\":0,\\"avg_ts\\":1000.0,\\"stddev_ts\\":1.0}]"; else echo "[{\\"n_prompt\\":0,\\"n_gen\\":128,\\"n_depth\\":128,\\"avg_ts\\":40.0,\\"stddev_ts\\":0.5}]"; fi;; esac\n' >build/bin/llama-bench; printf '#!/usr/bin/env bash\nfor a in "$@"; do [ "$prev" = --port ] && port="$a"; prev="$a"; done\nmkdir -p /tmp/mock-llama.$$/v1; echo "{\\"object\\":\\"list\\",\\"data\\":[{\\"id\\":\\"mock-model\\"}]}" >/tmp/mock-llama.$$/v1/models\ncd /tmp/mock-llama.$$ && exec python3 -m http.server "$port" >/dev/null 2>&1\n' >build/bin/llama-server; chmod +x build/bin/*; fi
M
cat >"$T/bin/poot-bench-runner" <<'M'
#!/usr/bin/env bash
[ "$1" = --version ] && { echo mock-runner; exit 0; }
sleep 1   # long enough for the 0.5 s VRAM sampler to take a sample
# argv is validated by the REAL runner (it fails on a flag it does not know before touching the model)
chk=$("$REAL_RUNNER" "$@" 2>&1 </dev/null); if echo "$chk" | grep -q 'unknown argument\|needs a value'; then echo "$chk" >&2; exit 2; fi
echo '{"framework":"poot","status":"ok","osl":128,"caveats":[],"curve":[{"isl":128,"iters":[{"ttft_ms":50.0,"tpot_ms":25.0,"tokens":128}],"roofline":null,"device_time":null}]}'
M
cat >"$T/bin/poot-serve" <<'M'
#!/usr/bin/env bash
# last arg is ADDR host:port
mkdir -p /tmp/mock-serve.$$/v1; echo '{"object":"list","data":[{"id":"mock-model"}]}' >/tmp/mock-serve.$$/v1/models
cd /tmp/mock-serve.$$ && exec python3 -m http.server "${@: -1}" 2>/dev/null
M
sed -i 's|exec python3 -m http.server "${@: -1}"|exec python3 -m http.server "${1##*:}"|' "$T/bin/poot-serve"
cat >"$T/venv/bin/vllm" <<'M'
#!/usr/bin/env bash
if [ "$1 $2" = "bench serve" ]; then
  if [ "$3" = --help ]; then echo "--temperature --num-warmups --disable-tqdm"; exit 0; fi
  # like the real client: a call without these flags would not hold the matched settings, so it fails here
  for need in --ignore-eos --max-concurrency --random-output-len; do case " $* " in *" $need "*) ;; *) echo "mock vllm: missing $need" >&2; exit 2;; esac; done
  d=""; f=""; while [ $# -gt 0 ]; do case "$1" in --result-dir) d="$2";; --result-filename) f="$2";; esac; shift; done
  echo '{"completed":4,"total_output_tokens":512,"random_output_len":128,"median_ttft_ms":10.0,"median_tpot_ms":5.0,"output_throughput":999.0,"request_throughput":7.0}' >"$d/$f"; exit 0
fi
[ "$1" = serve ] && { mkdir -p /tmp/mock-vllm.$$/v1; echo '{"object":"list","data":[{"id":"mock-model"}]}' >/tmp/mock-vllm.$$/v1/models; cd /tmp/mock-vllm.$$ && port=""; for a in "$@"; do [ "$prev" = --port ] && port="$a"; prev="$a"; done; exec python3 -m http.server "$port" 2>/dev/null; }
M
echo '__version__ = "0.0-mock"' >"$T/py/vllm.py"
cat >"$T/venv/bin/python" <<M
#!/usr/bin/env bash
PYTHONPATH="$T/py" exec python3 "\$@"
M
chmod +x "$MOCK"/* "$T"/bin/* "$T"/venv/bin/*
mkdir -p "$T/stage"; cp "$HERE/pod-sweep.sh" "$HERE/summarize.py" "$HERE/oracle_fixture.py" "$T/stage/"
# a tiny budget for the last phases proves the skip path: MAX_TOTAL_S large here, skip tested separately below
( cd "$T" && env -i PATH="$MOCK:$T/venv/bin:$PATH" HOME="$T" OUT="$T/results" WORK="$T/work" BIN="$T/bin" ORACLE_ARCHIVE="$T/oracle-lane.tar.zst" WS_TAR="$T/ws.tar" WS_DIR="$T/ws" EX_DIR="$T/ex" REAL_RUNNER="$RUNNER" PORT_POOT=18080 PORT_VLLM=18000 PORT_LLAMA=18081 \
    VENV_VLLM="$T/venv" LLAMACPP_TAG=bMOCK BATCH_ISLS=512 CONCS=4 BATCH_PROMPTS=8 ISLS=128 \
    bash "$T/stage/pod-sweep.sh" ) >"$T/sweep.out" 2>&1
[ -e "$T/results/SWEEP_DONE" ] && ok "mock sweep reached SWEEP_DONE" || { bad "mock sweep did not finish"; tail -20 "$T/sweep.out"; }
cp "$T/results.tgz" "$T/main.tgz" 2>/dev/null; [ -s "$T/results.tgz" ] && ok "results.tgz written" || bad "no results.tgz"
for ph in oracle_receipt download_fp8 download_gguf poot_curve poot_curve_roofline poot_curve_profile poot_isl512_c4 vllm_isl128_c1 llamacpp_build llamacpp_bench_pp llamacpp_bench_tg llamacpp_isl512_c4; do
  awk -F'\t' -v p="$ph" '$1==p && $2==0 {f=1} END {exit !f}' "$T/results/phases.tsv" && ok "phase $ph rc=0" || bad "phase $ph missing/non-zero: $(grep "^$ph" "$T/results/phases.tsv")"
done
grep -q 'poot' "$T/results/summary.tsv" && grep -q 'llamacpp' "$T/results/summary.tsv" && grep -q 'vllm' "$T/results/summary.tsv" && ok "summary.tsv has poot, vllm and llamacpp rows" || bad "summary.tsv incomplete: $(cat "$T/results/summary.tsv")"
[ -s "$T/results/commands.tsv" ] && ok "commands.tsv records exact commands ($(wc -l <"$T/results/commands.tsv") lines)" || bad "commands.tsv empty"
awk -F'\t' '$1=="poot_curve" && $4+0>0 {f=1} END {exit !f}' "$T/results/phases.tsv" && ok "peak VRAM recorded for a phase" || bad "no peak VRAM recorded"
python3 -I - "$T/results/crates/poot-llm/tests/data/qwen38_27b_fp8_oracle_top5.json" <<'P' && ok "oracle fixture JSON valid (ids + logprobs, 5 each)" || bad "oracle fixture invalid/missing"
import json,sys
d=json.load(open(sys.argv[1]))
assert d["oracle_first_token_top5"]["ids"]==[11751,524,264,31586,1076] and len(d["oracle_first_token_top5"]["logprobs"])==5
assert d["prompt"]=="The capital of France is" and d["prompt_ids"] and d["generated_ids"][0]==11751
P
# order: oracle, then the vLLM and llama.cpp baselines, then the poot curve, then poot-serve rows last
python3 -I - "$T/results/phases.tsv" <<'P' && ok "phase order: oracle < vllm < llamacpp < poot curve < poot serve" || bad "phase order wrong"
import sys
names = [l.split("\t")[0] for l in open(sys.argv[1]).read().splitlines()[1:]]
order = ["oracle_receipt", "vllm_isl128_c1", "llamacpp_bench_pp", "poot_curve", "poot_isl512_c4"]
pos = [names.index(n) for n in order]
assert pos == sorted(pos), (order, pos)
P
grep -q -- "CLIENT_FLAGS+=(--save-detailed)" "$HERE/pod-sweep.sh" && grep -q 'ROW_CAP_S" -- ' "$HERE/pod-sweep.sh" && ok "serve rows keep per-request errors and carry the ROW_CAP_S cap" || bad "serve rows lack --save-detailed or the row cap"
# the staged oracle archive (stage.sh) must extract: nextest rejects an archive whose entries are not under target/
STAGED="${STAGE:-/tmp/claude-1000/poot-1184-stage}"
if [ -s "$STAGED/oracle-lane.tar.zst" ] && command -v cargo-nextest >/dev/null && [ -s "$STAGED/ws.tar" ]; then
  X=$(mktemp -d); mkdir -p "$X/ex" "$X/ws"; tar -C "$X/ws" -xf "$STAGED/ws.tar"
  lo=$(cd /tmp && cargo-nextest nextest list --archive-file "$STAGED/oracle-lane.tar.zst" --extract-to "$X/ex" --workspace-remap "$X/ws" --run-ignored all -E 'test(/qwen35_27b_fp8_receipt::ptx$/)' 2>&1)
  echo "$lo" | grep -q 'qwen35_27b_fp8_receipt::ptx' && ok "oracle-lane.tar.zst extracts and lists qwen35_27b_fp8_receipt::ptx" || bad "oracle archive: $(echo "$lo" | tail -3)"
  rm -rf "$X"
else echo "SKIP  oracle archive extraction (no staged archive / cargo-nextest)"; fi
# budget skip: MAX_TOTAL_S=0 skips every phase but still finishes
( cd "$T" && env -i PATH="$MOCK:$T/venv/bin:$PATH" HOME="$T" OUT="$T/results0" WORK="$T/work0" BIN="$T/bin" \
    VENV_VLLM="$T/venv" MAX_TOTAL_S=0 bash "$T/stage/pod-sweep.sh" ) >"$T/sweep0.out" 2>&1
grep -q 'skipped: budget' "$T/results0/phases.tsv" && ok "MAX_TOTAL_S=0 skips phases" || bad "budget cap did not skip"
# launcher extract round trip: a fake exec stdout carrying the tarball
{ echo noise; echo ===RESULTS-TGZ-BEGIN===; base64 -w0 "$T/main.tgz"; echo; echo ===RESULTS-TGZ-END===; } >"$T/exec.out"
"$HERE/launch.sh" extract "$T/exec.out" "$T/unpacked" >/dev/null 2>&1 && [ -s "$T/unpacked/results/phases.tsv" ] && [ -s "$T/unpacked/results/crates/poot-llm/tests/data/qwen38_27b_fp8_oracle_top5.json" ] && ok "launch.sh extract round-trips the tarball" || bad "extract failed"
pkill_pids=$(jobs -p); [ -n "$pkill_pids" ] && kill $pkill_pids 2>/dev/null
echo "workdir $T"
exit $fail
