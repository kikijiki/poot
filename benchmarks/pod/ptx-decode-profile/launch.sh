#!/usr/bin/env bash
# POOT-1319 launcher (LOCAL). Starts ONE secure-cloud pod, runs pod.sh on it detached, keeps the pod for follow-ups.
#   launch.sh print | start | extract LOG DEST       (state the hourly price first; tear down BY ID afterwards)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${STAGE:-/tmp/claude-1000/poot-1319-stage}"
LOG="${LOG:-/tmp/claude-1000/poot-1319-exec.log}"
ORCH="${ORCH:-poot-orchestrator}"
GPU_TYPES="${GPU_TYPES:-NVIDIA RTX PRO 6000 Blackwell Server Edition,NVIDIA RTX PRO 6000 Blackwell Workstation Edition}"
WAIT_CEILING_S="${WAIT_CEILING_S:-14400}"
setup='set -e; mkdir -p /root/bin; mv /root/poot-bench-runner /root/bin/; chmod +x /root/bin/* /root/pod.sh; (command -v patchelf >/dev/null || (apt-get update -qq && apt-get install -y -qq patchelf)) >/dev/null 2>&1; patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 /root/bin/poot-bench-runner 2>/dev/null || true'
cmd='setsid nohup bash /root/pod.sh >/root/sweep.out 2>&1 </dev/null & n=0; while [ ! -e /root/results/SWEEP_DONE ] && [ "$n" -lt '"$WAIT_CEILING_S"' ]; do sleep 20; n=$((n+20)); done; echo ===RESULTS-TGZ-BEGIN===; base64 -w0 /root/results.tgz; echo; echo ===RESULTS-TGZ-END==='
args=(exec --image ghcr.io/kikijiki/poot-bench:latest --gpu-count 1 --gpu-types "$GPU_TYPES" --cloud SECURE
  --max-price-per-hour "${MAX_PRICE:-3.0}" --container-disk-gb 200 --provision-timeout-s 1800 --ssh-timeout-s 900
  --keep-warm "${KEEP_WARM_MIN:-240}" --keep --max-cmd "${MAX_CMD:-5h}" --pod-ttl "${POD_TTL_MIN:-480}"
  --upload "$STAGE/poot-bench-runner:/root/poot-bench-runner" --upload "$HERE/pod.sh:/root/pod.sh"
  --setup "$setup" --cmd "$cmd")
case "${1:-}" in
  print) printf '%q ' "$ORCH" "${args[@]}"; echo ;;
  start)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"
    [ -s "$STAGE/poot-bench-runner" ] || { echo "missing $STAGE/poot-bench-runner: run stage.sh" >&2; exit 1; }
    setsid nohup "$ORCH" "${args[@]}" >"$LOG" 2>"$LOG.err" </dev/null &
    echo "exec started, pid $!; pod id: grep 'pod .* created' $LOG" ;;
  extract)
    log="${2:?LOG}"; dest="${3:?DEST}"; mkdir -p "$dest"
    sed -n '/^===RESULTS-TGZ-BEGIN===$/,/^===RESULTS-TGZ-END===$/p' "$log" | sed '1d;$d' | tr -d '\n' | base64 -d >"$dest/results.tgz"
    tar -C "$dest" -xzf "$dest/results.tgz" && echo "unpacked into $dest/results" ;;
  *) echo "usage: $0 print|start|extract LOG DEST" >&2; exit 2 ;;
esac
