#!/usr/bin/env bash
# POOT-1184 launcher (LOCAL). Starts ONE pod, runs pod-sweep.sh on it detached, and returns the results.
#
#   launch.sh print                 print the exact poot-orchestrator command (no pod, no key needed)
#   launch.sh start                 setsid-nohup the exec; log -> $LOG; needs RUNPOD_API_KEY and the staged binaries
#   launch.sh extract LOG DEST      unpack the results tarball the exec printed into DEST
#
# The orchestrator creates the pod ON `start`; the orchestrator (a human/agent) states the hourly price first
# (POOT-ADR-0092): H100 SXM secure $3.99/h, fallback A100 SXM secure $1.79/h (the GPU list is tried in order and
# the sweep records which card it got). Tear down BY ID afterwards (the pod id is in $LOG: "pod <id> created").
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${STAGE:-/tmp/claude-1000/poot-1184-stage}"     # stage.sh output: poot-bench-runner, poot-serve (patchelf'd)
LOG="${LOG:-/tmp/claude-1000/poot-1184-exec.log}"
ORCH="${ORCH:-poot-orchestrator}"
GPU_TYPES="${GPU_TYPES:-NVIDIA H100 80GB HBM3,NVIDIA A100-SXM4-80GB}"
# vLLM 0.22.1 in the image is a cu130 wheel: it needs a host driver that supports CUDA 13.0.
MIN_CUDA="${MIN_CUDA:-13.0}"
KEEP_WARM_MIN="${KEEP_WARM_MIN:-120}"
WAIT_CEILING_S="${WAIT_CEILING_S:-21600}"

# Flat uploads; --setup moves them into place (nothing is uploaded to a nested path that may not exist yet).
setup='set -e; mkdir -p /root/bin; mv /root/poot-bench-runner /root/poot-serve /root/bin/; chmod +x /root/bin/* /root/pod-sweep.sh; (command -v patchelf >/dev/null || (apt-get update -qq && apt-get install -y -qq patchelf)) >/dev/null 2>&1; patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 /root/bin/poot-bench-runner /root/bin/poot-serve 2>/dev/null || true'
# The sweep runs detached on the pod (an SSH drop does not stop it); this command only waits for it and prints the
# results tarball between markers, because the exec has no download path.
cmd='setsid nohup bash /root/pod-sweep.sh >/root/sweep.out 2>&1 </dev/null & n=0; while [ ! -e /root/results/SWEEP_DONE ] && [ "$n" -lt '"$WAIT_CEILING_S"' ]; do sleep 20; n=$((n+20)); done; echo ===SUMMARY-BEGIN===; cat /root/results/summary.tsv; echo ===SUMMARY-END===; echo ===RESULTS-TGZ-BEGIN===; base64 -w0 /root/results.tgz; echo; echo ===RESULTS-TGZ-END==='

args=(exec
  --image ghcr.io/kikijiki/poot-bench:latest
  --gpu-count 1 --gpu-types "$GPU_TYPES" --cloud SECURE --min-cuda-version "$MIN_CUDA"
  --container-disk-gb 200 --provision-timeout-s 1800 --ssh-timeout-s 900
  --keep-warm "$KEEP_WARM_MIN" --keep
  --upload "$STAGE/poot-bench-runner:/root/poot-bench-runner"
  --upload "$STAGE/poot-serve:/root/poot-serve"
  --upload "$HERE/pod-sweep.sh:/root/pod-sweep.sh"
  --upload "$HERE/summarize.py:/root/summarize.py"
  --upload "$HERE/oracle_fixture.py:/root/oracle_fixture.py"
  --upload "$STAGE/oracle-lane.tar.zst:/root/oracle-lane.tar.zst"
  --upload "$STAGE/ws.tar:/root/ws.tar"
  --setup "$setup" --cmd "$cmd")

case "${1:-}" in
  print) printf '%q ' "$ORCH" "${args[@]}"; echo ;;
  start)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"
    for f in poot-bench-runner poot-serve oracle-lane.tar.zst ws.tar; do [ -s "$STAGE/$f" ] || { echo "missing $STAGE/$f: run stage.sh" >&2; exit 1; }; done
    setsid nohup "$ORCH" "${args[@]}" >"$LOG" 2>"$LOG.err" </dev/null &
    echo "exec started, pid $!; pod id: grep 'pod .* created' $LOG (also in ~/.local/state/poot-orchestrator/logs/exec-*.log)" ;;
  extract)
    log="${2:?LOG}"; dest="${3:?DEST}"; mkdir -p "$dest"
    sed -n '/^===RESULTS-TGZ-BEGIN===$/,/^===RESULTS-TGZ-END===$/p' "$log" | sed '1d;$d' | tr -d '\n' | base64 -d >"$dest/results.tgz"
    tar -C "$dest" -xzf "$dest/results.tgz" && echo "unpacked into $dest/results" ;;
  *) echo "usage: $0 print|start|extract LOG DEST" >&2; exit 2 ;;
esac
