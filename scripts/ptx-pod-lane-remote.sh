#!/bin/bash
# Pod side of scripts/ptx-pod-lane: runs on the RunPod pod (ubuntu, bench image, no python/just/cargo), after the
# driver has uploaded everything to /root. Prints the results back between markers; the driver runs the gate.
set -u
cd /root
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
ldconfig -p | grep libcuda.so.1 || { echo "no libcuda.so.1: not a GPU pod"; exit 1; }
mkdir -p ex ws stage bank
tar -C ws -xzf src.tar.gz
tar -C / -xzf toolchain.tar.gz            # the toolchain closure at its /nix/store paths
rust=$(cat rust-path)
chmod +x cargo-nextest
export PATH=$rust/bin:$PATH LD_LIBRARY_PATH=$rust/lib
export POOT_REQUIRE_PTX=1 POOT_LLC=$(command -v llc || echo /usr/local/bin/llc)
$POOT_LLC --version | head -2
nx="/root/cargo-nextest nextest run --archive-file /root/archive.tar.zst --extract-to /root/ex --extract-overwrite --workspace-remap /root/ws --no-tests=fail --no-fail-fast"
cd /root/ws
# the real-checkpoint rows (--models): the checkpoints first, every file checked against the manifest's sha256
if [ -s /root/checkpoint-filter.txt ]; then
  mkdir -p /root/models
  tar -C /root/models -xzf /root/models.tar.gz
  while IFS=$'\t' read -r path source _; do
    [ "$source" = ship ] && continue
    mkdir -p "/root/models/$(dirname "$path")"
    curl -fsSL --retry 5 --retry-delay 5 -o "/root/models/$path" "$source" || echo "DOWNLOAD FAILED: $path"
  done < /root/models.tsv
  awk -F'\t' '{ print $3 "  " $1 }' /root/models.tsv > /root/models.sha256
  (cd /root/models && sha256sum -c /root/models.sha256) | tail -60; sha=${PIPESTATUS[0]}; echo "MODELS_SHA_EXIT=$sha"
  df -h /root | tail -1
  if [ "$sha" -eq 0 ]; then
    POOT_MODELS_DIR=/root/models POOT_REQUIRE_MODELS=1 $nx --profile device-checkpoint-ptx --run-ignored all \
      --test-threads=1 -E "$(cat /root/checkpoint-filter.txt)" > /root/checkpoint.log 2>&1; crc=$?
  else
    # A checkpoint that is not the measured one is never run: the results file stays empty, and the driver fails.
    echo "a checkpoint file does not match its sha256: no row runs" > /root/checkpoint.log; crc=1
    mkdir -p /root/ws/target/nextest/device-checkpoint-ptx; : > /root/ws/target/nextest/device-checkpoint-ptx/results.xml
  fi
  grep -E '^NARROW-GATE|POOT-SKIP' /root/checkpoint.log; tail -60 /root/checkpoint.log; echo "CHECKPOINT_EXIT=$crc"
  echo ===CHECKPOINT-RESULTS-BEGIN===
  gzip -c /root/ws/target/nextest/device-checkpoint-ptx/results.xml | base64 -w0
  echo
  echo ===CHECKPOINT-RESULTS-END===
fi
if [ -e /root/no-device-lane ]; then exit 0; fi
$nx --profile device-ptx -E "$(cat /root/selection.txt)" > /root/lane.log 2>&1; rc=$?
tail -400 /root/lane.log; echo "NEXTEST_EXIT=$rc"
# the kernels `ptx-check` runs, staged fresh for this run (the justfile's recipe does the same)
POOT_PTX_STAGE_DIR=/root/stage $nx -E 'binary_id(=poot-kernelgen) | binary_id(=poot-kernelgen::emit_nvptx)' 2>&1 | tail -15
echo "EMIT_EXIT=${PIPESTATUS[0]}"
/root/ex/target/debug/ptx-check /root/stage 2>&1 | tail -30
echo "PTXCHECK_EXIT=${PIPESTATUS[0]}"
if [ -s /root/bank-cases.txt ]; then
  export POOT_BANK_DEVICE="$(nvidia-smi --query-gpu=name --format=csv,noheader)" POOT_BANK_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  for pass in $(seq 1 "$(cat /root/bank-passes)"); do
    while read -r c; do
      /root/performance_bank --bank /root/ws/benchmarks/performance-bank/cases.json --case "$c" --out "/root/bank/$c.$pass.json" \
        > "/root/bank/$c.$pass.log" 2>&1; echo "bank $c pass $pass exit $?"
    done < /root/bank-cases.txt
  done
  echo ===BANK-BEGIN===; tar -C /root/bank -czf - . | base64 -w0; echo; echo ===BANK-END===
fi
echo ===RESULTS-BEGIN===
gzip -c /root/ws/target/nextest/device-ptx/results.xml | base64 -w0
echo
echo ===RESULTS-END===
