#!/usr/bin/env bash
# poot runner build: a plain `cargo build --release` (kernels are generated at runtime by kernelgen; no
# cargo-poot lowering). Runs on the local workstation inside `nix develop`, not the pod; scp the binary to
# the pod afterwards. The wgpu path needs a Vulkan-capable GPU; the PTX path
# uses cudarc dynamic loading, so the pod needs only the NVIDIA driver.
set -euo pipefail
cd "$(dirname "$0")/../runners/poot"

echo "== build poot-bench-runner (cargo build, current graph-IR engine) =="
# The runner is a member of the root workspace (one Cargo.lock), so its binary lands in the workspace target
# dir: CARGO_TARGET_DIR when set (card 395), else <repo>/target.
cargo build --release -p poot-bench-runner

BIN="${CARGO_TARGET_DIR:-$(git rev-parse --show-toplevel)/target}/release/poot-bench-runner"
echo "built: $BIN"
echo
echo "Next (manual):"
echo "  1. scp \"$BIN\" root@<pod-host>:-p <port>:/root/poot-bench-runner"
echo "  2. on the pod: patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 /root/poot-bench-runner"
echo "  3. on the pod: export POOT_BENCH_POOT_BIN=/root/poot-bench-runner  (the harness picks it up)"
echo "  Note: one binary runs on the ptx, rocm and wgpu backends; POOT_BENCH_BACKEND selects it (default ptx)."
echo "  On a pod the ptx backend needs libcuda (the pod driver) at runtime. Use warmup >= 2 to amortize the JIT."
