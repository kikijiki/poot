#!/usr/bin/env bash
# POOT-1319: build poot-bench-runner LOCALLY (release, jobs capped, memory capped) and stage it patchelf'd.
# Run from the repo root inside the worktree. Output: $STAGE/poot-bench-runner and BUILD_SHA.
set -euo pipefail
STAGE="${STAGE:-/tmp/claude-1000/poot-1319-stage}"
mkdir -p "$STAGE"
exec 9>>/tmp/poot-heavy.lock; flock 9
RUSTC_WRAPPER= CARGO_BUILD_JOBS=8 systemd-run --user --scope -q --unit="poot-1319-stage-$$" -p MemoryMax=24G -p TasksMax=4000 \
  poot-cargo nix develop -c bash -c '
    set -euo pipefail
    cargo build --release -j 8 -p poot-bench-runner
    cp "${CARGO_TARGET_DIR:-target}/release/poot-bench-runner" "'"$STAGE"'/"
    patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath "'"$STAGE"'/poot-bench-runner"'
git rev-parse HEAD >"$STAGE/BUILD_SHA"
ls -l "$STAGE"
