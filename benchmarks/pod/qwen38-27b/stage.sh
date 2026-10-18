#!/usr/bin/env bash
# POOT-1184: build the two pod binaries LOCALLY (release, jobs capped, memory capped) and stage them patchelf'd.
# Run from the repo root inside the worktree. Output: $STAGE/{poot-bench-runner,poot-serve}.
set -euo pipefail
STAGE="${STAGE:-/tmp/claude-1000/poot-1184-stage}"
mkdir -p "$STAGE"
RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 systemd-run --user --scope -q -p MemoryMax=20G -p TasksMax=4000 \
  /home/kikijiki/.local/bin/poot-cargo nix develop -c bash -c '
    set -euo pipefail
    cargo build --release -j 4 -p poot-bench-runner -p poot-serve
    cp "$CARGO_TARGET_DIR/release/poot-bench-runner" "$CARGO_TARGET_DIR/release/poot-serve" "'"$STAGE"'/"
    patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath "'"$STAGE"'/poot-bench-runner" "'"$STAGE"'/poot-serve"'
git rev-parse HEAD >"$STAGE/BUILD_SHA"


# The oracle row (qwen35_27b_fp8_receipt) is an #[ignore]d test: ship it as a patchelf'd nextest archive plus the
# workspace tar the tests remap to (same recipe as the PTX device lane).
RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 systemd-run --user --scope -q -p MemoryMax=20G -p TasksMax=4000 \
  /home/kikijiki/.local/bin/poot-cargo nix develop -c bash -c '
    set -euo pipefail
    cargo nextest archive --release -p poot-llm --lib --archive-file "'"$STAGE"'/oracle-lane.raw.tar.zst"
    rm -rf "'"$STAGE"'/x" && mkdir "'"$STAGE"'/x" && cd "'"$STAGE"'/x" && zstd -dc ../oracle-lane.raw.tar.zst | tar x
    find . -type f | while IFS= read -r f; do if head -c4 "$f" | grep -q ELF; then
      patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath "$f" 2>/dev/null || patchelf --remove-rpath "$f" 2>/dev/null || true; fi; done
    tar -c target | zstd -T0 -q -f -o ../oracle-lane.tar.zst'
git archive HEAD -o "$STAGE/ws.tar"

ls -l "$STAGE"
