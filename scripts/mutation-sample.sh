#!/usr/bin/env bash
# The cargo-mutants sample behind scripts/mutation-sample.tsv (POOT-1221, R2-C5-009).
#
# Mutates the loader's header parsers against the whole poot-load test suite: all of the GGUF header
# parser (gguf/format.rs, in two shards of about 5 minutes) and the range, shard-name and safetensors
# header checks of gguf.rs, dtype.rs and safetensors.rs (one run of about 5 minutes). Then it compares the
# surviving mutants with the committed bound:
#
#   scripts/mutation-sample.sh [OUT_DIR]      run the sample (nightly or by hand; not part of `just test`)
#
# cargo-mutants comes from nixpkgs (`nix shell`); it is not in the dev shell. The run goes through
# poot-cargo like every cargo command. Exit status is the checker's: non-zero when survivors rose above
# the committed count.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-target/mutation-sample}"
rm -rf "$out"; mkdir -p "$out/shard-0" "$out/shard-1" "$out/ranges"
for shard in 0 1; do
  /home/kikijiki/.local/bin/poot-cargo nix develop -c nix shell nixpkgs#cargo-mutants -c \
    cargo mutants -p poot-load -f crates/poot-load/src/gguf/format.rs \
    --shard "$shard/2" --timeout 120 --no-times --output "$out/shard-$shard" || true
done
/home/kikijiki/.local/bin/poot-cargo nix develop -c nix shell nixpkgs#cargo-mutants -c \
  cargo mutants -p poot-load -f crates/poot-load/src/safetensors.rs -f crates/poot-load/src/dtype.rs \
  -f crates/poot-load/src/gguf.rs \
  --re 'parse_tensor_header_entry|plan_header_entries|plan_file|parse_safetensors_header|checked_header_end|tensor_range|shard_paths|split_name|gguf_plan|is_plain_file_name|shard_files' \
  --timeout 120 --no-times --output "$out/ranges" || true
scripts/check-mutation-sample.py "$out/shard-0" "$out/shard-1" "$out/ranges"
