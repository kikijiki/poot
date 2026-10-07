#!/usr/bin/env bash
# Prove that cargo's freshness check, as configured by this repository's .cargo/config.toml, is decided by
# source content and not by mtime. Several worktrees share one CARGO_TARGET_DIR (poot-cargo slots), and cargo's
# unit hash for path packages omits the worktree path, so a unit built from worktree A's source is found again
# when worktree B builds. Only the fingerprint method keeps B from accepting it (POOT-1037: the 2026-10-06 nightly made
# `-Zchecksum-freshness` alone a silent no-op; `build.fingerprint = "content"` is what turns it on).
#
# Two throwaway workspaces share one target dir and differ in one source line: `q::W` has a variant that `e`
# matches exhaustively. Building `e` in the second must fail with E0004; reusing the first's `q` is the bug.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

make_workspace() {
  local dir=$1 variants=$2
  mkdir -p "$dir/.cargo" "$dir/q/src" "$dir/e/src"
  cp "$root/.cargo/config.toml" "$dir/.cargo/config.toml"
  cp "$root/rust-toolchain.toml" "$dir/rust-toolchain.toml"
  printf '[workspace]\nmembers = ["q", "e"]\nresolver = "3"\n' >"$dir/Cargo.toml"
  printf '[package]\nname = "q"\nversion = "0.0.0"\nedition = "2024"\n' >"$dir/q/Cargo.toml"
  printf '[package]\nname = "e"\nversion = "0.0.0"\nedition = "2024"\n[dependencies]\nq = { path = "../q" }\n' >"$dir/e/Cargo.toml"
  printf 'pub enum W { %s }\n' "$variants" >"$dir/q/src/lib.rs"
  printf 'pub fn f(w: &q::W) -> u8 { match w { q::W::A => 1, q::W::B => 2 } }\n' >"$dir/e/src/lib.rs"
}

make_workspace "$work/a" "A, B"
make_workspace "$work/b" "A, B, C"

build() {
  (cd "$1" && RUSTC_WRAPPER= CARGO_TARGET_DIR="$work/target" cargo check -p e 2>&1)
}

build "$work/a" >/dev/null
if out=$(build "$work/b"); then
  echo "FAIL: worktree b reused worktree a's stale unit (cargo is not on content freshness)" >&2
  echo "$out" >&2
  exit 1
fi
if ! grep -q 'E0004' <<<"$out"; then
  echo "FAIL: worktree b failed, but not with the expected E0004:" >&2
  echo "$out" >&2
  exit 1
fi
echo "cargo freshness: a unit built from another worktree's source is rebuilt (E0004 seen in worktree b)"
