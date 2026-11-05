#!/usr/bin/env bash
# Build the patched wgpu-hal the workspace Cargo.toml's [patch.crates-io] points at (patches/wgpu-hal/README.md, POOT-1464).
#
#   scripts/wgpu-hal-patched.sh          unpack the pinned crate into .patched/wgpu-hal and apply the patch (no-op when current)
#   scripts/wgpu-hal-patched.sh check    assert the directory is current and that cargo resolves wgpu-hal to it
#
# Fails loudly: the archive must match the pinned SHA-256, and the patch is applied by `git apply`, which has no
# fuzz (every context line must match). Any failure leaves no half-built directory behind.
set -euo pipefail

version=29.0.4
sha256=97ace1c17727311c22a46e4e3faf56ea6de81af99dcc839bdfb54857b94d448d

root=$(cd "$(dirname "$0")/.." && pwd)
patch_file="$root/patches/wgpu-hal/0001-pass-begin-timestamp-at-top-of-pipe.patch"
out="$root/.patched/wgpu-hal"
stamp="$sha256 $(sha256sum "$patch_file" | cut -d' ' -f1)"

current() { [ -f "$out/.poot-patched" ] && [ "$(cat "$out/.poot-patched")" = "$stamp" ]; }

if [ "${1:-}" = check ]; then
  current || { echo "wgpu-hal-patched: $out is missing or stale; run scripts/wgpu-hal-patched.sh" >&2; exit 1; }
  grep -q "wgpu-hal = { path = \".patched/wgpu-hal\" }" "$root/Cargo.toml" ||
    { echo "wgpu-hal-patched: Cargo.toml has no [patch.crates-io] entry for .patched/wgpu-hal" >&2; exit 1; }
  locked=$(awk '$0 == "name = \"wgpu-hal\"" { getline; print $3 }' "$root/Cargo.lock" | tr -d '"')
  [ "$locked" = "$version" ] ||
    { echo "wgpu-hal-patched: Cargo.lock has wgpu-hal '$locked', the patch is pinned to $version" >&2; exit 1; }
  resolved=$(cd "$root" && cargo tree --offline -p wgpu-hal --depth 0 2>&1)
  grep -q "wgpu-hal v$version ($out)" <<<"$resolved" ||
    { echo "wgpu-hal-patched: cargo does not resolve wgpu-hal to the patched $out: $resolved" >&2; exit 1; }
  grep -q 'vk::PipelineStageFlags::TOP_OF_PIPE' "$out/src/vulkan/command.rs" ||
    { echo "wgpu-hal-patched: the patched source lacks the TOP_OF_PIPE pass-begin stamp" >&2; exit 1; }
  echo "wgpu-hal-patched: wgpu-hal $version resolves to $out with the pass-begin stamp at TOP_OF_PIPE"
  exit 0
fi

current && exit 0
rm -rf "$out" # a stale directory must not outlive a failed rebuild: cargo then fails on the missing path

work=$(mktemp -d "$root/.patched.XXXXXX" 2>/dev/null || mktemp -d)
trap 'rm -rf "$work"' EXIT
archive=
for candidate in "${CARGO_HOME:-$HOME/.cargo}"/registry/cache/*/wgpu-hal-$version.crate; do
  [ -f "$candidate" ] && archive=$candidate && break
done
if [ -z "$archive" ]; then
  archive="$work/wgpu-hal-$version.crate"
  curl --fail --silent --show-error --location --retry 3 \
    "https://static.crates.io/crates/wgpu-hal/wgpu-hal-$version.crate" -o "$archive"
fi
actual=$(sha256sum "$archive" | cut -d' ' -f1)
[ "$actual" = "$sha256" ] ||
  { echo "wgpu-hal-patched: $archive has SHA-256 $actual, expected $sha256" >&2; exit 1; }

tar -xzf "$archive" -C "$work"
(cd "$work/wgpu-hal-$version" && git apply --check "$patch_file" && git apply "$patch_file")
echo "$stamp" >"$work/wgpu-hal-$version/.poot-patched"
mkdir -p "$root/.patched"
mv "$work/wgpu-hal-$version" "$out"
echo "wgpu-hal-patched: wgpu-hal $version patched into $out"
