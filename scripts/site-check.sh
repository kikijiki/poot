#!/usr/bin/env bash
# Every check the website must pass: TypeScript typecheck, component tests, the benchmark grid, the navbar, a production
# build (broken links), names the code no longer has, the develop/ Rust snippets, the left-edge bar rule, internal card
# numbers, NPU summaries, prose length and the ascii rule over tracked text sources. git grep ignores untracked
# build/vendor files; bun.lock is generated metadata.
#
# `just site-check` and the cd-docs workflow run this one script, so the published site cannot pass a smaller set than
# the local gate (POOT-1467). The workflow passes --no-toolchain: the Rust snippets are `cargo check`ed against the
# workspace, which needs the poot toolchain (`nix develop`) that a hosted runner has not. Run from anywhere in the checkout,
# after `bun install --frozen-lockfile` in website/. Reports every stage even if one fails; non-zero exit if any did.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

toolchain=1
case "${1:-}" in
  "") ;;
  --no-toolchain) toolchain=0 ;;
  *)
    echo "usage: $0 [--no-toolchain]" >&2
    exit 2
    ;;
esac

rc=0
echo "-- typecheck --"
(cd website && bun run typecheck) || rc=1
echo "-- component tests (the Performance page's selection logic, the navbar check) --"
(cd website && bun test) || rc=1
echo "-- benchmark grid (every required cell is verified, pending or has a reason) --"
python3 benchmarks/harness/grid.py check || rc=1
echo "-- navbar (every top-level docs section has a navbar entry and its sidebar, every item a doc) --"
(cd website && bun scripts/check-navbar.mjs) || rc=1
echo "-- build (broken-link check) --"
(cd website && bun run build) || rc=1
echo "-- stale names (identifiers, variables, crates, paths, recipes, flags the code no longer has) --"
scripts/site-stale-names.sh || rc=1
if [ "$toolchain" = 1 ]; then
  echo "-- rust snippets (the develop/ pages' code compiles) --"
  scripts/site-rust-snippets.sh || rc=1
else
  echo "-- rust snippets: NOT RUN (--no-toolchain); \`just site-check\` runs them --"
fi
echo "-- edge bars (no coloured bar on the left edge of a card, callout, alert or active item) --"
scripts/site-edge-bars.sh || rc=1
echo "-- card numbers (the site never names an internal card or board id) --"
scripts/site-card-numbers.sh || rc=1
echo "-- NPU summaries (what an NPU runs is stated on architecture/npu.mdx only) --"
scripts/site-npu-summaries.sh || rc=1
echo "-- readability (no prose paragraph over the word limit) --"
scripts/site-readability.sh || rc=1
echo "-- ascii (no em-dashes / non-ascii) --"
ascii_rc=0
git grep -nIP '[^\x00-\x7F]' -- website ':!website/bun.lock' || ascii_rc=$?
if [ "$ascii_rc" -eq 0 ]; then
  echo "non-ASCII characters found (see above) - replace with ASCII equivalents"
  rc=1
elif [ "$ascii_rc" -ne 1 ]; then
  echo "ASCII scan failed with status $ascii_rc"
  rc=1
fi
exit $rc
