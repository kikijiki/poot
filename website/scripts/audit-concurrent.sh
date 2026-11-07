#!/usr/bin/env bash
# Two audits at once must both pass: lanes and the gate run them at the same time (POOT-1511). The self-test is the cheap
# audit (its own server and browser); a fixed port in either makes the second run fail to start.
set -uo pipefail
cd "$(dirname "$0")/.."
node scripts/audit-pages.mjs --self-test > /tmp/poot-audit-a.$$ 2>&1 &
first=$!
node scripts/audit-pages.mjs --self-test > /tmp/poot-audit-b.$$ 2>&1 &
second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
if [ "$rc" != 0 ]; then
  echo "concurrent audits: FAIL" >&2
  tail -n 20 /tmp/poot-audit-a.$$ /tmp/poot-audit-b.$$ >&2
else
  echo "concurrent audits: both passed"
fi
rm -f /tmp/poot-audit-a.$$ /tmp/poot-audit-b.$$
exit "$rc"
