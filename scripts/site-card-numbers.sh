#!/usr/bin/env bash
# Fails when a website page names an internal card, spec or board id. The site holds user-facing truth only.
# Run by `just site-check` and by the cd-docs workflow.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

rc=0
git grep -nIiP '\b(cards? #?[0-9]{3,4}[a-d]?|POOT-[0-9]+|POOT-ADR-[0-9]+)\b' -- website/docs website/src website/diagrams || rc=$?
case $rc in
    0) echo "internal card numbers found (see above) - the site holds user-facing truth only"; exit 1 ;;
    1) exit 0 ;;
    *) echo "card-number scan failed with status $rc"; exit 1 ;;
esac
