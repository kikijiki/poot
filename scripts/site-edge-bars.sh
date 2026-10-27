#!/usr/bin/env bash
# Fails when the site's or the dashboard's stylesheets draw a coloured bar on the left edge of an element (a thick
# border-left or border-inline-start, an inset box-shadow, or an Infima left-border width variable above 1px). Cards,
# callouts, alerts and active items use a tinted background, an icon or a label instead (the website rules, Backstage). A border
# trick that draws a triangle (a `transparent` side) is not a bar and is skipped.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

bars=$(git grep -nIE '(border-(left|inline-start)(-width)?|border-left-width)[[:space:]]*:[[:space:]]*([2-9]|[1-9][0-9])px|box-shadow:[[:space:]]*inset[[:space:]]+-?([2-9]|[1-9][0-9])px[[:space:]]+(0|[0-9]+px)?[[:space:]]*(0[[:space:]]+)?(var|#|rgb|[a-z])' -- 'website/src/*.css' 'crates/poot-orchestrator/ui/src/*.css' |
    grep -v transparent)
if [ -n "$bars" ]; then
    echo "left-edge accent bars (use a tinted background, an icon or a label):"
    echo "$bars" | sed 's/^/    /'
    exit 1
fi
