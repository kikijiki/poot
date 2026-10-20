#!/usr/bin/env bash
# Fails when a website page other than the NPU page and the generated matrix summarizes what an NPU runs
# (an op list such as "add, sub, mul", or "a few F32 programs"). Every such statement lives in
# architecture/npu.mdx; other pages link to it. Run by `just site-check`.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

hits=$(git ls-files -z 'website/docs/*.md' 'website/docs/*.mdx' \
    ':!website/docs/architecture/npu.mdx' ':!website/docs/reference/feature-matrix.md' |
    xargs -0 perl -0777 -ne '
        for my $para (split /\n\s*\n/) {
            next unless $para =~ /\b(NPUs?|XDNA2?)\b/;
            next unless $para =~ /\b(add|sub|mul)\b\W{1,6}(sub|mul)\b|\bfew F(16|32)\b|\bF(16|32) (elementwise|graphs|programs|ops)\b|\b(runs|run) only\b/i;
            (my $one = $para) =~ s/\n/ /g;
            print "$ARGV: $one\n";
        }
    ' 2>&1)
if [ -n "$hits" ]; then
    echo "$hits"
    echo "NPU capability summary outside architecture/npu.mdx - link to it instead"
    exit 1
fi
