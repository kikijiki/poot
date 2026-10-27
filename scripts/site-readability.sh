#!/usr/bin/env bash
# Fails when a website page has a prose paragraph over MAX_WORDS words (the Readability section of the website rules,
# Backstage). Tables, lists, code blocks, details blocks, headings, footnote definitions and JSX lines do not count.
# Also fails when a text code block holds an arrow chain (2+ ` -> ` on a line): a pipeline is a list, table or diagram.
# Also fails when a line inside a code block is wider than MAX_CODE_COLS (a code block must not scroll sideways).
# A command line that cannot be broken goes in CODE_WIDTH_EXCEPTIONS (file, with the reason beside it).
# Run by `just site-check` and by the cd-docs workflow.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

MAX_WORDS=120
MAX_CODE_COLS=86
CODE_WIDTH_EXCEPTIONS=""

git ls-files 'website/docs/*.md' 'website/docs/*.mdx' | xargs awk -v max="$MAX_WORDS" -v maxcols="$MAX_CODE_COLS" -v exc="$CODE_WIDTH_EXCEPTIONS" '
function flush() {
    if (words > max) {
        printf "%s:%d: prose paragraph of %d words (max %d): split it, or move the detail to a table, list or details block\n", FILENAME, start, words, max
        bad = 1
    }
    words = 0; skip = 0
}
FNR == 1 { flush(); fence = 0; details = 0; front = 0; start = 0 }
FNR == 1 && $0 == "---" { front = 1; next }
front { if ($0 == "---") front = 0; next }
/^[[:space:]]*(```|~~~)/ {
    flush()
    fence = !fence
    if (fence) { info = $0; gsub(/^[[:space:]]*(```|~~~)[[:space:]]*/, "", info); plain = (info == "" || info == "text") }
    next
}
fence {
    if (length($0) > maxcols && index(" " exc " ", " " FILENAME " ") == 0) {
        printf "%s:%d: code line of %d columns (max %d): break it, or move the comment above the line\n", FILENAME, FNR, length($0), maxcols
        bad = 1
    }
    if (plain && gsub(/ -> /, "&") >= 2) {
        printf "%s:%d: arrow chain in a text code block: use a table, a numbered list or a diagram component\n", FILENAME, FNR
        bad = 1
    }
    next
}
/^[[:space:]]*<details/ { flush(); details++; next }
/^[[:space:]]*<\/details/ { flush(); if (details > 0) details--; next }
/^[[:space:]]*$/ { flush(); next }
{
    if (words == 0 && !skip) {
        # first line of a block decides whether it is prose
        if (details > 0 || $0 ~ /^([[:space:]]|#|\||<|>|:::|[-*+][[:space:]]|[0-9]+[.)][[:space:]]|\[\^|import[[:space:]]|export[[:space:]]|\{)/) skip = 1
        else start = FNR
    }
    if (!skip) words += NF
}
END { flush(); exit bad }
' || exit 1
