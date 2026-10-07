#!/usr/bin/env bash
# Fails when the website cites a name the code no longer has. Public pages are the user-facing truth, and cards that
# delete or rename an API kept leaving the old name in prose and examples. The check runs from the site's side, so it
# needs no list of deleted names: anything the pages cite in backticks or <code> (identifiers, CamelCase type names,
# source files, POOT_* variables, crate names,
# repository paths, `just` recipes, command-line flags) must still exist in the tracked source. Identifiers must occur in
# code, not only in a comment: a name that survives in a `//` or `///` line is a leftover.
#
# A page may deliberately name something that does not exist (a setting the server does not read yet). List it in
# scripts/site-stale-names.allow, one name per line, with a reason in a comment above it.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

allow=scripts/site-stale-names.allow
src=(crates scripts justfile Cargo.toml flake.nix nix README.md benchmarks)
site=(website/docs website/src)
rc=0

allowed() { grep -qxF -- "$1" <(grep -v '^#' "$allow") ; }

# report KIND NAME WHERE: print one stale name with the page lines that cite it.
report() {
    allowed "$2" && return
    echo "stale $1 \`$2\` (no longer in the code), cited at:"
    git grep -nIF -- "$2" -- "${site[@]}" | sed 's/^/    /'
    rc=1
}

# Code corpus: every word in tracked source OUTSIDE comments. A name that survives only in a `//`, `///`, `//!` or
# block comment (or a `#` comment in toml, shell and just files) is a leftover, not code, so it does not keep a page
# claim alive. Rust files lose their comments with perl (block comments first; a `//` right after `:` is a URL scheme,
# not a comment); other text sources lose whole `#` comment lines.
corpus=$(mktemp)
stripped=$(mktemp)
trap 'rm -f "$corpus" "$stripped"' EXIT
{
    git ls-files -z -- 'crates/*.rs' 'benchmarks/*.rs' 'website/src/*.ts' 'website/src/*.tsx' 'website/*.ts' |
        xargs -0 perl -0777 -pe 's{/\*.*?\*/}{}gs; s{(?<!:)//[^\n]*}{}g'
    git ls-files -z -- crates scripts justfile Cargo.toml flake.nix nix benchmarks ':!*.rs' ':!*.json' ':!*.trace' ':!*.tsv' ':!*.md' |
        xargs -0 grep -hIv '^[[:space:]]*#' 2>/dev/null
} >"$stripped"
grep -oE '[A-Za-z_][A-Za-z0-9_]*' "$stripped" | sort -u >"$corpus"

in_code() { grep -qxF -- "$1" "$corpus"; }
# Tracked file names, for `kernel_name.rs`-style citations.
files=$(git ls-files | sed 's|.*/||' | sort -u)
stems=$(git ls-files -- '*.rs' | sed -E 's|.*/||; s|\.rs$||' | sort -u)

# Names cited on the site: backticked or <code> text in docs, components and static pages, plus bare CamelCase type
# names in prose (two or more capital humps, e.g. `FlashAttention`), which is how renamed types linger unquoted.
cited=$(mktemp)
trap 'rm -f "$corpus" "$stripped" "$cited"' EXIT
{
    git grep -hIoE '`[A-Za-z_][A-Za-z0-9_:.]*(\(\))?`' -- "${site[@]}" website/static | tr -d '`'
    git grep -hIoE '<code>[A-Za-z_][A-Za-z0-9_:.]*(\(\))?</code>' -- "${site[@]}" website/static | sed -E 's/<\/?code>//g'
    git grep -hIoE '\b[A-Z][a-z0-9]+([A-Z][a-z0-9]*)+\b' -- "${site[@]}" website/static
} | sed 's/()$//' | sort -u >"$cited"
checked=0

while read -r id; do
    case $id in POOT_*) continue ;; esac # variables have their own check below
    # An example, test or binary is cited by its file stem (`trace_and_fuse` runs `examples/trace_and_fuse.rs`).
    grep -qxF -- "$id" <<<"$stems" && { checked=$((checked + 1)); continue; }
    if [[ $id == *.rs ]]; then
        checked=$((checked + 1))
        grep -qxF -- "${id##*/}" <<<"$files" || report "source file" "$id"
        continue
    fi
    [[ $id == *.* ]] && continue # other file names and prose
    # `a::b::c`: every segment must be code, so a deleted module fails even when its leaf survives elsewhere. Segments
    # are checked one by one: modules are declared (`pub mod analysis;`), never written as `analysis::item` pairs.
    IFS=: read -ra parts <<<"${id//::/:}"
    for part in "${parts[@]}"; do
        # Short single words are prose; every segment of a path is a claim.
        [ "${#parts[@]}" -gt 1 ] || [ "${#part}" -ge 4 ] || continue
        checked=$((checked + 1))
        in_code "$part" || { report identifier "$id"; break; }
    done
done <"$cited"
echo "site-stale-names: $checked identifiers and file names checked against code"

# Environment variables: the full name must be read somewhere.
while read -r v; do
    git grep -qIw -- "$v" -- "${src[@]}" || report variable "$v"
done < <(git grep -hIoE '\bPOOT_[A-Z0-9_]*[A-Z0-9]\b' -- "${site[@]}" | sort -u)

# Crates and binaries.
while read -r c; do
    [ -d "crates/$c" ] || grep -qxF "name = \"$c\"" crates/*/Cargo.toml || report crate "$c"
done < <(git grep -hIoE '\bpoot-[a-z0-9]+(-[a-z0-9]+)*\b' -- "${site[@]}" | sort -u)

# Repository paths.
while read -r p; do
    [ -e "$p" ] || report path "$p"
done < <(git grep -hIoE '\b(crates|scripts|benchmarks|nix)/[A-Za-z0-9_./-]*[A-Za-z0-9_]' -- "${site[@]}" | sort -u)

# `just` recipes.
while read -r r; do
    grep -qE "^$r[ :]" justfile || report "just recipe" "just $r"
done < <(git grep -hIoE '`just [a-z][a-z0-9-]+' -- "${site[@]}" | sed 's/^`just //' | sort -u)

# Long command-line flags.
while read -r f; do
    git grep -qIF -- "$f" -- "${src[@]}" || report flag "$f"
done < <(git grep -hIoE '(^|[ `])--[a-z][a-z0-9-]+' -- website/docs | tr -d ' `' | sort -u)

exit $rc
