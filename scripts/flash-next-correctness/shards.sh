#!/usr/bin/env bash
# Sourced by pod.sh. A split GGUF is loaded from its first shard; the shard count is the repository's to change
# (the Flash-Next UD-Q2_K_XL set went from three shards to four), so the name is found, not written down.

# first_shard DIR: the path of DIR's `*-00001-of-NNNNN.gguf`; exits 1 unless there is exactly one.
first_shard() {
    local found=("$1"/*-00001-of-[0-9][0-9][0-9][0-9][0-9].gguf)
    if [ "${#found[@]}" -ne 1 ] || [ ! -f "${found[0]}" ]; then
        echo "first_shard: expected one *-00001-of-NNNNN.gguf in $1, found: ${found[*]}" >&2
        return 1
    fi
    echo "${found[0]}"
}
