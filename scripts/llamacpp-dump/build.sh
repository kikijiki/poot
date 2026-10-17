#!/usr/bin/env bash
# Build llamacpp-dump against the llama.cpp source the Nix `llama-cpp` on PATH was built from (fixed-output,
# hash-checked by nixpkgs: never a fresh clone). Prints the path of the binary.
#
#   scripts/llamacpp-dump/build.sh [out-dir]    # default: ${XDG_CACHE_HOME:-~/.cache}/poot/llamacpp-dump
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=${1:-${XDG_CACHE_HOME:-$HOME/.cache}/poot/llamacpp-dump}
mkdir -p "$out"

package=$(dirname "$(dirname "$(readlink -f "$(command -v llama-perplexity)")")")
deriver=$(nix-store --query --deriver "$package")
source_drv=$(nix-store --query --references "$deriver" | grep -- '-source\.drv$' | head -n1)
source=$(nix-store --realise "$source_drv" 2>/dev/null)
echo "llama.cpp source: $source" >&2

nix shell nixpkgs#cmake nixpkgs#ninja -c bash -euo pipefail -c '
    cmake -S "$1" -B "$2/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF >&2
    cmake --build "$2/build" --target llama -j "$(nproc)" >&2
' _ "$source" "$out"

g++ -O1 -std=c++17 "$here/dump.cpp" -I"$source/include" -I"$source/ggml/include" \
    -L"$out/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$out/build/bin" -o "$out/llamacpp-dump"
echo "$out/llamacpp-dump"
