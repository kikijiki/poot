#!/usr/bin/env bash
# Regenerate the committed llama.cpp references of the real Qwen3.8-27B GGUF: the activations of its first GDN
# layer (the `the_real_gguf_gdn_layer_matches_the_llama_cpp_activation` row) and the per-row dequantization
# table (the `the_real_gguf_rows_dequantize_to_the_llama_cpp_reference` row), both in
# crates/poot-models/src/architectures/qwen38_27b.rs.
#
#   scripts/llamacpp-dump/reference.sh <Qwen3.8-27B-UD-Q4_K_XL.gguf> [dump-build-dir]
#
# llama.cpp (the release `build.sh` prints, pinned by flake.lock) runs the file cut to its first four blocks
# (three GDN layers and the first full-attention layer, which llama.cpp needs to build the hybrid graph; the
# layer under test is block 0, whose quantized tensors llama.cpp's own `to_float` widens to f32 so that it
# multiplies in f32 like the oracle; the other blocks stay quantized) over the eight tokens in tokens.txt on the
# CPU. `attn_norm-0` is the GDN layer's input and `linear_attn_out-0` its output (after the output
# projection, before the residual add); both are [8 tokens, 5120] f32, token-major.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
model=$1
read -r dump truncate dequant < <("$here/build.sh" ${2:+"$2"} | tail -n1)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$truncate" "$model" "$work/four-blocks.gguf" 4 0
mkdir "$work/dump"
"$dump" "$work/four-blocks.gguf" "$work/dump" "$here/reference/tokens.txt" > "$work/dump.log" 2>&1

for name in attn_norm-0 linear_attn_out-0; do
    seq=$(awk -F'\t' -v n="$name" '$2 == n { print $1 }' "$work/dump/index.tsv")
    test "$(wc -w <<<"$seq")" -eq 1
    cp "$work/dump/$seq.f32" "$here/reference/qwen35_gdn_layer0_$name.f32"
done

# The per-row dequantization rows: llama.cpp's own `to_float` over single rows of the real file's tensors, one
# pair of rows (first and last) per quantization the file uses.
"$dequant" "$model" \
    blk.0.ssm_out.weight:0 blk.0.ssm_out.weight:4999 \
    blk.0.ffn_up.weight:0 blk.0.ffn_up.weight:17407 \
    token_embd.weight:0 token_embd.weight:248319 \
    blk.0.attn_qkv.weight:0 blk.0.attn_qkv.weight:10239 \
    output.weight:0 output.weight:248319 \
    blk.1.ffn_down.weight:0 blk.1.ffn_down.weight:5119 \
    blk.0.ffn_down.weight:0 blk.0.ffn_down.weight:5119 \
    blk.0.ffn_gate.weight:0 blk.0.ffn_gate.weight:17407 \
    blk.14.ffn_down.weight:0 blk.14.ffn_down.weight:5119 \
    > "$here/reference/qwen35_dequant_rows.tsv"
