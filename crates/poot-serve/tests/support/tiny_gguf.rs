//! A tiny all-F32 deepseek2-architecture GGUF (two layers: one dense, one routed with a shared
//! expert; a 4-token vocabulary): a decoder the family registry loads as a `ModelHandle`. Shared by the unit tests
//! and the production-binary test through `#[path]`, so there is one writer.

use poot_load::gguf::{GgufValue, write_gguf};

/// Write the fixture to `path`. `context_length` is the GGUF's `deepseek2.context_length`.
pub fn write_tiny_deepseek2_gguf(path: &std::path::Path, context_length: u32) {
    let (h, hq, r, kv_rank, nope, rope_d, vd) =
        (8usize, 2usize, 4usize, 4usize, 2usize, 2usize, 2usize);
    let qk_head_dim = nope + rope_d;
    let layers = 2usize;
    let vocab = 4usize;
    let dense_inter = 6usize;
    let (experts, top_k, n_group, topk_group, moe_inter, shared) =
        (6usize, 2usize, 3usize, 1usize, 4usize, 1usize);
    let shared_inter = moe_inter * shared;

    // A xorshift fill: non-degenerate magnitudes.
    fn fill(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }
    let mut seed = 1u64;
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    // A weight (0.1 scale), a norm scale (near 1, kind 'g') or a selection bias (kind 'b').
    let mut push = |name: String, dims: &[usize], kind: char| {
        seed = seed.wrapping_add(0x9E37_79B9);
        let n: usize = dims.iter().product();
        let bytes: Vec<u8> = fill(seed, n)
            .into_iter()
            .map(|v| match kind {
                'g' => 1.0 + v * 0.05,
                'b' => v * 0.5,
                _ => v * 0.1,
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        tensors.push((name, dims.iter().map(|&d| d as u64).collect(), 0, bytes));
    };
    push("token_embd.weight".into(), &[h, vocab], 'w');
    for li in 0..layers {
        let blk = |s: &str| format!("blk.{li}.{s}");
        push(blk("attn_norm.weight"), &[h], 'g');
        push(blk("ffn_norm.weight"), &[h], 'g');
        push(blk("attn_q_a.weight"), &[h, r], 'w');
        push(blk("attn_q_a_norm.weight"), &[r], 'g');
        push(blk("attn_q_b.weight"), &[r, hq * qk_head_dim], 'w');
        push(blk("attn_kv_a_mqa.weight"), &[h, kv_rank + rope_d], 'w');
        push(blk("attn_kv_a_norm.weight"), &[kv_rank], 'g');
        push(blk("attn_k_b.weight"), &[nope, kv_rank, hq], 'w');
        push(blk("attn_v_b.weight"), &[kv_rank, vd, hq], 'w');
        push(blk("attn_output.weight"), &[hq * vd, h], 'w');
        if li == 0 {
            push(blk("ffn_gate.weight"), &[h, dense_inter], 'w');
            push(blk("ffn_up.weight"), &[h, dense_inter], 'w');
            push(blk("ffn_down.weight"), &[dense_inter, h], 'w');
        } else {
            push(blk("ffn_gate_inp.weight"), &[h, experts], 'w');
            push(blk("exp_probs_b.bias"), &[experts], 'b');
            push(blk("ffn_gate_exps.weight"), &[h, moe_inter, experts], 'w');
            push(blk("ffn_up_exps.weight"), &[h, moe_inter, experts], 'w');
            push(blk("ffn_down_exps.weight"), &[moe_inter, h, experts], 'w');
            push(blk("ffn_gate_shexp.weight"), &[h, shared_inter], 'w');
            push(blk("ffn_up_shexp.weight"), &[h, shared_inter], 'w');
            push(blk("ffn_down_shexp.weight"), &[shared_inter, h], 'w');
        }
    }
    push("output_norm.weight".into(), &[h], 'g');
    push("output.weight".into(), &[h, vocab], 'w');

    let u = |v: usize| GgufValue::U32(v as u32);
    let kvs: Vec<(&str, GgufValue)> = vec![
        ("general.architecture", GgufValue::Str("deepseek2".into())),
        ("deepseek2.embedding_length", u(h)),
        ("deepseek2.block_count", u(layers)),
        ("deepseek2.attention.head_count", u(hq)),
        ("deepseek2.attention.q_lora_rank", u(r)),
        ("deepseek2.attention.kv_lora_rank", u(kv_rank)),
        ("deepseek2.attention.key_length_mla", u(qk_head_dim)),
        ("deepseek2.rope.dimension_count", u(rope_d)),
        ("deepseek2.attention.value_length_mla", u(vd)),
        (
            "deepseek2.attention.layer_norm_rms_epsilon",
            GgufValue::F32(1e-5),
        ),
        ("deepseek2.context_length", GgufValue::U32(context_length)),
        ("deepseek2.rope.freq_base", GgufValue::F32(10_000.0)),
        ("deepseek2.feed_forward_length", u(dense_inter)),
        ("deepseek2.expert_count", u(experts)),
        ("deepseek2.expert_used_count", u(top_k)),
        ("deepseek2.expert_feed_forward_length", u(moe_inter)),
        ("deepseek2.expert_shared_count", u(shared)),
        ("deepseek2.leading_dense_block_count", u(1)),
        ("deepseek2.expert_weights_scale", GgufValue::F32(1.7)),
        ("deepseek2.expert_group_count", u(n_group)),
        ("deepseek2.expert_group_used_count", u(topk_group)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(99)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "ab", "c"]
                    .iter()
                    .map(|s| GgufValue::Str((*s).to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let refs: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors
        .iter()
        .map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone()))
        .collect();
    std::fs::write(path, write_gguf(&kvs, &refs)).expect("write the tiny GGUF");
}
