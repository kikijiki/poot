use super::super::{gguf_config, gguf_proj_dtype};
use crate::checkpoint::gguf::arch_config::gguf_weights;
use crate::checkpoint::gguf::rope::{gguf_rope_tables, rope_tables};
use crate::checkpoint::place::unpermute_qk_rows;
use crate::core::runner::Runner;
use crate::text::tokenize::gguf_tokenizer;
use poot_load::RopeScaling;
use poot_load::gguf::{GgufIndex, GgufValue, IdentityNames, read_gguf, write_gguf};
use poot_models::mixtral::{MixtralParams, trace_mixtral_prefill};

// small non-degenerate f32 tensor bytes, keeping a forward finite.
fn small_f32(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (((i % 7) as f32) * 0.1 - 0.3).to_le_bytes())
        .collect()
}

/// The q/k un-permute undoes exactly llama.cpp's rope permute. `convert_hf_to_gguf.py`'s
/// `LlamaModel.permute` is `w.reshape(heads, 2, hd/2, K).swapaxes(1, 2).reshape(out, K)`: re-derived here
/// as an index map (stored row -> HF row) independently of `unpermute_qk_rows`, whose gather must
/// then restore every HF row. Checked directly because a wrong un-permute still gives a finite,
/// in-vocab token in the end-to-end smoke test. Mutation: swapping `2 * j` and `2 * j + 1` in
/// `unpermute_qk_rows` turns this red.
#[test]
fn unpermute_qk_rows_inverts_the_llama_cpp_rope_permute() {
    for (out, heads) in [(4, 1), (8, 2), (24, 3), (64, 4)] {
        let hd = out / heads;
        let half = hd / 2;
        // llama.cpp: stored[h][j][t] = hf[h][t][j] over (heads, half, 2) <- (heads, 2, half).
        let stored_from_hf: Vec<usize> = (0..out)
            .map(|stored| {
                let (h, rest) = (stored / hd, stored % hd);
                let (j, t) = (rest / 2, rest % 2);
                h * hd + t * half + j
            })
            .collect();
        let rows = unpermute_qk_rows(out, heads);
        let restored: Vec<usize> = rows.iter().map(|&stored| stored_from_hf[stored]).collect();
        assert_eq!(
            restored,
            (0..out).collect::<Vec<_>>(),
            "out {out} heads {heads}"
        );
    }
}

// GGUF `proj_dtype` probe tests: a missing required projection keeps the probe at F32.

/// Values exactly representable in BF16.
fn native16_exact_vals(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = ((i % 7) as f32) * 0.125 - 0.375;
            poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(v))
        })
        .collect()
}

fn native16_exact_f32_bytes(n: usize) -> Vec<u8> {
    native16_exact_vals(n)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn native16_exact_bf16_bytes(n: usize) -> Vec<u8> {
    native16_exact_vals(n)
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A tiny 1-layer qwen2 GGUF with the seven projection weights stored as BF16 and every other tensor
/// F32.
fn tiny_qwen2_gguf_bytes() -> Vec<u8> {
    const F32: u32 = 0;
    const BF16: u32 = 30;
    let proj_ty = BF16;
    let proj = native16_exact_bf16_bytes;
    let plain = native16_exact_f32_bytes;
    // dims: hidden 8, 2 heads / 1 KV -> q_dim 8 / kv_dim 4; FFN 16; vocab 4. ggml ne-dims = reversed.
    let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        ("token_embd.weight", vec![8, 4], F32, plain(32)),
        ("output.weight", vec![8, 4], F32, plain(32)),
        ("output_norm.weight", vec![8], F32, plain(8)),
        ("blk.0.attn_q.weight", vec![8, 8], proj_ty, proj(64)),
        ("blk.0.attn_k.weight", vec![8, 4], proj_ty, proj(32)),
        ("blk.0.attn_v.weight", vec![8, 4], proj_ty, proj(32)),
        ("blk.0.attn_output.weight", vec![8, 8], proj_ty, proj(64)),
        ("blk.0.attn_q.bias", vec![8], F32, plain(8)),
        ("blk.0.attn_k.bias", vec![4], F32, plain(4)),
        ("blk.0.attn_v.bias", vec![4], F32, plain(4)),
        ("blk.0.attn_norm.weight", vec![8], F32, plain(8)),
        ("blk.0.ffn_norm.weight", vec![8], F32, plain(8)),
        ("blk.0.ffn_gate.weight", vec![8, 16], proj_ty, proj(128)),
        ("blk.0.ffn_up.weight", vec![8, 16], proj_ty, proj(128)),
        ("blk.0.ffn_down.weight", vec![16, 8], proj_ty, proj(128)),
    ];
    let kvs = vec![
        ("general.architecture", GgufValue::Str("qwen2".into())),
        ("qwen2.embedding_length", GgufValue::U32(8)),
        ("qwen2.block_count", GgufValue::U32(1)),
        ("qwen2.attention.head_count", GgufValue::U32(2)),
        ("qwen2.attention.head_count_kv", GgufValue::U32(1)),
        ("qwen2.feed_forward_length", GgufValue::U32(16)),
        ("qwen2.context_length", GgufValue::U32(32)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "ab", "c"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    write_gguf(&kvs, &tensors)
}

#[test]
fn gguf_missing_required_projection_keeps_f32_proj_dtype() {
    let bytes = tiny_qwen2_gguf_bytes();
    let g = GgufIndex::from_bytes(&bytes).expect("parse all-BF16 fixture");
    let mut store = read_gguf(&g, bytes.as_slice(), &IdentityNames).expect("read all-BF16 fixture");
    store.remove("blk.0.ffn_down.weight");
    assert_eq!(
        gguf_proj_dtype(&store),
        poot_tensor::DType::F32,
        "a missing required projection must keep proj_dtype F32"
    );
}

#[test]
fn gguf_config_reads_sliding_window_when_present_and_none_when_absent() {
    // The GGUF config builder must read llama.cpp's `{arch}.attention.sliding_window` (a u32) into
    // `Qwen2Config::sliding_window`; otherwise a GGUF SWA model (e.g. Mistral/Qwen2 with a window) silently runs
    // full causal attention past the window and gives wrong logits once pos >= window. Exercises `gguf_config`
    // with the key present (-> Some(w)) and absent (-> None).
    let base_kvs = |extra: Vec<(&'static str, GgufValue)>| -> Vec<(&'static str, GgufValue)> {
        let mut kvs = vec![
            ("general.architecture", GgufValue::Str("qwen2".into())),
            ("qwen2.embedding_length", GgufValue::U32(8)),
            ("qwen2.block_count", GgufValue::U32(1)),
            ("qwen2.attention.head_count", GgufValue::U32(2)),
            ("qwen2.attention.head_count_kv", GgufValue::U32(1)),
            ("qwen2.feed_forward_length", GgufValue::U32(16)),
            ("qwen2.context_length", GgufValue::U32(32)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(
                    ["a", "b", "ab", "c"]
                        .iter()
                        .map(|s| GgufValue::Str(s.to_string()))
                        .collect(),
                ),
            ),
        ];
        kvs.extend(extra);
        kvs
    };

    // Present: qwen2.attention.sliding_window = 4096 -> Some(4096).
    let with_window = base_kvs(vec![(
        "qwen2.attention.sliding_window",
        GgufValue::U32(4096),
    )]);
    let g = GgufIndex::from_bytes(&write_gguf(&with_window, &[])).expect("parse written gguf");
    let cfg = gguf_config(&g, "qwen2", true).expect("gguf_config with sliding_window key");
    assert_eq!(cfg.sliding_window, Some(4096));

    // Absent: no sliding_window key at all -> None (non-SWA checkpoints unchanged).
    let without_window = base_kvs(vec![]);
    let g = GgufIndex::from_bytes(&write_gguf(&without_window, &[])).expect("parse written gguf");
    let cfg = gguf_config(&g, "qwen2", true).expect("gguf_config without sliding_window key");
    assert_eq!(cfg.sliding_window, None);
}

#[test]
fn gguf_tokenizer_builds_a_bpe_from_written_vocab_and_merges() {
    // The qwen2 GGUF tokenizer path: tokenizer.ggml.tokens (vocab) + tokenizer.ggml.merges -> a byte-level
    // BPE. A tiny vocab over printable ASCII (which the byte-level alphabet maps to itself) with one merge gives
    // a known tokenization, validating the gguf_tokenizer wiring without a tokenizer file.
    let kvs = vec![
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "ab"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]), // a + b -> ab
        ),
    ];
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    let tok = gguf_tokenizer(&g).expect("build tokenizer");
    // "ab" merges to the single token id 2; "ba" has no "b a" merge so stays [b=1, a=0].
    assert_eq!(tok.encode("ab", false).unwrap().get_ids(), &[2]);
    assert_eq!(tok.encode("a", false).unwrap().get_ids(), &[0]);
    assert_eq!(tok.encode("ba", false).unwrap().get_ids(), &[1, 0]);
    assert_eq!(tok.decode(&[2], false).unwrap(), "ab");
}

#[test]
fn gguf_tokenizer_drops_orphan_merges_instead_of_failing() {
    // Regression: a byte-level BPE GGUF whose merge list references a non-vocab piece or produces a non-vocab
    // result (an OLMo-2 GGUF-conversion quirk: mojibake-U+FFFD merges) used to fail the whole tokenizer build
    // with HuggingFace `Token ... out of vocabulary`. Such merges are dead (a merge fires only when both inputs
    // and the result exist), so gguf_tokenizer drops them, as llama.cpp does. Same tiny vocab as above plus two
    // orphan merges: `a z` (piece `z` not in vocab) and `b b` (result `bb` not in vocab).
    let kvs = vec![
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "ab"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(
                ["a b", "a z", "b b"] // only "a b" is live; "a z"/"b b" are orphans to be dropped
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
    ];
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    // must not return Err.
    let tok = gguf_tokenizer(&g).expect("build tokenizer despite orphan merges");
    // The live merge still fires (a+b -> id 2); dropping the dead merges changed nothing observable.
    assert_eq!(tok.encode("ab", false).unwrap().get_ids(), &[2]);
    assert_eq!(tok.encode("ba", false).unwrap().get_ids(), &[1, 0]);
}

#[test]
fn gguf_tokenizer_registers_control_tokens_as_atomic() {
    // Regression: a CONTROL/USER_DEFINED token (tokenizer.ggml.token_type 3/4), a chat turn marker like
    // `<start_of_turn>` / `<|user|>`, must encode atomically even adjacent to text, not shred into
    // byte/sub-word pieces. Chat templates emit `<marker>text` with no space boundary (gemma's
    // `<start_of_turn>model`, phi3's `<|user|>Hi`), and before `gguf_special_tokens` registered these the marker
    // went through BPE and fragmented. Vocab: a/b/ab plus the control token `<S>` (type 3); one merge
    // `a b -> ab`.
    let kvs = vec![
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "ab", "<S>"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(vec![
                GgufValue::U32(1), // NORMAL
                GgufValue::U32(1),
                GgufValue::U32(1),
                GgufValue::U32(3), // CONTROL -> must be registered atomic
            ]),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    let tok = gguf_tokenizer(&g).expect("build tokenizer");
    // The control token adjacent to text encodes as [ <S>=id3, ab=id2 ], not fragmented `<`,`S`,`>` bytes.
    assert_eq!(
        tok.encode("<S>ab", false).unwrap().get_ids(),
        &[3, 2],
        "control token must be atomic even when immediately followed by text"
    );
    // Plain text is unaffected (no spurious special matching).
    assert_eq!(tok.encode("ab", false).unwrap().get_ids(), &[2]);
}

#[test]
fn gguf_spm_tokenizer_applies_dummy_space_prefix() {
    // SentencePiece add_dummy_prefix (default true when tokenizer.ggml.add_space_prefix is absent, as for
    // phi3/llama/mistral): a leading word must tokenize with the ▁-prefixed piece, as llama.cpp does, not the
    // bare piece. Minimal "llama" SPM GGUF whose vocab has both "▁The"(1) and "The"(2): encoding "The" must
    // pick "▁The".
    let kvs = vec![
        ("tokenizer.ggml.model", GgufValue::Str("llama".into())),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["<unk>", "\u{2581}The", "The", "\u{2581}cat", "cat"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
    ];
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    let tok = gguf_tokenizer(&g).expect("build spm tokenizer");
    // Leading word gets the ▁ dummy prefix: "▁The" (id 1), not the bare "The" (id 2).
    assert_eq!(
        tok.encode("The", false).unwrap().get_ids(),
        &[1],
        "dummy prefix must make a leading word ▁-prefixed"
    );
    // Every space-separated word is ▁-prefixed (the prefix + space->▁ normalizer).
    assert_eq!(tok.encode("The cat", false).unwrap().get_ids(), &[1, 3]);
}

#[test]
fn spm_encode_is_score_greedy_with_byte_fallback() {
    use crate::text::tokenize::spm_encode;
    use std::collections::HashMap;
    // "xyz" can segment as [xy, z] or [x, yz] depending on which merge scores higher: the SentencePiece
    // score-greedy agglomeration llama.cpp uses, which a reconstructed-BPE path cannot reproduce.
    let toks = ["x", "y", "z", "xy", "yz", "<unk>", "<0x71>"];
    let vocab: HashMap<String, u32> = toks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), i as u32))
        .collect();
    // xy(-1.0) beats yz(-2.0): merge xy first -> [xy=3, z=2].
    let mut out = Vec::new();
    spm_encode(
        "xyz",
        &vocab,
        &[0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0],
        5,
        &mut out,
    );
    assert_eq!(out, vec![3, 2], "higher-scoring merge (xy) wins");
    // Flip the scores: yz(-1.0) beats xy(-2.0): merge yz first -> [x=0, yz=4]. Different segmentation.
    let mut out2 = Vec::new();
    spm_encode(
        "xyz",
        &vocab,
        &[0.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0],
        5,
        &mut out2,
    );
    assert_eq!(out2, vec![0, 4], "score order flips the segmentation");
    // byte-fallback: 'q' (0x71) is not a token but <0x71> is -> emit the byte token, not <unk>.
    let mut out3 = Vec::new();
    spm_encode(
        "xq",
        &vocab,
        &[0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0],
        5,
        &mut out3,
    );
    assert_eq!(
        out3,
        vec![0, 6],
        "byte-fallback maps an out-of-vocab char to its <0xNN> token"
    );
}

#[test]
fn spm_data_encode_edge_inputs_do_not_panic() {
    use crate::text::tokenize::SpmData;
    use std::collections::HashMap;
    // A tiny SPM tokenizer: "a"/"b"/"ab" text tokens, a control special "<S>", and one byte token.
    let toks = ["<unk>", "a", "b", "ab", "<S>", "<0x63>"];
    let vocab: HashMap<String, u32> = toks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), i as u32))
        .collect();
    let spm = SpmData {
        vocab,
        scores: vec![0.0, -1.0, -1.0, -0.5, 0.0, 0.0],
        unk_id: 0,
        add_prefix: false,
        specials: vec![("<S>".to_string(), 4)],
    };
    // Empty, whitespace, all-specials-adjacent, unicode/replacement, control bytes, long, many-specials.
    for input in [
        "",
        "   ",
        "<S><S><S>",
        "a<S>b<S>ab",
        "café 日本 \u{fffd} 𝕏",
        "\0\u{7f}",
        &"ab".repeat(3000),
        &"<S>".repeat(400),
    ] {
        // Must not panic; returns some (possibly empty) id list.
        let _ = spm.encode(input);
    }
    // Concrete behaviors: a control token adjacent to text stays atomic; "ab" merges (score -0.5 highest).
    assert_eq!(spm.encode(""), Vec::<u32>::new());
    assert_eq!(
        spm.encode("<S>ab"),
        vec![4, 3],
        "special atomic, then ab merged"
    );
    assert_eq!(
        spm.encode("c"),
        vec![5],
        "byte-fallback for out-of-vocab 'c' -> <0x63>"
    );
}

#[test]
fn gguf_weights_maps_a_written_qwen2_layer() {
    // Build a tiny 1-layer qwen2 model in a GGUF (all tensors gguf_weights requires, with ggml names and
    // ne-reversed dims) and confirm gguf_weights returns the HF-named weights with the right shapes/values.
    // Exercises the name mapping, dim reversal, and the t2 (transpose) vs asis paths. dims: hidden H=8,
    // n_heads=2, n_kv=1, head_dim=4 -> q_dim=8, kv_dim=4; inter I=16; vocab V=4.
    const F32: u32 = 0;
    let seq = |n: usize| -> Vec<u8> { (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect() };
    // (ggml-name, ne-dims = reversed logical, F32, data). Logical [out,in] -> ne [in,out].
    let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        ("token_embd.weight", vec![8, 4], F32, seq(32)), // logical [V=4,H=8]
        ("output.weight", vec![8, 4], F32, seq(32)),     // lm_head, logical [V,H]
        ("output_norm.weight", vec![8], F32, seq(8)),
        ("blk.0.attn_q.weight", vec![8, 8], F32, seq(64)), // [q_dim=8,H=8]
        ("blk.0.attn_k.weight", vec![8, 4], F32, seq(32)), // [kv_dim=4,H=8]
        ("blk.0.attn_v.weight", vec![8, 4], F32, seq(32)),
        ("blk.0.attn_output.weight", vec![8, 8], F32, seq(64)), // [H,q_dim]
        ("blk.0.attn_q.bias", vec![8], F32, seq(8)),
        ("blk.0.attn_k.bias", vec![4], F32, seq(4)),
        ("blk.0.attn_v.bias", vec![4], F32, seq(4)),
        ("blk.0.attn_norm.weight", vec![8], F32, seq(8)),
        ("blk.0.ffn_norm.weight", vec![8], F32, seq(8)),
        ("blk.0.ffn_gate.weight", vec![8, 16], F32, seq(128)), // [I=16,H=8]
        ("blk.0.ffn_up.weight", vec![8, 16], F32, seq(128)),
        ("blk.0.ffn_down.weight", vec![16, 8], F32, seq(128)), // [H=8,I=16]
    ];
    let kvs = vec![
        ("general.architecture", GgufValue::Str("qwen2".into())),
        ("qwen2.embedding_length", GgufValue::U32(8)),
        ("qwen2.block_count", GgufValue::U32(1)),
        ("qwen2.attention.head_count", GgufValue::U32(2)),
        ("qwen2.attention.head_count_kv", GgufValue::U32(1)),
        ("qwen2.feed_forward_length", GgufValue::U32(16)),
        ("qwen2.context_length", GgufValue::U32(32)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array((0..4).map(|i| GgufValue::Str(format!("t{i}"))).collect()),
        ),
    ];
    let bytes = write_gguf(&kvs, &tensors);
    let g = GgufIndex::from_bytes(&bytes).expect("parse written gguf");
    let store = read_gguf(&g, bytes.as_slice(), &IdentityNames).expect("read written gguf");
    let cfg = gguf_config(&g, "qwen2", true).unwrap();
    let (w, formats) = gguf_weights(&g, &store, &cfg, "qwen2").expect("gguf_weights");
    assert!(formats.is_empty(), "an all-F32 file stores nothing packed");

    // embeddings/norms/biases are loaded as-is (no transpose): shape + values pass straight through.
    let embed = &w["model.embed_tokens.weight"];
    assert_eq!(embed.as_host().expect("dense weight").shape(), vec![4, 8]);
    assert_eq!(
        embed.as_host().expect("dense weight").as_f32().unwrap(),
        &(0..32).map(|i| i as f32).collect::<Vec<_>>()[..]
    );
    let qbias = &w["model.layers.0.self_attn.q_proj.bias"];
    assert_eq!(qbias.as_host().expect("dense weight").shape(), vec![8]);
    assert_eq!(
        qbias.as_host().expect("dense weight").as_f32().unwrap(),
        &(0..8).map(|i| i as f32).collect::<Vec<_>>()[..]
    );
    // projection/FFN weights are transposed [out,in]->[in,out] (non-square so the transpose is visible).
    assert_eq!(
        w["model.layers.0.mlp.gate_proj.weight"]
            .as_host()
            .expect("dense weight")
            .shape(),
        vec![8, 16]
    ); // [H, I]
    assert_eq!(
        w["model.layers.0.mlp.down_proj.weight"]
            .as_host()
            .expect("dense weight")
            .shape(),
        vec![16, 8]
    ); // [I, H]
    assert_eq!(
        w["lm_head.weight"].as_host().expect("dense weight").shape(),
        vec![8, 4]
    ); // output [V,H] -> [H,V]
    // the standard qwen2 HF norm names are present.
    assert!(w.contains_key("model.layers.0.input_layernorm.weight"));
    assert!(w.contains_key("model.layers.0.post_attention_layernorm.weight"));
    assert!(w.contains_key("model.norm.weight"));
}

#[test]
fn gguf_config_parses_a_written_qwen2_metadata_block() {
    // Build a metadata-only GGUF in memory (via the writer) carrying a tiny qwen2 config, and confirm
    // gguf_config reads every field.
    let toks: Vec<GgufValue> = (0..7).map(|i| GgufValue::Str(format!("t{i}"))).collect(); // vocab = 7
    let kvs = vec![
        ("general.architecture", GgufValue::Str("qwen2".into())),
        ("qwen2.embedding_length", GgufValue::U32(64)),
        ("qwen2.block_count", GgufValue::U32(3)),
        ("qwen2.attention.head_count", GgufValue::U32(8)),
        ("qwen2.attention.head_count_kv", GgufValue::U32(2)),
        ("qwen2.feed_forward_length", GgufValue::U32(128)),
        ("qwen2.context_length", GgufValue::U32(4096)),
        (
            "qwen2.attention.layer_norm_rms_epsilon",
            GgufValue::F32(1e-5),
        ),
        ("tokenizer.ggml.tokens", GgufValue::Array(toks)),
    ];
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    let cfg = gguf_config(&g, "qwen2", true).expect("gguf_config");
    assert_eq!(cfg.vocab, 7);
    assert_eq!(cfg.hidden, 64);
    assert_eq!(cfg.layers, 3);
    assert_eq!(cfg.n_heads, 8);
    assert_eq!(cfg.n_kv_heads, 2);
    assert_eq!(cfg.inter, 128);
    assert_eq!(cfg.head_dim, 8); // key_length omitted -> hidden/n_heads
    assert_eq!(cfg.rotary_dim, 8); // rope.dimension_count omitted -> head_dim (full rotary)
    assert!((cfg.eps - 1e-5).abs() < 1e-9);
    assert_eq!(cfg.max_pos, 4096);
    assert!(cfg.qkv_bias);
    assert!(!cfg.qk_norm); // qwen2 (not gemma3/qwen3/olmo2)
}

// GGUF metadata wiring for linear/dynamic/yarn RoPE scaling and the LongRoPE long-context regime (card 245)

fn minimal_qwen2_kvs(extra: Vec<(&'static str, GgufValue)>) -> Vec<(&'static str, GgufValue)> {
    let mut kvs = vec![
        ("general.architecture", GgufValue::Str("qwen2".into())),
        ("qwen2.embedding_length", GgufValue::U32(8)),
        ("qwen2.block_count", GgufValue::U32(1)),
        ("qwen2.attention.head_count", GgufValue::U32(2)),
        ("qwen2.attention.head_count_kv", GgufValue::U32(2)),
        ("qwen2.feed_forward_length", GgufValue::U32(16)),
        ("qwen2.context_length", GgufValue::U32(64)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "ab", "c"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
    ];
    kvs.extend(extra);
    kvs
}

#[test]
fn gguf_rope_scaling_type_linear_wires_the_same_table_as_an_explicit_hf_config() {
    // `{arch}.rope.scaling.type` must be wired: "linear" (and "dynamic"/"yarn") used to fall through to plain
    // unscaled RoPE for a native-GGUF checkpoint, though the safetensors/HF-config path (card 236) handled
    // them. Keys follow llama.cpp's gguf-py `Keys.Rope`: type `{arch}.rope.scaling.type`, factor
    // `{arch}.rope.scaling.factor`.
    let kvs = minimal_qwen2_kvs(vec![
        ("qwen2.rope.scaling.type", GgufValue::Str("linear".into())),
        ("qwen2.rope.scaling.factor", GgufValue::F32(4.0)),
    ]);
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    let cfg = gguf_config(&g, "qwen2", true).expect("gguf_config");
    let store = poot_quant::weights::WeightStore::default();
    let (cos, sin) = gguf_rope_tables(&g, &store, &cfg, "qwen2").expect("gguf rope tables");

    let expected_scaling = RopeScaling {
        rope_type: "linear".to_string(),
        factor: 4.0,
        low_freq_factor: 0.0,
        high_freq_factor: 0.0,
        original_max_position_embeddings: 0,
        long_factor: None,
        short_factor: None,
        beta_fast: None,
        beta_slow: None,
        attention_factor: None,
        mscale: None,
        mscale_all_dim: None,
    };
    // qwen2's default freq_base (no rope.freq_base key in this fixture) is 1e6.
    let (want_cos, want_sin) = rope_tables(&cfg, 1_000_000.0, Some(&expected_scaling), None);
    assert_eq!(
        cos.as_f32().unwrap(),
        want_cos.as_f32().unwrap(),
        "linear-scaled GGUF table must match the equivalent explicit RopeScaling table"
    );
    assert_eq!(sin.as_f32().unwrap(), want_sin.as_f32().unwrap());
    // and it must differ from plain (unscaled) RoPE - proof the wiring did something.
    let (plain_cos, _) = rope_tables(&cfg, 1_000_000.0, None, None);
    assert_ne!(
        cos.as_f32().unwrap(),
        plain_cos.as_f32().unwrap(),
        "linear scaling must change the table vs plain RoPE"
    );
}

#[test]
fn gguf_rope_scaling_type_yarn_wires_factor_orig_ctx_attn_factor_and_betas() {
    // Exercises every yarn-specific GGUF key: type, factor, original_context_length, attn_factor (explicit
    // override), yarn_beta_fast, yarn_beta_slow (key strings from gguf-py `Keys.Rope`; `yarn_beta_fast`/
    // `yarn_beta_slow` are easy to misname).
    let kvs = minimal_qwen2_kvs(vec![
        ("qwen2.rope.scaling.type", GgufValue::Str("yarn".into())),
        ("qwen2.rope.scaling.factor", GgufValue::F32(8.0)),
        (
            "qwen2.rope.scaling.original_context_length",
            GgufValue::U32(2048),
        ),
        ("qwen2.rope.scaling.attn_factor", GgufValue::F32(2.5)),
        ("qwen2.rope.scaling.yarn_beta_fast", GgufValue::F32(16.0)),
        ("qwen2.rope.scaling.yarn_beta_slow", GgufValue::F32(2.0)),
    ]);
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    let cfg = gguf_config(&g, "qwen2", true).expect("gguf_config");
    let store = poot_quant::weights::WeightStore::default();
    let (cos, sin) = gguf_rope_tables(&g, &store, &cfg, "qwen2").expect("gguf rope tables");

    let expected_scaling = RopeScaling {
        rope_type: "yarn".to_string(),
        factor: 8.0,
        low_freq_factor: 0.0,
        high_freq_factor: 0.0,
        original_max_position_embeddings: 2048,
        long_factor: None,
        short_factor: None,
        beta_fast: Some(16.0),
        beta_slow: Some(2.0),
        attention_factor: Some(2.5),
        mscale: None,
        mscale_all_dim: None,
    };
    let (want_cos, want_sin) = rope_tables(&cfg, 1_000_000.0, Some(&expected_scaling), None);
    assert_eq!(
        cos.as_f32().unwrap(),
        want_cos.as_f32().unwrap(),
        "yarn-scaled GGUF table must match the equivalent explicit RopeScaling table (every field wired)"
    );
    assert_eq!(sin.as_f32().unwrap(), want_sin.as_f32().unwrap());
    // cos[0] (angle 0) must equal the explicit attn_factor override read from the GGUF.
    assert!(
        (cos.as_f32().unwrap()[0] - 2.5).abs() < 1e-5,
        "cos[0] should equal the wired attn_factor override: {}",
        cos.as_f32().unwrap()[0]
    );
}

#[test]
fn gguf_rope_scaling_type_dynamic_has_no_gguf_representation_and_is_not_wired() {
    // llama.cpp's GGUF conversion has no "dynamic" rope_scaling.type: `RopeScalingType` (gguf-py) and
    // `llama_rope_scaling_type_from_string` (src/llama-model.cpp) recognize only none/linear/yarn/longrope.
    // A GGUF that nonetheless carries `type: "dynamic"` must not crash or apply some other scaling; it degrades
    // to plain unscaled RoPE, like any unrecognized type string.
    let kvs = minimal_qwen2_kvs(vec![
        ("qwen2.rope.scaling.type", GgufValue::Str("dynamic".into())),
        ("qwen2.rope.scaling.factor", GgufValue::F32(4.0)),
        (
            "qwen2.rope.scaling.original_context_length",
            GgufValue::U32(16),
        ),
    ]);
    let g = GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse written gguf");
    let cfg = gguf_config(&g, "qwen2", true).expect("gguf_config");
    let store = poot_quant::weights::WeightStore::default();
    let (cos, sin) = gguf_rope_tables(&g, &store, &cfg, "qwen2").expect("gguf rope tables");
    let (plain_cos, plain_sin) = rope_tables(&cfg, 1_000_000.0, None, None);
    assert_eq!(
        cos.as_f32().unwrap(),
        plain_cos.as_f32().unwrap(),
        "an unrecognized/unrepresentable GGUF rope_scaling.type must fall back to plain RoPE, \
             not crash or misapply a different scaling"
    );
    assert_eq!(sin.as_f32().unwrap(), plain_sin.as_f32().unwrap());
}

#[test]
fn gguf_longrope_wires_the_long_factor_tensor_when_present() {
    // The GGUF phi3 LongRoPE branch must read `rope_factors_long.weight` as well as
    // rope_factors_short.weight, or `RopeScaling.long_factor` stays None and the long-context regime is
    // unreachable even for a checkpoint that ships the tensor. Cross-checks the resulting table against
    // `rope_tables` called directly with the same factors as an explicit RopeScaling, proving the tensor bytes
    // reach the struct field (regime selection is unit-tested in `rope_tests::longrope_factors_*`).
    const F32: u32 = 0;
    let short = [1.0f32, 1.0];
    let long = [2.0f32, 2.0];
    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        ("rope_factors_short.weight", vec![2], F32, f32_bytes(&short)),
        ("rope_factors_long.weight", vec![2], F32, f32_bytes(&long)),
    ];
    let kvs = vec![
        ("general.architecture", GgufValue::Str("phi3".into())),
        ("phi3.embedding_length", GgufValue::U32(8)),
        ("phi3.block_count", GgufValue::U32(1)),
        ("phi3.attention.head_count", GgufValue::U32(2)),
        ("phi3.attention.head_count_kv", GgufValue::U32(2)),
        ("phi3.feed_forward_length", GgufValue::U32(16)),
        ("phi3.context_length", GgufValue::U32(64)), // table capacity 64 > orig 16 -> long regime
        (
            "phi3.rope.scaling.original_context_length",
            GgufValue::U32(16),
        ),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "c", "d"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
    ];
    let bytes = write_gguf(&kvs, &tensors);
    let g = GgufIndex::from_bytes(&bytes).expect("parse written gguf");
    let store = read_gguf(&g, bytes.as_slice(), &IdentityNames).expect("read written gguf");
    let cfg = gguf_config(&g, "phi3", false).expect("gguf_config");
    assert_eq!(
        cfg.rotary_dim, 4,
        "head_dim=8/2=4, no rope.dimension_count key -> full rotary"
    );
    assert_eq!(cfg.max_pos, 64);
    let (cos, sin) = gguf_rope_tables(&g, &store, &cfg, "phi3").expect("gguf rope tables");

    let expected_scaling = RopeScaling {
        rope_type: "longrope".to_string(),
        factor: 0.0,
        low_freq_factor: 0.0,
        high_freq_factor: 0.0,
        original_max_position_embeddings: 16,
        long_factor: Some(long.to_vec()),
        short_factor: Some(short.to_vec()),
        beta_fast: None,
        beta_slow: None,
        attention_factor: None,
        mscale: None,
        mscale_all_dim: None,
    };
    // phi3's default freq_base (no rope.freq_base key in this fixture) is 1e4.
    let (want_cos, want_sin) = rope_tables(&cfg, 10_000.0, Some(&expected_scaling), None);
    assert_eq!(
        cos.as_f32().unwrap(),
        want_cos.as_f32().unwrap(),
        "GGUF-wired long_factor table must match the equivalent explicit RopeScaling table"
    );
    assert_eq!(sin.as_f32().unwrap(), want_sin.as_f32().unwrap());

    // Confirm the long regime is active, not silently still on short_factor=1.0 (numerically
    // identical to plain rope, which would hide a wiring bug): long_factor=2.0 halves inv_freq, so the table
    // must differ from a short-only equivalent.
    let short_only_scaling = RopeScaling {
        long_factor: None,
        ..expected_scaling
    };
    let (short_cos, _) = rope_tables(&cfg, 10_000.0, Some(&short_only_scaling), None);
    assert_ne!(
        cos.as_f32().unwrap(),
        short_cos.as_f32().unwrap(),
        "capacity 64 > orig 16 must engage the long regime, not stay on short_factor"
    );
}

#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn load_gguf_old_per_expert_mixtral_layout_matches_merged_layout() {
    // `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` (the most-downloaded real Mixtral-8x7B GGUF) predates
    // llama.cpp's merged 3D `ffn_{gate,up,down}_exps.weight` MoE convention and stores each expert as its own 2D
    // tensor: `blk.N.ffn_{gate,up,down}.{0..n_experts-1}.weight`. This builds the same tiny Mixtral checkpoint
    // twice, byte-identical per expert, once with the merged convention (as
    // `load_gguf_runs_a_forward_on_a_written_tiny_llama`, plus a router and experts) and once with the
    // per-expert convention, and asserts `Runner::load_gguf` + `next_token` give bit-exact identical output,
    // so `stack_experts_2d`'s fuse reconstructs the merged-branch weights exactly. dims: hidden 8, 2 heads / 1
    // KV (head_dim 4), 3 experts top-2, per-expert FFN 16, vocab 4.
    const F32: u32 = 0;
    let (hidden, kv_dim, inter, n_experts): (u64, u64, u64, usize) = (8, 4, 16, 3);

    // Deterministic, distinct-per-(stem,expert,index) weights: stem/expert are folded into the formula so no
    // two experts or projections are byte-identical (a stack_experts_2d ordering bug would otherwise hide behind
    // identical experts).
    let expert_bytes = |stem: usize, e: usize, n: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|i| ((((i + e * 17 + stem * 41) % 23) as f32) * 0.05 - 0.5).to_le_bytes())
            .collect()
    };
    let per_expert = |stem: usize| -> Vec<Vec<u8>> {
        (0..n_experts).map(|e| expert_bytes(stem, e, 128)).collect()
    };
    let (gate, up, down) = (per_expert(0), per_expert(1), per_expert(2));
    let merged_gate: Vec<u8> = gate.iter().flatten().copied().collect();
    let merged_up: Vec<u8> = up.iter().flatten().copied().collect();
    let merged_down: Vec<u8> = down.iter().flatten().copied().collect();

    let common_kvs = || -> Vec<(&'static str, GgufValue)> {
        vec![
            ("general.architecture", GgufValue::Str("llama".into())),
            ("llama.embedding_length", GgufValue::U32(hidden as u32)),
            ("llama.block_count", GgufValue::U32(1)),
            ("llama.attention.head_count", GgufValue::U32(2)),
            ("llama.attention.head_count_kv", GgufValue::U32(1)),
            ("llama.feed_forward_length", GgufValue::U32(inter as u32)),
            ("llama.context_length", GgufValue::U32(32)),
            ("llama.expert_count", GgufValue::U32(n_experts as u32)),
            ("llama.expert_used_count", GgufValue::U32(2)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(
                    ["a", "b", "ab", "c"]
                        .iter()
                        .map(|s| GgufValue::Str(s.to_string()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array(vec![GgufValue::Str("a b".into())]),
            ),
        ]
    };
    let base_tensors = || -> Vec<(&'static str, Vec<u64>, u32, Vec<u8>)> {
        vec![
            ("token_embd.weight", vec![hidden, 4], F32, small_f32(32)),
            ("output.weight", vec![hidden, 4], F32, small_f32(32)),
            ("output_norm.weight", vec![hidden], F32, small_f32(8)),
            (
                "blk.0.attn_q.weight",
                vec![hidden, hidden],
                F32,
                small_f32(64),
            ),
            (
                "blk.0.attn_k.weight",
                vec![hidden, kv_dim],
                F32,
                small_f32(32),
            ),
            (
                "blk.0.attn_v.weight",
                vec![hidden, kv_dim],
                F32,
                small_f32(32),
            ),
            (
                "blk.0.attn_output.weight",
                vec![hidden, hidden],
                F32,
                small_f32(64),
            ),
            ("blk.0.attn_norm.weight", vec![hidden], F32, small_f32(8)),
            ("blk.0.ffn_norm.weight", vec![hidden], F32, small_f32(8)),
            (
                "blk.0.ffn_gate_inp.weight",
                vec![hidden, n_experts as u64],
                F32,
                small_f32(hidden as usize * n_experts),
            ),
        ]
    };

    // Merged layout: one 3D tensor per projection, ggml ne = [H,I,E] (gate/up) / [I,H,E] (down).
    let mut merged_tensors = base_tensors();
    merged_tensors.push((
        "blk.0.ffn_gate_exps.weight",
        vec![hidden, inter, n_experts as u64],
        F32,
        merged_gate,
    ));
    merged_tensors.push((
        "blk.0.ffn_up_exps.weight",
        vec![hidden, inter, n_experts as u64],
        F32,
        merged_up,
    ));
    merged_tensors.push((
        "blk.0.ffn_down_exps.weight",
        vec![inter, hidden, n_experts as u64],
        F32,
        merged_down,
    ));
    let merged_path =
        poot_test_util::unique_temp_path("poot_gguf_fixture_mixtral_merged_layout.gguf");
    std::fs::write(&merged_path, write_gguf(&common_kvs(), &merged_tensors)).unwrap();

    // Old per-expert layout: one 2D tensor per expert, ggml ne = [H,I] (gate/up) / [I,H] (down), the
    // pre-2024 llama.cpp convention `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` uses.
    let mut old_tensors = base_tensors();
    // leak the per-expert name strings so they live in the tensors Vec (write_gguf takes &str); test-only,
    // small and short-lived.
    let nm = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    for e in 0..n_experts {
        old_tensors.push((
            nm(format!("blk.0.ffn_gate.{e}.weight")),
            vec![hidden, inter],
            F32,
            gate[e].clone(),
        ));
        old_tensors.push((
            nm(format!("blk.0.ffn_up.{e}.weight")),
            vec![hidden, inter],
            F32,
            up[e].clone(),
        ));
        old_tensors.push((
            nm(format!("blk.0.ffn_down.{e}.weight")),
            vec![inter, hidden],
            F32,
            down[e].clone(),
        ));
    }
    let old_path = poot_test_util::unique_temp_path("poot_gguf_fixture_mixtral_old_layout.gguf");
    std::fs::write(&old_path, write_gguf(&common_kvs(), &old_tensors)).unwrap();

    let merged_runner =
        Runner::load_gguf(&merged_path).expect("load_gguf of merged-layout mixtral");
    let old_runner = Runner::load_gguf(&old_path).expect("load_gguf of old-layout mixtral");

    assert_eq!(merged_runner.arch, "mixtral");
    assert_eq!(old_runner.arch, "mixtral");
    let merged_mp = merged_runner
        .mixtral
        .expect("Runner::load_gguf must set mixtral params for a mixtral checkpoint");
    let old_mp = old_runner
        .mixtral
        .expect("Runner::load_gguf must set mixtral params for a mixtral checkpoint");

    // Drive the Mixtral-specific prefill tracer directly: `next_token`'s generic prefill is not
    // MoE-aware and would look for dense `mlp.gate_proj.weight`, which neither fixture has (see
    // `Runner::generate_sampled`'s `self.mixtral.is_some()` arm). Compares the full logits vector, not just the
    // argmax, for a bit-exact check.
    let logits = |runner: &Runner, mp: MixtralParams, tokens: &[u32]| -> Vec<f32> {
        let g = trace_mixtral_prefill(runner.cfg, mp, tokens.len());
        let inputs = runner.bind(&g, tokens).expect("bind mixtral graph");
        crate::core::cpu_oracle::cpu_eval(&g, &inputs)
            .expect("eval mixtral graph")
            .as_f32()
            .unwrap()
            .to_vec()
    };

    // Several token-context prefixes, so a stack_experts_2d ordering/shape bug that only shows under a
    // different router selection cannot hide behind one lucky prefix.
    for tokens in [[0u32, 1, 2].as_slice(), &[2, 0], &[3, 1, 2, 0]] {
        let merged_logits = logits(&merged_runner, merged_mp, tokens);
        let old_logits = logits(&old_runner, old_mp, tokens);
        assert_eq!(
            merged_logits, old_logits,
            "old per-expert layout must produce BIT-EXACT logits vs. the merged layout for \
                 identical weights (tokens={tokens:?})"
        );
    }
}
