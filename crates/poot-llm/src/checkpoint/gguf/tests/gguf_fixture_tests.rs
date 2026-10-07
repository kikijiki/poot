use crate::text::tokenize::gguf_tokenizer;
use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};

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
