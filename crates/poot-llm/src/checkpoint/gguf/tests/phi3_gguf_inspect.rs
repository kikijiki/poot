use super::super::gguf_config;
use crate::checkpoint::gguf::rope::gguf_rope_tables;
use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf};

// LongRoPE-from-GGUF: a phi-4-mini GGUF carries partial rotary (rope.dimension_count 96 < head_dim 128),
// GQA (24/8), and the LongRoPE factors as `rope_factors_short.weight` +
// rope.scaling.original_context_length. This builds the rope tables from that metadata + the small
// rope_factors tensor (no model weights) and asserts they are finite with rotary_dim 96, which catches the
// infinite-attention-factor NaN bug on the GGUF metadata path. End-to-end coherence is covered by the
// phi-3-mini GGUF generate path and safetensors phi-4-mini (same `rope_tables` path, L40S-verified).
#[test]
#[ignore = "needs the phi-4-mini GGUF (Q4_K_M) under POOT_MODELS_DIR"]
fn phi4_mini_gguf_longrope_tables_finite() {
    let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "phi-4-mini-gguf/Phi-4-mini-instruct-Q4_K_M.gguf"
    )) else {
        return;
    };
    let g = GgufIndex::open(&path).expect("open phi-4-mini gguf index");
    let file = std::fs::File::open(&path).expect("open phi-4-mini gguf file");
    let store = read_gguf(&g, &file, &IdentityNames).expect("read phi-4-mini gguf");
    let cfg = gguf_config(&g, "phi3", false).expect("phi3 gguf config");
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(
        cfg.rotary_dim, 96,
        "partial rotary from rope.dimension_count"
    );
    assert_eq!(cfg.n_heads, 24);
    assert_eq!(cfg.n_kv_heads, 8, "GQA");
    let (cos, sin) = gguf_rope_tables(&g, &store, &cfg, "phi3").expect("phi3 gguf rope tables");
    assert_eq!(cos.shape(), vec![cfg.max_pos, cfg.rotary_dim]);
    // the bug was an infinite attention factor -> all-NaN tables. Assert finite (sample the used rows).
    let n = (5 * cfg.rotary_dim).min(cos.as_f32().unwrap().len());
    assert!(
        cos.as_f32().unwrap()[..n].iter().all(|v| v.is_finite())
            && sin.as_f32().unwrap()[..n].iter().all(|v| v.is_finite()),
        "rope tables must be finite (the LongRoPE attention factor must not blow up)"
    );
    // attn_factor ~= sqrt(1 + ln(131072/4096)/ln(4096)) ~= 1.19, so cos at pos 0 (angle 0) == attn_factor.
    eprintln!("cos[0] (== attn_factor) = {}", cos.as_f32().unwrap()[0]);
    assert!(
        (cos.as_f32().unwrap()[0] - 1.19).abs() < 0.05,
        "pos-0 cos should equal the ~1.19 attention factor, got {}",
        cos.as_f32().unwrap()[0]
    );
}
