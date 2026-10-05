//! Checks `rope_tables`' f32 position arithmetic at a real context length. Every RoPE test in `poot-eval`
//! uses `max_pos=16` and position 3. The `rope` op itself gathers a precomputed cos/sin row by position
//! (magnitude-independent), so the arithmetic that depends on position magnitude is `rope_tables`:
//! `ang = pos as f32 * inv_freq[j]`, then `ang.cos()/ang.sin()` in f32. This checks it at Qwen2.5-0.5B's
//! config (`Qwen2Config::qwen2_0_5b()`: rotary_dim=64, rope_theta=1e6) out to max_pos=32768 against an f64
//! ground truth.
//!
//! HF's `LlamaRotaryEmbedding` computes the same `position_ids.float() @ inv_freq.float()` product in
//! float32, so poot's table matches the reference's precision. The measured error (~1.1e-3 absolute near
//! pos=32768, larger than f32 epsilon because `cos`/`sin` of a large f32 argument amplify its rounding) is
//! expected. The test pins the tolerance so a later change (an integer overflow in a position counter, or
//! `ang` computed in lower precision) is caught.

use super::super::gguf_config;
use crate::checkpoint::gguf::rope::{gguf_rope_tables, rope_tables};
use poot_models::qwen2::Qwen2Config;

#[test]
fn rope_tables_stay_close_to_f64_ground_truth_at_real_context_length() {
    let mut cfg = Qwen2Config::qwen2_0_5b();
    assert_eq!(cfg.rotary_dim, 64);
    assert_eq!(cfg.max_pos, 32768, "real Qwen2.5-0.5B max_pos");
    // keep the table just long enough to cover the max_pos boundary (avoids building the full 32768-row table
    // to check the last row).
    cfg.max_pos = 32769;
    let theta = 1_000_000.0f32; // real Qwen2.5 rope_theta
    let (cos, sin) = rope_tables(&cfg, theta, None, None);
    let d = cfg.rotary_dim;
    let half = d / 2;
    assert_eq!(cos.shape(), vec![cfg.max_pos, d]);

    let mut worst = 0.0f64;
    let mut worst_pos = 0usize;
    for &pos in &[0usize, 1, 100, 4096, 8192, 16384, 32767, 32768] {
        for j in 0..half {
            let base64 = (theta as f64).powf(-((2 * j) as f64) / d as f64);
            let ang64 = pos as f64 * base64;
            let (c64, s64) = (ang64.cos(), ang64.sin());
            let c_got = cos.as_f32().unwrap()[pos * d + j] as f64;
            let s_got = sin.as_f32().unwrap()[pos * d + j] as f64;
            let err = (c_got - c64).abs().max((s_got - s64).abs());
            if err > worst {
                worst = err;
                worst_pos = pos;
            }
            assert!(
                c_got.is_finite() && s_got.is_finite(),
                "pos={pos} j={j}: non-finite"
            );
        }
    }
    eprintln!(
        "rope_tables f32 vs f64 ground truth over pos 0..=32768: worst_abs_err={worst:e} at pos={worst_pos}"
    );
    // 2e-3: looser than a typical f32-epsilon tolerance on purpose; this matches HF's f32 reference precision
    // at this magnitude (measured worst case ~1.1e-3 at pos=32767, with headroom).
    assert!(
        worst < 2e-3,
        "rope_tables diverged from f64 ground truth by more than the expected f32-trig-at-large-\
             argument budget: {worst:e} at pos={worst_pos}"
    );
}

/// End-to-end CPU-oracle check of long-context RoPE variants: traces a one-op graph applying the `rope` IR
/// primitive (poot-graph-ir) to a `rope_tables` output, runs it through `poot-eval`'s CPU oracle, and checks
/// the position dependence and numerical stability a long-context scaling variant must have. This covers
/// more than the host-side frequency math (`rope_tests`): plugging the table into the same `OpKind::Rope`
/// primitive as plain RoPE gives a position-varying, finite rotation. No IR/kernel changes were needed for
/// YaRN/dynamic-NTK/linear, only the host-side table builder.
#[test]
fn long_context_rope_variants_end_to_end_position_dependent_and_finite() {
    use poot_eval::{EvalBudget, EvalOptions, Value};
    use poot_graph_ir::Slot;
    use poot_graph_ir::builder::Builder;
    use poot_graph_ir::ops::rope;
    use poot_graph_ir::types::TensorType;
    use poot_load::RopeScaling;
    use poot_tensor::DType;
    use poot_tensor::HostTensor;
    use std::collections::HashMap;

    let mut cfg = Qwen2Config::qwen2_0_5b();
    cfg.rotary_dim = 64;
    cfg.max_pos = 8192; // cheap but comfortably beyond a small "original" context below
    let theta = 1_000_000.0f32;
    let d = cfg.rotary_dim;
    let orig = 2048usize; // "trained" context: well inside max_pos, so scaling engages

    let variants: [(&str, RopeScaling); 3] = [
        (
            "linear",
            RopeScaling {
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
            },
        ),
        (
            "dynamic",
            RopeScaling {
                rope_type: "dynamic".to_string(),
                factor: 4.0,
                low_freq_factor: 0.0,
                high_freq_factor: 0.0,
                original_max_position_embeddings: orig,
                long_factor: None,
                short_factor: None,
                beta_fast: None,
                beta_slow: None,
                attention_factor: None,
                mscale: None,
                mscale_all_dim: None,
            },
        ),
        (
            "yarn",
            RopeScaling {
                rope_type: "yarn".to_string(),
                factor: 4.0,
                low_freq_factor: 0.0,
                high_freq_factor: 0.0,
                original_max_position_embeddings: orig,
                long_factor: None,
                short_factor: None,
                beta_fast: None,
                beta_slow: None,
                attention_factor: None,
                mscale: None,
                mscale_all_dim: None,
            },
        ),
    ];

    for (name, scaling) in &variants {
        let (cos, sin) = rope_tables(&cfg, theta, Some(scaling), None);
        assert_eq!(cos.shape(), vec![cfg.max_pos, d]);
        assert!(
            cos.as_f32().unwrap().iter().all(|v| v.is_finite())
                && sin.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "{name}: rope table itself has a non-finite entry"
        );

        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
        let cos_c = b.constant("cos", TensorType::f32(vec![cfg.max_pos, d]));
        let sin_c = b.constant("sin", TensorType::f32(vec![cfg.max_pos, d]));
        let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
        let out = rope(&b, x, cos_c, sin_c, pos);
        let (xi, ci, si, pi) = (x.id, cos_c.id, sin_c.id, pos.id);
        let g = b.finish(out);

        let xd: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.037).sin()).collect();
        let run_at = |p: usize| {
            let mut inputs: HashMap<_, Value> = HashMap::new();
            inputs.insert(xi, HostTensor::f32(vec![1, 1, 1, d], xd.clone()).into());
            inputs.insert(ci, cos.clone().into());
            inputs.insert(si, sin.clone().into());
            inputs.insert(pi, HostTensor::i32(vec![], vec![p as i32]).into());
            poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("eval")
                .output
                .into_host()
                .expect("dense output")
        };

        // Well within the original context: a short and a longer-but-still-short position should both be
        // well-behaved and different (position matters).
        let near = run_at(16);
        let mid = run_at(orig / 2);
        assert!(
            near.as_f32().unwrap().iter().all(|v| v.is_finite())
                && mid.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "{name}: non-finite output within the original context"
        );
        assert_ne!(
            near.as_f32().unwrap(),
            mid.as_f32().unwrap(),
            "{name}: two different in-context positions produced identical output"
        );

        // Far beyond the original context, at the table's capacity boundary, where naive unscaled extrapolation
        // degrades: must still be finite and position-dependent relative to `mid`.
        let far = run_at(cfg.max_pos - 1);
        assert!(
            far.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "{name}: non-finite output far beyond the original context (pos={})",
            cfg.max_pos - 1
        );
        assert_ne!(
            mid.as_f32().unwrap(),
            far.as_f32().unwrap(),
            "{name}: a long-context position produced the same output as a mid-context one"
        );
    }
}

/// End-to-end CPU-oracle check that sources scaling from a written GGUF's metadata via
/// `gguf_rope_tables` (linear, yarn) and from its `rope_factors_{short,long}.weight` tensors (LongRoPE's
/// long-context regime), then traces the `rope` IR primitive against each table as the explicit-`RopeScaling`
/// test above does. Confirms GGUF metadata parsing, the frequency-table build, and the graph op agree end
/// to end.
#[test]
fn gguf_wired_and_longrope_long_regime_end_to_end_position_dependent_and_finite() {
    use poot_eval::{EvalBudget, EvalOptions, Value};
    use poot_graph_ir::Slot;
    use poot_graph_ir::builder::Builder;
    use poot_graph_ir::ops::rope;
    use poot_graph_ir::types::TensorType;
    use poot_load::gguf::{GgufIndex, GgufValue, IdentityNames, read_gguf, write_gguf};
    use poot_tensor::DType;
    use poot_tensor::HostTensor;
    use std::collections::HashMap;

    // Small qwen2 fixture, big enough for a meaningful "original" vs "table capacity" split.
    let base_kvs = |extra: Vec<(&'static str, GgufValue)>| -> Vec<(&'static str, GgufValue)> {
        let mut kvs = vec![
            ("general.architecture", GgufValue::Str("qwen2".into())),
            ("qwen2.embedding_length", GgufValue::U32(8)),
            ("qwen2.block_count", GgufValue::U32(1)),
            ("qwen2.attention.head_count", GgufValue::U32(2)),
            ("qwen2.attention.head_count_kv", GgufValue::U32(2)),
            ("qwen2.feed_forward_length", GgufValue::U32(16)),
            ("qwen2.context_length", GgufValue::U32(8192)),
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
    let orig = 2048u32; // "trained" context: well inside the 8192 table capacity above.

    // Build the three (name, cos, sin, d) fixtures via the real GGUF metadata/tensor path.
    let mut variants: Vec<(&str, HostTensor, HostTensor, usize)> = Vec::new();

    let linear_kvs = base_kvs(vec![
        ("qwen2.rope.scaling.type", GgufValue::Str("linear".into())),
        ("qwen2.rope.scaling.factor", GgufValue::F32(4.0)),
    ]);
    let linear_bytes = write_gguf(&linear_kvs, &[]);
    let g = GgufIndex::from_bytes(&linear_bytes).expect("parse linear gguf");
    let store = read_gguf(&g, linear_bytes.as_slice(), &IdentityNames).expect("read linear gguf");
    let cfg = gguf_config(&g, "qwen2", true).expect("gguf_config");
    let (cos, sin) =
        gguf_rope_tables(&g, &store, &cfg, "qwen2").expect("gguf rope tables (linear)");
    variants.push(("gguf-linear", cos, sin, cfg.rotary_dim));

    let yarn_kvs = base_kvs(vec![
        ("qwen2.rope.scaling.type", GgufValue::Str("yarn".into())),
        ("qwen2.rope.scaling.factor", GgufValue::F32(4.0)),
        (
            "qwen2.rope.scaling.original_context_length",
            GgufValue::U32(orig),
        ),
    ]);
    let yarn_bytes = write_gguf(&yarn_kvs, &[]);
    let g = GgufIndex::from_bytes(&yarn_bytes).expect("parse yarn gguf");
    let store = read_gguf(&g, yarn_bytes.as_slice(), &IdentityNames).expect("read yarn gguf");
    let cfg2 = gguf_config(&g, "qwen2", true).expect("gguf_config");
    let (cos, sin) = gguf_rope_tables(&g, &store, &cfg2, "qwen2").expect("gguf rope tables (yarn)");
    variants.push(("gguf-yarn", cos, sin, cfg2.rotary_dim));

    // LongRoPE long regime: phi3 arch, rope_factors_{short,long}.weight tensors, table capacity
    // (context_length) beyond original_context_length so the trace below exercises the long regime.
    const F32: u32 = 0;
    let half = 4usize; // head_dim 16/2 heads=8, no rope.dimension_count key -> rotary_dim=8, half=4
    let short_vals = vec![1.0f32; half];
    let long_vals = vec![1.6f32; half];
    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let phi3_tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "rope_factors_short.weight",
            vec![half as u64],
            F32,
            f32_bytes(&short_vals),
        ),
        (
            "rope_factors_long.weight",
            vec![half as u64],
            F32,
            f32_bytes(&long_vals),
        ),
    ];
    let phi3_kvs = vec![
        ("general.architecture", GgufValue::Str("phi3".into())),
        ("phi3.embedding_length", GgufValue::U32(16)),
        ("phi3.block_count", GgufValue::U32(1)),
        ("phi3.attention.head_count", GgufValue::U32(2)),
        ("phi3.attention.head_count_kv", GgufValue::U32(2)),
        ("phi3.feed_forward_length", GgufValue::U32(16)),
        ("phi3.context_length", GgufValue::U32(8192)), // capacity > orig -> long regime engaged
        (
            "phi3.rope.scaling.original_context_length",
            GgufValue::U32(orig),
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
    let phi3_bytes = write_gguf(&phi3_kvs, &phi3_tensors);
    let g = GgufIndex::from_bytes(&phi3_bytes).expect("parse phi3 gguf");
    let store = read_gguf(&g, phi3_bytes.as_slice(), &IdentityNames).expect("read phi3 gguf");
    let cfg3 = gguf_config(&g, "phi3", false).expect("gguf_config");
    assert_eq!(cfg3.rotary_dim, 8, "head_dim 16/2 heads=8 -> full rotary");
    let (cos, sin) =
        gguf_rope_tables(&g, &store, &cfg3, "phi3").expect("gguf rope tables (longrope)");
    variants.push(("gguf-longrope-long-regime", cos, sin, cfg3.rotary_dim));

    for (name, cos, sin, d) in variants {
        let cap = cos.shape()[0];
        assert!(
            cos.as_f32().unwrap().iter().all(|v| v.is_finite())
                && sin.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "{name}: rope table itself has a non-finite entry"
        );

        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
        let cos_c = b.constant("cos", TensorType::f32(vec![cap, d]));
        let sin_c = b.constant("sin", TensorType::f32(vec![cap, d]));
        let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
        let out = rope(&b, x, cos_c, sin_c, pos);
        let (xi, ci, si, pi) = (x.id, cos_c.id, sin_c.id, pos.id);
        let g = b.finish(out);

        let xd: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.037).sin()).collect();
        let run_at = |p: usize| {
            let mut inputs: HashMap<_, Value> = HashMap::new();
            inputs.insert(xi, HostTensor::f32(vec![1, 1, 1, d], xd.clone()).into());
            inputs.insert(ci, cos.clone().into());
            inputs.insert(si, sin.clone().into());
            inputs.insert(pi, HostTensor::i32(vec![], vec![p as i32]).into());
            poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("eval")
                .output
                .into_host()
                .expect("dense output")
        };

        let near = run_at(16);
        let mid = run_at((orig as usize) / 2);
        let far = run_at(cap - 1);
        assert!(
            near.as_f32().unwrap().iter().all(|v| v.is_finite())
                && mid.as_f32().unwrap().iter().all(|v| v.is_finite())
                && far.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "{name}: non-finite output somewhere in near/mid/far"
        );
        assert_ne!(
            near.as_f32().unwrap(),
            mid.as_f32().unwrap(),
            "{name}: two different in-context positions produced identical output"
        );
        assert_ne!(
            mid.as_f32().unwrap(),
            far.as_f32().unwrap(),
            "{name}: a long-context position produced the same output as a mid-context one"
        );
    }
}
