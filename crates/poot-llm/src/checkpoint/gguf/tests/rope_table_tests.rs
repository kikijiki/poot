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

use crate::checkpoint::gguf::rope::rope_tables;
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
