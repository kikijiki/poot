//! Card 198: `FlashAttentionPrefill{softcap: Some(c)}` must reproduce `ops::attention_prefill_softcap`'s
//! `c * tanh(scores/c)` clamp exactly. Gemma2 numbers: `d=256` (`gemma2_2b()` head_dim, also the imported
//! flash-prefill kernel's LDS cap), `cap=50.0` (`attn_logit_softcap`), `n_rep=4`/`hkv=2` as a representative GQA ratio.
//!
//! Q/K are `fill`'s `[-1,1)` amplified 8x so raw scaled scores reach into and past the cap; the sensitivity
//! assertion below proves softcap clamps (otherwise the test would pass with softcap silently dropped).
//!
//! Card 626: moved here from `poot-eval/src/tests/linear_attention_hybrid.rs` with the pass itself -
//! poot-eval must never depend on poot-graph-plan (its own architecture test); this crate already
//! dev-depends on poot-eval, so an integration test here can drive both.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::attention_prefill_softcap;
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;
use poot_test_util::{max_abs_error, max_abs_error_f64};
use std::collections::HashMap;

/// Deterministic pseudo-random fill in [-1, 1), no rng dependency (poot-eval's own test helper of the
/// same name, duplicated: this is an external integration test, so it cannot reach poot-eval's
/// `pub(super)` test helpers).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

use poot_graph_plan::passes_without_target as optimize;

#[test]
fn flash_prefill_softcap_matches_attention_prefill_softcap_and_f64_reference() {
    let (hkv, n_rep, d, l) = (2usize, 4usize, 256usize, 64usize);
    let hq = hkv * n_rep;
    let scale = 1.0 / (d as f32).sqrt();
    let cap = 50.0f32;

    let mut qd = fill(hq * l * d, 0x5A17_C0DE);
    let mut kd = fill(hkv * l * d, 0x0B16_B00B);
    let vd = fill(hkv * l * d, 0xFEED_FACE);
    for x in qd.iter_mut() {
        *x *= 24.0;
    }
    for x in kd.iter_mut() {
        *x *= 24.0;
    }
    // additive causal mask [1,1,L,L]: 0 if j<=t else -1e9.
    let mask: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();

    // `materialized(softcap_opt)` traces the plain decomposition and evals it as traced (the CPU oracle).
    // `autofused(softcap_opt)` runs the same decomposition through `optimize()` above, the pass pipeline
    // every real tracer goes through (fusion is always automatic, per AGENTS.md), exercising
    // `match_attention` on the softcap chain and the fused op's eval numerics end to end.
    let trace =
        |softcap_opt: Option<f32>| -> (poot_graph_ir::graph::Graph, HashMap<ValueId, Value>) {
            let b = Builder::new();
            let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
            let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
            let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
            let m = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
            let o = attention_prefill_softcap(&b, q, k, v, n_rep, scale, m, softcap_opt);
            let g = b.finish(o);
            let mut inp = HashMap::new();
            inp.insert(
                q.id,
                Value::from(HostTensor::f32(vec![1, hq, l, d], qd.clone())),
            );
            inp.insert(
                k.id,
                Value::from(HostTensor::f32(vec![1, hkv, l, d], kd.clone())),
            );
            inp.insert(
                v.id,
                Value::from(HostTensor::f32(vec![1, hkv, l, d], vd.clone())),
            );
            inp.insert(
                m.id,
                Value::from(HostTensor::f32(vec![1, 1, l, l], mask.clone())),
            );
            (g, inp)
        };
    let materialized = |softcap_opt: Option<f32>| -> HostTensor {
        let (g, inp) = trace(softcap_opt);
        eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs")
    };
    let autofused = |softcap_opt: Option<f32>| -> HostTensor {
        let (g, inp) = trace(softcap_opt);
        let opt = optimize(&g);
        // Structural check that fusion fired: exactly one flash-prefill op, carrying the same softcap passed in.
        let flash_ops: Vec<&poot_graph_ir::op::OpKind> = opt
            .eqns
            .iter()
            .map(|e| &e.op)
            .filter(|op| matches!(op, poot_graph_ir::op::OpKind::FlashAttentionPrefill { .. }))
            .collect();
        assert_eq!(
            flash_ops.len(),
            1,
            "expected optimize() to fuse this attention_prefill_softcap chain into exactly one flash op"
        );
        let got_softcap = match flash_ops[0] {
            poot_graph_ir::op::OpKind::FlashAttentionPrefill { softcap, .. } => *softcap,
            _ => unreachable!(),
        };
        assert_eq!(
            got_softcap, softcap_opt,
            "fused op's softcap parameter does not match the traced graph's attn_logit_softcap"
        );
        eval(&opt, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs")
    };

    let materialized_capped = materialized(Some(cap));
    let fused_capped = autofused(Some(cap));
    let materialized_uncapped = materialized(None);

    // Sensitivity: cap=50.0 at this magnitude must clip. If this fails, strengthen the amplification above;
    // otherwise the equivalence check below would also pass with softcap silently dropped.
    let drop_diff = max_abs_error(
        materialized_capped.as_f32().unwrap(),
        materialized_uncapped.as_f32().unwrap(),
    );
    assert!(
        drop_diff > 1.0,
        "chosen Q/K magnitude does not make cap=50.0 bite (materialized capped vs uncapped \
         max_abs={drop_diff}) - this test cannot detect a dropped softcap at this magnitude"
    );

    // The claim: FlashAttentionPrefill{softcap: Some(c)} == the materialized decomposition.
    let fuse_err = max_abs_error(
        fused_capped.as_f32().unwrap(),
        materialized_capped.as_f32().unwrap(),
    );
    assert!(
        fuse_err < 1e-3,
        "flash-fused softcapped prefill vs materialized attention_prefill_softcap: max_abs={fuse_err}"
    );

    // Catches a dropped softcap: if eval ignored `softcap`, `fused_capped` would equal `materialized_uncapped`. Since
    // `drop_diff > 1.0` (above) and `fuse_err < 1e-3`, the fused result cannot also be within 1e-3 of the uncapped oracle.
    let would_be_wrong_if_dropped = max_abs_error(
        fused_capped.as_f32().unwrap(),
        materialized_uncapped.as_f32().unwrap(),
    );
    assert!(
        would_be_wrong_if_dropped > 1.0,
        "fused result is suspiciously close to the NO-softcap oracle ({would_be_wrong_if_dropped}) - \
         something is wrong with this test's sensitivity, not just the fix"
    );

    // Catches a double-applied softcap: softcap is not idempotent once tanh saturates, c*tanh(c*tanh(x/c)/c) != c*tanh(x/c).
    // Demonstrated on real data from this test (the raw scaled score at [head=0, row=l-1, key=0]).
    let mut raw0 = 0.0f32;
    for c in 0..d {
        raw0 += qd[((hq - 1) * l + (l - 1)) * d + c] * kd[((hkv - 1) * l) * d + c];
    }
    raw0 *= scale;
    let once = cap * (raw0 / cap).tanh();
    let twice = cap * (once / cap).tanh();
    let double_apply_gap = (once - twice).abs();
    eprintln!(
        "sensitivity: drop_diff={drop_diff} fuse_err={fuse_err} \
         would_be_wrong_if_dropped={would_be_wrong_if_dropped} raw0={raw0} once={once} twice={twice} \
         double_apply_gap={double_apply_gap}"
    );
    assert!(
        double_apply_gap > 0.1,
        "softcap is not sufficiently non-idempotent at this sample's magnitude (once={once} \
         twice={twice} gap={double_apply_gap}) to prove a double-applied softcap would be caught by the \
         fuse_err < 1e-3 assertion above - the raw score magnitude needs to be closer to/past cap=50.0"
    );

    // Independent f64 ground truth: causal softmax(c*tanh(scale*QK^T/c)+mask)@V per (head, row), GQA-repeated, no poot code.
    let mut want = vec![0.0f64; hq * l * d];
    let cap64 = cap as f64;
    for hqi in 0..hq {
        let hk = hqi / n_rep;
        for i in 0..l {
            let mut scores = vec![0.0f64; i + 1];
            for j in 0..=i {
                let mut dot = 0.0f64;
                for c in 0..d {
                    dot += (qd[(hqi * l + i) * d + c] as f64) * (kd[(hk * l + j) * d + c] as f64);
                }
                let raw = dot * (scale as f64);
                scores[j] = cap64 * (raw / cap64).tanh();
            }
            let m64 = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - m64).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for c in 0..d {
                let mut acc = 0.0f64;
                for j in 0..=i {
                    acc += (exps[j] / sum) * (vd[(hk * l + j) * d + c] as f64);
                }
                want[(hqi * l + i) * d + c] = acc;
            }
        }
    }
    let worst_materialized = max_abs_error_f64(materialized_capped.as_f32().unwrap(), &want);
    let worst_fused = max_abs_error_f64(fused_capped.as_f32().unwrap(), &want);
    eprintln!(
        "flash softcapped prefill at real head_dim=256, cap=50.0 vs f64 ground truth: \
         materialized={worst_materialized:e} fused={worst_fused:e}"
    );
    assert!(
        worst_fused < 1e-3,
        "flash-fused softcapped prefill vs independent f64 ground truth: err={worst_fused:e}"
    );
}
