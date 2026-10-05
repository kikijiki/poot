//! Correction to spec 267's "out of scope" claim that `rope_fusion`'s matcher does not fire on sectioned rope: it does.
//! `match_rope_core` (`poot-graph-plan`'s passes) matches only the rotate-half chain rooted at `x` plus a shape check on
//! cos/sin's trailing width, never how cos/sin were produced. `rope_sectioned_partial` reuses `rope_partial` for that
//! chain, so the matcher collapses it to `OpKind::Rope` with the `Concat`-of-sliced-per-section-gathers as its
//! cos/sin operand. This is safe: `OpKind::Rope`'s eval (`poot_eval::lib`) uses whatever cos/sin values it is given.
//! This test checks it empirically (bit-exact fused vs unfused).
//!
//! Card 626: moved here from `poot-eval/src/tests/mrope.rs` with the pass itself - poot-eval must
//! never depend on poot-graph-plan (its own architecture test); this crate already dev-depends on
//! poot-eval, so an integration test here can drive both.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::Builder;
use poot_graph_ir::ops::rope_sectioned;
use poot_graph_ir::types::TensorType;
use poot_graph_plan::{cse, rope_fusion};
use poot_tensor::DType;
use poot_tensor::HostTensor;
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

#[test]
fn rope_fusion_matches_and_is_result_preserving_on_sectioned_rope() {
    let b = Builder::new();
    let d = 16usize;
    let max_pos = 16usize;
    let sections = [2usize, 3, 3];
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos_t = b.constant("pos_t", TensorType::scalar(DType::I32));
    let pos_h = b.constant("pos_h", TensorType::scalar(DType::I32));
    let pos_w = b.constant("pos_w", TensorType::scalar(DType::I32));
    let out = rope_sectioned(&b, x, cos, sin, &[pos_t, pos_h, pos_w], &sections);
    let (xi, ci, si, pti, phi, pwi) = (x.id, cos.id, sin.id, pos_t.id, pos_h.id, pos_w.id);
    let g = b.finish(out);
    let g2 = rope_fusion(&cse(&g));
    let has_rope_op = g2
        .eqns
        .iter()
        .any(|e| matches!(e.op, poot_graph_ir::op::OpKind::Rope { .. }));
    assert!(
        has_rope_op,
        "rope_fusion DOES match the sectioned composition's rotate-half chain (contrary to spec 267's \
         out-of-scope claim) - it just doesn't need a dedicated fusion recognizer since match_rope_core is \
         cos/sin-provenance-agnostic"
    );

    let xd = fill(d, 201);
    let cosd = fill(max_pos * d, 202);
    let sind = fill(max_pos * d, 203);
    let (pt, ph, pw) = (2usize, 9usize, 13usize);
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
    );
    inputs.insert(
        ci,
        Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
    );
    inputs.insert(
        si,
        Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
    );
    inputs.insert(pti, Value::from(HostTensor::i32(vec![], vec![pt as i32])));
    inputs.insert(phi, Value::from(HostTensor::i32(vec![], vec![ph as i32])));
    inputs.insert(pwi, Value::from(HostTensor::i32(vec![], vec![pw as i32])));

    let want = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = eval(&g2, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let wb: Vec<u32> = want.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    let gb: Vec<u32> = got.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        wb, gb,
        "rope_fusion must be result-preserving (invariant 6) on the sectioned composition, even though it \
         was not specifically designed to recognize it"
    );
}
