//! Card 536b (R481-007): pointwise-fusion legality admits `Recip`, `Tanh` and `Erf`, proven against
//! the real CPU oracle (`poot_eval::eval`).
//!
//! Integration test for the same reason as `transform_soundness.rs` (card 536a, see its header): the
//! crate dev-depends on `poot-eval`, which depends back on it, so a `#[cfg(test)]` unit test would see a
//! foreign `Graph` type. Only the oracle comparison lives here, through APIs already public before this
//! card (`fuse`, `OpKind` matching). The structural fusion-count assertion and the packed-recognizer
//! evidence (R467-009, which needs no oracle - a declined near-miss leaves the graph's equations
//! untouched, and the recognized case is checked by its declared type) stay `#[cfg(test)]` unit tests in
//! `crates/poot-graph-plan/src/passes/{fuse.rs,packed.rs}`.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Builder, Graph, OpKind, TensorType, UnOp};
use poot_graph_plan::fuse;
use poot_tensor::HostTensor;

fn bind_inputs(g: &Graph) -> HashMap<usize, HostTensor> {
    g.inputs
        .iter()
        .map(|&id| {
            let shape = g.aval(id).shape.clone();
            let n = shape.iter().product::<usize>().max(1);
            let data = (0..n)
                .map(|i| ((i as f32) * 0.171 - 0.4).sin() * 0.5)
                .collect();
            (id, HostTensor::f32(shape, data))
        })
        .collect()
}

/// ADR-0101 elementwise comparison: a non-finite value where the other side is finite fails, and NaN
/// always fails.
fn assert_close(a: &HostTensor, b: &HostTensor, tol: f32) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    for (i, (x, y)) in a
        .as_f32()
        .unwrap()
        .iter()
        .zip(b.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite element {i}: {x} vs {y}"
        );
        let diff = (x - y).abs();
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(diff <= tol * scale, "element {i}: {x} vs {y} (diff {diff})");
    }
}

/// SC-002 (R481-007): a `Tanh`/`Erf`/`Recip` pointwise chain fuses into one region (see
/// `passes::fuse::tests::recip_tanh_erf_fuse_into_one_region` for the structural half) and
/// matches the oracle within the tier-2 (`Reassociating`) tolerance. Mutation: drop one of the three
/// from `FUSABLE_FLOAT_UNARY_OPS`; the region no longer covers all three steps and this row's shape
/// assertion (one eqn) goes red.
#[test]
fn recip_tanh_erf_fused_region_matches_the_oracle() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 8]));
    let g = b.unary(UnOp::Tanh, x);
    let e = b.unary(UnOp::Erf, g);
    let r = b.unary(UnOp::Recip, e);
    let graph = b.finish(r);

    let fused = fuse(&graph);
    assert_eq!(fused.eqns.len(), 1, "expected one fused region: {fused:?}");
    assert!(matches!(fused.eqns[0].op, OpKind::Fused(_)));

    let inputs: HashMap<usize, Value> = bind_inputs(&graph)
        .into_iter()
        .map(|(k, v)| (k, Value::from(v)))
        .collect();
    let raw = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let fused_out = eval(&fused, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_close(&raw, &fused_out, 1e-4);
}
