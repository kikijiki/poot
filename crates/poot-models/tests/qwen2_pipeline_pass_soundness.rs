//! CSE/fuse preservation on a real qwen2 decomposition, moved
//! from `poot-eval/src/tests/qwen2_pipeline.rs` with the passes themselves (card 626) - poot-eval
//! must never depend on poot-graph-plan (its own architecture test); this crate already dev-depends
//! on poot-eval, so an integration test here can drive both.

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::op::OpKind;
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_graph_plan::{cse, fuse};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::test_support::dense::{Dense, Family, step};
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

/// The tiny qwen2 the pass tests trace (through the registry): the same dimensions the decomposition
/// tests always used.
fn tiny_qwen2() -> poot_models::test_support::weight_map::MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .f32_model()
}

/// `model`'s one-token decode step over `cap` positions.
fn decode_graph(model: &poot_models::test_support::weight_map::MappedModel, cap: usize) -> Graph {
    model
        .model
        .trace(
            &StepSpec::step(step(1, 1, cap, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// Every input of `g` bound: the token and position slots as one decode step at `pos`, state and
/// weights filled by input index (weights scaled by 0.1).
fn bind_decode(g: &Graph, token: i32, pos: i32) -> HashMap<ValueId, HostTensor> {
    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    for (i, &id) in g.inputs().iter().enumerate() {
        let aval = g.aval(id).clone();
        let t = match g.values()[id].storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(aval.shape, vec![token]),
            Storage::Slot(Slot::Pos) => HostTensor::i32(aval.shape, vec![pos]),
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Computed(c) => c.tensor(),
            Storage::Const | Storage::State => {
                let data: Vec<f32> = fill(aval.numel(), 100 + i as u64)
                    .iter()
                    .map(|v| v * 0.1)
                    .collect();
                HostTensor::f32(aval.shape, data)
            }
            Storage::Device => unreachable!(),
        };
        inputs.insert(id, t);
    }
    inputs
}

#[test]
fn cse_preserves_results() {
    // CSE only removes redundant identical computations, so eval(cse(g)) is bit-identical to eval(g).
    let pos = 3usize;
    let g = decode_graph(&tiny_qwen2(), 8);
    let c = cse(&g).unwrap();
    assert!(c.eqns().len() < g.eqns().len());

    let inputs = bind_decode(&g, 5, pos as i32);
    let a = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&g, &values, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    let b = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&c, &values, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    assert_eq!(
        a.as_f32().unwrap(),
        b.as_f32().unwrap(),
        "CSE changed the result"
    );
}

#[test]
fn fuse_preserves_qwen2_results_and_shrinks() {
    // G5 executor-equivalence oracle on a real decomposition: eval(fuse(g)) is bit-identical to eval(g) on the tiny qwen2
    // decode graph, with fewer eqns, and both fusion kinds fired: pointwise Fused regions (rope / residual chains) and
    // reduction-rooted FusedRow regions (each layer's RMSNorm + softmax).
    let pos = 3usize;
    let g = decode_graph(&tiny_qwen2(), 8);
    let fg = fuse(&g).unwrap();
    assert!(
        fg.eqns().len() < g.eqns().len(),
        "fusion should drop eqns: {} -> {}",
        g.eqns().len(),
        fg.eqns().len()
    );
    let n_fused = fg
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Fused(_)))
        .count();
    let n_row = fg
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::FusedRow(_)))
        .count();
    assert!(n_fused > 0, "pointwise fusion should fire");
    assert!(
        n_row > 0,
        "reduction-rooted fusion should fire (RMSNorm + softmax)"
    );
    eprintln!(
        "qwen2 tiny fuse: {} eqns -> {} ({} Fused, {} FusedRow)",
        g.eqns().len(),
        fg.eqns().len(),
        n_fused,
        n_row
    );

    let inputs = bind_decode(&g, 5, pos as i32);
    let a = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&g, &values, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    let b = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&fg, &values, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    assert_eq!(
        a.as_f32().unwrap(),
        b.as_f32().unwrap(),
        "fusion changed the result"
    );
}
