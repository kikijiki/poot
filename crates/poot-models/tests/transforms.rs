//! Integration of `poot-graph-ir`'s graph transforms with a real model graph. These live here rather than in
//! `poot-graph-ir` because they need a model tracer, which depends on the IR crate.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalError, EvalOptions, Value, eval};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::{
    BinOp, Eqn, ExecutionValidationFailure, Graph, MAX_VALIDATION_PACKET_BYTES, OpKind, Operand,
    Storage, ValidationId, ValidationOutput, ValidationOutputs, ValueId, ValueMeta,
};
use poot_graph_plan::{collapse_reshape_chains, cse, dce, elide_noop_transposes, fuse};
use poot_models::model::{LogitRows, Phase};
use poot_tensor::HostTensor;

/// A tiny qwen2 decode graph: `layers` layers over a cache of `cap` positions.
fn qwen2_decode(dense: &Dense, cap: usize) -> Graph {
    plain(
        dense
            .f32_model()
            .model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .expect("the qwen2 decode traces"),
    )
}

/// The qwen2 decode the transform rows run over: repeated layers, so CSE has work.
fn deep_qwen2_decode() -> Graph {
    qwen2_decode(
        &Dense::new(Family::Qwen2)
            .vocab(32)
            .dims(32, 64, 4)
            .heads(4, 2)
            .max_positions(16),
        9,
    )
}

#[test]
fn cse_shrinks_qwen2_and_stays_valid() {
    let g = deep_qwen2_decode();
    let c = cse(&g);
    c.validate().expect("cse graph should validate");
    assert!(
        c.eqns.len() < g.eqns.len(),
        "cse should remove eqns: {} -> {}",
        g.eqns.len(),
        c.eqns.len()
    );
    // output type unchanged.
    assert_eq!(c.aval(c.output).shape, g.aval(g.output).shape);
}

#[test]
fn cse_is_idempotent_on_qwen2() {
    let g = deep_qwen2_decode();
    let c1 = cse(&g);
    let c2 = cse(&c1);
    assert_eq!(c1.eqns.len(), c2.eqns.len());
}

#[test]
fn dce_keeps_qwen2_valid() {
    let g = deep_qwen2_decode();
    let d = dce(&g);
    d.validate().expect("dce graph should validate");
    assert_eq!(d.aval(d.output).shape, g.aval(g.output).shape);
}

#[test]
fn qwen2_value_preserving_transforms_keep_validation_bits() {
    let mut traced = qwen2_decode(
        &Dense::new(Family::Qwen2)
            .vocab(16)
            .dims(8, 16, 1)
            .heads(2, 2)
            .head_dim(4)
            .max_positions(8),
        2,
    );

    // Computed witnesses downstream of a real MatMul: `exact = h - h` is exactly zero for finite `h`; `fused = exact * h`
    // is a single-use pointwise consumer that fusion would absorb `exact` into unless validation roots are pinned;
    // `duplicate = h - h` is a structural duplicate CSE must remap; `failing = h * h + h` is a nonzero witness through a
    // fusible chain, declared by a second graph so the reported failure pins real nonzero bits across transforms.
    let h = traced
        .eqns
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias))
        .map(|eqn| eqn.out)
        .filter(|&value| traced.aval(value).dtype == DType::F32)
        .min_by_key(|&value| traced.aval(value).numel())
        .expect("tiny qwen2 decode has an F32 matmul output");
    let lanes = traced.aval(h).numel();
    assert!(
        (1..=MAX_VALIDATION_PACKET_BYTES / 16).contains(&lanes),
        "four witnesses of {lanes} lanes must fit the packet"
    );
    let mut append = |op: BinOp, left: ValueId, right: ValueId| -> ValueId {
        let out = traced.values.len();
        traced.values.push(ValueMeta::new(
            traced.aval(h).clone(),
            Storage::Device,
            None,
        ));
        traced.eqns.push(Eqn {
            op: OpKind::Binary(op),
            inputs: vec![Operand::Value(left), Operand::Value(right)],
            out,
            layer: None,
        });
        out
    };
    let exact = append(BinOp::Sub, h, h);
    let fused = append(BinOp::Mul, exact, h);
    let duplicate = append(BinOp::Sub, h, h);
    let square = append(BinOp::Mul, h, h);
    let failing = append(BinOp::Add, square, h);
    let declarations = |entries: &[(ValidationId, &str, ValueId)]| -> Vec<ValidationOutput> {
        entries
            .iter()
            .map(|&(id, name, value)| ValidationOutput {
                id,
                name: name.into(),
                value,
            })
            .collect()
    };
    let passing = [
        (ValidationId(31), "exact", exact),
        (ValidationId(32), "fused", fused),
        (ValidationId(33), "duplicate", duplicate),
    ];
    let failing_graph = traced.clone().with_validations(declarations(
        &[&passing[..], &[(ValidationId(34), "failing", failing)]].concat(),
    ));
    failing_graph.validate().unwrap();
    let graph = traced.with_validations(declarations(&passing));
    graph.validate().unwrap();

    let inputs: HashMap<ValueId, HostTensor> = graph
        .inputs
        .iter()
        .map(|&value| {
            let shape = graph.aval(value).shape.clone();
            let is_slot = matches!(graph.meta(value).storage, Storage::Slot(_));
            let tensor = if graph.aval(value).dtype == DType::I32 {
                let ints = if is_slot {
                    vec![0i32; graph.aval(value).numel()]
                } else {
                    (0..graph.aval(value).numel())
                        .map(|lane| ((value.wrapping_mul(17) + lane * 13) % 11) as i32 - 5)
                        .collect()
                };
                HostTensor::i32(shape, ints)
            } else {
                let data = if is_slot {
                    vec![0.0; graph.aval(value).numel()]
                } else {
                    (0..graph.aval(value).numel())
                        .map(|lane| {
                            let centered = ((value.wrapping_mul(17) + lane * 13) % 11) as f32 - 5.0;
                            centered * 0.01
                        })
                        .collect()
                };
                HostTensor::f32(shape, data)
            };
            (value, tensor)
        })
        .collect();

    fn execution_snapshot(
        graph: &Graph<ValidationOutputs>,
        inputs: &HashMap<ValueId, HostTensor>,
    ) -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
        let inputs_v: HashMap<ValueId, Value> =
            inputs.iter().map(|(&k, v)| (k, v.clone().into())).collect();
        let result = eval(
            graph,
            &inputs_v,
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .expect("validation-bearing graph evaluates");
        let environment: Vec<Option<HostTensor>> = result
            .environment
            .expect("keep_environment was set")
            .into_iter()
            .map(|slot| slot.map(|v| v.into_host().expect("dense environment slot")))
            .collect();
        let packet = graph
            .validation_outputs()
            .iter()
            .flat_map(|validation| {
                environment[validation.value]
                    .as_ref()
                    .expect("validation result is materialized")
                    .as_f32()
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
            })
            .collect();
        let primary = result.output.into_host().expect("dense primary output");
        let state: Vec<HostTensor> = result
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dense state"))
            .collect();
        (
            primary
                .as_f32()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            packet,
            state
                .iter()
                .map(|tensor| {
                    tensor
                        .as_f32()
                        .unwrap()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect()
                })
                .collect(),
        )
    }

    let expected = execution_snapshot(&graph, &inputs);
    let failure = |graph: &Graph<ValidationOutputs>| -> ExecutionValidationFailure {
        let inputs_v: HashMap<ValueId, Value> =
            inputs.iter().map(|(&k, v)| (k, v.clone().into())).collect();
        match eval(graph, &inputs_v, EvalOptions::new(EvalBudget::UNBOUNDED)) {
            Err(EvalError::Validation(failure)) => failure,
            other => panic!("expected a validation failure, got {other:?}"),
        }
    };
    let expected_failure = failure(&failing_graph);
    assert_eq!(expected_failure.id, ValidationId(34));
    assert_ne!(
        expected_failure.observed_bits & 0x7fff_ffff,
        0,
        "the failing witness must report nonzero bits"
    );
    type Transform = fn(&Graph<ValidationOutputs>) -> Graph<ValidationOutputs>;
    let transforms: [(&str, Transform); 5] = [
        ("cse", cse),
        ("dce", dce),
        ("fusion", fuse),
        ("reshape cleanup", collapse_reshape_chains),
        ("transpose elision", elide_noop_transposes),
    ];
    for (name, transform) in transforms {
        let transformed = transform(&graph);
        transformed.validate().unwrap();
        assert_eq!(
            poot_test_util::graph_fixtures::validation_packet_layout(&transformed).unwrap(),
            poot_test_util::graph_fixtures::validation_packet_layout(&graph).unwrap(),
            "{name} changed the canonical packet layout"
        );
        assert_eq!(
            execution_snapshot(&transformed, &inputs),
            expected,
            "{name} changed primary values, validation bits, or state"
        );
        assert_eq!(
            failure(&transform(&failing_graph)),
            expected_failure,
            "{name} changed the reported validation failure"
        );
    }
    assert_eq!(
        cse(&graph).validation_outputs()[2].value,
        exact,
        "CSE must remap the duplicate witness to its canonical producer"
    );
}
