//! Numerical legality through the production compiler and the existing primitive oracle.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{BinOp, Builder, Graph, OpKind, Scalar, StateRole, TensorType, UnOp};
use poot_graph_plan::canonicalize;
use poot_graph_plan::{CompileOptions, FusionPolicy, Submission, Target, compile};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::HostTensor;

/// Card 554d: `eval`'s full environment, dense at every slot this suite ever reads
/// back, inlined at each call site below rather than kept behind a renamed `eval_all` shim - the
/// deleted convenience's call shape (`env[value_id]` as `&Tensor`), rebuilt from
/// `eval(..).environment` plus `Value::into_dense`.
const OPTIONS: CompileOptions = CompileOptions {
    execution: Submission::Replay,
    fusion: FusionPolicy::Full,
    limits: poot_graph_plan::CompileLimits::STANDARD,
};

fn target(backend: Backend) -> Target {
    Target {
        backend,
        caps: match backend {
            Backend::SpirvVulkan => DeviceCaps::wgpu_rdna3_igpu(),
            Backend::Nvptx => DeviceCaps::ptx_default(),
            Backend::AmdGcn(_) => DeviceCaps::rocm_default(),
        },
    }
}

/// Compare every word of every published result, including both carried-state buffers. A NaN is
/// never accepted as numerical parity. Non-finite input semantics have separate classification tests.
fn assert_observables(graph: &Graph, transformed: &Graph, inputs: &HashMap<usize, HostTensor>) {
    let values: HashMap<usize, Value> = inputs
        .iter()
        .map(|(&k, v)| (k, Value::from(v.clone())))
        .collect();
    let before: Vec<Option<HostTensor>> = eval(
        graph,
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .expect("primitive oracle")
    .environment
    .unwrap()
    .into_iter()
    .map(|v| v.map(|v| v.into_host().unwrap()))
    .collect();
    let after: Vec<Option<HostTensor>> = eval(
        transformed,
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .expect("transformed oracle")
    .environment
    .unwrap()
    .into_iter()
    .map(|v| v.map(|v| v.into_host().unwrap()))
    .collect();
    let roots: Vec<_> = graph.liveness_roots().collect();
    let mapped: Vec<_> = transformed.liveness_roots().collect();
    assert_eq!(roots.len(), mapped.len(), "all outputs and states survive");
    for (root, mapped) in roots.into_iter().zip(mapped) {
        assert_eq!(
            graph.aval(root),
            transformed.aval(mapped),
            "result dtype and shape"
        );
        let a = before[root].as_ref().expect("primitive result");
        let b = after[mapped].as_ref().expect("transformed result");
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.as_f32().unwrap().len(), b.as_f32().unwrap().len());
        for (index, (&a, &b)) in a
            .as_f32()
            .unwrap()
            .iter()
            .zip(b.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                !a.is_nan() && !b.is_nan(),
                "NaN at result {root}, word {index}"
            );
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "result {root}, word {index}: {a} -> {b}"
            );
        }
    }
}

fn with_states(b: Builder, output: poot_graph_ir::Traced) -> Graph {
    let ty = b.aval(output);
    let first = b.state_input("first", ty.clone(), StateRole::Recurrent);
    let second = b.state_input("second", ty, StateRole::Recurrent);
    let next_first = b.binary(BinOp::Add, output, first);
    let next_second = b.binary(BinOp::Sub, second, output);
    b.finish_with_state(output, &[(first, next_first), (second, next_second)])
}

fn bind_states(graph: &Graph, inputs: &mut HashMap<usize, HostTensor>) {
    for (index, pair) in graph.state_pairs().enumerate() {
        let shape = graph.aval(pair.input).shape.clone();
        let count = shape.iter().product();
        inputs.insert(
            pair.input,
            HostTensor::f32(shape, vec![index as f32 + 2.0; count]),
        );
    }
}

#[test]
fn finite_scaled_contraction_survives_compile() {
    // The preserved review reproduction, unchanged in size and arithmetic.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([1, 1]));
    let w = b.constant("w", TensorType::f32([1, 1]));
    let scaled = b.binary_scalar(BinOp::Mul, x, Scalar::F32(1.0e-30));
    let output = b.matmul(scaled, w);
    let graph = with_states(b, output);
    let mut inputs = HashMap::from([
        (x.id, HostTensor::f32(vec![1, 1], vec![1.0e30])),
        (w.id, HostTensor::f32(vec![1, 1], vec![1.0e30])),
    ]);
    bind_states(&graph, &mut inputs);
    let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
    let values: HashMap<usize, Value> = inputs
        .iter()
        .map(|(&k, v)| (k, Value::from(v.clone())))
        .collect();
    let original: Vec<Option<HostTensor>> = eval(
        &graph,
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap()
    .into_iter()
    .map(|v| v.map(|v| v.into_host().unwrap()))
    .collect();
    let compiled: Vec<Option<HostTensor>> = eval(
        program.graph(),
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap()
    .into_iter()
    .map(|v| v.map(|v| v.into_host().unwrap()))
    .collect();
    let original = &original[graph.output].as_ref().unwrap().as_f32().unwrap();
    let compiled = &compiled[program.graph().output]
        .as_ref()
        .unwrap()
        .as_f32()
        .unwrap();
    println!("original={original:?} compiled={compiled:?}");
    assert_eq!(original[0].to_bits(), 1e30f32.to_bits());
    assert!(
        compiled[0].is_finite(),
        "compiler changed finite result to infinity"
    );
    assert_observables(&graph, program.graph(), &inputs);
}

#[test]
fn scalar_contraction_corner_corpus_is_bitwise() {
    let cases = [
        ("overflow", vec![1e30, 1e30], vec![1e30, -1e30], 1e-30),
        ("underflow lhs", vec![1e-30, 0.0], vec![1e30, 1e30], 1e-20),
        ("underflow rhs", vec![1e30, 1e30], vec![1e-30, 0.0], 1e-20),
        ("cancellation", vec![1.0000001, -1.0], vec![1.0, 1.0], 3.0),
        ("signed zero", vec![0.0, -0.0], vec![1.0, 1.0], -1.0),
        (
            "subnormal",
            vec![f32::MIN_POSITIVE, -f32::MIN_POSITIVE],
            vec![1.0, 0.5],
            0.5,
        ),
        (
            "identity",
            vec![f32::from_bits(1), -0.0],
            vec![1.0, -1.0],
            1.0,
        ),
    ];
    for (name, x_data, w_data, scale) in cases {
        for scale_rhs in [false, true] {
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32([2, 2]));
            let w = b.constant("w", TensorType::f32([2, 2]));
            let scaled = b.binary_scalar(
                BinOp::Mul,
                if scale_rhs { w } else { x },
                Scalar::F32(scale),
            );
            let result = if scale_rhs {
                b.matmul(x, scaled)
            } else {
                b.matmul(scaled, w)
            };
            let graph = with_states(b, result);
            let mut inputs = HashMap::from([
                (x.id, HostTensor::f32(vec![2, 2], x_data.repeat(2))),
                (
                    w.id,
                    HostTensor::f32(vec![2, 2], vec![w_data[0], w_data[0], w_data[1], w_data[1]]),
                ),
            ]);
            bind_states(&graph, &mut inputs);
            println!("case={name}, scale_rhs={scale_rhs}");
            assert_observables(&graph, &canonicalize(&graph), &inputs);
            let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
            assert_observables(&graph, program.graph(), &inputs);
        }
    }
}

#[test]
fn reciprocal_rounding_is_not_division() {
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32([4]));
    let d = b.constant("d", TensorType::f32([4]));
    let reciprocal = b.unary(UnOp::Recip, d);
    let output = b.binary(BinOp::Mul, a, reciprocal);
    let graph = with_states(b, output);
    let mut inputs = HashMap::from([
        (a.id, HostTensor::f32(vec![4], vec![3.0, -3.0, 0.0, -0.0])),
        (d.id, HostTensor::f32(vec![4], vec![7.0, 7.0, -3.0, -3.0])),
    ]);
    bind_states(&graph, &mut inputs);
    let values: HashMap<usize, Value> = inputs
        .iter()
        .map(|(&k, v)| (k, Value::from(v.clone())))
        .collect();
    let before: Vec<Option<HostTensor>> = eval(
        &graph,
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap()
    .into_iter()
    .map(|v| v.map(|v| v.into_host().unwrap()))
    .collect();
    assert_ne!(
        before[output.id].as_ref().unwrap().as_f32().unwrap()[0].to_bits(),
        (3.0f32 / 7.0).to_bits(),
        "the pinned reciprocal counterexample must distinguish division"
    );
    assert_observables(&graph, &canonicalize(&graph), &inputs);
    let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
    assert_observables(&graph, program.graph(), &inputs);
}

#[test]
fn narrow_contraction_rounds_before_fused_epilogue() {
    for (dtype, half_ulp) in [(DType::BF16, 1.0 / 256.0), (DType::F16, 1.0 / 2048.0)] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32([2, 2]));
        let w = b.constant("w", TensorType::f32([2, 2]));
        let nx = b.cast(x, dtype);
        let nw = b.cast(w, dtype);
        let product = b.matmul(nx, nw);
        assert_eq!(
            b.aval(product).dtype,
            dtype,
            "a real narrow contraction result"
        );
        let wide = b.cast(product, DType::F32);
        let bias = b.constant("bias", TensorType::f32([2]));
        let sum = b.binary(BinOp::Add, wide, bias);
        let output = b.binary_scalar(BinOp::Mul, sum, Scalar::F32(1.0));
        let graph = with_states(b, output);
        let mut inputs = HashMap::from([
            (
                x.id,
                HostTensor::f32(vec![2, 2], vec![1.0, 1.0, -1.0, -1.0]),
            ),
            (
                w.id,
                HostTensor::f32(vec![2, 2], vec![1.0, 2.0, half_ulp, half_ulp * 2.0]),
            ),
            (bias.id, HostTensor::f32(vec![2], vec![-1.0, -2.0])),
        ]);
        bind_states(&graph, &mut inputs);
        let program = compile(&graph, &target(Backend::Nvptx), &OPTIONS).unwrap();
        assert!(
            program
                .graph()
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Fused(_))),
            "the real compiler must select the F32 epilogue region"
        );
        let values: HashMap<usize, Value> = inputs
            .iter()
            .map(|(&k, v)| (k, Value::from(v.clone())))
            .collect();
        let compiled: Vec<Option<HostTensor>> = eval(
            program.graph(),
            &values,
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .unwrap()
        .environment
        .unwrap()
        .into_iter()
        .map(|v| v.map(|v| v.into_host().unwrap()))
        .collect();
        let actual = &compiled[program.graph().output]
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap();
        println!("{dtype:?} narrow contraction epilogue={actual:?}");
        assert_eq!(
            actual,
            &[0.0, 0.0, -2.0, -4.0],
            "round 1 + half-ulp before adding -1"
        );
        assert!(
            program
                .graph()
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::MatMul)
                    && program.graph().aval(eqn.out).dtype == dtype),
            "the contraction's narrow result remains a boundary"
        );
        assert_observables(&graph, program.graph(), &inputs);
    }
}

#[test]
fn explicit_narrow_casts_remain_between_contraction_and_epilogue() {
    for (dtype, half_ulp) in [(DType::BF16, 1.0 / 256.0), (DType::F16, 1.0 / 2048.0)] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32([1, 2]));
        let w = b.constant("w", TensorType::f32([2, 2]));
        let product = b.matmul(x, w);
        let narrow = b.cast(product, dtype);
        let wide = b.cast(narrow, DType::F32);
        let sum = b.binary_scalar(BinOp::Add, wide, Scalar::F32(-1.0));
        let output = b.binary_scalar(BinOp::Mul, sum, Scalar::F32(1.0));
        let graph = with_states(b, output);
        let mut inputs = HashMap::from([
            (x.id, HostTensor::f32(vec![1, 2], vec![1.0, 1.0])),
            (
                w.id,
                HostTensor::f32(vec![2, 2], vec![1.0, 1.0, half_ulp, half_ulp * 3.0]),
            ),
        ]);
        bind_states(&graph, &mut inputs);
        let program = compile(&graph, &target(Backend::Nvptx), &OPTIONS).unwrap();
        let values: HashMap<usize, Value> = inputs
            .iter()
            .map(|(&k, v)| (k, Value::from(v.clone())))
            .collect();
        let result: Vec<Option<HostTensor>> = eval(
            program.graph(),
            &values,
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .unwrap()
        .environment
        .unwrap()
        .into_iter()
        .map(|v| v.map(|v| v.into_host().unwrap()))
        .collect();
        let actual = &result[program.graph().output]
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap();
        println!("{dtype:?} explicit cast epilogue={actual:?}");
        assert_eq!(actual, &[0.0, half_ulp * 4.0]);
        assert_observables(&graph, program.graph(), &inputs);
    }
}

#[test]
fn nonfinite_inputs_keep_ieee_classification() {
    // This asserts defined IEEE classifications, not NaN equality or a device tolerance success.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([1, 2]));
    let w = b.constant("w", TensorType::f32([2, 2]));
    let scaled = b.binary_scalar(BinOp::Mul, x, Scalar::F32(0.5));
    let output = b.matmul(scaled, w);
    let graph = b.finish(output);
    let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
    for data in [vec![f32::INFINITY, 1.0], vec![f32::NAN, 1.0]] {
        let inputs = HashMap::from([
            (x.id, HostTensor::f32(vec![1, 2], data.clone())),
            (w.id, HostTensor::f32(vec![2, 2], vec![1.0, 0.0, 1.0, 1.0])),
        ]);
        let values: HashMap<usize, Value> = inputs
            .iter()
            .map(|(&k, v)| (k, Value::from(v.clone())))
            .collect();
        for candidate in [&graph, &canonicalize(&graph), program.graph()] {
            let env: Vec<Option<HostTensor>> = eval(
                candidate,
                &values,
                EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
            )
            .unwrap()
            .environment
            .unwrap()
            .into_iter()
            .map(|v| v.map(|v| v.into_host().unwrap()))
            .collect();
            let actual = &env[candidate.output].as_ref().unwrap().as_f32().unwrap();
            if data[0].is_infinite() {
                assert_eq!(actual[0], f32::INFINITY);
            } else {
                assert!(actual[0].is_nan());
            }
            assert!(actual[1].is_nan(), "zero times infinity/NaN stays NaN");
        }
    }
}

#[test]
fn legal_attention_regions_keep_selection_and_prescaled_query() {
    use poot_graph_ir::analysis::NumericalRewriteReason;
    use poot_graph_ir::{Operand, RedOp};

    for (name, prescale, reciprocal, commute) in [
        ("standard", false, false, false),
        ("commuted mask", false, false, true),
        ("prescaled query", true, false, false),
        ("reciprocal near-miss", false, true, false),
    ] {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32([1, 2, 1, 4]));
        let k = b.constant("k", TensorType::f32([1, 2, 3, 4]));
        let v = b.constant("v", TensorType::f32([1, 2, 3, 4]));
        let mask = b.constant("mask", TensorType::f32([1, 1, 1, 3]));
        let query = if prescale {
            b.binary_scalar(BinOp::Mul, q, Scalar::F32(1e-30))
        } else {
            q
        };
        let kt = b.transpose(k, vec![0, 1, 3, 2]);
        let qk = b.matmul(query, kt);
        let scaled = if prescale {
            qk
        } else {
            b.binary_scalar(BinOp::Mul, qk, Scalar::F32(0.5))
        };
        let scores = if commute {
            b.binary(BinOp::Add, mask, scaled)
        } else {
            b.binary(BinOp::Add, scaled, mask)
        };
        let max = b.reduce(RedOp::Max, scores, 3, true);
        let shifted = b.binary(BinOp::Sub, scores, max);
        let exp = b.unary(UnOp::Exp, shifted);
        let denom = b.reduce(RedOp::Sum, exp, 3, true);
        let p = if reciprocal {
            let recip = b.unary(UnOp::Recip, denom);
            b.binary(BinOp::Mul, exp, recip)
        } else {
            b.binary(BinOp::Div, exp, denom)
        };
        let output = b.matmul(p, v);
        let graph = with_states(b, output);
        let mut inputs = HashMap::from([
            (
                q.id,
                HostTensor::f32(
                    vec![1, 2, 1, 4],
                    if prescale {
                        vec![1e30; 8]
                    } else {
                        vec![0.25, -0.5, 0.75, 1.0, -0.75, 0.25, 1.25, -0.5]
                    },
                ),
            ),
            (
                k.id,
                HostTensor::f32(
                    vec![1, 2, 3, 4],
                    (0..24)
                        .map(|i| {
                            if prescale {
                                (i % 3) as f32 * 1e30
                            } else {
                                (i as f32 - 11.0) * 0.125
                            }
                        })
                        .collect(),
                ),
            ),
            (
                v.id,
                HostTensor::f32(
                    vec![1, 2, 3, 4],
                    (0..24).map(|i| (i as f32 - 12.0) * 0.2).collect(),
                ),
            ),
            (
                mask.id,
                HostTensor::f32(vec![1, 1, 1, 3], vec![0.0, -0.125, 0.25]),
            ),
        ]);
        bind_states(&graph, &mut inputs);
        let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
        let flash: Vec<_> = program
            .graph()
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::FlashAttentionDecode { .. }))
            .collect();
        assert_eq!(
            flash.len(),
            usize::from(!reciprocal),
            "{name}: real compile region selection"
        );
        if prescale {
            assert!(
                matches!(flash[0].inputs[0], Operand::Value(id) if id == query.id),
                "the fused kernel must consume the already-scaled query"
            );
            assert!(matches!(
                flash[0].op,
                OpKind::FlashAttentionDecode { scale: 1.0, .. }
            ));
        }
        if reciprocal {
            assert!(
                program
                    .numerical_declines()
                    .iter()
                    .any(|d| d.reason == NumericalRewriteReason::ReciprocalRounding)
            );
        }
        // Card 546b: `Program::reassociation_class` narrowed to `pub(crate)` (its one other
        // cross-crate reader was deleted with poot-gpu's cached-decode test suite); the general
        // "a reassociating pass publishes `Tier2Class::Reassociating`" mechanism this `else` branch
        // checked stays covered in-crate by
        // `compile::tests::compile_publishes_the_class_of_the_passes_that_changed_the_graph`.
        println!(
            "{name}: flash regions={}, diagnostics={:?}",
            flash.len(),
            program.numerical_declines()
        );
        assert_observables(&graph, program.graph(), &inputs);
    }
}

#[test]
fn softcap_requires_the_exact_reciprocal_literal() {
    use poot_graph_ir::{Operand, ops::attention_masked_softcap};
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32([1, 2, 3, 4]));
    let k = b.constant("k", TensorType::f32([1, 2, 3, 4]));
    let v = b.constant("v", TensorType::f32([1, 2, 3, 4]));
    let mask = b.constant("mask", TensorType::f32([1, 1, 3, 3]));
    let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(2.0));
    let original = with_states(b, out);
    let mut inputs = HashMap::from([
        (
            q.id,
            HostTensor::f32(
                vec![1, 2, 3, 4],
                (0..24).map(|i| (i as f32 * 0.31).sin()).collect(),
            ),
        ),
        (
            k.id,
            HostTensor::f32(
                vec![1, 2, 3, 4],
                (0..24).map(|i| (i as f32 * 0.71).cos()).collect(),
            ),
        ),
        (
            v.id,
            HostTensor::f32(
                vec![1, 2, 3, 4],
                (0..24).map(|i| i as f32 * 0.13 - 1.0).collect(),
            ),
        ),
        (mask.id, HostTensor::f32(vec![1, 1, 3, 3], vec![0.0; 9])),
    ]);
    bind_states(&original, &mut inputs);
    for exact in [true, false] {
        let mut graph = original.clone();
        if !exact {
            let mut changed = 0;
            for eqn in &mut graph.eqns {
                if matches!(eqn.op, OpKind::Binary(BinOp::Mul)) {
                    for operand in &mut eqn.inputs {
                        if matches!(operand, Operand::Lit(Scalar::F32(0.5))) {
                            *operand =
                                Operand::Lit(Scalar::F32(f32::from_bits(0.5f32.to_bits() + 1)));
                            changed += 1;
                        }
                    }
                }
            }
            assert_eq!(changed, 1, "exactly the reciprocal is perturbed by one ULP");
        }
        let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
        let flash = program
            .graph()
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::FlashAttentionPrefill { .. }))
            .count();
        assert_eq!(
            flash,
            usize::from(exact),
            "only the exact softcap composition is representable"
        );
        assert_observables(&graph, program.graph(), &inputs);
    }
}

fn underflow_before_contraction(scale_rhs: bool) {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([2, 2]));
    let w = b.constant("w", TensorType::f32([2, 2]));
    let scaled = b.binary_scalar(
        BinOp::Mul,
        if scale_rhs { w } else { x },
        Scalar::F32(1e-20),
    );
    let output = if scale_rhs {
        b.matmul(x, scaled)
    } else {
        b.matmul(scaled, w)
    };
    let graph = with_states(b, output);
    let (x_data, w_data) = if scale_rhs {
        (vec![1e30; 4], vec![1e-30, 1e-30, 0.0, 0.0])
    } else {
        (vec![1e-30, 0.0, 1e-30, 0.0], vec![1e30; 4])
    };
    let mut inputs = HashMap::from([
        (x.id, HostTensor::f32(vec![2, 2], x_data)),
        (w.id, HostTensor::f32(vec![2, 2], w_data)),
    ]);
    bind_states(&graph, &mut inputs);
    let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
    let values: HashMap<usize, Value> = inputs
        .iter()
        .map(|(&k, v)| (k, Value::from(v.clone())))
        .collect();
    let before: Vec<Option<HostTensor>> = eval(
        &graph,
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap()
    .into_iter()
    .map(|v| v.map(|v| v.into_host().unwrap()))
    .collect();
    let after: Vec<Option<HostTensor>> = eval(
        program.graph(),
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap()
    .into_iter()
    .map(|v| v.map(|v| v.into_host().unwrap()))
    .collect();
    let expected = &before[graph.output].as_ref().unwrap().as_f32().unwrap();
    let actual = &after[program.graph().output]
        .as_ref()
        .unwrap()
        .as_f32()
        .unwrap();
    println!("underflow scale_rhs={scale_rhs}: primitive={expected:?}, compiled={actual:?}");
    assert_eq!(
        expected, &[0.0; 4],
        "each tiny operand underflows before contraction"
    );
    assert_observables(&graph, program.graph(), &inputs);
}

#[test]
fn lhs_underflow_is_not_factored_out_of_contraction() {
    underflow_before_contraction(false);
}

#[test]
fn rhs_underflow_is_not_factored_out_of_contraction() {
    underflow_before_contraction(true);
}

#[test]
fn reciprocal_identity_reports_no_numerical_decline() {
    let b = Builder::new();
    let denominator = b.constant("denominator", TensorType::f32([2, 2]));
    let weight = b.constant("weight", TensorType::f32([2, 2]));
    let reciprocal = b.unary(UnOp::Recip, denominator);
    let identity = b.binary_scalar(BinOp::Mul, reciprocal, Scalar::F32(1.0));
    let output = b.matmul(identity, weight);
    let graph = with_states(b, output);
    let mut inputs = HashMap::from([
        (
            denominator.id,
            HostTensor::f32(vec![2, 2], vec![2.0, 4.0, 8.0, 16.0]),
        ),
        (
            weight.id,
            HostTensor::f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ),
    ]);
    bind_states(&graph, &mut inputs);
    let program = compile(&graph, &target(Backend::SpirvVulkan), &OPTIONS).unwrap();
    assert!(
        program.numerical_declines().is_empty(),
        "a proved identity was optimized, not declined: {:?}",
        program.numerical_declines()
    );
    assert!(program.passes().any(|pass| pass == "canonicalize"));
    assert_observables(&graph, program.graph(), &inputs);
}
