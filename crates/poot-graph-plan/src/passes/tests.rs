use super::dtype::transpose_is_noop;
use super::*;
use crate::passes::cse_dce::dce_with_roots;
use poot_graph_ir::Builder;
use poot_graph_ir::analysis::{
    NumericsError, NumericsProperty, PASS_DECLARATIONS, PassDeclaration, Tier2Class,
    dispatch_count, implied_numerics, peak_transient_bytes, verify_pass_numerics,
};
use poot_graph_ir::graph::StateRole;
use poot_graph_ir::op::{BinOp, RedOp};
use poot_graph_ir::ops::{attention_masked, attention_masked_softcap};
use poot_graph_ir::test_support::graph_fixtures::derive;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{KeyLayout, KvBound, value_ranges};
use poot_graph_ir::{LayerIndex, LayerTagError};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;

/// A new program derived from `graph` with `change` applied to its draft: the fixture shape for a
/// near miss whose operands the matcher must decline (well formed, so the edit check accepts it).
fn mutated(graph: &Graph, change: impl FnOnce(&mut poot_graph_ir::GraphEdit)) -> Graph {
    derive(graph, |draft| {
        change(draft);
        poot_graph_ir::test_support::graph_fixtures::reinfer(draft);
    })
    .expect("the near miss is a well-formed graph")
}

/// The graph-level packed-dequant preparation these tests exercise: eliminate dead
/// oracle-only decoding, recognize dense contractions, then reject every remaining escape. The planner's
/// production entry adds block-float claims on top of the same passes.
fn prepare_packed_dequant_production(g: &Graph) -> Result<Graph, PackedDequantProductionError> {
    let graph = dce_with_roots(g, &[]).unwrap();
    let graph = recognize_packed_contractions(&graph).unwrap();
    reject_packed_dequant_escapes(&graph)?;
    Ok(graph)
}

fn packed_graph() -> Result<Graph, poot_graph_ir::BuilderError> {
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
    let b = Builder::for_tests();
    let activation = b
        .constant("activation", TensorType::f32(vec![2, 35]))
        .unwrap();
    let output =
        poot_graph_ir::ops::packed_linear(&b, activation, "layer", descriptor, None, None)?;
    Ok(b.finish(output).unwrap())
}

fn packed_graph_with_extra_consumer() -> Result<Graph, poot_graph_ir::BuilderError> {
    let base = packed_graph()?;
    let mut graph = base.edit();
    let dequant = base.eqns()[0].out;
    let first_output = base.result();
    let transpose_meta = base.values()[base.eqns()[1].out].clone();
    let transpose2 = graph.values_mut().push(transpose_meta);
    graph.eqns_mut().push(Eqn {
        op: OpKind::Transpose { perm: vec![1, 0] },
        inputs: vec![Operand::Value(dequant)],
        out: transpose2,
        layer: None,
    });
    let activation = match base.eqns()[2].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    let second_output = graph.values_mut().push(base.values()[first_output].clone());
    graph.eqns_mut().push(Eqn {
        op: OpKind::MatMul,
        inputs: vec![Operand::Value(activation), Operand::Value(transpose2)],
        out: second_output,
        layer: None,
    });
    let combined = graph.values_mut().push(base.values()[first_output].clone());
    graph.eqns_mut().push(Eqn {
        op: OpKind::Binary(BinOp::Add),
        inputs: vec![Operand::Value(first_output), Operand::Value(second_output)],
        out: combined,
        layer: None,
    });
    graph.set_result(combined);
    Ok(graph.commit().unwrap())
}

fn packed_graph_with_extra_carrier_consumer() -> Result<Graph, poot_graph_ir::BuilderError> {
    let base = packed_graph()?;
    let mut graph = base.edit();
    let weight = match base.eqns()[0].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    let original_output = base.result();
    let cast = graph.values_mut().push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![3, 18]),
        Storage::Device,
        None,
    ));
    graph.eqns_mut().push(Eqn {
        op: OpKind::Cast { to: DType::F32 },
        inputs: vec![Operand::Value(weight)],
        out: cast,
        layer: None,
    });
    let reduced = graph.values_mut().push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![3]),
        Storage::Device,
        None,
    ));
    graph.eqns_mut().push(Eqn {
        op: OpKind::Reduce {
            op: RedOp::Sum,
            axis: 1,
            keepdim: false,
        },
        inputs: vec![Operand::Value(cast)],
        out: reduced,
        layer: None,
    });
    let broadcast = graph.values_mut().push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![2, 3]),
        Storage::Device,
        None,
    ));
    graph.eqns_mut().push(Eqn {
        op: OpKind::Broadcast { shape: vec![2, 3] },
        inputs: vec![Operand::Value(reduced)],
        out: broadcast,
        layer: None,
    });
    let combined = graph.values_mut().push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![2, 3]),
        Storage::Device,
        None,
    ));
    graph.eqns_mut().push(Eqn {
        op: OpKind::Binary(BinOp::Add),
        inputs: vec![Operand::Value(original_output), Operand::Value(broadcast)],
        out: combined,
        layer: None,
    });
    graph.set_result(combined);
    Ok(graph.commit().unwrap())
}

/// `g` with its transpose's permutation replaced by the identity and every type that follows retyped
/// to match, so the graph stays well formed and only the recognizer's permutation check can refuse it.
fn wrong_permutation(g: &Graph) -> Graph {
    let transpose = g
        .eqns()
        .iter()
        .position(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
        .unwrap();
    let matmul = g
        .eqns()
        .iter()
        .position(|eqn| matches!(eqn.op, OpKind::MatMul))
        .unwrap();
    let Operand::Value(activation) = g.eqns()[matmul].inputs[0] else {
        unreachable!()
    };
    let (transpose_out, matmul_out) = (g.eqns()[transpose].out, g.eqns()[matmul].out);
    derive(g, |draft| {
        draft.eqns_mut()[transpose].op = OpKind::Transpose { perm: vec![0, 1] };
        draft.values_mut()[transpose_out].aval = TensorType::f32(vec![3, 35]);
        draft.values_mut()[activation].aval = TensorType::f32(vec![2, 3]);
        draft.values_mut()[matmul_out].aval = TensorType::f32(vec![2, 35]);
    })
    .unwrap()
}

#[test]
fn packed_dequant_fusion_matches_only_exact_linear() -> Result<(), poot_graph_ir::BuilderError> {
    let graph = packed_graph()?;
    let fused = prepare_packed_dequant_production(&graph).unwrap();
    assert_eq!(fused.eqns().len(), 1);
    assert!(matches!(
        fused.eqns()[0].op,
        OpKind::PackedContraction { .. }
    ));

    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
    let builder = Builder::for_tests();
    let activation = builder
        .constant("alias.activation", TensorType::f32(vec![2, 35]))
        .unwrap();
    let output = poot_graph_ir::ops::packed_linear(
        &builder,
        activation,
        "alias",
        descriptor,
        Some(vec![35, 3]),
        None,
    )?;
    let alias_graph = builder.finish(output).unwrap();
    let alias_fused = prepare_packed_dequant_production(&alias_graph).unwrap();
    assert_eq!(alias_fused.eqns().len(), 1);
    assert!(matches!(
        alias_fused.eqns()[0].op,
        OpKind::PackedContraction { .. }
    ));
    assert_eq!(alias_fused.aval(alias_fused.result()).shape, vec![2, 3]);

    let wrong_perm = wrong_permutation(&graph);
    assert!(matches!(
        prepare_packed_dequant_production(&wrong_perm),
        Err(PackedDequantProductionError::TransposePermutation { .. })
    ));

    let extra_consumer = packed_graph_with_extra_consumer()?;
    assert!(matches!(
        prepare_packed_dequant_production(&extra_consumer),
        Err(PackedDequantProductionError::ConsumerCount { .. })
    ));
    Ok(())
}

#[test]
fn packed_block_diagonal_chain_is_recognized_as_one_contraction()
-> Result<(), poot_graph_ir::BuilderError> {
    // out=8, k=35, blocks=4 -> block_out=2. Card 385: `ops::packed_block_diagonal_linear`'s
    // Reshape-before-Transpose chain must fold like the canonical chain; an unrecognized packed
    // dequant is a compile-time escape rejection, not a silently slow fallback.
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
    let builder = Builder::for_tests();
    let activation = builder
        .constant("blocked.activation", TensorType::f32(vec![4, 5, 35]))
        .unwrap();
    let output = poot_graph_ir::ops::packed_block_diagonal_linear(
        &builder, activation, "blocked", descriptor, 4, None,
    )?;
    let graph = builder.finish(output).unwrap();
    let fused = prepare_packed_dequant_production(&graph).unwrap();
    assert_eq!(fused.eqns().len(), 1);
    assert!(matches!(
        fused.eqns()[0].op,
        OpKind::PackedContraction {
            rows: poot_graph_ir::ContractionRows::Blocks(4),
            ..
        }
    ));
    assert_eq!(fused.aval(fused.result()).shape, vec![4, 5, 2]);
    Ok(())
}

fn packed_block_diagonal_graph() -> Result<Graph, poot_graph_ir::BuilderError> {
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
    let b = Builder::for_tests();
    let activation = b
        .constant("blocked.activation", TensorType::f32(vec![4, 5, 35]))
        .unwrap();
    let output = poot_graph_ir::ops::packed_block_diagonal_linear(
        &b, activation, "blocked", descriptor, 4, None,
    )?;
    Ok(b.finish(output).unwrap())
}

/// A second full branch off the same `PackedDequant` output (reshape/transpose/matmul cloned from
/// the original, then summed into the graph's output). This gives the dequant a second live consumer
/// without hand-computing any new tensor shape.
fn packed_block_diagonal_graph_with_extra_consumer() -> Result<Graph, poot_graph_ir::BuilderError> {
    let base = packed_block_diagonal_graph()?;
    let mut graph = base.edit();
    let dequant = base.eqns()[0].out;
    let first_output = base.result();

    let reshape2 = graph
        .values_mut()
        .push(base.values()[base.eqns()[1].out].clone());
    graph.eqns_mut().push(Eqn {
        op: base.eqns()[1].op.clone(),
        inputs: vec![Operand::Value(dequant)],
        out: reshape2,
        layer: None,
    });
    let transpose2 = graph
        .values_mut()
        .push(base.values()[base.eqns()[2].out].clone());
    graph.eqns_mut().push(Eqn {
        op: base.eqns()[2].op.clone(),
        inputs: vec![Operand::Value(reshape2)],
        out: transpose2,
        layer: None,
    });
    let activation = match base.eqns()[3].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    let second_output = graph.values_mut().push(base.values()[first_output].clone());
    graph.eqns_mut().push(Eqn {
        op: OpKind::MatMul,
        inputs: vec![Operand::Value(activation), Operand::Value(transpose2)],
        out: second_output,
        layer: None,
    });
    let combined = graph.values_mut().push(base.values()[first_output].clone());
    graph.eqns_mut().push(Eqn {
        op: OpKind::Binary(BinOp::Add),
        inputs: vec![Operand::Value(first_output), Operand::Value(second_output)],
        out: combined,
        layer: None,
    });
    graph.set_result(combined);
    Ok(graph.commit().unwrap())
}

#[test]
fn packed_block_diagonal_production_rejects_extra_dequant_consumer()
-> Result<(), poot_graph_ir::BuilderError> {
    // The recognizer's admission of the blocked chain must still respect the single-consumer rule
    // every packed chain shares: a second live consumer off the same dequant is a graph escape, not a
    // second contraction to fold.
    let extra_consumer = packed_block_diagonal_graph_with_extra_consumer()?;
    assert!(matches!(
        prepare_packed_dequant_production(&extra_consumer),
        Err(PackedDequantProductionError::ConsumerCount { .. })
    ));
    Ok(())
}

#[test]
fn packed_dequant_production_rejects_live_extra_carrier_consumer()
-> Result<(), poot_graph_ir::BuilderError> {
    let graph = packed_graph_with_extra_carrier_consumer()?;
    assert!(matches!(
        prepare_packed_dequant_production(&graph),
        Err(PackedDequantProductionError::CarrierConsumerCount { consumers: 2, .. })
    ));
    Ok(())
}

#[test]
fn packed_recognition_keeps_validation_observed_members() -> Result<(), poot_graph_ir::BuilderError>
{
    let base = packed_graph()?;
    let dequant = base.eqns()[0].out;
    let graph = derive(&base, |draft| {
        draft.push_witness(
            poot_graph_ir::graph::ValidationId(7),
            "observes-dequant",
            dequant,
        );
    })
    .unwrap();

    let recognized = recognize_packed_contractions(&graph).unwrap();
    assert!(recognized.witnesses().eq(graph.witnesses()));
    assert!(
        recognized
            .eqns()
            .iter()
            .any(|eqn| eqn.out == dequant && matches!(eqn.op, OpKind::PackedDequant { .. })),
        "a witness-observed dequant must stay unrecognized"
    );
    assert!(
        !recognized
            .eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::PackedContraction { .. }))
    );
    assert!(matches!(
        reject_packed_dequant_escapes(&recognized),
        Err(PackedDequantProductionError::ValidationOutput { .. })
    ));
    Ok(())
}

#[test]
fn production_rejects_packed_dequant_escape_table() -> Result<(), poot_graph_ir::BuilderError> {
    let base = packed_graph()?;
    let dequant = base.eqns()[0].out;
    let transpose = base.eqns()[1].out;
    let weight = match base.eqns()[0].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    type Reject = fn(&PackedDequantProductionError) -> bool;
    let mut cases: Vec<(&str, Graph, Reject)> = Vec::new();

    let output = derive(&base, |draft| draft.set_result(dequant)).unwrap();
    cases.push(("graph output", output, |error| {
        matches!(error, PackedDequantProductionError::GraphOutput { .. })
    }));

    // R-644-1: a State value must carry a role, or the graph is malformed before this test's target
    // (CarrierStorage) is reached. `ValueMeta::new_state` is the only constructor that can produce one
    // (`state_role` is private outside `poot-graph-ir`).
    let as_state = |draft: &mut poot_graph_ir::GraphEdit, value: ValueId| {
        let meta = poot_graph_ir::ValueMeta::new_state(
            draft.values()[value].aval.clone(),
            draft.values()[value].name.clone(),
            poot_graph_ir::StateRole::Recurrent,
        );
        draft.values_mut()[value] = meta;
    };
    let state_carrier = derive(&base, |draft| as_state(draft, weight)).unwrap();
    cases.push(("state carrier", state_carrier, |error| {
        matches!(error, PackedDequantProductionError::CarrierStorage { .. })
    }));

    cases.push(("live-out", packed_graph_with_extra_consumer()?, |error| {
        matches!(error, PackedDequantProductionError::ConsumerCount { .. })
    }));

    for (label, state_out) in [("dequant state", dequant), ("transpose state", transpose)] {
        let graph = derive(&base, |draft| {
            let state_in = draft.values_mut().push(poot_graph_ir::ValueMeta::new_state(
                base.aval(state_out).clone(),
                Some(format!("{label}.state")),
                poot_graph_ir::StateRole::Recurrent,
            ));
            draft.inputs_mut().push(state_in);
            draft.consts_mut().push(state_in);
            draft.push_state(state_in, state_out);
        })
        .unwrap();
        cases.push((label, graph, |error| {
            matches!(error, PackedDequantProductionError::StateOutput { .. })
        }));
    }

    let candidate_state = derive(&base, |draft| {
        let state_meta = poot_graph_ir::ValueMeta::new_state(
            base.aval(base.result()).clone(),
            Some("candidate.state".into()),
            poot_graph_ir::StateRole::Recurrent,
        );
        let state_in = draft.values_mut().push(state_meta);
        draft.inputs_mut().push(state_in);
        draft.consts_mut().push(state_in);
        draft.push_state(state_in, base.result());
    })
    .unwrap();
    cases.push(("candidate state", candidate_state, |error| {
        matches!(error, PackedDequantProductionError::StateOutput { .. })
    }));

    let wrong = wrong_permutation(&base);
    cases.push(("wrong permutation", wrong, |error| {
        matches!(
            error,
            PackedDequantProductionError::TransposePermutation { .. }
        )
    }));

    let unmatched = derive(&base, |draft| {
        draft.eqns_mut().remove(1);
        draft.eqns_mut().remove(1);
        draft.set_result(dequant);
        let output = draft.values_mut().push(base.values()[dequant].clone());
        draft.eqns_mut().push(Eqn {
            op: OpKind::Reshape { shape: vec![3, 35] },
            inputs: vec![Operand::Value(dequant)],
            out: output,
            layer: None,
        });
        draft.set_result(output);
    })
    .unwrap();
    cases.push(("unsupported movement", unmatched, |error| {
        matches!(error, PackedDequantProductionError::Movement { .. })
    }));

    for (label, graph, expected) in cases {
        let error = prepare_packed_dequant_production(&graph).unwrap_err();
        assert!(
            expected(&error),
            "{label} returned the wrong rejection: {error:?}"
        );
    }

    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
    let builder = Builder::for_tests();
    let output = builder
        .constant("ordinary.output", TensorType::f32(vec![2, 3]))
        .unwrap();
    // `Builder::packed_component_constants` is `#[cfg(test)]` to `poot-graph-ir` itself (invisible to
    // this crate's tests); inlined here from its own small body.
    let sources: Vec<_> = poot_graph_ir::packed_source_constants("dead", descriptor)
        .into_iter()
        .map(|(name, tensor_type)| builder.constant(name.as_str(), tensor_type).unwrap())
        .collect();
    let _dead = builder.packed_dequant(&sources, descriptor).unwrap();
    let dead_graph = builder.finish(output).unwrap();
    let prepared = prepare_packed_dequant_production(&dead_graph).unwrap();
    assert!(prepared.eqns().is_empty(), "DCE removes oracle-only decode");
    Ok(())
}

/// A value head narrower than the key head fuses: the attention op types `dv != dk`, and the output is
/// `dv` wide. Mutation: require `dv == dk` again in `infer_attention`; the chain stays decomposed.
#[test]
fn flash_attention_fuses_a_distinct_value_head_width() {
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 3])).unwrap();
    let k = b.constant("k", TensorType::f32(vec![1, 2, 5, 3])).unwrap();
    let v = b.constant("v", TensorType::f32(vec![1, 2, 5, 2])).unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, 5]))
        .unwrap();
    let out = attention_masked(&b, q, k, v, 1, 1.0, mask).unwrap();
    let g = b.finish(out).unwrap();
    let optimized = flash_attention_capped(&g, None).unwrap();

    assert_eq!(optimized.aval(optimized.result()).shape, vec![1, 2, 1, 2]);
    assert!(
        optimized
            .eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Attention(_))),
        "the generated kernel takes a value width of its own"
    );
}

fn is_flash(op: &OpKind) -> bool {
    matches!(op, OpKind::Attention(_))
}

/// R467-001: K/V with a size-1 head axis (MQA written by broadcasting, not `repeat_kv`) or a size-1 batch
/// axis (one KV shared by every batch row) is a valid decomposition, but the fused op indexes K/V as
/// `[B, Hq/n_rep, cap, D]`. The matcher must decline both.
#[test]
fn flash_attention_declines_broadcast_kv() {
    let cases = [
        (
            "MQA by broadcast",
            [1, 4, 1, 16],
            [1, 1, 8, 16],
            [1, 1, 1, 8],
        ),
        (
            "KV shared over batch",
            [2, 4, 1, 16],
            [1, 4, 8, 16],
            [2, 1, 1, 8],
        ),
    ];
    for (name, qs, kvs, ms) in cases {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(qs.to_vec())).unwrap();
        let k = b.constant("k", TensorType::f32(kvs.to_vec())).unwrap();
        let v = b.constant("v", TensorType::f32(kvs.to_vec())).unwrap();
        let mask = b.constant("mask", TensorType::f32(ms.to_vec())).unwrap();
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask).unwrap();
        let optimized = flash_attention_capped(&b.finish(out).unwrap(), None).unwrap();
        let flash: Vec<String> = optimized
            .eqns()
            .iter()
            .filter(|e| is_flash(&e.op))
            .map(|e| e.op.name())
            .collect();
        assert!(flash.is_empty(), "{name}: fused into {flash:?}");
    }
}

/// A graph whose last equation is the fused attention op over dense MHA operands (`Hq = 4`, `n_rep = 1`),
/// with the ids of its `k`, `v` and `mask` inputs. The decode form uses batch 2 and `cap = 5`; the prefill
/// form batch 1 and `L = 3`.
fn flash_graph(decode: bool) -> (Graph, [ValueId; 3]) {
    let (bsz, hq, m, t, d) = if decode {
        (2, 4, 1, 5, 8)
    } else {
        (1, 4, 3, 3, 8)
    };
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![bsz, hq, m, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![bsz, hq, t, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![bsz, hq, t, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![bsz, 1, m, t]))
        .unwrap();
    let out = if decode {
        poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.25, mask).unwrap()
    } else {
        poot_graph_ir::ops::attention_prefill(&b, q, k, v, 1, 0.25, mask).unwrap()
    };
    // A tracer states attention as primitives; the flash op is the matcher's.
    let g = dce(&flash_attention_capped(&b.finish(out).unwrap(), None).unwrap()).unwrap();
    let flash = g.eqns().last().expect("the flash equation");
    assert!(
        matches!(flash.op, OpKind::Attention(_)),
        "the matcher forms the flash op: {:?}",
        flash.op
    );
    (g, [k.id(), v.id(), mask.id()])
}

/// R467-001: the edit check rejects a fused flash op whose K/V or mask the op would index past, from
/// any source (not just the matcher), with the typed `AttentionOperand` error.
#[test]
fn validate_rejects_flash_ops_with_inconsistent_kv_or_mask() {
    for decode in [true, false] {
        let name = if decode { "decode" } else { "prefill" };
        let (dense, [k, v, mask]) = flash_graph(decode);
        // (operand id, rank-4 axis to shrink to 1, operand name): a broadcast K head axis, a broadcast V
        // batch/row axis, and a mask whose head axis is neither 1 nor Hq.
        for (id, axis, size, operand) in [(k, 1, 1, "k"), (v, 0, 1, "v"), (mask, 1, 2, "mask")] {
            let expected = dense.values()[id].aval.shape.clone();
            let mut actual = expected.clone();
            actual[axis] = size;
            if actual == expected {
                // Prefill is batch 1, so shrinking V's batch axis changes nothing: shrink its rows.
                actual[2] = 1;
            }
            let bad = derive(&dense, |draft| {
                draft.values_mut()[id].aval.shape = actual.clone();
            });
            match bad {
                Err(poot_graph_ir::EditError::Invalid(
                    poot_graph_ir::GraphValidationError::EquationInference {
                        source:
                            poot_graph_ir::error::ShapeError::AttentionOperand {
                                operand: got,
                                expected: want,
                                actual: seen,
                            },
                        ..
                    },
                )) => {
                    assert_eq!(got, operand, "{name}");
                    assert_eq!(want, expected, "{name}: {operand} expected shape");
                    assert_eq!(seen, actual, "{name}: {operand} actual shape");
                }
                other => {
                    panic!("{name}: {operand} {actual:?} must be a typed rejection, got {other:?}")
                }
            }
        }
    }
}

/// Hand-built `ops::rope_partial` rotate-half chain with free x and cos/sin shapes (full rotary).
fn rope_chain(x_shape: Vec<usize>, table_shape: Vec<usize>) -> Graph {
    use poot_graph_ir::op::UnOp;
    let b = Builder::for_tests();
    let last = x_shape.len() - 1;
    let d = x_shape[last];
    let x = b.constant("x", TensorType::f32(x_shape)).unwrap();
    let cos = b
        .constant("cos", TensorType::f32(table_shape.clone()))
        .unwrap();
    let sin = b.constant("sin", TensorType::f32(table_shape)).unwrap();
    let x1 = b.slice(x, last, 0, d / 2).unwrap();
    let x2 = b.slice(x, last, d / 2, d).unwrap();
    let neg_x2 = b.unary(UnOp::Neg, x2).unwrap();
    let rotate_half = b.concat(last, &[neg_x2, x1]).unwrap();
    let xc = b.binary(BinOp::Mul, x, cos).unwrap();
    let rs = b.binary(BinOp::Mul, rotate_half, sin).unwrap();
    let out = b.binary(BinOp::Add, xc, rs).unwrap();
    b.finish(out).unwrap()
}

/// R467-002: a rotate-half chain whose per-head cos/sin broadcast a shared x up (`x[1,1,S,D]`,
/// `cos[1,H,S,D]`) outputs `[1,H,S,D]`, but `Rope` is shape-preserving. `rope_fusion` must decline it
/// (the graph stays valid) while still fusing the control that does not grow x.
#[test]
fn rope_fusion_declines_a_chain_that_broadcasts_x_up() {
    for (x, table, ropes) in [
        (vec![1, 1, 3, 8], vec![1, 4, 3, 8], 0),
        (vec![1, 4, 3, 8], vec![1, 1, 3, 8], 1),
    ] {
        let g = rope_chain(x.clone(), table.clone());
        // The pass's result is a checked graph: an ill-typed rewrite is an `EditError`, not a graph.
        let fused = rope_fusion(&g).unwrap_or_else(|e| {
            panic!("x {x:?}, cos {table:?}: rope_fusion left an ill-typed graph: {e}")
        });
        let got = fused
            .eqns()
            .iter()
            .filter(|e| matches!(e.op, OpKind::Rope { .. }))
            .count();
        assert_eq!(got, ropes, "x {x:?}, cos {table:?}: Rope ops");
    }
}

/// `Rope`'s typing rule rejects a table that would grow x, so `Graph::validate` catches an ill-typed
/// `Rope` from any source.
#[test]
fn rope_infer_rejects_a_table_that_grows_x() {
    let f = |s: &[usize]| TensorType::f32(s.to_vec());
    let rope = OpKind::Rope { rot: 8 };
    assert_eq!(
        rope.infer(&[f(&[1, 4, 3, 8]), f(&[1, 1, 3, 8]), f(&[3, 8])]),
        Ok(f(&[1, 4, 3, 8]))
    );
    assert_eq!(
        rope.infer(&[f(&[1, 1, 3, 8]), f(&[1, 4, 3, 8]), f(&[1, 1, 3, 8])]),
        Err(poot_graph_ir::error::ShapeError::RopeTable {
            operand: "cos",
            rotated: vec![1, 1, 3, 8],
            actual: vec![1, 4, 3, 8],
        })
    );
    assert_eq!(
        OpKind::Rope { rot: 4 }.infer(&[f(&[1, 1, 3, 8]), f(&[3, 4]), f(&[2, 3, 4])]),
        Err(poot_graph_ir::error::ShapeError::RopeTable {
            operand: "sin",
            rotated: vec![1, 1, 3, 4],
            actual: vec![2, 3, 4],
        })
    );
}

#[test]
fn dce_drops_dead_eqns() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let _dead = b.binary(BinOp::Add, x, x).unwrap(); // never used
    let out = b.binary(BinOp::Mul, x, x).unwrap();
    let g = b.finish(out).unwrap();
    let d = dce(&g).unwrap();
    assert!(
        d.eqns().len() < g.eqns().len(),
        "dce should drop the dead add"
    );
    assert_eq!(d.aval(d.result()).shape, vec![4]);
}

#[test]
fn peak_transient_bytes_counts_the_live_working_set() {
    // a linear chain x -> a -> b -> c (output), each [N] f32. At any step at most two intermediates are
    // live (the current output + the one feeding it), so the peak is 2*N*4 - NOT the sum of all four. The
    // const input x is persistent, not transient, so it is excluded.
    let n = 100usize;
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![n])).unwrap();
    let a = b.unary(poot_graph_ir::op::UnOp::Neg, x).unwrap();
    let bb = b.unary(poot_graph_ir::op::UnOp::Neg, a).unwrap();
    let c = b.unary(poot_graph_ir::op::UnOp::Neg, bb).unwrap();
    let g = b.finish(c).unwrap();
    // a born (live={a}=N), b born (live={a,b}=2N) then a freed, c born (live={b,c}=2N) then b freed.
    assert_eq!(peak_transient_bytes(&g), 2 * n * 4);
}

#[test]
fn dispatch_count_excludes_reshape_aliases() {
    // reshape is a buffer-aliasing view (no dispatch); the binary + the transpose are real dispatches.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
    let r = b.reshape(x, vec![3, 2]).unwrap(); // alias - not a dispatch
    let t = b.transpose(r, vec![1, 0]).unwrap(); // a real kernel
    let out = b.binary(BinOp::Add, t, t).unwrap(); // a real kernel
    let g = b.finish(out).unwrap();
    assert_eq!(g.eqns().len(), 3, "reshape + transpose + add");
    assert_eq!(dispatch_count(&g), 2, "reshape aliases, so 2 dispatches");
}

#[test]
fn cse_folds_duplicate_eqns_and_stays_valid() {
    // two structurally-identical adds collapse to one; the mul rewires to the canonical value.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let a1 = b.binary(BinOp::Add, x, x).unwrap();
    let a2 = b.binary(BinOp::Add, x, x).unwrap(); // a duplicate of a1
    let out = b.binary(BinOp::Mul, a1, a2).unwrap();
    let g = b.finish(out).unwrap();
    let c = cse(&g).unwrap();
    assert!(
        c.eqns().len() < g.eqns().len(),
        "cse should drop the duplicate add: {} -> {}",
        g.eqns().len(),
        c.eqns().len()
    );
    assert_eq!(c.aval(c.result()).shape, g.aval(g.result()).shape);
}

#[test]
fn cse_is_idempotent() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let a1 = b.binary(BinOp::Add, x, x).unwrap();
    let a2 = b.binary(BinOp::Add, x, x).unwrap();
    let out = b.binary(BinOp::Mul, a1, a2).unwrap();
    let g = b.finish(out).unwrap();
    let c1 = cse(&g).unwrap();
    let c2 = cse(&c1).unwrap();
    assert_eq!(c1.eqns().len(), c2.eqns().len());
}

#[test]
fn cse_merges_computed_constants_of_one_payload_and_drops_the_duplicate_binder() {
    // A tracer that declares the same computed table per use (a RoPE table per layer) still gathers it
    // once: two declarations of one payload are one value, and only the first stays a graph input.
    // A different payload stays a separate constant.
    let b = Builder::for_tests();
    let pos = b
        .constant("pos", TensorType::new(vec![1, 1], poot_tensor::DType::I32))
        .unwrap();
    let table = |len| {
        b.computed(poot_graph_ir::ComputedConst::Iota {
            len,
            dtype: poot_tensor::DType::F32,
        })
        .unwrap()
    };
    let (first, second, other) = (table(8), table(8), table(9));
    let rows = |t| b.gather(t, 0, pos).unwrap();
    let out = b.binary(BinOp::Add, rows(first), rows(second)).unwrap();
    let out = b.binary(BinOp::Add, out, rows(other)).unwrap();
    let g = b.finish(out).unwrap();
    let c = cse(&g).unwrap();
    let gathers = |g: &Graph| {
        g.eqns()
            .iter()
            .filter(|e| matches!(e.op, OpKind::Gather { .. }))
            .count()
    };
    assert_eq!(gathers(&g), 3);
    assert_eq!(gathers(&c), 2, "the two gathers of one payload merge");
    assert_eq!(c.consts().len(), g.consts().len() - 1);
    assert_eq!(c.inputs().len(), g.inputs().len() - 1);
}

fn f(b: &Builder, shape: &[usize]) -> poot_graph_ir::builder::Traced {
    b.constant("x", TensorType::f32(shape.to_vec())).unwrap()
}

#[test]
fn fuse_collapses_pointwise_chain() {
    // out = (a + b) * c, all [1,1,8] -> two eqns (add, mul) collapse to one Fused eqn over 3 leaves.
    let b = Builder::for_tests();
    let (a, c, d) = (f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]));
    let t = b.binary(BinOp::Add, a, c).unwrap();
    let out = b.binary(BinOp::Mul, t, d).unwrap();
    let g = b.finish(out).unwrap();
    assert_eq!(g.eqns().len(), 2);
    let fg = fuse(&g).unwrap();
    assert_eq!(
        fg.eqns().len(),
        1,
        "the add+mul chain should fuse to one eqn"
    );
    match &fg.eqns()[0].op {
        OpKind::Fused(r) => {
            assert_eq!(r.n_inputs, 3); // a, c, d
            assert_eq!(r.steps.len(), 2);
            assert_eq!(fg.eqns()[0].inputs.len(), 3);
        }
        other => panic!("expected a Fused eqn, got {}", other.name()),
    }
    assert_eq!(fg.aval(fg.result()).shape, vec![1, 1, 8]);
}

#[test]
fn fuse_fuses_a_multi_use_value_read_only_inside_the_region() {
    // t = a + b is consumed by TWO eqns that both end up in the one region (the diamond u, v -> out), so
    // t never escapes and fuses with them: 4 eqns -> 1 (Card 630's diamond rule; before it, t stayed a
    // standalone add). A value that is also read outside the region stays materialized: see
    // `fuse::tests::a_producer_read_outside_the_region_does_not_fuse`.
    let b = Builder::for_tests();
    let (a, bb, c, d) = (
        f(&b, &[1, 4]),
        f(&b, &[1, 4]),
        f(&b, &[1, 4]),
        f(&b, &[1, 4]),
    );
    let t = b.binary(BinOp::Add, a, bb).unwrap();
    let u = b.binary(BinOp::Mul, t, c).unwrap();
    let v = b.binary(BinOp::Sub, t, d).unwrap();
    let out = b.binary(BinOp::Add, u, v).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    assert_eq!(fg.eqns().len(), 1, "t joins the region that reads it twice");
    assert!(matches!(&fg.eqns()[0].op, OpKind::Fused(r) if r.steps.len() == 4));
}

#[test]
fn fuse_does_not_wrap_a_singleton() {
    // a lone pointwise eqn has no fusable single-use producer: it stays a plain Binary, not Fused.
    let b = Builder::for_tests();
    let (a, c) = (f(&b, &[3]), f(&b, &[3]));
    let out = b.binary(BinOp::Add, a, c).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    assert_eq!(fg.eqns().len(), 1);
    assert!(matches!(fg.eqns()[0].op, OpKind::Binary(_)));
}

#[test]
fn fuse_collapses_exact_i32_pointwise_chain() {
    let b = Builder::for_tests();
    let ty = TensorType::new(vec![4], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone()).unwrap();
    let y = b.constant("y", ty).unwrap();
    let sum = b.binary(BinOp::Add, x, y).unwrap();
    let out = b
        .binary_scalar(BinOp::GeU, sum, poot_graph_ir::types::Scalar::I32(-1))
        .unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    assert_eq!(fg.eqns().len(), 1, "same-shape I32 Add+GeU should fuse");
    match &fg.eqns()[0].op {
        OpKind::Fused(region) => assert_eq!(region.steps.len(), 2),
        other => panic!("expected Fused, got {}", other.name()),
    }
}

#[test]
fn fuse_i32_closed_dag_absorbs_non_escaping_multi_use() {
    let b = Builder::for_tests();
    let ty = TensorType::new(vec![4], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone()).unwrap();
    let y = b.constant("y", ty.clone()).unwrap();
    let z = b.constant("z", ty).unwrap();
    let t = b.binary(BinOp::And, x, y).unwrap();
    let u = b.binary(BinOp::Or, t, z).unwrap();
    let v = b.unary(UnOp::Not, t).unwrap();
    let out = b.select(u, v, t).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    assert_eq!(fg.eqns().len(), 1);
    match &fg.eqns()[0].op {
        OpKind::Fused(region) => {
            assert_eq!(region.steps.len(), 4);
            assert!(
                region
                    .steps
                    .iter()
                    .any(|step| matches!(step.op, FusedOp::Select))
            );
        }
        other => panic!("expected Fused, got {}", other.name()),
    }
}

/// POOT-1033: a float `Select` fuses into a pointwise float region; its I32 condition (made by an I32
/// equation, which a float region cannot hold) is a region input leaf, and the select is a step.
#[test]
fn fuse_float_select_is_a_step_of_a_float_region_with_the_condition_as_a_leaf() {
    let b = Builder::for_tests();
    let cond = b
        .constant("cond", TensorType::new(vec![4], poot_tensor::DType::I32))
        .unwrap();
    let if_true = b.constant("t", TensorType::f32(vec![4])).unwrap();
    let if_false = b.constant("f", TensorType::f32(vec![4])).unwrap();
    let selected = b.select(cond, if_true, if_false).unwrap();
    let out = b.unary(UnOp::Neg, selected).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    assert_eq!(fg.eqns().len(), 1);
    let OpKind::Fused(region) = &fg.eqns()[0].op else {
        panic!("expected Fused, got {}", fg.eqns()[0].op.name());
    };
    assert!(
        region
            .steps
            .iter()
            .any(|step| matches!(step.op, FusedOp::Select))
    );
    let leaf_dtypes: Vec<_> = fg.eqns()[0]
        .inputs
        .iter()
        .map(|input| match input {
            poot_graph_ir::Operand::Value(id) => fg.aval(*id).dtype,
            poot_graph_ir::Operand::Lit(_) => unreachable!("a region input is a value"),
        })
        .collect();
    assert_eq!(
        leaf_dtypes,
        [
            poot_tensor::DType::I32,
            poot_tensor::DType::F32,
            poot_tensor::DType::F32
        ]
    );
}

/// A `Select` between two reductions of a softmax-shaped chain ends the row region instead of entering it: a
/// row kernel reads float lanes only, so the I32 condition has no lane there. No `FusedRow` holds a `Select`.
/// Mutation: drop the `Select` exclusion in `is_row_candidate`; the select joins the row region and this
/// row goes red.
#[test]
fn fuse_keeps_a_float_select_out_of_row_regions() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 4])).unwrap();
    let cond = b
        .constant("cond", TensorType::new(vec![2, 4], poot_tensor::DType::I32))
        .unwrap();
    let floor = b.constant("floor", TensorType::f32(vec![2, 4])).unwrap();
    let max = b.reduce(RedOp::Max, x, 1, true).unwrap();
    let shifted = b.binary(BinOp::Sub, x, max).unwrap();
    let masked = b.select(cond, shifted, floor).unwrap();
    let exp = b.unary(UnOp::Exp, masked).unwrap();
    let sum = b.reduce(RedOp::Sum, exp, 1, true).unwrap();
    let out = b.binary(BinOp::Div, exp, sum).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    let mut selects = 0;
    for eqn in fg.eqns() {
        match &eqn.op {
            OpKind::FusedRow(region) => assert!(
                region.steps.iter().all(|step| !matches!(
                    step,
                    RowStep::Pointwise {
                        op: FusedOp::Select,
                        ..
                    }
                )),
                "a row region holds a Select"
            ),
            OpKind::Select => selects += 1,
            OpKind::Fused(region) => {
                selects += region
                    .steps
                    .iter()
                    .filter(|step| matches!(step.op, FusedOp::Select))
                    .count();
            }
            _ => {}
        }
    }
    assert_eq!(selects, 1, "the select survives outside the row regions");
}

#[test]
fn fuse_i32_pack_concat_keeps_shared_operands_in_one_region() {
    let b = Builder::for_tests();
    let ty = TensorType::new(vec![1], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone()).unwrap();
    let y = b.constant("y", ty).unwrap();
    let shared = b.binary(BinOp::And, x, y).unwrap();
    let low = b.binary(BinOp::Or, shared, x).unwrap();
    let high = b.binary(BinOp::Xor, shared, y).unwrap();
    let packed = b.concat(0, &[low, high]).unwrap();
    let g = b.finish(packed).unwrap();
    let fg = fuse(&g).unwrap();
    assert_eq!(fg.eqns().len(), 1);
    match &fg.eqns()[0].op {
        OpKind::Fused(region) => {
            assert_eq!(region.pack.len(), 2);
            assert_eq!(region.steps.len(), 3);
        }
        other => panic!("expected packed Fused, got {}", other.name()),
    }
}

#[test]
fn fuse_i32_pack_concat_absorbs_unit_last_axis_unpack_slices() {
    let b = Builder::for_tests();
    let ty = TensorType::new(vec![1], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone()).unwrap();
    let y = b.constant("y", ty).unwrap();
    let low = b.binary(BinOp::Or, x, y).unwrap();
    let high = b.binary(BinOp::Xor, x, y).unwrap();
    let packed = b.concat(0, &[low, high]).unwrap();
    let left = b.slice(packed, 0, 0, 1).unwrap();
    let right = b.slice(packed, 0, 1, 2).unwrap();
    let masked = b.binary(BinOp::And, left, right).unwrap();
    let out = b.binary(BinOp::Or, masked, left).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    assert!(
        !fg.eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Slice { .. })),
        "unit last-axis unpack slices must be absorbed"
    );
    let fused = fg
        .eqns()
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::Fused(_)))
        .count();
    assert_eq!(fused, 2, "one packed concat region and one unpack consumer");
    match &fg.eqns()[1].op {
        OpKind::Fused(region) => {
            assert_eq!(region.n_inputs, 1);
            assert!(region.steps.iter().any(|step| {
                step.inputs
                    .iter()
                    .any(|operand| matches!(operand, FusedOperand::PackLane { lane: 1, .. }))
            }));
        }
        other => panic!("expected consumer Fused, got {}", other.name()),
    }
}

#[test]
fn fuse_i32_pack_concat_keeps_a_pinned_unpack_slice() {
    let b = Builder::for_tests();
    let ty = TensorType::new(vec![1], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone()).unwrap();
    let y = b.constant("y", ty).unwrap();
    let low = b.binary(BinOp::Or, x, y).unwrap();
    let high = b.binary(BinOp::Xor, x, y).unwrap();
    let packed = b.concat(0, &[low, high]).unwrap();
    let left = b.slice(packed, 0, 0, 1).unwrap();
    let right = b.slice(packed, 0, 1, 2).unwrap();
    let masked = b.binary(BinOp::And, left, right).unwrap();
    let _keep = b.binary(BinOp::Or, masked, left).unwrap();
    let g = b.finish(left).unwrap();
    let fg = fuse(&g).expect("pinned unpack slice must remain a produced graph output");
    assert_eq!(fg.result(), left.id());
    assert!(
        fg.eqns().iter().any(|eqn| eqn.out == left.id()),
        "pinned unit last-axis slice must not be absorbed"
    );
}

#[test]
fn fuse_does_not_cross_a_matmul_boundary() {
    // add -> matmul -> add: the matmul is a forced boundary, so the two adds do not fuse together.
    let b = Builder::for_tests();
    let (a, c) = (f(&b, &[1, 4]), f(&b, &[1, 4]));
    let w = b.constant("w", TensorType::f32(vec![4, 4])).unwrap();
    let s = b.binary(BinOp::Add, a, c).unwrap(); // [1,4]
    let m = b.matmul(s, w).unwrap(); // [1,4], a boundary
    let e = f(&b, &[1, 4]);
    let out = b.binary(BinOp::Add, m, e).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    // matmul stays; neither add fuses with anything (each is a singleton across the boundary).
    assert!(fg.eqns().iter().any(|e| matches!(e.op, OpKind::MatMul)));
    assert!(!fg.eqns().iter().any(|e| matches!(e.op, OpKind::Fused(_))));
}

#[test]
fn fuse_forms_a_rmsnorm_row_region() {
    // the whole RMSNorm decomposition (sq -> sum -> *1/n -> +eps -> sqrt -> x/den -> *w) fuses to ONE
    // reduction-rooted FusedRow eqn.
    use poot_graph_ir::ops::rmsnorm;
    let b = Builder::for_tests();
    let n = 8usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, n])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![n])).unwrap();
    let out = rmsnorm(&b, x, w, 1e-6).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    let rows = fg
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::FusedRow(_)))
        .count();
    assert_eq!(rows, 1, "rmsnorm fuses to one FusedRow eqn");
    assert!(
        fg.eqns().len() < g.eqns().len(),
        "fusion drops eqns: {} -> {}",
        g.eqns().len(),
        fg.eqns().len()
    );
    if let OpKind::FusedRow(r) = &fg
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::FusedRow(_)))
        .unwrap()
        .op
    {
        assert_eq!(r.n_inputs, 2, "leaves are x and w");
        assert_eq!(r.axis, 2, "reduction over the last axis");
    }
}

#[test]
fn fuse_does_not_form_a_reduced_output_row_region() {
    // A FusedRow whose output is the reduced value (axis size 1, not the full row width) breaks the
    // fused_row_parallel kernel: it derives n_cols from the output shape and would iterate 1 column
    // instead of the full row. Such regions are skipped (the reduce stays a plain `Reduce`, the
    // pointwise falls to pointwise fusion). Two shapes of reduced output, both accepted by `row_fits`
    // (its `== 1` clause), so the region would otherwise form: (a) a bare reduce root, and (b) a
    // scalar pointwise applied to a reduce (the `quant_scale_kv = reduce_max(|x|)*(1/127)` shape that
    // gave garbage on real KV-quant weights). Fast unit-level guard for the bug the GPU fuzzer and a
    // real-model test caught.
    use poot_graph_ir::Scalar;
    use poot_graph_ir::op::RedOp;
    let has_row = |g: &Graph| g.eqns().iter().any(|e| matches!(e.op, OpKind::FusedRow(_)));
    let has_reduce = |g: &Graph| {
        g.eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Reduce { .. }))
    };

    // (a) bare reduce of a pointwise: sum(x + x) over the last axis -> output [1,4,1].
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 4, 8])).unwrap();
    let s = b.binary(BinOp::Add, x, x).unwrap();
    let r = b.reduce(RedOp::Sum, s, 2, true).unwrap();
    let fg = fuse(&b.finish(r).unwrap()).unwrap();
    assert!(
        !has_row(&fg),
        "bare reduce-output region must not be a FusedRow (0196)"
    );
    assert!(has_reduce(&fg), "the reduce stays a plain Reduce op");

    // (b) scalar pointwise after a reduce (the quant_scale_kv shape): reduce_max(x) * (1/127) -> [1,4,1].
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 4, 8])).unwrap();
    let m = b.reduce(RedOp::Max, x, 2, true).unwrap();
    let sc = b
        .binary_scalar(BinOp::Mul, m, Scalar::F32(1.0 / 127.0))
        .unwrap();
    let fg = fuse(&b.finish(sc).unwrap()).unwrap();
    assert!(
        !has_row(&fg),
        "scalar-after-reduce reduced output must not be a FusedRow (0197)"
    );
    assert!(has_reduce(&fg), "the reduce stays a plain Reduce op");
}

#[test]
fn fuse_is_idempotent() {
    let b = Builder::for_tests();
    let (a, c, d) = (f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]));
    let t = b.binary(BinOp::Add, a, c).unwrap();
    let out = b.binary(BinOp::Mul, t, d).unwrap();
    let g = b.finish(out).unwrap();
    let f1 = fuse(&g).unwrap();
    let f2 = fuse(&f1).unwrap();
    assert_eq!(f1.eqns().len(), f2.eqns().len());
}

#[test]
fn elide_noop_transpose_unit() {
    // bare predicate: only size-1 axes shuffle -> no-op; non-unit axes reorder -> a real move.
    assert!(transpose_is_noop(&[1, 4, 1, 8], &[0, 2, 1, 3])); // decode q/k/v reshape
    assert!(transpose_is_noop(&[1, 1, 8], &[1, 0, 2])); // swap two size-1 axes
    assert!(!transpose_is_noop(&[3, 5], &[1, 0])); // real 2D transpose moves data
    assert!(!transpose_is_noop(&[1, 4, 8], &[0, 2, 1])); // H,D reorder moves data
}

#[test]
fn elide_noop_transposes_rewrites_only_noops() {
    // a seq-1 attention-style transpose [1,H,1,D] -> [1,1,H,D] is rewritten to a (free) reshape.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 4, 1, 8])).unwrap();
    let t = b.transpose(x, vec![0, 2, 1, 3]).unwrap();
    let g = elide_noop_transposes(&b.finish(t).unwrap()).unwrap();
    assert!(
        g.eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Reshape { .. })),
        "no-op transpose -> reshape"
    );
    assert!(
        !g.eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Transpose { .. })),
        "no transpose dispatch remains"
    );
    // a genuine data-moving transpose is left intact.
    let b2 = Builder::for_tests();
    let y = b2.constant("y", TensorType::f32(vec![3, 5])).unwrap();
    let ty = b2.transpose(y, vec![1, 0]).unwrap();
    let g2 = elide_noop_transposes(&b2.finish(ty).unwrap()).unwrap();
    assert!(
        g2.eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Transpose { .. })),
        "real transpose preserved"
    );
}

#[test]
fn collapse_reshape_chains_merges_a_chain() {
    // reshape -> reshape -> reshape collapses to ONE reshape reading straight from the source, with the
    // final shape preserved; the two dead intermediates are DCE'd.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 3, 4])).unwrap();
    let r1 = b.reshape(x, vec![6, 4]).unwrap();
    let r2 = b.reshape(r1, vec![24]).unwrap();
    let r3 = b.reshape(r2, vec![4, 6]).unwrap();
    let g = b.finish(r3).unwrap();
    assert_eq!(g.eqns().len(), 3);
    let c = collapse_reshape_chains(&g).unwrap();
    assert_eq!(c.eqns().len(), 1, "the reshape chain collapses to one");
    assert!(matches!(c.eqns()[0].op, OpKind::Reshape { .. }));
    assert!(
        matches!(c.eqns()[0].inputs[0], Operand::Value(v) if v == x.id()),
        "reads the source"
    );
    assert_eq!(
        c.aval(c.result()).shape,
        vec![4, 6],
        "final shape preserved"
    );
}

#[test]
fn collapse_keeps_a_still_used_intermediate() {
    // if an intermediate reshape feeds another live consumer, it must stay (only the second reshape's
    // input is rewired past it); the graph stays valid and both outputs resolve.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, 4])).unwrap();
    let r1 = b.reshape(x, vec![16]).unwrap(); // used by r2 AND by add
    let r2 = b.reshape(r1, vec![2, 8]).unwrap();
    let add = b.binary(BinOp::Add, r1, r1).unwrap(); // second consumer of r1 -> r1 stays live
    let out = b
        .binary(BinOp::Add, b.reshape(r2, vec![16]).unwrap(), add)
        .unwrap();
    let g = b.finish(out).unwrap();
    let c = collapse_reshape_chains(&g).unwrap();
    assert!(
        c.eqns().iter().any(|e| e.out == r1.id()),
        "the multi-use intermediate reshape is retained"
    );
}

#[test]
fn fuse_collapses_the_reshape_after_eliding_a_noop_transpose() {
    // the decode attention shape: reshape -> (no-op) transpose -> reshape. elide turns the transpose into
    // a reshape, then collapse merges the run, so `fuse` emits a single reshape (no transpose, no chain).
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, 32])).unwrap();
    let r1 = b.reshape(x, vec![1, 1, 4, 8]).unwrap();
    let t = b.transpose(r1, vec![0, 2, 1, 3]).unwrap(); // [1,4,1,8] - no-op at seq 1
    let out = b.reshape(t, vec![1, 4, 8]).unwrap();
    let g = b.finish(out).unwrap();
    let fg = fuse(&g).unwrap();
    assert!(
        !fg.eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Transpose { .. })),
        "no transpose dispatch remains"
    );
    assert_eq!(
        fg.eqns()
            .iter()
            .filter(|e| matches!(e.op, OpKind::Reshape { .. }))
            .count(),
        1,
        "the reshape/transpose/reshape run collapses to one reshape"
    );
    assert_eq!(fg.aval(fg.result()).shape, vec![1, 4, 8]);
}

#[test]
fn rope_fusion_full_collapses_the_rotate_half_chain() {
    use poot_graph_ir::graph::Slot;
    use poot_graph_ir::op::UnOp;
    use poot_tensor::DType;
    // full rotary (rot == D): the whole slice/slice/neg/concat/mul/mul/add chain -> one Rope op.
    let b = Builder::for_tests();
    let (d, max_pos) = (8usize, 16usize);
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d])).unwrap();
    let cos = b
        .constant("cos", TensorType::f32(vec![max_pos, d]))
        .unwrap();
    let sin = b
        .constant("sin", TensorType::f32(vec![max_pos, d]))
        .unwrap();
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32)).unwrap();
    let out = poot_graph_ir::ops::rope(&b, x, cos, sin, pos).unwrap();
    let g = b.finish(out).unwrap();
    let before = dispatch_count(&g);
    let fused = dce(&rope_fusion(&cse(&g).unwrap()).unwrap()).unwrap();
    let ropes = fused
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Rope { .. }))
        .count();
    assert_eq!(ropes, 1, "exactly one fused Rope op");
    // the rotate-half remnants (neg/concat/the split slices) are dead and gone.
    assert!(
        !fused
            .eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Unary(UnOp::Neg) | OpKind::Concat { .. })),
        "rotate-half neg/concat should be fused away"
    );
    let after = dispatch_count(&fused);
    assert!(
        after < before,
        "fusion cuts dispatches: {before} -> {after}"
    );
    let rope = fused
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Rope { .. }))
        .unwrap();
    assert!(
        matches!(rope.op, OpKind::Rope { rot } if rot == d),
        "rot == D"
    );
}

#[test]
fn rope_fusion_partial_absorbs_passthrough() {
    use poot_graph_ir::graph::Slot;
    use poot_graph_ir::op::UnOp;
    use poot_tensor::DType;
    // partial rotary (rot < D): the outer slice/concat passthrough is absorbed too - Rope over the FULL x.
    let b = Builder::for_tests();
    let (d, rot, max_pos) = (8usize, 4usize, 16usize);
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d])).unwrap();
    let cos = b
        .constant("cos", TensorType::f32(vec![max_pos, rot]))
        .unwrap();
    let sin = b
        .constant("sin", TensorType::f32(vec![max_pos, rot]))
        .unwrap();
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32)).unwrap();
    let out = poot_graph_ir::ops::rope(&b, x, cos, sin, pos).unwrap();
    let g = b.finish(out).unwrap();
    let fused = dce(&rope_fusion(&cse(&g).unwrap()).unwrap()).unwrap();
    let ropes: Vec<&Eqn> = fused
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Rope { .. }))
        .collect();
    assert_eq!(ropes.len(), 1, "one fused Rope op");
    assert!(
        matches!(ropes[0].op, OpKind::Rope { rot: r } if r == rot),
        "rot < D preserved"
    );
    // the Rope input is the FULL head-dim tensor (kernel does the passthrough), so no leftover
    // slice/concat/neg movement remains.
    let x0 = match ropes[0].inputs[0] {
        Operand::Value(v) => v,
        _ => panic!("rope x is a value"),
    };
    assert_eq!(
        fused.aval(x0).shape.last().copied(),
        Some(d),
        "Rope over full x"
    );
    assert!(
        !fused.eqns().iter().any(|e| matches!(
            e.op,
            OpKind::Unary(UnOp::Neg) | OpKind::Concat { .. } | OpKind::Slice { .. }
        )),
        "partial rope absorbs the passthrough slice/concat"
    );
}

/// `compile`'s pipeline runs `cse`, then `rope_fusion`, then (eventually) `dce`: a rotate-half chain
/// `rope_fusion` fuses to one `Rope` must leave no rotate-half remnants once `dce` runs. Without this a
/// pipeline that silently dropped the pass would only show up in a whole-model dispatch count.
/// Mutation observed red: dropping `rope_fusion` from this chain.
#[test]
fn optimize_fuses_the_rope_chain_into_one_rope_op() {
    use poot_graph_ir::graph::Slot;
    use poot_graph_ir::op::UnOp;
    use poot_tensor::DType;
    let b = Builder::for_tests();
    let (d, max_pos) = (8usize, 16usize);
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d])).unwrap();
    let cos = b
        .constant("cos", TensorType::f32(vec![max_pos, d]))
        .unwrap();
    let sin = b
        .constant("sin", TensorType::f32(vec![max_pos, d]))
        .unwrap();
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32)).unwrap();
    let out = poot_graph_ir::ops::rope(&b, x, cos, sin, pos).unwrap();
    let g = b.finish(out).unwrap();
    let optimized = dce(&rope_fusion(&cse(&g).unwrap()).unwrap()).unwrap();
    assert_eq!(
        optimized
            .eqns()
            .iter()
            .filter(|e| matches!(e.op, OpKind::Rope { .. }))
            .count(),
        1,
        "cse, rope_fusion then dce must leave exactly one fused Rope op"
    );
    assert!(
        !optimized
            .eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Unary(UnOp::Neg) | OpKind::Concat { .. })),
        "dce must leave no rotate-half neg/concat remnants"
    );
}

#[test]
fn rope_fusion_ignores_a_non_rope_add_of_two_muls() {
    // legality guard: a plain `a*b + c*d` (no rotate-half concat) must NOT match.
    let b = Builder::for_tests();
    let p = b.constant("p", TensorType::f32(vec![4])).unwrap();
    let q = b.constant("q", TensorType::f32(vec![4])).unwrap();
    let m0 = b.binary(BinOp::Mul, p, q).unwrap();
    let m1 = b.binary(BinOp::Mul, q, p).unwrap();
    let out = b.binary(BinOp::Add, m0, m1).unwrap();
    let g = b.finish(out).unwrap();
    let fused = rope_fusion(&cse(&g).unwrap()).unwrap();
    assert!(
        !fused
            .eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Rope { .. })),
        "a non-rope add of two muls must not match"
    );
}

#[test]
fn cse_remaps_validation_roots() {
    use poot_graph_ir::ValidationId;

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32([4])).unwrap();
    let y = b.constant("y", TensorType::f32([4])).unwrap();
    let primary = b.binary(BinOp::Add, x, y).unwrap();
    let duplicate = b.binary(BinOp::Add, x, y).unwrap();
    let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
        b,
        primary,
        &[(ValidationId(3), "duplicate", duplicate)],
    )
    .unwrap();

    let transformed = cse(&graph).unwrap();
    assert_eq!(transformed.eqns().len(), 1);
    assert_eq!(
        transformed.witnesses().next().unwrap().value,
        transformed.result()
    );
}

#[test]
fn dce_keeps_validation_only_producer() {
    use poot_graph_ir::ValidationId;

    let b = Builder::for_tests();
    let primary = b.constant("primary", TensorType::f32([4])).unwrap();
    let x = b.constant("x", TensorType::f32([4])).unwrap();
    let y = b.constant("y", TensorType::f32([4])).unwrap();
    let witness = b.binary(BinOp::Mul, x, y).unwrap();
    let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
        b,
        primary,
        &[(ValidationId(3), "dce", witness)],
    )
    .unwrap();

    let transformed = dce(&graph).unwrap();
    assert!(transformed.eqns().iter().any(|eqn| eqn.out == witness.id()));
}

#[test]
fn fusion_pins_validation_packet_values() {
    use poot_graph_ir::ValidationId;

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32([4])).unwrap();
    let y = b.constant("y", TensorType::f32([4])).unwrap();
    let witness = b.binary(BinOp::Mul, x, y).unwrap();
    let primary = b.binary(BinOp::Add, witness, y).unwrap();
    let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
        b,
        primary,
        &[(ValidationId(3), "fusion", witness)],
    )
    .unwrap();

    let transformed = fuse(&graph).unwrap();
    assert!(
        transformed.eqns().iter().any(|eqn| eqn.out == witness.id()),
        "the validation root must not be absorbed into its consumer"
    );
}

#[test]
fn movement_cleanup_preserves_validation_only_root() {
    use poot_graph_ir::ValidationId;

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32([1, 2, 1])).unwrap();
    let witness = b.transpose(x, vec![0, 2, 1]).unwrap();
    let reshaped = b.reshape(witness, vec![2]).unwrap();
    let primary = b.reshape(reshaped, vec![1, 2]).unwrap();
    let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
        b,
        primary,
        &[(ValidationId(6), "movement", witness)],
    )
    .unwrap();

    let transformed = collapse_reshape_chains(&elide_noop_transposes(&graph).unwrap()).unwrap();
    assert!(transformed.witnesses().eq(graph.witnesses()));
    assert!(
        transformed
            .eqns()
            .iter()
            .any(|eqn| eqn.out == witness.id() && matches!(eqn.op, OpKind::Reshape { .. })),
        "the validation-only movement value must stay independently materialized"
    );
}

// ---- ADR-0099 S2: the stage split ---------------------------------------------------------

/// A `layers`-block graph: one Activation input, one carried state pair per block, and one shared
/// constant. Three equations per block, so block `l` owns equations `3l..3l + 3`.
fn stage_split_layered_graph(layers: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let w = b.constant("w", TensorType::f32(vec![4])).unwrap();
    let mut hidden = x;
    let mut state = Vec::new();
    for layer in 0..layers {
        let scaled = b.binary(BinOp::Mul, hidden, w).unwrap();
        let k_in = b
            .state_input(
                &format!("k{layer}"),
                TensorType::f32(vec![4]),
                StateRole::Recurrent,
            )
            .unwrap();
        let k_out = b.binary(BinOp::Add, k_in, scaled).unwrap();
        let next = b.binary(BinOp::Add, scaled, k_out).unwrap();
        state.push((k_in, k_out));
        hidden = next;
    }
    b.finish(poot_graph_ir::Outputs::new(hidden).with_state(&state))
        .unwrap()
}

#[test]
fn adr0099_s2_stage_graphs_partition_every_equation_and_validate() {
    let graph = stage_split_layered_graph(3);
    assert_eq!(graph.eqns().len(), 9, "three equations per block");

    for (cuts, expected_boundaries, expected_pairs) in [
        (vec![3usize], 1usize, vec![vec![0usize], vec![1, 2]]),
        (vec![3usize, 6], 2usize, vec![vec![0], vec![1], vec![2]]),
    ] {
        let assignment =
            crate::unit_helpers::assignment_before(&graph, &cuts).expect("the cuts are legal");
        let split = split_stages(&graph, &assignment).expect("the graph splits");
        assert_eq!(split.stages.len(), cuts.len() + 1);
        assert_eq!(split.boundaries.len(), expected_boundaries);

        // Every equation lands in exactly one stage, in order, and none is invented. Stage graphs
        // renumber their values, so the equations are compared by operation and layer, not by id.
        let shape_of = |graph: &Graph| -> Vec<String> {
            graph
                .eqns()
                .iter()
                .map(|eqn| format!("{:?}/{:?}", eqn.op, eqn.layer))
                .collect()
        };
        let mut seen = Vec::new();
        for stage in &split.stages {
            seen.extend(shape_of(&stage.graph));
        }
        assert_eq!(seen, shape_of(&graph));

        // Carried state stays whole: stage `s` owns exactly the pairs listed for it (by declared
        // name), and a pair the stage does not own never reaches its binding tables.
        let state_name = |graph: &Graph, id: ValueId| graph.meta(id).name.clone().unwrap();
        for (stage_index, stage) in split.stages.iter().enumerate() {
            let owned: Vec<String> = stage
                .graph
                .state()
                .map(|(state_input, _)| state_name(&stage.graph, state_input))
                .collect();
            let expected: Vec<String> = expected_pairs[stage_index]
                .iter()
                .map(|&pair| state_name(&graph, graph.state().nth(pair).unwrap().0))
                .collect();
            assert_eq!(
                owned, expected,
                "stage {stage_index} owns its own carried state"
            );
            let bound_state: Vec<String> = stage
                .graph
                .inputs()
                .iter()
                .filter(|&&id| stage.graph.meta(id).storage == Storage::State)
                .map(|&id| state_name(&stage.graph, id))
                .collect();
            assert_eq!(
                bound_state, expected,
                "stage {stage_index} must not bind another stage's cache"
            );
        }

        // Each crossing value is an equation result of its producer, an explicit export there, and a
        // keyed activation input of every consumer.
        for boundary in &split.boundaries {
            let producer = &split.stages[boundary.producing_stage].graph;
            let export = producer
                .exports()
                .find(|export| export.id == boundary.id)
                .unwrap_or_else(|| {
                    panic!(
                        "{:?} is an explicit export of stage {}",
                        boundary.id, boundary.producing_stage
                    )
                });
            assert!(
                producer.eqns().iter().any(|eqn| eqn.out == export.value),
                "the producing stage still defines {:?}",
                boundary.id
            );
            assert_eq!(producer.values()[export.value].storage, Storage::Device);
            for &consumer in &boundary.consuming_stages {
                let stage = &split.stages[consumer].graph;
                let input = stage
                    .inputs()
                    .iter()
                    .copied()
                    .find(|&id| stage.meta(id).slot_key() == Some(&boundary.id.slot_key()))
                    .unwrap_or_else(|| {
                        panic!("{:?} is an explicit input of stage {consumer}", boundary.id)
                    });
                assert_eq!(
                    stage.values()[input].storage,
                    Storage::Slot(poot_graph_ir::graph::Slot::Activation),
                    "{:?} is bound as an activation input of stage {consumer}",
                    boundary.id
                );
                assert!(
                    stage
                        .slots()
                        .contains(&(input, poot_graph_ir::graph::Slot::Activation)),
                    "{:?} has a slot binding in stage {consumer}",
                    boundary.id
                );
                assert!(
                    !stage.eqns().iter().any(|eqn| eqn.out == input),
                    "stage {consumer} reads {:?}, it does not redefine it",
                    boundary.id
                );
            }
        }
    }
}

#[test]
fn adr0099_s2_boundary_descriptors_carry_dtype_shape_and_byte_len() {
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![3, 4]),
        )
        .unwrap();
    let dense = b.binary(BinOp::Add, x, x).unwrap();
    let bf16 = b.cast(dense, DType::BF16).unwrap();
    let i32 = b.cast(dense, DType::I32).unwrap();
    // F32 -> I8 directly is refused at `infer` time (card 555): go through the I32 value
    // already produced above, same as every other I32 -> I8 cast in the IR.
    let i8 = b.cast(i32, DType::I8).unwrap();
    let back_bf16 = b.cast(bf16, DType::F32).unwrap();
    let back_i32 = b.cast(i32, DType::F32).unwrap();
    let back_i8 = b.cast(i8, DType::F32).unwrap();
    let mixed = b.binary(BinOp::Add, back_bf16, back_i32).unwrap();
    let out = b.binary(BinOp::Add, mixed, back_i8).unwrap();
    let graph = b.finish(out).unwrap();
    assert_eq!(graph.eqns().len(), 9);

    let assignment =
        crate::unit_helpers::assignment_before(&graph, &[4]).expect("the cut is legal");
    let split = split_stages(&graph, &assignment).expect("the graph splits");

    // Four values are produced in stage 0, but `dense` is read only there, so exactly three cross -
    // one per dtype, each sized from its own aval (12 elements x 2/4/1 bytes).
    for boundary in &split.boundaries {
        assert_eq!(
            boundary.aval.shape,
            vec![3, 4],
            "a cast keeps the shape, so only the dtype changes the byte count"
        );
    }
    let carried: Vec<(ValueId, DType, usize, usize, Vec<usize>)> = split
        .boundaries
        .iter()
        .map(|boundary| {
            (
                boundary.id.0,
                boundary.aval.dtype,
                boundary.byte_len,
                boundary.producing_stage,
                boundary.consuming_stages.clone(),
            )
        })
        .collect();
    assert_eq!(
        carried,
        vec![
            (bf16.id(), DType::BF16, 24, 0, vec![1]),
            (i32.id(), DType::I32, 48, 0, vec![1]),
            (i8.id(), DType::I8, 12, 0, vec![1]),
        ]
    );
}

#[test]
fn adr0099_s2_state_pair_split_across_stages_is_rejected() {
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let k_in = b
        .state_input("k", TensorType::f32(vec![4]), StateRole::Recurrent)
        .unwrap();
    let early = b.binary(BinOp::Add, x, k_in).unwrap();
    let mid = b.binary(BinOp::Mul, early, early).unwrap();
    let k_out = b.binary(BinOp::Add, mid, mid).unwrap();
    let graph = b
        .finish(poot_graph_ir::Outputs::new(k_out).with_state(&[(k_in, k_out)]))
        .unwrap();

    let assignment =
        crate::unit_helpers::assignment_before(&graph, &[1]).expect("the cut is legal");
    assert_eq!(
        split_stages(&graph, &assignment).expect_err("the pair spans both stages"),
        StageSplitError::StatePairSplit {
            pair: 0,
            state_input: k_in.id(),
            state_output: k_out.id(),
            first_stage: 0,
            second_stage: 1,
        }
    );
}

/// Four chained equations, for cut-position errors.
fn four_eqn_graph() -> Graph {
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let mut hidden = x;
    for _ in 0..4 {
        hidden = b.binary(BinOp::Add, hidden, x).unwrap();
    }
    b.finish(hidden).unwrap()
}

#[test]
fn adr0099_s2_assignment_must_be_contiguous_and_cover_the_graph() {
    assert_eq!(
        StageAssignment::from_stage_per_equation(&[]).expect_err("an empty assignment is no split"),
        StageSplitError::EmptyAssignment
    );
    assert_eq!(
        StageAssignment::from_stage_per_equation(&[1, 1, 1])
            .expect_err("the first equation must be in stage 0"),
        StageSplitError::NonContiguousStage {
            equation: 0,
            stage: 1,
            expected: 0,
        }
    );
    assert_eq!(
        StageAssignment::from_stage_per_equation(&[0, 0, 1, 1, 0])
            .expect_err("a stage may not reopen after a later one started"),
        StageSplitError::NonContiguousStage {
            equation: 4,
            stage: 0,
            expected: 1,
        }
    );
    assert_eq!(
        StageAssignment::from_stage_per_equation(&[0, 2]).expect_err("stage ids may not skip"),
        StageSplitError::NonContiguousStage {
            equation: 1,
            stage: 2,
            expected: 1,
        }
    );
    assert_eq!(
        crate::unit_helpers::assignment_before(&four_eqn_graph(), &[0])
            .expect_err("a cut at 0 empties the first stage"),
        StageSplitError::CutOutOfRange {
            index: 0,
            cut: 0,
            eqn_count: 4,
        }
    );
    assert_eq!(
        crate::unit_helpers::assignment_before(&four_eqn_graph(), &[4])
            .expect_err("a cut at the end empties the last stage"),
        StageSplitError::CutOutOfRange {
            index: 0,
            cut: 4,
            eqn_count: 4,
        }
    );
    assert_eq!(
        crate::unit_helpers::assignment_before(&four_eqn_graph(), &[2, 2])
            .expect_err("cuts must strictly increase"),
        StageSplitError::CutOutOfOrder {
            index: 1,
            cut: 2,
            previous: 2,
        }
    );

    let graph = stage_split_layered_graph(3);
    let assignment = StageAssignment::from_stage_per_equation(&[0, 0]).expect("a legal sequence");
    assert_eq!(
        split_stages(&graph, &assignment).expect_err("the assignment does not cover the graph"),
        StageSplitError::AssignmentLength {
            assigned: 2,
            eqn_count: 9,
        }
    );
}

#[test]
fn adr0099_s2_a_stage_that_hands_off_nothing_is_rejected() {
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let t0 = b.binary(BinOp::Add, x, x).unwrap();
    let _d1 = b.binary(BinOp::Mul, t0, t0).unwrap();
    let out = b.binary(BinOp::Mul, t0, t0).unwrap();
    let graph = b.finish(out).unwrap();

    let assignment =
        crate::unit_helpers::assignment_before(&graph, &[1, 2]).expect("the cuts are legal");
    assert_eq!(
        split_stages(&graph, &assignment)
            .expect_err("stage 1 reads only its own results, so it never feeds stage 2"),
        StageSplitError::StageWithoutExit { stage: 1 }
    );
}

#[test]
fn adr0099_s2_a_final_stage_that_never_sees_the_output_is_rejected() {
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let t0 = b.binary(BinOp::Add, x, x).unwrap();
    let t1 = b.binary(BinOp::Mul, t0, t0).unwrap();
    let _after_t1 = b.binary(BinOp::Mul, t1, t1).unwrap();
    let graph = b.finish(t0).unwrap();

    let assignment =
        crate::unit_helpers::assignment_before(&graph, &[1, 2]).expect("the cuts are legal");
    assert_eq!(
        split_stages(&graph, &assignment)
            .expect_err("the output is defined in stage 0 and reaches no later stage"),
        StageSplitError::FinalStageMissingOutput {
            stage: 2,
            output: t0.id(),
        }
    );
}

#[test]
fn adr0099_s2_each_stage_keeps_only_the_witnesses_it_defines() {
    use poot_graph_ir::ValidationId;

    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let t0 = b.binary(BinOp::Add, x, x).unwrap();
    let w0 = b.binary(BinOp::Mul, x, x).unwrap();
    let t1 = b.binary(BinOp::Mul, t0, t0).unwrap();
    let w1 = b.binary(BinOp::Sub, t1, t1).unwrap();
    let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
        b,
        t1,
        &[
            (ValidationId(1), "stage0_witness", w0),
            (ValidationId(2), "stage1_witness", w1),
        ],
    )
    .expect("the witness values are f32 tensors");

    let assignment =
        crate::unit_helpers::assignment_before(&graph, &[2]).expect("the cut is legal");
    let split = split_stages(&graph, &assignment).expect("the graph splits");

    // Stage graphs renumber their values, so a witness is identified by what computes it: the
    // producing equation's op and what each operand is (an equation's op, or the binder's storage).
    fn describe(graph: &Graph, value: ValueId) -> String {
        match graph.eqns().iter().find(|eqn| eqn.out == value) {
            Some(eqn) => format!("{:?}", eqn.op),
            None => format!("{:?}", graph.meta(value).storage),
        }
    }
    fn computation(graph: &Graph, value: ValueId) -> (String, Vec<String>) {
        let eqn = graph
            .eqns()
            .iter()
            .find(|eqn| eqn.out == value)
            .expect("a witness is an equation result");
        let operands = eqn
            .inputs
            .iter()
            .map(|operand| match operand {
                poot_graph_ir::Operand::Value(v) => describe(graph, *v),
                poot_graph_ir::Operand::Lit(scalar) => format!("{scalar:?}"),
            })
            .collect();
        (format!("{:?}", eqn.op), operands)
    }
    let declarations =
        |stage: usize| -> Vec<(poot_graph_ir::ValidationId, String, (String, Vec<String>))> {
            let stage_graph = &split.stages[stage].graph;
            stage_graph
                .witnesses()
                .map(|declaration| {
                    (
                        declaration.id,
                        declaration.name.to_owned(),
                        computation(stage_graph, declaration.value),
                    )
                })
                .collect()
        };
    assert_eq!(
        declarations(0),
        vec![(
            ValidationId(1),
            "stage0_witness".to_owned(),
            computation(&graph, w0.id())
        )]
    );
    assert_eq!(
        declarations(1),
        vec![(
            ValidationId(2),
            "stage1_witness".to_owned(),
            computation(&graph, w1.id())
        )]
    );
}

/// A traced graph with an untagged preamble, three scoped layers of four equations each, and an
/// untagged head - the shape a layer-pipeline placement cuts. Within a layer: a `Mul`, a duplicate
/// of that `Mul` (CSE folds it), a dead `Add` (DCE drops it), and the `Add` that carries the
/// hidden state forward.
fn layer_scoped_graph() -> Graph {
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let w = b.constant("w", TensorType::f32(vec![4])).unwrap();
    let _preamble = b.binary(BinOp::Add, x, x).unwrap();
    let mut hidden = x;
    for layer in 0..3usize {
        let _layer_scope = b.layer_scope(layer);
        let scaled = b.binary(BinOp::Mul, hidden, w).unwrap();
        let duplicate = b.binary(BinOp::Mul, hidden, w).unwrap();
        let _dead = b.binary(BinOp::Add, duplicate, duplicate).unwrap();
        hidden = b.binary(BinOp::Add, scaled, x).unwrap();
    }
    let out = b.binary(BinOp::Mul, hidden, hidden).unwrap();
    b.finish(out).unwrap()
}

fn layer_tags(graph: &Graph) -> Vec<Option<usize>> {
    graph
        .eqns()
        .iter()
        .map(|eqn| eqn.layer.map(|layer| layer.0))
        .collect()
}

#[test]
fn adr0099_s2_layer_scope_tags_equations_and_layer_cuts_split_between_layers() {
    let graph = layer_scoped_graph();
    assert_eq!(
        layer_tags(&graph),
        vec![
            None, // preamble
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(2),
            Some(2),
            None, // head
        ],
        "the scope tags exactly the equations emitted inside it"
    );

    // `after_layer = [0, 1]` is the cut list a three-device plan records; each cut lands on the
    // first equation of the next layer, so the preamble rides stage 0 and the head the last stage.
    let split = crate::unit_helpers::split_after_layers(&graph, &[0, 1])
        .expect("both cuts map onto a boundary");
    let per_stage: Vec<Vec<Option<usize>>> = split
        .stages
        .iter()
        .map(|stage| layer_tags(&stage.graph))
        .collect();
    assert_eq!(
        per_stage,
        vec![
            vec![None, Some(0), Some(0), Some(0), Some(0)],
            vec![Some(1), Some(1), Some(1), Some(1)],
            vec![Some(2), Some(2), Some(2), Some(2), None],
        ]
    );
    assert_eq!(
        split
            .boundaries
            .iter()
            .map(|boundary| (boundary.producing_stage, boundary.consuming_stages.clone()))
            .collect::<Vec<_>>(),
        vec![(0, vec![1]), (1, vec![2])],
        "one crossing value per layer hand-off, each into the next stage"
    );

    // A single cut after layer 0: stage 0 holds layer 0, stage 1 everything from layer 1 on.
    let two = crate::unit_helpers::split_after_layers(&graph, &[0]).expect("after_layer 0 maps");
    assert_eq!(two.stages.len(), 2);
    assert_eq!(
        layer_tags(&two.stages[0].graph),
        vec![None, Some(0), Some(0), Some(0), Some(0)]
    );
    assert_eq!(
        layer_tags(&two.stages[1].graph),
        vec![
            Some(1),
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(2),
            Some(2),
            None
        ]
    );
    assert_eq!(two.boundaries.len(), 1, "one value crosses the layer-0 cut");
}

#[test]
fn adr0099_s2_layer_cut_errors_are_typed() {
    // An untagged equation between two tagged layers: which side would a cut give it?
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let in_layer_zero = {
        let _layer_scope = b.layer_scope(0);
        b.binary(BinOp::Mul, x, x).unwrap()
    };
    let hole = b.binary(BinOp::Add, x, x).unwrap();
    let in_layer_one = {
        let _layer_scope = b.layer_scope(1);
        b.binary(BinOp::Mul, hole, hole).unwrap()
    };
    let holey = b.finish(in_layer_one).unwrap();
    let _ = in_layer_zero;
    assert_eq!(
        crate::unit_helpers::assignment_after_layers(&holey, &[0]),
        Err(StageSplitError::LayerTags(LayerTagError::Untagged {
            equation: 1
        }))
    );

    // A skipped layer id: layer 2 appears where layer 1 should start.
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    {
        let _layer_scope = b.layer_scope(0);
        b.binary(BinOp::Mul, x, x).unwrap();
    }
    let skipped = {
        let _layer_scope = b.layer_scope(2);
        b.binary(BinOp::Mul, x, x).unwrap()
    };
    let skipped = b.finish(skipped).unwrap();
    assert_eq!(
        crate::unit_helpers::assignment_after_layers(&skipped, &[0]),
        Err(StageSplitError::LayerTags(LayerTagError::NonContiguous {
            equation: 1,
            layer: 2,
            expected: 1,
        }))
    );

    // A graph whose first tagged equation is not layer 0 never had a layer 0.
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let started_at_one = {
        let _layer_scope = b.layer_scope(1);
        b.binary(BinOp::Mul, x, x).unwrap()
    };
    let started_at_one = b.finish(started_at_one).unwrap();
    assert_eq!(
        crate::unit_helpers::assignment_after_layers(&started_at_one, &[0]),
        Err(StageSplitError::LayerTags(LayerTagError::NonContiguous {
            equation: 0,
            layer: 1,
            expected: 0,
        }))
    );

    // A cut past the last layer would produce an empty stage, and a cut on an untagged graph has
    // no layer to land on at all. (A cut at layer 1 of three is legal - it separates 1 from 2.)
    let tagged = layer_scoped_graph();
    assert!(crate::unit_helpers::assignment_after_layers(&tagged, &[1]).is_ok());
    assert_eq!(
        crate::unit_helpers::assignment_after_layers(&tagged, &[2]),
        Err(StageSplitError::LayerCutOutOfRange { cut: 2, layers: 3 })
    );
    let b = Builder::for_tests();
    let x = b
        .slot(
            poot_graph_ir::graph::Slot::Activation,
            TensorType::f32(vec![4]),
        )
        .unwrap();
    let untagged_value = b.binary(BinOp::Add, x, x).unwrap();
    let untagged = b.finish(untagged_value).unwrap();
    assert_eq!(
        crate::unit_helpers::assignment_after_layers(&untagged, &[0]),
        Err(StageSplitError::LayerCutOutOfRange { cut: 0, layers: 0 })
    );

    // Cuts must also strictly increase, the same contract an equation cut enforces.
    assert_eq!(
        crate::unit_helpers::assignment_after_layers(&tagged, &[0, 0]),
        Err(StageSplitError::CutOutOfOrder {
            index: 1,
            cut: 5,
            previous: 5,
        })
    );
}

#[test]
fn adr0099_s2_cse_and_dce_preserve_layer_tags() {
    let graph = layer_scoped_graph();
    let original: HashMap<ValueId, Option<LayerIndex>> = graph
        .eqns()
        .iter()
        .map(|eqn| (eqn.out, eqn.layer))
        .collect();

    let folded = cse(&graph).unwrap();
    assert!(
        folded.eqns().len() < graph.eqns().len(),
        "cse must fold the duplicate Mul in each layer, or this row proves nothing"
    );
    for eqn in folded.eqns() {
        assert_eq!(
            eqn.layer, original[&eqn.out],
            "cse changed the layer tag of {}",
            eqn.out
        );
    }

    let trimmed = dce(&graph).unwrap();
    assert!(
        trimmed.eqns().len() < graph.eqns().len(),
        "dce must drop the dead layer equation, or this row proves nothing"
    );
    for eqn in trimmed.eqns() {
        assert_eq!(
            eqn.layer, original[&eqn.out],
            "dce changed the layer tag of {}",
            eqn.out
        );
    }

    // Preserved tags mean a layer cut still maps on the transformed graphs.
    for (label, transformed) in [("cse", &folded), ("dce", &trimmed)] {
        let split = crate::unit_helpers::split_after_layers(transformed, &[0, 1])
            .unwrap_or_else(|error| panic!("{label} must keep the tags a cut maps onto: {error}"));
        assert_eq!(split.stages.len(), 3, "{label} keeps three layers");
        assert_eq!(
            split
                .stages
                .iter()
                .map(|stage| stage.graph.eqns().iter().any(|eqn| eqn.layer.is_some()))
                .collect::<Vec<_>>(),
            vec![true, true, true],
            "{label} leaves every layer with equations of its own"
        );
    }
}

/// Card 533 SC-003: a pipeline-level check compares the witness outputs before and after every pass the
/// `optimize` pipeline runs. The structural half lives here: the declaration survives and the witness
/// producer stays defined. The value half over the CPU oracle lives with the evaluator
/// (`poot-eval/src/exact_i32.rs`).
///
/// Mutation: drop validation roots from `fuse`'s pinned set; the witness producer is absorbed and this
/// test goes red with "fuse dropped the witness producer".
#[test]
fn card533_optimize_passes_keep_witness_outputs() {
    use poot_graph_ir::ValidationId;
    type PassFn = fn(&Graph) -> Result<Graph, poot_graph_ir::EditError>;

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32([4])).unwrap();
    let y = b.constant("y", TensorType::f32([4])).unwrap();
    let witness = b.binary(BinOp::Mul, x, y).unwrap();
    let primary = b.binary(BinOp::Add, witness, y).unwrap();
    let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
        b,
        primary,
        &[(ValidationId(533), "card533", witness)],
    )
    .unwrap();

    let passes: [(&str, PassFn); 6] = [
        ("cse", cse),
        ("rope_fusion", rope_fusion),
        ("flash_attention_capped", |g| {
            flash_attention_capped(g, None)
        }),
        ("dce", dce),
        ("fuse_contraction_epilogues", |g| {
            fuse_contraction_epilogues(g, |_, _| true)
        }),
        ("fuse", fuse),
    ];
    for (name, pass) in passes {
        let transformed = pass(&graph).unwrap();
        assert!(
            transformed.witnesses().eq(graph.witnesses()),
            "{name} dropped or changed a witness declaration"
        );
        assert!(
            transformed.witnesses().all(|output| {
                transformed.eqns().iter().any(|eqn| eqn.out == output.value)
                    || transformed.consts().contains(&output.value)
            }),
            "{name} dropped the witness producer"
        );
    }
}

/// Card 533 SC-004: a pass whose declaration is weaker than the ops it actually produced is refused,
/// and the pipeline publishes the tier-2 class its graph exhibits.
///
/// Mutation: set the `fuse` row of `PASS_DECLARATIONS` to `Exact`; `verify_pass_numerics`
/// returns `UnderDeclared { pass: "fuse", declared: Exact, implied: Reassociating }` and this test goes red.
#[test]
fn card533_pass_numerics_are_checked_against_produced_ops() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32([4])).unwrap();
    let y = b.constant("y", TensorType::f32([4])).unwrap();
    let product = b.binary(BinOp::Mul, x, y).unwrap();
    let sum = b.binary(BinOp::Add, product, y).unwrap();
    let graph = b.finish(sum).unwrap();

    let fused = fuse(&graph).unwrap();
    assert!(
        fused
            .eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Fused(_))),
        "the pointwise chain must fuse, or this row proves nothing"
    );
    assert_eq!(implied_numerics(&fused), NumericsProperty::Reassociating);

    let declaration = PASS_DECLARATIONS
        .iter()
        .find(|declaration| declaration.pass == "fuse")
        .expect("fuse is declared");
    assert_eq!(declaration.property, NumericsProperty::Reassociating);
    verify_pass_numerics(declaration, &fused).unwrap();

    let misdeclared = PassDeclaration {
        property: NumericsProperty::Exact,
        ..*declaration
    };
    assert_eq!(
        verify_pass_numerics(&misdeclared, &fused).unwrap_err(),
        NumericsError::UnderDeclared {
            pass: "fuse",
            declared: NumericsProperty::Exact,
            implied: NumericsProperty::Reassociating,
        }
    );
}

/// Card 533: an `M == 1` decode `MatMul` survives `optimize` unfused, and a real device is not bit-exact against the CPU oracle for ordinary float
/// contraction (ADR-0101 Context: the unoptimized wgpu route differs in every logit). The published
/// class must be the unfused hardware baseline, not `BitExact`.
#[test]
fn card533_unfused_decode_matmul_is_not_bit_exact() {
    let b = Builder::for_tests();
    let activation = b.constant("activation", TensorType::f32([1, 8])).unwrap();
    let weight = b.constant("weight", TensorType::f32([8, 8])).unwrap();
    let out = b.matmul(activation, weight).unwrap();
    let graph = b.finish(out).unwrap();
    assert!(
        matches!(
            graph.eqns(),
            [eqn] if matches!(eqn.op, OpKind::MatMul)
        ),
        "the M==1 decode MatMul must stay a bare MatMul (no pass retags it), or this row proves nothing"
    );
    let property = implied_numerics(&graph);
    assert_eq!(property, NumericsProperty::Unfused);
    assert_ne!(property.tier2_class(), Tier2Class::BitExact);
    assert_eq!(property.tier2_class(), Tier2Class::Unfused);

    // BitExact stays reachable for a graph that runs no device-executed float arithmetic: movement
    // only, bit-preserving.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32([2, 2])).unwrap();
    let moved = b.reshape(x, vec![4]).unwrap();
    let movement = b.finish(moved).unwrap();
    assert_eq!(
        implied_numerics(&movement).tier2_class(),
        Tier2Class::BitExact
    );
}

// ---- Card 536a: matchers are legality checks ----------------
// Pure-reason / pure-structure tests: no oracle (`poot_eval`), so these run as ordinary
// `#[cfg(test)]` unit tests against `pub(super)` internals, same crate. The oracle-comparison half
// of the suite (SC-001 metamorphic fuse, the tiling/batched-decode typing oracles, and the
// oracle-unchanged half of the near-miss/head-dim checks) lives in the integration test
// `crates/poot-graph-ir/tests/transform_soundness.rs`, which cannot see these `pub(super)` items
// (it's a separate crate - see that file's header for why it has to be one).
use super::attention_match::{
    AttentionDecline, RopeDecline, canonicalize, match_attention, match_rope_core,
};
use super::guard_witness::producer_map;

fn output_eqn(g: &Graph) -> &Eqn {
    g.eqns()
        .iter()
        .find(|eqn| eqn.out == g.result())
        .expect("the graph output has a producer")
}

/// A full-rotary rotate-half rope with free `x`/`cos`/`sin` shapes; `commute` writes `Mul(cos, x)`.
fn hand_rope(x_shape: Vec<usize>, cos_shape: Vec<usize>, commute: bool) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(x_shape.clone())).unwrap();
    let cos = b
        .constant("cos", TensorType::f32(cos_shape.clone()))
        .unwrap();
    let sin = b.constant("sin", TensorType::f32(cos_shape)).unwrap();
    let last = x_shape.len() - 1;
    let d = x_shape[last];
    let half = d / 2;
    let x1 = b.slice(x, last, 0, half).unwrap();
    let x2 = b.slice(x, last, half, d).unwrap();
    let neg_x2 = b.unary(UnOp::Neg, x2).unwrap();
    let rotate_half = b.concat(last, &[neg_x2, x1]).unwrap();
    let xc = if commute {
        b.binary(BinOp::Mul, cos, x).unwrap()
    } else {
        b.binary(BinOp::Mul, x, cos).unwrap()
    };
    let rs = b.binary(BinOp::Mul, rotate_half, sin).unwrap();
    let out = b.binary(BinOp::Add, xc, rs).unwrap();
    b.finish(out).unwrap()
}

/// SC-002 near-miss attention decline reasons, by semantic detail (the oracle-unchanged half of
/// each case lives in `transform_soundness.rs`'s `card536a_attention_near_misses_decline_with_a_reason`).
/// Mutation observed red: widen the matching check named in the reason; the fixture fuses and this
/// reason assertion fails.
#[test]
fn card536a_attention_near_misses_decline_with_a_reason() {
    // (name, graph, expected reason)
    let mut cases: Vec<(&str, Graph, AttentionDecline)> = Vec::new();

    // Axis: the row softmax reduces the wrong axis.
    let wrong_axis = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
            .unwrap();
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask).unwrap();
        b.finish(out).unwrap()
    };
    let wrong_axis = mutated(&wrong_axis, |draft| {
        for eqn in draft.eqns_mut().iter_mut() {
            if let OpKind::Reduce { axis, .. } = &mut eqn.op {
                *axis = 0;
            }
        }
    });
    cases.push(("wrong reduce axis", wrong_axis, AttentionDecline::Denom));

    // Scale: a softcap chain whose two constants are not reciprocals.
    let wrong_softcap = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
            .unwrap();
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(2.0)).unwrap();
        b.finish(out).unwrap()
    };
    let mut rewrote = false;
    let wrong_softcap = mutated(&wrong_softcap, |draft| {
        for eqn in draft.eqns_mut().iter_mut() {
            if let OpKind::Binary(BinOp::Mul) = eqn.op
                && let Some(Operand::Lit(poot_graph_ir::Scalar::F32(0.5))) = eqn.inputs.get(1)
            {
                eqn.inputs[1] = Operand::Lit(poot_graph_ir::Scalar::F32(0.6));
                rewrote = true;
            }
        }
    });
    assert!(rewrote, "the softcap reciprocal literal must be present");
    cases.push((
        "softcap constants not reciprocal",
        wrong_softcap,
        AttentionDecline::SoftcapRecip,
    ));

    // Shape: a softcap chain whose `Tanh` slot holds another unary (here `Neg`): same literals, same
    // arity, but not a tanh. Mutation: drop the `Tanh` check in `match_pre_mask`; this fuses.
    let not_tanh = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
            .unwrap();
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(2.0)).unwrap();
        b.finish(out).unwrap()
    };
    let mut swapped = 0;
    let not_tanh = mutated(&not_tanh, |draft| {
        for eqn in draft.eqns_mut().iter_mut() {
            if matches!(eqn.op, OpKind::Unary(poot_graph_ir::UnOp::Tanh)) {
                eqn.op = OpKind::Unary(poot_graph_ir::UnOp::Neg);
                swapped += 1;
            }
        }
    });
    assert_eq!(swapped, 1, "exactly the softcap tanh is replaced");
    cases.push((
        "softcap tanh slot is not a tanh",
        not_tanh,
        AttentionDecline::SoftcapShape,
    ));

    // Type gate: a shared single-head key the decomposition merely broadcasts (MQA written by broadcasting
    // instead of `repeat_kv`); the fused op indexes one key head per group, so its typing rule refuses.
    let broadcast_key = {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 3])).unwrap();
        let k = b.constant("k", TensorType::f32(vec![1, 1, 5, 3])).unwrap();
        let v = b.constant("v", TensorType::f32(vec![1, 1, 5, 2])).unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, 1, 5]))
            .unwrap();
        let out = attention_masked(&b, q, k, v, 1, 1.0, mask).unwrap();
        b.finish(out).unwrap()
    };
    cases.push((
        "a size-1 key head broadcast by the matmul",
        broadcast_key,
        AttentionDecline::TypeMismatch,
    ));

    for (name, graph, expected) in cases {
        let reason = match_attention(
            &graph,
            &producer_map(&graph),
            &value_ranges(&graph),
            output_eqn(&graph),
        );
        assert_eq!(reason.err(), Some(expected), "{name}: wrong decline reason");
    }
}

/// SC-002/SC-003 near-miss rope, by semantic detail, with typed reasons. Mutation observed red: widen
/// the named check; the near-miss fuses or returns a different reason.
#[test]
fn card536a_rope_near_misses_decline_with_a_reason() {
    let producer =
        |graph: &Graph| match_rope_core(graph, &producer_map(graph), output_eqn(graph)).err();
    let mut cases: Vec<(&str, Graph, RopeDecline)> = Vec::new();

    // The two slices do not split the same tensor: cut the keep slice from a different tensor.
    let wrong_source = mutated(
        &hand_rope(vec![1, 4, 3, 8], vec![1, 1, 3, 8], false),
        |draft| {
            let other = draft
                .values_mut()
                .push(poot_graph_ir::graph::ValueMeta::new(
                    TensorType::f32(vec![1, 4, 3, 8]),
                    Storage::Const,
                    Some("other".into()),
                ));
            draft.inputs_mut().push(other);
            draft.consts_mut().push(other);
            for eqn in draft.eqns_mut().iter_mut() {
                let keep_slice = matches!(&eqn.op, OpKind::Slice { start: 0, .. });
                if keep_slice && let Some(Operand::Value(value)) = eqn.inputs.first_mut() {
                    *value = other;
                }
            }
        },
    );
    cases.push((
        "slices from different tensors",
        wrong_source,
        RopeDecline::SliceMismatch,
    ));

    for (name, graph, expected) in cases {
        assert_eq!(
            producer(&graph),
            Some(expected),
            "{name}: wrong decline reason"
        );
    }
}

/// SC-003: each surviving legality guard turns a test red when mutated. This pins the
/// reasons the three surviving R486 guards produce, so dropping one changes its fixture's reason:
///
/// - the softcap reciprocal-literal check (`SoftcapRecip`),
/// - the K/V GQA repeat cross-check (`GqaRepeatMismatch`),
/// - the row-softmax reduce-axis check (`Denom`, the surviving form of the deleted softmax matcher's
///   `softmax.rs:27` guard).
#[test]
fn card536a_guards_have_killing_reasons() {
    // A softcapped decode matches: the generated kernel takes the softcap on every phase. Mutation: restore a
    // decode-softcap decline in `match_attention`; this match fails with that decline.
    let decode_softcap = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hq, cap, d]))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
            .unwrap();
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(30.0)).unwrap();
        b.finish(out).unwrap()
    };
    let matched = match_attention(
        &decode_softcap,
        &producer_map(&decode_softcap),
        &value_ranges(&decode_softcap),
        output_eqn(&decode_softcap),
    );
    assert_eq!(
        matched.as_ref().err(),
        None,
        "a softcapped decode must match"
    );
    assert_eq!(matched.unwrap().softcap, Some(30.0));

    // GQA repeat cross-check: k is repeated by 2, v is not. The chain stays a valid decomposition
    // (V keeps four heads) but the repeated K carries a different factor.
    let (hq, cap, d, hkv) = (4usize, 6usize, 16usize, 2usize);
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    let kt = b
        .transpose(
            poot_graph_ir::ops::repeat_kv(&b, k, 2).unwrap(),
            vec![0, 1, 3, 2],
        )
        .unwrap();
    let scores = b.matmul(q, kt).unwrap();
    let scores = b
        .binary_scalar(BinOp::Mul, scores, poot_graph_ir::Scalar::F32(0.25))
        .unwrap();
    let scores = b.binary(BinOp::Add, scores, mask).unwrap();
    let last = b.aval(scores).unwrap().rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true).unwrap();
    let e = b
        .unary(UnOp::Exp, b.binary(BinOp::Sub, scores, m).unwrap())
        .unwrap();
    let denom = b.reduce(RedOp::Sum, e, last, true).unwrap();
    let p = b.binary(BinOp::Div, e, denom).unwrap();
    let gqa_out = b.matmul(p, v).unwrap();
    let gqa = b.finish(gqa_out).unwrap();
    assert_eq!(
        match_attention(
            &gqa,
            &producer_map(&gqa),
            &value_ranges(&gqa),
            output_eqn(&gqa)
        )
        .err(),
        Some(AttentionDecline::GqaRepeatMismatch)
    );
}

/// The matcher puts no head-dim limit on a match: the planner's device caps decide whether a kernel fits,
/// so a decode and a prefill at head dim 512 both fuse. Mutation: restore a head-dim decline in
/// `match_attention`; this assertion fails with that decline.
#[test]
fn a_wide_head_dim_still_matches() {
    let build = |m: usize, d: usize| {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, 2, m, d])).unwrap();
        let k = b.constant("k", TensorType::f32(vec![1, 2, 5, d])).unwrap();
        let v = b.constant("v", TensorType::f32(vec![1, 2, 5, d])).unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, m, 5]))
            .unwrap();
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask).unwrap();
        b.finish(out).unwrap()
    };
    for m in [1, 3] {
        let wide = build(m, 512);
        let matched = match_attention(
            &wide,
            &producer_map(&wide),
            &value_ranges(&wide),
            output_eqn(&wide),
        );
        assert!(
            matched.is_ok(),
            "M={m} D=512 must match: {:?}",
            matched.err()
        );
    }
}

/// MLA: a split key (`concat(k_nope, repeat_kv(k_pe))`, the form `deepseek2` traces) with a value narrower
/// than the key matches as a `Split` attention with `dk != dv`, over the per-head part and the shared part.
/// Mutation: drop the split-key recognition in `peel_key`; the key stays one `Concat` and the match fails
/// the typing gate (`TypeMismatch`).
#[test]
fn a_split_key_with_a_narrower_value_matches_as_a_split_attention() {
    let (hq, cap, nope, rope, dv) = (2usize, 5usize, 6usize, 4usize, 3usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, 1, nope + rope]))
        .unwrap();
    let k_nope = b
        .constant("k_nope", TensorType::f32(vec![1, hq, cap, nope]))
        .unwrap();
    let k_pe = b
        .constant("k_pe", TensorType::f32(vec![1, 1, cap, rope]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hq, cap, dv]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    let k = b
        .concat(
            3,
            &[k_nope, poot_graph_ir::ops::repeat_kv(&b, k_pe, hq).unwrap()],
        )
        .unwrap();
    let out = attention_masked(&b, q, k, v, 1, 0.25, mask).unwrap();
    let graph = b.finish(out).unwrap();
    let matched = match_attention(
        &graph,
        &producer_map(&graph),
        &value_ranges(&graph),
        output_eqn(&graph),
    )
    .unwrap_or_else(|decline| panic!("the MLA chain must match, declined {decline:?}"));
    assert_eq!(matched.k, k_nope.id());
    assert_eq!(matched.k_shared, Some(k_pe.id()));
    assert_eq!(matched.key, KeyLayout::Split { shared_dim: rope });
    assert_eq!(matched.len, None);
    let OpKind::Attention(attrs) = matched.op() else {
        panic!("the match collapses to an attention op")
    };
    assert_eq!(
        (attrs.key, attrs.bound),
        (KeyLayout::Split { shared_dim: rope }, KvBound::Capacity)
    );
}

/// A live-capacity decode over `cap` keys, built by the one composite ([`ops::attention_live`]): batch 2, two
/// key heads shared by four query heads, an additive mask, an optional softcap, a split key when `split`,
/// and the raw I32 `len`. The length witness is declared as a validation output when `validated`.
fn live_attention_graph(split: bool, validated: bool) -> poot_graph_ir::Graph {
    let (batch, hq, hkv, cap, d, dv) = (2usize, 4usize, 2usize, 6usize, 8usize, 5usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![batch, hq, 1, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![batch, hkv, cap, dv]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![batch, 1, 1, cap]))
        .unwrap();
    let len = b
        .constant("len", TensorType::new(vec![batch], poot_tensor::DType::I32))
        .unwrap();
    let keys = if split {
        poot_graph_ir::ops::AttentionKeys::Split {
            head: b
                .constant("k_head", TensorType::f32(vec![batch, hkv, cap, d - 3]))
                .unwrap(),
            shared: b
                .constant("k_shared", TensorType::f32(vec![batch, 1, cap, 3]))
                .unwrap(),
        }
    } else {
        poot_graph_ir::ops::AttentionKeys::Single(
            b.constant("k", TensorType::f32(vec![batch, hkv, cap, d]))
                .unwrap(),
        )
    };
    let live =
        poot_graph_ir::ops::attention_live(&b, q, keys, v, hq / hkv, 0.25, mask, Some(30.0), len)
            .expect("the live composite builds");
    let outputs = if validated {
        poot_graph_ir::Outputs::new(live.out).with_witness(
            poot_graph_ir::ValidationId(1),
            "attention.len",
            live.witness,
        )
    } else {
        poot_graph_ir::Outputs::new(live.out)
    };
    b.finish(outputs)
        .expect("the live composite is a valid graph")
}

/// The float selects of a live composite in construction order: the key source(s), the value, the scores.
fn float_selects(graph: &poot_graph_ir::Graph) -> Vec<usize> {
    graph
        .eqns()
        .iter()
        .enumerate()
        .filter(|(_, eqn)| {
            matches!(eqn.op, OpKind::Select) && graph.aval(eqn.out).dtype == poot_tensor::DType::F32
        })
        .map(|(index, _)| index)
        .collect()
}

/// Replace the select at `index` with a reshape of its data operand: the same value, no guard.
fn remove_guard(graph: &poot_graph_ir::Graph, index: usize) -> poot_graph_ir::Graph {
    let data = graph.eqns()[index].inputs[1];
    let shape = graph.aval(graph.eqns()[index].out).shape.clone();
    mutated(graph, |draft| {
        let eqn = &mut draft.eqns_mut()[index];
        eqn.op = OpKind::Reshape { shape };
        eqn.inputs = vec![data];
    })
}

/// The guarded live-capacity composite matches as a `Live` attention over the proved length, for a dense and a
/// split key. Mutation: emit the live bound from the score select alone (accept an unguarded source in
/// `peel_live_rows`); the near-miss rows below stop declining.
#[test]
fn the_guarded_live_composite_matches_as_a_live_attention() {
    for split in [false, true] {
        let graph = live_attention_graph(split, true);
        let matched = match_attention(
            &graph,
            &producer_map(&graph),
            &value_ranges(&graph),
            output_eqn(&graph),
        )
        .unwrap_or_else(|decline| panic!("split={split}: declined {decline:?}"));
        let len = matched.len.expect("a live match carries its length");
        assert_eq!(matched.k_shared.is_some(), split);
        let OpKind::Attention(attrs) = matched.op() else {
            panic!("the match collapses to an attention op")
        };
        assert_eq!(attrs.bound, KvBound::Live);
        assert_eq!(matched.operands().last(), Some(&len));
        assert_eq!(graph.aval(len).dtype, poot_tensor::DType::I32);
        assert_eq!(attrs.softcap, Some(30.0));
    }
}

/// Everything short of the full guard stays the generic chain, with a typed reason: a source without its
/// select (K, V), a condition one key off, a condition over another length, a length no declared witness
/// bounds by the capacity, and a length with no witness at all (POOT-726: the length's range comes from the
/// guard's clamp, so its witness must be a declared validation output). Mutation: accept the named shape in the
/// matcher; the row's near miss matches instead.
#[test]
fn live_near_misses_decline_with_a_reason() {
    use poot_graph_ir::{BinOp as B, Scalar};
    let mut cases: Vec<(&str, poot_graph_ir::Graph, AttentionDecline)> = Vec::new();

    let base = live_attention_graph(false, true);
    let no_key_guard = remove_guard(&base, float_selects(&base)[0]);
    cases.push((
        "K without its guard",
        no_key_guard,
        AttentionDecline::LiveKeyGuard,
    ));

    let base = live_attention_graph(false, true);
    let no_value_guard = remove_guard(&base, float_selects(&base)[1]);
    cases.push((
        "V without its guard",
        no_value_guard,
        AttentionDecline::LiveKeyGuard,
    ));

    let base = live_attention_graph(true, true);
    let no_shared_guard = remove_guard(&base, float_selects(&base)[1]);
    cases.push((
        "the shared part of a split key without its guard",
        no_shared_guard,
        AttentionDecline::LiveKeyGuard,
    ));

    // One key off: the key positions start at 2 instead of 1.
    let off_by_one = live_attention_graph(false, true);
    let iotas: Vec<ValueId> = off_by_one
        .eqns()
        .iter()
        .filter(|eqn| {
            matches!(
                eqn.op,
                OpKind::Iota {
                    dtype: poot_tensor::DType::I32,
                    ..
                }
            )
        })
        .map(|eqn| eqn.out)
        .collect();
    let mut changed = 0;
    let off_by_one = mutated(&off_by_one, |draft| {
        for eqn in draft.eqns_mut().iter_mut() {
            if matches!(eqn.op, OpKind::Binary(B::Add))
                && matches!(eqn.inputs.first(), Some(Operand::Value(id)) if iotas.contains(id))
                && matches!(eqn.inputs.get(1), Some(Operand::Lit(Scalar::I32(1))))
            {
                eqn.inputs[1] = Operand::Lit(Scalar::I32(2));
                changed += 1;
            }
        }
    });
    assert_eq!(changed, 1, "exactly the key-position offset is rewritten");
    cases.push((
        "conditions one key off",
        off_by_one,
        AttentionDecline::LiveScoreGuard,
    ));

    // The key rows select by another length than the scores do.
    let base = live_attention_graph(false, true);
    let rows = base
        .eqns()
        .iter()
        .position(|eqn| {
            matches!(eqn.op, OpKind::Binary(B::Ge)) && base.aval(eqn.out).shape == [2, 1, 6, 1]
        })
        .expect("the key-row condition");
    let other_len = mutated(&base, |draft| {
        let other = draft
            .values_mut()
            .push(poot_graph_ir::graph::ValueMeta::new(
                TensorType::new(vec![2], poot_tensor::DType::I32),
                Storage::Const,
                Some("other_len".into()),
            ));
        draft.consts_mut().push(other);
        draft.inputs_mut().push(other);
        let view = draft
            .values_mut()
            .push(poot_graph_ir::graph::ValueMeta::new(
                TensorType::new(vec![2, 1, 1, 1], poot_tensor::DType::I32),
                Storage::Device,
                None,
            ));
        draft.eqns_mut().insert(
            rows,
            Eqn {
                op: OpKind::Reshape {
                    shape: vec![2, 1, 1, 1],
                },
                inputs: vec![Operand::Value(other)],
                out: view,
                layer: None,
            },
        );
        draft.eqns_mut()[rows + 1].inputs[0] = Operand::Value(view);
    });
    cases.push((
        "K/V select by another length",
        other_len,
        AttentionDecline::LiveKeyGuard,
    ));

    // A guard bounding the length by one more than the capacity, and no witness at all.
    let wrong_bound = mutated(&live_attention_graph(false, true), |draft| {
        for eqn in draft.eqns_mut().iter_mut() {
            if matches!(eqn.op, OpKind::Binary(B::GeU)) {
                eqn.inputs[1] = Operand::Lit(Scalar::I32(7));
            }
        }
    });
    cases.push((
        "length bounded by capacity + 1",
        wrong_bound,
        AttentionDecline::LiveLengthUnproved,
    ));

    cases.push((
        "length with no declared witness",
        live_attention_graph(false, false),
        AttentionDecline::LiveLengthUnproved,
    ));

    for (name, graph, expected) in cases {
        let reason = match_attention(
            &graph,
            &producer_map(&graph),
            &value_ranges(&graph),
            output_eqn(&graph),
        );
        assert_eq!(reason.err(), Some(expected), "{name}: wrong decline reason");
    }
}

/// Canonicalization retains unproved arithmetic and reports why; only F32 multiply-by-one is
/// removed from a contraction operand. The CPU bitwise corpus lives in numerical_legality.rs.
#[test]
fn canonicalize_preserves_unproved_spellings_and_removes_identity() {
    for scale in [0.25, 1.0] {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 4, 8])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![8, 5])).unwrap();
        let scaled = b
            .binary_scalar(BinOp::Mul, x, poot_graph_ir::Scalar::F32(scale))
            .unwrap();
        let output = b.matmul(scaled, w).unwrap();
        let graph = b.finish(output).unwrap();
        let (canonical, declines) = canonicalize_with_declines(&graph).unwrap();
        let root = canonical.eqns().last().unwrap();
        assert!(matches!(root.op, OpKind::MatMul));
        assert_eq!(root.out, graph.result());
        if scale == 1.0 {
            assert!(matches!(root.inputs[0], Operand::Value(id) if id == x.id()));
            assert!(declines.is_empty());
        } else {
            assert_eq!(
                format!("{:?}", canonical.eqns()),
                format!("{:?}", graph.eqns())
            );
            assert_eq!(
                declines,
                vec![NumericalRewriteDecline {
                    value: output.id(),
                    reason: NumericalRewriteReason::ScalarAcrossContraction,
                }]
            );
        }
        assert_eq!(
            format!("{:?}", canonicalize(&canonical).unwrap().eqns()),
            format!("{:?}", canonical.eqns())
        );
    }
    let b = Builder::for_tests();
    let e = b.constant("e", TensorType::f32(vec![3, 4])).unwrap();
    let denom = b.constant("denom", TensorType::f32(vec![3, 1])).unwrap();
    let recip = b.unary(UnOp::Recip, denom).unwrap();
    let mul = b.binary(BinOp::Mul, e, recip).unwrap();
    let graph = b.finish(mul).unwrap();
    let (canonical, declines) = canonicalize_with_declines(&graph).unwrap();
    assert_eq!(
        format!("{:?}", canonical.eqns()),
        format!("{:?}", graph.eqns())
    );
    assert_eq!(
        declines,
        vec![NumericalRewriteDecline {
            value: mul.id(),
            reason: NumericalRewriteReason::ReciprocalRounding,
        }]
    );
}

/// Card 536a/R467-005 scalar-placement near-miss: a scale carried as a runtime *value* (not a
/// literal) cannot be lifted into the fused op's `f32` parameter, so the matcher declines it with a
/// typed reason rather than dropping the scale.
#[test]
fn card536a_scale_as_a_runtime_value_declines() {
    let (hq, cap, d) = (4usize, 6usize, 16usize);
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    let scale = b.constant("scale", TensorType::f32(vec![1])).unwrap();
    let kt = b.transpose(k, vec![0, 1, 3, 2]).unwrap();
    let qk = b.matmul(q, kt).unwrap();
    let scores = b.binary(BinOp::Mul, scale, qk).unwrap();
    let scores = b.binary(BinOp::Add, scores, mask).unwrap();
    let last = b.aval(scores).unwrap().rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true).unwrap();
    let e = b
        .unary(UnOp::Exp, b.binary(BinOp::Sub, scores, m).unwrap())
        .unwrap();
    let denom = b.reduce(RedOp::Sum, e, last, true).unwrap();
    let p = b.binary(BinOp::Div, e, denom).unwrap();
    let graph_out = b.matmul(p, v).unwrap();
    let graph = b.finish(graph_out).unwrap();

    assert_eq!(
        match_attention(
            &graph,
            &producer_map(&graph),
            &value_ranges(&graph),
            output_eqn(&graph)
        )
        .err(),
        Some(AttentionDecline::ScoreProduct),
        "a runtime-valued scale is a typed decline, not a dropped parameter"
    );
}

/// Card 557 SC-003 (R-557-1), restated for the contraction epilogue: a traced bias linear is primitives
/// only, `matmul` then a broadcast `add` with no `ContractionEpilogue`, and the one contraction fuse rule
/// forms exactly one `ContractionEpilogue { head: MatMul, region: head + leaf }` over `(x, w, bias)`, with
/// the operands in the order every planner arm and the oracle read (the head's two operands, then the
/// region's extra inputs). The fused graph evaluates bit for bit like the traced one. Mutations: emit the
/// epilogue from `ops::linear` (the traced assertion goes red); put the bias before the head operands (the
/// operand assertion goes red).
#[test]
fn a_traced_bias_linear_is_primitive_and_fuses_into_one_contraction_epilogue() {
    use poot_eval::{EvalBudget, EvalOptions, Value, eval};
    use poot_graph_ir::OpClass;
    use poot_tensor::HostTensor;

    let (m, k, n) = (3usize, 5usize, 4usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let out = poot_graph_ir::ops::linear(&b, x, w, Some(bias)).unwrap();
    let traced = b.finish(out).unwrap();
    assert!(
        traced
            .eqns()
            .iter()
            .all(|eqn| eqn.op.class() == OpClass::Primitive),
        "a tracer emits primitives only: {:?}",
        traced.eqns()
    );

    let fused = fuse_contraction_epilogues(&traced, |_, _| true).unwrap();
    let epilogues: Vec<&Eqn> = fused
        .eqns()
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::ContractionEpilogue { .. }))
        .collect();
    assert_eq!(
        epilogues.len(),
        1,
        "one ContractionEpilogue: {:?}",
        fused.eqns()
    );
    assert!(
        matches!(
            epilogues[0].inputs.as_slice(),
            [Operand::Value(a), Operand::Value(b), Operand::Value(c)]
                if [*a, *b, *c] == [x.id(), w.id(), bias.id()]
        ),
        "ContractionEpilogue reads (activation, weight, bias): {:?}",
        epilogues[0].inputs
    );
    assert_eq!(
        fused.eqns().len(),
        1,
        "the matmul and the add are one equation"
    );
    assert_eq!(
        epilogues[0].out,
        traced.result(),
        "the add's output id is kept"
    );

    let data = |len: usize, seed: f32| -> Vec<f32> {
        (0..len).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
    };
    let inputs: HashMap<ValueId, Value> = [
        (x.id(), HostTensor::f32(vec![m, k], data(m * k, 1.0))),
        (w.id(), HostTensor::f32(vec![k, n], data(k * n, 2.0))),
        (bias.id(), HostTensor::f32(vec![n], data(n, 3.0))),
    ]
    .into_iter()
    .map(|(id, tensor)| (id, Value::from(tensor)))
    .collect();
    let run = |g: &Graph| -> Vec<u32> {
        eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("the graph evaluates")
            .output
            .into_host()
            .expect("a dense output")
            .as_f32()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    };
    assert_eq!(
        run(&fused),
        run(&traced),
        "the epilogue is the same arithmetic"
    );
}

/// The `ContractionEpilogue` equations of `g`.
fn epilogues_of(g: &Graph) -> Vec<&Eqn> {
    g.eqns()
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::ContractionEpilogue { .. }))
        .collect()
}

/// The oracle's output bits of `g` over `[m, k]`, `[k, n]` and `[n]` data and any further `[m, n]` inputs
/// the fixture declares, filled deterministically.
fn eval_bits(g: &Graph) -> Vec<u32> {
    use poot_eval::{EvalBudget, EvalOptions, Value, eval};
    use poot_tensor::HostTensor;

    let inputs: HashMap<ValueId, Value> = g
        .inputs()
        .iter()
        .map(|&id| {
            let shape = g.aval(id).shape.clone();
            let numel: usize = shape.iter().product();
            let data = (0..numel)
                .map(|i| ((i as f32 + id.index() as f32) * 0.37).sin())
                .collect();
            (id, Value::from(HostTensor::f32(shape, data)))
        })
        .collect();
    eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("the graph evaluates")
        .output
        .into_host()
        .expect("a dense output")
        .as_f32()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

/// Contraction epilogue near misses (the fuse rule's legality): a head with a second consumer, a reducing
/// consumer, a step that changes the shape, a pinned head output and a step over another dtype each stay
/// exactly as traced, and every fused fixture evaluates bit for bit like its unfused graph. Mutation: drop the
/// single-use check in `fuse_contraction_epilogues` (`single_use`): the two-consumer graph fuses and its second
/// consumer then reads the epilogue's result instead of the contraction's, so this row goes red.
#[test]
fn contraction_epilogue_near_misses_stay_unfused() {
    let (m, k, n) = (3usize, 5usize, 4usize);
    let linear = |b: &Builder| {
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
        (b.matmul(x, w).unwrap(), bias)
    };

    // The fixture that does fuse: head, bias, residual.
    let b = Builder::for_tests();
    let (head, bias) = linear(&b);
    let residual = b.constant("residual", TensorType::f32(vec![m, n])).unwrap();
    let out = b
        .binary(
            BinOp::Add,
            b.binary(BinOp::Add, head, bias).unwrap(),
            residual,
        )
        .unwrap();
    let fused_case = b.finish(out).unwrap();
    let fused = fuse_contraction_epilogues(&fused_case, |_, _| true).unwrap();
    assert_eq!(epilogues_of(&fused).len(), 1, "{:?}", fused.eqns());
    assert_eq!(fused.eqns().len(), 1, "head and both adds are one equation");
    assert_eq!(eval_bits(&fused), eval_bits(&fused_case), "fused fixture");

    // A head read twice: the add and a second consumer.
    let b = Builder::for_tests();
    let (head, bias) = linear(&b);
    let out = b
        .binary(BinOp::Mul, b.binary(BinOp::Add, head, bias).unwrap(), head)
        .unwrap();
    let two_consumers = b.finish(out).unwrap();
    // A reducing consumer.
    let b = Builder::for_tests();
    let (head, _) = linear(&b);
    let out = b.reduce(RedOp::Sum, head, 1, true).unwrap();
    let reducing = b.finish(out).unwrap();
    // A step whose output is larger than the head: a leaf of another shape broadcast against it.
    let b = Builder::for_tests();
    let (head, _) = linear(&b);
    let wide = b.constant("wide", TensorType::f32(vec![2, m, n])).unwrap();
    let out = b.binary(BinOp::Add, head, wide).unwrap();
    let shape_changing = b.finish(out).unwrap();
    // The head is the graph output, so a step beyond it cannot absorb it.
    let b = Builder::for_tests();
    let (head, _) = linear(&b);
    let pinned = b.finish(head).unwrap();
    for (name, graph) in [
        ("a head with two consumers", &two_consumers),
        ("a reducing consumer", &reducing),
        ("a shape-changing step", &shape_changing),
        ("a head that is the graph output", &pinned),
    ] {
        let after = fuse_contraction_epilogues(graph, |_, _| true).unwrap();
        assert!(
            epilogues_of(&after).is_empty(),
            "{name} must stay unfused: {:?}",
            after.eqns()
        );
        assert_eq!(after.eqns().len(), graph.eqns().len(), "{name}: untouched");
        assert_eq!(eval_bits(&after), eval_bits(graph), "{name}: same bits");
    }
}

/// An explicit `Broadcast` of a bias feeding the chain is read through: the epilogue's leaf is the `[N]` bias
/// itself, not an expanded `[M, N]` copy, and the broadcast equation is gone. A broadcast another equation also
/// reads stays (its expanded value is still needed). Mutation: stop folding the broadcast; the fused equation
/// then holds the broadcast's output as its leaf and the broadcast equation remains, so the leaf and equation
/// count assertions go red.
#[test]
fn a_single_use_bias_broadcast_folds_into_the_epilogue_leaf() {
    let (m, k, n) = (3usize, 5usize, 4usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let expanded = b.broadcast(bias, vec![m, n]).unwrap();
    let out = b
        .binary(BinOp::Add, b.matmul(x, w).unwrap(), expanded)
        .unwrap();
    let traced = b.finish(out).unwrap();
    let fused = fuse_contraction_epilogues(&traced, |_, _| true).unwrap();
    let epilogues = epilogues_of(&fused);
    assert_eq!(epilogues.len(), 1, "{:?}", fused.eqns());
    assert_eq!(
        fused.eqns().len(),
        1,
        "the broadcast folds away: {:?}",
        fused.eqns()
    );
    assert!(
        matches!(
            epilogues[0].inputs.as_slice(),
            [Operand::Value(a), Operand::Value(bw), Operand::Value(leaf)]
                if [*a, *bw, *leaf] == [x.id(), w.id(), bias.id()]
        ),
        "the leaf is the [N] bias: {:?}",
        epilogues[0].inputs
    );
    assert_eq!(eval_bits(&fused), eval_bits(&traced));

    // The same broadcast read by a second equation is not folded away.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let expanded = b.broadcast(bias, vec![m, n]).unwrap();
    let out = b
        .binary(
            BinOp::Mul,
            b.binary(BinOp::Add, b.matmul(x, w).unwrap(), expanded)
                .unwrap(),
            expanded,
        )
        .unwrap();
    let shared = b.finish(out).unwrap();
    let after = fuse_contraction_epilogues(&shared, |_, _| true).unwrap();
    assert!(
        after
            .eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Broadcast { .. })),
        "a shared broadcast stays: {:?}",
        after.eqns()
    );
    assert_eq!(eval_bits(&after), eval_bits(&shared));
}

/// POOT-1033: a float `Select` is a select, not float arithmetic: it moves a branch's bits, so a graph of
/// nothing else is bit-exact on a device (tier 1). Mutation: put `Select` back with `Binary`/`Unary` in
/// `implied_op_numerics`; the graph reads `Unfused` and this row goes red.
#[test]
fn float_select_is_exact_not_float_arithmetic() {
    let b = Builder::for_tests();
    let cond = b
        .constant("cond", TensorType::new(vec![4], poot_tensor::DType::I32))
        .unwrap();
    let if_true = b.constant("t", TensorType::f32(vec![4])).unwrap();
    let if_false = b.constant("f", TensorType::f32(vec![4])).unwrap();
    let selected = b.select(cond, if_true, if_false).unwrap();
    let graph = b.finish(selected).unwrap();
    assert!(matches!(graph.eqns(), [eqn] if matches!(eqn.op, OpKind::Select)));
    assert_eq!(implied_numerics(&graph), NumericsProperty::Exact);
    assert_eq!(implied_numerics(&graph).tier2_class(), Tier2Class::BitExact);
}
