use super::*;
use poot_graph_ir::Builder;
use poot_graph_ir::LayerIndex;
use poot_graph_ir::analysis::{
    NumericsError, NumericsProperty, PASS_DECLARATIONS, PassDeclaration, Tier2Class,
    dispatch_count, implied_numerics, peak_transient_bytes, verify_pass_numerics,
};
use poot_graph_ir::op::{BinOp, RedOp};
use poot_graph_ir::ops::{attention_masked, attention_masked_softcap};
use poot_graph_ir::types::TensorType;
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;

/// The graph-level packed-dequant preparation these tests exercise: validate, eliminate dead
/// oracle-only decoding, recognize dense contractions, then reject every remaining escape. The planner's
/// production entry adds block-float claims on top of the same passes.
fn prepare_packed_dequant_production(g: &Graph) -> Result<Graph, PackedDequantProductionError> {
    g.validate()?;
    reject_preexisting_packed_contractions(g)?;
    let graph = dce_with_roots(g, &[]);
    let graph = recognize_packed_contractions(&graph);
    reject_packed_dequant_escapes(&graph)?;
    graph.validate()?;
    Ok(graph)
}

fn packed_graph() -> Result<Graph, poot_graph_ir::BuilderAppendError> {
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
    let b = Builder::new();
    let activation = b.constant("activation", TensorType::f32(vec![2, 35]));
    let output =
        poot_graph_ir::ops::packed_linear(&b, activation, "layer", descriptor, None, None)?;
    Ok(b.finish(output))
}

fn packed_graph_with_extra_consumer() -> Result<Graph, poot_graph_ir::BuilderAppendError> {
    let mut graph = packed_graph()?;
    let dequant = graph.eqns[0].out;
    let first_output = graph.output;
    let transpose2 = graph.values.len();
    let transpose_meta = graph.values[graph.eqns[1].out].clone();
    graph.values.push(transpose_meta);
    graph.eqns.push(Eqn {
        op: OpKind::Transpose { perm: vec![1, 0] },
        inputs: vec![Operand::Value(dequant)],
        out: transpose2,
        layer: None,
    });
    let activation = match graph.eqns[2].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    let second_output = graph.values.len();
    graph.values.push(graph.values[first_output].clone());
    graph.eqns.push(Eqn {
        op: OpKind::MatMul,
        inputs: vec![Operand::Value(activation), Operand::Value(transpose2)],
        out: second_output,
        layer: None,
    });
    let combined = graph.values.len();
    graph.values.push(graph.values[first_output].clone());
    graph.eqns.push(Eqn {
        op: OpKind::Binary(BinOp::Add),
        inputs: vec![Operand::Value(first_output), Operand::Value(second_output)],
        out: combined,
        layer: None,
    });
    graph.output = combined;
    Ok(graph)
}

fn packed_graph_with_extra_carrier_consumer() -> Result<Graph, poot_graph_ir::BuilderAppendError> {
    let mut graph = packed_graph()?;
    let weight = match graph.eqns[0].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    let original_output = graph.output;
    let cast = graph.values.len();
    graph.values.push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![3, 18]),
        Storage::Device,
        None,
    ));
    graph.eqns.push(Eqn {
        op: OpKind::Cast { to: DType::F32 },
        inputs: vec![Operand::Value(weight)],
        out: cast,
        layer: None,
    });
    let reduced = graph.values.len();
    graph.values.push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![3]),
        Storage::Device,
        None,
    ));
    graph.eqns.push(Eqn {
        op: OpKind::Reduce {
            op: RedOp::Sum,
            axis: 1,
            keepdim: false,
        },
        inputs: vec![Operand::Value(cast)],
        out: reduced,
        layer: None,
    });
    let broadcast = graph.values.len();
    graph.values.push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![2, 3]),
        Storage::Device,
        None,
    ));
    graph.eqns.push(Eqn {
        op: OpKind::Broadcast { shape: vec![2, 3] },
        inputs: vec![Operand::Value(reduced)],
        out: broadcast,
        layer: None,
    });
    let combined = graph.values.len();
    graph.values.push(poot_graph_ir::ValueMeta::new(
        TensorType::f32(vec![2, 3]),
        Storage::Device,
        None,
    ));
    graph.eqns.push(Eqn {
        op: OpKind::Binary(BinOp::Add),
        inputs: vec![Operand::Value(original_output), Operand::Value(broadcast)],
        out: combined,
        layer: None,
    });
    graph.output = combined;
    Ok(graph)
}

#[test]
fn packed_dequant_fusion_matches_only_exact_linear() -> Result<(), poot_graph_ir::BuilderAppendError>
{
    let graph = packed_graph()?;
    let fused = prepare_packed_dequant_production(&graph).unwrap();
    assert_eq!(fused.eqns.len(), 1);
    assert!(matches!(fused.eqns[0].op, OpKind::PackedContraction { .. }));

    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
    let builder = Builder::new();
    let activation = builder.constant("alias.activation", TensorType::f32(vec![2, 35]));
    let output = poot_graph_ir::ops::packed_linear(
        &builder,
        activation,
        "alias",
        descriptor,
        Some(vec![35, 3]),
        None,
    )?;
    let alias_graph = builder.finish(output);
    let alias_fused = prepare_packed_dequant_production(&alias_graph).unwrap();
    assert_eq!(alias_fused.eqns.len(), 1);
    assert!(matches!(
        alias_fused.eqns[0].op,
        OpKind::PackedContraction { .. }
    ));
    assert_eq!(alias_fused.aval(alias_fused.output).shape, vec![2, 3]);

    let mut wrong_perm = graph.clone();
    let transpose = wrong_perm
        .eqns
        .iter_mut()
        .find(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
        .unwrap();
    transpose.op = OpKind::Transpose { perm: vec![0, 1] };
    wrong_perm.values[transpose.out].aval = TensorType::f32(vec![3, 35]);
    let matmul = wrong_perm
        .eqns
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::MatMul))
        .unwrap();
    let Operand::Value(activation) = matmul.inputs[0] else {
        unreachable!()
    };
    wrong_perm.values[activation].aval = TensorType::f32(vec![2, 3]);
    wrong_perm.values[matmul.out].aval = TensorType::f32(vec![2, 35]);
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
-> Result<(), poot_graph_ir::BuilderAppendError> {
    // out=8, k=35, blocks=4 -> block_out=2. Card 385: `ops::packed_block_diagonal_linear`'s
    // Reshape-before-Transpose chain must fold like the canonical chain; an unrecognized packed
    // dequant is a compile-time escape rejection, not a silently slow fallback.
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
    let builder = Builder::new();
    let activation = builder.constant("blocked.activation", TensorType::f32(vec![4, 5, 35]));
    let output = poot_graph_ir::ops::packed_block_diagonal_linear(
        &builder, activation, "blocked", descriptor, 4, None,
    )?;
    let graph = builder.finish(output);
    let fused = prepare_packed_dequant_production(&graph).unwrap();
    assert_eq!(fused.eqns.len(), 1);
    assert!(matches!(
        fused.eqns[0].op,
        OpKind::PackedContraction { blocks: 4, .. }
    ));
    assert_eq!(fused.aval(fused.output).shape, vec![4, 5, 2]);
    Ok(())
}

fn packed_block_diagonal_graph() -> Result<Graph, poot_graph_ir::BuilderAppendError> {
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
    let b = Builder::new();
    let activation = b.constant("blocked.activation", TensorType::f32(vec![4, 5, 35]));
    let output = poot_graph_ir::ops::packed_block_diagonal_linear(
        &b, activation, "blocked", descriptor, 4, None,
    )?;
    Ok(b.finish(output))
}

/// A second full branch off the same `PackedDequant` output (reshape/transpose/matmul cloned from
/// the original, then summed into the graph's output). This gives the dequant a second live consumer
/// without hand-computing any new tensor shape.
fn packed_block_diagonal_graph_with_extra_consumer()
-> Result<Graph, poot_graph_ir::BuilderAppendError> {
    let mut graph = packed_block_diagonal_graph()?;
    let dequant = graph.eqns[0].out;
    let first_output = graph.output;

    let reshape2 = graph.values.len();
    graph.values.push(graph.values[graph.eqns[1].out].clone());
    graph.eqns.push(Eqn {
        op: graph.eqns[1].op.clone(),
        inputs: vec![Operand::Value(dequant)],
        out: reshape2,
        layer: None,
    });
    let transpose2 = graph.values.len();
    graph.values.push(graph.values[graph.eqns[2].out].clone());
    graph.eqns.push(Eqn {
        op: graph.eqns[2].op.clone(),
        inputs: vec![Operand::Value(reshape2)],
        out: transpose2,
        layer: None,
    });
    let activation = match graph.eqns[3].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    let second_output = graph.values.len();
    graph.values.push(graph.values[first_output].clone());
    graph.eqns.push(Eqn {
        op: OpKind::MatMul,
        inputs: vec![Operand::Value(activation), Operand::Value(transpose2)],
        out: second_output,
        layer: None,
    });
    let combined = graph.values.len();
    graph.values.push(graph.values[first_output].clone());
    graph.eqns.push(Eqn {
        op: OpKind::Binary(BinOp::Add),
        inputs: vec![Operand::Value(first_output), Operand::Value(second_output)],
        out: combined,
        layer: None,
    });
    graph.output = combined;
    Ok(graph)
}

#[test]
fn packed_block_diagonal_production_rejects_extra_dequant_consumer()
-> Result<(), poot_graph_ir::BuilderAppendError> {
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
-> Result<(), poot_graph_ir::BuilderAppendError> {
    let graph = packed_graph_with_extra_carrier_consumer()?;
    graph.validate().unwrap();
    assert!(matches!(
        prepare_packed_dequant_production(&graph),
        Err(PackedDequantProductionError::CarrierConsumerCount { consumers: 2, .. })
    ));
    Ok(())
}

#[test]
fn packed_recognition_keeps_validation_observed_members()
-> Result<(), poot_graph_ir::BuilderAppendError> {
    let base = packed_graph()?;
    let dequant = base.eqns[0].out;
    let graph = base.with_validations(vec![poot_graph_ir::graph::ValidationOutput {
        id: poot_graph_ir::graph::ValidationId(7),
        name: "observes-dequant".into(),
        value: dequant,
    }]);
    graph.validate().unwrap();

    let recognized = recognize_packed_contractions(&graph);
    assert_eq!(recognized.validation_outputs(), graph.validation_outputs());
    assert!(
        recognized
            .eqns
            .iter()
            .any(|eqn| eqn.out == dequant && matches!(eqn.op, OpKind::PackedDequant { .. })),
        "a witness-observed dequant must stay unrecognized"
    );
    assert!(
        !recognized
            .eqns
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
fn production_rejects_packed_dequant_escape_table() -> Result<(), poot_graph_ir::BuilderAppendError>
{
    let base = packed_graph()?;
    let dequant = base.eqns[0].out;
    let transpose = base.eqns[1].out;
    let weight = match base.eqns[0].inputs[0] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    type Reject = fn(&PackedDequantProductionError) -> bool;
    let mut cases: Vec<(&str, Graph, Reject)> = Vec::new();

    let mut output = base.clone();
    output.output = dequant;
    cases.push(("graph output", output, |error| {
        matches!(error, PackedDequantProductionError::GraphOutput { .. })
    }));

    let mut state_carrier = base.clone();
    // R-644-1: a State value must carry a role, or validate()'s own MissingStateRole check fires
    // before this test's target (CarrierStorage) gets a chance to. `ValueMeta::new_state` is the only
    // constructor that can produce one (`state_role` is private outside `poot-graph-ir`).
    state_carrier.values[weight] = poot_graph_ir::ValueMeta::new_state(
        state_carrier.values[weight].aval.clone(),
        state_carrier.values[weight].name.clone(),
        poot_graph_ir::StateRole::Recurrent,
    );
    cases.push(("state carrier", state_carrier, |error| {
        matches!(error, PackedDequantProductionError::CarrierStorage { .. })
    }));

    let mut direct_storage = prepare_packed_dequant_production(&base).unwrap();
    let direct_weight = match direct_storage.eqns[0].inputs[1] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!(),
    };
    direct_storage.values[direct_weight] = poot_graph_ir::ValueMeta::new_state(
        direct_storage.values[direct_weight].aval.clone(),
        direct_storage.values[direct_weight].name.clone(),
        poot_graph_ir::StateRole::Recurrent,
    );
    cases.push(("direct candidate state carrier", direct_storage, |error| {
        matches!(
            error,
            PackedDequantProductionError::PreexistingContraction { .. }
        )
    }));

    let mut direct_identity = prepare_packed_dequant_production(&base).unwrap();
    direct_identity.values[direct_weight].name = Some("layer.not_a_packed_weight".into());
    cases.push((
        "direct candidate carrier identity",
        direct_identity,
        |error| {
            matches!(
                error,
                PackedDequantProductionError::PreexistingContraction { .. }
            )
        },
    ));

    cases.push(("live-out", packed_graph_with_extra_consumer()?, |error| {
        matches!(error, PackedDequantProductionError::ConsumerCount { .. })
    }));

    for (label, state_out) in [("dequant state", dequant), ("transpose state", transpose)] {
        let mut graph = base.clone();
        let state_in = graph.values.len();
        graph.values.push(poot_graph_ir::ValueMeta::new_state(
            graph.aval(state_out).clone(),
            Some(format!("{label}.state")),
            poot_graph_ir::StateRole::Recurrent,
        ));
        graph.inputs.push(state_in);
        graph.consts.push(state_in);
        graph.state.push((state_in, state_out));
        cases.push((label, graph, |error| {
            matches!(error, PackedDequantProductionError::StateOutput { .. })
        }));
    }

    let mut candidate_state = base.clone();
    let state_in = candidate_state.values.len();
    candidate_state
        .values
        .push(poot_graph_ir::ValueMeta::new_state(
            candidate_state.aval(candidate_state.output).clone(),
            Some("candidate.state".into()),
            poot_graph_ir::StateRole::Recurrent,
        ));
    candidate_state.inputs.push(state_in);
    candidate_state.consts.push(state_in);
    candidate_state
        .state
        .push((state_in, candidate_state.output));
    cases.push(("candidate state", candidate_state, |error| {
        matches!(error, PackedDequantProductionError::StateOutput { .. })
    }));

    let mut wrong = base.clone();
    let transpose_eqn = &mut wrong.eqns[1];
    transpose_eqn.op = OpKind::Transpose { perm: vec![0, 1] };
    wrong.values[transpose_eqn.out].aval = TensorType::f32(vec![3, 35]);
    let matmul = &wrong.eqns[2];
    let Operand::Value(activation) = matmul.inputs[0] else {
        unreachable!()
    };
    wrong.values[activation].aval = TensorType::f32(vec![2, 3]);
    wrong.values[matmul.out].aval = TensorType::f32(vec![2, 35]);
    cases.push(("wrong permutation", wrong, |error| {
        matches!(
            error,
            PackedDequantProductionError::TransposePermutation { .. }
        )
    }));

    let mut unmatched = base.clone();
    unmatched.eqns.remove(1);
    let matmul = unmatched.eqns.remove(1);
    unmatched.output = dequant;
    let output = unmatched.values.len();
    unmatched.values.push(unmatched.values[dequant].clone());
    unmatched.eqns.push(Eqn {
        op: OpKind::Reshape { shape: vec![3, 35] },
        inputs: vec![Operand::Value(dequant)],
        out: output,
        layer: None,
    });
    unmatched.output = output;
    let _ = matmul;
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
    let builder = Builder::new();
    let output = builder.constant("ordinary.output", TensorType::f32(vec![2, 3]));
    // `Builder::packed_component_constants` is `#[cfg(test)]` to `poot-graph-ir` itself (invisible to
    // this crate's tests); inlined here from its own small body.
    let sources: Vec<_> = poot_graph_ir::packed_source_constants("dead", descriptor)
        .into_iter()
        .map(|(name, tensor_type)| builder.constant(name.as_str(), tensor_type))
        .collect();
    let _dead = builder.packed_dequant(&sources, descriptor);
    let dead_graph = builder.finish(output);
    let prepared = prepare_packed_dequant_production(&dead_graph).unwrap();
    assert!(prepared.eqns.is_empty(), "DCE removes oracle-only decode");
    Ok(())
}

#[test]
fn flash_attention_keeps_distinct_value_head_width_decomposed() {
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 3]));
    let k = b.constant("k", TensorType::f32(vec![1, 2, 5, 3]));
    let v = b.constant("v", TensorType::f32(vec![1, 2, 5, 2]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, 5]));
    let out = attention_masked(&b, q, k, v, 1, 1.0, mask);
    let g = b.finish(out);
    let optimized = flash_attention_capped(&g, None);

    assert_eq!(optimized.aval(optimized.output).shape, vec![1, 2, 1, 2]);
    assert!(
        optimized.eqns.iter().all(|eqn| !matches!(
            eqn.op,
            OpKind::FlashAttentionDecode { .. } | OpKind::FlashAttentionPrefill { .. }
        )),
        "flash kernels have one shared Q/K/V head width"
    );
}

fn is_flash(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::FlashAttentionDecode { .. } | OpKind::FlashAttentionPrefill { .. }
    )
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
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(qs.to_vec()));
        let k = b.constant("k", TensorType::f32(kvs.to_vec()));
        let v = b.constant("v", TensorType::f32(kvs.to_vec()));
        let mask = b.constant("mask", TensorType::f32(ms.to_vec()));
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask);
        let optimized = flash_attention_capped(&b.finish(out), None);
        optimized
            .validate()
            .unwrap_or_else(|e| panic!("{name}: optimized graph invalid: {e}"));
        let flash: Vec<String> = optimized
            .eqns
            .iter()
            .filter(|e| is_flash(&e.op))
            .map(|e| e.op.name())
            .collect();
        assert!(flash.is_empty(), "{name}: fused into {flash:?}");
    }
}

/// A graph whose last equation is the fused flash `op` over dense MHA operands (`Hq = 4`, `n_rep = 1`),
/// with the ids of its `k`, `v` and `mask` inputs. Decode forms use batch 2 and `cap = 5`; prefill forms
/// use batch 1 and `L = 3`.
fn flash_graph(op: OpKind) -> (Graph, [ValueId; 3]) {
    let decode = matches!(op, OpKind::FlashAttentionDecode { .. });
    let (bsz, hq, m, t, d) = if decode {
        (2, 4, 1, 5, 8)
    } else {
        (1, 4, 3, 3, 8)
    };
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![bsz, hq, m, d]));
    let k = b.constant("k", TensorType::f32(vec![bsz, hq, t, d]));
    let v = b.constant("v", TensorType::f32(vec![bsz, hq, t, d]));
    let mask = b.constant("mask", TensorType::f32(vec![bsz, 1, m, t]));
    let out = if decode {
        poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.25, mask)
    } else {
        poot_graph_ir::ops::attention_prefill(&b, q, k, v, 1, 0.25, mask)
    };
    // A tracer states attention as primitives; the flash op is the matcher's.
    let mut g = dce(&flash_attention_capped(&b.finish(out), None));
    let flash = g.eqns.last_mut().expect("the flash equation");
    assert!(
        matches!(
            flash.op,
            OpKind::FlashAttentionDecode { .. } | OpKind::FlashAttentionPrefill { .. }
        ),
        "the matcher forms the flash op: {:?}",
        flash.op
    );
    flash.op = op;
    (g, [k.id, v.id, mask.id])
}

/// R467-001: `Graph::validate` rejects a fused flash op whose K/V or mask the op would index past, from
/// any source (not just the matcher), with the typed `FlashAttentionOperand` error.
#[test]
fn validate_rejects_flash_ops_with_inconsistent_kv_or_mask() {
    let ops = [
        OpKind::FlashAttentionDecode {
            n_rep: 1,
            scale: 0.25,
        },
        OpKind::FlashAttentionPrefill {
            n_rep: 1,
            scale: 0.25,
            softcap: None,
        },
    ];
    for op in ops {
        let name = op.name();
        let (dense, [k, v, mask]) = flash_graph(op);
        dense
            .validate()
            .unwrap_or_else(|e| panic!("{name}: dense operands must validate: {e}"));
        // (operand id, rank-4 axis to shrink to 1, operand name): a broadcast K head axis, a broadcast V
        // batch/row axis, and a mask whose head axis is neither 1 nor Hq.
        for (id, axis, size, operand) in [(k, 1, 1, "k"), (v, 0, 1, "v"), (mask, 1, 2, "mask")] {
            let mut bad = dense.clone();
            let expected = bad.values[id].aval.shape.clone();
            let mut actual = expected.clone();
            actual[axis] = size;
            if actual == expected {
                // Prefill is batch 1, so shrinking V's batch axis changes nothing: shrink its rows.
                actual[2] = 1;
            }
            bad.values[id].aval.shape = actual.clone();
            match bad.validate() {
                Err(GraphValidationError::EquationInference {
                    source:
                        poot_graph_ir::error::ShapeError::FlashAttentionOperand {
                            operand: got,
                            expected: want,
                            actual: seen,
                        },
                    ..
                }) => {
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
    let b = Builder::new();
    let last = x_shape.len() - 1;
    let d = x_shape[last];
    let x = b.constant("x", TensorType::f32(x_shape));
    let cos = b.constant("cos", TensorType::f32(table_shape.clone()));
    let sin = b.constant("sin", TensorType::f32(table_shape));
    let x1 = b.slice(x, last, 0, d / 2);
    let x2 = b.slice(x, last, d / 2, d);
    let neg_x2 = b.unary(UnOp::Neg, x2);
    let rotate_half = b.concat(last, &[neg_x2, x1]);
    let xc = b.binary(BinOp::Mul, x, cos);
    let rs = b.binary(BinOp::Mul, rotate_half, sin);
    let out = b.binary(BinOp::Add, xc, rs);
    b.finish(out)
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
        let fused = rope_fusion(&g);
        fused.validate().unwrap_or_else(|e| {
            panic!("x {x:?}, cos {table:?}: rope_fusion left an ill-typed graph: {e}")
        });
        let got = fused
            .eqns
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4]));
    let _dead = b.binary(BinOp::Add, x, x); // never used
    let out = b.binary(BinOp::Mul, x, x);
    let g = b.finish(out);
    let d = dce(&g);
    d.validate().expect("dce graph valid");
    assert!(d.eqns.len() < g.eqns.len(), "dce should drop the dead add");
    assert_eq!(d.aval(d.output).shape, vec![4]);
}

#[test]
fn peak_transient_bytes_counts_the_live_working_set() {
    // a linear chain x -> a -> b -> c (output), each [N] f32. At any step at most two intermediates are
    // live (the current output + the one feeding it), so the peak is 2*N*4 - NOT the sum of all four. The
    // const input x is persistent, not transient, so it is excluded.
    let n = 100usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let a = b.unary(poot_graph_ir::op::UnOp::Neg, x);
    let bb = b.unary(poot_graph_ir::op::UnOp::Neg, a);
    let c = b.unary(poot_graph_ir::op::UnOp::Neg, bb);
    let g = b.finish(c);
    // a born (live={a}=N), b born (live={a,b}=2N) then a freed, c born (live={b,c}=2N) then b freed.
    assert_eq!(peak_transient_bytes(&g), 2 * n * 4);
}

#[test]
fn dispatch_count_excludes_reshape_aliases() {
    // reshape is a buffer-aliasing view (no dispatch); the binary + the transpose are real dispatches.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2, 3]));
    let r = b.reshape(x, vec![3, 2]); // alias - not a dispatch
    let t = b.transpose(r, vec![1, 0]); // a real kernel
    let out = b.binary(BinOp::Add, t, t); // a real kernel
    let g = b.finish(out);
    assert_eq!(g.eqns.len(), 3, "reshape + transpose + add");
    assert_eq!(dispatch_count(&g), 2, "reshape aliases, so 2 dispatches");
}

#[test]
fn cse_folds_duplicate_eqns_and_stays_valid() {
    // two structurally-identical adds collapse to one; the mul rewires to the canonical value.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4]));
    let a1 = b.binary(BinOp::Add, x, x);
    let a2 = b.binary(BinOp::Add, x, x); // a duplicate of a1
    let out = b.binary(BinOp::Mul, a1, a2);
    let g = b.finish(out);
    let c = cse(&g);
    c.validate().expect("cse graph should validate");
    assert!(
        c.eqns.len() < g.eqns.len(),
        "cse should drop the duplicate add: {} -> {}",
        g.eqns.len(),
        c.eqns.len()
    );
    assert_eq!(c.aval(c.output).shape, g.aval(g.output).shape);
}

#[test]
fn cse_is_idempotent() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4]));
    let a1 = b.binary(BinOp::Add, x, x);
    let a2 = b.binary(BinOp::Add, x, x);
    let out = b.binary(BinOp::Mul, a1, a2);
    let g = b.finish(out);
    let c1 = cse(&g);
    let c2 = cse(&c1);
    assert_eq!(c1.eqns.len(), c2.eqns.len());
}

fn f(b: &Builder, shape: &[usize]) -> poot_graph_ir::builder::Traced {
    b.constant("x", TensorType::f32(shape.to_vec()))
}

#[test]
fn fuse_collapses_pointwise_chain() {
    // out = (a + b) * c, all [1,1,8] -> two eqns (add, mul) collapse to one Fused eqn over 3 leaves.
    let b = Builder::new();
    let (a, c, d) = (f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]));
    let t = b.binary(BinOp::Add, a, c);
    let out = b.binary(BinOp::Mul, t, d);
    let g = b.finish(out);
    assert_eq!(g.eqns.len(), 2);
    let fg = fuse(&g);
    fg.validate().expect("fused graph valid");
    assert_eq!(fg.eqns.len(), 1, "the add+mul chain should fuse to one eqn");
    match &fg.eqns[0].op {
        OpKind::Fused(r) => {
            assert_eq!(r.n_inputs, 3); // a, c, d
            assert_eq!(r.steps.len(), 2);
            assert_eq!(fg.eqns[0].inputs.len(), 3);
        }
        other => panic!("expected a Fused eqn, got {}", other.name()),
    }
    assert_eq!(fg.aval(fg.output).shape, vec![1, 1, 8]);
}

#[test]
fn fuse_fuses_a_multi_use_value_read_only_inside_the_region() {
    // t = a + b is consumed by TWO eqns that both end up in the one region (the diamond u, v -> out), so
    // t never escapes and fuses with them: 4 eqns -> 1 (Card 630's diamond rule; before it, t stayed a
    // standalone add). A value that is also read outside the region stays materialized: see
    // `fuse::tests::a_producer_read_outside_the_region_does_not_fuse`.
    let b = Builder::new();
    let (a, bb, c, d) = (
        f(&b, &[1, 4]),
        f(&b, &[1, 4]),
        f(&b, &[1, 4]),
        f(&b, &[1, 4]),
    );
    let t = b.binary(BinOp::Add, a, bb);
    let u = b.binary(BinOp::Mul, t, c);
    let v = b.binary(BinOp::Sub, t, d);
    let out = b.binary(BinOp::Add, u, v);
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("fused graph valid");
    assert_eq!(fg.eqns.len(), 1, "t joins the region that reads it twice");
    assert!(matches!(&fg.eqns[0].op, OpKind::Fused(r) if r.steps.len() == 4));
}

#[test]
fn fuse_does_not_wrap_a_singleton() {
    // a lone pointwise eqn has no fusable single-use producer: it stays a plain Binary, not Fused.
    let b = Builder::new();
    let (a, c) = (f(&b, &[3]), f(&b, &[3]));
    let out = b.binary(BinOp::Add, a, c);
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("valid");
    assert_eq!(fg.eqns.len(), 1);
    assert!(matches!(fg.eqns[0].op, OpKind::Binary(_)));
}

#[test]
fn fuse_collapses_exact_i32_pointwise_chain() {
    let b = Builder::new();
    let ty = TensorType::new(vec![4], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone());
    let y = b.constant("y", ty);
    let sum = b.binary(BinOp::Add, x, y);
    let out = b.binary_scalar(BinOp::GeU, sum, poot_graph_ir::types::Scalar::I32(-1));
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("fused exact-I32 graph remains valid");
    assert_eq!(fg.eqns.len(), 1, "same-shape I32 Add+GeU should fuse");
    match &fg.eqns[0].op {
        OpKind::Fused(region) => assert_eq!(region.steps.len(), 2),
        other => panic!("expected Fused, got {}", other.name()),
    }
}

#[test]
fn fuse_i32_closed_dag_absorbs_non_escaping_multi_use() {
    let b = Builder::new();
    let ty = TensorType::new(vec![4], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone());
    let y = b.constant("y", ty.clone());
    let z = b.constant("z", ty);
    let t = b.binary(BinOp::And, x, y);
    let u = b.binary(BinOp::Or, t, z);
    let v = b.unary(UnOp::Not, t);
    let out = b.select(u, v, t);
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("fused I32 DAG remains valid");
    assert_eq!(fg.eqns.len(), 1);
    match &fg.eqns[0].op {
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

#[test]
fn fuse_i32_pack_concat_keeps_shared_operands_in_one_region() {
    let b = Builder::new();
    let ty = TensorType::new(vec![1], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone());
    let y = b.constant("y", ty);
    let shared = b.binary(BinOp::And, x, y);
    let low = b.binary(BinOp::Or, shared, x);
    let high = b.binary(BinOp::Xor, shared, y);
    let packed = b.concat(0, &[low, high]);
    let g = b.finish(packed);
    let fg = fuse(&g);
    fg.validate().expect("packed I32 concat remains valid");
    assert_eq!(fg.eqns.len(), 1);
    match &fg.eqns[0].op {
        OpKind::Fused(region) => {
            assert_eq!(region.pack.len(), 2);
            assert_eq!(region.steps.len(), 3);
        }
        other => panic!("expected packed Fused, got {}", other.name()),
    }
}

#[test]
fn fuse_i32_pack_concat_absorbs_unit_last_axis_unpack_slices() {
    let b = Builder::new();
    let ty = TensorType::new(vec![1], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone());
    let y = b.constant("y", ty);
    let low = b.binary(BinOp::Or, x, y);
    let high = b.binary(BinOp::Xor, x, y);
    let packed = b.concat(0, &[low, high]);
    let left = b.slice(packed, 0, 0, 1);
    let right = b.slice(packed, 0, 1, 2);
    let masked = b.binary(BinOp::And, left, right);
    let out = b.binary(BinOp::Or, masked, left);
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("packed unpack fusion remains valid");
    assert!(
        !fg.eqns
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Slice { .. })),
        "unit last-axis unpack slices must be absorbed"
    );
    let fused = fg
        .eqns
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::Fused(_)))
        .count();
    assert_eq!(fused, 2, "one packed concat region and one unpack consumer");
    match &fg.eqns[1].op {
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
    let b = Builder::new();
    let ty = TensorType::new(vec![1], poot_tensor::DType::I32);
    let x = b.constant("x", ty.clone());
    let y = b.constant("y", ty);
    let low = b.binary(BinOp::Or, x, y);
    let high = b.binary(BinOp::Xor, x, y);
    let packed = b.concat(0, &[low, high]);
    let left = b.slice(packed, 0, 0, 1);
    let right = b.slice(packed, 0, 1, 2);
    let masked = b.binary(BinOp::And, left, right);
    let _keep = b.binary(BinOp::Or, masked, left);
    let g = b.finish(left);
    let fg = fuse(&g);
    fg.validate()
        .expect("pinned unpack slice must remain a produced graph output");
    assert_eq!(fg.output, left.id);
    assert!(
        fg.eqns.iter().any(|eqn| eqn.out == left.id),
        "pinned unit last-axis slice must not be absorbed"
    );
}

#[test]
fn fuse_does_not_cross_a_matmul_boundary() {
    // add -> matmul -> add: the matmul is a forced boundary, so the two adds do not fuse together.
    let b = Builder::new();
    let (a, c) = (f(&b, &[1, 4]), f(&b, &[1, 4]));
    let w = b.constant("w", TensorType::f32(vec![4, 4]));
    let s = b.binary(BinOp::Add, a, c); // [1,4]
    let m = b.matmul(s, w); // [1,4], a boundary
    let e = f(&b, &[1, 4]);
    let out = b.binary(BinOp::Add, m, e);
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("valid");
    // matmul stays; neither add fuses with anything (each is a singleton across the boundary).
    assert!(fg.eqns.iter().any(|e| matches!(e.op, OpKind::MatMul)));
    assert!(!fg.eqns.iter().any(|e| matches!(e.op, OpKind::Fused(_))));
}

#[test]
fn fuse_forms_a_rmsnorm_row_region() {
    // the whole RMSNorm decomposition (sq -> sum -> *1/n -> +eps -> sqrt -> x/den -> *w) fuses to ONE
    // reduction-rooted FusedRow eqn.
    use poot_graph_ir::ops::rmsnorm;
    let b = Builder::new();
    let n = 8usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let out = rmsnorm(&b, x, w, 1e-6);
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("fused graph valid");
    let rows = fg
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FusedRow(_)))
        .count();
    assert_eq!(rows, 1, "rmsnorm fuses to one FusedRow eqn");
    assert!(
        fg.eqns.len() < g.eqns.len(),
        "fusion drops eqns: {} -> {}",
        g.eqns.len(),
        fg.eqns.len()
    );
    if let OpKind::FusedRow(r) = &fg
        .eqns
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
    let has_row = |g: &Graph| g.eqns.iter().any(|e| matches!(e.op, OpKind::FusedRow(_)));
    let has_reduce = |g: &Graph| g.eqns.iter().any(|e| matches!(e.op, OpKind::Reduce { .. }));

    // (a) bare reduce of a pointwise: sum(x + x) over the last axis -> output [1,4,1].
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 4, 8]));
    let s = b.binary(BinOp::Add, x, x);
    let r = b.reduce(RedOp::Sum, s, 2, true);
    let fg = fuse(&b.finish(r));
    fg.validate().expect("valid");
    assert!(
        !has_row(&fg),
        "bare reduce-output region must not be a FusedRow (0196)"
    );
    assert!(has_reduce(&fg), "the reduce stays a plain Reduce op");

    // (b) scalar pointwise after a reduce (the quant_scale_kv shape): reduce_max(x) * (1/127) -> [1,4,1].
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 4, 8]));
    let m = b.reduce(RedOp::Max, x, 2, true);
    let sc = b.binary_scalar(BinOp::Mul, m, Scalar::F32(1.0 / 127.0));
    let fg = fuse(&b.finish(sc));
    fg.validate().expect("valid");
    assert!(
        !has_row(&fg),
        "scalar-after-reduce reduced output must not be a FusedRow (0197)"
    );
    assert!(has_reduce(&fg), "the reduce stays a plain Reduce op");
}

#[test]
fn fuse_is_idempotent() {
    let b = Builder::new();
    let (a, c, d) = (f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]), f(&b, &[1, 1, 8]));
    let t = b.binary(BinOp::Add, a, c);
    let out = b.binary(BinOp::Mul, t, d);
    let g = b.finish(out);
    let f1 = fuse(&g);
    let f2 = fuse(&f1);
    assert_eq!(f1.eqns.len(), f2.eqns.len());
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 4, 1, 8]));
    let t = b.transpose(x, vec![0, 2, 1, 3]);
    let g = elide_noop_transposes(&b.finish(t));
    assert!(
        g.eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Reshape { .. })),
        "no-op transpose -> reshape"
    );
    assert!(
        !g.eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Transpose { .. })),
        "no transpose dispatch remains"
    );
    // a genuine data-moving transpose is left intact.
    let b2 = Builder::new();
    let y = b2.constant("y", TensorType::f32(vec![3, 5]));
    let ty = b2.transpose(y, vec![1, 0]);
    let g2 = elide_noop_transposes(&b2.finish(ty));
    assert!(
        g2.eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Transpose { .. })),
        "real transpose preserved"
    );
}

#[test]
fn collapse_reshape_chains_merges_a_chain() {
    // reshape -> reshape -> reshape collapses to ONE reshape reading straight from the source, with the
    // final shape preserved; the two dead intermediates are DCE'd.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2, 3, 4]));
    let r1 = b.reshape(x, vec![6, 4]);
    let r2 = b.reshape(r1, vec![24]);
    let r3 = b.reshape(r2, vec![4, 6]);
    let g = b.finish(r3);
    assert_eq!(g.eqns.len(), 3);
    let c = collapse_reshape_chains(&g);
    c.validate().expect("collapsed graph valid");
    assert_eq!(c.eqns.len(), 1, "the reshape chain collapses to one");
    assert!(matches!(c.eqns[0].op, OpKind::Reshape { .. }));
    assert!(
        matches!(c.eqns[0].inputs[0], Operand::Value(v) if v == x.id),
        "reads the source"
    );
    assert_eq!(c.aval(c.output).shape, vec![4, 6], "final shape preserved");
}

#[test]
fn collapse_keeps_a_still_used_intermediate() {
    // if an intermediate reshape feeds another live consumer, it must stay (only the second reshape's
    // input is rewired past it); the graph stays valid and both outputs resolve.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 4]));
    let r1 = b.reshape(x, vec![16]); // used by r2 AND by add
    let r2 = b.reshape(r1, vec![2, 8]);
    let add = b.binary(BinOp::Add, r1, r1); // second consumer of r1 -> r1 stays live
    let out = b.binary(BinOp::Add, b.reshape(r2, vec![16]), add);
    let g = b.finish(out);
    let c = collapse_reshape_chains(&g);
    c.validate().expect("collapsed graph valid");
    assert!(
        c.eqns.iter().any(|e| e.out == r1.id),
        "the multi-use intermediate reshape is retained"
    );
}

#[test]
fn fuse_collapses_the_reshape_after_eliding_a_noop_transpose() {
    // the decode attention shape: reshape -> (no-op) transpose -> reshape. elide turns the transpose into
    // a reshape, then collapse merges the run, so `fuse` emits a single reshape (no transpose, no chain).
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, 32]));
    let r1 = b.reshape(x, vec![1, 1, 4, 8]);
    let t = b.transpose(r1, vec![0, 2, 1, 3]); // [1,4,1,8] - no-op at seq 1
    let out = b.reshape(t, vec![1, 4, 8]);
    let g = b.finish(out);
    let fg = fuse(&g);
    fg.validate().expect("fused graph valid");
    assert!(
        !fg.eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Transpose { .. })),
        "no transpose dispatch remains"
    );
    assert_eq!(
        fg.eqns
            .iter()
            .filter(|e| matches!(e.op, OpKind::Reshape { .. }))
            .count(),
        1,
        "the reshape/transpose/reshape run collapses to one reshape"
    );
    assert_eq!(fg.aval(fg.output).shape, vec![1, 4, 8]);
}

#[test]
fn rope_fusion_full_collapses_the_rotate_half_chain() {
    use poot_graph_ir::graph::Slot;
    use poot_graph_ir::op::UnOp;
    use poot_tensor::DType;
    // full rotary (rot == D): the whole slice/slice/neg/concat/mul/mul/add chain -> one Rope op.
    let b = Builder::new();
    let (d, max_pos) = (8usize, 16usize);
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let out = poot_graph_ir::ops::rope(&b, x, cos, sin, pos);
    let g = b.finish(out);
    let before = dispatch_count(&g);
    let fused = dce(&rope_fusion(&cse(&g)));
    fused.validate().expect("fused graph valid");
    let ropes = fused
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::Rope { .. }))
        .count();
    assert_eq!(ropes, 1, "exactly one fused Rope op");
    // the rotate-half remnants (neg/concat/the split slices) are dead and gone.
    assert!(
        !fused
            .eqns
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
        .eqns
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
    let b = Builder::new();
    let (d, rot, max_pos) = (8usize, 4usize, 16usize);
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, rot]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, rot]));
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let out = poot_graph_ir::ops::rope(&b, x, cos, sin, pos);
    let g = b.finish(out);
    let fused = dce(&rope_fusion(&cse(&g)));
    fused.validate().expect("fused graph valid");
    let ropes: Vec<&Eqn> = fused
        .eqns
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
        !fused.eqns.iter().any(|e| matches!(
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
    let b = Builder::new();
    let (d, max_pos) = (8usize, 16usize);
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let out = poot_graph_ir::ops::rope(&b, x, cos, sin, pos);
    let g = b.finish(out);
    let optimized = dce(&rope_fusion(&cse(&g)));
    optimized.validate().expect("optimized graph valid");
    assert_eq!(
        optimized
            .eqns
            .iter()
            .filter(|e| matches!(e.op, OpKind::Rope { .. }))
            .count(),
        1,
        "cse, rope_fusion then dce must leave exactly one fused Rope op"
    );
    assert!(
        !optimized
            .eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Unary(UnOp::Neg) | OpKind::Concat { .. })),
        "dce must leave no rotate-half neg/concat remnants"
    );
}

#[test]
fn rope_fusion_ignores_a_non_rope_add_of_two_muls() {
    // legality guard: a plain `a*b + c*d` (no rotate-half concat) must NOT match.
    let b = Builder::new();
    let p = b.constant("p", TensorType::f32(vec![4]));
    let q = b.constant("q", TensorType::f32(vec![4]));
    let m0 = b.binary(BinOp::Mul, p, q);
    let m1 = b.binary(BinOp::Mul, q, p);
    let out = b.binary(BinOp::Add, m0, m1);
    let g = b.finish(out);
    let fused = rope_fusion(&cse(&g));
    assert!(
        !fused
            .eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Rope { .. })),
        "a non-rope add of two muls must not match"
    );
}

#[test]
fn cse_remaps_validation_roots() {
    use poot_graph_ir::ValidationId;

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([4]));
    let y = b.constant("y", TensorType::f32([4]));
    let primary = b.binary(BinOp::Add, x, y);
    let duplicate = b.binary(BinOp::Add, x, y);
    let graph = crate::test_support::finish_with_validations(
        b,
        primary,
        &[(ValidationId(3), "duplicate", duplicate)],
    )
    .unwrap();

    let transformed = cse(&graph);
    transformed.validate().unwrap();
    assert_eq!(transformed.eqns.len(), 1);
    assert_eq!(
        transformed.validation_outputs()[0].value,
        transformed.output
    );
}

#[test]
fn dce_keeps_validation_only_producer() {
    use poot_graph_ir::ValidationId;

    let b = Builder::new();
    let primary = b.constant("primary", TensorType::f32([4]));
    let x = b.constant("x", TensorType::f32([4]));
    let y = b.constant("y", TensorType::f32([4]));
    let witness = b.binary(BinOp::Mul, x, y);
    let graph = crate::test_support::finish_with_validations(
        b,
        primary,
        &[(ValidationId(3), "dce", witness)],
    )
    .unwrap();

    let transformed = dce(&graph);
    transformed.validate().unwrap();
    assert!(transformed.eqns.iter().any(|eqn| eqn.out == witness.id));
}

#[test]
fn fusion_pins_validation_packet_values() {
    use poot_graph_ir::ValidationId;

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([4]));
    let y = b.constant("y", TensorType::f32([4]));
    let witness = b.binary(BinOp::Mul, x, y);
    let primary = b.binary(BinOp::Add, witness, y);
    let graph = crate::test_support::finish_with_validations(
        b,
        primary,
        &[(ValidationId(3), "fusion", witness)],
    )
    .unwrap();

    let transformed = fuse(&graph);
    transformed.validate().unwrap();
    assert!(
        transformed.eqns.iter().any(|eqn| eqn.out == witness.id),
        "the validation root must not be absorbed into its consumer"
    );
}

#[test]
fn movement_cleanup_preserves_validation_only_root() {
    use poot_graph_ir::ValidationId;

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([1, 2, 1]));
    let witness = b.transpose(x, vec![0, 2, 1]);
    let reshaped = b.reshape(witness, vec![2]);
    let primary = b.reshape(reshaped, vec![1, 2]);
    let graph = crate::test_support::finish_with_validations(
        b,
        primary,
        &[(ValidationId(6), "movement", witness)],
    )
    .unwrap();

    let transformed = collapse_reshape_chains(&elide_noop_transposes(&graph));
    transformed.validate().unwrap();
    assert_eq!(transformed.validations, graph.validations);
    assert!(
        transformed
            .eqns
            .iter()
            .any(|eqn| eqn.out == witness.id && matches!(eqn.op, OpKind::Reshape { .. })),
        "the validation-only movement value must stay independently materialized"
    );
}

// ---- ADR-0099 S2: the stage split ---------------------------------------------------------

/// A `layers`-block graph: one Activation input, one carried state pair per block, and one shared
/// constant. Three equations per block, so block `l` owns equations `3l..3l + 3`.
fn stage_split_layered_graph(layers: usize) -> Graph {
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let w = b.constant("w", TensorType::f32(vec![4]));
    let mut hidden = x;
    let mut state = Vec::new();
    for layer in 0..layers {
        let scaled = b.binary(BinOp::Mul, hidden, w);
        let k_in = b.state_input(
            &format!("k{layer}"),
            TensorType::f32(vec![4]),
            StateRole::Recurrent,
        );
        let k_out = b.binary(BinOp::Add, k_in, scaled);
        let next = b.binary(BinOp::Add, scaled, k_out);
        state.push((k_in, k_out));
        hidden = next;
    }
    b.finish_with_state(hidden, &state)
}

#[test]
fn adr0099_s2_stage_graphs_partition_every_equation_and_validate() {
    let graph = stage_split_layered_graph(3);
    assert_eq!(graph.eqns.len(), 9, "three equations per block");

    for (cuts, expected_boundaries, expected_pairs) in [
        (vec![3usize], 1usize, vec![vec![0usize], vec![1, 2]]),
        (vec![3usize, 6], 2usize, vec![vec![0], vec![1], vec![2]]),
    ] {
        let assignment =
            StageAssignment::from_cuts(&cuts, graph.eqns.len()).expect("the cuts are legal");
        let split = split_stages(&graph, &assignment).expect("the graph splits");
        assert_eq!(split.stages.len(), cuts.len() + 1);
        assert_eq!(split.boundaries.len(), expected_boundaries);

        // Every equation lands in exactly one stage, in order, and none is invented.
        let mut seen = Vec::new();
        for stage in &split.stages {
            seen.extend(stage.graph.eqns.iter().map(|eqn| eqn.out));
        }
        assert_eq!(
            seen,
            graph.eqns.iter().map(|eqn| eqn.out).collect::<Vec<_>>()
        );

        // Carried state stays whole: stage `s` owns exactly the pairs listed for it, and a pair the
        // stage does not own never reaches its binding tables.
        for (stage_index, stage) in split.stages.iter().enumerate() {
            let owned: Vec<(usize, (ValueId, ValueId))> = graph
                .state
                .iter()
                .enumerate()
                .filter(|&(pair, _)| stage.graph.state.contains(&graph.state[pair]))
                .map(|(pair, &entry)| (pair, entry))
                .collect();
            assert_eq!(
                owned,
                expected_pairs[stage_index]
                    .iter()
                    .map(|&pair| (pair, graph.state[pair]))
                    .collect::<Vec<_>>(),
                "stage {stage_index} owns its own carried state"
            );
            let foreign: Vec<ValueId> = graph
                .state
                .iter()
                .flat_map(|&(state_input, _)| [state_input])
                .filter(|value| stage.graph.inputs.contains(value))
                .filter(|value| {
                    !expected_pairs[stage_index]
                        .iter()
                        .any(|&p| graph.state[p].0 == *value)
                })
                .collect();
            assert!(
                foreign.is_empty(),
                "stage {stage_index} must not bind another stage's cache: {foreign:?}"
            );
        }

        // Each crossing value is an equation result of its producer and an activation input of every
        // consumer, explicitly listed in both places.
        for boundary in &split.boundaries {
            let producer = &split.stages[boundary.producing_stage];
            assert!(
                producer.outputs.contains(&boundary.value),
                "v{} is an explicit output of stage {}",
                boundary.value,
                boundary.producing_stage
            );
            assert!(
                producer
                    .graph
                    .eqns
                    .iter()
                    .any(|eqn| eqn.out == boundary.value),
                "the producing stage still defines v{}",
                boundary.value
            );
            assert_eq!(
                producer.graph.values[boundary.value].storage,
                Storage::Device
            );
            for &consumer in &boundary.consuming_stages {
                let stage = &split.stages[consumer];
                assert!(
                    stage.graph.inputs.contains(&boundary.value),
                    "v{} is an explicit input of stage {consumer}",
                    boundary.value
                );
                assert_eq!(
                    stage.graph.values[boundary.value].storage,
                    Storage::Slot(poot_graph_ir::graph::Slot::Activation),
                    "v{} is bound as an activation input of stage {consumer}",
                    boundary.value
                );
                assert!(
                    stage
                        .graph
                        .slots
                        .contains(&(boundary.value, poot_graph_ir::graph::Slot::Activation)),
                    "v{} has a slot binding in stage {consumer}",
                    boundary.value
                );
                assert!(
                    !stage.graph.eqns.iter().any(|eqn| eqn.out == boundary.value),
                    "stage {consumer} reads v{}, it does not redefine it",
                    boundary.value
                );
            }
        }
    }
}

#[test]
fn adr0099_s2_boundary_descriptors_carry_dtype_shape_and_byte_len() {
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![3, 4]),
    );
    let dense = b.binary(BinOp::Add, x, x);
    let bf16 = b.cast(dense, DType::BF16);
    let i32 = b.cast(dense, DType::I32);
    // F32 -> I8 directly is refused at `infer` time (card 555): go through the I32 value
    // already produced above, same as every other I32 -> I8 cast in the IR.
    let i8 = b.cast(i32, DType::I8);
    let back_bf16 = b.cast(bf16, DType::F32);
    let back_i32 = b.cast(i32, DType::F32);
    let back_i8 = b.cast(i8, DType::F32);
    let mixed = b.binary(BinOp::Add, back_bf16, back_i32);
    let out = b.binary(BinOp::Add, mixed, back_i8);
    let graph = b.finish(out);
    assert_eq!(graph.eqns.len(), 9);

    let assignment = StageAssignment::from_cuts(&[4], graph.eqns.len()).expect("the cut is legal");
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
                boundary.value,
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
            (bf16.id, DType::BF16, 24, 0, vec![1]),
            (i32.id, DType::I32, 48, 0, vec![1]),
            (i8.id, DType::I8, 12, 0, vec![1]),
        ]
    );
    assert!(
        split
            .stages
            .iter()
            .all(|stage| stage.graph.validate().is_ok()),
        "every stage graph validates"
    );
}

#[test]
fn adr0099_s2_state_pair_split_across_stages_is_rejected() {
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let k_in = b.state_input("k", TensorType::f32(vec![4]), StateRole::Recurrent);
    let early = b.binary(BinOp::Add, x, k_in);
    let mid = b.binary(BinOp::Mul, early, early);
    let k_out = b.binary(BinOp::Add, mid, mid);
    let graph = b.finish_with_state(k_out, &[(k_in, k_out)]);

    let assignment = StageAssignment::from_cuts(&[1], graph.eqns.len()).expect("the cut is legal");
    assert_eq!(
        split_stages(&graph, &assignment).expect_err("the pair spans both stages"),
        StageSplitError::StatePairSplit {
            pair: 0,
            state_input: k_in.id,
            state_output: k_out.id,
            first_stage: 0,
            second_stage: 1,
        }
    );
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
        StageAssignment::from_cuts(&[0], 4).expect_err("a cut at 0 empties the first stage"),
        StageSplitError::CutOutOfRange {
            index: 0,
            cut: 0,
            eqn_count: 4,
        }
    );
    assert_eq!(
        StageAssignment::from_cuts(&[4], 4).expect_err("a cut at the end empties the last stage"),
        StageSplitError::CutOutOfRange {
            index: 0,
            cut: 4,
            eqn_count: 4,
        }
    );
    assert_eq!(
        StageAssignment::from_cuts(&[2, 2], 4).expect_err("cuts must strictly increase"),
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
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let t0 = b.binary(BinOp::Add, x, x);
    let _d1 = b.binary(BinOp::Mul, t0, t0);
    let out = b.binary(BinOp::Mul, t0, t0);
    let graph = b.finish(out);

    let assignment =
        StageAssignment::from_cuts(&[1, 2], graph.eqns.len()).expect("the cuts are legal");
    assert_eq!(
        split_stages(&graph, &assignment)
            .expect_err("stage 1 reads only its own results, so it never feeds stage 2"),
        StageSplitError::StageWithoutExit { stage: 1 }
    );
}

#[test]
fn adr0099_s2_a_final_stage_that_never_sees_the_output_is_rejected() {
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let t0 = b.binary(BinOp::Add, x, x);
    let t1 = b.binary(BinOp::Mul, t0, t0);
    let _after_t1 = b.binary(BinOp::Mul, t1, t1);
    let graph = b.finish(t0);

    let assignment =
        StageAssignment::from_cuts(&[1, 2], graph.eqns.len()).expect("the cuts are legal");
    assert_eq!(
        split_stages(&graph, &assignment)
            .expect_err("the output is defined in stage 0 and reaches no later stage"),
        StageSplitError::FinalStageMissingOutput {
            stage: 2,
            output: t0.id,
        }
    );
}

#[test]
fn adr0099_s2_each_stage_keeps_only_the_witnesses_it_defines() {
    use poot_graph_ir::ValidationId;

    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let t0 = b.binary(BinOp::Add, x, x);
    let w0 = b.binary(BinOp::Mul, x, x);
    let t1 = b.binary(BinOp::Mul, t0, t0);
    let w1 = b.binary(BinOp::Sub, t1, t1);
    let graph = crate::test_support::finish_with_validations(
        b,
        t1,
        &[
            (ValidationId(1), "stage0_witness", w0),
            (ValidationId(2), "stage1_witness", w1),
        ],
    )
    .expect("the witness values are f32 tensors");

    let assignment = StageAssignment::from_cuts(&[2], graph.eqns.len()).expect("the cut is legal");
    let split = split_stages(&graph, &assignment).expect("the graph splits");

    let declarations = |stage: usize| -> Vec<(poot_graph_ir::ValidationId, &str, ValueId)> {
        split.stages[stage]
            .graph
            .validation_outputs()
            .iter()
            .map(|declaration| (declaration.id, declaration.name.as_str(), declaration.value))
            .collect()
    };
    assert_eq!(
        declarations(0),
        vec![(ValidationId(1), "stage0_witness", w0.id)]
    );
    assert_eq!(
        declarations(1),
        vec![(ValidationId(2), "stage1_witness", w1.id)]
    );
}

/// A traced graph with an untagged preamble, three scoped layers of four equations each, and an
/// untagged head - the shape a layer-pipeline placement cuts. Within a layer: a `Mul`, a duplicate
/// of that `Mul` (CSE folds it), a dead `Add` (DCE drops it), and the `Add` that carries the
/// hidden state forward.
fn layer_scoped_graph() -> Graph {
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let w = b.constant("w", TensorType::f32(vec![4]));
    let _preamble = b.binary(BinOp::Add, x, x);
    let mut hidden = x;
    for layer in 0..3usize {
        let _layer_scope = b.layer_scope(layer);
        let scaled = b.binary(BinOp::Mul, hidden, w);
        let duplicate = b.binary(BinOp::Mul, hidden, w);
        let _dead = b.binary(BinOp::Add, duplicate, duplicate);
        hidden = b.binary(BinOp::Add, scaled, x);
    }
    let out = b.binary(BinOp::Mul, hidden, hidden);
    b.finish(out)
}

fn layer_tags(graph: &Graph) -> Vec<Option<usize>> {
    graph
        .eqns
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
    let split = split_stages_after_layers(&graph, &[0, 1]).expect("both cuts map onto a boundary");
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
    let two = split_stages_after_layers(&graph, &[0]).expect("after_layer 0 maps");
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
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let in_layer_zero = {
        let _layer_scope = b.layer_scope(0);
        b.binary(BinOp::Mul, x, x)
    };
    let hole = b.binary(BinOp::Add, x, x);
    let in_layer_one = {
        let _layer_scope = b.layer_scope(1);
        b.binary(BinOp::Mul, hole, hole)
    };
    let holey = b.finish(in_layer_one);
    let _ = in_layer_zero;
    assert_eq!(
        StageAssignment::from_layer_cuts(&holey, &[0]),
        Err(StageSplitError::UntaggedLayerEquation { equation: 1 })
    );

    // A skipped layer id: layer 2 appears where layer 1 should start.
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    {
        let _layer_scope = b.layer_scope(0);
        b.binary(BinOp::Mul, x, x);
    }
    let skipped = {
        let _layer_scope = b.layer_scope(2);
        b.binary(BinOp::Mul, x, x)
    };
    let skipped = b.finish(skipped);
    assert_eq!(
        StageAssignment::from_layer_cuts(&skipped, &[0]),
        Err(StageSplitError::NonContiguousLayer {
            equation: 1,
            layer: 2,
            expected: 1,
        })
    );

    // A graph whose first tagged equation is not layer 0 never had a layer 0.
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let started_at_one = {
        let _layer_scope = b.layer_scope(1);
        b.binary(BinOp::Mul, x, x)
    };
    let started_at_one = b.finish(started_at_one);
    assert_eq!(
        StageAssignment::from_layer_cuts(&started_at_one, &[]),
        Err(StageSplitError::NonContiguousLayer {
            equation: 0,
            layer: 1,
            expected: 0,
        })
    );

    // A cut past the last layer would produce an empty stage, and a cut on an untagged graph has
    // no layer to land on at all. (A cut at layer 1 of three is legal - it separates 1 from 2.)
    let tagged = layer_scoped_graph();
    assert!(StageAssignment::from_layer_cuts(&tagged, &[1]).is_ok());
    assert_eq!(
        StageAssignment::from_layer_cuts(&tagged, &[2]),
        Err(StageSplitError::LayerCutOutOfRange { cut: 2, layers: 3 })
    );
    let b = Builder::new();
    let x = b.slot(
        poot_graph_ir::graph::Slot::Activation,
        TensorType::f32(vec![4]),
    );
    let untagged_value = b.binary(BinOp::Add, x, x);
    let untagged = b.finish(untagged_value);
    assert_eq!(
        StageAssignment::from_layer_cuts(&untagged, &[0]),
        Err(StageSplitError::LayerCutOutOfRange { cut: 0, layers: 0 })
    );

    // Cuts must also strictly increase, the same contract `from_cuts` enforces.
    assert_eq!(
        StageAssignment::from_layer_cuts(&tagged, &[0, 0]),
        Err(StageSplitError::CutOutOfOrder {
            index: 1,
            cut: 0,
            previous: 0,
        })
    );
}

#[test]
fn adr0099_s2_cse_and_dce_preserve_layer_tags() {
    let graph = layer_scoped_graph();
    let original: HashMap<ValueId, Option<LayerIndex>> =
        graph.eqns.iter().map(|eqn| (eqn.out, eqn.layer)).collect();

    let folded = cse(&graph);
    assert!(
        folded.eqns.len() < graph.eqns.len(),
        "cse must fold the duplicate Mul in each layer, or this row proves nothing"
    );
    for eqn in &folded.eqns {
        assert_eq!(
            eqn.layer, original[&eqn.out],
            "cse changed the layer tag of v{}",
            eqn.out
        );
    }

    let trimmed = dce(&graph);
    assert!(
        trimmed.eqns.len() < graph.eqns.len(),
        "dce must drop the dead layer equation, or this row proves nothing"
    );
    for eqn in &trimmed.eqns {
        assert_eq!(
            eqn.layer, original[&eqn.out],
            "dce changed the layer tag of v{}",
            eqn.out
        );
    }

    // Preserved tags mean a layer cut still maps on the transformed graphs.
    for (label, transformed) in [("cse", &folded), ("dce", &trimmed)] {
        let split = split_stages_after_layers(transformed, &[0, 1])
            .unwrap_or_else(|error| panic!("{label} must keep the tags a cut maps onto: {error}"));
        assert_eq!(split.stages.len(), 3, "{label} keeps three layers");
        assert_eq!(
            split
                .stages
                .iter()
                .map(|stage| stage.graph.eqns.iter().any(|eqn| eqn.layer.is_some()))
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
    type PassFn =
        fn(&Graph<poot_graph_ir::ValidationOutputs>) -> Graph<poot_graph_ir::ValidationOutputs>;

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([4]));
    let y = b.constant("y", TensorType::f32([4]));
    let witness = b.binary(BinOp::Mul, x, y);
    let primary = b.binary(BinOp::Add, witness, y);
    let graph = crate::test_support::finish_with_validations(
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
        ("fuse_bias_epilogues", fuse_bias_epilogues),
        ("fuse", fuse),
    ];
    for (name, pass) in passes {
        let transformed = pass(&graph);
        transformed.validate().unwrap();
        assert_eq!(
            transformed.validation_outputs(),
            graph.validation_outputs(),
            "{name} dropped or changed a witness declaration"
        );
        assert!(
            transformed.validation_outputs().iter().all(|output| {
                transformed.eqns.iter().any(|eqn| eqn.out == output.value)
                    || transformed.consts.contains(&output.value)
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([4]));
    let y = b.constant("y", TensorType::f32([4]));
    let product = b.binary(BinOp::Mul, x, y);
    let sum = b.binary(BinOp::Add, product, y);
    let graph = b.finish(sum);

    let fused = fuse(&graph);
    assert!(
        fused
            .eqns
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
    let b = Builder::new();
    let activation = b.constant("activation", TensorType::f32([1, 8]));
    let weight = b.constant("weight", TensorType::f32([8, 8]));
    let out = b.matmul(activation, weight);
    let graph = b.finish(out);
    assert!(
        matches!(
            graph.eqns.as_slice(),
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([2, 2]));
    let moved = b.reshape(x, vec![4]);
    let movement = b.finish(moved);
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
    AttentionDecline, RopeDecline, canonicalize, match_attention, match_rope_core, producer_map,
};

fn output_eqn(g: &Graph) -> &Eqn {
    g.eqns
        .iter()
        .find(|eqn| eqn.out == g.output)
        .expect("the graph output has a producer")
}

/// A full-rotary rotate-half rope with free `x`/`cos`/`sin` shapes; `commute` writes `Mul(cos, x)`.
fn hand_rope(x_shape: Vec<usize>, cos_shape: Vec<usize>, commute: bool) -> Graph {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(x_shape.clone()));
    let cos = b.constant("cos", TensorType::f32(cos_shape.clone()));
    let sin = b.constant("sin", TensorType::f32(cos_shape));
    let last = x_shape.len() - 1;
    let d = x_shape[last];
    let half = d / 2;
    let x1 = b.slice(x, last, 0, half);
    let x2 = b.slice(x, last, half, d);
    let neg_x2 = b.unary(UnOp::Neg, x2);
    let rotate_half = b.concat(last, &[neg_x2, x1]);
    let xc = if commute {
        b.binary(BinOp::Mul, cos, x)
    } else {
        b.binary(BinOp::Mul, x, cos)
    };
    let rs = b.binary(BinOp::Mul, rotate_half, sin);
    let out = b.binary(BinOp::Add, xc, rs);
    b.finish(out)
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
    let mut wrong_axis = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask);
        b.finish(out)
    };
    for eqn in wrong_axis.eqns.iter_mut() {
        if let OpKind::Reduce { axis, .. } = &mut eqn.op {
            *axis = 0;
        }
    }
    cases.push(("wrong reduce axis", wrong_axis, AttentionDecline::Denom));

    // Scale: a softcap chain whose two constants are not reciprocals.
    let mut wrong_softcap = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(2.0));
        b.finish(out)
    };
    let mut rewrote = false;
    for eqn in wrong_softcap.eqns.iter_mut() {
        if let OpKind::Binary(BinOp::Mul) = eqn.op
            && let Some(Operand::Lit(poot_graph_ir::Scalar::F32(0.5))) = eqn.inputs.get(1)
        {
            eqn.inputs[1] = Operand::Lit(poot_graph_ir::Scalar::F32(0.6));
            rewrote = true;
        }
    }
    assert!(rewrote, "the softcap reciprocal literal must be present");
    cases.push((
        "softcap constants not reciprocal",
        wrong_softcap,
        AttentionDecline::SoftcapRecip,
    ));

    // Shape: a softcap chain whose `Tanh` slot holds another unary (here `Neg`): same literals, same
    // arity, but not a tanh. Mutation: drop the `Tanh` check in `match_pre_mask`; this fuses.
    let mut not_tanh = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(2.0));
        b.finish(out)
    };
    let mut swapped = 0;
    for eqn in not_tanh.eqns.iter_mut() {
        if matches!(eqn.op, OpKind::Unary(poot_graph_ir::UnOp::Tanh)) {
            eqn.op = OpKind::Unary(poot_graph_ir::UnOp::Neg);
            swapped += 1;
        }
    }
    assert_eq!(swapped, 1, "exactly the softcap tanh is replaced");
    cases.push((
        "softcap tanh slot is not a tanh",
        not_tanh,
        AttentionDecline::SoftcapShape,
    ));

    // Mask: the value width differs from the QK width (MLA); the fused op indexes one width.
    let mla = {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 3]));
        let k = b.constant("k", TensorType::f32(vec![1, 2, 5, 3]));
        let v = b.constant("v", TensorType::f32(vec![1, 2, 5, 2]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, 5]));
        let out = attention_masked(&b, q, k, v, 1, 1.0, mask);
        b.finish(out)
    };
    cases.push((
        "V width differs from QK",
        mla,
        AttentionDecline::TypeMismatch,
    ));

    for (name, graph, expected) in cases {
        let reason = match_attention(&graph, &producer_map(&graph), output_eqn(&graph));
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

    // The rotate-half concat joins on a non-last axis.
    let mut wrong_axis = hand_rope(vec![1, 4, 3, 8], vec![1, 1, 3, 8], false);
    for eqn in wrong_axis.eqns.iter_mut() {
        if let OpKind::Concat { axis } = &mut eqn.op {
            *axis = 0;
        }
    }
    cases.push((
        "rotate-half on axis 0",
        wrong_axis,
        RopeDecline::RotateHalfConcat,
    ));

    // The two slices do not split the same tensor: cut the keep slice from a different tensor.
    let mut wrong_source = hand_rope(vec![1, 4, 3, 8], vec![1, 1, 3, 8], false);
    let other = wrong_source.values.len();
    wrong_source
        .values
        .push(poot_graph_ir::graph::ValueMeta::new(
            TensorType::f32(vec![1, 4, 3, 8]),
            Storage::Const,
            Some("other".into()),
        ));
    wrong_source.consts.push(other);
    for eqn in wrong_source.eqns.iter_mut() {
        let keep_slice = matches!(&eqn.op, OpKind::Slice { start: 0, .. });
        if keep_slice && let Some(Operand::Value(value)) = eqn.inputs.first_mut() {
            *value = other;
        }
    }
    cases.push((
        "slices from different tensors",
        wrong_source,
        RopeDecline::SliceMismatch,
    ));

    // The cos/sin table is not the rotated width.
    let mut wrong_width = hand_rope(vec![1, 4, 3, 8], vec![1, 1, 3, 8], false);
    for value in wrong_width.values.iter_mut() {
        if value.name.as_deref() == Some("cos") {
            value.aval = TensorType::f32(vec![1, 1, 3, 4]);
        }
    }
    cases.push((
        "cos is not the rotated width",
        wrong_width,
        RopeDecline::TableWidth,
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
/// reasons the four R486 guards produce, so dropping one changes its fixture's reason:
///
/// - the softcap reciprocal-literal check (`SoftcapRecip`),
/// - the K/V GQA repeat cross-check (`GqaRepeatMismatch`),
/// - the decode-softcap refusal (`DecodeSoftcap`),
/// - the row-softmax reduce-axis check (`Denom`, the surviving form of the deleted softmax matcher's
///   `softmax.rs:27` guard).
#[test]
fn card536a_guards_have_killing_reasons() {
    // Decode softcap refusal.
    let decode_softcap = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(30.0));
        b.finish(out)
    };
    assert_eq!(
        match_attention(
            &decode_softcap,
            &producer_map(&decode_softcap),
            output_eqn(&decode_softcap)
        )
        .err(),
        Some(AttentionDecline::DecodeSoftcap)
    );

    // GQA repeat cross-check: k is repeated by 2, v is not. The chain stays a valid decomposition
    // (V keeps four heads) but the repeated K carries a different factor.
    let (hq, cap, d, hkv) = (4usize, 6usize, 16usize, 2usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let kt = b.transpose(poot_graph_ir::ops::repeat_kv(&b, k, 2), vec![0, 1, 3, 2]);
    let scores = b.matmul(q, kt);
    let scores = b.binary_scalar(BinOp::Mul, scores, poot_graph_ir::Scalar::F32(0.25));
    let scores = b.binary(BinOp::Add, scores, mask);
    let last = b.aval(scores).rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true);
    let e = b.unary(UnOp::Exp, b.binary(BinOp::Sub, scores, m));
    let denom = b.reduce(RedOp::Sum, e, last, true);
    let p = b.binary(BinOp::Div, e, denom);
    let gqa_out = b.matmul(p, v);
    let gqa = b.finish(gqa_out);
    assert_eq!(
        match_attention(&gqa, &producer_map(&gqa), output_eqn(&gqa)).err(),
        Some(AttentionDecline::GqaRepeatMismatch)
    );
}

/// Head-dim awareness decline reason: a decode match wider than `FLASH_LDS_CAP` declines
/// with `HeadDimExceedsCap` (the oracle-unchanged half of this fixture lives in
/// `transform_soundness.rs`'s `card536a_flash_head_dim_is_aware`). Mutation: remove the head-dim
/// gate; this assertion fails (the match succeeds instead of declining).
#[test]
fn card536a_flash_head_dim_is_aware_decline_reason() {
    let build = |m: usize, d: usize| {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, 2, m, d]));
        let k = b.constant("k", TensorType::f32(vec![1, 2, 5, d]));
        let v = b.constant("v", TensorType::f32(vec![1, 2, 5, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, m, 5]));
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask);
        b.finish(out)
    };
    let wide = build(1, 512);
    assert_eq!(
        match_attention(&wide, &producer_map(&wide), output_eqn(&wide)).err(),
        Some(AttentionDecline::HeadDimExceedsCap),
        "D=512 decode must decline"
    );
}

/// Canonicalization retains unproved arithmetic and reports why; only F32 multiply-by-one is
/// removed from a contraction operand. The CPU bitwise corpus lives in numerical_legality.rs.
#[test]
fn canonicalize_preserves_unproved_spellings_and_removes_identity() {
    for scale in [0.25, 1.0] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 4, 8]));
        let w = b.constant("w", TensorType::f32(vec![8, 5]));
        let scaled = b.binary_scalar(BinOp::Mul, x, poot_graph_ir::Scalar::F32(scale));
        let output = b.matmul(scaled, w);
        let graph = b.finish(output);
        let (canonical, declines) = canonicalize_with_declines(&graph);
        canonical.validate().unwrap();
        let root = canonical.eqns.last().unwrap();
        assert!(matches!(root.op, OpKind::MatMul));
        assert_eq!(root.out, graph.output);
        if scale == 1.0 {
            assert!(matches!(root.inputs[0], Operand::Value(id) if id == x.id));
            assert!(declines.is_empty());
        } else {
            assert_eq!(format!("{:?}", canonical.eqns), format!("{:?}", graph.eqns));
            assert_eq!(
                declines,
                vec![NumericalRewriteDecline {
                    value: output.id,
                    reason: NumericalRewriteReason::ScalarAcrossContraction,
                }]
            );
        }
        assert_eq!(
            format!("{:?}", canonicalize(&canonical).eqns),
            format!("{:?}", canonical.eqns)
        );
    }
    let b = Builder::new();
    let e = b.constant("e", TensorType::f32(vec![3, 4]));
    let denom = b.constant("denom", TensorType::f32(vec![3, 1]));
    let recip = b.unary(UnOp::Recip, denom);
    let mul = b.binary(BinOp::Mul, e, recip);
    let graph = b.finish(mul);
    let (canonical, declines) = canonicalize_with_declines(&graph);
    assert_eq!(format!("{:?}", canonical.eqns), format!("{:?}", graph.eqns));
    assert_eq!(
        declines,
        vec![NumericalRewriteDecline {
            value: mul.id,
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
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let scale = b.constant("scale", TensorType::f32(vec![1]));
    let kt = b.transpose(k, vec![0, 1, 3, 2]);
    let qk = b.matmul(q, kt);
    let scores = b.binary(BinOp::Mul, scale, qk);
    let scores = b.binary(BinOp::Add, scores, mask);
    let last = b.aval(scores).rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true);
    let e = b.unary(UnOp::Exp, b.binary(BinOp::Sub, scores, m));
    let denom = b.reduce(RedOp::Sum, e, last, true);
    let p = b.binary(BinOp::Div, e, denom);
    let graph_out = b.matmul(p, v);
    let graph = b.finish(graph_out);

    assert_eq!(
        match_attention(&graph, &producer_map(&graph), output_eqn(&graph)).err(),
        Some(AttentionDecline::ScoreProduct),
        "a runtime-valued scale is a typed decline, not a dropped parameter"
    );
}

/// Card 557 SC-003 (R-557-1): a traced bias linear is primitives only, `matmul` then a broadcast `add`
/// with no `MatMulBias`, and the one contraction fuse rule forms exactly one `MatMulBias(x, w, bias)`
/// from it, with the operands in the order every planner arm and the oracle read (`inputs[2]` is the
/// `[N]` bias). The fused graph evaluates bit for bit like the traced one. Mutations: emit `MatMulBias`
/// from `ops::linear` (the traced assertion goes red); emit the bias as the rule's first operand (the
/// operand assertion goes red).
#[test]
fn a_traced_bias_linear_is_primitive_and_fuses_into_one_matmul_bias() {
    use poot_eval::{EvalBudget, EvalOptions, Value, eval};
    use poot_graph_ir::OpClass;
    use poot_tensor::HostTensor;

    let (m, k, n) = (3usize, 5usize, 4usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let out = poot_graph_ir::ops::linear(&b, x, w, Some(bias));
    let traced = b.finish(out);
    assert!(
        traced
            .eqns
            .iter()
            .all(|eqn| eqn.op.class() == OpClass::Primitive),
        "a tracer emits primitives only: {:?}",
        traced.eqns
    );

    let fused = fuse_bias_epilogues(&traced);
    let epilogues: Vec<&Eqn> = fused
        .eqns
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::MatMulBias))
        .collect();
    assert_eq!(epilogues.len(), 1, "one MatMulBias: {:?}", fused.eqns);
    assert!(
        matches!(
            epilogues[0].inputs.as_slice(),
            [Operand::Value(a), Operand::Value(b), Operand::Value(c)]
                if [*a, *b, *c] == [x.id, w.id, bias.id]
        ),
        "MatMulBias reads (activation, weight, bias): {:?}",
        epilogues[0].inputs
    );
    assert_eq!(
        fused.eqns.len(),
        1,
        "the matmul and the add are one equation"
    );
    assert_eq!(
        epilogues[0].out, traced.output,
        "the add's output id is kept"
    );

    let data = |len: usize, seed: f32| -> Vec<f32> {
        (0..len).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
    };
    let inputs: HashMap<ValueId, Value> = [
        (x.id, HostTensor::f32(vec![m, k], data(m * k, 1.0))),
        (w.id, HostTensor::f32(vec![k, n], data(k * n, 2.0))),
        (bias.id, HostTensor::f32(vec![n], data(n, 3.0))),
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
