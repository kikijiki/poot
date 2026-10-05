//! Card 380 and card 381: planned-graph evidence for the two Qwen3.8-27B `[vocab, hidden]` BF16 sources
//! on wgpu - the LM head weight and the embedding table.
//!
//! The head is `MatMul(F32 activation, Transpose(BF16 weight)) -> F32`. Asserting on the traced graph
//! (card 376) cannot see what preparation does: `widen_mismatched_matmul_dtypes` used to put a
//! `Cast(BF16 -> F32)` on the head weight, a 5.09 GB F32 copy of a 2.54 GB matrix at the pinned dims.
//!
//! This test inspects the PREPARED graph, through the same `prepare_target_graph` the executor runs.
//! Both `[vocab, hidden]` BF16 sources (head weight and embedding table) lower without an F32 copy.
//! The file keeps its card-380 name so update 1257's verification command still resolves.
//!
//! The Qwen3.8-27B wgpu lane is still blocked: each packed matrix is 2_542_796_800 bytes against
//! wgpu's 2_147_483_647-byte `maxBufferSize`, so neither fits one buffer. This is host-side planning
//! evidence only.

use poot_graph_ir::{OpKind, Operand, ValueId};
use poot_graph_plan::prepare_target_graph;
use poot_models::qwen38_27b::{Qwen35TextConfig, trace_qwen38_text_decode};
use poot_target::Backend;
use poot_tensor::DType;

const HEAD: &str = "lm_head.weight";
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";

/// `Operand` carries no `PartialEq`, so read an operand's value id instead of comparing operands.
fn operand_value(eqn: &poot_graph_ir::Eqn, position: usize) -> Option<ValueId> {
    match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(*value),
        _ => None,
    }
}

fn value_named(graph: &poot_graph_ir::Graph, name: &str) -> ValueId {
    graph
        .values
        .iter()
        .position(|value| value.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("the traced graph declares {name}"))
}

/// SC-001 and SC-002. The prepared decode graph lowers the head as one `DenseContraction` over the BF16
/// constant itself, and holds no widened copy of it.
#[test]
fn qwen38_wgpu_prepared_graph_holds_no_widened_lm_head_weight() {
    let config = Qwen35TextConfig::pinned();
    let vocab_matrix = config.vocab * config.hidden;
    let traced = trace_qwen38_text_decode(&config, 0, 2).expect("the pinned config traces");

    // The traced head is the Card 376 shape: a BF16 constant, transposed, then used as a matmul weight.
    let head = value_named(&traced, HEAD);
    assert_eq!(traced.aval(head).dtype, DType::BF16);
    assert_eq!(traced.aval(head).shape, vec![config.vocab, config.hidden]);

    let caps = poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan);
    let prepared = prepare_target_graph(&traced, Backend::SpirvVulkan, &caps);

    // The typed walk uploads the checkpoint's BF16 matrix as packed u32 lanes, so preparation must
    // neither retype nor widen it.
    let prepared_head = value_named(&prepared, HEAD);
    assert_eq!(
        prepared.aval(prepared_head).dtype,
        DType::BF16,
        "a widened or retyped head weight is the F32 copy this card exists to remove"
    );

    // Exactly one BF16 contraction reading the constant directly, with no transpose (a second full-size
    // buffer; card 258 tripped the iGPU watchdog with that dispatch shape). The model's F32 projections are
    // contractions too since Card 645, so the head is the only one over a BF16 weight.
    let contractions = prepared
        .eqns
        .iter()
        .filter(|eqn| {
            matches!(
                eqn.op,
                OpKind::DenseContraction {
                    weight: DType::BF16
                }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        contractions.len(),
        1,
        "the LM head is the only BF16 contraction"
    );
    assert_eq!(
        contractions[0].op,
        OpKind::DenseContraction {
            weight: DType::BF16
        }
    );
    assert_eq!(
        operand_value(contractions[0], 1),
        Some(prepared_head),
        "the contraction reads the checkpoint constant, not a copy of it"
    );
    assert!(
        !prepared.eqns.iter().any(|eqn| {
            matches!(eqn.op, OpKind::Transpose { .. })
                && operand_value(eqn, 0) == Some(prepared_head)
        }),
        "the head weight must not be transposed on device"
    );

    // No equation anywhere widens a vocabulary-sized BF16 value.
    for eqn in &prepared.eqns {
        let OpKind::Cast { to: DType::F32 } = eqn.op else {
            continue;
        };
        let Some(source) = operand_value(eqn, 0) else {
            continue;
        };
        assert!(
            prepared.aval(source).dtype != DType::BF16
                || prepared.aval(source).numel() < vocab_matrix,
            "Cast(BF16 -> F32) over a vocabulary-sized value {:?}",
            prepared.meta(source).name
        );
    }

    // No value in the prepared graph is an F32 matrix of that size (covers head and embedding).
    for value in &prepared.values {
        if value.aval.dtype != DType::F32 {
            continue;
        }
        assert!(
            value.aval.numel() < vocab_matrix,
            "F32 vocabulary matrix {:?} in the prepared graph",
            value.name
        );
    }
}

/// Card 381 SC-001 and SC-002. The embedding lowers as one `DenseRowGather` over the BF16 constant:
/// gather and widening cast are one equation, so no copy of the table exists.
#[test]
fn qwen38_wgpu_prepared_graph_holds_no_widened_embedding_table() {
    let config = Qwen35TextConfig::pinned();
    let vocab_matrix = config.vocab * config.hidden;
    let traced = trace_qwen38_text_decode(&config, 0, 2).expect("the pinned config traces");

    // The traced embedding is the Card 376 shape: a BF16 constant gathered by the token id, then widened.
    let embedding = value_named(&traced, EMBEDDING);
    assert_eq!(traced.aval(embedding).dtype, DType::BF16);
    assert_eq!(
        traced.aval(embedding).shape,
        vec![config.vocab, config.hidden]
    );
    assert!(
        traced.eqns.iter().any(|eqn| {
            matches!(eqn.op, OpKind::Gather { axis: 0 }) && operand_value(eqn, 0) == Some(embedding)
        }),
        "the traced graph must still gather the table, or this test proves nothing"
    );

    let caps = poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan);
    let prepared = prepare_target_graph(&traced, Backend::SpirvVulkan, &caps);

    // The typed walk uploads the BF16 table as packed u32 lanes, so preparation must neither retype
    // nor widen it.
    let prepared_embedding = value_named(&prepared, EMBEDDING);
    assert_eq!(
        prepared.aval(prepared_embedding).dtype,
        DType::BF16,
        "a retyped embedding table is the 5.09 GB F32 copy this card exists to remove"
    );

    let gathers = prepared
        .eqns
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::DenseRowGather { .. }))
        .collect::<Vec<_>>();
    assert_eq!(gathers.len(), 1, "the embedding is the only row gather");
    assert_eq!(
        gathers[0].op,
        OpKind::DenseRowGather {
            source: DType::BF16
        }
    );
    assert_eq!(
        operand_value(gathers[0], 0),
        Some(prepared_embedding),
        "the row gather reads the checkpoint constant, not a copy of it"
    );
    assert!(
        !prepared
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Gather { .. })
                && operand_value(eqn, 0) == Some(prepared_embedding)),
        "no BF16 gather over the table may survive: SPIR-V cannot emit one"
    );

    // No equation casts a vocabulary-sized value in either direction (the widening pass's `Gather`
    // rule used to put a `Cast { to: BF16 }` on the retyped table, a 2.54 GB copy).
    for eqn in &prepared.eqns {
        if !matches!(eqn.op, OpKind::Cast { .. }) {
            continue;
        }
        let sizes = operand_value(eqn, 0)
            .into_iter()
            .chain(std::iter::once(eqn.out))
            .map(|value| prepared.aval(value).numel());
        for numel in sizes {
            assert!(
                numel < vocab_matrix,
                "{} over a vocabulary-sized value in the prepared graph",
                eqn.op.name()
            );
        }
    }
}

/// FR-009. PTX and ROCm keep the widening cast (a lowering there needs its own hardware receipt).
/// Their pipelines never run the fold, so the head stays a `MatMul`. The ROCm typed packed walk is a
/// separate pipeline (card 450 ROCm): it runs the fold and lowers the row gather on AmdGcn, so this
/// row prepares WITHOUT the fold - exactly what `RocmGraphExecutor::prepare_dtypes` does - and pins
/// the resident answer, not the typed walk's.
///
/// Card 1011: nothing retypes the embedding. A `Gather` consumer is not a packed reader, so on both
/// backends the table stays BF16 in its native two-byte lane (no 5.09 GB F32 copy at pinned dims, no
/// cast back to BF16 for the gather).
#[test]
fn ptx_and_rocm_preparation_is_untouched_by_this_card() {
    let config = Qwen35TextConfig::pinned();
    let traced = trace_qwen38_text_decode(&config, 0, 2).expect("the pinned config traces");
    for backend in [
        poot_target::Backend::Nvptx,
        poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
    ] {
        let caps = poot_test_util::device_caps::default_caps_for(backend);
        let prepared = poot_graph_plan::widen_mismatched_matmul_dtypes(&traced, backend, &caps);
        assert!(
            !prepared
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::DenseContraction { .. })),
            "no DenseContraction may reach {backend:?}, whose planner rejects it by name"
        );
        assert!(
            !prepared
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::DenseRowGather { .. })),
            "no DenseRowGather may reach {backend:?}, whose planner rejects it by name"
        );
        let embedding = value_named(&prepared, EMBEDDING);
        assert_eq!(prepared.aval(embedding).dtype, DType::BF16);
        assert!(
            !prepared.eqns.iter().any(|eqn| {
                matches!(eqn.op, OpKind::Cast { to: DType::BF16 })
                    && operand_value(eqn, 0) == Some(embedding)
            }),
            "{backend:?}: the BF16 embedding table is gathered as stored, not widened and cast back"
        );
    }
}
