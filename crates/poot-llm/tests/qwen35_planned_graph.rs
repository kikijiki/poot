//! Planned-graph evidence for Qwen3.5 exact (Qwen3.8-27B) on the one pipeline: the BF16 `[vocab, hidden]`
//! head and embedding sources on wgpu (cards 380 and 381), and the family's decode on every backend
//! that plans it, with the target's buffer cap honored (POOT-572 SC-006).
//!
//! The head is `MatMul(F32 activation, Transpose(BF16 weight)) -> F32`. Asserting on the traced graph
//! (card 376) cannot see what preparation does: `widen_mismatched_matmul_dtypes` used to put a
//! `Cast(BF16 -> F32)` on the head weight, a 5.09 GB F32 copy of a 2.54 GB matrix at the pinned dims.
//!
//! The family is the registry's `qwen35` fixture (`Model::trace`, no per-mode tracer): the planning
//! decisions under test (a BF16 contraction over the checkpoint constant, a row gather over the BF16
//! table, the device buffer cap) depend on the graph's structure and dtypes, not on its widths. The
//! real-dimensions receipt is POOT-572 SC-007, on a pod.
//!
//! These rows prepare and compile on the host: no device runs here.

use std::num::NonZeroUsize;

use poot_graph_ir::{Graph, OpKind, Operand, ValidationOutputs, ValueId};
use poot_graph_plan::{
    CompileError, CompileLimits, CompileOptions, LegalizeError, Submission, Target, WeightFormats,
    bind_packed_weights, compile, prepare_target_graph,
};
use poot_models::model::{KvLayout, LogitRows, Model, Phase, StepShape};
use poot_models::registry::{Fixture, Registry};
use poot_quant::weights::{WeightId, WeightRole};
use poot_target::{AmdArch, Backend};
use poot_tensor::DType;

/// The KV positions of every traced step.
const CAPACITY: usize = 8;

fn qwen35() -> (Box<dyn Model>, Fixture) {
    qwen35_stored_as("")
}

/// The family's fixture as `variant` stores it: `""` is the BF16 checkpoint, `"fp8"` the FP8 one (every
/// projection a packed `E4m3Block128` payload).
fn qwen35_stored_as(variant: &str) -> (Box<dyn Model>, Fixture) {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "qwen35")
        .expect("qwen35 is a registered family");
    let fixture = match variant {
        "" => (entry.fixture)(),
        variant => {
            let (_, make) = entry
                .variants
                .iter()
                .find(|(name, _)| *name == variant)
                .unwrap_or_else(|| panic!("qwen35 has no fixture variant {variant}"));
            make()
        }
    };
    let model = (entry.build)(&fixture.raw(), &fixture.store).expect("the fixture builds");
    (model, fixture)
}

fn shape(tokens: usize) -> StepShape {
    let n = |v| NonZeroUsize::new(v).unwrap();
    StepShape {
        rows: n(1),
        tokens: n(tokens),
        capacity: n(CAPACITY),
        kv: KvLayout::Contiguous,
        logits: LogitRows::Last,
    }
}

/// The family's decode graph, as the planner receives it.
fn decode_graph(model: &dyn Model) -> Graph<ValidationOutputs> {
    let traced = model.trace(Phase::Decode, shape(1)).expect("decode traces");
    bind_packed_weights(&traced, &WeightFormats::from_weight_map(model.weights()))
        .expect("the weights bind")
}

fn head() -> String {
    WeightId::model(WeightRole::Head).const_name()
}

fn embedding() -> String {
    WeightId::model(WeightRole::Embed).const_name()
}

/// `Operand` carries no `PartialEq`, so read an operand's value id instead of comparing operands.
fn operand_value(eqn: &poot_graph_ir::Eqn, position: usize) -> Option<ValueId> {
    match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(*value),
        _ => None,
    }
}

fn value_named<V: poot_graph_ir::ValidationChannel>(graph: &Graph<V>, name: &str) -> ValueId {
    graph
        .values
        .iter()
        .position(|value| value.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("the graph declares {name}"))
}

/// Card 380 SC-001 and SC-002. The prepared decode graph lowers the head as one `DenseContraction` over
/// the BF16 constant itself, and holds no widened copy of it.
#[test]
fn qwen35_wgpu_prepared_graph_holds_no_widened_lm_head_weight() {
    let (model, fixture) = qwen35();
    let traced = decode_graph(&*model);
    let (vocab, hidden) = (
        fixture.config["text_config"]["vocab_size"]
            .as_u64()
            .unwrap() as usize,
        fixture.config["text_config"]["hidden_size"]
            .as_u64()
            .unwrap() as usize,
    );

    // The traced head is the Card 376 shape: a BF16 constant, transposed, then used as a matmul weight.
    let head_name = head();
    let traced_head = value_named(&traced, &head_name);
    assert_eq!(traced.aval(traced_head).dtype, DType::BF16);
    assert_eq!(traced.aval(traced_head).shape, vec![vocab, hidden]);

    let prepared = prepare_target_graph(&traced, Backend::SpirvVulkan);

    // The typed walk uploads the checkpoint's BF16 matrix as packed u32 lanes, so preparation must
    // neither retype nor widen it.
    let prepared_head = value_named(&prepared, &head_name);
    assert_eq!(
        prepared.aval(prepared_head).dtype,
        DType::BF16,
        "a widened or retyped head weight is the F32 copy this card exists to remove"
    );

    // Exactly one contraction reads the head constant directly, with no transpose (a second full-size
    // buffer; card 258 tripped the iGPU watchdog with that dispatch shape).
    let readers = prepared
        .eqns
        .iter()
        .filter(|eqn| operand_value(eqn, 1) == Some(prepared_head))
        .collect::<Vec<_>>();
    assert_eq!(readers.len(), 1, "one equation reads the head weight");
    assert_eq!(
        readers[0].op,
        OpKind::DenseContraction {
            weight: DType::BF16
        },
        "the head is a BF16 contraction over the checkpoint constant, not a copy of it"
    );
    assert!(
        !prepared.eqns.iter().any(|eqn| {
            matches!(eqn.op, OpKind::Transpose { .. })
                && operand_value(eqn, 0) == Some(prepared_head)
        }),
        "the head weight must not be transposed on device"
    );

    // No equation widens the head, and no value is an F32 `[vocab, hidden]` matrix.
    for eqn in &prepared.eqns {
        let OpKind::Cast { to: DType::F32 } = eqn.op else {
            continue;
        };
        assert_ne!(
            operand_value(eqn, 0),
            Some(prepared_head),
            "Cast(BF16 -> F32) over the head weight"
        );
    }
    for value in &prepared.values {
        assert!(
            value.aval.dtype != DType::F32 || value.aval.shape != vec![vocab, hidden],
            "F32 vocabulary matrix {:?} in the prepared graph",
            value.name
        );
    }
}

/// Card 381 SC-001 and SC-002. The embedding lowers as one `DenseRowGather` over the BF16 constant:
/// gather and widening cast are one equation, so no copy of the table exists.
#[test]
fn qwen35_wgpu_prepared_graph_holds_no_widened_embedding_table() {
    let (model, fixture) = qwen35();
    let traced = decode_graph(&*model);
    let (vocab, hidden) = (
        fixture.config["text_config"]["vocab_size"]
            .as_u64()
            .unwrap() as usize,
        fixture.config["text_config"]["hidden_size"]
            .as_u64()
            .unwrap() as usize,
    );

    // The traced embedding is the Card 376 shape: a BF16 constant gathered by the token id, then widened.
    let name = embedding();
    let table = value_named(&traced, &name);
    assert_eq!(traced.aval(table).dtype, DType::BF16);
    assert_eq!(traced.aval(table).shape, vec![vocab, hidden]);
    assert!(
        traced.eqns.iter().any(|eqn| {
            matches!(eqn.op, OpKind::Gather { axis: 0 }) && operand_value(eqn, 0) == Some(table)
        }),
        "the traced graph must still gather the table, or this test proves nothing"
    );

    let prepared = prepare_target_graph(&traced, Backend::SpirvVulkan);

    // The typed walk uploads the BF16 table as packed u32 lanes, so preparation must neither retype
    // nor widen it.
    let prepared_table = value_named(&prepared, &name);
    assert_eq!(
        prepared.aval(prepared_table).dtype,
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
        Some(prepared_table),
        "the row gather reads the checkpoint constant, not a copy of it"
    );
    assert!(
        !prepared
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Gather { .. })
                && operand_value(eqn, 0) == Some(prepared_table)),
        "no BF16 gather over the table may survive: SPIR-V cannot emit one"
    );

    // No equation casts a vocabulary-sized value in either direction (the widening pass's `Gather`
    // rule used to put a `Cast { to: BF16 }` on the retyped table, a 2.54 GB copy).
    for eqn in &prepared.eqns {
        if !matches!(eqn.op, OpKind::Cast { .. }) {
            continue;
        }
        let shapes = operand_value(eqn, 0)
            .into_iter()
            .chain(std::iter::once(eqn.out))
            .map(|value| prepared.aval(value).shape.clone());
        for cast_shape in shapes {
            assert_ne!(
                cast_shape,
                vec![vocab, hidden],
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
    let (model, _) = qwen35();
    let traced = decode_graph(&*model);
    for backend in [Backend::Nvptx, Backend::AmdGcn(AmdArch::gfx1151())] {
        let prepared = poot_graph_plan::widen_mismatched_matmul_dtypes(&traced, backend);
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
        let table = value_named(&prepared, &embedding());
        assert_eq!(prepared.aval(table).dtype, DType::BF16);
        assert!(
            !prepared.eqns.iter().any(|eqn| {
                matches!(eqn.op, OpKind::Cast { to: DType::BF16 })
                    && operand_value(eqn, 0) == Some(table)
            }),
            "{backend:?}: the BF16 embedding table is gathered as stored, not widened and cast back"
        );
    }
}

fn options() -> CompileOptions {
    CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
}

fn target(backend: Backend) -> Target {
    Target {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    }
}

/// The three planner targets (the deleted exact route admitted the family on the CPU and wgpu only).
/// The targets that report a binding single-buffer cap: wgpu's and ROCm's. PTX reports `u64::MAX`
/// (a CUDA allocation is limited by the device's memory, not a per-buffer cap).
fn capped_backends() -> [Backend; 2] {
    [Backend::SpirvVulkan, Backend::AmdGcn(AmdArch::gfx1151())]
}

fn backends() -> [Backend; 3] {
    [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ]
}

/// POOT-572 SC-006 (R469-020): Qwen3.5 decode and prefill plan on SpirvVulkan, AmdGcn and Nvptx through
/// `compile`, over the BF16 checkpoint and the FP8 one (packed `E4m3Block128` projections). Mutation:
/// make `plan_eqn` refuse the KV-cache write (`DynamicUpdateSlice`) on AmdGcn; the AmdGcn rows name it and
/// go red.
#[test]
fn qwen35_decode_and_prefill_plan_on_every_backend() {
    for stored_as in ["", "fp8"] {
        let (model, _) = qwen35_stored_as(stored_as);
        let formats = WeightFormats::from_weight_map(model.weights());
        for backend in backends() {
            for (phase, tokens) in [(Phase::Decode, 1), (Phase::Prefill, 4)] {
                let traced = model.trace(phase, shape(tokens)).unwrap();
                let graph = bind_packed_weights(&traced, &formats).unwrap();
                compile(&graph, &target(backend), &options()).unwrap_or_else(|e| {
                    panic!("{stored_as:?} {phase:?} does not compile for {backend:?}: {e}")
                });
            }
        }
    }
}

/// The byte size of a named const of `graph`.
fn const_bytes<V: poot_graph_ir::ValidationChannel>(graph: &Graph<V>, id: ValueId) -> u64 {
    let aval = graph.aval(id);
    aval.shape.iter().product::<usize>() as u64 * aval.dtype.byte_size() as u64
}

/// The largest projection the decode graph reads that is not the embedding or the head, by bytes.
fn largest_projection(graph: &Graph<ValidationOutputs>) -> (String, u64, u64) {
    let skip = [embedding(), head()];
    graph
        .consts
        .iter()
        .filter_map(|&id| {
            let name = graph.meta(id).name.clone()?;
            let shape = &graph.aval(id).shape;
            (shape.len() == 2 && !skip.contains(&name)).then(|| {
                let row = shape[1] as u64 * graph.aval(id).dtype.byte_size() as u64;
                (name, const_bytes(graph, id), row)
            })
        })
        .max_by_key(|(_, bytes, _)| *bytes)
        .expect("the graph reads projections")
}

/// POOT-572 SC-006: a target whose single-buffer cap is below the largest non-embedding projection
/// still plans: `legalize` splits the projection into row chunks that each fit, and the compiled
/// program holds no constant over the cap. Mutation: make the cap check ignore the projection (`legalize`
/// reads `u64::MAX` as the limit for the split and the refusal); the compiled program keeps the
/// oversized projection and this row goes red.
#[test]
fn a_cap_below_the_largest_projection_splits_it_on_every_backend() {
    let (model, _) = qwen35();
    let graph = decode_graph(&*model);
    let (name, bytes, row) = largest_projection(&graph);
    // Below the projection, above everything the split cannot help (the carried conv window is the
    // largest unsplittable constant).
    let cap = bytes / 2;
    assert!(
        cap >= row,
        "{name}: a row ({row} bytes) must fit the cap {cap}"
    );
    for backend in capped_backends() {
        let mut target = target(backend);
        target.caps.max_buffer_bytes = cap;
        let program = compile(&graph, &target, &options()).unwrap_or_else(|e| {
            panic!("{backend:?}: a cap of {cap} bytes over {name} ({bytes}): {e}")
        });
        let planned = program.graph();
        // The split survives the pipeline's revert of rewrites the target cannot plan: the compiled
        // program holds the projection as row chunks, not as the original oversized constant.
        assert!(
            !planned
                .consts
                .iter()
                .any(|&id| planned.meta(id).name.as_deref() == Some(name.as_str())),
            "{backend:?}: {name} is still one constant after compile"
        );
        // The embedding gather is hosted off the device (`Slot::TokenEmbed`) where the target admits
        // it, so the cap binds every other constant: the projections, split, and the head, split.
        for &id in planned
            .consts
            .iter()
            .filter(|&&id| planned.meta(id).name.as_deref() != Some(&*embedding()))
        {
            assert!(
                const_bytes(planned, id) <= cap,
                "{backend:?}: {:?} is {} bytes, over the {cap}-byte cap",
                planned.meta(id).name,
                const_bytes(planned, id)
            );
        }
    }
}

/// POOT-572 SC-006: a cap below one row of a projection cannot be met by any split: `compile` refuses
/// it typed, before any device allocation, naming the constant and its size. Mutation: skip the refusal
/// (`refuse_oversized_constants` returns `Ok`); the program plans with the oversized constant and this
/// row goes red.
#[test]
fn a_cap_below_one_projection_row_is_a_typed_refusal_naming_the_constant() {
    let (model, _) = qwen35();
    let graph = decode_graph(&*model);
    let (_, _, row) = largest_projection(&graph);
    let cap = row - 1;
    for backend in capped_backends() {
        let mut target = target(backend);
        target.caps.max_buffer_bytes = cap;
        match compile(&graph, &target, &options()) {
            Err(CompileError::Legalize(error)) => match *error {
                LegalizeError::OversizedConstant { name, bytes, limit } => {
                    assert_eq!(limit, cap);
                    assert!(bytes > cap, "{name}: {bytes} bytes fits the cap {cap}");
                    assert!(
                        graph
                            .consts
                            .iter()
                            .any(|&id| graph.meta(id).name.as_deref() == Some(name.as_str())),
                        "{name} is a constant of the graph"
                    );
                }
                other => {
                    panic!("{backend:?}: expected the oversized-constant refusal, got {other}")
                }
            },
            Ok(_) => {
                panic!("{backend:?}: planned a program whose constants exceed the {cap}-byte cap")
            }
            Err(other) => panic!("{backend:?}: expected a typed legalize refusal, got {other}"),
        }
    }
}
