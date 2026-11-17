//! POOT-1175: what the Intel NPU lowering admits of a real small-Qwen graph.
//!
//! The graph is the shipped Qwen2 tracer's output at Qwen2.5-0.5B dimensions (hidden 896, 14 query heads over 2
//! key/value heads of 64, feed-forward 4864), one decoder layer, an 8-token prefill, synthetic weights. Every
//! equation is planned for the NPU on its own and the refusals are grouped by op and typed capability (the
//! first refusal alone would hide every other op class). The part the lowering admits runs on the device in
//! `poot-npu`'s `vpu_qwen` rows. The caps are SYNTHETIC (an NPU4 with a made-up compiler version): the table
//! depends on the target's arithmetic width, not on a measurement.

use poot_graph_plan::staged::StagedCompileError;
use poot_graph_plan::{CompileError, NpuTarget, PlanError, Target};
use poot_models::model::{KvDtype, StepShape, StepSpec};
use poot_models::registry::{RawConfig, Registry};
use poot_models::test_support::dense::{Dense, Family};
use poot_npu::vpu_refusals;
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_target::{NarrowingContract, Queried, StepWidths, VendorCompilerId, VpuCaps, VpuTargetId};
use poot_tensor::DType;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

const HIDDEN: usize = 896;
const INTER: usize = 4864;
const HEADS: usize = 14;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const TOKENS: usize = 8;

fn caps() -> VpuCaps {
    VpuCaps {
        target: VpuTargetId::Npu4,
        compiler: Queried::Known(VendorCompilerId { major: 7, minor: 5 }),
        memory_budget_bytes: Queried::Unknown,
        narrowing: None,
        step_widths: StepWidths::new([TOKENS]).expect("8 is a power of two"),
    }
}

/// [`caps`] with the numerical contract F32 values are narrowed to FP16 under. SYNTHETIC like the rest: the
/// planner reads only that a contract exists; the recorded bound the device is held to is `poot-npu`'s.
fn narrowing_caps() -> VpuCaps {
    VpuCaps {
        narrowing: Some(NarrowingContract {
            max_relative_error_ppm: 10_000,
            max_magnitude: 65_504,
        }),
        ..caps()
    }
}

/// The Qwen2 tracer's prefill graph at Qwen2.5-0.5B dimensions, one layer, a vocabulary of 64.
fn traced_block() -> poot_graph_ir::Graph {
    let dense = Dense::new(Family::Qwen2)
        .vocab(64)
        .dims(HIDDEN, INTER, 1)
        .heads(HEADS, KV_HEADS)
        .head_dim(HEAD_DIM)
        .max_positions(64);
    let model = dense.f32_model();
    let shape = StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::new(TOKENS).unwrap(),
        NonZeroUsize::new(TOKENS).unwrap(),
    );
    model
        .model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// The traced block's refusals for the NPU, one line per (op, capability) with the equation count.
///
/// Without a recorded narrowing contract the F32 activations the tracer emits are refused as
/// `NarrowingContract { F32 }` on every arithmetic op the opset has; an op the opset lacks would be refused by
/// op whatever its dtype. So the table separates "the device computes in FP16 and the tracer emits F32" (one
/// cause, many ops) from the ops with no lowering at all.
fn refusal_table(
    graph: &poot_graph_ir::Graph,
    caps: &poot_target::VpuCaps,
) -> BTreeMap<String, usize> {
    let mut table = BTreeMap::new();
    for refusal in vpu_refusals(graph, caps).unwrap() {
        *table
            .entry(format!(
                "{:<28} {:?}",
                refusal.op.kind_name(),
                refusal.missing
            ))
            .or_default() += 1;
    }
    table
}

/// The traced block's refusals on a target that records no narrowing contract, pinned: (op, missing capability)
/// and the number of equations. Five groups of the 105 equations. Every op the block uses has an opset lowering
/// (POOT-1177), so what is left is one cause: the tracer emits F32 where the device computes FP16 and nothing
/// bounds narrowing it (POOT-1178), and the `cast` is a conversion through BF16 or an integer that changes the
/// value. The 45 equations the lowering admits are the ones that move data, the two `dynamic_update_slice` and
/// the `iota` among them.
const UNCONTRACTED_REFUSALS: [(&str, usize); 5] = [
    (
        "binary                       NarrowingContract { dtype: F32 }",
        36,
    ),
    ("cast                         DtypeLowering", 1),
    (
        "matmul                       NarrowingContract { dtype: F32 }",
        10,
    ),
    (
        "reduce                       NarrowingContract { dtype: F32 }",
        5,
    ),
    (
        "unary                        NarrowingContract { dtype: F32 }",
        8,
    ),
];

/// The same block on a target that records the contract: every refusal is gone. The F32 arithmetic is narrowed
/// at the region boundary, the additive attention-mask equation (`sub` by the literal `1e9`) saturates to
/// FP16's largest value (POOT-1180), the multiplicative one (`mul` by `1e9` of a factor comparisons alone fix
/// to -1, 0 or 1) saturates the same way because its product is exactly 0 or the sentinel (POOT-1278), and the
/// `cast` of the position into the mask's comparison is exact because the tracer declares the position below
/// the cache capacity (`IndexRange::below`), which FP16 holds exactly up to 2048.
const CONTRACTED_REFUSALS: [(&str, usize); 0] = [];

fn pinned(rows: &[(&str, usize)]) -> BTreeMap<String, usize> {
    rows.iter()
        .map(|&(line, count)| (line.to_string(), count))
        .collect()
}

/// Every equation of the Qwen2 tracer's block is planned for the NPU on its own and the refusals are listed
/// by op and typed capability, for a target without and with a narrowing contract. Then the block goes through
/// the production entry, which the planner refuses with a typed error naming the missing contract.
///
/// Mutation: in `compile` run `fuse_index_casts` for every target; the outcome is the untyped
/// `BinaryValueDtypeMismatch` edit error and the compile-outcome row fails.
///
/// Mutation: in `opset_op` delete the `OpKind::Iota` arm; the `iota` equation comes back as a refusal and the
/// table no longer matches.
///
/// Mutation: drop the `narrowed` clamp in `arithmetic_scalar` (poot-graph-plan); the `sub` by `1e9` comes
/// back as a `binary DtypeLowering` refusal and the contracted table no longer matches.
///
/// Mutation: make `computes` (poot-graph-plan) ignore `caps.narrowing`; the contracted table loses its F32
/// arithmetic admission or the uncontracted one its refusals, and a table no longer matches.
#[test]
fn the_traced_qwen2_block_is_refused_op_class_by_op_class() {
    let caps = caps();
    let graph = traced_block();
    let table = refusal_table(&graph, &caps);
    println!(
        "traced Qwen2.5-0.5B block, {} equations:",
        graph.eqns().len()
    );
    for (line, count) in &table {
        println!("{count:4}  {line}");
    }
    assert_eq!(table, pinned(&UNCONTRACTED_REFUSALS));
    let contracted = refusal_table(&graph, &narrowing_caps());
    println!("with the narrowing contract:");
    for (line, count) in &contracted {
        println!("{count:4}  {line}");
    }
    assert_eq!(contracted, pinned(&CONTRACTED_REFUSALS));
    // The pass pipeline leaves the graph valid for a non-GPU target (`fuse_index_casts` is GPU-only), so the block
    // reaches the planner and is refused with its first typed capability gap (POOT-1176).
    let outcome = compile_outcome(
        graph.clone(),
        Target::Npu(NpuTarget::Vpu {
            target: caps.target,
            caps,
        }),
    );
    println!("production compile of the traced block: {outcome}");
    assert!(
        outcome.contains("Refused")
            && outcome.contains("op: Binary(Mul)")
            && outcome.contains("missing: NarrowingContract { dtype: F32 }"),
        "the traced block compiles differently than pinned: {outcome}"
    );
    // With the contract the whole block compiles (the width check is a model step's, and this block is one).
    let contracted_outcome = compile_outcome(
        graph,
        Target::Npu(NpuTarget::Vpu {
            target: caps.target,
            caps: narrowing_caps(),
        }),
    );
    println!("production compile with the contract: {contracted_outcome}");
    assert_eq!(contracted_outcome, "compiled");
}

/// What `compile_staged` does with `graph` for `target`: the error it returns, in words.
fn compile_outcome(graph: poot_graph_ir::Graph, target: Target) -> String {
    use poot_graph_plan::multi_device::topology::DeviceId;
    use poot_graph_plan::staged::{
        DevicePlacement, ExpertPlacement, Partition, TargetSet, compile_staged,
    };
    use poot_graph_plan::{CompileLimits, CompileOptions};
    match compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            npu: Some(poot_npu::npu_lowering()),
            ..CompileOptions::new(CompileLimits::STANDARD)
        },
    ) {
        Ok(_) => "compiled".to_string(),
        Err(error) => format!("{error:?}"),
    }
}

/// The chunk width of the wide step the NPU prepares, in tokens of one row.
const WIDE: usize = 128;
/// The cache capacity the wide step's KV states are declared with: two chunks of a prompt.
const CAPACITY: usize = 256;

/// Qwen2.5-0.5B's real dimensions: 24 layers, the vocabulary of 151936, untied head.
fn qwen2_5_0_5b() -> Dense {
    Dense::new(Family::Qwen2)
        .vocab(151_936)
        .dims(HIDDEN, INTER, 24)
        .heads(HEADS, KV_HEADS)
        .head_dim(HEAD_DIM)
        .max_positions(32_768)
}

/// The checkpoint's own weight formats, as the official `qwen2.5-0.5b-instruct-fp16.gguf` stores them: every matrix
/// F16 (the projections, the embedding and the head), every vector (norms, the Q/K/V biases) F32. F16 binds as
/// stored on the NPU (design npu-targets-1093 4.6): nothing is re-encoded. The values are zeros; the plan reads
/// only the graph's structure and the stored formats.
fn f16_gguf_store(dense: &Dense) -> WeightStore {
    let mut store = WeightStore::builder();
    for (key, shape) in dense.tensors() {
        let numel: usize = shape.iter().product();
        let (dtype, width) = if shape.len() == 1 {
            (DType::F32, 4)
        } else {
            (DType::F16, 2)
        };
        let weight =
            DenseWeight::try_new(dtype, shape, StoredBytes::from(vec![0u8; width * numel]))
                .unwrap();
        store.insert(key, WeightEntry::Dense(weight)).unwrap();
    }
    store.build()
}

/// Qwen2.5-0.5B's step graph of `tokens` tokens against [`CAPACITY`] cache positions, over the F16 checkpoint's
/// formats, with an F16 KV cache.
fn wide_step_graph(tokens: usize) -> poot_graph_ir::Graph {
    let dense = qwen2_5_0_5b();
    let store = f16_gguf_store(&dense);
    let config = dense.config();
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    let model = Registry::builtin()
        .unwrap()
        .build(&raw, &store)
        .unwrap()
        .model;
    let shape = StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::new(tokens).unwrap(),
        NonZeroUsize::new(CAPACITY).unwrap(),
    )
    .with_kv_dtype(KvDtype::F16);
    model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// The wide step's target: the operator's narrowing contract and the widths it configured.
fn wide_target(widths: &[usize]) -> Target {
    let caps = VpuCaps {
        step_widths: StepWidths::new(widths.iter().copied()).unwrap(),
        ..narrowing_caps()
    };
    Target::Npu(NpuTarget::Vpu {
        target: caps.target,
        caps,
    })
}

/// POOT-1278 SC-001: the step graph of Qwen2.5-0.5B at its real dimensions (24 layers, vocabulary 151936, the
/// F16 checkpoint's own weight formats), 128 tokens of one row against a 256-position cache, plans on the Intel
/// NPU under the operator's `NarrowingContract` with no refusal: one graph region, whole-graph. Every equation of
/// the wide step is lowered, among them the two `DynamicUpdateSlice` per layer that write the KV cache at the
/// position the step carries as data.
///
/// Mutation: in `opset_op` (poot-npu `vpu_lowering`) delete the `OpKind::DynamicUpdateSlice` arm; the plan is
/// refused at the first KV write with `VpuOpset` and this row goes red.
///
/// Mutation: in `lower_eqn` drop the exactness check of the integer-to-float `Cast` (`exact` is always false); the
/// position cast of the causal mask is refused with `IntegerToFloat` and this row goes red.
///
/// Mutation: in `narrow_to_device_precision` make `mask_scale` always false; the mask's `mul` by `1e9` is refused
/// as `NarrowingRange` and this row goes red.
#[test]
fn the_qwen2_5_0_5b_wide_step_plans_on_the_npu_with_no_refusal() {
    let graph = wide_step_graph(WIDE);
    println!(
        "Qwen2.5-0.5B step of {WIDE} tokens: {} equations",
        graph.eqns().len()
    );
    let staged = compile_staged_for(graph, wide_target(&[WIDE])).expect("the wide step plans");
    let stages: Vec<_> = staged.stages().collect();
    assert_eq!(stages.len(), 1, "one program");
    let program = stages[0].2;
    assert_eq!(program.regions().len(), 1, "one whole-graph region");
    let updates = program
        .graph()
        .eqns()
        .iter()
        .filter(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::DynamicUpdateSlice { .. }))
        .count();
    assert_eq!(
        updates,
        2 * 24,
        "every layer writes its K and V cache rows in the region"
    );
}

/// POOT-1278 SC-003 (the plan's half): the widths the operator configured are the only widths a model step plans
/// for. The same Qwen2.5-0.5B step of 128 tokens on a device configured for 64 and 256 is a typed
/// `StepWidth` refusal naming both, before one equation is lowered; a device configured for none refuses it too.
///
/// Mutation: in `plan_vpu_region` skip the width check; the 64/256 target plans the 128-token step and this row
/// goes red.
#[test]
fn a_step_of_a_width_the_operator_did_not_configure_is_a_typed_refusal() {
    for widths in [&[64, 256][..], &[][..]] {
        let error = compile_staged_for(wide_step_graph(WIDE), wide_target(widths))
            .err()
            .unwrap_or_else(|| panic!("{widths:?}: the 128-token step planned"));
        let configured = StepWidths::new(widths.iter().copied()).unwrap();
        let StagedCompileError::Compile(compile) = &error else {
            panic!("{widths:?}: {error:?}")
        };
        let CompileError::Plan(plan) = compile else {
            panic!("{widths:?}: {error:?}")
        };
        assert!(
            matches!(
                &**plan,
                PlanError::StepWidth { width: WIDE, configured: got } if *got == configured
            ),
            "{widths:?}: {error:?}"
        );
    }
}

/// `graph` compiled for `target` through the production entry, over one device.
fn compile_staged_for(
    graph: poot_graph_ir::Graph,
    target: Target,
) -> Result<poot_graph_plan::staged::StagedProgram, StagedCompileError> {
    use poot_graph_plan::multi_device::topology::DeviceId;
    use poot_graph_plan::staged::{
        DevicePlacement, ExpertPlacement, Partition, TargetSet, compile_staged,
    };
    use poot_graph_plan::{CompileLimits, CompileOptions};
    compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            npu: Some(poot_npu::npu_lowering()),
            ..CompileOptions::new(CompileLimits::STANDARD)
        },
    )
}
