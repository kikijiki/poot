//! POOT-1175: what the Intel NPU lowering admits of a real small-Qwen graph.
//!
//! The graph is the shipped Qwen2 tracer's output at Qwen2.5-0.5B dimensions (hidden 896, 14 query heads over 2
//! key/value heads of 64, feed-forward 4864), one decoder layer, an 8-token prefill, synthetic weights. Every
//! equation is planned for the NPU on its own and the refusals are grouped by op and typed capability (the
//! first refusal alone would hide every other op class). The part the lowering admits runs on the device in
//! `poot-npu`'s `vpu_qwen` rows. The caps are SYNTHETIC (an NPU4 with a made-up compiler version): the table
//! depends on the target's arithmetic width, not on a measurement.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use poot_graph_plan::production_plan::Submission;
use poot_graph_plan::{Target, vpu_refusals};
use poot_models::model::{KvLayout, LogitRows, Phase, StepShape};
use poot_models::test_support::dense::{Dense, Family};
use poot_target::{NarrowingContract, Queried, VendorCompilerId, VpuCaps, VpuTargetId};

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
    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(TOKENS).unwrap(),
        capacity: NonZeroUsize::new(TOKENS).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::Last,
        hidden: None,
    };
    model
        .model
        .trace(
            Phase::Prefill,
            shape,
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
/// and the number of equations. Five groups of the 111 equations. Every op the block uses has an opset lowering
/// (POOT-1177), so what is left is one cause: the tracer emits F32 where the device computes FP16 and nothing
/// bounds narrowing it (POOT-1178), and the `cast` is a conversion through BF16 or an integer that changes the
/// value. The 49 equations the lowering admits are the ones that move data, the two `dynamic_update_slice` and
/// the two `iota` among them.
const UNCONTRACTED_REFUSALS: [(&str, usize); 5] = [
    (
        "binary                       NarrowingContract { dtype: F32 }",
        38,
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

/// The same block on a target that records the contract: every F32 arithmetic refusal is gone (the region is
/// narrowed at its boundary instead) and so is the additive attention-mask equation (`sub` by the literal
/// `1e9`), whose finite literal saturates to FP16's largest value under the contract (POOT-1180). Two
/// refusals remain and each is owned: the `mul` by `1e9` that scales the causal mask (a multiplicative literal
/// past FP16's range is never saturated and is refused as `NarrowingRange`, because `x * 65504` is not `x * 1e9`; the mask must be built with an
/// additive sentinel or a representable scale), and the `cast` of the I32 position range into the mask's
/// comparison, which FP16 holds exactly only up to 2048 and nothing the planner reads bounds.
const CONTRACTED_REFUSALS: [(&str, usize); 2] = [
    (
        "binary                       NarrowingRange { dtype: F32 }",
        1,
    ),
    (
        "cast                         IntegerToFloat { to: F16, exact_limit: 2048 }",
        1,
    ),
];

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
/// Mutation: in `opset_op` delete the `OpKind::Iota` arm; the two `iota` equations come back as a refusal and the
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
        Target::Vpu {
            target: caps.target,
            caps,
        },
    );
    println!("production compile of the traced block: {outcome}");
    assert!(
        outcome.contains("Refused")
            && outcome.contains("op: Binary(Mul)")
            && outcome.contains("missing: NarrowingContract { dtype: F32 }"),
        "the traced block compiles differently than pinned: {outcome}"
    );
    // With the contract the production entry stops at the first equation no contract covers, the
    // integer `cast` (the multiplicative `1e9` mask scale refuses after it).
    let contracted_outcome = compile_outcome(
        graph,
        Target::Vpu {
            target: caps.target,
            caps: narrowing_caps(),
        },
    );
    println!("production compile with the contract: {contracted_outcome}");
    assert!(
        contracted_outcome.contains("Refused")
            && contracted_outcome.contains("op: Cast")
            && contracted_outcome
                .contains("missing: IntegerToFloat { to: F16, exact_limit: 2048 }"),
        "the contracted block compiles differently than pinned: {contracted_outcome}"
    );
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
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    ) {
        Ok(_) => "compiled".to_string(),
        Err(error) => format!("{error:?}"),
    }
}
