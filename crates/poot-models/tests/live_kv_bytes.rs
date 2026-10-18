//! POOT-1209: a decode step reads only the live part of the KV cache.
//!
//! Qwen3-0.6B's dimensions, one decode token at position 15 over a 4096-position cache, compiled for every
//! reference backend through `compile`. Each attention launch's K and V bytes are counted from the program:
//! a launch bounded by capacity streams every stored position; a launch bounded by a live length streams
//! the positions that length names, which the test evaluates on the CPU oracle from the position input
//! the launch's own length operand derives from. The row fails if decode reads the whole cache (R2-A3-001:
//! 0.94 GB per token at this capacity).

use std::collections::{HashMap, HashSet};

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Graph, GraphEdit, KvBound, OpKind, Operand, Slot, ValueId};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Program, Target, compile, production_plan::Submission,
};
use poot_models::model::{LogitRows, Phase};
use poot_models::test_support::dense::{Dense, Family, step};
use poot_target::{AmdArch, Backend};
use poot_tensor::{DType, HostTensor};
use poot_test_util::device_caps::default_caps_for;

const CAPACITY: usize = 4096;
const POS: i32 = 15;

/// Qwen3-0.6B's dimensions (no bias, per-head Q/K norm, explicit head dimension).
fn qwen3_0_6b() -> Dense {
    Dense::new(Family::Qwen3)
        .vocab(151936)
        .dims(1024, 3072, 28)
        .heads(16, 8)
        .head_dim(128)
        .max_positions(40960)
}

fn decode_graph() -> Graph {
    qwen3_0_6b()
        .zeroed_model(DType::BF16)
        .model
        .trace(
            Phase::Decode,
            step(1, 1, CAPACITY, LogitRows::Last),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("Qwen3 traces a decode step")
}

fn compiled(g: &Graph, backend: Backend) -> Program {
    let target = Target::Gpu {
        backend,
        caps: default_caps_for(backend),
    };
    let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    compile(g, &target, &options).expect("the decode graph compiles")
}

/// The value `len` of `g` when the position input is `POS`, evaluated on the CPU oracle over only the
/// equations `len` derives from.
fn live_length(g: &Graph, len: ValueId) -> i32 {
    let mut needed: HashSet<ValueId> = HashSet::from([len]);
    let mut keep = Vec::new();
    for eqn in g.eqns().iter().rev() {
        if needed.contains(&eqn.out) {
            needed.extend(eqn.inputs.iter().filter_map(|operand| match operand {
                Operand::Value(id) => Some(*id),
                Operand::Lit(_) => None,
            }));
            keep.push(eqn.clone());
        }
    }
    keep.reverse();
    let (pos, _) = g
        .slots()
        .iter()
        .find(|(_, slot)| *slot == Slot::Pos)
        .expect("a decode graph has a position input");
    assert!(
        needed.contains(pos)
            && needed
                .iter()
                .all(|id| g.slots().iter().any(|(s, _)| s == id)
                    || keep.iter().any(|e| e.out == *id)),
        "the live length derives from the position input alone"
    );
    let mut draft = GraphEdit::new();
    *draft.values_mut() = g.values().clone();
    *draft.eqns_mut() = keep;
    *draft.inputs_mut() = g
        .inputs()
        .iter()
        .copied()
        .filter(|id| needed.contains(id))
        .collect();
    *draft.slots_mut() = g
        .slots()
        .iter()
        .copied()
        .filter(|(id, _)| needed.contains(id))
        .collect();
    *draft.consts_mut() = g
        .consts()
        .iter()
        .copied()
        .filter(|id| needed.contains(id))
        .collect();
    draft.set_result(len);
    let pruned = draft.commit().expect("the length slice is a graph");
    let inputs = HashMap::from([(*pos, Value::from(HostTensor::i32(vec![1, 1], vec![POS])))]);
    let out = eval(&pruned, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("the length evaluates")
        .output
        .into_host()
        .expect("a dense length");
    assert_eq!(out.as_i32().expect("an I32 length").len(), 1);
    out.as_i32().unwrap()[0]
}

/// K and V bytes the attention launches of `program` read in one step.
fn attention_kv_bytes(program: &Program) -> (usize, usize) {
    let g = program.graph();
    let (mut launches, mut bytes) = (0, 0);
    for eqn in g.eqns() {
        let OpKind::Attention(attrs) = &eqn.op else {
            continue;
        };
        let operand = |i: usize| match eqn.inputs[i] {
            Operand::Value(id) => g.aval(id),
            Operand::Lit(_) => unreachable!("attention operands are values"),
        };
        let (k, v) = (operand(1), operand(2));
        let positions = match attrs.bound {
            KvBound::Capacity => k.shape[2],
            KvBound::Live => {
                let Some(Operand::Value(len)) = eqn.inputs.last() else {
                    unreachable!("a live bound reads its length operand")
                };
                live_length(g, *len) as usize
            }
        };
        let per_position =
            |t: &poot_graph_ir::TensorType| t.shape[1] * t.shape[3] * t.dtype.byte_size();
        launches += 1;
        bytes += positions * (per_position(k) + per_position(v));
    }
    (launches, bytes)
}

/// SC-001: decode at position 15 over a 4096-position cache reads at most twice the K/V bytes of 16
/// positions, in every layer, on every backend.
#[test]
fn live_short_decode_reads_the_live_positions_not_the_cache() {
    let g = decode_graph();
    for (name, backend) in [
        ("wgpu", Backend::SpirvVulkan),
        ("ROCm", Backend::AmdGcn(AmdArch::gfx1151())),
        ("PTX", Backend::Nvptx),
    ] {
        let program = compiled(&g, backend);
        let (launches, bytes) = attention_kv_bytes(&program);
        assert_eq!(launches, 28, "{name}: one attention launch per layer");
        // Per layer: 8 kv heads x 128 dims x (K + V) x 2 bytes, per position.
        let sixteen = 28 * 16 * 8 * 128 * 2 * 2;
        assert!(
            bytes <= 2 * sixteen,
            "{name}: decode at pos {POS} reads {bytes} K/V bytes; 16 positions are {sixteen} \
             (the whole {CAPACITY}-position cache is {})",
            28 * CAPACITY * 8 * 128 * 2 * 2
        );
    }
}
