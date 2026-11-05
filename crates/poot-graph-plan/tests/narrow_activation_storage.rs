//! POOT-1448: a value whose every reader narrows it is stored narrow, rounded once by the region that computes it
//! (`store_narrowed_activations`, `receipts/1448-activation-narrowed-once/design.md`).
//!
//! Every row drives the production `compile` on a transformer block (a norm, three projections of it, a gated MLP
//! and the residual adds) and reads the program it returns.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::op::BinOp;
use poot_graph_ir::ops::rmsnorm;
use poot_graph_ir::ops::silu;
use poot_graph_ir::{Builder, Graph, MatmulPrecision, OpKind, TensorType, ValueId};
use poot_graph_plan::{CompileLimits, CompileOptions, Program, Target, compile};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried, SubgroupSupport};
use poot_tensor::{DType, HostTensor};

const M: usize = 160;
const K: usize = 64;
const F: usize = 96;

/// ROCm caps with the RDNA3 WMMA lowering, as the ROCm runtime reports them on gfx11: native two-byte elements.
fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

/// The same device with no native two-byte element (a SPIR-V device's buffers hold BF16 and F16 only as packed
/// words): the planner has nowhere to store a narrowed value.
fn rocm_wmma_without_two_byte_elements() -> Target {
    let Target::Gpu { backend, mut caps } = rocm_wmma() else {
        unreachable!("a GPU target")
    };
    caps.native_two_byte_floats = false;
    Target::Gpu { backend, caps }
}

/// The block, and its constants by id (every input is a constant, so the oracle evaluates it as compiled):
/// `h = rmsnorm(x)`, `out = x + h Wq^T + (silu(h Wg^T) * h Wu^T) Wd^T`, BF16 checkpoint-order weights.
/// `extra_reader` also adds `h` itself to the output, an f32 reader of the normed value.
fn block(extra_reader: bool) -> (Graph, HashMap<ValueId, Value>) {
    blocks(1, extra_reader)
}

/// `layers` blocks, each reading the previous one's output as its `x`.
fn blocks(layers: usize, extra_reader: bool) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::for_tests();
    let mut inputs = HashMap::new();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    inputs.insert(
        x.id(),
        Value::Host(HostTensor::f32(vec![M, K], values(M * K, 1))),
    );
    let mut out = x;
    for layer in 0..layers {
        let x = out;
        let norm = b
            .constant(&format!("norm{layer}"), TensorType::f32(vec![K]))
            .unwrap();
        inputs.insert(
            norm.id(),
            Value::Host(HostTensor::f32(vec![K], values(K, 2 + 10 * layer as u64))),
        );
        let mut weight = |name: &str, rows: usize, cols: usize, seed: u64| {
            let (name, seed) = (format!("{name}{layer}"), seed + 10 * layer as u64);
            let name = name.as_str();
            let w = b
                .constant(name, TensorType::new(vec![rows, cols], DType::BF16))
                .unwrap();
            let bits = values(rows * cols, seed)
                .into_iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect();
            inputs.insert(
                w.id(),
                Value::Host(HostTensor::bf16(vec![rows, cols], bits)),
            );
            b.transpose(w, vec![1, 0]).unwrap()
        };
        let (wq, wg, wu, wd) = (
            weight("wq", K, K, 3),
            weight("wg", F, K, 4),
            weight("wu", F, K, 5),
            weight("wd", K, F, 6),
        );
        let h = rmsnorm(&b, x, norm, 1e-6).unwrap();
        let q = b.matmul(h, wq).unwrap();
        let gate = silu(&b, b.matmul(h, wg).unwrap()).unwrap();
        let act = b
            .binary(BinOp::Mul, gate, b.matmul(h, wu).unwrap())
            .unwrap();
        let down = b.matmul(act, wd).unwrap();
        out = b
            .binary(BinOp::Add, b.binary(BinOp::Add, x, q).unwrap(), down)
            .unwrap();
        if extra_reader {
            out = b.binary(BinOp::Add, out, h).unwrap();
        }
    }
    (b.finish(out).unwrap(), inputs)
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn compiled(g: &Graph, target: &Target, policy: MatmulPrecision) -> Program {
    compile(
        g,
        target,
        &CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy),
    )
    .expect("the block compiles")
}

/// The op name and dtype of every value the program computes in a two-byte float.
fn narrow_values(program: &Program) -> Vec<(String, DType)> {
    program
        .planned()
        .filter_map(|(eqn, _)| {
            let dtype = program.graph().aval(eqn.out).dtype;
            matches!(dtype, DType::BF16 | DType::F16)
                .then(|| (eqn.op.kind_name().to_string(), dtype))
        })
        .collect()
}

/// The number of kernel launches the program's plans hold.
fn dispatches(program: &Program) -> usize {
    program
        .planned()
        .map(|(_, plan)| plan.dispatches().len())
        .sum()
}

fn output_bits(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Vec<u32> {
    eval(g, inputs, EvalOptions::default())
        .expect("the graph evaluates")
        .output
        .into_host()
        .expect("a dense graph")
        .as_f32()
        .expect("an f32 output")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Under the narrow policy on a device with native two-byte elements, the norm's output (read by the three
/// projections) and the gated activation (read by the down projection) are stored bf16, written by the row region
/// and the up projection's epilogue that compute them; the residual stream, which f32 arithmetic reads, stays f32.
/// The program launches what the same program does with nothing stored narrow (no pass, no copy). Mutation: in
/// `store_narrowed_activations` commit a value without probing (`if true`); the device without two-byte elements
/// then narrows too and the row goes red.
#[test]
fn a_value_every_reader_narrows_is_stored_narrow_by_its_producer() {
    let (g, _) = block(false);
    let narrowed = compiled(&g, &rocm_wmma(), MatmulPrecision::Narrow);
    assert_eq!(
        narrow_values(&narrowed),
        vec![
            ("fused_row".to_string(), DType::BF16),
            ("contraction_epilogue".to_string(), DType::BF16),
        ],
        "the norm and the gated activation are stored narrow, nothing else"
    );
    assert!(
        narrowed
            .passes()
            .any(|pass| pass == "store_narrowed_activations")
    );
    let unnarrowed = compiled(
        &g,
        &rocm_wmma_without_two_byte_elements(),
        MatmulPrecision::Narrow,
    );
    assert_eq!(
        narrow_values(&unnarrowed),
        Vec::new(),
        "no buffer to store it in"
    );
    assert_eq!(
        (narrowed.graph().eqns().len(), dispatches(&narrowed)),
        (unnarrowed.graph().eqns().len(), dispatches(&unnarrowed)),
        "no equation and no dispatch is added"
    );
}

/// A value one f32 reader reads is not stored narrow, whatever its other readers do. Mutation: in
/// `narrowing_dtype` accept the readers when any one narrows (`dtypes.any(..)`); the norm is narrowed and the
/// f32 add that reads it no longer types.
#[test]
fn a_value_an_f32_reader_reads_stays_f32() {
    let (g, _) = block(true);
    let program = compiled(&g, &rocm_wmma(), MatmulPrecision::Narrow);
    assert_eq!(
        narrow_values(&program),
        vec![("contraction_epilogue".to_string(), DType::BF16)],
        "only the gated activation, which the down projection alone reads"
    );
}

/// The narrowed program computes what the declared semantics compute, bit for bit: the oracle of the compiled
/// graph, whose values are stored narrow, equals the oracle of the traced graph under the same operand precision,
/// whose values are f32 and rounded by each reader (the producer's `Cast` and the readers' narrowing are one
/// rounding). Mutation: in `OperandPrecision::stored_activation` map `Narrow(Bf16)` to `F16`; the readers round an
/// f16 value to bf16 again and the bits differ.
#[test]
fn the_narrowed_program_evaluates_bit_for_bit_as_the_declared_one() {
    let (g, inputs) = block(false);
    let target = rocm_wmma();
    let program = compiled(&g, &target, MatmulPrecision::Narrow);
    assert!(!narrow_values(&program).is_empty(), "the program narrows");
    let declared =
        poot_graph_plan::with_operand_precision(&g, &target, MatmulPrecision::Narrow).unwrap();
    assert_eq!(
        output_bits(program.graph(), &inputs),
        output_bits(&declared, &inputs)
    );
}

/// Under the exact policy no reader narrows, so nothing is stored narrow and the pass leaves no trace: the exact
/// program is the one it was before the pass existed (its plan digests do not move, `plan_summary`). Mutation: in
/// `narrowing_dtype` accept an exact reader (`stored_activation()` of `Narrow(Bf16)` for every precision); the exact
/// block narrows.
#[test]
fn an_exact_program_stores_nothing_narrow() {
    let (g, _) = block(false);
    let program = compiled(&g, &rocm_wmma(), MatmulPrecision::Exact);
    assert_eq!(narrow_values(&program), Vec::new());
    assert!(
        !program
            .passes()
            .any(|pass| pass == "store_narrowed_activations")
    );
    assert!(
        program
            .graph()
            .eqns()
            .iter()
            .all(|eqn| !matches!(eqn.op, OpKind::Cast { .. }))
    );
}

/// POOT-1450: a value stored narrow shares an untyped arena slot of bytes with f32 values whose lifetimes it
/// does not overlap, so narrowing never costs the f32 values the slots they shared before; every non-export
/// slot is untyped. Mutation: color the arena per storage class again (group `plan_buffers`' keys by the value's
/// storage); no bf16 value shares a slot with an f32 one and the row goes red.
#[test]
fn a_value_stored_narrow_shares_its_arena_slot_with_f32_values() {
    let (g, _) = blocks(2, false);
    let program = compiled(&g, &rocm_wmma(), MatmulPrecision::Narrow);
    let plan = program.buffer_plan();
    let slot_storage: HashMap<_, _> = plan
        .slots()
        .map(|(slot, storage, _)| (slot, storage))
        .collect();
    let mut occupants: HashMap<_, Vec<DType>> = HashMap::new();
    for (eqn, _) in program.planned() {
        if let Some(slot) = plan.slot(eqn.out) {
            occupants
                .entry(slot)
                .or_default()
                .push(program.graph().aval(eqn.out).dtype);
        }
    }
    let mixed = occupants
        .iter()
        .filter(|(_, dtypes)| dtypes.contains(&DType::BF16) && dtypes.contains(&DType::F32))
        .count();
    assert!(
        mixed > 0,
        "a bf16 value and an f32 value share a slot: {occupants:?}"
    );
    for (slot, dtypes) in &occupants {
        if !plan.slot_outlives_stage(*slot) {
            assert_eq!(
                slot_storage[slot],
                poot_target::BufferStorage::untyped(),
                "slot {slot:?} of {dtypes:?}"
            );
        }
    }
}
