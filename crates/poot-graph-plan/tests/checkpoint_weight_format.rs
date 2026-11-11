//! POOT-1535 (design POOT-1534): a weight is resident in its checkpoint format in every program of a driver, and
//! precision belongs to the contraction that reads it. `compile` never retypes a weight; a contraction the planner
//! narrows to f16 rounds the staged weight in its fragment staging, and declares the weight finite in f16
//! ([`ValueBound::FiniteIn`]) for the binder to check. Every row drives the production `compile`.

use poot_graph_ir::{
    Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, Operand, OperandPrecision, Slot,
    TensorType, ValueBound, ValueId,
};
use poot_graph_plan::{CompileLimits, CompileOptions, Program, Target, compile};
use poot_target::{
    AmdArch, Backend, BufferStorage, DeviceCaps, MatrixUnits, Queried, SubgroupSupport,
};
use poot_tensor::DType;

const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);

/// wgpu over RADV: an opaque f16 cooperative matrix, one wave per workgroup, where nothing narrows.
fn radv() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

/// gfx1151: the RDNA3 matrix unit, where a prefill-sized projection narrows to f16.
fn gfx1151() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.register_file = AmdArch::gfx1151().register_file();
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

fn compiled(g: &Graph, target: &Target, policy: MatmulPrecision) -> Program {
    compile(
        g,
        target,
        &CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy),
    )
    .expect("the step compiles")
}

/// `h = x[m, k] @ w[n, k]^T` over a BF16 checkpoint-order weight `w`.
fn projection(m: usize, n: usize, k: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![n, k], DType::BF16))
        .unwrap();
    let h = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(h).unwrap()
}

/// A tied-embedding LM step with `m` rows: `embed` gathered by token and widened, `h = x @ w^T` over a BF16
/// projection, `logits = h @ head^T` over the tied BF16 head (its own const of the embedding's values).
fn lm_step(m: usize) -> Graph {
    const K: usize = 2560;
    const VOCAB: usize = 2048;
    let b = Builder::for_tests();
    let embed = b
        .constant("embed", TensorType::new(vec![VOCAB, K], DType::BF16))
        .unwrap();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![m], DType::I32))
        .unwrap();
    let x = b
        .cast(b.gather(embed, 0, token).unwrap(), DType::F32)
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![K, K], DType::BF16))
        .unwrap();
    let head = b
        .constant("head", TensorType::new(vec![VOCAB, K], DType::BF16))
        .unwrap();
    let h = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let logits = b.matmul(h, b.transpose(head, vec![1, 0]).unwrap()).unwrap();
    b.finish(logits).unwrap()
}

/// The constant named `name` of the program's graph.
fn constant(program: &Program, name: &str) -> ValueId {
    let g = program.graph();
    g.values()
        .ids()
        .find(|&id| g.meta(id).name().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no constant {name}"))
}

/// `(declared dtype, buffer storage)` of the program's constant `name`.
fn format_of(program: &Program, name: &str) -> (DType, BufferStorage) {
    let id = constant(program, name);
    (
        program.graph().aval(id).dtype,
        program.storage().storage(id).buffer_storage(),
    )
}

/// The precision of the contraction that reads `name` as its weight.
fn reader_precision(program: &Program, name: &str) -> OperandPrecision {
    let id = constant(program, name);
    program
        .planned()
        .find(|(eqn, _)| {
            eqn.op.dense_weight().is_some()
                && matches!(eqn.inputs.get(1), Some(Operand::Value(read)) if *read == id)
        })
        .and_then(|(eqn, _)| eqn.op.operand_precision())
        .unwrap_or_else(|| panic!("no dense contraction reads {name}"))
}

const CHECKPOINT: (DType, BufferStorage) = (DType::BF16, BufferStorage::bf16_packed());

/// What a program is, as an executor sees it: every equation's op with the kernel the planner chose for it and the
/// dispatches it launches, and every const's declared dtype, storage and bound.
fn identity(program: &Program) -> String {
    let planned: Vec<_> = program
        .planned()
        .map(|(eqn, plan)| (&eqn.op, program.kernel_choice(eqn), plan.dispatches().len()))
        .collect();
    let consts: Vec<_> = program
        .graph()
        .values()
        .ids()
        .filter(|&id| program.graph().meta(id).name().is_some())
        .map(|id| {
            (
                program.graph().meta(id).name(),
                program.graph().aval(id).dtype,
                program.storage().storage(id).buffer_storage(),
                program.graph().meta(id).bound(),
            )
        })
        .collect();
    format!("{planned:?}\n{consts:?}")
}

/// The policy is a permission: a `Narrow` program in which the planner narrows nothing is its `Exact` twin, with the
/// same ops, kernels, weight formats and bounds. The shapes are the wgpu cells that lost to exact when a store pass
/// rounded their weights (the `k9728` down projection and a decode head), on RADV where the one-wave fragment
/// workgroup never wins.
///
/// Mutation: in `select_operand_precision` skip the planner's answer (`if true || on_matrix_unit(g, &probe)`); the
/// narrow program narrows and differs from its twin.
#[test]
fn a_narrow_program_that_narrows_nothing_is_its_exact_twin() {
    for (name, graph) in [
        ("k9728 down", projection(2048, 2560, 9728)),
        ("decode head", projection(1, 151936, 2560)),
    ] {
        let narrow = compiled(&graph, &radv(), MatmulPrecision::Narrow);
        let exact = compiled(&graph, &radv(), MatmulPrecision::Exact);
        assert_eq!(
            reader_precision(&narrow, "w"),
            OperandPrecision::Exact,
            "{name}: nothing narrows on RADV"
        );
        assert_eq!(identity(&narrow), identity(&exact), "{name}");
        assert_eq!(format_of(&narrow, "w"), CHECKPOINT, "{name}");
    }
}

/// A weight has its checkpoint format in every program of a driver, on a target where the programs disagree on
/// precision: the prefill projection narrows to f16 (asserted first) and the one-row decode projection stays exact,
/// and `w` is bf16 in the packed lane in both, so the two programs upload one copy.
///
/// Mutation: in `dense_contraction_weight_storage` return `Some(StorageKind::F16Packed)` for BF16; `w` is not the
/// bf16 lane in either program.
#[test]
fn a_weight_has_its_checkpoint_format_in_prefill_and_decode() {
    let prefill = compiled(
        &projection(512, 2560, 9728),
        &gfx1151(),
        MatmulPrecision::Narrow,
    );
    let decode = compiled(
        &projection(1, 2560, 9728),
        &gfx1151(),
        MatmulPrecision::Narrow,
    );
    assert_eq!(
        reader_precision(&prefill, "w"),
        F16,
        "the prefill reader narrows"
    );
    assert_eq!(
        reader_precision(&decode, "w"),
        OperandPrecision::Exact,
        "the decode reader stays exact"
    );
    assert_eq!(format_of(&prefill, "w"), CHECKPOINT);
    assert_eq!(format_of(&decode, "w"), CHECKPOINT);
}

/// Two consts over one stored tensor (a tied embedding and head) are both the checkpoint's bf16, so they share one
/// upload: the residency key is `(runs, storage)`. The head's reader narrows (asserted first), and the embedding's
/// gather reads bf16 as before.
///
/// Mutation: as the row above; the head is not the bf16 lane.
#[test]
fn a_tied_embedding_and_head_share_one_format() {
    let program = compiled(&lm_step(512), &gfx1151(), MatmulPrecision::Narrow);
    assert_eq!(reader_precision(&program, "head"), F16, "the head narrows");
    assert_eq!(format_of(&program, "head"), CHECKPOINT);
    assert_eq!(format_of(&program, "embed"), CHECKPOINT);
}

/// A narrowed contraction declares its BF16 weight finite in f16, which the binder checks before it looks up the
/// resident copy; a weight whose reader stays exact (a decode row, `Exact` policy) declares nothing.
///
/// Mutation: drop the `declare_bound` loop in `select_operand_precision`; the narrowed weight declares no bound.
#[test]
fn a_narrowed_contraction_declares_its_weight_finite_in_f16() {
    let bound = |program: &Program| program.graph().meta(constant(program, "w")).bound();
    let narrowed = compiled(
        &projection(512, 2560, 9728),
        &gfx1151(),
        MatmulPrecision::Narrow,
    );
    assert_eq!(reader_precision(&narrowed, "w"), F16);
    assert_eq!(
        bound(&narrowed),
        Some(ValueBound::FiniteIn(FragmentDtype::F16))
    );

    let decode = compiled(
        &projection(1, 2560, 9728),
        &gfx1151(),
        MatmulPrecision::Narrow,
    );
    let exact = compiled(
        &projection(512, 2560, 9728),
        &gfx1151(),
        MatmulPrecision::Exact,
    );
    assert_eq!(bound(&decode), None, "a decode reader stays exact");
    assert_eq!(bound(&exact), None, "the Exact policy narrows nothing");
}

/// No op reads a weight under another dtype than the checkpoint's: the narrowed reader's weight attribute is the
/// declared bf16.
#[test]
fn a_narrowed_reader_keeps_its_bf16_weight_attribute() {
    let program = compiled(
        &projection(512, 2560, 9728),
        &gfx1151(),
        MatmulPrecision::Narrow,
    );
    assert!(
        program.planned().any(|(eqn, _)| matches!(
            eqn.op,
            OpKind::DenseContraction {
                weight: DType::BF16,
                precision: F16
            }
        )),
        "{:?}",
        program.planned().map(|(e, _)| &e.op).collect::<Vec<_>>()
    );
}
