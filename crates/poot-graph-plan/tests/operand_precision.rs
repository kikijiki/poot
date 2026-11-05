//! POOT-1430: the `select_operand_precision` pass writes an op's operand precision from the driver's policy, the
//! target's matrix units and the operands' dtypes, and nothing else.
//!
//! Every row drives the production `compile` and reads what the compiled `Program` holds
//! (the precision each planned op carries, the kernel request behind a plan).

use poot_graph_ir::{
    AttentionAttrs, BinOp, Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, Operand,
    OperandPrecision, TensorType,
};
use poot_graph_plan::{CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{AttentionSchedule, ContractionSpec, KernelRequest};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried};
use poot_tensor::DType;

const BF16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::Bf16);
const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);

/// ROCm caps with the RDNA3 WMMA lowering (bf16 and f16 fragments), as the ROCm runtime reports them on gfx11.
fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

/// RADV through wgpu or raw Vulkan: cooperative matrix over f16 only.
fn radv() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
    }
}

/// ROCm caps with no matrix unit.
fn rocm_without_units() -> Target {
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps: DeviceCaps::rocm_default(),
    }
}

fn compiled(g: &Graph, target: &Target, policy: MatmulPrecision) -> Program {
    let options =
        CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD).with_matmul_precision(policy);
    compile(g, target, &options).expect("the graph compiles")
}

/// How many equations of `program` read their operands under each precision, in the order they first appear.
fn summary(program: &Program) -> Vec<(OperandPrecision, usize)> {
    let mut counts: Vec<(OperandPrecision, usize)> = Vec::new();
    for precision in program
        .planned()
        .filter_map(|(eqn, _)| eqn.op.operand_precision())
    {
        match counts.iter_mut().find(|(seen, _)| *seen == precision) {
            Some((_, count)) => *count += 1,
            None => counts.push((precision, 1)),
        }
    }
    counts
}

/// `matmul(x, transpose(w))` over F32 operands: the projection the dense fold turns into a `DenseContraction`.
fn projection() -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![33, 200])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![37, 200])).unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

/// One `Attention` equation over an F32 query and a key and value cache stored as `kv`.
fn attention(kv: DType) -> Graph {
    let (hq, hkv, cap, d) = (4usize, 2usize, 64usize, 16usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, 32, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::new(vec![1, hkv, cap, d], kv))
        .unwrap();
    let v = b
        .constant("v", TensorType::new(vec![1, hkv, cap, d], kv))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 32, cap]))
        .unwrap();
    let mut plan = b.append_plan(0).unwrap();
    let out = plan
        .equation(
            OpKind::Attention(AttentionAttrs::new(hq / hkv, 0.25, None)),
            [q, k, v, mask].map(|t| Operand::Value(t.id())).to_vec(),
        )
        .expect("the attention types");
    plan.declare_result(out).unwrap();
    let mut prepared = b.preflight_append(plan).unwrap();
    let out = b.commit_append(&mut prepared).unwrap();
    b.finish(out).unwrap()
}

/// The precision of the first compiled equation `want` selects.
fn precision_of(program: &Program, want: impl Fn(&OpKind) -> bool) -> OperandPrecision {
    program
        .planned()
        .find(|(eqn, _)| want(&eqn.op))
        .and_then(|(eqn, _)| eqn.op.operand_precision())
        .expect("the equation is planned and carries a precision")
}

fn is_dense(op: &OpKind) -> bool {
    matches!(op, OpKind::DenseContraction { .. })
}

fn is_attention(op: &OpKind) -> bool {
    matches!(op, OpKind::Attention(_))
}

/// `Exact` never narrows: whatever the target's units, no equation reads a narrowed operand and the pass leaves
/// no trace in the compiled program. Mutation: in `select_operand_precision` ignore the policy (drop the early
/// return, and let a `Narrow` choice stand for an `Exact` driver); the ROCm row publishes `Narrow` ops and goes red.
#[test]
fn an_exact_policy_never_narrows() {
    for (target, graph) in [
        (rocm_wmma(), projection()),
        (radv(), projection()),
        (rocm_wmma(), attention(DType::F16)),
    ] {
        let program = compiled(&graph, &target, MatmulPrecision::Exact);
        assert!(
            summary(&program)
                .iter()
                .all(|(precision, _)| precision.is_exact()),
            "{:?}",
            summary(&program)
        );
        assert!(
            !program
                .passes()
                .any(|pass| pass == "select_operand_precision")
        );
    }
}

/// ROCm caps (bf16 and f16 fragments) give a bf16 contraction and an f16 attention over the default f16 KV
/// cache; both are published, and the program's class is the narrow-storage one. Mutation: make the
/// contraction rule prefer f16 (swap the candidates) and the contraction row goes red.
#[test]
fn rocm_caps_give_bf16_contractions_and_f16_attention() {
    let dense = compiled(&projection(), &rocm_wmma(), MatmulPrecision::Narrow);
    assert_eq!(precision_of(&dense, is_dense), BF16);
    assert_eq!(summary(&dense), [(BF16, 1)]);
    assert_eq!(
        dense.numerics_class(),
        poot_graph_ir::analysis::Tier2Class::NarrowStorage
    );
    assert!(
        dense
            .passes()
            .any(|pass| pass == "select_operand_precision")
    );

    let attention = compiled(
        &attention(DType::F16),
        &rocm_wmma(),
        MatmulPrecision::Narrow,
    );
    assert_eq!(precision_of(&attention, is_attention), F16);
    assert_eq!(
        attention.numerics_class(),
        poot_graph_ir::analysis::Tier2Class::NarrowStorage
    );
}

/// RADV exposes f16 fragments only (Mesa disables bf16 on gfx11), so both families get f16 there and the
/// planner never sees a bf16 fragment request. Mutation: report bf16 fragments on the cooperative-matrix
/// lowering (`FragmentUnits::of`); the contraction row picks bf16 and goes red.
#[test]
fn radv_caps_give_f16() {
    let dense = compiled(&projection(), &radv(), MatmulPrecision::Narrow);
    assert_eq!(precision_of(&dense, is_dense), F16);
    let attention = compiled(&attention(DType::F16), &radv(), MatmulPrecision::Narrow);
    assert_eq!(precision_of(&attention, is_attention), F16);
}

/// A target without a matrix unit compiles `Exact` under any policy: narrowing there would lose accuracy for no
/// speed. Mutation: let `FragmentUnits::of` admit f16 for `MatrixUnits::NONE`; this row goes red.
#[test]
fn caps_without_a_matrix_unit_give_exact() {
    for policy in [MatmulPrecision::Narrow, MatmulPrecision::Split] {
        for graph in [projection(), attention(DType::F16)] {
            let program = compiled(&graph, &rocm_without_units(), policy);
            assert!(
                summary(&program)
                    .iter()
                    .all(|(precision, _)| precision.is_exact()),
                "{policy:?}: {:?}",
                summary(&program)
            );
        }
    }
}

/// An attention takes the dtype its keys are stored in where the target has a fragment for it (f16 is the
/// default KV dtype, so K and V enter exactly and only Q and P round): bf16 KV is bf16 on ROCm and f16 on RADV (no
/// bf16 fragments; the SPIR-V planner has no bf16 cache lane to compile it), f32 KV is f16 on both. Mutation: ignore the KV dtype in `fragment_dtype` (always f16); the
/// bf16-on-ROCm row goes red.
#[test]
fn attention_takes_the_kv_storage_dtype() {
    let cases = [
        (rocm_wmma(), DType::F16, F16),
        (rocm_wmma(), DType::BF16, BF16),
        (rocm_wmma(), DType::F32, F16),
        (radv(), DType::F16, F16),
        (radv(), DType::F32, F16),
    ];
    for (target, kv, want) in cases {
        let program = compiled(&attention(kv), &target, MatmulPrecision::Narrow);
        assert_eq!(precision_of(&program, is_attention), want, "{kv}");
    }
}

/// A `Split` policy gives `Split` ops, publishes the reassociating-or-looser class, and plans an attention on
/// the streaming schedule (the tiled schedule stages one tile per operand, so a split operand never takes it).
#[test]
fn a_split_policy_splits_and_plans_attention_off_the_tiled_schedule() {
    let program = compiled(&attention(DType::F16), &rocm_wmma(), MatmulPrecision::Split);
    let split = OperandPrecision::Split(FragmentDtype::F16);
    assert_eq!(precision_of(&program, is_attention), split);
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| is_attention(&eqn.op))
        .unwrap();
    let KernelChoice::Generated(KernelRequest::Attention(spec)) = program.kernel_choice(eqn) else {
        panic!("a generated attention");
    };
    assert_eq!(
        spec.precision, split,
        "the request carries the declared precision"
    );
    assert!(
        !matches!(spec.schedule, AttentionSchedule::Tiled { .. }),
        "{:?}",
        spec.schedule
    );
}

/// The planner hands the declared precision to the kernel request of every contraction schedule it picks: the
/// register-tiled GEMM at `M > 1` and the decode GEMV at `M = 1` both name `Narrow(bf16)`.
#[test]
fn the_kernel_request_carries_the_declared_precision() {
    let program = compiled(&projection(), &rocm_wmma(), MatmulPrecision::Narrow);
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| is_dense(&eqn.op))
        .unwrap();
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(spec))) => {
            assert_eq!(spec.precision, BF16);
        }
        other => panic!("the register-tiled GEMM, got {other:?}"),
    }
    let decode = {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 200])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![37, 200])).unwrap();
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        b.finish(out).unwrap()
    };
    let program = compiled(&decode, &rocm_wmma(), MatmulPrecision::Narrow);
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| is_dense(&eqn.op))
        .unwrap();
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
            precision,
            ..
        })) => assert_eq!(*precision, BF16),
        other => panic!("the decode GEMV, got {other:?}"),
    }
}

/// The pass reads the equation's own operands and nothing else: a precision an op already declares stands
/// (it is not an `Exact` op for the pass to rewrite), whatever the policy.
#[test]
fn a_declared_precision_stands_under_any_policy() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![16, 32])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![32, 16])).unwrap();
    let declared = OperandPrecision::Split(FragmentDtype::Bf16);
    let out = b.matmul_as(x, w, declared).unwrap();
    let out = b
        .binary_scalar(BinOp::Mul, out, poot_graph_ir::Scalar::F32(2.0))
        .unwrap();
    let g = b.finish(out).unwrap();
    for policy in [MatmulPrecision::Exact, MatmulPrecision::Narrow] {
        let program = compiled(&g, &rocm_wmma(), policy);
        // A scalar multiply fuses behind the matmul as its epilogue; the head carries the declared precision.
        let found = program
            .planned()
            .find_map(|(eqn, _)| eqn.op.operand_precision());
        assert_eq!(found, Some(declared), "{policy:?}");
    }
}
