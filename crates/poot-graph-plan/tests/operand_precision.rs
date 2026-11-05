//! POOT-1430: the `select_operand_precision` pass writes an op's operand precision from the driver's policy, the
//! target's matrix units, the operands' dtypes and whether the planner runs the narrowed op on a matrix unit
//! (POOT-1476), and nothing else. The shapes of these rows are real projection and prefill dimensions: a matrix unit
//! wins for them, and the rows that show it does not (a decode row, one wave) say so.
//!
//! Every row drives the production `compile` and reads what the compiled `Program` holds
//! (the precision each planned op carries, the kernel request behind a plan).

use poot_graph_ir::op::ContractionHead;
use poot_graph_ir::{
    AttentionAttrs, BinOp, Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, Operand,
    OperandPrecision, TensorType,
};
use poot_graph_plan::{CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried};
use poot_tensor::DType;

const BF16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::Bf16);
const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);

/// ROCm caps with the RDNA3 WMMA lowering (bf16 and f16 fragments), as the ROCm runtime reports them on gfx11.
fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.register_file = AmdArch::gfx1151().register_file();
    // The wavefront is the device's: a live agent query reports it, the fixture states it.
    caps.subgroup = Queried::Known(poot_target::SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

/// RADV through raw Vulkan: cooperative matrix over f16 only, several subgroups per workgroup.
fn radv() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::vulkan_rdna3_igpu(),
    }
}

/// RADV through wgpu: the same unit, but the runtime cannot widen the pipeline's subgroups, so a fragment workgroup is
/// one wave.
fn wgpu_one_wave() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
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

/// `matmul(x, transpose(w))` over F32 operands: the projection the dense fold turns into a `DenseContraction`, over
/// the rows of a prefill chunk.
fn projection() -> Graph {
    projection_of(2048)
}

/// [`projection`] over `rows` activation rows.
fn projection_of(rows: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, 1024])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![3072, 1024])).unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

/// One prefill `Attention` equation over an F32 query and a key and value cache stored as `kv`.
fn attention(kv: DType) -> Graph {
    let (hq, hkv, cap, d, lq) = (16usize, 8usize, 2048usize, 128usize, 2048usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, lq, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::new(vec![1, hkv, cap, d], kv))
        .unwrap();
    let v = b
        .constant("v", TensorType::new(vec![1, hkv, cap, d], kv))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, lq, cap]))
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

/// A dense contraction, bare or as the head of an epilogue.
fn is_dense(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::DenseContraction { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Dense { .. },
                ..
            }
    )
}

fn is_attention(op: &OpKind) -> bool {
    matches!(op, OpKind::Attention(_))
}

/// POOT-1461: options that name no policy compile under `Narrow`. The row compiles with `CompileOptions::new` and
/// reads what the program holds: the projection reads f16 operands on ROCm, and compiling the same graph under
/// `Exact` does not, so the default is the narrow one and `Exact` stays selectable. Mutation: set
/// `MatmulPrecision::DEFAULT` back to `Exact`; the first assertion goes red.
#[test]
fn options_that_name_no_policy_compile_under_narrow() {
    let default = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    assert_eq!(default.matmul_precision, MatmulPrecision::Narrow);
    assert_eq!(MatmulPrecision::default(), MatmulPrecision::DEFAULT);
    let program = compile(&projection(), &rocm_wmma(), &default).expect("the graph compiles");
    assert_eq!(precision_of(&program, is_dense), F16);
    let exact = compiled(&projection(), &rocm_wmma(), MatmulPrecision::Exact);
    assert!(precision_of(&exact, is_dense).is_exact());
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

/// ROCm caps (bf16 and f16 fragments) give an f16 contraction (POOT-1458: a contraction takes f16 where the
/// unit has it) and an f16 attention over the default f16 KV cache; both are published, and the program's
/// class is the narrow-storage one. Mutation: make the contraction rule prefer bf16 (swap the candidates) and the
/// contraction row goes red.
#[test]
fn rocm_caps_give_f16_contractions_and_f16_attention() {
    let dense = compiled(&projection(), &rocm_wmma(), MatmulPrecision::Narrow);
    assert_eq!(precision_of(&dense, is_dense), F16);
    assert_eq!(summary(&dense), [(F16, 1)]);
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

/// RADV exposes f16 fragments only (Mesa disables bf16 on gfx11), so an attention that narrows there reads f16 and the
/// planner never sees a bf16 fragment request. A contraction does not narrow on RADV at all: the cost model finds the
/// exact register tiling cheaper than the cooperative-matrix one, whose operands are rounded in integer arithmetic
/// (POOT-1474 measured narrowed wgpu prefill slower than exact). Mutation: report bf16 fragments on the
/// cooperative-matrix lowering (`FragmentUnits::of`); the attention row picks bf16 and goes red.
#[test]
fn radv_caps_give_f16_attention_and_exact_contractions() {
    let dense = compiled(&projection(), &radv(), MatmulPrecision::Narrow);
    assert!(precision_of(&dense, is_dense).is_exact());
    let attention = compiled(&attention(DType::F16), &radv(), MatmulPrecision::Narrow);
    assert_eq!(precision_of(&attention, is_attention), F16);
}

/// An op the matrix unit does not win stays `Exact` under the `narrow` policy (POOT-1476): a decode projection is a
/// GEMV, and a wgpu workgroup of one wave is slower than the register tiles for attention. The planner then plans
/// the exact schedule, with no refusal. Mutation: in `select_operand_precision` skip the planner's answer (narrow
/// every op the dtype rule admits); the decode and one-wave rows compile to a refusal and go red.
#[test]
fn an_op_a_matrix_unit_does_not_win_stays_exact() {
    for (label, target, graph) in [
        ("decode on rocm", rocm_wmma(), projection_of(1)),
        ("decode on radv", radv(), projection_of(1)),
        (
            "attention, one wave",
            wgpu_one_wave(),
            attention(DType::F16),
        ),
    ] {
        let program = compiled(&graph, &target, MatmulPrecision::Narrow);
        assert!(
            summary(&program)
                .iter()
                .all(|(precision, _)| precision.is_exact()),
            "{label}: {:?}",
            summary(&program)
        );
    }
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

/// An attention takes f16 over any cache dtype where the target has an f16 fragment (no operand is range-scaled, so
/// no cache dtype is kept out of f16: POOT-1471): f16, bf16
/// and f32 KV all give f16 on ROCm and RADV (RADV has no bf16 cache lane to compile a bf16 cache). Mutation: in
/// `fragment_dtype` make an attention over a bf16 cache offer bf16 alone; the bf16-on-ROCm row goes red.
#[test]
fn attention_takes_f16_over_any_cache_dtype() {
    let cases = [
        (rocm_wmma(), DType::F16),
        (rocm_wmma(), DType::BF16),
        (rocm_wmma(), DType::F32),
        (radv(), DType::F16),
        (radv(), DType::F32),
    ];
    for (target, kv) in cases {
        let program = compiled(&attention(kv), &target, MatmulPrecision::Narrow);
        assert_eq!(precision_of(&program, is_attention), F16, "{kv}");
    }
}

/// A target whose units offer bf16 fragments alone (gfx11 cut down with `fragment_only`) gets bf16 for every
/// contraction and every attention over a bf16 or f32 cache: f16 is the first candidate, bf16 the fallback.
/// Mutation: in `fragment_dtype` drop `Bf16` from the contraction candidates; the dense row stays exact and goes red.
#[test]
fn bf16_fragments_alone_give_bf16() {
    let options = CompileOptions {
        fragment_only: Some(FragmentDtype::Bf16),
        ..CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD)
            .with_matmul_precision(MatmulPrecision::Narrow)
    };
    let dense = compile(&projection(), &rocm_wmma(), &options).expect("the graph compiles");
    assert_eq!(precision_of(&dense, is_dense), BF16);
    for kv in [DType::BF16, DType::F32] {
        let program = compile(&attention(kv), &rocm_wmma(), &options).expect("the graph compiles");
        assert_eq!(precision_of(&program, is_attention), BF16, "{kv}");
    }
}

/// A `Split` policy gives `Split` contractions on a matrix unit. The tiled attention stages one tile per operand, so
/// a split operand has no matrix-unit attention schedule and the attention stays `Exact` (the streaming schedule is
/// exact only). Mutation: let `attention_tiling` offer the register tiles to a split spec; the declared row below
/// plans a scalar split attention and goes red.
#[test]
fn a_split_policy_splits_contractions_and_leaves_attention_exact() {
    let split = OperandPrecision::Split(FragmentDtype::F16);
    let dense = compiled(&projection(), &rocm_wmma(), MatmulPrecision::Split);
    assert_eq!(precision_of(&dense, is_dense), split);
    let attention = compiled(&attention(DType::F16), &rocm_wmma(), MatmulPrecision::Split);
    assert!(precision_of(&attention, is_attention).is_exact());
}

/// The planner hands the declared precision to the kernel request of the matrix-unit schedule it picks: the
/// register-tiled GEMM at `M > 1` names `Narrow(f16)`, and the decode GEMV at `M = 1` (a request with no precision) is
/// planned for an exact operation.
#[test]
fn the_kernel_request_carries_the_declared_precision() {
    let program = compiled(&projection(), &rocm_wmma(), MatmulPrecision::Narrow);
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| is_dense(&eqn.op))
        .unwrap();
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(spec))) => {
            assert_eq!(spec.precision, F16);
        }
        other => panic!("the register-tiled GEMM, got {other:?}"),
    }
    let decode = {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 1024])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![3072, 1024])).unwrap();
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
            ..
        })) => {
            assert!(precision_of(&program, is_dense).is_exact());
        }
        other => panic!("the decode GEMV, got {other:?}"),
    }
}

/// The pass reads the equation's own operands and nothing else: a precision an op already declares stands
/// (it is not an `Exact` op for the pass to rewrite), whatever the policy.
#[test]
fn a_declared_precision_stands_under_any_policy() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2048, 1024])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![1024, 3072])).unwrap();
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
