//! Unit tests for the planner: dispatch-predicate gating, cache-key stability, strided-view promotion,
//! real-config dispatch-count regressions.

use crate::dtype_widen::{bf16_const_feeds_only_packed_readers, widen_mismatched_matmul_dtypes};
use crate::graph_validation::ValidationPacketSource;
use crate::planner::{Planned, fty_ty};
use crate::predicates::{
    DenseGemvLaunch, ELEMENTWISE_2D_WIDTH, GEMV_WIDTH, GraphTables, dense_decode_gemv_in,
    dense_gemv_launch, is_elementwise_2d_in,
};
use crate::refusal::RefusalDtypes;
use crate::types::{DispatchPlanError, Launch, ScratchDecl};
use poot_graph_ir::test_support::graph_fixtures::derive;
use poot_graph_ir::{
    AttentionAttrs, Builder, Eqn, Graph, OpKind, Operand, RedOp, Scalar, ShapeError, Storage,
    TensorType, ValidationId, ValueId,
};
use poot_kernel_ir::{BinOp, Body, Ty};
use poot_kernelgen::{
    CastSpec, ContractionSpec, DenseTiledContraction, DenseTiledWeight, FusedInput, FusedScalarOp,
    KernelRequest, LeafTy, PointwiseForm, PointwiseSpec, Schedule,
};
use poot_target::{
    AmdArch, Backend, DECODE_GEMV_CHUNK_TRIGGER, DeviceCaps, TargetId, WatchdogBudget,
};
use poot_tensor::DType as GDt;
use poot_test_util::device_caps::default_caps_for;
use std::collections::HashMap;

use super::*;
use poot_graph_ir::op::BinOp as GBinOp;
use poot_graph_ir::op::UnOp as GUnOp;
use poot_kernelgen as kg;

/// The one dispatch of a plan that launches one kernel and binds no metadata buffer.
fn plain(plan: &Plan) -> &Dispatch {
    match plan.single() {
        Some(dispatch) if dispatch.launch().meta.is_empty() => dispatch,
        _ => panic!("expected one dispatch without a metadata buffer, got {plan:?}"),
    }
}

/// The one dispatch of a plan that launches one kernel and binds a metadata buffer.
fn with_meta(plan: &Plan) -> &Dispatch {
    match plan.single() {
        Some(dispatch) if !dispatch.launch().meta.is_empty() => dispatch,
        _ => panic!("expected one dispatch with a metadata buffer, got {plan:?}"),
    }
}

fn is_plain(plan: &Plan) -> bool {
    plan.single()
        .is_some_and(|dispatch| dispatch.launch().meta.is_empty())
}

/// [`plain`], taking the plan by value.
fn into_plain(plan: Plan) -> Dispatch {
    plain(&plan);
    one_dispatch(plan)
}

/// [`with_meta`], taking the plan by value.
fn into_with_meta(plan: Plan) -> Dispatch {
    with_meta(&plan);
    one_dispatch(plan)
}

fn one_dispatch(plan: Plan) -> Dispatch {
    match plan {
        Plan::Dispatches(dispatches) if dispatches.len() == 1 => {
            dispatches.into_dispatches().remove(0)
        }
        other => panic!("expected one dispatch, got {other:?}"),
    }
}

/// Dispatch count with the planner's strided-view promotion applied: a promoted eqn counts as 0
/// dispatches, like `Reshape`.
fn dispatch_count_planned(g: &Graph, backend: Backend) -> usize {
    let views = compute_views(g, backend);
    g.eqns()
        .iter()
        .filter(|e| !matches!(e.op, OpKind::Reshape { .. }) && !views.contains_key(&e.out))
        .count()
}

fn is_elementwise_2d(
    g: &Graph,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
) -> bool {
    is_elementwise_2d_in(
        g.into(),
        eqn,
        out_shape,
        out_numel,
        &default_caps_for(backend),
    )
}

#[test]
fn validation_packet_plan_matches_layout() {
    let b = Builder::for_tests();
    let primary = b.constant("primary", TensorType::f32([1])).unwrap();
    let first = b.constant("first", TensorType::f32([2])).unwrap();
    let second = b.constant("second", TensorType::f32([3])).unwrap();
    let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
        b,
        primary,
        &[
            (ValidationId(2), "first", first),
            (ValidationId(5), "second", second),
        ],
    )
    .unwrap();

    let plan = crate::device_validation::validation_packet_plan(&graph);
    assert_eq!(plan.layout.lane_count, 5);
    assert_eq!(plan.layout.byte_len, 20);
    assert_eq!(
        plan.sources,
        [
            ValidationPacketSource {
                value: first.id(),
                first_lane: 0,
                lane_count: 2,
            },
            ValidationPacketSource {
                value: second.id(),
                first_lane: 2,
                lane_count: 3,
            },
        ]
    );
}

// Build a single bf16/f32 matmul graph `[1,m,k] @ [k,n]`; returns (graph, the matmul eqn index).
fn matmul_graph(m: usize, k: usize, n: usize, dt: GDt) -> Graph {
    let bld = Builder::for_tests();
    let a = bld
        .constant("a", TensorType::new(vec![1, m, k], dt))
        .unwrap();
    let w = bld.constant("w", TensorType::new(vec![k, n], dt)).unwrap();
    let mm = bld.matmul(a, w).unwrap();
    bld.finish(mm).unwrap()
}

/// The plan and the kernel choice behind it for one equation (single-device, no views).
fn planned_with_choice(
    g: &Graph,
    eqn: &Eqn,
    backend: Backend,
) -> Result<(Plan, KernelChoice), PlanError> {
    planned_with_choice_caps(g, eqn, backend, &default_caps_for(backend))
}

/// [`planned_with_choice`] against explicit device caps.
fn planned_with_choice_caps(
    g: &Graph,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<(Plan, KernelChoice), PlanError> {
    plan_eqn_choice(
        g,
        eqn,
        &Target::Gpu {
            backend,
            caps: *caps,
        },
        &HashMap::new(),
        &crate::test_support::roomy_body_limits(),
        None,
    )
    .map(|Planned { plan, choice }| (plan, choice))
}

/// Whether a choice is the shipped kernel `kernel`.
fn is_imported(choice: &KernelChoice, kernel: ImportedKernel) -> bool {
    matches!(choice, KernelChoice::Imported { kernel: k, .. } if *k == kernel)
}

/// Whether the choice is the generated decode GEMV (`kg::gemv_lds`): the one body every decode `[K, N]` GEMV
/// runs, whole or split.
fn is_decode_gemv(choice: &KernelChoice) -> bool {
    matches!(contraction(choice), Some(ContractionSpec::GemvChunk { .. }))
}

/// The kernel choice for one equation. Tests assert on this record, never on the plan's opaque key.
fn choice_of(g: &Graph, eqn: &Eqn, backend: Backend) -> KernelChoice {
    planned_with_choice(g, eqn, backend).unwrap().1
}

/// The kernel choice of the graph's one `MatMul`.
fn matmul_choice(g: &Graph, backend: Backend) -> KernelChoice {
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .expect("a matmul eqn");
    choice_of(g, eqn, backend)
}

/// The contraction request a choice generated, if it generated one.
fn contraction(choice: &KernelChoice) -> Option<&ContractionSpec> {
    match choice {
        KernelChoice::Generated(KernelRequest::Contraction(spec)) => Some(spec),
        _ => None,
    }
}

/// The serial reference matmul's element type, if the choice generated that kernel.
fn serial_matmul_dtype(choice: &KernelChoice) -> Option<&Ty> {
    match contraction(choice) {
        Some(ContractionSpec::Serial {
            dt, epilogue: None, ..
        }) => Some(dt),
        _ => None,
    }
}

#[test]
fn exact_i32_binary_plans_typed_integer_kernels_on_every_compute_backend() {
    let b = Builder::for_tests();
    let a = b
        .constant("a", TensorType::new(vec![2, 1], GDt::I32))
        .unwrap();
    let c = b
        .constant("c", TensorType::new(vec![1, 3], GDt::I32))
        .unwrap();
    let add = b.binary(GBinOp::Add, a, c).unwrap();
    let geu = b
        .binary_scalar(GBinOp::GeU, add, Scalar::I32(i32::MIN))
        .unwrap();
    let remu = b
        .binary_scalar(GBinOp::RemU, geu, Scalar::I32(0x7fff_ffffu32 as i32))
        .unwrap();
    let g = b.finish(remu).unwrap();

    for (eqn, expected_name) in [
        (&g.eqns()[0], "Add"),
        (&g.eqns()[1], "GeU"),
        (&g.eqns()[2], "RemU"),
    ] {
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let (planned_plan, choice) = planned_with_choice(&g, eqn, backend).unwrap();
            let Dispatch { body, .. } = into_plain(planned_plan);
            // The request names the exact-I32 operation, not the f32 lane's: a one-step exact-I32 pointwise
            // region over I32 leaves.
            let named = match &choice {
                KernelChoice::Generated(KernelRequest::Pointwise(PointwiseSpec {
                    kernel,
                    form: PointwiseForm::ExactI32 { .. },
                    ..
                })) => {
                    let expected_op = match expected_name {
                        "Add" => FusedScalarOp::Binary(BinOp::Add),
                        "GeU" => FusedScalarOp::GeU,
                        _ => FusedScalarOp::RemU,
                    };
                    kernel.leaf_tys.iter().all(|&leaf| leaf == LeafTy::I32)
                        && matches!(kernel.steps.as_slice(), [step] if step.op == expected_op)
                }
                _ => false,
            };
            assert!(named, "{expected_name} on {backend:?}: {choice:?}");
            for local in &body.locals[1..=body.param_count as usize] {
                let Ty::Ref { pointee, .. } = &local.ty else {
                    panic!("binary parameter must be a slice reference: {:?}", local.ty);
                };
                assert_eq!(**pointee, Ty::Slice(Box::new(Ty::I32)));
            }
            if expected_name == "RemU" {
                use poot_kernel_ir::{Rvalue, Statement};
                let statements: Vec<_> = body
                    .blocks
                    .iter()
                    .flat_map(|block| &block.statements)
                    .collect();
                assert!(
                    statements.iter().any(|statement| matches!(
                        statement,
                        Statement::Assign(_, Rvalue::Bitcast { to: Ty::U32, .. })
                    )),
                    "RemU must bitcast operands to U32: {choice:?}"
                );
                assert!(
                    statements.iter().any(|statement| matches!(
                        statement,
                        Statement::Assign(_, Rvalue::BinaryOp(BinOp::Rem, _, _))
                    )),
                    "RemU must emit kernel-IR Rem: {choice:?}"
                );
            }
        }
    }
}

#[test]
fn exact_i32_bit_select_and_closed_fusion_plan_as_i32() {
    let b = Builder::for_tests();
    let ty = TensorType::new(vec![4], GDt::I32);
    let x = b.constant("x", ty.clone()).unwrap();
    let y = b.constant("y", ty.clone()).unwrap();
    let z = b.constant("z", ty).unwrap();
    let masked = b.binary(GBinOp::And, x, y).unwrap();
    let shifted = b
        .binary_scalar(GBinOp::Shl, masked, Scalar::I32(33))
        .unwrap();
    let counted = b.unary(GUnOp::Clz, shifted).unwrap();
    let inverted = b.unary(GUnOp::Not, counted).unwrap();
    let selected = b.select(inverted, y, z).unwrap();
    let g = b.finish(selected).unwrap();

    for eqn in g.eqns() {
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let (planned_plan, choice) = planned_with_choice(&g, eqn, backend).unwrap();
            let Dispatch { body, .. } = into_plain(planned_plan);
            // Every step runs over exact I32 storage: the exact-I32 pointwise forms.
            let exact = match &choice {
                KernelChoice::Generated(KernelRequest::Pointwise(spec)) => {
                    matches!(
                        spec.form,
                        PointwiseForm::ExactI32 { .. } | PointwiseForm::ExactI32Packed { .. }
                    )
                }
                _ => false,
            };
            assert!(exact, "{choice:?}");
            for local in &body.locals[1..=body.param_count as usize] {
                let Ty::Ref { pointee, .. } = &local.ty else {
                    panic!("parameter must be a slice reference: {:?}", local.ty);
                };
                assert_eq!(
                    **pointee,
                    Ty::Slice(Box::new(Ty::I32)),
                    "{:?} {choice:?}",
                    eqn.op
                );
            }
        }
    }

    let fused = crate::fuse(&g).unwrap();
    assert_eq!(fused.eqns().len(), 1);
    let eqn = &fused.eqns()[0];
    let (planned_plan, choice) = planned_with_choice(&fused, eqn, Backend::Nvptx).unwrap();
    let Dispatch { body, .. } = into_plain(planned_plan);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Pointwise(PointwiseSpec {
                form: PointwiseForm::ExactI32Packed { .. },
                ..
            }))
        ),
        "{choice:?}"
    );
    for local in &body.locals[1..=body.param_count as usize] {
        let Ty::Ref { pointee, .. } = &local.ty else {
            panic!("fused parameter must be a slice reference: {:?}", local.ty);
        };
        assert_eq!(**pointee, Ty::Slice(Box::new(Ty::I32)));
    }
}

#[test]
fn exact_i32_select_past_the_grid_ceiling_emits_a_2d_fold() {
    let over_ceiling = 65_535 * 256 + 256;
    let b = Builder::for_tests();
    let ty = TensorType::new(vec![over_ceiling], GDt::I32);
    let cond = b.constant("cond", ty.clone()).unwrap();
    let if_true = b.constant("t", ty.clone()).unwrap();
    let if_false = b.constant("f", ty).unwrap();
    let selected = b.select(cond, if_true, if_false).unwrap();
    let g = b.finish(selected).unwrap();
    let eqn = &g.eqns()[0];
    let (planned_plan, choice) = planned_with_choice(&g, eqn, Backend::SpirvVulkan).unwrap();
    let Dispatch { body, .. } = into_plain(planned_plan);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Pointwise(PointwiseSpec {
                form: PointwiseForm::ExactI32 {
                    fold: kg::Fold { two_d: true, .. }
                },
                ..
            }))
        ),
        "a Select past the grid ceiling must request the 2-D fold: {choice:?}"
    );
    assert!(
        body.blocks.iter().any(|block| matches!(
            block.terminator,
            poot_kernel_ir::Terminator::ThreadIndexCall {
                dim: poot_kernel_ir::IndexAxis::GroupY,
                ..
            }
        )),
        "2-D Select kernel must read GroupY"
    );
}

/// POOT-1033: a float `Select` plans as one float pointwise kernel whose condition leaf is I32 and whose
/// branch leaves are F32, on every compute backend: the kernel's parameters carry both element types, so
/// the I32 condition is never read as float bits. Mutation: type every leaf `F32` in the planner's `Select`
/// arm; the condition parameter turns into an f32 slice and this row goes red.
#[test]
fn float_select_plans_a_float_pointwise_kernel_with_an_i32_condition_leaf() {
    let b = Builder::for_tests();
    let cond = b
        .constant("cond", TensorType::new(vec![4], GDt::I32))
        .unwrap();
    let if_true = b.constant("t", TensorType::new(vec![4], GDt::F32)).unwrap();
    let if_false = b.constant("f", TensorType::new(vec![4], GDt::F32)).unwrap();
    let selected = b.select(cond, if_true, if_false).unwrap();
    let g = b.finish(selected).unwrap();
    let eqn = &g.eqns()[0];
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let (planned_plan, choice) = planned_with_choice(&g, eqn, backend).unwrap();
        let Dispatch { body, .. } = into_plain(planned_plan);
        let KernelChoice::Generated(KernelRequest::Pointwise(spec)) = &choice else {
            panic!("a float Select is one pointwise kernel: {choice:?}");
        };
        assert!(
            matches!(spec.form, PointwiseForm::Float { .. }),
            "{choice:?}"
        );
        assert_eq!(
            spec.kernel.leaf_tys,
            [kg::LeafTy::I32, kg::LeafTy::F32, kg::LeafTy::F32]
        );
        let element = |index: usize| {
            let Ty::Ref { pointee, .. } = &body.locals[index].ty else {
                panic!("parameter {index} must be a slice reference");
            };
            (**pointee).clone()
        };
        assert_eq!(element(1), Ty::Slice(Box::new(Ty::I32)), "{backend:?}");
        assert_eq!(element(2), Ty::Slice(Box::new(Ty::F32)), "{backend:?}");
        assert_eq!(element(3), Ty::Slice(Box::new(Ty::F32)), "{backend:?}");
        assert_eq!(element(4), Ty::Slice(Box::new(Ty::F32)), "{backend:?}");
    }
}

/// POOT-1301: a `Select` branch that is an inline literal is an inline operand of the kernel's step, not a leaf:
/// the index guard's `Select(invalid, 0, raw)` plans as a two-leaf exact-I32 kernel (the flag and `raw`) on every
/// compute backend, and a literal condition has no kernel and is refused by name.
///
/// Mutation: make the planner's `Select` arm push a leaf for a literal branch; the leaf count and the step's
/// operands go red.
#[test]
fn a_select_with_a_literal_branch_plans_the_literal_inline() {
    let b = Builder::for_tests();
    let raw = b
        .constant("raw", TensorType::new(vec![4], GDt::I32))
        .unwrap();
    let guard = b.guard_index_bounds(raw, 3).unwrap();
    let g = b.finish(guard.guarded).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Select))
        .expect("the guard selects");
    assert!(matches!(
        eqn.inputs.as_slice(),
        [
            poot_graph_ir::Operand::Value(_),
            poot_graph_ir::Operand::Lit(poot_graph_ir::Scalar::I32(0)),
            poot_graph_ir::Operand::Value(_)
        ]
    ));
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let (_, choice) = planned_with_choice(&g, eqn, backend).unwrap();
        let KernelChoice::Generated(KernelRequest::Pointwise(spec)) = &choice else {
            panic!("a Select is one pointwise kernel: {choice:?}");
        };
        assert!(
            matches!(spec.form, PointwiseForm::ExactI32 { .. }),
            "{backend:?}: {choice:?}"
        );
        assert_eq!(spec.leaves.len(), 2, "{backend:?}: the flag and raw");
        assert_eq!(spec.kernel.leaf_tys, [kg::LeafTy::I32, kg::LeafTy::I32]);
        assert!(
            matches!(
                spec.kernel.steps[0].inputs.as_slice(),
                [
                    kg::FusedInput::Leaf(flag),
                    kg::FusedInput::LitI32(0),
                    kg::FusedInput::Leaf(raw)
                ] if flag.index() == 0 && raw.index() == 1
            ),
            "{backend:?}: {:?}",
            spec.kernel.steps[0].inputs
        );
    }

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::new(vec![4], GDt::I32)).unwrap();
    let y = b.constant("y", TensorType::new(vec![4], GDt::I32)).unwrap();
    let selected = b.select(x, x, y).unwrap();
    let g = b.finish(selected).unwrap();
    let mut edit = g.edit();
    edit.eqns_mut()[0].inputs[0] = poot_graph_ir::Operand::Lit(poot_graph_ir::Scalar::I32(1));
    let g = edit.commit().unwrap();
    let refusal = planned_with_choice(&g, &g.eqns()[0], Backend::SpirvVulkan)
        .expect_err("a literal condition has no kernel");
    let PlanError::Refused(refusal) = refusal else {
        panic!("a typed refusal: {refusal:?}")
    };
    assert_eq!(refusal.missing, Capability::LiteralFirstOperand);
}

/// A float `Select` past the grid ceiling takes the same 2-D fold the exact-I32 `Select` does, standalone and
/// inside a fused float region (the K/V zero-select of a long batched cache is far past 65535 * 256 elements):
/// the request carries the fold and the body reads `GroupY`. Mutation: pass `PointwiseForm::Float`'s fold as
/// flat in `request::pointwise` (or drop `OpKind::Fused` from `is_elementwise_2d_in`); the body reads no
/// `GroupY` and this row goes red.
#[test]
fn float_select_past_the_grid_ceiling_emits_a_2d_fold() {
    let over_ceiling = 65_535 * 256 + 256;
    let b = Builder::for_tests();
    let cond = b
        .constant("cond", TensorType::new(vec![over_ceiling], GDt::I32))
        .unwrap();
    let if_true = b
        .constant("t", TensorType::new(vec![over_ceiling], GDt::F32))
        .unwrap();
    let if_false = b
        .constant("f", TensorType::new(vec![over_ceiling], GDt::F32))
        .unwrap();
    let selected = b.select(cond, if_true, if_false).unwrap();
    let standalone = b.finish(selected).unwrap();
    // The same select followed by a pointwise neighbour fuses into one float region.
    let b = Builder::for_tests();
    let cond = b
        .constant("cond", TensorType::new(vec![over_ceiling], GDt::I32))
        .unwrap();
    let if_true = b
        .constant("t", TensorType::new(vec![over_ceiling], GDt::F32))
        .unwrap();
    let if_false = b
        .constant("f", TensorType::new(vec![over_ceiling], GDt::F32))
        .unwrap();
    let negated = b
        .unary(GUnOp::Neg, b.select(cond, if_true, if_false).unwrap())
        .unwrap();
    let fused = crate::fuse(&b.finish(negated).unwrap()).unwrap();
    assert!(matches!(fused.eqns()[..], [ref eqn] if matches!(eqn.op, OpKind::Fused(_))));
    for (what, g) in [("standalone", &standalone), ("fused", &fused)] {
        let eqn = g.eqns().last().unwrap();
        // The capped-grid target: NVPTX's X grid limit is far above this size, so it needs no fold.
        let backend = Backend::SpirvVulkan;
        {
            let (planned_plan, choice) = planned_with_choice(g, eqn, backend).unwrap();
            let Dispatch { body, .. } = into_plain(planned_plan);
            assert!(
                matches!(
                    &choice,
                    KernelChoice::Generated(KernelRequest::Pointwise(PointwiseSpec {
                        form: PointwiseForm::Float {
                            fold: kg::Fold { two_d: true, .. }
                        },
                        ..
                    }))
                ),
                "{what} {backend:?}: a float pointwise kernel past the ceiling must request the 2-D fold: {choice:?}"
            );
            assert!(
                body.blocks.iter().any(|block| matches!(
                    block.terminator,
                    poot_kernel_ir::Terminator::ThreadIndexCall {
                        dim: poot_kernel_ir::IndexAxis::GroupY,
                        ..
                    }
                )),
                "{what} {backend:?}: the folded float kernel must read GroupY"
            );
        }
    }
}

/// Only I32 and F32 branches have a kernel today: an F16 `Select` traces and evaluates (the oracle moves its
/// bits) but the planner refuses it with the typed dtype-lowering refusal, never a wrong kernel.
#[test]
fn half_width_select_is_refused_as_a_dtype_lowering_gap() {
    let b = Builder::for_tests();
    let cond = b
        .constant("cond", TensorType::new(vec![4], GDt::I32))
        .unwrap();
    let if_true = b.constant("t", TensorType::new(vec![4], GDt::F16)).unwrap();
    let if_false = b.constant("f", TensorType::new(vec![4], GDt::F16)).unwrap();
    let selected = b.select(cond, if_true, if_false).unwrap();
    let g = b.finish(selected).unwrap();
    let eqn = &g.eqns()[0];
    let refusal = refusal_of(
        plan_eqn(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "an F16 Select",
    );
    assert_eq!(refusal.missing, Capability::DtypeLowering);
}

#[test]
fn i32_is_its_own_kernel_element_type() {
    // POOT-725: no I32 value is computed through f32. Its element type is I32 wherever it is read.
    assert_eq!(fty_ty(GDt::I32), Ty::I32);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::new(vec![2], GDt::I32)).unwrap();
    let y = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1)).unwrap();
    let g = b.finish(y).unwrap();
    let eqn = &g.eqns()[0];
    let planned_plan = plan_eqn(
        &g,
        eqn,
        Backend::Nvptx,
        &default_caps_for(Backend::Nvptx),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch { body, .. } = into_plain(planned_plan);
    assert!(matches!(
        body.locals[1].ty,
        Ty::Ref {
            pointee: ref p,
            ..
        } if **p == Ty::Slice(Box::new(Ty::I32))
    ));
}

fn slice_element_ty(body: &Body, param: usize) -> &Ty {
    let Ty::Ref { pointee, .. } = &body.locals[param].ty else {
        panic!(
            "parameter {param} is not a reference: {:?}",
            body.locals[param].ty
        );
    };
    let Ty::Slice(element) = &**pointee else {
        panic!("parameter {param} is not a slice: {pointee:?}");
    };
    element
}

/// An I32 movement moves I32 words whatever reads its result: the same kernel (same request, same plan key)
/// whether an arithmetic op or a `Gather` index consumes it.
///
/// Mutation: have `movement_storage` carry an I32 value through `Ty::F32`; both carriers go red.
#[test]
fn i32_movement_carries_i32_whatever_consumes_it() {
    let movement_plan = |arithmetic: bool| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::new(vec![1], GDt::I32)).unwrap();
        let moved = b.broadcast(x, vec![2]).unwrap();
        let output = if arithmetic {
            b.binary_scalar(GBinOp::Add, moved, Scalar::I32(1)).unwrap()
        } else {
            let table = b.constant("table", TensorType::f32(vec![4, 1])).unwrap();
            b.gather(table, 0, moved).unwrap()
        };
        let g = b.finish(output).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Broadcast { .. }))
            .unwrap();
        let (planned_plan, choice) = planned_with_choice(&g, eqn, Backend::Nvptx).unwrap();
        let Dispatch { body, key, .. } = into_plain(planned_plan);
        (body, key, choice)
    };

    let (arith_body, arith_key, arith_choice) = movement_plan(true);
    let (index_body, index_key, index_choice) = movement_plan(false);
    assert_eq!(slice_element_ty(&arith_body, 1), &Ty::I32);
    assert_eq!(slice_element_ty(&index_body, 1), &Ty::I32);
    let broadcast_dtype = |choice: &KernelChoice| match choice {
        KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
            storage: kg::Storage::Elements { carrier },
            op: kg::MovementOp::Broadcast { .. },
            ..
        })) => carrier.clone(),
        other => panic!("a broadcast must be generated: {other:?}"),
    };
    assert_eq!(broadcast_dtype(&arith_choice), Ty::I32);
    assert_eq!(broadcast_dtype(&index_choice), Ty::I32);
    assert_eq!(arith_key, index_key);
}

/// An I32 `Add` plans one I32 kernel whether the sum is observed, or only feeds a `Gather` index: the kernel
/// does not depend on its consumer.
///
/// Mutation: have `fty_ty` map I32 back to `Ty::F32`; the unobserved case's param goes `Slice<f32>`.
#[test]
fn an_i32_binary_plans_the_same_kernel_whatever_consumes_it() {
    let binary_plan = |observed: bool| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::new(vec![2], GDt::I32)).unwrap();
        let y = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1)).unwrap();
        let output = if observed {
            y
        } else {
            let table = b.constant("table", TensorType::f32(vec![4, 1])).unwrap();
            b.gather(table, 0, y).unwrap()
        };
        let g = b.finish(output).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Binary(_)))
            .unwrap();
        let (planned_plan, choice) = planned_with_choice(&g, eqn, Backend::Nvptx).unwrap();
        let Dispatch { body, key, .. } = into_plain(planned_plan);
        (body, key, choice)
    };
    let (observed_body, observed_key, observed_choice) = binary_plan(true);
    let (index_body, index_key, index_choice) = binary_plan(false);
    for (body, choice) in [
        (&observed_body, &observed_choice),
        (&index_body, &index_choice),
    ] {
        assert_eq!(slice_element_ty(body, 1), &Ty::I32, "{choice:?}");
        assert!(
            matches!(
                choice,
                KernelChoice::Generated(KernelRequest::Pointwise(PointwiseSpec {
                    form: PointwiseForm::ExactI32 { .. },
                    kernel,
                    ..
                })) if matches!(
                    kernel.steps.as_slice(),
                    [step] if matches!(step.inputs.as_slice(), [FusedInput::Leaf(_), FusedInput::LitI32(_)])
                )
            ),
            "{choice:?}"
        );
    }
    assert_eq!(observed_key, index_key);
}

/// A `Gather` reads its index through `Slice<i32>` on axis 0 and axis 1, whether the index is a plain I32 value
/// or computed by I32 arithmetic, and the movement request names no index element type.
///
/// Mutation: read the index param as `Slice<f32>` in the movement generator; the param assertions go red.
#[test]
fn gather_reads_i32_indices_through_slice_i32() {
    let gather_plan = |computed: bool, axis: usize| {
        let b = Builder::for_tests();
        let table = if axis == 0 {
            b.constant("table", TensorType::f32(vec![4, 2])).unwrap()
        } else {
            b.constant("table", TensorType::f32(vec![2, 4, 2])).unwrap()
        };
        let index = b
            .constant("index", TensorType::new(vec![2], GDt::I32))
            .unwrap();
        let index = if computed {
            b.binary_scalar(GBinOp::Add, index, Scalar::I32(1)).unwrap()
        } else {
            b.broadcast(index, vec![2]).unwrap()
        };
        let gathered = b.gather(table, axis, index).unwrap();
        let g = b.finish(gathered).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .unwrap();
        planned_with_choice(&g, eqn, Backend::Nvptx).unwrap()
    };

    for axis in [0, 1] {
        for computed in [true, false] {
            let (planned_plan, choice) = gather_plan(computed, axis);
            let Dispatch { body, .. } = into_plain(planned_plan);
            assert_eq!(slice_element_ty(&body, 1), &Ty::F32, "axis {axis}");
            assert_eq!(
                slice_element_ty(&body, 2),
                &Ty::I32,
                "axis {axis} computed {computed}: {choice:?}"
            );
        }
    }
}

/// A Gather over I32 data moves the table's words as i32 and reads its index as i32.
///
/// Mutation: have `movement_storage` carry an I32 table through `Ty::F32`; the table and output params go
/// `Slice<f32>` and this row goes red.
#[test]
fn gather_over_i32_data_reads_the_table_on_the_i32_lane() {
    for axis in [0, 1] {
        let b = Builder::for_tests();
        let table_shape = if axis == 0 { vec![4, 2] } else { vec![2, 4, 2] };
        let table = b
            .constant("table", TensorType::new(table_shape, GDt::I32))
            .unwrap();
        let index = b
            .constant("index", TensorType::new(vec![3], GDt::I32))
            .unwrap();
        let gathered = b.gather(table, axis, index).unwrap();
        let g = b.finish(gathered).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .unwrap();
        let body = match plan_eqn(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap()
        {
            plan if is_plain(&plan) => one_dispatch(plan).body().clone(),
            other => panic!("axis {axis}: I32 Gather planned as {other:?}"),
        };
        assert_eq!(slice_element_ty(&body, 1), &Ty::I32, "axis {axis} data");
        assert_eq!(slice_element_ty(&body, 3), &Ty::I32, "axis {axis} out");
        assert_eq!(slice_element_ty(&body, 2), &Ty::I32, "axis {axis} index");
    }
}

#[test]
fn an_isolated_i32_gather_plans_the_i32_carrier_with_an_i32_index() {
    for (axis, table_shape) in [(0, vec![4, 2]), (1, vec![2, 4, 2])] {
        let b = Builder::for_tests();
        let table = b
            .constant("table", TensorType::new(table_shape, GDt::I32))
            .unwrap();
        let index = b
            .constant("index", TensorType::new(vec![3], GDt::I32))
            .unwrap();
        let gathered = b.gather(table, axis, index).unwrap();
        let g = b.finish(gathered).unwrap();
        let eqn = &g.eqns()[0];
        let (planned_plan, choice) = planned_with_choice(&g, eqn, Backend::SpirvVulkan).unwrap();
        let Dispatch { body, .. } = into_plain(planned_plan);
        let (in_shape, out_shape) = if axis == 0 {
            (vec![4, 2], vec![3, 2])
        } else {
            (vec![2, 4, 2], vec![2, 3, 2])
        };
        let KernelChoice::Generated(KernelRequest::Movement(spec)) = &choice else {
            panic!("axis {axis}: {choice:?}");
        };
        assert_eq!(
            spec,
            &kg::MovementSpec::new(
                kg::Storage::Elements { carrier: Ty::I32 },
                out_shape,
                kg::MovementOp::Gather {
                    in_shape,
                    axis,
                    index_numel: 3,
                },
            ),
            "axis {axis}"
        );
        assert_eq!(slice_element_ty(&body, 1), &Ty::I32, "axis {axis} data");
        assert_eq!(slice_element_ty(&body, 2), &Ty::I32, "axis {axis} index");
    }
}

#[test]
fn planner_rejects_literal_first_binary_locally_without_panicking() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2])).unwrap();
    let output = b.binary_scalar(GBinOp::Sub, x, Scalar::F32(1.0)).unwrap();
    let g = derive(&b.finish(output).unwrap(), |draft| {
        draft.eqns_mut()[0].inputs.swap(0, 1);
    })
    .unwrap();
    let refusal = refusal_of(
        plan_eqn(
            &g,
            &g.eqns()[0],
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "literal-first Binary",
    );
    assert_eq!(refusal.missing, Capability::LiteralFirstOperand);
}

/// Card 558a: `Iota` is a pure compile-time constant, so the planner has no kernel for it and refuses
/// an unfolded equation with the typed [`Capability::UnfoldedIota`]. `transform::fold_iota` is the one
/// route to a planned iota.
#[test]
fn planner_refuses_an_unfolded_iota_with_a_typed_refusal() {
    let b = Builder::for_tests();
    let output = b.cast(b.iota(4).unwrap(), GDt::F32).unwrap();
    let graph = b.finish(output).unwrap();
    let iota = graph
        .eqns()
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::Iota { .. }))
        .expect("the graph must hold an Iota equation");
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            iota,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "unfolded Iota",
    );
    assert_eq!(refusal.missing, Capability::UnfoldedIota);
}

#[test]
fn exact_i32_planner_rejects_static_numel_overflow() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32)).unwrap();
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1)).unwrap();
    let g = derive(&b.finish(output).unwrap(), |draft| {
        // The operand and the result agree, so the graph is well formed; only the element count overflows.
        draft.values_mut()[x.id()].aval.shape = vec![usize::MAX, 2];
        draft.values_mut()[output.id()].aval.shape = vec![usize::MAX, 2];
    })
    .unwrap();
    // card 525: plan_eqn now derives out_shape/out_numel from this same mutated aval itself (it can no
    // longer be told a different shape than the graph's own), so the overflow it must survive without
    // panicking is now computed inside plan_eqn, not passed in by the caller.
    let error = plan_eqn(
        &g,
        &g.eqns()[0],
        Backend::Nvptx,
        &default_caps_for(Backend::Nvptx),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("overflows usize"), "{error}");
}

/// Card 525 SC-003 / R469-019: `plan_eqn` derives `out_shape`/`out_numel` from `g.aval(eqn.out).shape`
/// itself; it no longer takes a caller-supplied shape that could disagree with the graph's own
/// inferred one. This proves the planned kernel's launch dims track the graph's real output shape.
///
/// MUTATION (recorded here, never left in the tree): in `planner.rs`'s `plan_eqn_views`, `let
/// out_shape = g.aval(eqn.out).shape.clone();` was changed to `let out_shape = vec![1, 1];`,
/// reproducing the pre-card-525 shape of bug this closes (a caller-suppliable shape read unchecked).
/// Rerunning this test then panicked: `assertion `left == right` failed left: [1, 1, 1] right: [32,
/// 1, 1]` - the mutated code baked the wrong `[1,1]` shape into the plan instead of the graph's real
/// `[4,8]`. Restoring the `g.aval` read made it green again.
#[test]
fn plan_eqn_derives_shape_from_the_graph_not_a_caller_argument() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, 8])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![4, 8])).unwrap();
    let out = b.binary(GBinOp::Add, x, y).unwrap();
    let g = b.finish(out).unwrap();
    let eqn = &g.eqns()[0];
    let planned_plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch {
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    // 4*8 = 32 elements, one thread each (well under the elementwise-2D fold's threshold): the grid
    // must reflect the graph's real output shape, which plan_eqn can no longer be told to ignore.
    assert_eq!(grid, [32, 1, 1]);
}

/// The typed refusal a planning result carries; any other outcome fails the test with `what`.
pub(crate) fn refusal_of(result: Result<Plan, PlanError>, what: &str) -> Refusal {
    match result {
        Err(PlanError::Refused(refusal)) => *refusal,
        other => panic!("{what}: expected a typed refusal, got {other:?}"),
    }
}

/// Like `matmul_key` but surfaces the planner's `Result`, for the mixed-dtype cases the planner rejects.
fn try_matmul_plan_planned(g: &Graph, backend: Backend) -> Result<(Plan, KernelChoice), PlanError> {
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .expect("a matmul eqn");
    planned_with_choice(g, eqn, backend)
}

fn try_matmul_plan(g: &Graph, backend: Backend) -> Result<Plan, PlanError> {
    try_matmul_plan_planned(g, backend).map(|(plan, _)| plan)
}

/// A biased matmul whose weight dtype differs from its activation's is not a contraction epilogue: the
/// fuse rule only forms one over a head whose operands and result share a dtype (a mixed pairing is the
/// decode-GEMV arm's contract, and `widen_mismatched_matmul_dtypes` casts the rest), so
/// the graph keeps its matmul and its add. Mutation: drop the dtype check in `epilogue_head`; the mixed
/// matmul fuses and this row goes red.
#[test]
fn a_biased_matmul_with_mismatched_operand_dtypes_is_not_fused_into_an_epilogue() {
    let bld = Builder::for_tests();
    let a = bld
        .constant("a", TensorType::new(vec![1, 1, 32], GDt::F32))
        .unwrap();
    let w = bld
        .constant("w", TensorType::new(vec![32, 16], GDt::BF16))
        .unwrap();
    let bias = bld
        .constant("bias", TensorType::new(vec![16], GDt::F32))
        .unwrap();
    let mm = poot_graph_ir::ops::linear(&bld, a, w, Some(bias)).unwrap();
    let g =
        crate::passes::fuse_contraction_epilogues(&bld.finish(mm).unwrap(), |_, _| true).unwrap();
    assert!(
        g.eqns()
            .iter()
            .all(|e| !matches!(e.op, OpKind::ContractionEpilogue { .. })),
        "a mixed-dtype matmul must keep its add: {:?}",
        g.eqns()
    );
}

/// Card 645's review item, closed: a biased projection against a checkpoint-orientation `[N, K]` weight
/// (`x @ w^T + bias`, the qwen2 q/k/v shape) compiles on every backend to exactly one equation, a
/// `ContractionEpilogue` over a dense head reading the weight as stored: no device `Transpose`, no standalone
/// `MatMul` or add. The head is the `M == 1` decode Gemv, and the bias rides in its store. Mutation: put the
/// epilogue rule back before `fold_dense_contractions` (the pre-fold order, where the rule saw only a plain
/// `MatMul` over a `Transpose`): the weight keeps its device transpose and this row goes red.
#[test]
fn a_biased_checkpoint_orientation_projection_is_one_contraction_with_bias_and_no_transpose() {
    use poot_graph_ir::op::ContractionHead;

    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        for weight in [GDt::F32, GDt::BF16] {
            let bld = Builder::for_tests();
            let x = bld
                .constant("x", TensorType::new(vec![1, 1, 32], GDt::F32))
                .unwrap();
            let w = bld
                .constant("w", TensorType::new(vec![16, 32], weight))
                .unwrap();
            let bias = bld
                .constant("bias", TensorType::new(vec![16], GDt::F32))
                .unwrap();
            let projection = bld
                .matmul(x, bld.transpose(w, vec![1, 0]).unwrap())
                .unwrap();
            let out = bld
                .binary(poot_graph_ir::op::BinOp::Add, projection, bias)
                .unwrap();
            let g = bld.finish(out).unwrap();
            let program = compile(
                &g,
                &Target::Gpu {
                    backend,
                    caps: default_caps_for(backend),
                },
                &CompileOptions::new(CompileLimits::STANDARD),
            )
            .unwrap_or_else(|e| panic!("{backend:?} {weight:?}: {e}"));
            let ops: Vec<&OpKind> = program.graph().eqns().iter().map(|e| &e.op).collect();
            assert!(
                matches!(
                    ops.as_slice(),
                    [OpKind::ContractionEpilogue {
                        head: ContractionHead::Dense { weight: stored },
                        ..
                    }] if *stored == weight
                ),
                "{backend:?} {weight:?}: one dense contraction with its bias, got {ops:?}"
            );
        }
    }
}

#[test]
fn e4m3fn_graph_execution_is_rejected_by_name_for_every_unwired_arithmetic_class() {
    // Card 529: infer now rejects E4M3FN arithmetic (Unary/Binary/Reduce) directly, so these
    // three never reach the planner any more - they are IR-level `ShapeError`s, not planner refusals.
    let unary_err = OpKind::Unary(poot_graph_ir::UnOp::Neg)
        .infer(&[TensorType::new(vec![2, 4], GDt::E4M3FN)])
        .expect_err("E4M3FN Unary must be rejected by infer");
    assert!(matches!(
        unary_err,
        ShapeError::DtypeOp {
            op: "Unary",
            dtype: GDt::E4M3FN
        }
    ));

    let binary_err = OpKind::Binary(poot_graph_ir::BinOp::Add)
        .infer(&[
            TensorType::new(vec![2, 4], GDt::E4M3FN),
            TensorType::new(vec![2, 4], GDt::E4M3FN),
        ])
        .expect_err("E4M3FN Binary must be rejected by infer");
    assert!(matches!(
        binary_err,
        ShapeError::DtypeOp {
            op: "Binary",
            dtype: GDt::E4M3FN
        }
    ));

    let reduce_err = OpKind::Reduce {
        op: poot_graph_ir::RedOp::Sum,
        axis: 1,
        keepdim: false,
    }
    .infer(&[TensorType::new(vec![2, 4], GDt::E4M3FN)])
    .expect_err("E4M3FN Reduce must be rejected by infer");
    assert!(matches!(
        reduce_err,
        ShapeError::DtypeOp {
            op: "Reduce",
            dtype: GDt::E4M3FN
        }
    ));

    // Card 623: infer now rejects E4M3FN MatMul directly too (via `reject_non_arithmetic`,
    // the same class check Unary/Binary/Reduce already had), so it also never reaches the planner any
    // more - an IR-level `ShapeError`, not a planner refusal.
    let matmul_err = OpKind::matmul()
        .infer(&[
            TensorType::new(vec![1, 4], GDt::E4M3FN),
            TensorType::new(vec![4, 2], GDt::E4M3FN),
        ])
        .expect_err("E4M3FN MatMul must be rejected by infer");
    assert!(matches!(
        matmul_err,
        ShapeError::DtypeOp {
            op: "MatMul",
            dtype: GDt::E4M3FN
        }
    ));
}

#[test]
fn e4m3fn_scatter_update_materializes_exact_packed_output_and_keys_all_shapes() {
    let build = |source_rows| {
        let builder = Builder::for_tests();
        let base = builder
            .constant("base", TensorType::new(vec![4, 2, 5], GDt::E4M3FN))
            .unwrap();
        let src = builder
            .constant("src", TensorType::new(vec![source_rows, 2, 5], GDt::E4M3FN))
            .unwrap();
        let slots = builder
            .constant("slots", TensorType::new(vec![source_rows], GDt::I32))
            .unwrap();
        let output = builder.scatter_update(base, src, slots).unwrap();
        builder.finish(output).unwrap()
    };
    let graph = build(3);
    let eqn = &graph.eqns()[0];
    let (planned_plan, choice) = planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap();
    let Dispatch {
        body,
        key,
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    // The kernel reads the source rows and the slot map; the base is the output's own buffer.
    assert_eq!(body.param_count, 3);
    let KernelChoice::Generated(KernelRequest::Movement(spec)) = &choice else {
        panic!("expected the packed E4M3FN scatter-update request: {choice:?}");
    };
    assert_eq!(spec.storage, kg::Storage::BYTE_LANES);
    assert_eq!(spec.out_shape, [4, 2, 5]);
    assert_eq!(
        spec.op,
        kg::MovementOp::ScatterRows {
            src_shape: vec![3, 2, 5],
            window: None
        }
    );
    // One thread per packed source word: 3 rows x 2 pool rows x ceil(5 / 4) words.
    assert_eq!(grid, [12, 1, 1]);

    let other = build(2);
    let other_eqn = &other.eqns()[0];
    let planned_plan = plan_eqn(
        &other,
        other_eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch { key: other_key, .. } = into_plain(planned_plan);
    assert_ne!(key, other_key, "source row count must separate plan keys");

    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &crate::test_support::roomy_body_limits()
        )
        .unwrap()
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN ScatterUpdate on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, OpKind::ScatterUpdate);
    assert_eq!(refusal.target, TargetId::Gpu(backend));

    // A rank-1 packed pool has one lane per slot: slots share words, so there is no whole unit to copy and
    // the request is refused by name instead of racing sub-word stores.
    let builder = Builder::for_tests();
    let base = builder
        .constant("base_rank1", TensorType::new(vec![5], GDt::E4M3FN))
        .unwrap();
    let src = builder
        .constant("src_rank1", TensorType::new(vec![3], GDt::E4M3FN))
        .unwrap();
    let slots = builder
        .constant("slots_rank1", TensorType::new(vec![3], GDt::I32))
        .unwrap();
    let output = builder.scatter_update(base, src, slots).unwrap();
    let rank1 = builder.finish(output).unwrap();
    let refusal = refusal_of(
        plan_eqn(
            &rank1,
            &rank1.eqns()[0],
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "rank-1 E4M3FN ScatterUpdate",
    );
    assert!(
        matches!(
            refusal.missing,
            Capability::KernelGen(kg::KernelGenError::Unsupported { .. })
        ),
        "{:?}",
        refusal.missing
    );
}

#[test]
fn e4m3fn_dynamic_update_slice_plans_static_and_runtime_exact_packed_outputs() {
    let builder = Builder::for_tests();
    let operand = builder
        .constant("operand", TensorType::new(vec![2, 4, 5], GDt::E4M3FN))
        .unwrap();
    let update = builder
        .constant("update", TensorType::new(vec![2, 2, 5], GDt::E4M3FN))
        .unwrap();
    let output = builder.dynamic_update_slice(operand, update, 1, 1).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let (planned_plan, choice) = planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap();
    let Dispatch {
        body,
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(body.param_count, 3);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
                storage: kg::Storage::PackedRows { lane_bits: 8 },
                out_shape,
                op: kg::MovementOp::UpdateSlice {
                    update_shape,
                    axis: 1,
                    index: kg::UpdateIndex::Static(1),
                },
                ..
            })) if out_shape == &[2, 4, 5] && update_shape == &[2, 2, 5]
        ),
        "{choice:?}"
    );
    assert_eq!(grid, [16, 1, 1]);

    let builder = Builder::for_tests();
    let operand = builder
        .constant("operand", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let update = builder
        .constant("update", TensorType::new(vec![2, 2], GDt::E4M3FN))
        .unwrap();
    let index = builder
        .constant("index", TensorType::new(vec![], GDt::I32))
        .unwrap();
    let output = builder
        .dynamic_update_slice_dyn(operand, update, index, 1)
        .unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let (planned_plan, choice) = planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap();
    let Dispatch {
        body,
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(body.param_count, 4);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
                storage: kg::Storage::PackedRows { lane_bits: 8 },
                out_shape,
                op: kg::MovementOp::UpdateSlice {
                    update_shape,
                    axis: 1,
                    index: kg::UpdateIndex::Runtime,
                },
                ..
            })) if out_shape == &[2, 5] && update_shape == &[2, 2]
        ),
        "{choice:?}"
    );
    assert_eq!(grid, [4, 1, 1]);
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &crate::test_support::roomy_body_limits()
        )
        .unwrap()
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN DynamicUpdateSlice on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, TargetId::Gpu(backend));
}

#[test]
fn e4m3fn_scatter_update_rejects_empty_storage_and_mixed_dtypes() {
    for (pool, source_rows) in [(0, 1), (2, 0)] {
        let builder = Builder::for_tests();
        let base = builder
            .constant("base", TensorType::new(vec![pool, 5], GDt::E4M3FN))
            .unwrap();
        let src = builder
            .constant("src", TensorType::new(vec![source_rows, 5], GDt::E4M3FN))
            .unwrap();
        let slots = builder
            .constant("slots", TensorType::new(vec![source_rows], GDt::I32))
            .unwrap();
        let output = builder.scatter_update(base, src, slots).unwrap();
        let graph = builder.finish(output).unwrap();
        let eqn = &graph.eqns()[0];
        let refusal = refusal_of(
            plan_eqn(
                &graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &crate::test_support::roomy_body_limits(),
            ),
            "empty E4M3FN ScatterUpdate",
        );
        assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
        assert_eq!(refusal.op, OpKind::ScatterUpdate);
    }

    // An E4M3FN base under an f32 source never reaches the planner: `infer` types a `ScatterUpdate` over one
    // dtype only (POOT-1241), so it is an IR-level `ShapeError`.
    let err = OpKind::ScatterUpdate
        .infer(&[
            TensorType::new(vec![2, 5], GDt::E4M3FN),
            TensorType::f32(vec![1, 5]),
            TensorType::new(vec![1], GDt::I32),
        ])
        .expect_err(
            "a ScatterUpdate over a base and a source of two dtypes must be rejected by infer",
        );
    assert!(matches!(err, ShapeError::Dtype { .. }));

    // Card 529: an E4M3FN slot map no longer reaches the planner - `infer` rejects it for every
    // `ScatterUpdate`, so this is an IR-level `ShapeError` now, not a planner refusal.
    let err = OpKind::ScatterUpdate
        .infer(&[
            TensorType::new(vec![2, 5], GDt::E4M3FN),
            TensorType::new(vec![1, 5], GDt::E4M3FN),
            TensorType::new(vec![1], GDt::E4M3FN),
        ])
        .expect_err("E4M3FN ScatterUpdate with an E4M3FN slot map must be rejected by infer");
    assert!(matches!(
        err,
        ShapeError::DtypeOp {
            op: "ScatterUpdate",
            dtype: GDt::E4M3FN
        }
    ));
}

#[test]
fn e4m3fn_dynamic_update_slice_cache_keys_separate_forms_axes_windows_and_layouts() {
    fn key(
        operand_shape: Vec<usize>,
        update_shape: Vec<usize>,
        axis: usize,
        index: Option<usize>,
    ) -> String {
        let builder = Builder::for_tests();
        let operand = builder
            .constant("operand", TensorType::new(operand_shape, GDt::E4M3FN))
            .unwrap();
        let update = builder
            .constant("update", TensorType::new(update_shape, GDt::E4M3FN))
            .unwrap();
        let output = if let Some(index) = index {
            builder
                .dynamic_update_slice(operand, update, index, axis)
                .unwrap()
        } else {
            let index = builder
                .constant("index", TensorType::new(vec![], GDt::I32))
                .unwrap();
            builder
                .dynamic_update_slice_dyn(operand, update, index, axis)
                .unwrap()
        };
        let graph = builder.finish(output).unwrap();
        let eqn = &graph.eqns()[0];
        let planned_plan = plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap();
        let Dispatch { key, .. } = into_plain(planned_plan);
        key
    }

    let keys = [
        key(vec![3, 5], vec![1, 5], 0, Some(0)),
        key(vec![3, 5], vec![1, 5], 0, Some(1)),
        key(vec![3, 5], vec![2, 5], 0, Some(0)),
        key(vec![3, 5], vec![3, 2], 1, Some(1)),
        key(vec![3, 6], vec![1, 6], 0, Some(0)),
        key(vec![3, 5], vec![1, 5], 0, None),
    ];
    for left in 0..keys.len() {
        for right in left + 1..keys.len() {
            assert_ne!(keys[left], keys[right], "keys {left} and {right} collided");
        }
    }
}

#[test]
fn e4m3fn_dynamic_update_slice_rejects_zero_invalid_window_and_index_storage() {
    let builder = Builder::for_tests();
    let operand = builder
        .constant("operand", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let update = builder
        .constant("update", TensorType::new(vec![2, 2], GDt::E4M3FN))
        .unwrap();
    let output = builder.dynamic_update_slice(operand, update, 4, 1).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN DynamicUpdateSlice window past the axis",
    );
    assert_eq!(
        refusal.missing,
        Capability::OperandShape {
            value: operand.id(),
            shape: vec![2, 5],
            expected: vec![2, 6],
        }
    );
    assert_eq!(refusal.op, eqn.op);

    let builder = Builder::for_tests();
    let operand = builder
        .constant("operand", TensorType::new(vec![0, 5], GDt::E4M3FN))
        .unwrap();
    let update = builder
        .constant("update", TensorType::new(vec![0, 5], GDt::E4M3FN))
        .unwrap();
    let output = builder.dynamic_update_slice(operand, update, 0, 0).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "zero-element E4M3FN DynamicUpdateSlice",
    );
    assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    assert_eq!(refusal.op, eqn.op);

    // An F32 runtime index never reaches the planner: `infer` admits an I32 index only.
    let err = OpKind::DynamicUpdateSlice { axis: 1 }
        .infer(&[
            TensorType::new(vec![2, 5], GDt::E4M3FN),
            TensorType::new(vec![2, 2], GDt::E4M3FN),
            TensorType::scalar(GDt::F32),
        ])
        .expect_err("an F32 DynamicUpdateSlice index must be rejected by infer");
    assert!(matches!(
        err,
        ShapeError::DtypeOp {
            op: "DynamicUpdateSlice",
            dtype: GDt::F32
        }
    ));
}

/// POOT-725: a runtime `DynamicUpdateSlice` index is I32 and nothing else. An F32 index is refused by shape
/// inference (it never reaches the planner), and the I32 form plans one request whose index buffer the body
/// reads as `Slice<i32>`.
///
/// Mutation: admit `F32` in `check_index_dtype` (`op.rs`); the F32 row goes red (the index infers).
#[test]
fn dynamic_update_slice_runtime_index_is_i32_only() {
    let err = OpKind::DynamicUpdateSlice { axis: 0 }
        .infer(&[
            TensorType::f32(vec![4, 5]),
            TensorType::f32(vec![2, 5]),
            TensorType::scalar(GDt::F32),
        ])
        .expect_err("an F32 DynamicUpdateSlice index must be rejected by infer");
    assert!(matches!(
        err,
        ShapeError::DtypeOp {
            op: "DynamicUpdateSlice",
            dtype: GDt::F32
        }
    ));

    let builder = Builder::for_tests();
    let operand = builder
        .constant("operand", TensorType::f32(vec![4, 5]))
        .unwrap();
    let update = builder
        .constant("update", TensorType::f32(vec![2, 5]))
        .unwrap();
    let index = builder
        .constant("index", TensorType::scalar(GDt::I32))
        .unwrap();
    let output = builder
        .dynamic_update_slice_dyn(operand, update, index, 0)
        .unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = graph
        .eqns()
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::DynamicUpdateSlice { .. }))
        .unwrap();
    let (planned_plan, choice) = planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap();
    let Dispatch { body, .. } = into_plain(planned_plan);
    assert!(
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
                op: kg::MovementOp::UpdateSlice {
                    index: kg::UpdateIndex::Runtime,
                    ..
                },
                ..
            }))
        ),
        "{choice:?}"
    );
    // (operand, update, index, out): the index is the third data param.
    assert_eq!(slice_element_ty(&body, 3), &Ty::I32);
}

#[test]
fn e4m3fn_concat_preserves_n_input_contract_and_aliases_only_one_input_on_wgpu() {
    let builder = Builder::for_tests();
    let p0 = builder
        .constant("p0", TensorType::new(vec![2, 2, 3], GDt::E4M3FN))
        .unwrap();
    let p1 = builder
        .constant("p1", TensorType::new(vec![2, 1, 3], GDt::E4M3FN))
        .unwrap();
    let p2 = builder
        .constant("p2", TensorType::new(vec![2, 3, 3], GDt::E4M3FN))
        .unwrap();
    let output = builder.concat(1, &[p0, p1, p2]).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let shape = graph.aval(eqn.out).shape.clone();
    let (plan, choice) = planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap();
    let planned_plan = plan;
    let Dispatch {
        body,
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(shape, [2, 6, 3]);
    assert_eq!(body.param_count, 4);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
                storage: kg::Storage::PackedRows { lane_bits: 8 },
                op: kg::MovementOp::Concat { axis: 1, in_shapes },
                ..
            })) if in_shapes.len() == 3
        ),
        "{choice:?}"
    );
    assert_eq!(grid, [12, 1, 1]);
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &crate::test_support::roomy_body_limits()
        )
        .unwrap()
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN Concat on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, TargetId::Gpu(backend));

    let builder = Builder::for_tests();
    let a = builder
        .constant("a", TensorType::new(vec![2, 3], GDt::E4M3FN))
        .unwrap();
    let b = builder
        .constant("b", TensorType::new(vec![2, 2], GDt::E4M3FN))
        .unwrap();
    let output = builder.concat(1, &[a, b]).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let shape = graph.aval(eqn.out).shape.clone();
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch {
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(shape, [2, 5]);
    assert_eq!(grid, [4, 1, 1]);

    let builder = Builder::for_tests();
    let only = builder
        .constant("only", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let output = builder.concat(1, &[only]).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &crate::test_support::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == only.id()
       ));
}

#[test]
fn e4m3fn_concat_rejects_zero_element_storage_and_mixed_dtypes() {
    let builder = Builder::for_tests();
    let empty = builder
        .constant("empty", TensorType::new(vec![0, 5], GDt::E4M3FN))
        .unwrap();
    let full = builder
        .constant("full", TensorType::new(vec![1, 5], GDt::E4M3FN))
        .unwrap();
    let output = builder.concat(0, &[empty, full]).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "zero-element E4M3FN Concat",
    );
    assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    assert_eq!(refusal.op, eqn.op);

    // A concat of an E4M3FN part and a dense one is an IR-level `ShapeError`: `infer` types the parts of a
    // `Concat` over one dtype only (POOT-1241), so it never reaches the planner.
    let err = OpKind::Concat { axis: 0 }
        .infer(&[
            TensorType::new(vec![1, 5], GDt::E4M3FN),
            TensorType::new(vec![1, 5], GDt::F32),
        ])
        .expect_err("a Concat over two dtypes must be rejected by infer");
    assert!(matches!(err, ShapeError::Dtype { .. }));
}

#[test]
fn e4m3fn_broadcast_materializes_expansions_and_aliases_only_identity_on_wgpu() {
    for (input_shape, output_shape, expected_words) in [
        (vec![5], vec![3, 5], 6),
        (vec![2, 1, 5], vec![2, 3, 5], 12),
        (vec![1, 3], vec![2, 1, 3], 2),
    ] {
        let builder = Builder::for_tests();
        let input = builder
            .constant("raw", TensorType::new(input_shape.clone(), GDt::E4M3FN))
            .unwrap();
        let output = builder.broadcast(input, output_shape.clone()).unwrap();
        let graph = builder.finish(output).unwrap();
        let eqn = &graph.eqns()[0];
        let shape = graph.aval(eqn.out).shape.clone();
        assert_eq!(shape, output_shape);
        let planned_plan = plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap();
        let Dispatch {
            launch: Launch { grid, .. },
            ..
        } = into_plain(planned_plan);
        assert_eq!(grid, [expected_words, 1, 1]);
        assert!(is_plain(
            &plan_eqn(
                &graph,
                eqn,
                Backend::Nvptx,
                &default_caps_for(Backend::Nvptx),
                &crate::test_support::roomy_body_limits()
            )
            .unwrap()
        ));
        let backend = Backend::AmdGcn(AmdArch::gfx1151());
        let refusal = refusal_of(
            plan_eqn(
                &graph,
                eqn,
                backend,
                &default_caps_for(backend),
                &crate::test_support::roomy_body_limits(),
            ),
            "E4M3FN Broadcast on ROCm",
        );
        assert_eq!(refusal.missing, Capability::E4m3Storage);
        assert_eq!(refusal.op, eqn.op);
        assert_eq!(refusal.target, TargetId::Gpu(backend));
    }

    let builder = Builder::for_tests();
    let input = builder
        .constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let output = builder.broadcast(input, vec![2, 5]).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &crate::test_support::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == input.id()
       ));
}

#[test]
fn e4m3fn_empty_output_broadcast_is_rejected_by_name() {
    let builder = Builder::for_tests();
    let input = builder
        .constant("raw", TensorType::new(vec![1, 5], GDt::E4M3FN))
        .unwrap();
    let output = builder.broadcast(input, vec![0, 5]).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN Broadcast to an empty output",
    );
    assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    assert_eq!(refusal.op, eqn.op);
}

#[test]
fn e4m3fn_cast_and_reshape_plan_only_for_spirv_wgpu_value_executor() {
    let decode = {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![2, 5], GDt::E4M3FN))
            .unwrap();
        let y = b.cast(x, GDt::F32).unwrap();
        b.finish(y).unwrap()
    };
    let encode = {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![2, 5], GDt::F32))
            .unwrap();
        let y = b.cast(x, GDt::E4M3FN).unwrap();
        b.finish(y).unwrap()
    };
    let repack = {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![2, 5], GDt::E4M3FN))
            .unwrap();
        let y = b.reshape(x, vec![5, 2]).unwrap();
        b.finish(y).unwrap()
    };

    for graph in [&decode, &encode, &repack] {
        let eqn = &graph.eqns()[0];
        let shape = graph.aval(eqn.out).shape.clone();
        let planned_plan = plan_eqn(
            graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap();
        let Dispatch {
            launch: Launch { grid, .. },
            ..
        } = into_plain(planned_plan);
        if graph.aval(eqn.out).dtype == GDt::E4M3FN {
            let layout_words = shape[..shape.len() - 1].iter().product::<usize>()
                * shape.last().unwrap().div_ceil(4);
            assert_eq!(grid, [layout_words as u32, 1, 1]);
        }

        assert!(matches!(
            plan_eqn(graph,
                eqn,
                Backend::Nvptx,
                &default_caps_for(Backend::Nvptx),
                &crate::test_support::roomy_body_limits()
            )
            .unwrap(),
            plan if is_plain(&plan) || matches!(plan, Plan::Alias(_))
        ));
        let backend = Backend::AmdGcn(AmdArch::gfx1151());
        let refusal = refusal_of(
            plan_eqn(
                graph,
                eqn,
                backend,
                &default_caps_for(backend),
                &crate::test_support::roomy_body_limits(),
            ),
            "E4M3FN Cast and Reshape on ROCm",
        );
        assert_eq!(refusal.missing, Capability::E4m3Storage);
        assert_eq!(refusal.target, TargetId::Gpu(backend));
    }
}

#[test]
fn e4m3fn_materializing_cast_and_reshape_reject_zero_element_outputs() {
    let encode = {
        let builder = Builder::for_tests();
        let input = builder
            .constant("empty", TensorType::new(vec![0, 5], GDt::F32))
            .unwrap();
        let output = builder.cast(input, GDt::E4M3FN).unwrap();
        builder.finish(output).unwrap()
    };
    let decode = {
        let builder = Builder::for_tests();
        let input = builder
            .constant("empty", TensorType::new(vec![0, 5], GDt::E4M3FN))
            .unwrap();
        let output = builder.cast(input, GDt::F32).unwrap();
        builder.finish(output).unwrap()
    };
    let repack = {
        let builder = Builder::for_tests();
        let input = builder
            .constant("empty", TensorType::new(vec![0, 5], GDt::E4M3FN))
            .unwrap();
        let output = builder.reshape(input, vec![0, 7]).unwrap();
        builder.finish(output).unwrap()
    };

    for graph in [&encode, &decode, &repack] {
        let eqn = &graph.eqns()[0];
        let refusal = refusal_of(
            plan_eqn(
                graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &crate::test_support::roomy_body_limits(),
            ),
            "zero-element E4M3FN Cast and Reshape",
        );
        assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    }
}

#[test]
fn e4m3fn_materializing_dispatch_rejects_a_grid_beyond_wgpu_x_limit() {
    let grid_cap = default_caps_for(Backend::SpirvVulkan).max_grid[0] as usize;
    let rows = grid_cap * 256 + 1;
    let encode = {
        let builder = Builder::for_tests();
        let input = builder
            .constant("large", TensorType::new(vec![rows, 1], GDt::F32))
            .unwrap();
        let output = builder.cast(input, GDt::E4M3FN).unwrap();
        builder.finish(output).unwrap()
    };
    let decode = {
        let builder = Builder::for_tests();
        let input = builder
            .constant("large", TensorType::new(vec![rows, 1], GDt::E4M3FN))
            .unwrap();
        let output = builder.cast(input, GDt::F32).unwrap();
        builder.finish(output).unwrap()
    };

    for graph in [&encode, &decode] {
        let eqn = &graph.eqns()[0];
        let refusal = refusal_of(
            plan_eqn(
                graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &crate::test_support::roomy_body_limits(),
            ),
            "E4M3FN dispatch past the wgpu x limit",
        );
        let Capability::DispatchGridLimit {
            threads,
            workgroups_x,
            limit,
            ..
        } = refusal.missing
        else {
            panic!("expected a dispatch-grid refusal, got {refusal:?}");
        };
        assert_eq!(limit, grid_cap);
        assert!(workgroups_x > limit, "{workgroups_x} <= {limit}");
        assert!(threads > 0);
    }
}

/// SC-003 (card 546b): the planner's generic one-thread-per-output workgroup
/// bump (`finalize`, `planner.rs`) reads the dispatch cap from the caller's own `DeviceCaps.max_grid`,
/// never a baked wgpu-specific `65_535` literal - a fixture whose `max_grid` is far below the real
/// wgpu cap must still get a dispatch that respects IT.
///
/// MUTATION (recorded here, never left in the tree): restore the `65_535` literal `finalize` used to
/// compare `out_numel.div_ceil(wg0)` against. At this fixture's `max_grid = 1024`, `out_numel =
/// 1024*64 + 1` needs 16385 default-width groups - under the restored 65535 the bump never fires
/// (`wg0` stays the kernel's stock 64), so the planned dispatch of `16385 > 1024` workgroups on X
/// silently exceeds this fixture's real cap and the row goes red (the `wg0 == 256` assertion fails,
/// `wg0 == 64` instead). Restored: green again.
#[test]
fn sc003_grid_cap_bump_respects_custom_device_caps_not_hardcoded_65535() {
    let small_caps = poot_target::DeviceCaps {
        max_grid: [1024, 1024, 1024],
        ..default_caps_for(Backend::SpirvVulkan)
    };
    // 1024*64 + 1 default-width (64-lane) groups clears the real wgpu cap (65535) comfortably but
    // overshoots this fixture's 1024 - the wg-bump to 256 lanes must fire to bring workgroups_x back
    // under 1024 (1024*64+1 div_ceil 256 = 257 <= 1024).
    let out_numel = 1024 * 64 + 1;
    let bld = Builder::for_tests();
    let a = bld.constant("a", TensorType::f32(vec![out_numel])).unwrap();
    let b = bld.constant("b", TensorType::f32(vec![out_numel])).unwrap();
    let y = bld.binary(GBinOp::Add, a, b).unwrap();
    let g = bld.finish(y).unwrap();
    let eqn = &g.eqns()[0];
    let plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &small_caps,
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let planned_plan = plan;
    let Dispatch { body, .. } = into_plain(planned_plan);
    let wg0 = body.workgroup_size[0] as usize;
    assert_eq!(
        wg0, 256,
        "the generic wg-bump must fire under the fixture's own max_grid=1024, not the old 65535"
    );
    assert!(
        out_numel.div_ceil(wg0) <= 1024,
        "bumped dispatch ({} groups) must clear the fixture's max_grid=1024",
        out_numel.div_ceil(wg0)
    );
}

/// SC-003 extension (card 546b): `is_elementwise_2d_in`'s
/// fold-to-2D threshold must also read the caller's own `DeviceCaps.max_grid`, not a baked
/// `65_535 * 256` wgpu-specific ceiling. Above, `sc003_grid_cap_bump_respects_custom_device_caps_not_hardcoded_65535`
/// only exercises `finalize`'s generic 1-D wg-bump (its `out_numel` is nowhere near
/// `65535*256`); this fixture's `out_numel` sits ABOVE the fixture's own fold ceiling
/// (`max_grid[0] * ELEMENTWISE_2D_WIDTH = 262144`) but far BELOW the old hardcoded
/// `65535*256 = 16,776,960` one, so it is the one case that tells the two ceilings apart: the
/// generic wg-bump alone (max wg0 = 256) cannot bring a flat 1-D dispatch of this size back under
/// `max_grid[0] = 1024`, so a correct plan must fold to 2-D instead.
///
/// MUTATION (recorded here, never left in the tree): restore the `65_535 * 256` literal
/// `is_elementwise_2d_in`'s `OpKind::Binary` arm compared `out_numel` against. At this fixture's
/// `out_numel = 262145`, that restored literal is nowhere near `16,776,960`, so the op stays on the
/// 1-D path; `finalize`'s wg-bump maxes out at `wg0 = 256`, giving `workgroups_x =
/// 262145.div_ceil(256) = 1025 > 1024` - past the fixture's own cap - and the row goes red (the
/// `workgroups_x <= 1024` assertion fails, `1025 > 1024`). Restored: green again.
#[test]
fn sc003_elementwise_2d_fold_respects_custom_device_caps_not_hardcoded_65535x256() {
    let small_caps = poot_target::DeviceCaps {
        max_grid: [1024, 1024, 1024],
        ..default_caps_for(Backend::SpirvVulkan)
    };
    // One past this fixture's own fold ceiling (max_grid[0] * ELEMENTWISE_2D_WIDTH = 1024*256 =
    // 262144); nowhere near the old hardcoded 65535*256 = 16,776,960 ceiling.
    let out_numel = 1024 * 256 + 1;
    let bld = Builder::for_tests();
    let a = bld.constant("a", TensorType::f32(vec![out_numel])).unwrap();
    let b = bld.constant("b", TensorType::f32(vec![out_numel])).unwrap();
    let y = bld.binary(GBinOp::Add, a, b).unwrap();
    let g = bld.finish(y).unwrap();
    let eqn = &g.eqns()[0];
    let plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &small_caps,
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let planned_plan = plan;
    let Dispatch {
        body,
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    let wg0 = body.workgroup_size[0].max(1) as usize;
    let wg1 = body.workgroup_size[1].max(1) as usize;
    // `grid` is the total thread count (card 525: `types.rs`'s `Launch` doc); the executor
    // derives the dispatched workgroup count per axis by CEILING division against `workgroup_size`
    // (`poot-wgpu-runtime`'s `workgroup_count`), never a floor division that would silently truncate.
    let workgroups_x = (grid[0] as usize).div_ceil(wg0);
    let workgroups_y = (grid[1] as usize).div_ceil(wg1);
    assert!(
        workgroups_x > 1,
        "sanity: the fixture must actually need more than one X group, or this test proves nothing \
         about the 2-D fold"
    );
    assert!(
        workgroups_x <= small_caps.max_grid[0] as usize,
        "X dispatch ({workgroups_x} groups) must clear the fixture's max_grid[0]=1024: the generic \
         wg-bump alone cannot, so the plan must have folded to a 2-D grid"
    );
    assert!(
        workgroups_y <= small_caps.max_grid[1] as usize,
        "Y dispatch ({workgroups_y} groups) must clear the fixture's max_grid[1]=1024"
    );
}

#[test]
fn e4m3fn_same_dtype_cast_is_an_alias_only_on_spirv_wgpu_value_path() {
    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let same_dtype = builder.cast(x, GDt::E4M3FN).unwrap();
    let graph = builder.finish(same_dtype).unwrap();
    let eqn = &graph.eqns()[0];

    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &crate::test_support::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::Nvptx,
       &default_caps_for(Backend::Nvptx),
    &crate::test_support::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &crate::test_support::roomy_body_limits(),
        ),
        "same-dtype E4M3FN Cast on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.target, TargetId::Gpu(backend));
}

#[test]
fn e4m3fn_transpose_materializes_nonidentity_and_aliases_only_identity_on_wgpu() {
    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let transposed = builder.transpose(x, vec![1, 0]).unwrap();
    let graph = builder.finish(transposed).unwrap();
    let eqn = &graph.eqns()[0];
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch {
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(grid, [5, 1, 1]);
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &crate::test_support::roomy_body_limits()
        )
        .unwrap()
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN Transpose on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, TargetId::Gpu(backend));

    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let identity = builder.transpose(x, vec![0, 1]).unwrap();
    let graph = builder.finish(identity).unwrap();
    let eqn = &graph.eqns()[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &crate::test_support::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));

    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![2, 1, 5], GDt::E4M3FN))
        .unwrap();
    let moved_unit_axis = builder.transpose(x, vec![1, 0, 2]).unwrap();
    let graph = builder.finish(moved_unit_axis).unwrap();
    let eqn = &graph.eqns()[0];
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits()
        )
        .unwrap()
    ));
}

#[test]
fn e4m3fn_zero_element_transpose_is_rejected_by_name() {
    for perm in [vec![1, 0], vec![0, 1]] {
        let builder = Builder::for_tests();
        let x = builder
            .constant("raw", TensorType::new(vec![0, 5], GDt::E4M3FN))
            .unwrap();
        let transposed = builder.transpose(x, perm).unwrap();
        let graph = builder.finish(transposed).unwrap();
        let eqn = &graph.eqns()[0];
        let refusal = refusal_of(
            plan_eqn(
                &graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &crate::test_support::roomy_body_limits(),
            ),
            "zero-element E4M3FN Transpose",
        );
        assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
        assert_eq!(refusal.op, eqn.op);
    }
}

#[test]
fn e4m3fn_slice_materializes_proper_ranges_and_aliases_only_full_range_on_wgpu() {
    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let sliced = builder.slice(x, 1, 1, 4).unwrap();
    let graph = builder.finish(sliced).unwrap();
    let eqn = &graph.eqns()[0];
    let shape = graph.aval(eqn.out).shape.clone();
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch {
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(shape, [2, 3]);
    assert_eq!(grid, [2, 1, 1]);
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &crate::test_support::roomy_body_limits()
        )
        .unwrap()
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN Slice on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, TargetId::Gpu(backend));

    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let identity = builder.slice(x, 1, 0, 5).unwrap();
    let graph = builder.finish(identity).unwrap();
    let eqn = &graph.eqns()[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &crate::test_support::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));

    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![3, 5], GDt::E4M3FN))
        .unwrap();
    let contiguous_rows = builder.slice(x, 0, 1, 3).unwrap();
    let graph = builder.finish(contiguous_rows).unwrap();
    let eqn = &graph.eqns()[0];
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch {
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(grid, [4, 1, 1]);
}

#[test]
fn e4m3fn_empty_output_slice_is_rejected_by_name() {
    let builder = Builder::for_tests();
    let x = builder
        .constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN))
        .unwrap();
    let empty = builder.slice(x, 1, 2, 2).unwrap();
    let graph = builder.finish(empty).unwrap();
    let eqn = &graph.eqns()[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN Slice to an empty output",
    );
    assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    assert_eq!(refusal.op, eqn.op);
}

#[test]
fn e4m3fn_gather_always_materializes_exact_packed_output_on_wgpu() {
    for (data_shape, index_shape, axis, expected_shape, expected_words) in [
        (vec![4, 5], vec![2], 0, vec![2, 5], 4),
        (vec![2, 3, 5], vec![2], 1, vec![2, 2, 5], 8),
        (vec![3, 5], vec![], 0, vec![5], 2),
    ] {
        let builder = Builder::for_tests();
        let data = builder
            .constant("raw", TensorType::new(data_shape, GDt::E4M3FN))
            .unwrap();
        let index = builder
            .constant("index", TensorType::new(index_shape, GDt::I32))
            .unwrap();
        let gathered = builder.gather(data, axis, index).unwrap();
        let graph = builder.finish(gathered).unwrap();
        let eqn = &graph.eqns()[0];
        let shape = graph.aval(eqn.out).shape.clone();
        assert_eq!(shape, expected_shape);
        let planned_plan = plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap();
        let Dispatch {
            launch: Launch { grid, .. },
            ..
        } = into_plain(planned_plan);
        assert_eq!(grid, [expected_words, 1, 1]);
    }

    let builder = Builder::for_tests();
    let data = builder
        .constant("raw", TensorType::new(vec![4, 5], GDt::E4M3FN))
        .unwrap();
    let index = builder
        .constant("index", TensorType::new(vec![4], GDt::I32))
        .unwrap();
    let gathered = builder.gather(data, 0, index).unwrap();
    let graph = builder.finish(gathered).unwrap();
    let eqn = &graph.eqns()[0];
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &crate::test_support::roomy_body_limits()
        )
        .unwrap()
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN Gather on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, TargetId::Gpu(backend));
}

#[test]
fn e4m3fn_gather_rejects_empty_output_and_non_i32_indices_by_name() {
    let builder = Builder::for_tests();
    let data = builder
        .constant("raw", TensorType::new(vec![4, 5], GDt::E4M3FN))
        .unwrap();
    let index = builder
        .constant("index", TensorType::new(vec![0], GDt::I32))
        .unwrap();
    let gathered = builder.gather(data, 0, index).unwrap();
    let graph = builder.finish(gathered).unwrap();
    let eqn = &graph.eqns()[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        ),
        "E4M3FN Gather to an empty output",
    );
    assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    assert_eq!(refusal.op, eqn.op);

    // Card 529: an E4M3FN index no longer reaches the planner at all - `infer` rejects it
    // for every `Gather`, whatever the data dtype, so these two cases are IR-level `ShapeError`s now,
    // not planner refusals.
    let err = OpKind::Gather { axis: 0 }
        .infer(&[
            TensorType::new(vec![4, 5], GDt::E4M3FN),
            TensorType::new(vec![2], GDt::E4M3FN),
        ])
        .expect_err("E4M3FN Gather with an E4M3FN index must be rejected by infer");
    assert!(matches!(
        err,
        ShapeError::DtypeOp {
            op: "Gather",
            dtype: GDt::E4M3FN
        }
    ));

    let err = OpKind::Gather { axis: 0 }
        .infer(&[
            TensorType::f32(vec![4, 5]),
            TensorType::new(vec![2], GDt::E4M3FN),
        ])
        .expect_err("E4M3FN Gather over dense data with an E4M3FN index must be rejected by infer");
    assert!(matches!(
        err,
        ShapeError::DtypeOp {
            op: "Gather",
            dtype: GDt::E4M3FN
        }
    ));
}

/// A 16-aligned matmul `[1,16,32] @ [32,16]` over bf16/f16 operands that accumulates and stores F32.
fn mixed_matmul_graph(op_dt: GDt) -> Graph {
    let (m, k, n) = (16, 32, 16);
    let bld = Builder::for_tests();
    let a = bld
        .constant("a", TensorType::new(vec![1, m, k], op_dt))
        .unwrap();
    let w = bld
        .constant("w", TensorType::new(vec![k, n], op_dt))
        .unwrap();
    let mm = bld.matmul_f32_accumulate(a, w).unwrap();
    bld.finish(mm).unwrap()
}

#[test]
fn attention_rank4_matmul_routes_through_batched_tiled_gemm() {
    // A prefill-attention-shaped matmul [B,H,M,K] @ [B,H,K,N] -> [B,H,M,N] (M>1) routes through the register-tiled
    // GEMM with one `[K, N]` slab per block (B*H blocks), not the naive batched kernel.
    let bld = Builder::for_tests();
    let q = bld
        .constant("q", TensorType::new(vec![1, 14, 64, 64], GDt::F32))
        .unwrap();
    let kt = bld
        .constant("kt", TensorType::new(vec![1, 14, 64, 64], GDt::F32))
        .unwrap();
    let mm = bld.matmul(q, kt).unwrap();
    let g = bld.finish(mm).unwrap();
    let (plan, choice) = planned_with_choice(&g, &g.eqns()[0], Backend::SpirvVulkan).unwrap();
    assert!(
        kn_tiled(&choice).is_some_and(|spec| spec.meta[..3] == [64, 64, 64]),
        "attention matmul should take the tiled path: {choice:?}"
    );
    assert_eq!(
        one_dispatch(plan).body().params().count(),
        4,
        "[q, kt, meta, out]: the 14 blocks are the output's own length over [m_per_block, out_per_block]"
    );
    // At a long context the same body splits into dispatches over disjoint tile ranges (the device's watchdog
    // budget), each a tiled body, never the naive batched kernel.
    let bld2 = Builder::for_tests();
    let q2 = bld2
        .constant("q", TensorType::new(vec![1, 14, 2048, 2048], GDt::F32))
        .unwrap();
    let k2 = bld2
        .constant("k", TensorType::new(vec![1, 14, 2048, 2048], GDt::F32))
        .unwrap();
    let mm2 = bld2.matmul(q2, k2).unwrap();
    let g2 = bld2.finish(mm2).unwrap();
    let eqn2 = &g2.eqns()[0];
    let (plan2, _) = planned_with_choice(&g2, eqn2, Backend::SpirvVulkan).unwrap();
    assert!(
        plan2.dispatches().len() > 1
            && plan2
                .dispatches()
                .iter()
                .all(|d| kn_tiled(d.choice())
                    .is_some_and(|spec| spec.meta[..3] == [2048, 2048, 2048])),
        "over-cap attention matmul should split into tiled dispatches: {plan2:?}"
    );
}

#[test]
fn decode_attn_scores_v_matmul_lowers_to_the_lds_gemv_kernel() {
    // Card 143 Lever 1b: decode `scores @ V` (M=1, Hq rows already GQA-repeat_kv-expanded on both
    // operands, cap=300 not a multiple of GEMV_WIDTH=128) takes the LDS-parallel attn-scores-v GEMV on
    // every backend that lowers WorkgroupLocalWrite/Barrier, like is_decode_gemv's M=1 gate.
    let bld = Builder::for_tests();
    let probs = bld
        .constant("probs", TensorType::new(vec![1, 8, 1, 300], GDt::F32))
        .unwrap();
    let v = bld
        .constant("v", TensorType::new(vec![1, 8, 300, 64], GDt::F32))
        .unwrap();
    let mm = bld.matmul(probs, v).unwrap();
    let g = bld.finish(mm).unwrap();
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let choice = matmul_choice(&g, backend);
        assert!(
            matches!(
                contraction(&choice),
                Some(ContractionSpec::AttnScoresVGemv { .. })
            ),
            "decode scores@V should take the LDS-parallel attn GEMV on {backend:?}: got {choice:?}"
        );
    }

    // Q @ K^T has the same batched-M==1 shape (q[1,Hq,1,D] @ kt[1,Hq,D,cap] -> [1,Hq,1,cap]) but its
    // second operand is a Transpose eqn's direct output and must stay off this kernel (already well
    // parallelized: large output Hq*cap, short reduction D).
    let bldq = Builder::for_tests();
    let q = bldq
        .constant("q", TensorType::new(vec![1, 8, 1, 64], GDt::F32))
        .unwrap();
    let k = bldq
        .constant("k", TensorType::new(vec![1, 8, 300, 64], GDt::F32))
        .unwrap();
    let kt = bldq.transpose(k, vec![0, 1, 3, 2]).unwrap();
    let scores = bldq.matmul(q, kt).unwrap();
    let gq = bldq.finish(scores).unwrap();
    let choice_q = matmul_choice(&gq, Backend::SpirvVulkan);
    assert!(
        !matches!(
            contraction(&choice_q),
            Some(ContractionSpec::AttnScoresVGemv { .. })
        ),
        "Q@K^T must not take the scores@V attn GEMV path: got {choice_q:?}"
    );

    // M>1 (prefill) stays off this M=1-only path (the batched tiled GEMM handles it).
    let bldp = Builder::for_tests();
    let probs_p = bldp
        .constant("probs", TensorType::new(vec![1, 8, 4, 300], GDt::F32))
        .unwrap();
    let v_p = bldp
        .constant("v", TensorType::new(vec![1, 8, 300, 64], GDt::F32))
        .unwrap();
    let mm_p = bldp.matmul(probs_p, v_p).unwrap();
    let gp = bldp.finish(mm_p).unwrap();
    let choice_p = matmul_choice(&gp, Backend::SpirvVulkan);
    assert!(
        !matches!(
            contraction(&choice_p),
            Some(ContractionSpec::AttnScoresVGemv { .. })
        ),
        "M>1 scores@V should stay off the decode-only attn GEMV: got {choice_p:?}"
    );
}

#[test]
fn gdn_decode_matmuls_route_as_expected_on_amdgcn() {
    // Pins the routing of `gated_delta_net_decode`'s 3 per-layer decode MatMul eqns
    // (`crates/poot-graph-ir/src/ops.rs`, H=32, D=128) on `Backend::AmdGcn`:
    //   - `kv = k @ s_decayed`   [1,32,1,128] @ [1,32,128,128] -> [1,32,1,128]  (M=1, operand-2 not a
    //     Transpose output) hits `is_decode_attn_gemv` -> the LDS-parallel `attn_scores_v_gemv_lds`
    //     kernel, same as decode `scores @ V` above.
    //   - `o = q_scaled @ s_out` has the same shape/operand-producer as `kv` -> same kernel.
    //   - `outer = k_col @ delta` [1,32,128,1] @ [1,32,1,128] -> [1,32,128,128] (M=128) takes the generated
    //     batched-weight tiled GEMM on every backend: the LDS-array bodies compile for AmdGcn (hardware-verified,
    //     POOT-1237), so no backend withholds it.
    let amd = Backend::AmdGcn(AmdArch::gfx1151());
    const H: usize = 32;
    const D: usize = 128;

    // kv = k[1,H,1,D] @ s_decayed[1,H,D,D] -> [1,H,1,D]
    let bld_kv = Builder::for_tests();
    let k = bld_kv
        .constant("k", TensorType::new(vec![1, H, 1, D], GDt::F32))
        .unwrap();
    let s_decayed = bld_kv
        .constant("s_decayed", TensorType::new(vec![1, H, D, D], GDt::F32))
        .unwrap();
    let kv = bld_kv.matmul(k, s_decayed).unwrap();
    let g_kv = bld_kv.finish(kv).unwrap();
    let choice_kv = matmul_choice(&g_kv, amd);
    assert!(
        matches!(
            contraction(&choice_kv),
            Some(ContractionSpec::AttnScoresVGemv {
                cap: 128,
                d: 128,
                numel: 4096,
                ..
            })
        ),
        "GDN kv=k@s_decayed should take the LDS-parallel attn GEMV on AmdGcn: got {choice_kv:?}"
    );

    // o = q_scaled[1,H,1,D] @ s_out[1,H,D,D] -> [1,H,1,D] (same shape/producer pattern as kv)
    let bld_o = Builder::for_tests();
    let q_scaled = bld_o
        .constant("q_scaled", TensorType::new(vec![1, H, 1, D], GDt::F32))
        .unwrap();
    let s_out = bld_o
        .constant("s_out", TensorType::new(vec![1, H, D, D], GDt::F32))
        .unwrap();
    let o = bld_o.matmul(q_scaled, s_out).unwrap();
    let g_o = bld_o.finish(o).unwrap();
    let choice_o = matmul_choice(&g_o, amd);
    assert!(
        matches!(
            contraction(&choice_o),
            Some(ContractionSpec::AttnScoresVGemv {
                cap: 128,
                d: 128,
                numel: 4096,
                ..
            })
        ),
        "GDN o=q_scaled@s_out should take the LDS-parallel attn GEMV on AmdGcn: got {choice_o:?}"
    );

    // outer = k_col[1,H,D,1] @ delta[1,H,1,D] -> [1,H,D,D]  (M=128: rank-1 state update)
    let bld_outer = Builder::for_tests();
    let k_col = bld_outer
        .constant("k_col", TensorType::new(vec![1, H, D, 1], GDt::F32))
        .unwrap();
    let delta = bld_outer
        .constant("delta", TensorType::new(vec![1, H, 1, D], GDt::F32))
        .unwrap();
    let outer = bld_outer.matmul(k_col, delta).unwrap();
    let g_outer = bld_outer.finish(outer).unwrap();
    let choice_outer = matmul_choice(&g_outer, amd);
    assert!(
        kn_tiled(&choice_outer).is_some_and(|spec| spec.meta[..3] == [1, 128, 128]),
        "GDN outer=k_col@delta should take the generated batched tiled GEMM on AmdGcn: got {choice_outer:?}"
    );
}

/// A mixed matmul (bf16 or f16 operands, f32 output) has no lowering on any backend: every matmul body reads one
/// element type, so planned at f32 it would read the two-byte operand buffers as f32 words. The planner refuses
/// the pair by name, and the widening pass is what makes it plannable. Mutation: drop the operand-dtype guard in
/// `planner/matmul.rs`; the pair plans the tiled-or-serial f32 body and every row goes red.
#[test]
fn mixed_dtype_matmul_is_refused_on_every_backend() {
    for operand in [GDt::BF16, GDt::F16] {
        let gm = mixed_matmul_graph(operand);
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
            Backend::AmdGcn(AmdArch::new("gfx942", 64)),
        ] {
            let r = try_matmul_plan(&gm, backend);
            assert!(
                matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
                "{operand:?} -> f32 matmul on {backend:?} must be refused, not planned at f32: got {r:?}"
            );
        }
    }
}

// --- card 235: the decode-time mixed-operand-dtype MatMul widen-cast pass ---

/// A matmul `[1,m,k] @ [k,n]` with a realistic operand-dtype mismatch: an F32 activation against a BF16
/// weight (the loader types every native-bf16 checkpoint's projection weights as bf16, independent of
/// the activation). `MatMul::infer` takes the output dtype from the first operand, so the output is F32
/// without a white-box override. This is the shape `plan_eqn`'s mixed-operand-dtype guard sees on a real
/// decode graph.
fn mixed_operand_matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let bld = Builder::for_tests();
    let a = bld
        .constant("a", TensorType::new(vec![1, m, k], GDt::F32))
        .unwrap();
    let w = bld
        .constant("w", TensorType::new(vec![k, n], GDt::BF16))
        .unwrap();
    let mm = bld.matmul(a, w).unwrap();
    bld.finish(mm).unwrap()
}

#[test]
fn widen_inserts_cast_for_decode_shaped_mixed_matmul() {
    // Card 235: every matmul body reads one element type, so without the widen pass every backend that is not
    // the refused decode projection (NVPTX, and any other shape) rejects this graph.
    let g = mixed_operand_matmul_graph(1, 32, 16);
    // NVPTX has no refused column Gemv (PTX has its own prepare rule), so it still needs the widening Cast.
    let backend = Backend::Nvptx;
    let r = try_matmul_plan(&g, backend);
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "unwidened decode-shaped mixed matmul should be rejected on {backend:?}: got {r:?}"
    );

    let gw = widen_mismatched_matmul_dtypes(&g, &default_caps_for(backend)).unwrap();
    let mm = gw
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .expect("matmul eqn retained");
    // both matmul operands read F32 (the odt): `a` already was, `w` gained a Cast.
    for (i, op) in mm.inputs.iter().take(2).enumerate() {
        if let poot_graph_ir::Operand::Value(v) = op {
            assert_eq!(
                gw.aval(*v).dtype,
                GDt::F32,
                "widened operand {i} should be F32 on {backend:?}"
            );
        }
    }
    let n_casts = gw
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Cast { to: GDt::F32 }))
        .count();
    assert_eq!(
        n_casts, 1,
        "only the mismatched (BF16) operand should gain a widening cast on {backend:?}: {n_casts}"
    );
    let r = try_matmul_plan(&gw, backend);
    assert!(
        r.is_ok(),
        "widened decode-shaped mixed matmul should now plan on {backend:?}: got {r:?}"
    );

    // SpirvVulkan / AmdGcn: a BF16 `[K, N]` decode projection has no lowering (the column-tile body is
    // retired), so the widening pass leaves it mixed with no Cast and `plan_eqn` refuses it by name.
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::AmdGcn(AmdArch::new("gfx942", 64)),
    ] {
        let r = try_matmul_plan(&g, backend);
        assert!(
            matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
            "a BF16 [K, N] decode projection must be refused on {backend:?}: got {r:?}"
        );
        let gw = widen_mismatched_matmul_dtypes(&g, &default_caps_for(backend)).unwrap();
        assert!(
            !gw.eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Cast { .. })),
            "the refused decode projection must not gain a widening Cast on {backend:?}"
        );
        assert_eq!(
            gw.aval(const_id(&gw, "w")).dtype,
            GDt::BF16,
            "the weight must stay BF16 on {backend:?}"
        );
    }
}

/// A 16-aligned mixed matmul (bf16 operands, f32 output) gains a widening `Cast` on each operand on every backend
/// and then plans the register-tiled GEMM: no backend reads the narrow pair directly. Mutation: skip the widening
/// of a 16-aligned matmul in `widen_mismatched_matmul_dtypes`; no cast is added and the row goes red.
#[test]
fn widen_casts_an_aligned_mixed_matmul_onto_the_tiled_gemm() {
    let gm = mixed_matmul_graph(GDt::BF16);
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let gw = widen_mismatched_matmul_dtypes(&gm, &default_caps_for(backend)).unwrap();
        let casts = gw
            .eqns()
            .iter()
            .filter(|e| matches!(e.op, OpKind::Cast { to: GDt::F32 }))
            .count();
        assert_eq!(casts, 2, "both bf16 operands are widened on {backend:?}");
        let choice = matmul_choice(&gw, backend);
        assert!(
            kn_tiled(&choice).is_some(),
            "the widened matmul takes the register-tiled GEMM on {backend:?}: got {choice:?}"
        );
    }
}

/// Round an f32 to bf16 precision (round-to-nearest-even), widened back to f32 (bf16 is the top 16 bits
/// of an f32). Test-only copy of `poot_eval`'s private `round_bf16` (see `crates/poot-eval/src/lib.rs`),
/// to simulate the values of a bf16 safetensors checkpoint without depending on poot_eval internals.
fn round_bf16(f: f32) -> f32 {
    if f.is_nan() {
        return f;
    }
    let bits = f.to_bits();
    let bias = 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(bias)) & 0xFFFF_0000)
}

#[test]
fn widen_preserves_cpu_oracle_numerics() {
    // Card 235: the widen-then-matmul round trip must not change results beyond the bf16 precision loss
    // already incurred when the checkpoint was loaded as bf16 (simulated by rounding `w`'s data through
    // bf16 before it enters the graph). `poot_eval`'s `Cast{to: F32}` is the identity on f32-resident data
    // and `MatMul`'s eval rounds only by output dtype, so the CPU oracle is unaffected by widening: the
    // pass changes what `plan_eqn` accepts, not what the graph computes.
    let (m, k, n) = (1, 37, 41); // deliberately non-16-aligned: decode-realistic
    let g = mixed_operand_matmul_graph(m, k, n);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.037).sin() * 3.0).collect();
    // Round through bf16 up front, as a bf16 safetensors checkpoint already is (the loader
    // reads the raw bf16 bytes, widened to f32 by the loader).
    let w_data: Vec<f32> = (0..k * n)
        .map(|i| round_bf16((i as f32 * 0.019).cos() * 7.0))
        .collect();

    let mut inputs: HashMap<ValueId, poot_eval::Value> = HashMap::new();
    inputs.insert(
        g.inputs()[0],
        poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![1, m, k], a_data.clone())),
    );
    inputs.insert(
        g.inputs()[1],
        // `w_data` is already bf16-exact, so its words are the high halves of the f32 bit patterns.
        poot_eval::Value::from(poot_tensor::HostTensor::bf16(
            vec![k, n],
            w_data.iter().map(|v| (v.to_bits() >> 16) as u16).collect(),
        )),
    );
    let unwidened = poot_eval::eval(&g, &inputs, poot_eval::EvalOptions::default())
        .expect("cpu eval (unwidened)")
        .output
        .into_host()
        .expect("dense unwidened output");

    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let gw = widen_mismatched_matmul_dtypes(&g, &default_caps_for(backend)).unwrap();
        let widened = poot_eval::eval(&gw, &inputs, poot_eval::EvalOptions::default())
            .expect("cpu eval (widened)")
            .output
            .into_host()
            .expect("dense widened output");
        assert_eq!(
            widened.as_f32().unwrap(),
            unwidened.as_f32().unwrap(),
            "widening must not change the CPU-oracle result on {backend:?}"
        );
    }

    // It also matches a plain all-f32 reference graph fed the same (bf16-rounded) weight data: widening
    // adds no precision loss beyond the bf16 rounding applied above.
    let gf = matmul_graph(m, k, n, GDt::F32);
    let mut inputs_f32: HashMap<ValueId, poot_eval::Value> = HashMap::new();
    inputs_f32.insert(
        gf.inputs()[0],
        poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![1, m, k], a_data)),
    );
    inputs_f32.insert(
        gf.inputs()[1],
        poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![k, n], w_data)),
    );
    let f32_ref = poot_eval::eval(&gf, &inputs_f32, poot_eval::EvalOptions::default())
        .expect("cpu eval (f32 reference)")
        .output
        .into_host()
        .expect("dense f32 reference output");
    assert_eq!(
        unwidened.as_f32().unwrap(),
        f32_ref.as_f32().unwrap(),
        "the mismatched-dtype graph's oracle result should already match the all-f32 reference \
         (poot_eval ignores operand dtype for MatMul, only the Cast/output-dtype rounding matters)"
    );
}

fn cast_planned(in_dt: GDt, to: GDt, backend: Backend) -> Result<(Plan, KernelChoice), PlanError> {
    let bld = Builder::for_tests();
    let x = bld.constant("x", TensorType::new(vec![8], in_dt)).unwrap();
    let c = bld.cast(x, to).unwrap();
    let g = bld.finish(c).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Cast { .. }))
        .expect("a cast eqn");
    planned_with_choice(&g, eqn, backend)
}

fn cast_plan(in_dt: GDt, to: GDt, backend: Backend) -> Result<Plan, PlanError> {
    cast_planned(in_dt, to, backend).map(|(plan, _)| plan)
}

#[test]
fn cast_arm_matches_dtype_pairs_exhaustively() {
    // The "cast(i32->f32) emits a bf16 kernel" landmine: the Cast arm used to dispatch on the target
    // dtype only, so every non-F16 source of a Cast-to-F32 got `cast_bf16_to_f32`, reinterpreting an i32
    // source's bytes as packed bf16 halves and diverging from the CPU oracle (which treats the unwired
    // cast as identity). The arm now matches the (input, output) pair: each wired pair requests its own
    // kernel, same-dtype casts alias (identity), unwired pairs are rejected by name.
    for (in_dt, to) in [
        (GDt::F32, GDt::BF16),
        (GDt::F32, GDt::F16),
        (GDt::F16, GDt::F32),
    ] {
        let want = match (in_dt, to) {
            (GDt::F32, GDt::BF16) => CastSpec::F32ToBf16 { numel: 8 },
            (GDt::F32, GDt::F16) => CastSpec::F32ToF16 { numel: 8 },
            _ => CastSpec::F16ToF32 { numel: 8 },
        };
        // The native bf16 body has no SPIR-V type to store into (see below), so the bf16 pair is checked
        // where the device has two-byte lanes.
        let backend = if to == GDt::BF16 {
            Backend::Nvptx
        } else {
            Backend::SpirvVulkan
        };
        match cast_planned(in_dt, to, backend) {
            Ok((compute_plan, KernelChoice::Generated(KernelRequest::Cast(got))))
                if is_plain(&compute_plan) =>
            {
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{in_dt:?}->{to:?}")
            }
            other => panic!("cast {in_dt:?}->{to:?} should plan a Compute kernel: {other:?}"),
        }
    }
    // POOT-1395: SPIR-V has no bf16 type, so a native bf16 cast body (an f32 narrowed into, or a computed
    // bf16 widened out of, a bf16 buffer) cannot be generated there; both are refused by name at planning
    // instead of failing in code generation.
    let r = cast_plan(GDt::F32, GDt::BF16, Backend::SpirvVulkan);
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "cast f32->bf16 on SPIR-V must be refused by name: got {r:?}"
    );
    // Card 1011: a BF16 const read only by this cast is stored as packed `u32` lanes on every backend, so the
    // cast plans the imported packed body there. A BF16 computed value is the native two-byte cast.
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        match cast_planned(GDt::BF16, GDt::F32, backend) {
            Ok((compute_plan, choice)) if is_plain(&compute_plan) => assert!(
                is_imported(&choice, ImportedKernel::PackedBf16ToF32),
                "{backend:?}: {choice:?}"
            ),
            other => panic!(
                "a packed BF16 const cast should plan the imported body on {backend:?}: {other:?}"
            ),
        }
    }
    {
        let bld = Builder::for_tests();
        let x = bld
            .slot(
                poot_graph_ir::Slot::Activation,
                TensorType::new(vec![8], GDt::BF16),
            )
            .unwrap();
        let widened = bld.cast(x, GDt::F32).unwrap();
        let g = bld.finish(widened).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::Cast { .. }))
            .unwrap();
        match planned_with_choice(&g, eqn, Backend::Nvptx) {
            Ok((compute_plan, KernelChoice::Generated(KernelRequest::Cast(got))))
                if is_plain(&compute_plan) =>
            {
                assert_eq!(
                    format!("{got:?}"),
                    format!("{:?}", CastSpec::Bf16ToF32 { numel: 8 })
                )
            }
            other => panic!("a computed BF16 cast should plan the native cast: {other:?}"),
        }
    }
    // A same-dtype cast is the identity: alias, not a kernel.
    assert!(
        matches!(
            cast_plan(GDt::F32, GDt::F32, Backend::SpirvVulkan),
            Ok(Plan::Alias(_))
        ),
        "identity cast must alias"
    );
    // An unwired pair is rejected by name, never widened through another pair's kernel.
    let r = cast_plan(GDt::BF16, GDt::F16, Backend::SpirvVulkan);
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "cast bf16->f16 must be rejected by name: got {r:?}"
    );
}

/// ADR-0104 decision 2: a refusal names the equation, the op, the dtypes, the target and the missing
/// capability. The same unwired cast is refused on every backend, and each refusal carries all five.
#[test]
fn a_refusal_names_the_equation_op_dtypes_target_and_missing_capability_on_every_backend() {
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let bld = Builder::for_tests();
        let x = bld
            .constant("x", TensorType::new(vec![8], GDt::BF16))
            .unwrap();
        let c = bld.cast(x, GDt::F16).unwrap();
        let g = bld.finish(c).unwrap();
        let eqn = &g.eqns()[0];
        let refusal = refusal_of(
            plan_eqn(
                &g,
                eqn,
                backend,
                &default_caps_for(backend),
                &crate::test_support::roomy_body_limits(),
            ),
            "an unwired cast pair",
        );
        assert_eq!(
            refusal,
            Refusal {
                eqn: c.id(),
                op: OpKind::Cast { to: GDt::F16 },
                dtypes: RefusalDtypes {
                    operands: vec![GDt::BF16],
                    output: GDt::F16,
                },
                target: TargetId::Gpu(backend),
                missing: Capability::DtypeLowering,
            },
            "on {backend:?}"
        );
    }
}

/// An I32->F32 cast of an exact-I32 source (a `GeU` output, as card 372c's guarded selector Cast reads).
fn exact_i32_cast_planned(backend: Backend) -> Result<(Plan, KernelChoice), PlanError> {
    let bld = Builder::for_tests();
    let x = bld
        .constant("x", TensorType::new(vec![8], GDt::I32))
        .unwrap();
    let flag = bld.binary_scalar(GBinOp::GeU, x, Scalar::I32(0)).unwrap();
    let c = bld.cast(flag, GDt::F32).unwrap();
    let g = bld.finish(c).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Cast { .. }))
        .expect("a cast eqn");
    planned_with_choice(&g, eqn, backend)
}

#[test]
fn cast_i32_to_f32_plans_an_i32_in_f32_out_kernel() {
    // Card 372c wires the i32->f32 pair. The key alone would not catch a regression to
    // `cast_bf16_to_f32` (which reads i32 words as packed bf16 halves), so the operand and output element
    // types are asserted from the body.
    let (plan, choice) = exact_i32_cast_planned(Backend::SpirvVulkan).expect("i32->f32 is wired");
    assert!(
        is_plain(&plan),
        "cast i32->f32 should plan a Compute kernel: {plan:?}"
    );
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Cast(CastSpec::I32ToF32 { numel: 8 }))
        ),
        "{choice:?}"
    );
    assert_eq!(
        body_param1_ty(&plan),
        Ty::Ref {
            mutable: false,
            pointee: Box::new(Ty::Slice(Box::new(Ty::I32))),
        },
        "the source operand must be an i32 slice, not bf16 or f32 bytes"
    );
    let body = &plan.single().expect("one dispatch").body();
    assert_eq!(
        body.locals[2].ty,
        Ty::Ref {
            mutable: true,
            pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
        },
        "the destination must be a mutable f32 slice"
    );
}

#[test]
fn cast_f32_to_i32_is_still_unwired() {
    // Card 372c needs one direction only. The reverse pair stays a named rejection.
    let r = cast_plan(GDt::F32, GDt::I32, Backend::SpirvVulkan);
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "cast f32->i32 must stay unwired: got {r:?}"
    );
}

// --- spec 135 (F16 graph dtype, phase 1) ---

fn matmul_plan(g: &Graph, backend: Backend) -> Plan {
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .expect("a matmul eqn");
    plan_eqn(
        g,
        eqn,
        backend,
        &default_caps_for(backend),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap()
}

/// The `Ty` of a single-dispatch plan's body's first non-return parameter local (`_1`, the
/// `a` operand). Asserts the actual kernel element type the `fty` bridge chose, which the cache-key
/// string alone would not catch (a bug mapped `DType::F16 -> Ty::F32` while the key still said "f16:").
fn body_param1_ty(plan: &Plan) -> Ty {
    let body = &plan
        .single()
        .unwrap_or_else(|| panic!("expected a single-dispatch plan, got {plan:?}"))
        .body();
    body.locals[1].ty.clone()
}

#[test]
fn f16_matmul_bridges_to_ty_f16_not_f32() {
    // SC-003 / FR-003: a whole-graph F16 matmul (F16 operands, F16 output) must bridge to Ty::F16 kernel
    // operands (the `fty` bridge at plan_eqn_views's top); mapping F16 to Ty::F32 would read 2-byte f16
    // storage as 4-byte f32.
    let g = matmul_graph(1, 32, 16, GDt::F16);
    let plan = matmul_plan(&g, Backend::SpirvVulkan);
    let ty = body_param1_ty(&plan);
    assert_eq!(
        ty,
        Ty::Ref {
            mutable: false,
            pointee: Box::new(Ty::Slice(Box::new(Ty::F16))),
        },
        "an F16 matmul operand must lower to a Ty::F16 slice, not Ty::F32: got {ty:?}"
    );
    let choice = matmul_choice(&g, Backend::SpirvVulkan);
    assert_eq!(
        serial_matmul_dtype(&choice),
        Some(&Ty::F16),
        "the F16 matmul request must name the f16 element type: got {choice:?}"
    );
}

#[test]
fn f32_matmul_request_is_the_f32_serial_kernel_after_f16_dtype_added() {
    // SC-003: adding DType::F16 (and its `fty` arm) must not change the request an existing F32 matmul
    // plans. A batched `[2, 1, 32] @ [2, 32, 16]` steers past every fast path (decode-GEMV needs a rank-2
    // weight, the attention GEMV a Transpose-producer-free operand pair, the generated tiled GEMMs `M > 1`)
    // to the naive serial fallback; the Transpose on the weight rules out the attention GEMV.
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![2, 1, 32], GDt::F32))
        .unwrap();
    let wt = b
        .constant("wt", TensorType::new(vec![2, 16, 32], GDt::F32))
        .unwrap();
    let w = b.transpose(wt, vec![0, 2, 1]).unwrap();
    let out = b.matmul(x, w).unwrap();
    let g = b.finish(out).unwrap();
    let program = compiled(&g, Backend::AmdGcn(AmdArch::gfx1151()));
    let (_, choice) = compiled_matmul(&program);
    assert_eq!(
        serial_matmul_dtype(choice),
        Some(&Ty::F32),
        "an F32 matmul must request the f32 serial kernel: got {choice:?}"
    );
}

#[test]
fn f16_non_wired_op_rejected_with_diagnostic() {
    // FR-004: an F16-typed equation on an op not on the wired allow-list must be rejected with a clear
    // diagnostic, not silently emit an f32 kernel over 2-byte storage. IndexedMatMul's output dtype follows
    // its `x` operand (op.rs infer), so an F16 `x` gives an F16-typed IndexedMatMul eqn, and IndexedMatMul
    // is not in the BF16/F16 wired list (MatMul/ContractionEpilogue/Reshape/Transpose/Slice/Broadcast/Binary/
    // Unary/Gather/Concat/DynamicUpdateSlice/Reduce/Cast).
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![4, 8], GDt::F16))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![2, 8, 16], GDt::F16))
        .unwrap();
    let idx = b
        .constant("idx", TensorType::new(vec![4], GDt::I32))
        .unwrap();
    let out = b.indexed_matmul(x, w, idx).unwrap();
    let g = b.finish(out).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::IndexedMatMul))
        .expect("an indexed_matmul eqn");
    assert_eq!(
        g.aval(eqn.out).dtype,
        GDt::F16,
        "sanity: output follows x's F16 dtype"
    );
    let res = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    );
    assert!(
        res.is_err(),
        "an F16-typed non-wired op must be rejected, not silently planned: got {res:?}"
    );
}

/// Card 557: the contraction choice (`is_tileable_matmul_in`) takes the generated tiled GEMM only for
/// an M>1 matmul over F32 operands and a rank-2 weight, on every backend. A BF16 or F16 matmul takes the
/// dtype-generic serial kernel (the generated body is
/// f32-only; spec 135 FR-012). Mutation: drop the F32-operand clause from `is_tileable_matmul_in`; the
/// BF16 and F16 rows plan the tiled GEMM and this row goes red.
#[test]
fn only_an_f32_m_gt_1_matmul_takes_the_tiled_gemm() {
    let amd = Backend::AmdGcn(AmdArch::gfx1151());
    for (backend, dtype, m, tiled) in [
        (Backend::SpirvVulkan, GDt::F32, 16, true),
        (amd, GDt::F32, 16, true),
        (Backend::Nvptx, GDt::F32, 16, true),
        (Backend::SpirvVulkan, GDt::F32, 1, false),
        (amd, GDt::BF16, 16, false),
        (Backend::Nvptx, GDt::BF16, 16, false),
        (Backend::SpirvVulkan, GDt::F16, 16, false),
    ] {
        let g = matmul_graph(m, 32, 16, dtype);
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul { .. }))
            .expect("a matmul eqn");
        let planned = plan_eqn_choice(
            &g,
            eqn,
            &Target::Gpu {
                backend,
                caps: default_caps_for(backend),
            },
            &HashMap::new(),
            &crate::test_support::roomy_body_limits(),
            None,
        )
        .unwrap_or_else(|e| panic!("{backend:?} {dtype:?} M={m}: {e:?}"));
        assert_eq!(
            is_kn_tiled(&planned.choice),
            tiled,
            "{backend:?} {dtype:?} M={m}: {:?}",
            planned.choice
        );
    }
}

#[test]
fn f16_matmul_never_reaches_the_tiled_gemm() {
    // FR-012: an F16 whole-graph matmul with M>1 (the tiled-GEMM shape) must not take the f32-only
    // generated tiled GEMM (`is_tileable_matmul_in`, the contraction choice); that kernel reads all
    // buffers as f32, so an F16 operand would be read as 4-byte f32 over 2-byte storage (a fault). M=32
    // (>1) is the shape the tiled choice targets for f32.
    let g = matmul_graph(32, 32, 16, GDt::F16);
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .expect("a matmul eqn");
    let planned = plan_eqn_choice(
        &g,
        eqn,
        &Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: default_caps_for(Backend::SpirvVulkan),
        },
        &HashMap::new(),
        &crate::test_support::roomy_body_limits(),
        None,
    )
    .expect("the F16 matmul plans");
    assert!(
        !matches!(
            planned.choice,
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(_)))
        ),
        "an F16 M>1 matmul must never take the f32-only generated tiled GEMM: {:?}",
        planned.choice
    );
    // and the MatMul's plan bridges to Ty::F16, as in the M=1 case.
    let plan = matmul_plan(&g, Backend::SpirvVulkan);
    let ty = body_param1_ty(&plan);
    assert_eq!(
        ty,
        Ty::Ref {
            mutable: false,
            pointee: Box::new(Ty::Slice(Box::new(Ty::F16))),
        },
        "the M>1 F16 matmul must still bridge to Ty::F16 via the serial fallback: got {ty:?}"
    );
}

/// `g` compiled for `backend` through the full pipeline.
fn compiled(g: &Graph, backend: Backend) -> Program {
    let target = Target::Gpu {
        backend,
        caps: default_caps_for(backend),
    };
    let options = CompileOptions::new(crate::CompileLimits::STANDARD);
    compile(g, &target, &options).expect("the matmul compiles")
}

/// The compiled MatMul's plan and kernel choice.
fn compiled_matmul(program: &Program) -> (&Plan, &KernelChoice) {
    let (eqn, plan) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::MatMul { .. }))
        .expect("a matmul eqn");
    (plan, program.kernel_choice(eqn))
}

/// The register-tiled GEMM over an f32 `[K, N]` weight, if `choice` is one.
fn kn_tiled(choice: &KernelChoice) -> Option<&DenseTiledContraction> {
    match contraction(choice) {
        Some(ContractionSpec::DenseTiled(spec)) if spec.weight == DenseTiledWeight::KnF32 => {
            Some(spec)
        }
        _ => None,
    }
}

fn is_kn_tiled(choice: &KernelChoice) -> bool {
    kn_tiled(choice).is_some()
}

/// `g` compiled for `backend` on `caps` through the full pipeline.
fn compiled_on(g: &Graph, backend: Backend, caps: DeviceCaps) -> Program {
    let options = CompileOptions::new(crate::CompileLimits::STANDARD);
    compile(g, &Target::Gpu { backend, caps }, &options).expect("the matmul compiles")
}

#[test]
fn tiled_matmul_gets_the_tiled_grid_not_the_naive_fallthrough() {
    // Card 099b: the generated tiled GEMM (the contraction choice for an M>1 F32 shared-weight MatMul, Card 557)
    // gets its schedule's launch grid (`tiles * lanes^2` threads, `Schedule::grid_threads`), not the
    // one-thread-per-output default.
    let (m, k, n) = (64usize, 32usize, 48usize);
    let g = matmul_graph(m, k, n, GDt::F32);
    let out_numel = m * n;

    let program = compiled(&g, Backend::SpirvVulkan);
    let (plan, choice) = compiled_matmul(&program);
    assert!(
        is_kn_tiled(choice),
        "the compile chooses the tiled GEMM: {choice:?}"
    );
    let tiled = with_meta(plan).launch().grid;
    let schedule = kn_tiled(choice).expect("the tiled GEMM").schedule;
    assert_eq!(
        tiled,
        [schedule.grid_threads(1, m, n) as u32, 1, 1],
        "the generated tiled GEMM must get its schedule's grid"
    );
    assert_ne!(
        tiled,
        [out_numel as u32, 1, 1],
        "must NOT be the naive one-thread-per-output fallthrough"
    );
}

/// The generated tiled GEMM over the RADV tile ceiling (card 096) splits into sub-cap dispatches over contiguous
/// tile ranges, each launching whole workgroups of its schedule's lanes, whose tiles sum to the whole range. The
/// ceiling is shrunk to 100 so a modest shape crosses it, and the watchdog budget removed so the ceiling is the
/// only bound. Mutation: in `register_tiled_launch`, drop the `tiled_gemm_max_workgroups` bound; the plan is one
/// dispatch and this row goes red.
#[test]
fn a_tiled_gemm_over_the_tile_ceiling_splits_into_disjoint_launches() {
    let (m, k, n) = (4096usize, 8usize, 4096usize);
    let ceiling = 100usize;
    let base = default_caps_for(Backend::SpirvVulkan);
    let caps = DeviceCaps {
        watchdog_budget: None,
        known_miscompiles: poot_target::KnownMiscompiles {
            tiled_gemm_max_workgroups: Some(ceiling as u32),
        },
        ..base
    };
    let g = matmul_graph(m, k, n, GDt::F32);
    let program = compiled_on(&g, Backend::SpirvVulkan, caps);
    let (plan, choice) = compiled_matmul(&program);
    let schedule = kn_tiled(choice)
        .expect("the compile chooses the tiled GEMM")
        .schedule;
    let (tile_rows, tile_cols) = schedule.output_tile().expect("a tiled schedule");
    let tiles = m.div_ceil(tile_rows) * n.div_ceil(tile_cols);
    assert!(tiles >= ceiling, "the fixture must reach the ceiling");
    let dispatches = plan.dispatches();
    assert!(dispatches.len() > 1, "{plan:?}");
    let mut covered = 0usize;
    for dispatch in dispatches {
        let launch = dispatch.launch();
        let lanes = dispatch.body().workgroup_size[0] as usize;
        let chunk = launch.grid[0] as usize / lanes;
        assert_eq!(
            launch.grid,
            [(chunk * lanes) as u32, 1, 1],
            "a split dispatch launches whole tile workgroups"
        );
        assert!(chunk < ceiling, "each launch stays under the ceiling");
        assert_eq!(launch.meta[4] as usize, covered, "contiguous tile ranges");
        covered += chunk;
    }
    assert_eq!(covered, tiles, "the launches tile the whole output");
}

/// Card 557: the generated tiled GEMM owns its launch (workgroups of its schedule's `lanes^2` lanes), and the op
/// alone no longer says so: `finalize` reads the recorded choice. A MatMul whose output is past the grid cap at
/// one workgroup of lanes per tile must keep its schedule-wide workgroups. Mutation: drop the choice from
/// `finalize`'s custom-grid test; the wg-bump widens the body to 256 lanes and this row goes red.
#[test]
fn a_large_tiled_matmul_keeps_its_tile_workgroup_on_amdgcn() {
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    // A 65 535-wide X grid (the HSA fixture's is `u32::MAX`, past any `u32` thread count): the output below is
    // past it at 64 threads per workgroup, where the generic bump would widen a one-thread-per-output body.
    let caps = DeviceCaps {
        max_grid: [65_535, 65_535, 65_535],
        ..default_caps_for(backend)
    };
    let (k, n) = (8usize, 2048usize);
    let m = (caps.max_grid[0] as usize * 64).div_ceil(n) + 1;
    let g = matmul_graph(m, k, n, GDt::F32);
    let program = compiled_on(&g, backend, caps);
    let (plan, choice) = compiled_matmul(&program);
    let Some(Schedule::RegisterTiled { lanes, .. }) = kn_tiled(choice).map(|spec| spec.schedule)
    else {
        panic!("an M>1 F32 matmul takes the generated tiled GEMM: {choice:?}");
    };
    let lanes = lanes.get() as usize;
    assert!(
        !plan.dispatches().is_empty(),
        "the tiled GEMM plans a dispatch: {plan:?}"
    );
    for dispatch in plan.dispatches() {
        assert_eq!(
            dispatch.body().workgroup_size[0] as usize,
            lanes * lanes,
            "the tiled GEMM's workgroup must stay its lane grid"
        );
    }
}

/// Where a launch figure came from: the `const`'s name, its value and the evidence behind it.
struct Provenance {
    name: &'static str,
    value: u64,
    source: &'static str,
}

/// Where each launch literal in `predicates.rs` comes from: one entry per `const` item that is a number. The
/// `every_launch_literal_has_provenance` test reads this file's own source and fails on a numeric `const` with no
/// entry here, and on an entry whose `const` no longer exists or has another value. A figure measured on one
/// device (`gfx1151`, wgpu/RADV unless noted) says so; one that is a structural limit says that instead.
const LAUNCH_PROVENANCE: &[Provenance] = &[
    prov(
        "GEMV_WIDTH",
        128,
        "measured: Qwen2.5-0.5B decode on RADV STRIX_HALO, fused ms/tok w=32 50.9, w=64 53.7, w=128 39.8, w=256 45.6",
    ),
    prov(
        "DENSE_GEMV_WIDTH",
        256,
        "card 727: sweep on gfx1151 over the qwen2.5-0.5b BF16 decode shapes",
    ),
    prov("DENSE_GEMV_UNROLL", 4, "card 727: same sweep"),
    prov(
        "DENSE_GEMV_WIDE_COLS",
        4,
        "card 727: wgpu Gemv device time per token cols 64 7.1 ms, 32 6.0, 16 5.3, 8 4.9, 4 4.7; ROCm decode 78-81 tok/s from cols 16 to 2",
    ),
    prov(
        "DENSE_GEMV_NARROW_COLS",
        2,
        "card 727: the fallback that keeps a small N from starving the device",
    ),
    prov(
        "DENSE_GEMV_WORKGROUPS_PER_CU",
        2,
        "card 727: at least two workgroups per compute unit before the wide column count",
    ),
    prov(
        "ELEMENTWISE_2D_WIDTH",
        256,
        "structural: the portable Vulkan workgroup maximum; a folded body derives its index from this width",
    ),
    prov(
        "SCATTER_UPDATE_WG",
        64,
        "structural: a generated scatter body's default workgroup width",
    ),
    prov(
        "FUSED_ROW_MAX_WIDTH",
        128,
        "design bound, not swept: the widest row workgroup the portable FusedRow reduction runs; a wider row loops",
    ),
    prov(
        "MAX_LANES",
        64,
        "design bound, not swept: more lanes only lengthen the serial per-lane combine",
    ),
    prov(
        "SELECTED_SLICES",
        2,
        "measured (POOT-1338, RTX PRO 6000, one prefill chunk of 1024 queries, 2052 candidates, ms): 1 slice 54.4, 2 slices 46.8; 4 and more do not fit 48 KiB of workgroup memory at 12 head lanes of 256",
    ),
    prov(
        "MAX_ATTENTION_SPLITS",
        16,
        "design bound, not swept: past it the combine stage's per-row reduction costs more than the extra workgroups hide (Flash-Decoding occupancy rule)",
    ),
    prov(
        "MIN_KEYS_PER_SPLIT",
        128,
        "design bound, not swept: a shorter chunk leaves a workgroup's lanes idle and the launch is the combine's overhead",
    ),
    prov(
        "WORKGROUPS_PER_COMPUTE_UNIT",
        2,
        "design bound, not swept: two resident workgroups keep a CU busy while one waits on its K/V loads",
    ),
    prov(
        "INDEXED_GEMV_MAX_ROWS",
        32,
        "design bound, not swept: a decode step's selected experts or a few steps batched; above it the register-tiled expert walk shares expert weight tiles",
    ),
    prov(
        "TILED_MAX_ROWS",
        128,
        "design bound, not swept: past 128 rows the query tile alone (128 x d words) exceeds a 64 KiB workgroup memory at d 128",
    ),
    prov(
        "TILED_ACCUMULATORS_PER_LANE",
        64,
        "POOT-1402 bank sweep (Strix Halo wgpu): 40 accumulators beat 80; half of the 128-register occupancy point",
    ),
    prov(
        "TILED_MAX_LANE_COLS",
        16,
        "design bound, not swept: a lane's columns are one register each, as wide as the widest key block it splits",
    ),
];

/// A provenance entry; the value is read from the `const` of the same name by the test.
const fn prov(name: &'static str, value: u64, source: &'static str) -> Provenance {
    Provenance {
        name,
        value,
        source,
    }
}

/// POOT-1233 SC-003: every numeric `const` in `predicates.rs` (the file's launch literals) has an entry in
/// `LAUNCH_PROVENANCE` with the same value and a stated source, and every entry names a `const` that exists.
/// The test reads the file's own source, so a literal added without an entry cannot pass. It covers every
/// numeric `const` item; the inline arithmetic that remains is structural, not a launch choice: `4 * (..)` and
/// `2 * lanes` in `attention_lanes` (bytes per f32 and the online-softmax `m`, `l` scalars). The register-tiled lane figures live in `poot-target`'s own
/// table. Mutation: add
/// `const UNSOURCED_LAUNCH: usize = 7;` to `predicates.rs`; this goes red naming it.
#[test]
fn every_launch_literal_has_provenance() {
    let source = include_str!("predicates.rs");
    let declared: Vec<(&str, u64)> = source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            (!line.starts_with("//")).then_some(())?;
            let item = line.split_once("const ")?.1;
            let (name, rest) = item.split_once(": ")?;
            let (ty, value) = rest.split_once(" = ")?;
            matches!(ty, "usize" | "u32" | "u64").then_some(())?;
            let value = value.split_once(';')?.0;
            Some((name, value.replace('_', "").parse().unwrap_or_else(|_| {
                panic!("launch literal {name} = {value} is not a plain integer; give it a plain literal")
            })))
        })
        .collect();
    let table = LAUNCH_PROVENANCE;
    for &(name, value) in &declared {
        if name == "LAUNCH_PROVENANCE" {
            continue;
        }
        let entry = table
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("launch literal {name} has no provenance entry"));
        assert_eq!(entry.value, value, "{name}: the entry states another value");
        assert!(!entry.source.is_empty(), "{name} states no source");
    }
    for entry in table {
        assert!(
            declared.iter().any(|&(name, _)| name == entry.name),
            "provenance entry {} names no const in predicates.rs",
            entry.name
        );
    }
}

/// The routed (indexed) contraction shape the lane-grid rows plan: past the decode Gemv's rows and large enough to
/// fill a device many times over, two experts.
const ROUTED: (usize, usize, usize, usize) = (4096, 256, 1024, 2);

/// The register-tiled lane-grid edge a routed (indexed) contraction of [`ROUTED`]'s shape plans on `backend` with
/// `caps`, for the dense f32-stack form (`packed == false`) and the packed-stack form.
fn indexed_lane_edge(backend: Backend, caps: &DeviceCaps, packed: bool) -> usize {
    let (rows, k, n, experts) = ROUTED;
    assert!(rows > crate::predicates::INDEXED_GEMV_MAX_ROWS);
    let b = Builder::for_tests();
    let x = b
        .slot_named(
            poot_graph_ir::Slot::Activation,
            "x",
            TensorType::f32(vec![rows, k]),
        )
        .unwrap();
    let ids = b
        .slot_named_ranged(
            poot_graph_ir::Slot::Activation,
            "ids",
            TensorType::new(vec![rows], GDt::I32),
            poot_graph_ir::IndexRange::below(experts).unwrap(),
        )
        .unwrap();
    let out = if packed {
        let table = poot_graph_ir::test_support::packed_graphs::PackedExpertStack {
            stack: poot_graph_ir::test_support::packed_graphs::TEST_STACK,
            descriptor: poot_quant::PackedWeight::try_new(
                poot_quant::format::WeightFormat::Q8_0,
                [n, k],
            )
            .unwrap(),
            experts,
        };
        poot_graph_ir::test_support::packed_graphs::packed_indexed_linear(&b, x, ids, &table)
            .unwrap()
    } else {
        let w = b
            .constant("w", TensorType::f32(vec![experts, k, n]))
            .unwrap();
        b.indexed_matmul(x, w, ids).unwrap()
    };
    let g = b.finish(out).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| {
            matches!(
                e.op,
                OpKind::IndexedMatMul | OpKind::PackedContraction { .. }
            )
        })
        .expect("a routed contraction eqn");
    let (_, choice) = planned_with_choice_caps(&g, eqn, backend, caps)
        .unwrap_or_else(|error| panic!("{backend:?} packed={packed}: {error}"));
    let KernelChoice::Generated(KernelRequest::Contraction(spec)) = choice else {
        panic!("{backend:?} packed={packed}: not a generated contraction: {choice:?}");
    };
    let schedule = match spec {
        ContractionSpec::Indexed(indexed) => indexed.schedule,
        ContractionSpec::Packed(packed) => packed.schedule,
        other => panic!("{backend:?} packed={packed}: not a routed contraction: {other:?}"),
    };
    match schedule {
        kg::Schedule::RegisterTiled { lanes, .. } => lanes.get() as usize,
        other => {
            panic!(
                "{backend:?} packed={packed}: expected the RegisterTiled schedule, got {other:?}"
            )
        }
    }
}

/// POOT-1233 SC-001, POOT-1403: the routed M>1 contraction takes the cost model's lane grid for its shape on every
/// backend: the wide 16x16 grid on ROCm, PTX and RADV caps for a contraction that fills the device, in both the
/// dense and the packed form (the RADV wave64 LDS defect that once kept RADV to 8x8 does not reproduce, see
/// `REGISTER_LANE_EDGE_WIDE`'s provenance). Mutation: pin the narrow lane edge in `RegisterTiling::candidates`; the
/// rows read 8 and go red.
#[test]
fn the_routed_lane_grid_is_the_cost_models_on_every_backend() {
    for backend in [
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
        Backend::SpirvVulkan,
    ] {
        let caps = default_caps_for(backend);
        for packed in [false, true] {
            assert_eq!(
                indexed_lane_edge(backend, &caps, packed),
                16,
                "{backend:?} packed={packed}: a device-filling routed contraction takes the wide grid"
            );
        }
    }
}

/// POOT-1233: a dispatch carries a planner-computed cost, apportioned over a split plan by the threads each
/// launches, and the output element count is conserved (up to rounding). The tile ceiling is shrunk to 100 so
/// the plan splits with a short tail. Mutation: apportion evenly instead of by grid threads; the last, smaller
/// chunk weighs as much as a full one and this goes red.
#[test]
fn a_split_plans_dispatches_carry_work_in_proportion_to_their_launch() {
    let (m, k, n) = (4096usize, 8usize, 4096usize);
    let base = default_caps_for(Backend::SpirvVulkan);
    let caps = DeviceCaps {
        watchdog_budget: None,
        known_miscompiles: poot_target::KnownMiscompiles {
            tiled_gemm_max_workgroups: Some(100),
        },
        ..base
    };
    let g = matmul_graph(m, k, n, GDt::F32);
    let program = compiled_on(&g, Backend::SpirvVulkan, caps);
    let (plan, _) = compiled_matmul(&program);
    let dispatches = plan.dispatches();
    assert!(dispatches.len() > 1, "{plan:?}");
    let threads = |d: &Dispatch| {
        d.launch()
            .grid
            .iter()
            .map(|&e| u64::from(e))
            .product::<u64>()
    };
    let total: u64 = dispatches.iter().map(Dispatch::work).sum();
    let out = (m * n) as u64;
    assert!(
        total <= out && out - total < dispatches.len() as u64,
        "the shares sum to the output count: {total} vs {out}"
    );
    let (full, last) = (&dispatches[0], dispatches.last().unwrap());
    assert!(
        threads(last) < threads(full),
        "the fixture needs a short tail chunk"
    );
    assert!(last.work() < full.work(), "work must follow the launch");
    assert_eq!(
        full.work() * threads(last) / threads(full),
        last.work(),
        "work is proportional to the launched threads"
    );
}

// --- a collective is a cut between stages, never an equation a stage program plans ---

/// POOT-747 SC-005: a collective built into one stage graph (uncut) is refused by `compile` on every
/// backend. The baseline lowered it to `Plan::Alias` and `compile` returned `Ok`, silently dropping the
/// cross-device exchange the node stands for.
#[test]
fn a_collective_inside_a_stage_graph_is_refused_by_compile() {
    let world = std::num::NonZeroU32::new(2).unwrap();
    type Collective<'a> = &'a dyn Fn(&Builder, poot_graph_ir::Traced) -> poot_graph_ir::Traced;
    let all_reduce = |bld: &Builder, y| bld.all_reduce(y, RedOp::Sum, 1).unwrap();
    let all_gather = |bld: &Builder, y| bld.all_gather(y, 1, world).unwrap();
    let cases: [(&str, Collective); 2] = [("all_reduce", &all_reduce), ("all_gather", &all_gather)];
    for backend in [Backend::Nvptx, Backend::SpirvVulkan] {
        for (name, collective) in cases {
            let bld = Builder::for_tests();
            let x = bld.constant("x", TensorType::f32(vec![2, 8])).unwrap();
            let w = bld.constant("w", TensorType::f32(vec![8, 4])).unwrap();
            let y = collective(&bld, bld.matmul(x, w).unwrap());
            let reduced = y.id();
            let g = bld.finish(y).unwrap();
            let target = Target::Gpu {
                backend,
                caps: default_caps_for(backend),
            };
            let options = CompileOptions::new(CompileLimits::STANDARD);
            match compile(&g, &target, &options) {
                Err(CompileError::Plan(error)) => match *error {
                    PlanError::Refused(refusal) => {
                        assert_eq!(
                            refusal.missing,
                            Capability::CollectiveInStage,
                            "{backend:?} {name}"
                        );
                        assert_eq!(refusal.eqn, reduced, "{backend:?} {name}");
                        assert_eq!(refusal.target, TargetId::Gpu(backend));
                    }
                    other => panic!("{backend:?} {name}: expected a typed refusal, got {other:?}"),
                },
                Ok(_) => panic!("{backend:?} {name}: compile accepted a collective in a stage"),
                Err(other) => panic!("{backend:?} {name}: expected a refusal, got {other:?}"),
            }
        }
    }
}

// --- spec 132 phase 1 (strided views) ---

/// `a -(movement)-> t`, with `t` fed into `binary_add(t, t)` so `t`'s only consumer is strided-capable
/// and `compute_views` promotes it to a view. Returns the graph, `t`'s value id, and the movement eqn's
/// index (callers pull its `OpKind` off `g.eqns`, matching what `compute_views`/`apply_movement` see).
fn view_test_graph(
    a_shape: Vec<usize>,
    movement: impl FnOnce(&Builder, poot_graph_ir::Traced) -> poot_graph_ir::Traced,
) -> (Graph, ValueId, usize) {
    let bld = Builder::for_tests();
    let a = bld.constant("a", TensorType::f32(a_shape)).unwrap();
    let t = movement(&bld, a);
    let y = bld.binary(GBinOp::Add, t, t).unwrap();
    let g = bld.finish(y).unwrap();
    let t_id = t.id();
    let move_eqn = g
        .eqns()
        .iter()
        .position(|e| e.out == t_id)
        .expect("the movement eqn produces t");
    (g, t_id, move_eqn)
}

#[test]
fn compute_views_never_promotes_validation_values() {
    let (g, t_id, _) = view_test_graph(vec![3, 2, 4], |bld, a| {
        bld.transpose(a, vec![2, 0, 1]).unwrap()
    });
    assert!(
        compute_views(&g, Backend::SpirvVulkan).contains_key(&t_id),
        "control: without a witness the transpose is a view"
    );
    let observed = derive(&g, |draft| {
        draft.push_witness(ValidationId(3), "transposed", t_id);
    })
    .unwrap();
    let views = compute_views(&observed, Backend::SpirvVulkan);
    assert!(
        !views.contains_key(&t_id),
        "a validation value keeps its own materialized buffer"
    );
    assert_eq!(dispatch_count_planned(&observed, Backend::SpirvVulkan), 2);
}

#[test]
fn transpose_into_binary_add_lowers_to_view_zero_dispatch() {
    // AS-001/AS-003: a Transpose whose only consumer is a pointwise Binary emits Plan::View (no dispatch),
    // not the materialize copy.
    let (g, t_id, move_eqn) = view_test_graph(vec![3, 2, 4], |bld, a| {
        bld.transpose(a, vec![2, 0, 1]).unwrap()
    });
    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        views.contains_key(&t_id),
        "a Transpose feeding only a Binary(Add) must be promoted to a view"
    );
    let eqn = &g.eqns()[move_eqn];
    match plan_eqn_views(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &views,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap()
    {
        Plan::View { src, .. } => assert_eq!(src, value_ids(eqn)[0]),
        other => panic!("expected Plan::View, got a dispatching plan instead: {other:?}"),
    }
    // AS-003: with an empty views map (every other backend) the same eqn still takes the materialize
    // copy, so views change nothing for a caller that does not opt in.
    match plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap()
    {
        copy_plan if is_plain(&copy_plan) => {}
        other => panic!("expected the unchanged materialize copy plan, got {other:?}"),
    }
}

#[test]
fn transpose_into_matmul_keeps_the_materialize_copy() {
    // AS-002: a Transpose feeding a MatMul (a contraction kernel, not strided-capable) keeps its one
    // materialize copy; MatMul is absent from `is_strided_capable`.
    let bld = Builder::for_tests();
    let a = bld.constant("a", TensorType::f32(vec![4, 3])).unwrap();
    let t = bld.transpose(a, vec![1, 0]).unwrap(); // [3,4]
    let w = bld.constant("w", TensorType::f32(vec![4, 5])).unwrap();
    let mm = bld.matmul(t, w).unwrap(); // [3,5]
    let g = bld.finish(mm).unwrap();
    let t_id = t.id();

    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        !views.contains_key(&t_id),
        "a Transpose feeding a MatMul must NOT be promoted to a view (FR-004)"
    );
    let move_eqn = g.eqns().iter().find(|e| e.out == t_id).unwrap();
    match plan_eqn_views(
        &g,
        move_eqn,
        Backend::SpirvVulkan,
        &views,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap()
    {
        // exactly one dispatch: the generated movement copy.
        copy_plan if is_plain(&copy_plan) => {}
        other => panic!("expected the unchanged materialize copy plan, got {other:?}"),
    }
}

#[test]
fn view_cache_key_never_collides_with_a_plain_buffer_of_the_same_shape() {
    // A leaf's strides/offset are baked as consts into the generated Body, so a Binary(Add) reading a
    // view must get a different cache key than one reading a plain contiguous buffer of the same shape,
    // else the pipeline/disk cache could dispatch the wrong shader. Build one graph with both cases (same
    // operand shapes, op, out_shape) and assert the keys and the generated Bodies differ.
    let bld = Builder::for_tests();
    let a = bld.constant("a", TensorType::f32(vec![3, 2])).unwrap(); // pre-transpose source
    let t = bld.transpose(a, vec![1, 0]).unwrap(); // view candidate, shape [2,3]
    let b = bld.constant("b", TensorType::f32(vec![2, 3])).unwrap(); // shared operand
    let y_view = bld.binary(GBinOp::Add, t, b).unwrap(); // t (a view) + b
    let c = bld.constant("c", TensorType::f32(vec![2, 3])).unwrap(); // a PLAIN buffer, same shape as t
    let y_plain = bld.binary(GBinOp::Add, c, b).unwrap(); // c (plain) + b - same shapes as y_view's eqn
    let out = bld.binary(GBinOp::Add, y_view, y_plain).unwrap();
    let g = bld.finish(out).unwrap();

    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        views.contains_key(&t.id()),
        "t's only consumer (y_view's Binary) is strided-capable"
    );

    let eqn_view = g.eqns().iter().find(|e| e.out == y_view.id()).unwrap();
    let eqn_plain = g.eqns().iter().find(|e| e.out == y_plain.id()).unwrap();
    let Dispatch {
        key: key_view,
        body: body_view,
        ..
    } = into_plain(
        plan_eqn_views(
            &g,
            eqn_view,
            Backend::SpirvVulkan,
            &views,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap(),
    );
    let Dispatch {
        key: key_plain,
        body: body_plain,
        ..
    } = into_plain(
        plan_eqn_views(
            &g,
            eqn_plain,
            Backend::SpirvVulkan,
            &views,
            &default_caps_for(Backend::SpirvVulkan),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap(),
    );
    assert_ne!(
        key_view, key_plain,
        "a view operand and a plain buffer of the same shape must not share a cache key"
    );
    assert_ne!(
        format!("{body_view:?}"),
        format!("{body_plain:?}"),
        "the generated Bodies must differ too (not just the label) - a same-key collision would \
         have silently reused one Body's compiled shader for the other's inputs"
    );
}

/// Row-major strides of `shape`, the reference the view tests decode coordinates against.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Seeded xorshift64 for the fuzz tests below (no external RNG dependency).
fn xorshift(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// For every output coordinate of `out_shape`, decode it against `strides`/`offset` and assert
/// `src_data[computed_index] == oracle_data[linear_index]`: the strided view reads what the CPU oracle
/// (`poot_eval::apply_movement`, contiguous/logical) computed. Exercises the index math `compute_views`
/// derives and `kg::view_eff_strides` reconstructs at the consumer against an independent reference.
fn assert_view_matches_oracle(
    out_shape: &[usize],
    strides: &[usize],
    offset: usize,
    src_data: &[f32],
    oracle_data: &[f32],
) {
    let out_strides = row_major_strides(out_shape);
    let n = numel(out_shape);
    for (i, &want) in oracle_data.iter().enumerate().take(n) {
        let mut rem = i;
        let mut src = offset;
        for d in 0..out_shape.len() {
            let coord = rem / out_strides[d];
            rem -= coord * out_strides[d];
            src += coord * strides[d];
        }
        assert_eq!(
            src_data[src], want,
            "view/oracle mismatch at output index {i} (coord decode via out_shape {out_shape:?})"
        );
    }
}

#[test]
fn view_transpose_matches_oracle_fuzz() {
    let mut seed = 0xdead_beef_1234_5678u64;
    for _ in 0..200 {
        let a_shape = vec![
            2 + (xorshift(&mut seed) % 3) as usize,
            2 + (xorshift(&mut seed) % 3) as usize,
            2 + (xorshift(&mut seed) % 3) as usize,
        ];
        let mut perm = vec![0usize, 1, 2];
        for i in (1..3).rev() {
            let j = (xorshift(&mut seed) as usize) % (i + 1);
            perm.swap(i, j);
        }
        let (g, t_id, move_eqn) = view_test_graph(a_shape.clone(), |bld, a| {
            bld.transpose(a, perm.clone()).unwrap()
        });
        let views = compute_views(&g, Backend::SpirvVulkan);
        let layout = views
            .get(&t_id)
            .cloned()
            .expect("a Transpose into Binary(Add) must be a view");

        let a_numel = numel(&a_shape);
        let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32 + 0.5).collect();
        let a_tensor = poot_tensor::HostTensor::f32(a_shape.clone(), a_data.clone());
        let oracle = poot_eval::apply_movement(&g.eqns()[move_eqn].op, &[a_tensor]).unwrap();

        let out_shape = g.aval(t_id).shape.clone();
        assert_view_matches_oracle(
            &out_shape,
            &layout.strides,
            layout.offset,
            &a_data,
            oracle.as_f32().unwrap(),
        );
    }
}

#[test]
fn view_slice_matches_oracle_fuzz() {
    let mut seed = 0x0bad_c0de_f00d_1234u64;
    for _ in 0..200 {
        let a_shape = vec![
            2 + (xorshift(&mut seed) % 4) as usize,
            2 + (xorshift(&mut seed) % 4) as usize,
            3 + (xorshift(&mut seed) % 4) as usize,
        ];
        let axis = (xorshift(&mut seed) as usize) % 3;
        let len = a_shape[axis];
        let start = (xorshift(&mut seed) as usize) % len;
        let end = start + 1 + (xorshift(&mut seed) as usize) % (len - start);
        let (g, t_id, move_eqn) = view_test_graph(a_shape.clone(), |bld, a| {
            bld.slice(a, axis, start, end).unwrap()
        });
        let views = compute_views(&g, Backend::SpirvVulkan);
        let layout = views
            .get(&t_id)
            .cloned()
            .expect("a Slice into Binary(Add) must be a view");

        let a_numel = numel(&a_shape);
        let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32 + 0.5).collect();
        let a_tensor = poot_tensor::HostTensor::f32(a_shape.clone(), a_data.clone());
        let oracle = poot_eval::apply_movement(&g.eqns()[move_eqn].op, &[a_tensor]).unwrap();

        let out_shape = g.aval(t_id).shape.clone();
        assert_view_matches_oracle(
            &out_shape,
            &layout.strides,
            layout.offset,
            &a_data,
            oracle.as_f32().unwrap(),
        );
    }
}

#[test]
fn view_broadcast_matches_oracle_fuzz() {
    let mut seed = 0xfeed_face_cafe_babeu64;
    for _ in 0..200 {
        // a_shape mixes real and size-1 dims, so the broadcast zero-stride override is exercised too.
        // Every non-1 dim of `a_shape` is copied into `out_shape` (numpy broadcast only widens a size-1
        // dim); only a size-1 dim or the new leading dim varies.
        let a_shape = vec![
            if xorshift(&mut seed).is_multiple_of(2) {
                1
            } else {
                2
            },
            2 + (xorshift(&mut seed) % 3) as usize, // never 1: must match out exactly
            if xorshift(&mut seed).is_multiple_of(3) {
                1
            } else {
                3
            },
        ];
        let lead = 2 + (xorshift(&mut seed) % 2) as usize;
        let o0 = if a_shape[0] == 1 {
            2 + (xorshift(&mut seed) % 3) as usize
        } else {
            a_shape[0]
        };
        let o2 = if a_shape[2] == 1 {
            2 + (xorshift(&mut seed) % 3) as usize
        } else {
            a_shape[2]
        };
        let out_shape = vec![lead, o0, a_shape[1], o2];
        let (g, t_id, move_eqn) = view_test_graph(a_shape.clone(), |bld, a| {
            bld.broadcast(a, out_shape.clone()).unwrap()
        });
        let views = compute_views(&g, Backend::SpirvVulkan);
        let layout = views
            .get(&t_id)
            .cloned()
            .expect("a Broadcast into Binary(Add) must be a view");

        let a_numel = numel(&a_shape);
        let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32 + 0.5).collect();
        let a_tensor = poot_tensor::HostTensor::f32(a_shape.clone(), a_data.clone());
        let oracle = poot_eval::apply_movement(&g.eqns()[move_eqn].op, &[a_tensor]).unwrap();

        let got_out_shape = g.aval(t_id).shape.clone();
        assert_eq!(got_out_shape, out_shape);
        assert_view_matches_oracle(
            &got_out_shape,
            &layout.strides,
            layout.offset,
            &a_data,
            oracle.as_f32().unwrap(),
        );
    }
}

/// Dispatches `program` runs per step: every planned equation but the free ones (an alias or a strided view).
fn dispatches(program: &Program) -> usize {
    program
        .planned()
        .filter(|(_, plan)| !matches!(plan, Plan::Alias(_) | Plan::View { .. }))
        .count()
}

// --- spec 132 phase 2 (view-chain composition) ---

#[test]
fn transpose_then_slice_into_binary_add_composes_both_into_views() {
    // A Transpose feeding a Slice feeding only a Binary(Add). Phase 1's one-hop restriction promoted at
    // most the Slice; the Transpose stayed a materialize copy because its consumer, the Slice, is not
    // `is_strided_capable`. Phase 2 must promote both: the Slice is a view of a view, and the chain
    // composes into one Layout per link, both relative to the original contiguous buffer.
    let bld = Builder::for_tests();
    let a = bld.constant("a", TensorType::f32(vec![3, 2, 4])).unwrap();
    let t = bld.transpose(a, vec![2, 0, 1]).unwrap(); // [3,2,4] -> [4,3,2]
    let s = bld.slice(t, 1, 1, 3).unwrap(); // axis 1 (len 3) [1,3) -> [4,2,2]
    let y = bld.binary(GBinOp::Add, s, s).unwrap();
    let g = bld.finish(y).unwrap();
    let (t_id, s_id) = (t.id(), s.id());

    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        views.contains_key(&t_id),
        "the Transpose link must ALSO promote to a view now (chain composition) - phase 1 kept it \
         as a materialize copy since its consumer (Slice) was never `is_strided_capable`"
    );
    assert!(
        views.contains_key(&s_id),
        "the Slice link (already promotable in phase 1, one-hop) must still promote"
    );

    // Hand-computed composition: a's row-major strides (shape [3,2,4]) are [8,4,1].
    // transpose(perm=[2,0,1]): out dim d reads
    // in_strides[perm[d]] -> [in_strides[2], in_strides[0], in_strides[1]] = [1,8,4], offset 0.
    let t_layout = &views[&t_id];
    assert_eq!(t_layout.strides, vec![1, 8, 4]);
    assert_eq!(t_layout.offset, 0);
    // slice(axis=1,start=1) of that view: strides unchanged (a slice never reorders dims), offset
    // advances by start * the view's own stride for axis 1 (8, not a's in_strides[1]=4).
    let s_layout = &views[&s_id];
    assert_eq!(s_layout.strides, vec![1, 8, 4]);
    assert_eq!(s_layout.offset, 8);

    // 0 movement dispatches for the whole chain: only Binary(Add) counts. Phase 1 counted 2 (the
    // Transpose materialize copy + the Binary); see also
    // `dispatch_count_planned_chain_drops_further_than_phase_1_one_hop` below.
    let n = dispatch_count_planned(&g, Backend::SpirvVulkan);
    assert_eq!(
        n, 1,
        "both movement ops must be free views; only Binary(Add) should dispatch"
    );
}

#[test]
fn dispatch_count_planned_chain_drops_further_than_phase_1_one_hop() {
    // Same chain as above, as a direct phase-1-vs-phase-2 dispatch count comparison, so a regression to
    // one-hop composition is caught even if the strides assertions are weakened. Phase 1: the Transpose
    // is a materialize copy (its consumer, Slice, isn't `is_strided_capable`) -> 1; the Slice still
    // promotes as the last hop -> 0; Binary -> 1; total 2. Phase 2: both movement ops are free views;
    // total 1.
    const PHASE_1_ONE_HOP_DISPATCHES: usize = 2;

    let bld = Builder::for_tests();
    let a = bld.constant("a", TensorType::f32(vec![3, 2, 4])).unwrap();
    let t = bld.transpose(a, vec![2, 0, 1]).unwrap();
    let s = bld.slice(t, 1, 1, 3).unwrap();
    let y = bld.binary(GBinOp::Add, s, s).unwrap();
    let g = bld.finish(y).unwrap();

    let n = dispatch_count_planned(&g, Backend::SpirvVulkan);
    assert!(
        n < PHASE_1_ONE_HOP_DISPATCHES,
        "phase 2 chain composition must drop the dispatch count below phase 1's one-hop bound \
         ({PHASE_1_ONE_HOP_DISPATCHES}): got {n}"
    );
    assert_eq!(
        n, 1,
        "the whole 2-op chain must collapse to 0 dispatches, leaving only Binary(Add)"
    );
}

#[test]
fn view_chain_matches_oracle_fuzz() {
    // Phase 2 fuzz (extends the phase-1 `view_transpose/slice/broadcast_matches_oracle_fuzz` tests to a
    // 2-3 op chain mixing all three op kinds): a random chain feeding only a Binary(Add) must promote
    // every link to a view, and each link's composed Layout must read exactly what
    // `poot_eval::apply_movement`, folded step by step over the graph's own eqns, computes for the chain
    // up to that link (bit exact).
    let mut seed = 0x1357_9bdf_2468_ace0u64;
    for _ in 0..300 {
        let bld = Builder::for_tests();
        let mut shape: Vec<usize> = (0..3)
            .map(|_| 2 + (xorshift(&mut seed) % 3) as usize)
            .collect();
        let a = bld.constant("a", TensorType::f32(shape.clone())).unwrap();
        let mut cur = a;
        let mut link_ids: Vec<ValueId> = Vec::new();
        let chain_len = 2 + (xorshift(&mut seed) as usize) % 2; // 2 or 3 ops
        for _ in 0..chain_len {
            match xorshift(&mut seed) % 3 {
                0 => {
                    // Transpose: a random permutation of the current rank.
                    let rank = shape.len();
                    let mut perm: Vec<usize> = (0..rank).collect();
                    for i in (1..rank).rev() {
                        let j = (xorshift(&mut seed) as usize) % (i + 1);
                        perm.swap(i, j);
                    }
                    cur = bld.transpose(cur, perm.clone()).unwrap();
                    shape = perm.iter().map(|&p| shape[p]).collect();
                }
                1 => {
                    // Slice: a random axis/range of the current shape.
                    let rank = shape.len();
                    let axis = (xorshift(&mut seed) as usize) % rank;
                    let len = shape[axis];
                    let start = (xorshift(&mut seed) as usize) % len;
                    let end = start + 1 + (xorshift(&mut seed) as usize) % (len - start);
                    cur = bld.slice(cur, axis, start, end).unwrap();
                    shape[axis] = end - start;
                }
                _ => {
                    // Broadcast: prepend one new leading dim (always valid for the current shape).
                    let lead = 2 + (xorshift(&mut seed) % 3) as usize;
                    let mut new_shape = vec![lead];
                    new_shape.extend_from_slice(&shape);
                    cur = bld.broadcast(cur, new_shape.clone()).unwrap();
                    shape = new_shape;
                }
            }
            link_ids.push(cur.id());
        }
        let y = bld.binary(GBinOp::Add, cur, cur).unwrap();
        let g = bld.finish(y).unwrap();

        let views = compute_views(&g, Backend::SpirvVulkan);
        for &id in &link_ids {
            assert!(
                views.contains_key(&id),
                "every link of a chain feeding only Binary(Add) must promote to a view \
                 (chain shape: {shape:?}, {} links)",
                link_ids.len()
            );
        }

        let a_shape = g.aval(a.id()).shape.clone();
        let a_numel = numel(&a_shape);
        let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32 + 0.5).collect();
        let mut oracle = poot_tensor::HostTensor::f32(a_shape, a_data.clone());
        for &id in &link_ids {
            let eqn = g
                .eqns()
                .iter()
                .find(|e| e.out == id)
                .expect("every link id is some eqn's output");
            oracle = poot_eval::apply_movement(&eqn.op, &[oracle]).unwrap();
            let layout = views.get(&id).expect("checked promoted above");
            let out_shape = g.aval(id).shape.clone();
            assert_view_matches_oracle(
                &out_shape,
                &layout.strides,
                layout.offset,
                &a_data,
                oracle.as_f32().unwrap(),
            );
        }
    }
}

/// The scatter's cost follows the source rows, not the pool (POOT-1148). At gemma4-dense's local-layer pool
/// (`pool_slots = 1025, Hkv = 16, D = 256`, 4.2M elements, which the old pool-wide kernel had to chunk for the
/// wgpu watchdog) a real prompt (`l = 49` rows) is one dispatch of `49 * Hkv * D` threads. A wide batched
/// prefill (`l = 512`, past `scatter_update_work` source units) splits into dispatches over disjoint,
/// full-coverage source-unit windows, each reading the source rows and the slot map only. NVPTX (no display
/// watchdog) stays one dispatch at the same shape.
#[test]
fn scatter_update_launches_over_source_rows_and_chunks_only_a_wide_prefill_on_wgpu() {
    let (pool_slots, hkv, d) = (1025usize, 16usize, 256usize);
    let build = |rows: usize| {
        let b = Builder::for_tests();
        let base = b
            .constant("base", TensorType::f32(vec![pool_slots, hkv, d]))
            .unwrap();
        let src = b
            .constant("src", TensorType::f32(vec![rows, hkv, d]))
            .unwrap();
        let slots = b
            .constant("slots", TensorType::new(vec![rows], GDt::I32))
            .unwrap();
        let out = b.scatter_update(base, src, slots).unwrap();
        b.finish(out).unwrap()
    };
    let scatter = |g: &Graph| {
        g.eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::ScatterUpdate))
            .unwrap()
            .clone()
    };
    let window_of = |dispatch: &Dispatch| match dispatch.choice() {
        KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
            op: kg::MovementOp::ScatterRows { window, .. },
            ..
        })) => *window,
        other => panic!("expected a ScatterRows request: {other:?}"),
    };
    let wgpu = default_caps_for(Backend::SpirvVulkan);
    let budget = wgpu.watchdog_budget.unwrap().scatter_update_work as usize;

    // A real prompt: 49 rows x 4096 elements, far below the budget, in a 4.2M-element pool.
    let short = build(49);
    let eqn = scatter(&short);
    let plan = plan_eqn(
        &short,
        &eqn,
        Backend::SpirvVulkan,
        &wgpu,
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    assert!(
        49 * hkv * d <= budget,
        "premise: the prompt is under the budget"
    );
    let [dispatch] = plan.dispatches() else {
        panic!("a short prefill is one dispatch: {plan:?}")
    };
    assert_eq!(window_of(dispatch), None);
    assert_eq!(
        dispatch.launch().grid[0] as usize,
        49 * hkv * d,
        "one thread per source element, whatever the pool"
    );

    // A wide prefill: windows are disjoint, in order and cover every source unit.
    let wide = build(512);
    let eqn = scatter(&wide);
    let units = 512 * hkv * d;
    assert!(units > budget, "premise: the prefill is past the budget");
    let plan = plan_eqn(
        &wide,
        &eqn,
        Backend::SpirvVulkan,
        &wgpu,
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let windows: Vec<kg::UnitWindow> = plan
        .dispatches()
        .iter()
        .map(|dispatch| window_of(dispatch).expect("a chunk names its window"))
        .collect();
    assert!(windows.len() > 1, "{plan:?}");
    let mut next = 0;
    for window in &windows {
        assert_eq!(window.offset, next, "windows are consecutive");
        next += window.threads;
    }
    assert!(next >= units, "windows cover every source unit");
    assert!(
        windows[..windows.len() - 1]
            .iter()
            .all(|w| w.offset + w.threads <= units),
        "only the last window may launch past the source"
    );

    // NVPTX has no display watchdog: one whole dispatch at the wide shape.
    let (nvptx, choice) = planned_with_choice(&wide, &eqn, Backend::Nvptx).unwrap();
    assert!(is_plain(&nvptx), "{nvptx:?}");
    assert!(
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
                op: kg::MovementOp::ScatterRows { window: None, .. },
                ..
            }))
        ),
        "{choice:?}"
    );
}

/// Card 258: at BLOOM's tied-`lm_head` decode-GEMV scale (`M=1, K=1024, N=250880`, the shape that hung
/// real hardware, over
/// `DECODE_GEMV_CHUNK_TRIGGER`), the wgpu decode-GEMV plan must split into several dispatches
/// (kernelgen `gemv_lds` with a baked `elem_offset`, not the fixed-signature imported kernel) covering
/// disjoint, full-coverage output-column ranges, like `serial_kquant_dequant_plan` (card 163). gpt-oss's
/// scale (`N=201088`, at/under the trigger) stays a single unchunked imported dispatch. AmdGcn (no display
/// watchdog) stays a single imported dispatch even at BLOOM's shape; chunking is `Backend::SpirvVulkan`-only.
#[test]
fn decode_gemv_chunks_on_wgpu_at_bloom_lm_head_shape() {
    let (k, n) = (1024usize, 250_880usize); // BLOOM-560m's real tied lm_head (hidden, vocab)
    let g = matmul_graph(1, k, n, GDt::F32);
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .unwrap();
    let out_shape = g.aval(eqn.out).shape.clone();
    let out_numel = numel(&out_shape);
    assert_eq!(out_numel, n);
    assert!(
        out_numel as u64 > DECODE_GEMV_CHUNK_TRIGGER,
        "test fixture must exceed the chunk trigger"
    );

    let plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    let chunks = plan.dispatches();
    assert!(
        chunks.len() > 1,
        "BLOOM-scale wgpu decode-GEMV should split into >1 dispatch: {plan:?}"
    );
    // every chunk's workgroup count must clear the wgpu per-dim grid cap (chunk size is capped at
    // caps.max_grid[0]; each chunk is a plain 1-D grid).
    let grid_cap = default_caps_for(Backend::SpirvVulkan).max_grid[0] as usize;
    let groups = |c: &Dispatch| c.launch().grid[0] as usize / c.body().workgroup_size[0] as usize;
    for c in chunks {
        assert!(groups(c) <= grid_cap, "chunk grid must clear the grid cap");
        assert!(
            c.launch().meta.is_empty(),
            "a kernelgen chunk bakes its offset as a const, no meta buffer"
        );
    }
    // chunks must be disjoint and fully cover out_numel (one workgroup per output element, so
    // `groups` sums exactly to out_numel, with no workgroup-size rounding slack).
    let covered: usize = chunks.iter().map(groups).sum();
    assert_eq!(
        covered, out_numel,
        "chunks must exactly cover the full output"
    );

    // gpt-oss's scale (at/under the trigger) stays the single unchunked dispatch of the same body.
    let (k2, n2) = (32usize, 201_088usize);
    let g2 = matmul_graph(1, k2, n2, GDt::F32);
    let eqn2 = g2
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .unwrap();
    match planned_with_choice(&g2, eqn2, Backend::SpirvVulkan).unwrap() {
        (compute_plan, choice) if is_plain(&compute_plan) => {
            assert!(is_decode_gemv(&choice), "{choice:?}")
        }
        other => panic!("gpt-oss-scale wgpu decode-GEMV should stay one dispatch: {other:?}"),
    }

    // AmdGcn has no display watchdog: stays the single unchunked dispatch even at BLOOM's
    // large shape (chunking is SpirvVulkan-only, as in card 163).
    let amdgcn = Backend::AmdGcn(AmdArch::gfx1151());
    match planned_with_choice(&g, eqn, amdgcn).unwrap() {
        (compute_plan, choice) if is_plain(&compute_plan) => {
            assert!(is_decode_gemv(&choice), "{choice:?}")
        }
        other => panic!("AmdGcn decode-GEMV should stay one dispatch (no chunking): {other:?}"),
    }
}

/// R486-013 / SC-006, POOT-1232 SC-001: the decode-GEMV is kept only below the device's measured width
/// (`caps.decode_gemv_max_out_cols`), and the ceiling belongs to the caps, not to the backend. The NVPTX fixture
/// carries the Ampere figure (256), so a 300-column decode projection plans the one-thread-per-output
/// fallback there; the same backend with no measured ceiling keeps the coalesced GEMV, and the SPIR-V backend
/// with the same ceiling drops it. Mutation: hard-code the 256 ceiling instead of reading the caps field: the
/// NVPTX row without a ceiling (N=300) loses the GEMV (`assertion failed: gemv_at(Backend::Nvptx, None, 300)`).
#[test]
fn decode_gemv_width_ceiling_follows_the_caps_not_the_backend() {
    let gemv_at = |backend: Backend, ceiling: Option<u32>, n: usize| {
        let g = matmul_graph(1, 64, n, GDt::F32);
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul { .. }))
            .expect("a matmul eqn");
        let caps = DeviceCaps {
            decode_gemv_max_out_cols: ceiling,
            ..default_caps_for(backend)
        };
        let (_, choice) = planned_with_choice_caps(&g, eqn, backend, &caps).unwrap();
        eprintln!("{backend:?} N={n} ceiling={ceiling:?}: {choice:?}");
        is_decode_gemv(&choice)
    };
    // The Ampere fixture: the boundary is inclusive.
    assert!(
        gemv_at(Backend::Nvptx, Some(256), 256),
        "N=256 is at the ceiling"
    );
    assert!(
        !gemv_at(Backend::Nvptx, Some(256), 300),
        "N=300 is above it"
    );
    // No measured ceiling: the GEMV is kept at any width, on the backend that used to be special-cased.
    assert!(gemv_at(Backend::Nvptx, None, 300));
    // A ceiling on another backend applies the same way.
    assert!(!gemv_at(Backend::SpirvVulkan, Some(256), 300));
    assert!(gemv_at(Backend::SpirvVulkan, Some(256), 256));
}

/// POOT-1232 SC-001: whether an E4M3FN tensor has native storage is the device's `e4m3_storage`, not its
/// backend. The ROCm backend with native E4M3FN storage admits the `ScatterUpdate` it refuses without, and the
/// SPIR-V backend without it refuses by name. Mutation: gate on the backend again (`Backend::SpirvVulkan |
/// Backend::Nvptx`): the ROCm row with storage is refused and the SPIR-V row without it is admitted.
#[test]
fn e4m3_storage_admission_follows_the_caps_not_the_backend() {
    let builder = Builder::for_tests();
    let base = builder
        .constant("base", TensorType::new(vec![4, 2, 5], GDt::E4M3FN))
        .unwrap();
    let src = builder
        .constant("src", TensorType::new(vec![3, 2, 5], GDt::E4M3FN))
        .unwrap();
    let slots = builder
        .constant("slots", TensorType::new(vec![3], GDt::I32))
        .unwrap();
    let output = builder.scatter_update(base, src, slots).unwrap();
    let graph = builder.finish(output).unwrap();
    let eqn = &graph.eqns()[0];
    let plan_with = |backend: Backend, e4m3_storage: bool| {
        let caps = DeviceCaps {
            e4m3_storage: if e4m3_storage {
                poot_target::E4m3Storage::PackedLanes
            } else {
                poot_target::E4m3Storage::None
            },
            ..default_caps_for(backend)
        };
        planned_with_choice_caps(&graph, eqn, backend, &caps).map(|(plan, _)| plan)
    };
    let rocm = Backend::AmdGcn(AmdArch::gfx1151());
    assert!(
        plan_with(rocm, true).is_ok(),
        "native storage admits it on ROCm"
    );
    let refusal = refusal_of(
        plan_with(rocm, false),
        "E4M3FN ScatterUpdate without native storage",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    let refusal = refusal_of(
        plan_with(Backend::SpirvVulkan, false),
        "E4M3FN ScatterUpdate on SPIR-V without native storage",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert!(plan_with(Backend::SpirvVulkan, true).is_ok());
}

/// POOT-1232 SC-001: no planner rule compares a backend variant. Every non-test source file of the crate
/// (predicates, the planner entry point and its family planners, the dtype rules, the kernel mapping, the packed
/// lowering table, the multi-device topology) reads the device's caps instead, so a rule cannot be keyed on a name. Mutation:
/// put `matches!(backend, Backend::Nvptx)` back in any one of them; the test names the file and line.
#[test]
fn predicates_and_planner_do_not_compare_backend_variants() {
    fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    rust_files(&poot_test_util::manifest_dir().join("src"), &mut files);
    // Test sources build backends to plan for; the rules are the subject.
    files.retain(|f| {
        !f.components().any(|c| c.as_os_str() == "tests")
            && !f.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                n == "test_support.rs" || n == "tests.rs" || n.starts_with("tests_")
            })
    });
    assert!(
        files.len() > 30,
        "the planner sources are listed: {}",
        files.len()
    );
    let mut offenders = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        // The unit tests of a file build backends to plan for; the rules above them are the subject.
        let rules = text.split("\n#[cfg(test)]").next().unwrap();
        for (i, line) in rules.lines().enumerate() {
            if line.contains("Backend::") && !line.trim_start().starts_with("//") {
                offenders.push(format!(
                    "{}:{}: {}",
                    file.file_name().unwrap().to_string_lossy(),
                    i + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a planner rule keys on the backend name, not on device caps:\n{}",
        offenders.join("\n")
    );
}

/// Card 522 SC-002: the planner reads the display-watchdog chunk trigger from `DeviceCaps`, not from a
/// backend-keyed constant. Two synthetic `DeviceCaps` for the same `Backend::SpirvVulkan`, differing
/// only in `watchdog_budget.decode_gemv_out_elems`, plan the same decode-GEMV shape differently: the
/// lower trigger chunks it, the higher trigger keeps it a single dispatch. At the baseline (this test,
/// before any mutation) the two plans differ, proving the planner reads the caps value rather than
/// ignoring it.
#[test]
fn decode_gemv_chunking_follows_the_caps_watchdog_trigger_not_a_fixed_constant() {
    let (k, n) = (1024usize, 230_000usize); // between gpt-oss's 201088 and BLOOM's 250880
    let g = matmul_graph(1, k, n, GDt::F32);
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul { .. }))
        .unwrap();
    let out_shape = g.aval(eqn.out).shape.clone();
    let out_numel = numel(&out_shape);
    assert_eq!(out_numel, n);

    let caps_for = |decode_gemv_out_elems: u64| DeviceCaps {
        watchdog_budget: Some(WatchdogBudget {
            decode_gemv_out_elems,
            ..DeviceCaps::wgpu_rdna3_igpu().watchdog_budget.unwrap()
        }),
        ..DeviceCaps::wgpu_rdna3_igpu()
    };

    // Low trigger (under this shape's 230_000 elements): chunks, like the BLOOM-scale case above.
    let low = caps_for(100_000);
    let chunked = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &low,
        &crate::test_support::roomy_body_limits(),
    )
    .unwrap();
    assert!(
        chunked.dispatches().len() > 1,
        "low watchdog trigger should chunk this shape: {chunked:?}"
    );

    // High trigger (over this shape's 230_000 elements): a single unchunked dispatch, even though this
    // is the exact same backend and the exact same shape that chunked above.
    let high = caps_for(1_000_000);
    match planned_with_choice_caps(&g, eqn, Backend::SpirvVulkan, &high).unwrap() {
        (compute_plan, choice) if is_plain(&compute_plan) => {
            assert!(is_decode_gemv(&choice), "{choice:?}")
        }
        other => panic!("high watchdog trigger should stay a single dispatch: {other:?}"),
    }
}

/// One attention equation to plan: `q[B,Hq,Lq,D]`, `k`/`v[B,Hq/n_rep,Lk,D]`, `mask[B,Hm,Lq,Lk]`.
#[derive(Clone, Copy)]
struct AttentionFixture {
    name: &'static str,
    batch: usize,
    hq: usize,
    n_rep: usize,
    lq: usize,
    lk: usize,
    d: usize,
    per_head_mask: bool,
    softcap: Option<f32>,
}

impl AttentionFixture {
    /// Decode: one query row against a `cap`-key cache.
    const fn decode(
        name: &'static str,
        batch: usize,
        hq: usize,
        n_rep: usize,
        cap: usize,
        d: usize,
    ) -> Self {
        Self {
            name,
            batch,
            hq,
            n_rep,
            lq: 1,
            lk: cap,
            d,
            per_head_mask: false,
            softcap: None,
        }
    }

    /// Prefill: `l` query rows against the same `l` keys.
    const fn prefill(name: &'static str, hq: usize, n_rep: usize, l: usize, d: usize) -> Self {
        Self {
            name,
            batch: 1,
            hq,
            n_rep,
            lq: l,
            lk: l,
            d,
            per_head_mask: false,
            softcap: None,
        }
    }

    fn alibi(self) -> Self {
        Self {
            per_head_mask: true,
            ..self
        }
    }

    fn softcapped(self, cap: f32) -> Self {
        Self {
            softcap: Some(cap),
            ..self
        }
    }

    /// The fused equation, staged directly: the planner answers for any valid equation, whatever the matcher
    /// would have formed.
    fn graph(self) -> Graph {
        let hkv = self.hq / self.n_rep;
        let mask_heads = if self.per_head_mask { self.hq } else { 1 };
        let bld = Builder::for_tests();
        let operand = |name: &str, shape: Vec<usize>| {
            bld.constant(name, TensorType::new(shape, GDt::F32))
                .unwrap()
        };
        let q = operand("q", vec![self.batch, self.hq, self.lq, self.d]);
        let k = operand("k", vec![self.batch, hkv, self.lk, self.d]);
        let v = operand("v", vec![self.batch, hkv, self.lk, self.d]);
        let mask = operand("mask", vec![self.batch, mask_heads, self.lq, self.lk]);
        let mut plan = bld.append_plan(0).unwrap();
        let out = plan
            .equation(
                OpKind::Attention(AttentionAttrs::new(
                    self.n_rep,
                    1.0 / (self.d as f32).sqrt(),
                    self.softcap,
                )),
                [q, k, v, mask]
                    .map(|operand| Operand::Value(operand.id()))
                    .to_vec(),
            )
            .unwrap_or_else(|e| panic!("{}: stage the attention: {e:?}", self.name));
        plan.declare_result(out).expect("declare the result");
        let mut prepared = bld.preflight_append(plan).expect("preflight");
        let committed = bld.commit_append(&mut prepared).expect("commit");
        bld.finish(committed).unwrap()
    }
}

fn attention_planned(
    fixture: AttentionFixture,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<(Plan, KernelChoice), PlanError> {
    let g = fixture.graph();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Attention(_)))
        .expect("an attention eqn");
    planned_with_choice_caps(&g, eqn, backend, caps)
}

/// The attention request a choice generated.
fn attention_spec(choice: &KernelChoice) -> &kg::AttentionSpec {
    match choice {
        KernelChoice::Generated(KernelRequest::Attention(spec)) => spec,
        other => panic!("expected a generated attention request, got {other:?}"),
    }
}

const ATTENTION_TARGETS: [fn() -> Backend; 3] = [
    || Backend::SpirvVulkan,
    || Backend::AmdGcn(AmdArch::gfx1151()),
    || Backend::Nvptx,
];

/// Every attention shape the five old bodies covered plans one generated attention request on all three
/// targets: Llama GQA decode (`B = 1` and `B = 4`), the D = 256 decode the imported body capped at, the
/// D = 512 decode SPIR-V used to refuse, a plain prefill, gemma-style softcapped prefill, and the ALiBi
/// per-head-mask decode and prefill. The request carries each equation's own extents, mask kind and score
/// transform. Mutation: map the per-head mask to `MaskKind::Broadcast` (or drop the per-head case from the
/// planner's spec) and the ALiBi rows go red.
#[test]
fn every_attention_fixture_plans_the_generated_attention_family_on_every_target() {
    let fixtures = [
        AttentionFixture::decode("llama decode", 1, 8, 4, 64, 64),
        AttentionFixture::decode("llama decode B=4", 4, 8, 4, 64, 64),
        AttentionFixture::decode("decode D=256", 1, 4, 2, 32, 256),
        AttentionFixture::decode("decode D=512", 1, 4, 2, 32, 512),
        AttentionFixture::prefill("prefill", 8, 4, 32, 64),
        AttentionFixture::prefill("gemma softcap prefill", 4, 2, 24, 256).softcapped(50.0),
        AttentionFixture::decode("alibi decode", 2, 4, 1, 16, 64).alibi(),
        AttentionFixture::prefill("alibi prefill", 4, 1, 16, 64).alibi(),
    ];
    for backend in ATTENTION_TARGETS.map(|target| target()) {
        for fixture in fixtures {
            let (_, choice) = attention_planned(fixture, backend, &default_caps_for(backend))
                .unwrap_or_else(|e| panic!("{} on {backend:?}: {e:?}", fixture.name));
            let spec = attention_spec(&choice);
            let want_mask = if fixture.per_head_mask {
                kg::MaskKind::PerHead
            } else {
                kg::MaskKind::Broadcast
            };
            let scale_bits = (1.0 / (fixture.d as f32).sqrt()).to_bits();
            let want_score = match fixture.softcap {
                None => kg::ScoreTransform::Scale { scale_bits },
                Some(cap) => kg::ScoreTransform::ScaleSoftcap {
                    scale_bits,
                    cap_bits: cap.to_bits(),
                },
            };
            let got = (
                (
                    spec.batch, spec.hq, spec.n_rep, spec.lq, spec.lk, spec.dk, spec.dv,
                ),
                spec.mask,
                spec.score,
            );
            let want = (
                (
                    fixture.batch,
                    fixture.hq,
                    fixture.n_rep,
                    fixture.lq,
                    fixture.lk,
                    fixture.d,
                    fixture.d,
                ),
                want_mask,
                want_score,
            );
            assert_eq!(got, want, "{} on {backend:?}", fixture.name);
        }
    }
}

/// The streaming schedule's lane count is the widest power of two whose partials fit the device's workgroup
/// memory: `4 * (lanes * dv + 2 * lanes)` bytes. The same decode plans 64, 64, 32 and 16 lanes at value dim 64,
/// 128, 256 and 512 on the Vulkan cap (65536 bytes, the adapter's own limit: wgpu no longer asks for the
/// downlevel 16 KiB), the same as ROCm, and fewer on the PTX cap (49152 bytes). A fixture
/// device whose workgroup memory cannot hold even one lane at `dv = 512` (2056 bytes) refuses with
/// `AttentionLds { dv: 512, .. }`, one byte over the boundary plans a single lane, and the same graph plans
/// on the real caps. Mutation: take the lane width from a constant 256 instead of `caps.lds_bytes`; the
/// small-caps device plans and the refusal row goes red.
#[test]
fn attention_lanes_follow_the_device_workgroup_memory() {
    // (backend, dv) -> lanes, by hand from `4 * (lanes * dv + 2 * lanes) <= lds_bytes` and the 64 cap.
    let lanes = |backend: Backend, d: usize| {
        let fixture = AttentionFixture::decode("lanes", 1, 2, 1, 16, d);
        let (_, choice) = attention_planned(fixture, backend, &default_caps_for(backend))
            .unwrap_or_else(|e| panic!("{backend:?} D={d}: {e:?}"));
        attention_spec(&choice).schedule.lanes()
    };
    let wgpu = Backend::SpirvVulkan;
    assert_eq!(
        [64, 128, 256, 512].map(|d| lanes(wgpu, d)),
        [64, 64, 32, 16],
        "65536 bytes of workgroup memory"
    );
    let rocm = Backend::AmdGcn(AmdArch::gfx1151());
    assert_eq!(
        [64, 128, 256, 512].map(|d| lanes(rocm, d)),
        [64, 64, 32, 16],
        "65536 bytes of workgroup memory"
    );
    assert_eq!(
        [64, 128, 256, 512].map(|d| lanes(Backend::Nvptx, d)),
        [64, 64, 32, 16],
        "49152 bytes of workgroup memory"
    );

    let fixture = AttentionFixture::decode("dv 512", 1, 2, 1, 16, 512);
    let with_lds = |lds_bytes: u32| DeviceCaps {
        lds_bytes,
        ..default_caps_for(wgpu)
    };
    let refusal = refusal_of(
        attention_planned(fixture, wgpu, &with_lds(2055)).map(|(plan, _)| plan),
        "dv 512 on 2055 bytes of workgroup memory",
    );
    assert_eq!(
        refusal.missing,
        Capability::AttentionLds {
            dv: 512,
            lds_bytes: 2055
        }
    );
    let (_, one_lane) = attention_planned(fixture, wgpu, &with_lds(2056))
        .expect("2056 bytes hold exactly one lane at dv 512");
    assert_eq!(
        attention_spec(&one_lane).schedule,
        kg::AttentionSchedule::Streaming { lanes: 1 }
    );
    attention_planned(fixture, wgpu, &default_caps_for(wgpu))
        .expect("the same graph plans on the real caps");
}

/// The selected-attention equation of a Flash-Next layer at `lq` queries over `lk` stored positions (24 query
/// heads over 2 key/value heads of 256, blocks of four, 512 selected), and what planning it chose.
fn selected_attention_planned(
    lq: usize,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<(Plan, KernelChoice), PlanError> {
    let lk = 8448;
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, 24, lq, 256]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, 2, lk, 256]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, 2, lk, 256]))
        .unwrap();
    let ids = b
        .constant("ids", TensorType::new(vec![1, 1, lq, 512], GDt::I32))
        .unwrap();
    let pos = b
        .constant("pos", TensorType::new(vec![1, lq], GDt::I32))
        .unwrap();
    let out = poot_graph_ir::test_support::composites::selected_attention(
        &b,
        poot_graph_ir::SelectedAttentionAttrs {
            n_rep: 12,
            scale: 0.0625,
            softcap: None,
            ratio: 4,
            k: 512,
        },
        [q, k, v, ids, pos],
    )
    .unwrap();
    let g = b.finish(poot_graph_ir::Outputs::new(out)).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::SelectedAttention(_)))
        .expect("a selected attention eqn");
    planned_with_choice_caps(&g, eqn, backend, caps)
}

/// POOT-1338: a prefill chunk of a block-selected attention serves the 12 query heads of a key/value head in one
/// workgroup (the K/V rows load once for the group): the request is `grouped`, its lanes a multiple of 12 and
/// its workgroups one per `(kv head, query)`. A decode row, whose two groups would leave the device idle, keeps a
/// workgroup per head. A device whose workgroup memory cannot hold the 12 head lanes plans the chunk ungrouped
/// (the schedule it ran before grouping existed), and refuses by name (`AttentionLds`) only when not even one
/// head's lane fits. Mutation: make `groups_selected_heads` return `false`; the chunk rows go red.
#[test]
fn a_selected_attention_chunk_groups_the_heads_of_a_key_value_head() {
    for backend in ATTENTION_TARGETS.map(|b| b()) {
        let caps = DeviceCaps {
            watchdog_budget: None,
            ..default_caps_for(backend)
        };
        let (_, chunk) = selected_attention_planned(1024, backend, &caps)
            .unwrap_or_else(|e| panic!("{backend:?} chunk: {e:?}"));
        let spec = attention_spec(&chunk);
        assert!(
            matches!(spec.mask, kg::MaskKind::Blocks { grouped: true, .. }),
            "{backend:?}: the chunk is grouped: {:?}",
            spec.mask
        );
        assert_eq!(spec.schedule.lanes() % 12, 0, "{backend:?}: head lanes");
        assert_eq!(
            spec.workgroups(),
            2 * 1024,
            "{backend:?}: one per (kv head, query)"
        );

        let (_, decode) = selected_attention_planned(1, backend, &caps)
            .unwrap_or_else(|e| panic!("{backend:?} decode: {e:?}"));
        assert!(
            matches!(
                attention_spec(&decode).mask,
                kg::MaskKind::Blocks { grouped: false, .. }
            ),
            "{backend:?}: decode keeps a workgroup per head"
        );
    }

    // 4 * (12 * 256 + 2 * 12) bytes: the partials of exactly the 12 head lanes.
    let wgpu = Backend::SpirvVulkan;
    let with_lds = |lds_bytes: u32| DeviceCaps {
        lds_bytes,
        watchdog_budget: None,
        ..default_caps_for(wgpu)
    };
    let (_, small) = selected_attention_planned(1024, wgpu, &with_lds(12383))
        .expect("12383 bytes cannot hold 12 head lanes, so the chunk plans ungrouped");
    let small = attention_spec(&small);
    assert!(
        matches!(small.mask, kg::MaskKind::Blocks { grouped: false, .. }),
        "12383 bytes: ungrouped, got {:?}",
        small.mask
    );
    assert_eq!(
        small.workgroups(),
        24 * 1024,
        "a workgroup per head and query"
    );
    let refusal = refusal_of(
        selected_attention_planned(1024, wgpu, &with_lds(1023)).map(|(plan, _)| plan),
        "not one head lane on 1023 bytes",
    );
    assert_eq!(
        refusal.missing,
        Capability::AttentionLds {
            dv: 256,
            lds_bytes: 1023
        }
    );
    let (_, twelve) = selected_attention_planned(1024, wgpu, &with_lds(12384))
        .expect("12384 bytes hold exactly the 12 head lanes");
    let twelve = attention_spec(&twelve);
    assert!(matches!(
        twelve.mask,
        kg::MaskKind::Blocks { grouped: true, .. }
    ));
    assert_eq!(
        twelve.schedule,
        kg::AttentionSchedule::Streaming { lanes: 12 }
    );
}

/// POOT-1338 x POOT-1313: a grouped chunk on a device with a display watchdog splits into runs of row groups
/// (one per `(kv head, query)`, each costing its `n_rep` rows), not runs of rows: the run counts are in the
/// workgroups the kernel launches. Mutation: drop the `heads_per_workgroup` factor from
/// `attention_rows_per_dispatch`; the budget then admits all 2048 groups in one run and the chunk is not split.
#[test]
fn a_grouped_chunk_under_a_watchdog_splits_into_runs_of_row_groups() {
    let wgpu = Backend::SpirvVulkan;
    let caps = default_caps_for(wgpu);
    assert!(
        caps.watchdog_budget.is_some(),
        "the wgpu fixture has a watchdog"
    );
    let (_, first) = selected_attention_planned(1024, wgpu, &caps).expect("the chunk plans");
    let KernelChoice::Generated(KernelRequest::AttentionRows(run)) = &first else {
        panic!("a chunk past the budget plans runs, got {first:?}");
    };
    assert!(matches!(
        run.spec.mask,
        kg::MaskKind::Blocks { grouped: true, .. }
    ));
    assert_eq!(run.spec.row_groups(), 2 * 1024);
    assert_eq!(run.first, 0);
    assert!(
        0 < run.count && run.count < run.spec.row_groups(),
        "a run is some of the {} row groups, got {}",
        run.spec.row_groups(),
        run.count
    );
}

/// The attention request of a plan's first dispatch: the whole attention, or the first run of its row groups
/// a device with a display watchdog splits it into.
fn first_attention_spec(choice: &KernelChoice) -> &kg::AttentionSpec {
    match choice {
        KernelChoice::Generated(KernelRequest::Attention(spec)) => spec,
        KernelChoice::Generated(KernelRequest::AttentionRows(run)) => &run.spec,
        other => panic!("expected a generated attention request, got {other:?}"),
    }
}

/// POOT-1402: the bank's long prefill (16 query heads over 8 key/value heads, d 128, 2048 rows) plans the tiled
/// schedule on every target, within the device's limits: its workgroup is within the workgroup caps, its body declares exactly the workgroup memory the tiles need and
/// no more than the device has, and its row groups fill the device's compute units. A decode of the same heads
/// keeps a streaming or split-KV schedule, whose rows are one query each.
///
/// Mutation: in `planner::attention::plan`, skip `attention_tiling` (plan the streaming spec); every prefill row
/// goes red with the streaming schedule.
#[test]
fn a_long_prefill_plans_the_tiled_schedule_and_a_decode_does_not() {
    for target in ATTENTION_TARGETS {
        let backend = target();
        let caps = default_caps_for(backend);
        let fixture = AttentionFixture::prefill("bank prefill 2048", 16, 2, 2048, 128);
        let (plan, choice) = attention_planned(fixture, backend, &caps)
            .unwrap_or_else(|e| panic!("{backend:?}: {e:?}"));
        let spec = first_attention_spec(&choice);
        let kg::AttentionSchedule::Tiled { lanes, .. } = spec.schedule else {
            panic!(
                "{backend:?}: a long prefill plans the tiled schedule, got {:?}",
                spec.schedule
            );
        };
        assert!(
            lanes <= caps.max_workgroup_invocations as usize,
            "{backend:?}: {lanes} lanes"
        );
        assert!(
            spec.row_groups() >= caps.compute_units as usize,
            "{backend:?}: {} row groups on {} compute units",
            spec.row_groups(),
            caps.compute_units
        );
        let dispatch = match &plan {
            Plan::Dispatches(dispatches) => &dispatches[0],
            other => other.single().expect("one dispatch"),
        };
        let used = kg::BodyUse::measure(dispatch.body()).expect("measurable workgroup memory");
        let tiles = spec
            .tiled_lds_bytes()
            .expect("a tiled schedule")
            .expect("its tiles fit the shape");
        assert_eq!(used.lds_bytes, tiles as u64, "{backend:?}: LDS");
        assert!(
            used.lds_bytes <= u64::from(caps.lds_bytes),
            "{backend:?}: LDS budget"
        );
        assert_eq!(
            used.workgroup,
            [lanes as u32, 1, 1],
            "{backend:?}: workgroup"
        );

        let decode = AttentionFixture::decode("bank decode", 1, 16, 2, 2048, 128);
        let (_, choice) = attention_planned(decode, backend, &caps)
            .unwrap_or_else(|e| panic!("{backend:?} decode: {e:?}"));
        let schedule = first_attention_spec(&choice).schedule;
        assert!(
            !matches!(schedule, kg::AttentionSchedule::Tiled { .. }),
            "{backend:?}: a decode keeps a per-row schedule, got {schedule:?}"
        );
    }
}

/// POOT-1402: the tiled schedule is chosen only within what the device holds. The same long prefill on a device
/// whose workgroup memory holds no tiling (8 KiB: the narrowest staged key block, 16 keys of a 128-wide head, is
/// 8.5 KiB alone) keeps the streaming schedule, and so does a short prefill whose row groups cannot give each compute unit one;
/// a device that launches at most 64 invocations per workgroup gets a 64-lane tiling, where the Vulkan caps launch a
/// wider one.
///
/// Mutation: drop the `lds <= caps.lds_bytes` test in `attention_tiling`; the 8 KiB device plans a tiling its
/// memory cannot hold and the first row goes red. Mutation: drop the `lanes <= lanes_limit` filter; the
/// 64-invocation device is given a 256-lane tiling its launch check refuses (`WorkgroupInvocations`), and the
/// last rows go red.
#[test]
fn the_tiled_schedule_is_chosen_only_within_the_devices_resources() {
    let wgpu = Backend::SpirvVulkan;
    let caps = default_caps_for(wgpu);
    let long = AttentionFixture::prefill("bank prefill 2048", 16, 2, 2048, 128);
    let small_lds = DeviceCaps {
        lds_bytes: 8 * 1024,
        ..caps
    };
    let (_, choice) = attention_planned(long, wgpu, &small_lds).expect("the prefill plans");
    assert!(
        matches!(
            first_attention_spec(&choice).schedule,
            kg::AttentionSchedule::Streaming { .. }
        ),
        "8 KiB of workgroup memory holds no tiling, got {:?}",
        first_attention_spec(&choice).schedule
    );

    let short = AttentionFixture::prefill("short prefill", 2, 2, 16, 64);
    let (_, choice) = attention_planned(short, wgpu, &caps).expect("the prefill plans");
    assert!(
        matches!(
            first_attention_spec(&choice).schedule,
            kg::AttentionSchedule::Streaming { .. }
        ),
        "16 rows of one key/value head cannot fill {} compute units",
        caps.compute_units
    );

    let lanes_of = |caps: &DeviceCaps| {
        let (_, choice) = attention_planned(long, wgpu, caps).expect("the prefill plans");
        first_attention_spec(&choice).schedule.lanes()
    };
    assert!(
        lanes_of(&caps) > 64,
        "the Vulkan caps launch a workgroup wider than one wave64"
    );
    let narrow = DeviceCaps {
        max_workgroup_invocations: 64,
        ..caps
    };
    assert_eq!(
        lanes_of(&narrow),
        64,
        "a 64-invocation device gets a 64-lane tiling"
    );
}

/// The plan records the implemented streaming schedule and the exact resources it needs, over a fixed bank:
/// short and long prefill, low- and high-head decode, and a D = 256 softcapped prefill. Per row the request
/// names `Streaming { lanes }`, the dispatch body states `lanes * (dv + 2) * 4` bytes of workgroup memory
/// and a `[lanes, 1, 1]` workgroup.
#[test]
fn the_streaming_schedule_records_its_exact_resources() {
    // (fixture, backend, lanes, workgroup-memory bytes), from the shapes by hand.
    let wgpu = Backend::SpirvVulkan;
    let rows = [
        (
            AttentionFixture::prefill("short prefill", 8, 4, 16, 64),
            wgpu,
            64usize,
            4 * (64 * 64 + 2 * 64),
        ),
        (
            // One key/value head of 32 rows gives the 40 compute units too few row groups to tile.
            AttentionFixture::prefill("one-group prefill", 8, 8, 32, 128),
            wgpu,
            64,
            4 * (64 * 128 + 2 * 64),
        ),
        // 192 keys is under two 128-key chunks: the rows are few but a split has nothing to gain.
        (
            AttentionFixture::decode("low-head short-cache decode", 1, 2, 1, 192, 128),
            wgpu,
            64,
            4 * (64 * 128 + 2 * 64),
        ),
        (
            AttentionFixture::decode("high-head decode", 4, 32, 4, 2048, 128),
            Backend::AmdGcn(AmdArch::gfx1151()),
            64,
            4 * (64 * 128 + 2 * 64),
        ),
        (
            AttentionFixture::prefill("softcap prefill D=256", 4, 2, 24, 256).softcapped(50.0),
            Backend::Nvptx,
            32,
            4 * (32 * 256 + 2 * 32),
        ),
    ];
    for (fixture, backend, lanes, lds_bytes) in rows {
        let caps = default_caps_for(backend);
        let (plan, choice) = attention_planned(fixture, backend, &caps)
            .unwrap_or_else(|e| panic!("{}: {e:?}", fixture.name));
        assert_eq!(
            attention_spec(&choice).schedule,
            kg::AttentionSchedule::Streaming { lanes },
            "{}",
            fixture.name
        );
        let dispatch = plan.single().expect("one dispatch");
        let used = kg::BodyUse::measure(dispatch.body()).expect("measurable workgroup memory");
        assert_eq!(used.lds_bytes, lds_bytes as u64, "{}: LDS", fixture.name);
        assert_eq!(
            used.workgroup,
            [lanes as u32, 1, 1],
            "{}: workgroup",
            fixture.name
        );
    }
}

/// POOT-1113: the split-KV plan records the exact resources of both its launches. A 2-row decode over 1024
/// keys on the Vulkan caps plans 8 splits of 128 keys over 64 lanes: stage one is `rows * splits` workgroups of
/// `lanes` threads with `lanes * (dv + 2) * 4` bytes of workgroup memory; the combine is one workgroup of
/// `lanes` threads per row with no workgroup memory.
#[test]
fn the_split_kv_plan_records_the_exact_resources_of_both_launches() {
    let backend = Backend::SpirvVulkan;
    let caps = default_caps_for(backend);
    let fixture = AttentionFixture::decode("low-head long decode", 1, 2, 1, 1024, 128);
    let (plan, choice) = attention_planned(fixture, backend, &caps).expect("the decode plans");
    let (lanes, splits) = (64usize, 8usize);
    assert_eq!(
        attention_spec(&choice).schedule,
        kg::AttentionSchedule::SplitKv { lanes, splits }
    );
    let Plan::Dispatches(dispatches) = &plan else {
        panic!("a dispatching plan: {plan:?}");
    };
    assert_eq!(dispatches.len(), 2, "partials, then combine");
    let used = |dispatch: &Dispatch| kg::BodyUse::measure(dispatch.body()).expect("measurable");
    assert_eq!(
        used(&dispatches[0]).lds_bytes,
        (lanes * (128 + 2) * 4) as u64
    );
    assert_eq!(used(&dispatches[0]).workgroup, [lanes as u32, 1, 1]);
    assert_eq!(used(&dispatches[1]).lds_bytes, 0);
    assert_eq!(used(&dispatches[1]).workgroup, [lanes as u32, 1, 1]);
}

/// Regression: `is_elementwise_2d` used to gate every arm on `dtype == F32`, including Binary/Unary/
/// MatMul, whose kernel bodies are dtype-parameterized (`plan_eqn` derives `dt` from the eqn's output
/// dtype independently of this predicate; see `matmul_batched_dt_grid`'s call site feeding it
/// `fty(odt)`). A BF16/F16 elementwise/matmul op above the wg-bump's `65535*256` element ceiling would
/// keep the 1-D grid and hit the `GridCap` crash this predicate prevents. Only the imported-body-only
/// arms (ScatterUpdate/DynamicUpdateSlice/Transpose/Slice/Broadcast/Concat) are F32-locked (no
/// other-dtype imported variant), so those stay gated.
#[test]
fn is_elementwise_2d_folds_binary_and_unary_for_a_non_f32_dtype() {
    use poot_graph_ir::Eqn;
    use poot_graph_ir::op::{BinOp as GBinOp, UnOp as GUnOp};
    let bld = Builder::for_tests();
    let over_ceiling = 65_535 * 256 + 256; // one workgroup past the ceiling
    let x = bld
        .constant("x", TensorType::new(vec![over_ceiling], GDt::BF16))
        .unwrap();
    let g = bld.finish(x).unwrap();
    let out_shape = vec![over_ceiling];

    let binary_eqn = Eqn {
        op: OpKind::Binary(GBinOp::Add),
        inputs: vec![],
        out: x.id(),
        layer: None,
    };
    assert!(
        is_elementwise_2d(
            &g,
            Backend::SpirvVulkan,
            &binary_eqn,
            &out_shape,
            over_ceiling
        ),
        "a BF16 Binary op past the ceiling must still 2-D-fold"
    );

    let unary_eqn = Eqn {
        op: OpKind::Unary(GUnOp::Exp),
        inputs: vec![],
        out: x.id(),
        layer: None,
    };
    assert!(
        is_elementwise_2d(
            &g,
            Backend::SpirvVulkan,
            &unary_eqn,
            &out_shape,
            over_ceiling
        ),
        "a BF16 Unary op past the ceiling must still 2-D-fold"
    );

    // Below the ceiling, neither folds regardless of dtype (1-D dispatch is cheaper).
    let small_shape = vec![256usize];
    assert!(!is_elementwise_2d(
        &g,
        Backend::SpirvVulkan,
        &binary_eqn,
        &small_shape,
        256
    ));

    // A ScatterUpdate launches one thread per source element, so it folds when the source rows pass the
    // ceiling (not when the pool does), and a source below it stays on the flat launch.
    let ceiling =
        default_caps_for(Backend::SpirvVulkan).max_grid[0] as usize * ELEMENTWISE_2D_WIDTH;
    let row = 1024usize;
    let scatter = |rows: usize| {
        let b = Builder::for_tests();
        let base = b.constant("base", TensorType::f32(vec![4, row])).unwrap();
        let src = b.constant("src", TensorType::f32(vec![rows, row])).unwrap();
        let slots = b
            .constant("slots", TensorType::new(vec![rows], GDt::I32))
            .unwrap();
        let out = b.scatter_update(base, src, slots).unwrap();
        b.finish(out).unwrap()
    };
    let wide = scatter(ceiling / row + 1);
    let narrow = scatter(1);
    let folds = |g: &Graph| {
        let eqn = g.eqns().last().unwrap();
        let out_shape = g.aval(eqn.out).shape.clone();
        let out_numel = numel(&out_shape);
        is_elementwise_2d(g, Backend::SpirvVulkan, eqn, &out_shape, out_numel)
    };
    assert!(folds(&wide), "a source past the ceiling must 2-D-fold");
    assert!(
        !folds(&narrow),
        "a small source stays flat although the pool is the same"
    );
}

/// Find the named const's value id in `g`.
fn const_id(g: &Graph, name: &str) -> ValueId {
    g.values()
        .ids()
        .find(|&id| g.meta(id).name().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("const {name} not found"))
}

/// Card 298, Card 1011: the operand shape a natively-bf16 safetensors checkpoint traces for every projection
/// matmul: an F32 activation against a BF16 weight const, F32 output. `plan_eqn` refuses it on every
/// backend unwidened (olmo2-1b's prefill shape, M=128, K=2048, N=8192): every matmul body reads one element
/// type.
///
/// `widen_mismatched_matmul_dtypes`, run by `compile`, makes the eqn plan with an explicit
/// `Cast{BF16 -> F32}` of the weight: the weight const stays BF16 (the stored words upload as they are) and
/// the cast reads them from the lane the storage plan gave the const.
#[test]
fn card298_prefill_shaped_mixed_matmul_plans_after_widen() {
    let g = mixed_operand_matmul_graph(128, 2048, 8192);
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::AmdGcn(AmdArch::new("gfx942", 64)),
    ] {
        let r = try_matmul_plan(&g, backend);
        assert!(
            matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
            "a raw F32xBF16 -> F32 matmul must still be rejected on {backend:?}: got {r:?}"
        );

        let gp = widen_mismatched_matmul_dtypes(&g, &default_caps_for(backend)).unwrap();
        assert_eq!(
            gp.aval(const_id(&gp, "w")).dtype,
            GDt::BF16,
            "the BF16 weight const keeps its stored dtype on {backend:?}"
        );
        assert_eq!(
            gp.eqns()
                .iter()
                .filter(|e| matches!(e.op, OpKind::Cast { to: GDt::F32 }))
                .count(),
            1,
            "the widening is one planned Cast equation on {backend:?}"
        );
        let r = try_matmul_plan(&gp, backend);
        assert!(
            r.is_ok(),
            "the widened F32xBF16 -> F32 matmul must plan on {backend:?}: got {r:?}"
        );
    }
}

/// Card 298: the same shape at decode's M=1, which 16-alignment can never admit. On SpirvVulkan/AmdGcn a BF16
/// `[K, N]` decode projection is the retired column-tile body's shape: `widen_mismatched_matmul_dtypes` keeps
/// the weight narrow (no Cast) and `plan_eqn` refuses it with `DtypeLowering`.
#[test]
fn card298_decode_shaped_mixed_matmul_is_refused_after_widen() {
    let g = mixed_operand_matmul_graph(1, 2048, 2048);
    for backend in [Backend::SpirvVulkan, Backend::AmdGcn(AmdArch::gfx1151())] {
        let gp = widen_mismatched_matmul_dtypes(&g, &default_caps_for(backend)).unwrap();
        let r = try_matmul_plan(&gp, backend);
        assert!(
            matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
            "decode-shaped mixed matmul must be refused on {backend:?}: got {r:?}"
        );
        assert!(
            !gp.eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Cast { .. })),
            "decode-shaped mixed matmul must not gain a Cast on {backend:?}"
        );
        assert_eq!(
            gp.aval(const_id(&gp, "w")).dtype,
            GDt::BF16,
            "the refused decode weight must stay BF16 on {backend:?}"
        );
    }
}

/// Card 298, Card 1011: SPIR-V has no bf16 compute (RADV has no `SPV_KHR_bfloat16`), so a bf16 x bf16 pair is
/// widened by explicit casts that read the packed lanes. The consts keep
/// their BF16 dtype; nothing reaches codegen as a BF16 matmul.
#[test]
fn card298_spirv_widens_bf16_pairs_with_casts_not_retypes() {
    let gm = mixed_matmul_graph(GDt::BF16);
    let gp = widen_mismatched_matmul_dtypes(&gm, &default_caps_for(Backend::SpirvVulkan)).unwrap();
    for name in ["a", "w"] {
        assert_eq!(
            gp.aval(const_id(&gp, name)).dtype,
            GDt::BF16,
            "const {name} keeps its stored dtype"
        );
    }
    assert!(
        gp.eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Cast { to: GDt::F32 })),
        "the matmul reads F32 through the widening casts"
    );
    assert!(try_matmul_plan(&gp, Backend::SpirvVulkan).is_ok());
}

/// The Qwen3.8-27B LM head shape after `fold_dense_contractions`: a BF16 `[n, k]` checkpoint const
/// contracted directly, with no transpose or widening cast.
fn dense_contraction_graph(m: usize, k: usize, n: usize) -> Graph {
    let bld = Builder::for_tests();
    let a = bld
        .constant("a", TensorType::new(vec![1, m, k], GDt::F32))
        .unwrap();
    let w = bld
        .constant("w", TensorType::new(vec![n, k], GDt::BF16))
        .unwrap();
    let mm = bld
        .matmul(a, bld.transpose(w, vec![1, 0]).unwrap())
        .unwrap();
    crate::dtype_widen::fold_dense_contractions(&bld.finish(mm).unwrap()).unwrap()
}

fn try_dense_contraction_plan_planned(
    g: &Graph,
    backend: Backend,
) -> Result<(Plan, KernelChoice), PlanError> {
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::DenseContraction { .. }))
        .expect("a dense contraction eqn");
    planned_with_choice(g, eqn, backend)
}

/// Card 380 FR-009, narrowed by cards 450 ROCm and 450 PTX; the body is Card 1024's. The lowering is scoped by
/// name to the backends that have a typed packed walk: wgpu's (card 380), ROCm's (card 450 ROCm) and PTX's
/// (card 450 PTX). Above M == 1 all three plan the same register-tiled `[N, K]` body over the packed-BF16 lanes
/// with the same shape metadata; the tiling (and so the workgroup count) is each device's own
/// (`RegisterTiling::choose` reads its compute units and launch figures), the one field they may differ in.
#[test]
fn card450_dense_contraction_plans_on_nvptx_with_the_wgpu_key() {
    let g = dense_contraction_graph(2, 32, 16);
    let requests: Vec<_> = [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ]
    .into_iter()
    .map(|backend| {
        let Ok((_, choice)) = try_dense_contraction_plan_planned(&g, backend) else {
            panic!("the packed-BF16 contraction must plan on {backend:?}");
        };
        let KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(
            DenseTiledContraction {
                weight,
                schedule,
                epilogue,
                meta,
                ..
            },
        ))) = choice
        else {
            panic!("{backend:?}: expected the generated dense Tiled body, got {choice:?}");
        };
        assert_eq!(
            weight,
            DenseTiledWeight::Nk(poot_target::BufferStorage::bf16_packed())
        );
        assert!(
            matches!(schedule, Schedule::RegisterTiled { .. }),
            "{backend:?}: {schedule:?}"
        );
        format!("{:?} {weight:?} {epilogue:?}", &meta[..3])
    })
    .collect();
    assert!(
        requests.windows(2).all(|pair| pair[0] == pair[1]),
        "{requests:?}"
    );
}

/// Card 727: at M == 1 every backend plans the same generated dense Gemv request over the packed-BF16
/// `[N, K]` weight: one selection and one identical request on every backend. Host-side plan
/// evidence only; the device rows are the dense-Gemv parity rows in `poot-wgpu-device` and `poot-rocm-device`.
#[test]
fn an_m1_dense_bf16_contraction_plans_the_generated_gemv_on_every_backend() {
    let g = dense_contraction_graph(1, 32, 16);
    let requests: Vec<_> = [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ]
    .into_iter()
    .map(|backend| {
        let Ok((planned_plan, choice)) = try_dense_contraction_plan_planned(&g, backend) else {
            panic!(
                "the M == 1 contraction must plan as the Gemv's metadata-launch step on {backend:?}"
            );
        };
        let Dispatch {
            launch: Launch { meta, .. },
            ..
        } = into_with_meta(planned_plan);
        assert_eq!(meta, vec![32, 16], "{backend:?}: metadata is [K, N]");
        let KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
            weight,
            schedule,
            k,
            n,
            epilogue,
        })) = choice
        else {
            panic!("{backend:?}: expected the generated dense Gemv, got {choice:?}");
        };
        assert_eq!(
            weight,
            poot_target::BufferStorage::bf16_packed(),
            "{backend:?}"
        );
        // POOT-1405: the one request differs only in how the target reads the activation (both reads give the
        // same bits; `a_gemv_reads_activation_vectors_only_where_every_run_is_a_whole_aligned_group`).
        let kg::Schedule::Gemv {
            width,
            cols,
            unroll,
            activation,
        } = schedule
        else {
            panic!("{backend:?}: the Gemv schedule, got {schedule:?}");
        };
        assert_eq!(
            activation,
            crate::predicates::gemv_activation(backend, k, unroll),
            "{backend:?}"
        );
        format!("{weight:?} {width} {cols} {unroll} {k} {n} {epilogue:?}")
    })
    .collect();
    assert!(
        requests.windows(2).all(|pair| pair[0] == pair[1]),
        "{requests:?}"
    );
}

/// Card 380 FR-010. A BF16 const is packed-lane when every consumer is a
/// `DenseContraction`/`DenseRowGather`/`Cast` packed-lane reader (typed walk). No const is retyped (Card 1011): one with any other
/// consumer stays BF16 and that consumer is planned or refused on its own.
#[test]
fn card380_bf16_const_stays_narrow_only_for_a_contraction_weight() {
    let folded = dense_contraction_graph(1, 32, 16);
    let prepared =
        widen_mismatched_matmul_dtypes(&folded, &default_caps_for(Backend::SpirvVulkan)).unwrap();
    let weight = const_id(&prepared, "w");
    assert_eq!(
        prepared.aval(weight).dtype,
        GDt::BF16,
        "a contraction weight must stay BF16 on wgpu: the typed walk uploads it as packed u32 lanes"
    );
    assert!(
        !prepared
            .eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Cast { .. })),
        "the fold plus the narrowed retype leave no widening cast"
    );
    assert!(bf16_const_feeds_only_packed_readers(
        &folded,
        &Default::default(),
        weight
    ));

    // An ordinary mixed matmul at M>1 (prefill, not a decode GEMV) has no packed-lane consumer until the
    // widening pass puts a `Cast` in front of it; the weight is never retyped.
    let mixed = mixed_operand_matmul_graph(128, 32, 16);
    assert!(!bf16_const_feeds_only_packed_readers(
        &mixed,
        &Default::default(),
        const_id(&mixed, "w")
    ));
    let widened =
        widen_mismatched_matmul_dtypes(&mixed, &default_caps_for(Backend::SpirvVulkan)).unwrap();
    assert_eq!(widened.aval(const_id(&mixed, "w")).dtype, GDt::BF16);
    assert!(bf16_const_feeds_only_packed_readers(
        &widened,
        &Default::default(),
        const_id(&widened, "w")
    ));

    // The decode-shaped mixed matmul is not a packed reader either: its weight stays BF16 (never retyped) and
    // the matmul is refused at planning.
    let decode_mixed = mixed_operand_matmul_graph(1, 32, 16);
    assert!(!bf16_const_feeds_only_packed_readers(
        &decode_mixed,
        &Default::default(),
        const_id(&decode_mixed, "w")
    ));
    assert_eq!(
        widen_mismatched_matmul_dtypes(&decode_mixed, &default_caps_for(Backend::SpirvVulkan))
            .unwrap()
            .aval(const_id(&decode_mixed, "w"))
            .dtype,
        GDt::BF16
    );

    // A widening `Cast` is a packed reader (Card 1011); a second consumer that is not one disqualifies the
    // const: that consumer would read f32 lanes.
    let shared = derive(&folded, |draft| {
        let widened = draft
            .values_mut()
            .push(poot_graph_ir::graph::ValueMeta::new(
                TensorType::new(vec![16, 32], GDt::F32),
                Storage::Device,
                None,
            ));
        draft.eqns_mut().push(poot_graph_ir::Eqn {
            op: OpKind::Cast { to: GDt::F32 },
            inputs: vec![poot_graph_ir::Operand::Value(weight)],
            out: widened,
            layer: None,
        });
    })
    .unwrap();
    assert!(bf16_const_feeds_only_packed_readers(
        &shared,
        &Default::default(),
        weight
    ));
    let shared = derive(&shared, |draft| {
        let negated = draft
            .values_mut()
            .push(poot_graph_ir::graph::ValueMeta::new(
                TensorType::new(vec![16, 32], GDt::BF16),
                Storage::Device,
                None,
            ));
        draft.eqns_mut().push(poot_graph_ir::Eqn {
            op: OpKind::Unary(poot_graph_ir::UnOp::Neg),
            inputs: vec![poot_graph_ir::Operand::Value(weight)],
            out: negated,
            layer: None,
        });
    })
    .unwrap();
    assert!(!bf16_const_feeds_only_packed_readers(
        &shared,
        &Default::default(),
        weight
    ));

    // A const nothing reads is not eligible: the binder would upload two-byte words no kernel can
    // interpret.
    let mut orphan = None;
    let orphaned = derive(&folded, |draft| {
        let id = draft
            .values_mut()
            .push(poot_graph_ir::graph::ValueMeta::new(
                TensorType::new(vec![4, 4], GDt::BF16),
                Storage::Const,
                Some("orphan".into()),
            ));
        draft.inputs_mut().push(id);
        draft.consts_mut().push(id);
        orphan = Some(id);
    })
    .unwrap();
    let orphan = orphan.unwrap();
    assert!(!bf16_const_feeds_only_packed_readers(
        &orphaned,
        &Default::default(),
        orphan
    ));
}

/// A Qwen3.8-27B token-embedding shape (BF16 source) or a Qwen4Exp PLE shard shape (E4M3FN source)
/// after `fold_dense_bf16_row_gathers`: a narrow `[v, r]` checkpoint const gathered by an I32 index and
/// widened in one equation, with no cast or F32 table copy. One builder for both source dtypes, so the
/// two plans differ only in the dtype the planner sees.
fn dense_row_gather_graph(v: usize, r: usize, rows: usize, source: GDt) -> Graph {
    let bld = Builder::for_tests();
    let table = bld
        .constant("embed", TensorType::new(vec![v, r], source))
        .unwrap();
    let token = bld
        .slot(
            poot_graph_ir::Slot::Token,
            TensorType::new(vec![rows], GDt::I32),
        )
        .unwrap();
    let widened = bld
        .cast(bld.gather(table, 0, token).unwrap(), GDt::F32)
        .unwrap();
    crate::dtype_widen::fold_dense_bf16_row_gathers(&bld.finish(widened).unwrap()).unwrap()
}

fn try_dense_row_gather_plan_planned(
    g: &Graph,
    backend: Backend,
) -> Result<(Plan, KernelChoice), PlanError> {
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::DenseRowGather { .. }))
        .expect("a dense row gather eqn");
    planned_with_choice(g, eqn, backend)
}

fn try_dense_row_gather_plan(g: &Graph, backend: Backend) -> Result<Plan, PlanError> {
    try_dense_row_gather_plan_planned(g, backend).map(|(plan, _)| plan)
}

/// Card 381 FR-008, narrowed by cards 450 ROCm and 450 PTX. Scoped by name to the backends that have a
/// kernel: wgpu's typed walk landed it (card 381), the ROCm typed walk the same lowering on AmdGcn
/// (card 450 ROCm), and the PTX typed walk the same lowering on Nvptx (card 450 PTX).
#[test]
fn card450_dense_row_gather_plans_on_nvptx_with_the_wgpu_key() {
    let g = dense_row_gather_graph(8, 4, 3, GDt::BF16);
    let Ok((planned_plan, choice)) = try_dense_row_gather_plan_planned(&g, Backend::Nvptx) else {
        panic!("the row gather must plan on Nvptx as a plain Compute");
    };
    let Dispatch { .. } = into_plain(planned_plan);
    assert!(
        is_imported(&choice, ImportedKernel::PackedBf16RowGather),
        "{choice:?}"
    );
}

/// Card 381 FR-009, widened by Card 449 D1 and POOT-1060. `E4M3FN` is the second admitted row-gather
/// source dtype; its packed-lane kernel plans under its own key on all three typed packed walks
/// (SpirvVulkan, Nvptx, AmdGcn), with no metadata buffer. The refusal rows the older revisions held
/// for Nvptx (`DtypeLowering`) and AmdGcn (`E4m3Storage`) are gone: the device rows
/// (`dense_e4m3_row_gather_is_bit_exact_*`) run the body on hardware.
#[test]
fn card449_d1_dense_row_gather_e4m3_plans_on_every_typed_packed_backend() {
    let g = dense_row_gather_graph(8, 4, 3, GDt::E4M3FN);
    assert!(
        g.eqns().iter().any(|e| matches!(
            e.op,
            OpKind::DenseRowGather {
                source: GDt::E4M3FN
            }
        )),
        "the fold must have produced a DenseRowGather over the E4M3FN table"
    );
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let Ok((planned_plan, choice)) = try_dense_row_gather_plan_planned(&g, backend) else {
            panic!("the E4M3FN row gather must plan on {backend:?}, with no metadata buffer");
        };
        let Dispatch { .. } = into_plain(planned_plan);
        assert!(
            is_imported(&choice, ImportedKernel::PackedE4m3RowGather),
            "the choice must name the E4M3FN source dtype's kernel on {backend:?}: {choice:?}"
        );
    }
}

/// Card 450 ROCm: the very same imported packed-BF16 row-gather body plans on AmdGcn, with the key
/// unchanged (the key names the source dtype, not the backend, as the wgpu row pins), so the ROCm
/// typed walk dispatches the identical kernel the wgpu walk does. Host-side plan evidence only; the
/// device receipt is `card450rocm_exact_decode_two_steps_match_cpu_oracle`.
#[test]
fn card450_dense_row_gather_plans_on_amdgcn_with_the_wgpu_key() {
    let g = dense_row_gather_graph(8, 4, 3, GDt::BF16);
    let Ok((planned_plan, choice)) =
        try_dense_row_gather_plan_planned(&g, Backend::AmdGcn(AmdArch::gfx1151()))
    else {
        panic!("the row gather must plan on AmdGcn as a plain Compute");
    };
    let Dispatch { .. } = into_plain(planned_plan);
    assert!(
        is_imported(&choice, ImportedKernel::PackedBf16RowGather),
        "{choice:?}"
    );
}

/// Card 381 FR-009. The wgpu retype rule keeps an embedding table narrow for the same reason as a
/// contraction weight: the imported row-gather kernel reads it as packed u32 lanes. Operand position
/// matters: a const read as the row gather's index is an `&[i32]` operand of that kernel, so keeping it
/// narrow would feed two-byte words to a four-byte parameter.
#[test]
fn card381_bf16_const_stays_narrow_for_a_row_gather_table() {
    let folded = dense_row_gather_graph(8, 4, 3, GDt::BF16);
    let table = const_id(&folded, "embed");
    assert!(bf16_const_feeds_only_packed_readers(
        &folded,
        &Default::default(),
        table
    ));
    let prepared =
        widen_mismatched_matmul_dtypes(&folded, &default_caps_for(Backend::SpirvVulkan)).unwrap();
    assert_eq!(
        prepared.aval(table).dtype,
        GDt::BF16,
        "an embedding table must stay BF16 on wgpu: the typed walk uploads it as packed u32 lanes"
    );
    assert!(
        !prepared
            .eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Cast { .. })),
        "the fold plus the narrowed retype leave no widening cast"
    );

    // The un-folded traced shape (a plain `Gather` of the BF16 table): the table is not a packed-lane
    // reader's operand, and it is never retyped to F32 (Card 1011). The storage plan refuses the gather's
    // BF16-element body against the packed lane (`bf16_const_reads_refuse_a_reader_without_the_lane`).
    let bld = Builder::for_tests();
    let unfolded_table = bld
        .constant("embed", TensorType::new(vec![8, 4], GDt::BF16))
        .unwrap();
    let token = bld
        .slot(
            poot_graph_ir::Slot::Token,
            TensorType::new(vec![3], GDt::I32),
        )
        .unwrap();
    let unfolded_widened = bld
        .cast(bld.gather(unfolded_table, 0, token).unwrap(), GDt::F32)
        .unwrap();
    let unfolded = bld.finish(unfolded_widened).unwrap();
    let id = const_id(&unfolded, "embed");
    assert!(!bf16_const_feeds_only_packed_readers(
        &unfolded,
        &Default::default(),
        id
    ));
    let prepared =
        widen_mismatched_matmul_dtypes(&unfolded, &default_caps_for(Backend::SpirvVulkan)).unwrap();
    assert_eq!(prepared.aval(id).dtype, GDt::BF16);
}

/// Card 381. On wgpu the row gather plans to the imported packed kernel as a plain dispatch with no
/// metadata buffer (the row width is derived in-kernel). The key must name the source dtype, so a
/// packed-BF16 gather never shares a cache entry or disk artifact with the f32 table gather of the same
/// shapes (the card-128 Bug-1 class) - nor, since Card 449 D1, with the packed-E4M3FN gather of the
/// same shapes, which decodes a different byte layout.
#[test]
fn card381_row_gather_plan_key_names_the_source_dtype() {
    let folded = dense_row_gather_graph(8, 4, 3, GDt::BF16);
    let Ok((planned_plan, choice)) =
        try_dense_row_gather_plan_planned(&folded, Backend::SpirvVulkan)
    else {
        panic!("the row gather must plan on SpirvVulkan, with no metadata buffer");
    };
    let Dispatch { key, .. } = into_plain(planned_plan);
    assert!(
        is_imported(&choice, ImportedKernel::PackedBf16RowGather),
        "the choice must name the source dtype's kernel: {choice:?}"
    );

    // The packed-E4M3FN gather of the same shapes is a different body and must not collide either.
    let e4m3_folded = dense_row_gather_graph(8, 4, 3, GDt::E4M3FN);
    let Ok(e4m3_plan) = try_dense_row_gather_plan(&e4m3_folded, Backend::SpirvVulkan) else {
        panic!("the E4M3FN row gather must plan on SpirvVulkan, with no metadata buffer");
    };
    let e4m3_key = into_plain(e4m3_plan).key().to_string();
    assert_ne!(
        key, e4m3_key,
        "the BF16 and E4M3FN row gathers of the same shapes must not share a cache entry"
    );

    // The f32 table gather of the same shapes uses a different kernel and must not collide.
    let f32_gather = {
        let bld = Builder::for_tests();
        let table = bld.constant("embed", TensorType::f32(vec![8, 4])).unwrap();
        let token = bld
            .slot(
                poot_graph_ir::Slot::Token,
                TensorType::new(vec![3], GDt::I32),
            )
            .unwrap();
        let gathered = bld.gather(table, 0, token).unwrap();
        bld.finish(gathered).unwrap()
    };
    let eqn = f32_gather
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Gather { .. }))
        .expect("an f32 gather eqn");
    let (f32_plan, choice) = planned_with_choice(&f32_gather, eqn, Backend::SpirvVulkan).unwrap();
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
                storage: kg::Storage::Elements { carrier: Ty::F32 },
                op: kg::MovementOp::Gather { axis: 0, .. },
                ..
            }))
        ),
        "{choice:?}"
    );
    let f32_key = one_dispatch(f32_plan).key().to_string();
    assert_ne!(key, f32_key);
}

/// Card 380. The plan key must name the operand dtype, so a packed-BF16 kernel never shares a cache entry
/// or disk artifact with an f32-operand kernel of the same shapes (the card-128 Bug-1 class).
///
/// At M == 1 the plan is the generated dense Gemv (see [`dense_bf16_decode_gemv_selection_rows`]), whose
/// request names the packed-BF16 weight storage, so the key this row pins is that body's; the
/// M > 1 half is in the selection test.
#[test]
fn card380_dense_contraction_plan_key_names_the_operand_dtype() {
    let contraction = dense_contraction_graph(1, 32, 16);
    let Ok((planned_plan, choice)) =
        try_dense_contraction_plan_planned(&contraction, Backend::SpirvVulkan)
    else {
        panic!(
            "the M == 1 contraction must plan as the Gemv's metadata-launch step on SpirvVulkan"
        );
    };
    let Dispatch { key, .. } = into_with_meta(planned_plan);
    assert!(
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
                weight,
                ..
            })) if weight == poot_target::BufferStorage::bf16_packed()
        ),
        "the choice must name the packed-BF16 weight: {choice:?}"
    );

    // An ordinary f32 matmul of the same shapes must not produce the same key.
    let f32_matmul = {
        let bld = Builder::for_tests();
        let a = bld
            .constant("a", TensorType::new(vec![1, 1, 32], GDt::F32))
            .unwrap();
        let w = bld
            .constant("w", TensorType::new(vec![32, 16], GDt::F32))
            .unwrap();
        let mm = bld.matmul(a, w).unwrap();
        bld.finish(mm).unwrap()
    };
    let f32_key = one_dispatch(try_matmul_plan(&f32_matmul, Backend::SpirvVulkan).unwrap())
        .key()
        .to_string();
    assert_ne!(key, f32_key);
}

/// POOT-1405: the planner reads a Gemv's activation in 4-lane vector groups exactly where it can prove every full
/// run starts on a group and the target lowers a mixed vector and scalar read of one param: on NVPTX for a dense
/// contraction whose `K` is a multiple of four (the dense run is four), never for `K = 30`, and never on SPIR-V or
/// AMDGPU; a packed contraction (its `K` a whole number of 32-value blocks) on NVPTX only. Driven through the
/// planner's own entry: the planned kernel request carries the choice.
///
/// Mutation: drop the `k % 4` condition in `gemv_activation`; the `K = 30` NVPTX row goes red. Mutation: drop the
/// `mixed_vector_reads` condition; the SPIR-V and AMDGPU rows go red.
#[test]
fn a_gemv_reads_activation_vectors_only_where_every_run_is_a_whole_aligned_group() {
    use kg::GemvActivation::{Scalar, Vec4};
    let planned_read = |choice: &KernelChoice| match choice {
        KernelChoice::Generated(KernelRequest::Contraction(
            ContractionSpec::DenseGemv { schedule, .. }
            | ContractionSpec::Packed(kg::PackedContraction { schedule, .. }),
        )) => match schedule {
            kg::Schedule::Gemv { activation, .. } => *activation,
            other => panic!("a decode contraction plans the Gemv, got {other:?}"),
        },
        other => panic!("a generated contraction, got {other:?}"),
    };
    for (backend, k, want) in [
        (Backend::Nvptx, 32, Vec4),
        (Backend::Nvptx, 30, Scalar),
        (Backend::SpirvVulkan, 32, Scalar),
        (Backend::AmdGcn(AmdArch::gfx1151()), 32, Scalar),
    ] {
        let g = dense_contraction_graph(1, k, 16);
        let (_, choice) = try_dense_contraction_plan_planned(&g, backend).unwrap();
        assert_eq!(planned_read(&choice), want, "dense {backend:?} K={k}");
    }
    for (backend, want) in [
        (Backend::Nvptx, Vec4),
        (Backend::SpirvVulkan, Scalar),
        (Backend::AmdGcn(AmdArch::gfx1151()), Scalar),
    ] {
        let builder = Builder::for_tests();
        let x = builder
            .slot_named(
                poot_graph_ir::Slot::Activation,
                "x",
                TensorType::f32(vec![1, 256]),
            )
            .unwrap();
        let weight =
            poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q4_K, [64, 256])
                .unwrap();
        let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
            &builder, x, "layer", weight, None, None,
        )
        .unwrap();
        let target = Target::Gpu {
            backend,
            caps: default_caps_for(backend),
        };
        let program = compile(
            &builder.finish(out).unwrap(),
            &target,
            &CompileOptions::new(CompileLimits::STANDARD),
        )
        .unwrap_or_else(|error| panic!("packed {backend:?}: {error}"));
        let (eqn, _) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. }))
            .expect("a planned packed contraction");
        assert_eq!(
            planned_read(program.kernel_choice(eqn)),
            want,
            "packed {backend:?}"
        );
    }
}

/// POOT-1405: the planner narrows a Gemv dispatch to 32-bit index math exactly where it proves the generator's
/// index bound keeps every index of the longest planned binding within the 32-bit range and the target lowers
/// that: a Q4_K `[64, 256]` and a packed-BF16 `[16, 32]` Gemv on NVPTX narrow (each under its generator's bound,
/// with a key of its own), as does a Q4_K `[65536, 5120]` (47.2 M words, so 1.51 G bit offsets: the bound counts the
/// source's words, not its carrier's graph shape); a Q4_K `[65536, 8192]` (75.5 M words, 2.4 G bit offsets) stays
/// native on NVPTX; nothing
/// narrows on SPIR-V or AMDGPU. Driven through `compile` and the planner's own dense entry.
///
/// Mutation: drop the `fits_u32` condition in `build`; the `[65536, 8192]` NVPTX row goes red. Mutation: drop the
/// `narrow_index` condition; the SPIR-V and AMDGPU rows go red. Mutation: count the carrier operands in
/// `try_plan_contraction`'s longest binding; the `[65536, 5120]` row goes red.
#[test]
fn the_planner_narrows_index_math_only_where_its_proof_holds_on_a_target_that_lowers_it() {
    let packed_dispatch = |backend: Backend, n: usize, k: usize| -> Dispatch {
        let builder = Builder::for_tests();
        let x = builder
            .slot_named(
                poot_graph_ir::Slot::Activation,
                "x",
                TensorType::f32(vec![1, k]),
            )
            .unwrap();
        let weight =
            poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q4_K, [n, k])
                .unwrap();
        let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
            &builder, x, "layer", weight, None, None,
        )
        .unwrap();
        let target = Target::Gpu {
            backend,
            caps: default_caps_for(backend),
        };
        let program = compile(
            &builder.finish(out).unwrap(),
            &target,
            &CompileOptions::new(CompileLimits::STANDARD),
        )
        .unwrap_or_else(|error| panic!("packed {backend:?} [{n}, {k}]: {error}"));
        let (_, plan) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. }))
            .expect("a planned packed contraction");
        plan.single()
            .unwrap_or_else(|| panic!("one Gemv dispatch, got {plan:?}"))
            .clone()
    };
    // The generator's Gemv overshoot at the CUDA class's 256 lanes and 16-element run.
    let gemv_slack = 256 * 16 + 16 + 256 + 64;
    let narrowed = packed_dispatch(Backend::Nvptx, 64, 256);
    assert_eq!(
        narrowed.body().index_width,
        poot_kernel_ir::IndexWidth::U32(poot_kernel_ir::IndexBound {
            scale: 32,
            slack: gemv_slack
        }),
        "Q4_K [64, 256] on NVPTX"
    );
    assert!(narrowed.key().ends_with(":u32"), "{}", narrowed.key());
    let wide = packed_dispatch(Backend::Nvptx, 65536, 8192);
    assert_eq!(
        wide.body().index_width,
        poot_kernel_ir::IndexWidth::Native,
        "Q4_K [65536, 8192] on NVPTX: its bit offsets pass 2^31"
    );
    assert!(!wide.key().ends_with(":u32"), "{}", wide.key());
    // 47.2 M words, 1.51 G bit offsets: its graph-shaped carrier is far longer than that, its binding is not.
    assert!(
        matches!(
            packed_dispatch(Backend::Nvptx, 65536, 5120)
                .body()
                .index_width,
            poot_kernel_ir::IndexWidth::U32(_)
        ),
        "Q4_K [65536, 5120] on NVPTX narrows on its source's word count"
    );
    for backend in [Backend::SpirvVulkan, Backend::AmdGcn(AmdArch::gfx1151())] {
        assert_eq!(
            packed_dispatch(backend, 64, 256).body().index_width,
            poot_kernel_ir::IndexWidth::Native,
            "Q4_K [64, 256] on {backend:?}"
        );
    }
    let g = dense_contraction_graph(1, 32, 16);
    let (plan, _) = try_dense_contraction_plan_planned(&g, Backend::Nvptx).unwrap();
    let dense = into_with_meta(plan);
    assert!(
        matches!(
            dense.body().index_width,
            poot_kernel_ir::IndexWidth::U32(poot_kernel_ir::IndexBound { scale: 2, .. })
        ),
        "packed-BF16 [16, 32] on NVPTX: {:?}",
        dense.body().index_width
    );
}

/// Body, launch and selection for one packed-BF16 `DenseContraction` are chosen together (Card 727): at
/// M == 1, on every backend, the generated dense Gemv takes it with the launch `dense_gemv_launch` picks,
/// its workgroup the schedule's `width` and its grid the schedule's own `grid_threads`, `[K, N]` riding in
/// metadata; above M == 1 the generated `[N, K]` Tiled body takes it (Card 1024).
///
/// Mutation: make `dense_decode_gemv_in` return `None`; the M == 1 rows go red with "the dense Gemv must
/// take the M == 1 contraction".
#[test]
fn dense_bf16_decode_gemv_selection_rows() {
    let (k, n) = (32usize, 16usize);

    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let caps = default_caps_for(backend);
        let g = dense_contraction_graph(1, k, n);
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::DenseContraction { .. }))
            .expect("a dense contraction eqn");
        let out_shape = g.aval(eqn.out).shape.clone();
        let out_numel = numel(&out_shape);
        assert_eq!(out_numel, n, "M == 1: one row of {n} outputs");
        let launch = dense_decode_gemv_in(eqn, &out_shape, out_numel, &caps).unwrap_or_else(|| {
            panic!("{backend:?}: the dense Gemv must take the M == 1 contraction")
        });
        assert_eq!(launch, dense_gemv_launch(n, caps.compute_units));
        let DenseGemvLaunch { width, unroll, .. } = launch;
        // K = 32 and the 4-element run put every run on a 16-byte group: NVPTX reads them as vectors.
        let activation = if backend == Backend::Nvptx {
            poot_kernelgen::GemvActivation::Vec4
        } else {
            poot_kernelgen::GemvActivation::Scalar
        };
        let schedule = poot_kernelgen::Schedule::Gemv {
            width,
            cols: launch.cols,
            unroll,
            activation,
        };
        assert_eq!(launch.schedule(backend, k), schedule, "{backend:?}");

        let (planned_plan, choice) = try_dense_contraction_plan_planned(&g, backend).unwrap();
        let Dispatch {
            body,
            launch: Launch { grid, meta, .. },
            ..
        } = into_with_meta(planned_plan);
        assert!(
            matches!(
                &choice,
                KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
                    schedule: chosen,
                    ..
                })) if *chosen == schedule
            ),
            "{backend:?}: the Gemv body: {choice:?}"
        );
        assert_eq!(body.workgroup_size, [width, 1, 1], "{backend:?}");
        assert_eq!(
            grid,
            [schedule.grid_threads(1, 1, n) as u32, 1, 1],
            "{backend:?}: the schedule's own grid"
        );
        assert_eq!(meta, vec![k as u32, n as u32], "{backend:?}: [K, N]");
    }

    // M > 1 runs the register-tiled `[N, K]` body on every backend: no Gemv, no imported kernel.
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let g = dense_contraction_graph(2, k, n);
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::DenseContraction { .. }))
            .expect("a dense contraction eqn");
        let out_shape = g.aval(eqn.out).shape.clone();
        let out_numel = numel(&out_shape);
        assert_eq!(out_numel, 2 * n, "M == 2: two rows of {n} outputs");
        assert_eq!(
            dense_decode_gemv_in(eqn, &out_shape, out_numel, &default_caps_for(backend)),
            None,
            "{backend:?}: the Gemv is M == 1, so M > 1 must decline"
        );
        let (_, choice) = try_dense_contraction_plan_planned(&g, backend).unwrap();
        assert!(
            matches!(
                &choice,
                KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(
                    DenseTiledContraction {
                        weight: DenseTiledWeight::Nk(storage),
                        meta,
                        ..
                    }
                ))) if *storage == poot_target::BufferStorage::bf16_packed() && meta[2] == 2
            ),
            "{backend:?}: M > 1 must plan the generated dense Tiled body: {choice:?}"
        );
    }
}

/// The dense Gemv arm of the packed-reader planner passes the one finalizer like every other arm: on a device
/// whose workgroup holds one invocation fewer than the Gemv's `width`, planning the folded M == 1 contraction is
/// a typed `WorkgroupInvocations` refusal naming both figures, not a plan a loader would receive. (`compile`
/// reverts a fold it cannot plan, so this reads the folded equation's own plan.)
///
/// Mutation: in `planner/packed.rs` return the Gemv arm's `Planned` before `finalize` runs; the over-limit plan
/// succeeds and this row goes red ("expected a WorkgroupInvocations refusal").
#[test]
fn the_dense_gemv_arm_is_finalized() {
    let backend = Backend::SpirvVulkan;
    let g = dense_contraction_graph(1, 64, 256);
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::DenseContraction { .. }))
        .expect("a dense contraction eqn");
    let (planned_plan, _) = planned_with_choice(&g, eqn, backend).unwrap();
    let Dispatch { body, .. } = into_with_meta(planned_plan);
    let width = body.workgroup_size[0];
    let mut small = default_caps_for(backend);
    small.max_workgroup_invocations = width - 1;
    match planned_with_choice_caps(&g, eqn, backend, &small) {
        Err(PlanError::Refused(refusal)) => assert_eq!(
            refusal.missing,
            Capability::KernelResources(
                crate::device_validation::ResourceRefusal::WorkgroupInvocations {
                    invocations: u64::from(width),
                    limit: width - 1,
                }
            )
        ),
        other => panic!(
            "expected a WorkgroupInvocations refusal, got {:?}",
            other.map(|(_, choice)| choice)
        ),
    }
}

/// Decode attention `scores @ V` at `hq` heads, cache length `cap` and head dim `d`: `[1, hq, 1, cap] @ [1, hq,
/// cap, d]`, the shape `is_decode_attn_gemv_in` routes to the generated LDS-reduction GEMV.
fn attn_scores_v_graph(hq: usize, cap: usize, d: usize) -> Graph {
    let bld = Builder::for_tests();
    let scores = bld
        .constant("scores", TensorType::f32(vec![1, hq, 1, cap]))
        .unwrap();
    let v = bld
        .constant("v", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let out = bld.matmul(scores, v).unwrap();
    bld.finish(out).unwrap()
}

/// Card 727 SC-005, plan side: a contraction plan whose body sizes its LDS reduction by its baked
/// width keeps that width through the planner's launch shaping, by its request type. The fixture is the decode
/// attention GEMV (`GEMV_WIDTH` lanes) on a device whose X grid holds a single workgroup, so the generic bump
/// (raise a one-thread-per-output body to 256 lanes when `out_numel / width` passes the X cap) would fire; the
/// plan keeps `GEMV_WIDTH` and an LDS array of exactly that many lanes. The device half (the result equals the
/// oracle) is `a_contraction_keeps_its_lds_width_through_the_planner_*` in `poot-wgpu-device` and `poot-rocm-device`.
///
/// Mutation: in `kernel_mapping::shape_launch` drop the request-type exemption (`contraction = false`); the
/// workgroup becomes 256 and this row goes red.
#[test]
fn a_contraction_plan_keeps_its_lds_width_through_the_planner() {
    let g = attn_scores_v_graph(4, 64, 64);
    let eqn = &g.eqns()[0];
    let mut caps = default_caps_for(Backend::SpirvVulkan);
    caps.max_grid[0] = 1;
    let (plan, choice) = planned_with_choice_caps(&g, eqn, Backend::SpirvVulkan, &caps).unwrap();
    assert!(
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::AttnScoresVGemv { .. }
            ))
        ),
        "{choice:?}"
    );
    let Some(Dispatch { body, .. }) = plan.single() else {
        panic!("a single dispatch: {plan:?}");
    };
    assert_eq!(body.workgroup_size, [GEMV_WIDTH as u32, 1, 1]);
    assert_eq!(
        body.workgroup_locals
            .iter()
            .map(|decl| decl.len)
            .collect::<Vec<_>>(),
        vec![GEMV_WIDTH as u32],
        "the reduction's LDS is sized by the same width"
    );
}

/// What `compile` makes of a decode projection `x [1, 1, k] f32 @ w bf16` on `backend`, with `w` held `[k, n]`
/// (`transposed: false`) or in checkpoint `[n, k]` order read through a transpose (`transposed: true`): the
/// single contraction's kernel choice, or the compile error.
fn bf16_projection_contraction_choice(
    backend: Backend,
    transposed: bool,
    k: usize,
    n: usize,
) -> Result<KernelChoice, CompileError> {
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![1, 1, k], GDt::F32))
        .unwrap();
    let out = if transposed {
        let w = b
            .constant("w", TensorType::new(vec![n, k], GDt::BF16))
            .unwrap();
        b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap()
    } else {
        let w = b
            .constant("w", TensorType::new(vec![k, n], GDt::BF16))
            .unwrap();
        b.matmul(x, w).unwrap()
    };
    let graph = b.finish(out).unwrap();
    let options = CompileOptions::new(CompileLimits::STANDARD);
    let target = Target::Gpu {
        backend,
        caps: default_caps_for(backend),
    };
    let program = compile(&graph, &target, &options)?;
    let mut contractions = program.planned().filter(|(eqn, _)| {
        matches!(
            eqn.op,
            OpKind::MatMul { .. } | OpKind::DenseContraction { .. }
        )
    });
    let (eqn, _) = contractions
        .next()
        .unwrap_or_else(|| panic!("{backend:?} transposed={transposed}: no contraction planned"));
    assert!(contractions.next().is_none(), "one contraction");
    Ok(program.kernel_choice(eqn).clone())
}

/// POOT-739 SC-006, at planning: through `compile`, a BF16 decode projection (M == 1) whose weight is held in
/// checkpoint `[N, K]` order plans the generated dense Gemv on every backend. The same projection over a
/// `[K, N]` weight is a typed `DtypeLowering` refusal on SPIR-V and ROCm (the retired column-tile body was its
/// only lowering; a per-column Gemv over BF16 would read `[K, N]` at a stride of `N` per lane), and on NVPTX the
/// weight is widened to f32 and the projection plans the f32 decode GEMV, never the BF16 dense Gemv. Mutation: in `dtype_widen::matmul_bf16_column_gemv` return `false`; the weight is widened by a
/// Cast, the projection plans the f32 decode GEMV and this row goes red.
#[test]
fn a_k_by_n_bf16_decode_projection_is_refused_on_the_packed_lane_backends() {
    let (k, n) = (896, 256);
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let nk = bf16_projection_contraction_choice(backend, true, k, n)
            .unwrap_or_else(|e| panic!("{backend:?} [N, K]: {e}"));
        assert!(
            matches!(
                nk,
                KernelChoice::Generated(KernelRequest::Contraction(
                    ContractionSpec::DenseGemv { .. }
                ))
            ),
            "{backend:?} [N, K]: {nk:?}"
        );
        let kn = bf16_projection_contraction_choice(backend, false, k, n);
        if backend == Backend::Nvptx {
            let kn = kn.unwrap_or_else(|e| panic!("{backend:?} [K, N]: {e}"));
            assert!(
                matches!(
                    kn,
                    KernelChoice::Generated(KernelRequest::Contraction(
                        ContractionSpec::GemvChunk { .. }
                    ))
                ),
                "{backend:?} [K, N]: {kn:?}"
            );
        } else {
            assert!(
                matches!(
                    &kn,
                    Err(CompileError::Plan(e))
                        if matches!(&**e, PlanError::Refused(r) if r.missing == Capability::DtypeLowering)
                ),
                "{backend:?} [K, N]: expected a DtypeLowering refusal, got {kn:?}"
            );
        }
    }
}

/// POOT-1108 on the production path: an exact-I32 chain, its cast and the float arithmetic after it plan as ONE
/// launch on every planned backend, and the launch is the pointwise generator's (no `I32ToF32` kernel).
/// Mutation: remove `fuse_index_casts` from `compile`; the cast launches on its own and the count is 2 (the
/// region and the cast), or 3 where the chain does not fuse.
#[test]
fn an_i32_chain_cast_and_float_scale_compile_to_one_launch_on_every_backend() {
    let b = Builder::for_tests();
    let a = b
        .slot_named(
            poot_graph_ir::Slot::Token,
            "a",
            TensorType::new(vec![2, 4], GDt::I32),
        )
        .unwrap();
    let c = b
        .slot_named(
            poot_graph_ir::Slot::Token,
            "c",
            TensorType::new(vec![2, 4], GDt::I32),
        )
        .unwrap();
    let w = b.constant("w", TensorType::f32(vec![2, 4])).unwrap();
    let hit = b.binary(GBinOp::Ge, a, c).unwrap();
    let out = b
        .binary(GBinOp::Mul, b.cast(hit, GDt::F32).unwrap(), w)
        .unwrap();
    let g = b.finish(out).unwrap();
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let program = compiled(&g, backend);
        assert_eq!(
            dispatches(&program),
            1,
            "{backend:?}: the chain, its cast and the scale are one launch"
        );
        let (eqn, plan) = program
            .planned()
            .find(|(_, plan)| !matches!(plan, Plan::Alias(_) | Plan::View { .. }))
            .expect("one launching equation");
        assert!(
            matches!(eqn.op, OpKind::Fused(_)),
            "{backend:?}: planned as a fused region, got {}",
            eqn.op.name()
        );
        assert!(
            matches!(
                program.kernel_choice(eqn),
                KernelChoice::Generated(KernelRequest::Pointwise(_))
            ),
            "{backend:?}: the pointwise generator writes it"
        );
        assert!(matches!(plan, Plan::Dispatches(d) if d.len() == 1));
    }
}

/// Rebuild `plan`'s dispatches with `rebind` applied to every binding, through the production
/// `DispatchPlan::new`.
fn rebound(plan: Plan, rebind: impl Fn(&Binding) -> Binding) -> Plan {
    let Plan::Dispatches(dispatches) = plan else {
        panic!("expected a dispatching plan");
    };
    let rebuilt = dispatches
        .into_dispatches()
        .into_iter()
        .map(|dispatch| {
            let binding = rebind(dispatch.binding());
            dispatch.with_binding(binding)
        })
        .collect();
    Plan::Dispatches(DispatchPlan::new(rebuilt, Vec::new()).expect("the rebound plan validates"))
}

/// POOT-1115: a dispatch binding an operand the equation lacks is refused by `finalize`, the one place the
/// bindings and the equation are in hand together. [`DispatchPlan::new`] cannot catch it: it knows kernel
/// params, not equations.
///
/// Mutation: delete the `check_operand_bindings` call in `finalize`; the rebound plan is accepted and this
/// row goes red.
#[test]
fn a_binding_naming_an_operand_the_equation_lacks_is_refused_by_finalize() {
    use crate::kernel_mapping::finalize;

    let b = Builder::for_tests();
    let table = b
        .constant("table", TensorType::new(vec![8, 3], GDt::F32))
        .unwrap();
    let index = b
        .constant("index", TensorType::new(vec![2], GDt::I32))
        .unwrap();
    let gathered = b.gather(table, 0, index).unwrap();
    let g = b.finish(gathered).unwrap();
    let backend = Backend::SpirvVulkan;
    let caps = default_caps_for(backend);
    let eqn = g
        .eqns()
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
        .expect("the gather");
    let (plan, choice) = planned_with_choice(&g, eqn, backend).expect("the gather plans");
    let operands = value_ids(eqn).len();
    let plan = rebound(plan, |binding| {
        let mut reads = binding.reads().to_vec();
        reads[0] = Source::Operand(operands);
        Binding::new(reads, binding.write())
    });
    let mut planned = Planned { plan, choice };
    let shape = g.aval(eqn.out).shape.clone();
    let error = finalize(
        &mut planned,
        GraphTables::from(&g),
        eqn,
        &shape,
        numel(&shape),
        backend,
        g.aval(eqn.out).dtype,
        &caps,
    )
    .expect_err("an operand the equation lacks");
    assert!(
        matches!(
            error,
            PlanError::InvalidDispatchPlan(DispatchPlanError::OperandOutOfRange {
                dispatch: 0,
                operand,
                operands: have,
            }) if operand == operands && have == operands
        ),
        "got {error:?}"
    );
}

/// POOT-1113: the planner's own fit check picks streaming over a split whose scratch does not fit, so a
/// plan that still declares a scratch past `max_buffer_bytes` (another emitter, a future consumer) reaches
/// `validate_planned_resources` and is refused there with a typed `ScratchBytes`, never as an allocation
/// failure at an executor. The plan is the production split plan with its scratch declaration enlarged,
/// through the production `finalize`.
///
/// Mutation: delete the scratch loop of `validate_planned_resources`; the enlarged plan is accepted and
/// this row goes red.
#[test]
fn a_scratch_past_the_buffer_limit_is_refused_by_finalize() {
    use crate::kernel_mapping::finalize;
    use poot_target::BufferStorage;

    let backend = Backend::SpirvVulkan;
    let caps = default_caps_for(backend);
    let fixture = AttentionFixture::decode("split decode", 1, 2, 1, 1024, 64);
    let g = fixture.graph();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Attention(_)))
        .expect("an attention eqn");
    let (plan, choice) = planned_with_choice_caps(&g, eqn, backend, &caps).expect("it plans");
    let Plan::Dispatches(dispatches) = plan else {
        panic!("a dispatching plan");
    };
    let declared = dispatches.scratch()[0];
    let run = |decl: ScratchDecl| {
        let rebuilt = Plan::Dispatches(
            DispatchPlan::new(dispatches.clone().into_dispatches(), vec![decl])
                .expect("the rebuilt plan validates"),
        );
        let mut planned = Planned {
            plan: rebuilt,
            choice: choice.clone(),
        };
        let shape = g.aval(eqn.out).shape.clone();
        finalize(
            &mut planned,
            GraphTables::from(&g),
            eqn,
            &shape,
            numel(&shape),
            backend,
            g.aval(eqn.out).dtype,
            &caps,
        )
    };
    run(declared).expect("the planner's own scratch fits");
    let huge_elems = (caps.max_buffer_bytes / 4 + 1) as usize;
    let error = run(ScratchDecl::new(BufferStorage::f32(), huge_elems))
        .expect_err("a scratch past max_buffer_bytes");
    let PlanError::Refused(refusal) = error else {
        panic!("expected a typed refusal, got {error:?}");
    };
    assert_eq!(
        refusal.missing,
        Capability::ScratchBytes {
            needed: huge_elems as u64 * 4,
            max_buffer_bytes: caps.max_buffer_bytes,
        }
    );
}

/// Per-family dtype admission (POOT-1298): each family planner owns the storage dtypes it lowers, and the
/// router in `plan_gpu_eqn` only asks it. These rows drive the production planner entry with graphs whose
/// op is one family's and whose dtype that family refuses or admits. E4M3FN is IR-rejected as an arithmetic,
/// contraction or sampling operand (Card 529), so only the movement, cast and row-gather families can meet
/// it here; the other families' E4M3FN refusal is a defense behind `infer`.
mod family_dtype_admission {
    use super::*;

    fn caps_with_e4m3(storage: poot_target::E4m3Storage) -> DeviceCaps {
        DeviceCaps {
            e4m3_storage: storage,
            ..default_caps_for(Backend::SpirvVulkan)
        }
    }

    fn plan_first(g: &Graph, caps: &DeviceCaps) -> Result<Plan, PlanError> {
        planned_with_choice_caps(g, &g.eqns()[0], Backend::SpirvVulkan, caps).map(|(plan, _)| plan)
    }

    fn missing(result: Result<Plan, PlanError>, what: &str) -> Capability {
        refusal_of(result, what).missing
    }

    /// A scatter writes in its buffer's element type: F16 (the paged KV write, POOT-1395) is native on
    /// SPIR-V, BF16 is not, so the BF16 scatter is refused there by name. Mutation: drop the bf16-write
    /// refusal in `movement::admits_dtypes`; the BF16 scatter plans a row write SPIR-V has no type for and
    /// the BF16 assertion goes red.
    #[test]
    fn movement_refuses_a_bf16_scatter_and_a_native_less_e4m3_and_admits_a_half_slice() {
        let caps = default_caps_for(Backend::SpirvVulkan);
        let scatter = |dtype| {
            let b = Builder::for_tests();
            let base = b
                .constant("base", TensorType::new(vec![2, 5], dtype))
                .unwrap();
            let src = b
                .constant("src", TensorType::new(vec![1, 5], dtype))
                .unwrap();
            let slots = b.constant("s", TensorType::new(vec![1], GDt::I32)).unwrap();
            let scatter = b.scatter_update(base, src, slots).unwrap();
            b.finish(scatter).unwrap()
        };
        assert_eq!(
            missing(plan_first(&scatter(GDt::BF16), &caps), "BF16 ScatterUpdate"),
            Capability::DtypeLowering
        );
        assert!(
            plan_first(&scatter(GDt::F16), &caps).is_ok(),
            "an F16 scatter writes native 2-byte elements"
        );

        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![4, 6], GDt::F16))
            .unwrap();
        let sliced = b.slice(x, 0, 0, 2).unwrap();
        let g = b.finish(sliced).unwrap();
        assert!(
            plan_first(&g, &caps).is_ok(),
            "an F16 Slice moves 2-byte elements"
        );

        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![4, 8], GDt::E4M3FN))
            .unwrap();
        let sliced = b.slice(x, 0, 0, 2).unwrap();
        let g = b.finish(sliced).unwrap();
        let none = caps_with_e4m3(poot_target::E4m3Storage::None);
        assert_eq!(
            missing(plan_first(&g, &none), "E4M3FN Slice without native storage"),
            Capability::E4m3Storage
        );
        let packed = caps_with_e4m3(poot_target::E4m3Storage::PackedLanes);
        assert!(plan_first(&g, &packed).is_ok());
    }

    /// Mutation: drop the `collective ||` arm of the half refusal in `movement::admits_dtypes`; the F16
    /// all-reduce stops being refused by name here.
    #[test]
    fn movement_refuses_a_half_collective() {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![4, 8], GDt::F16))
            .unwrap();
        let reduced = b.all_reduce(x, poot_graph_ir::op::RedOp::Sum, 0).unwrap();
        let g = b.finish(reduced).unwrap();
        assert_eq!(
            missing(
                plan_first(&g, &default_caps_for(Backend::SpirvVulkan)),
                "F16 AllReduce"
            ),
            Capability::DtypeLowering
        );
    }

    /// Mutation: in `elementwise::admits_dtypes`, drop the `Fused(_)` pattern; the BF16 fused region plans a
    /// kernel over f32 storage and the refusal assertion goes red.
    #[test]
    fn elementwise_refuses_a_half_fused_region_and_admits_a_half_binary() {
        let caps = default_caps_for(Backend::SpirvVulkan);
        let b = Builder::for_tests();
        let slot = |name: &str| {
            let _ = name;
            b.slot(
                poot_graph_ir::Slot::Activation,
                TensorType::new(vec![4, 8], GDt::BF16),
            )
            .unwrap()
        };
        let (x, s, bias) = (slot("x"), slot("s"), slot("bias"));
        let scaled = b.binary(GBinOp::Mul, x, s).unwrap();
        let e = b.unary(GUnOp::Exp, scaled).unwrap();
        let out = b.binary(GBinOp::Add, e, bias).unwrap();
        let g = b.finish(out).unwrap();
        let plain = &g.eqns()[0];
        assert!(
            planned_with_choice_caps(&g, plain, Backend::SpirvVulkan, &caps).is_ok(),
            "a BF16 Binary is wired"
        );
        let fused = crate::passes::fuse(&g).unwrap();
        let region = fused
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Fused(_)))
            .expect("fuse fuses the chain");
        let refusal = refusal_of(
            planned_with_choice_caps(&fused, region, Backend::SpirvVulkan, &caps)
                .map(|(plan, _)| plan),
            "BF16 Fused region",
        );
        assert_eq!(refusal.missing, Capability::DtypeLowering);
    }

    /// Mutation: make `matmul::admits_dtypes` return `Err(Capability::DtypeLowering)` for every dtype; the
    /// BF16 matmul stops planning.
    #[test]
    fn matmul_admits_a_half_matmul() {
        let g = matmul_graph(16, 32, 16, GDt::BF16);
        assert!(try_matmul_plan(&g, Backend::SpirvVulkan).is_ok());
    }

    /// The indexed matmul's half refusal is stated twice: by `moe::admits_dtypes` and by
    /// `indexed_matmul_lowers` inside the planner body, which answers the same typed refusal, so no single
    /// mutation of the router arm turns this red; mutating `indexed_matmul_lowers` alone does not either.
    /// Both together do.
    #[test]
    fn moe_refuses_a_half_indexed_matmul() {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![4, 8], GDt::BF16))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![2, 8, 16], GDt::BF16))
            .unwrap();
        let idx = b.constant("i", TensorType::new(vec![4], GDt::I32)).unwrap();
        let out = b.indexed_matmul(x, w, idx).unwrap();
        let g = b.finish(out).unwrap();
        assert_eq!(
            g.aval(g.eqns()[0].out).dtype,
            GDt::BF16,
            "a half output over a half weight, which the weight lowering alone would admit as packed lanes"
        );
        assert_eq!(
            missing(
                plan_first(&g, &default_caps_for(Backend::SpirvVulkan)),
                "F16 IndexedMatMul"
            ),
            Capability::DtypeLowering
        );
    }

    /// Mutation: in `packed::admits_dtypes`, delete the `DenseRowGather` arm; the E4M3FN row gather then
    /// refuses on a device with no native E4M3FN storage. A dense contraction is F32 out (`infer`), so
    /// the packed family has no half-output case to build.
    #[test]
    fn packed_admits_the_e4m3_row_gather_without_native_storage() {
        let g = dense_row_gather_graph(8, 4, 3, GDt::E4M3FN);
        let gather = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::DenseRowGather { .. }))
            .expect("a dense row gather eqn");
        let none = caps_with_e4m3(poot_target::E4m3Storage::None);
        assert!(
            planned_with_choice_caps(&g, gather, Backend::SpirvVulkan, &none).is_ok(),
            "the E4M3FN row gather reads row-padded words on a device with no native storage"
        );
    }

    /// Mutation: make `attention::admits_dtypes` (via `StorageDtypes::refuse_unwired`) return `Ok(())`; the
    /// F16 attention plans an f32 kernel over 2-byte storage and the refusal assertion goes red.
    #[test]
    fn attention_refuses_a_half_output() {
        let b = Builder::for_tests();
        let slot = |name: &str, shape: Vec<usize>| {
            b.slot_named(
                poot_graph_ir::Slot::Activation,
                name,
                TensorType::new(shape, GDt::F16),
            )
            .unwrap()
        };
        let q = slot("q", vec![1, 2, 1, 16]);
        let k = slot("k", vec![1, 2, 8, 16]);
        let v = slot("v", vec![1, 2, 8, 16]);
        let mask = b
            .constant("mask", TensorType::new(vec![1, 1, 1, 8], GDt::F16))
            .unwrap();
        let mut plan = b.append_plan(0).unwrap();
        let out = plan
            .equation(
                OpKind::Attention(poot_graph_ir::op::AttentionAttrs::new(1, 0.25, None)),
                [q, k, v, mask]
                    .map(|operand| poot_graph_ir::Operand::Value(operand.id()))
                    .to_vec(),
            )
            .expect("stage the F16 attention");
        plan.declare_result(out).expect("declare the result");
        let mut prepared = b.preflight_append(plan).expect("preflight");
        let committed = b.commit_append(&mut prepared).expect("commit");
        let g = b.finish(committed).unwrap();
        assert_eq!(
            missing(
                plan_first(&g, &default_caps_for(Backend::SpirvVulkan)),
                "F16 Attention"
            ),
            Capability::DtypeLowering
        );
    }
}
