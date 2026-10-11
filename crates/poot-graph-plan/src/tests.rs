//! Unit tests for the planner: dispatch-predicate gating, cache-key stability, strided-view promotion,
//! real-config dispatch-count regressions.

use poot_graph_ir::{AttentionAttrs, Builder, ShapeError, Storage, TensorType, ValidationId};
use poot_target::{
    AmdArch, Backend, DECODE_GEMV_CHUNK_TRIGGER, DeviceCaps, Queried, TensorCoreSupport,
    WatchdogBudget,
};
use poot_tensor::DType as GDt;
use poot_test_util::device_caps::default_caps_for;

use super::*;
use crate::device_validation::ResourceRefusal;

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
fn dispatch_count_planned<V: ValidationChannel>(g: &Graph<V>, backend: Backend) -> usize {
    let views = compute_views(g, backend);
    g.eqns
        .iter()
        .filter(|e| !matches!(e.op, OpKind::Reshape { .. }) && !views.contains_key(&e.out))
        .count()
}

fn is_elementwise_2d<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
) -> bool {
    is_elementwise_2d_in(
        g.into(),
        backend,
        eqn,
        out_shape,
        out_numel,
        &default_caps_for(backend),
    )
}

#[test]
fn validation_packet_plan_matches_layout() {
    let b = Builder::new();
    let primary = b.constant("primary", TensorType::f32([1]));
    let first = b.constant("first", TensorType::f32([2]));
    let second = b.constant("second", TensorType::f32([3]));
    let graph = crate::test_support::finish_with_validations(
        b,
        primary,
        &[
            (ValidationId(2), "first", first),
            (ValidationId(5), "second", second),
        ],
    )
    .unwrap();

    let plan = plan_validation_packet(&graph).unwrap();
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

use crate::passes_without_target as optimize;

/// The graph `Model::trace` gives for one step of a real checkpoint's dimensions (zero weights: only
/// the structure is read), as an ordinary graph.
fn dense_graph(
    dense: poot_executor_parity::dense::Dense,
    phase: poot_models::model::Phase,
    tokens: usize,
    cap: usize,
    logits: poot_models::model::LogitRows,
) -> Graph {
    use poot_executor_parity::dense::{plain, step};
    plain(
        dense
            .zeroed_model(GDt::BF16)
            .model
            .trace(phase, step(1, tokens, cap, logits))
            .expect("the family traces this step"),
    )
}

/// Qwen2.5-0.5B's dimensions (biased projections, no Q/K norm).
fn qwen2_0_5b() -> poot_executor_parity::dense::Dense {
    use poot_executor_parity::dense::{Dense, Family};
    Dense::new(Family::Qwen2)
        .vocab(151936)
        .dims(896, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(32768)
}

/// Qwen3-0.6B's dimensions (no bias, per-head Q/K norm, explicit head dimension).
fn qwen3_0_6b() -> poot_executor_parity::dense::Dense {
    use poot_executor_parity::dense::{Dense, Family};
    Dense::new(Family::Qwen3)
        .vocab(151936)
        .dims(1024, 3072, 28)
        .heads(16, 8)
        .head_dim(128)
        .max_positions(40960)
}

/// One decode token of a real config over `cap` positions.
fn real_decode_graph(dense: poot_executor_parity::dense::Dense, cap: usize) -> Graph {
    dense_graph(
        dense,
        poot_models::model::Phase::Decode,
        1,
        cap,
        poot_models::model::LogitRows::Last,
    )
}

// Build a single bf16/f32 matmul graph `[1,m,k] @ [k,n]`; returns (graph, the matmul eqn index).
fn matmul_graph(m: usize, k: usize, n: usize, dt: GDt) -> Graph {
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::new(vec![1, m, k], dt));
    let w = bld.constant("w", TensorType::new(vec![k, n], dt));
    let mm = bld.matmul(a, w);
    bld.finish(mm)
}

/// Like [`matmul_graph`] but with an explicit leading batch dim: `[batch,m,k] @ [k,n]`. NVPTX WMMA
/// (`matmul_nvptx_tc_eligible`) allows batch dims, unlike the AMD arm, so this is the shape needed to
/// exercise the `tc` arm's batch factor (card 525).
fn batched_matmul_graph(batch: usize, m: usize, k: usize, n: usize, dt: GDt) -> Graph {
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::new(vec![batch, m, k], dt));
    let w = bld.constant("w", TensorType::new(vec![k, n], dt));
    let mm = bld.matmul(a, w);
    bld.finish(mm)
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
        backend,
        &HashMap::new(),
        caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .map(|Planned { plan, choice }| (plan, choice))
}

/// Whether a choice is the shipped kernel `kernel`.
fn is_imported(choice: &KernelChoice, kernel: ImportedKernel) -> bool {
    matches!(choice, KernelChoice::Imported { kernel: k, .. } if *k == kernel)
}

/// The kernel choice for one equation. Tests assert on this record, never on the plan's opaque key.
fn choice_of(g: &Graph, eqn: &Eqn, backend: Backend) -> KernelChoice {
    planned_with_choice(g, eqn, backend).unwrap().1
}

/// The kernel choice of the graph's one `MatMul`.
fn matmul_choice(g: &Graph, backend: Backend) -> KernelChoice {
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
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

/// Whether the choice generated a tensor-core matmul with this element type.
fn is_tensor_core(choice: &KernelChoice) -> bool {
    matches!(
        contraction(choice),
        Some(ContractionSpec::TensorCore { .. })
    )
}

#[test]
fn exact_i32_binary_plans_typed_integer_kernels_on_every_compute_backend() {
    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![2, 1], GDt::I32));
    let c = b.constant("c", TensorType::new(vec![1, 3], GDt::I32));
    let add = b.binary(GBinOp::Add, a, c);
    let geu = b.binary_scalar(GBinOp::GeU, add, Scalar::I32(i32::MIN));
    let remu = b.binary_scalar(GBinOp::RemU, geu, Scalar::I32(0x7fff_ffffu32 as i32));
    let g = b.finish(remu);

    for (eqn, expected_name) in [
        (&g.eqns[0], "Add"),
        (&g.eqns[1], "GeU"),
        (&g.eqns[2], "RemU"),
    ] {
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let (planned_plan, choice) = planned_with_choice(&g, eqn, backend).unwrap();
            let Dispatch { body, .. } = into_plain(planned_plan);
            // The request names the exact-I32 operation, not the f32 lane's.
            let named = match (&choice, expected_name) {
                (
                    KernelChoice::Generated(KernelRequest::Elementwise(ElementwiseSpec::Binary {
                        op: ValueBinary::Basic(BinOp::Add, Ty::I32),
                        ..
                    })),
                    "Add",
                ) => true,
                (
                    KernelChoice::Generated(KernelRequest::Elementwise(
                        ElementwiseSpec::ScalarI32 { op, .. },
                    )),
                    name,
                ) => matches!(
                    (op, name),
                    (I32Binary::GeU, "GeU") | (I32Binary::RemU, "RemU")
                ),
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
    let b = Builder::new();
    let ty = TensorType::new(vec![4], GDt::I32);
    let x = b.constant("x", ty.clone());
    let y = b.constant("y", ty.clone());
    let z = b.constant("z", ty);
    let masked = b.binary(GBinOp::And, x, y);
    let shifted = b.binary_scalar(GBinOp::Shl, masked, Scalar::I32(33));
    let counted = b.unary(GUnOp::Clz, shifted);
    let inverted = b.unary(GUnOp::Not, counted);
    let selected = b.select(inverted, y, z);
    let g = b.finish(selected);

    for eqn in &g.eqns {
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let (planned_plan, choice) = planned_with_choice(&g, eqn, backend).unwrap();
            let Dispatch { body, .. } = into_plain(planned_plan);
            // Every step runs over exact I32 storage: an I32 element type, the exact-I32 pointwise
            // forms, or the unsigned-literal kernels.
            let exact = match &choice {
                KernelChoice::Generated(KernelRequest::Elementwise(spec)) => match spec {
                    ElementwiseSpec::Unary { dt, .. } => *dt == Ty::I32,
                    ElementwiseSpec::ScalarI32 { .. } => true,
                    ElementwiseSpec::Binary { op, .. } => {
                        matches!(op, ValueBinary::Basic(_, Ty::I32))
                    }
                    _ => false,
                },
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

    let fused = crate::fuse(&g);
    fused.validate().unwrap();
    assert_eq!(fused.eqns.len(), 1);
    let eqn = &fused.eqns[0];
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
    let b = Builder::new();
    let ty = TensorType::new(vec![over_ceiling], GDt::I32);
    let cond = b.constant("cond", ty.clone());
    let if_true = b.constant("t", ty.clone());
    let if_false = b.constant("f", ty);
    let selected = b.select(cond, if_true, if_false);
    let g = b.finish(selected);
    let eqn = &g.eqns[0];
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
    let b = Builder::new();
    let cond = b.constant("cond", TensorType::new(vec![4], GDt::I32));
    let if_true = b.constant("t", TensorType::new(vec![4], GDt::F32));
    let if_false = b.constant("f", TensorType::new(vec![4], GDt::F32));
    let selected = b.select(cond, if_true, if_false);
    let g = b.finish(selected);
    let eqn = &g.eqns[0];
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

/// A float `Select` past the grid ceiling takes the same 2-D fold the exact-I32 `Select` does, standalone and
/// inside a fused float region (the K/V zero-select of a long batched cache is far past 65535 * 256 elements):
/// the request carries the fold and the body reads `GroupY`. Mutation: pass `PointwiseForm::Float`'s fold as
/// flat in `request::pointwise` (or drop `OpKind::Fused` from `is_elementwise_2d_in`); the body reads no
/// `GroupY` and this row goes red.
#[test]
fn float_select_past_the_grid_ceiling_emits_a_2d_fold() {
    let over_ceiling = 65_535 * 256 + 256;
    let b = Builder::new();
    let cond = b.constant("cond", TensorType::new(vec![over_ceiling], GDt::I32));
    let if_true = b.constant("t", TensorType::new(vec![over_ceiling], GDt::F32));
    let if_false = b.constant("f", TensorType::new(vec![over_ceiling], GDt::F32));
    let selected = b.select(cond, if_true, if_false);
    let standalone = b.finish(selected);
    // The same select followed by a pointwise neighbour fuses into one float region.
    let b = Builder::new();
    let cond = b.constant("cond", TensorType::new(vec![over_ceiling], GDt::I32));
    let if_true = b.constant("t", TensorType::new(vec![over_ceiling], GDt::F32));
    let if_false = b.constant("f", TensorType::new(vec![over_ceiling], GDt::F32));
    let negated = b.unary(GUnOp::Neg, b.select(cond, if_true, if_false));
    let fused = crate::fuse(&b.finish(negated));
    assert!(matches!(fused.eqns[..], [ref eqn] if matches!(eqn.op, OpKind::Fused(_))));
    for (what, g) in [("standalone", &standalone), ("fused", &fused)] {
        let eqn = g.eqns.last().unwrap();
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
    let b = Builder::new();
    let cond = b.constant("cond", TensorType::new(vec![4], GDt::I32));
    let if_true = b.constant("t", TensorType::new(vec![4], GDt::F16));
    let if_false = b.constant("f", TensorType::new(vec![4], GDt::F16));
    let selected = b.select(cond, if_true, if_false);
    let g = b.finish(selected);
    let eqn = &g.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "an F16 Select",
    );
    assert_eq!(refusal.missing, Capability::DtypeLowering);
}

#[test]
fn i32_is_its_own_kernel_element_type() {
    // POOT-725: no I32 value is computed through f32. Its element type is I32 wherever it is read.
    assert_eq!(fty_ty(GDt::I32), Ty::I32);

    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![2], GDt::I32));
    let y = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let g = b.finish(y);
    let eqn = &g.eqns[0];
    let planned_plan = plan_eqn(
        &g,
        eqn,
        Backend::Nvptx,
        &default_caps_for(Backend::Nvptx),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
        let moved = b.broadcast(x, vec![2]);
        let output = if arithmetic {
            b.binary_scalar(GBinOp::Add, moved, Scalar::I32(1))
        } else {
            let table = b.constant("table", TensorType::f32(vec![4, 1]));
            b.gather(table, 0, moved)
        };
        let g = b.finish(output);
        let eqn = g
            .eqns
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
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![2], GDt::I32));
        let y = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
        let output = if observed {
            y
        } else {
            let table = b.constant("table", TensorType::f32(vec![4, 1]));
            b.gather(table, 0, y)
        };
        let g = b.finish(output);
        let eqn = g
            .eqns
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
                KernelChoice::Generated(KernelRequest::Elementwise(
                    ElementwiseSpec::ScalarI32 { .. }
                ))
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
        let b = Builder::new();
        let table = if axis == 0 {
            b.constant("table", TensorType::f32(vec![4, 2]))
        } else {
            b.constant("table", TensorType::f32(vec![2, 4, 2]))
        };
        let index = b.constant("index", TensorType::new(vec![2], GDt::I32));
        let index = if computed {
            b.binary_scalar(GBinOp::Add, index, Scalar::I32(1))
        } else {
            b.broadcast(index, vec![2])
        };
        let gathered = b.gather(table, axis, index);
        let g = b.finish(gathered);
        let eqn = g
            .eqns
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
        let b = Builder::new();
        let table_shape = if axis == 0 { vec![4, 2] } else { vec![2, 4, 2] };
        let table = b.constant("table", TensorType::new(table_shape, GDt::I32));
        let index = b.constant("index", TensorType::new(vec![3], GDt::I32));
        let gathered = b.gather(table, axis, index);
        let g = b.finish(gathered);
        let eqn = g
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .unwrap();
        let body = match plan_eqn(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        let b = Builder::new();
        let table = b.constant("table", TensorType::new(table_shape, GDt::I32));
        let index = b.constant("index", TensorType::new(vec![3], GDt::I32));
        let gathered = b.gather(table, axis, index);
        let g = b.finish(gathered);
        let eqn = &g.eqns[0];
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2]));
    let output = b.binary_scalar(GBinOp::Sub, x, Scalar::F32(1.0));
    let mut g = b.finish(output);
    g.eqns[0].inputs.swap(0, 1);
    g.validate().expect("literal-first Binary is graph-valid");
    let refusal = refusal_of(
        plan_eqn(
            &g,
            &g.eqns[0],
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let b = Builder::new();
    let output = b.cast(b.iota(4), GDt::F32);
    let graph = b.finish(output);
    let iota = graph
        .eqns
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::Iota { .. }))
        .expect("the graph must hold an Iota equation");
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            iota,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "unfolded Iota",
    );
    assert_eq!(refusal.missing, Capability::UnfoldedIota);
}

#[test]
fn planner_structural_boundary_rejects_invalid_binding_table_first() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let mut graph = b.finish(output);
    graph.consts.push(ValueId::new(u32::MAX));

    assert!(matches!(
        validate_graph_execution(&graph),
        Err(PlanError::InvalidGraph(
            poot_graph_ir::GraphValidationError::ConstOutOfRange {
                value,
                value_count,
            }
        )) if value == ValueId::new(u32::MAX) && value_count == graph.values.len()
    ));
}

#[test]
fn i32_graph_inspection_is_total_for_a_malformed_graph() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let mut g = b.finish(output);
    g.eqns[0].inputs[0] = Operand::Value(g.values.next_id().unwrap());

    assert!(matches!(
        validate_graph_execution(&g),
        Err(PlanError::InvalidGraph(
            poot_graph_ir::GraphValidationError::EquationOperandNotDefined {
                equation: 0,
                value,
            }
        )) if Some(value) == g.values.next_id()
    ));
}

#[test]
fn exact_i32_planner_rejects_static_numel_overflow() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let mut g = b.finish(output);
    g.values[output.id()].aval.shape = vec![usize::MAX, 2];
    // card 525: plan_eqn now derives out_shape/out_numel from this same mutated aval itself (it can no
    // longer be told a different shape than the graph's own), so the overflow it must survive without
    // panicking is now computed inside plan_eqn, not passed in by the caller.
    let error = plan_eqn(
        &g,
        &g.eqns[0],
        Backend::Nvptx,
        &default_caps_for(Backend::Nvptx),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 8]));
    let y = b.constant("y", TensorType::f32(vec![4, 8]));
    let out = b.binary(GBinOp::Add, x, y);
    let g = b.finish(out);
    let eqn = &g.eqns[0];
    let planned_plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    planned_with_choice(g, eqn, backend)
}

fn try_matmul_plan(g: &Graph, backend: Backend) -> Result<Plan, PlanError> {
    try_matmul_plan_planned(g, backend).map(|(plan, _)| plan)
}

/// A biased matmul whose weight dtype differs from its activation's is not a contraction epilogue: the
/// fuse rule only forms one over a head whose operands and result share a dtype (a mixed pairing is the
/// tensor-core and decode-GEMV arms' contract, and `widen_mismatched_matmul_dtypes` casts the rest), so
/// the graph keeps its matmul and its add. Mutation: drop the dtype check in `epilogue_head`; the mixed
/// matmul fuses and this row goes red.
#[test]
fn a_biased_matmul_with_mismatched_operand_dtypes_is_not_fused_into_an_epilogue() {
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::new(vec![1, 1, 32], GDt::F32));
    let w = bld.constant("w", TensorType::new(vec![32, 16], GDt::BF16));
    let bias = bld.constant("bias", TensorType::new(vec![16], GDt::F32));
    let mm = poot_graph_ir::ops::linear(&bld, a, w, Some(bias));
    let g = crate::passes::fuse_contraction_epilogues(&bld.finish(mm), |_, _| true);
    assert!(
        g.eqns
            .iter()
            .all(|e| !matches!(e.op, OpKind::ContractionEpilogue { .. })),
        "a mixed-dtype matmul must keep its add: {:?}",
        g.eqns
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
            let bld = Builder::new();
            let x = bld.constant("x", TensorType::new(vec![1, 1, 32], GDt::F32));
            let w = bld.constant("w", TensorType::new(vec![16, 32], weight));
            let bias = bld.constant("bias", TensorType::new(vec![16], GDt::F32));
            let projection = bld.matmul(x, bld.transpose(w, vec![1, 0]));
            let out = bld.binary(poot_graph_ir::op::BinOp::Add, projection, bias);
            let g = bld.finish(out);
            let program = compile(
                &g,
                &Target {
                    backend,
                    caps: default_caps_for(backend),
                },
                &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
            )
            .unwrap_or_else(|e| panic!("{backend:?} {weight:?}: {e}"));
            let ops: Vec<&OpKind> = program.graph().eqns.iter().map(|e| &e.op).collect();
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
    let matmul_err = OpKind::MatMul
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
        let builder = Builder::new();
        let base = builder.constant("base", TensorType::new(vec![4, 2, 5], GDt::E4M3FN));
        let src = builder.constant("src", TensorType::new(vec![source_rows, 2, 5], GDt::E4M3FN));
        let inverse = builder.constant("inverse", TensorType::new(vec![4], GDt::I32));
        let output = builder.scatter_update(base, src, inverse);
        builder.finish(output)
    };
    let graph = build(3);
    let eqn = &graph.eqns[0];
    let (planned_plan, choice) = planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap();
    let Dispatch {
        body,
        key,
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(body.param_count, 4);
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
    assert_eq!(grid, [16, 1, 1]);

    let other = build(2);
    let other_eqn = &other.eqns[0];
    let planned_plan = plan_eqn(
        &other,
        other_eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
            &poot_test_util::graph_fixtures::roomy_body_limits()
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN ScatterUpdate on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, OpKind::ScatterUpdate);
    assert_eq!(refusal.target, backend);

    let builder = Builder::new();
    let base = builder.constant("base_rank1", TensorType::new(vec![5], GDt::E4M3FN));
    let src = builder.constant("src_rank1", TensorType::new(vec![3], GDt::E4M3FN));
    let inverse = builder.constant("inverse_rank1", TensorType::new(vec![5], GDt::I32));
    let output = builder.scatter_update(base, src, inverse);
    let rank1 = builder.finish(output);
    let rank1_eqn = &rank1.eqns[0];
    let (planned_plan, choice) =
        planned_with_choice(&rank1, rank1_eqn, Backend::SpirvVulkan).unwrap();
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
                op: kg::MovementOp::ScatterRows { src_shape, window: None, .. },
                ..
            })) if out_shape == &[5] && src_shape == &[3]
        ),
        "{choice:?}"
    );
    assert_eq!(
        grid,
        [2, 1, 1],
        "rank-1 POOL=5 output is two physical packed words, not five logical bytes"
    );
}

#[test]
fn e4m3fn_dynamic_update_slice_plans_static_and_runtime_exact_packed_outputs() {
    let builder = Builder::new();
    let operand = builder.constant("operand", TensorType::new(vec![2, 4, 5], GDt::E4M3FN));
    let update = builder.constant("update", TensorType::new(vec![2, 2, 5], GDt::E4M3FN));
    let output = builder.dynamic_update_slice(operand, update, 1, 1);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
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

    let builder = Builder::new();
    let operand = builder.constant("operand", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let update = builder.constant("update", TensorType::new(vec![2, 2], GDt::E4M3FN));
    let index = builder.constant("index", TensorType::new(vec![], GDt::I32));
    let output = builder.dynamic_update_slice_dyn(operand, update, index, 1);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
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
            &poot_test_util::graph_fixtures::roomy_body_limits()
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN DynamicUpdateSlice on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, backend);
}

#[test]
fn e4m3fn_scatter_update_rejects_empty_and_mixed_storage_by_name() {
    for (pool, source_rows) in [(0, 1), (2, 0)] {
        let builder = Builder::new();
        let base = builder.constant("base", TensorType::new(vec![pool, 5], GDt::E4M3FN));
        let src = builder.constant("src", TensorType::new(vec![source_rows, 5], GDt::E4M3FN));
        let inverse = builder.constant("inverse", TensorType::new(vec![pool], GDt::I32));
        let output = builder.scatter_update(base, src, inverse);
        let graph = builder.finish(output);
        let eqn = &graph.eqns[0];
        let refusal = refusal_of(
            plan_eqn(
                &graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &poot_test_util::graph_fixtures::roomy_body_limits(),
            ),
            "empty E4M3FN ScatterUpdate",
        );
        assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
        assert_eq!(refusal.op, OpKind::ScatterUpdate);
    }

    let builder = Builder::new();
    let base = builder.constant("base", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let src = builder.constant("src", TensorType::f32(vec![1, 5]));
    let inverse = builder.constant("inverse", TensorType::new(vec![2], GDt::I32));
    let output = builder.scatter_update(base, src, inverse);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN ScatterUpdate over an f32 source",
    );
    assert_eq!(refusal.missing, Capability::DtypeLowering);
    assert_eq!(
        refusal.dtypes.operands,
        [GDt::E4M3FN, GDt::F32, GDt::I32],
        "the refusal must carry the mixed operand dtypes"
    );

    // Card 529: an E4M3FN `inv` no longer reaches the planner - `infer` rejects it for every
    // `ScatterUpdate`, so this is an IR-level `ShapeError` now, not a planner refusal.
    let err = OpKind::ScatterUpdate
        .infer(&[
            TensorType::new(vec![2, 5], GDt::E4M3FN),
            TensorType::new(vec![1, 5], GDt::E4M3FN),
            TensorType::new(vec![2], GDt::E4M3FN),
        ])
        .expect_err("E4M3FN ScatterUpdate with an E4M3FN inverse must be rejected by infer");
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
        let builder = Builder::new();
        let operand = builder.constant("operand", TensorType::new(operand_shape, GDt::E4M3FN));
        let update = builder.constant("update", TensorType::new(update_shape, GDt::E4M3FN));
        let output = if let Some(index) = index {
            builder.dynamic_update_slice(operand, update, index, axis)
        } else {
            let index = builder.constant("index", TensorType::new(vec![], GDt::I32));
            builder.dynamic_update_slice_dyn(operand, update, index, axis)
        };
        let graph = builder.finish(output);
        let eqn = &graph.eqns[0];
        let planned_plan = plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let builder = Builder::new();
    let operand = builder.constant("operand", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let update = builder.constant("update", TensorType::new(vec![2, 2], GDt::E4M3FN));
    let output = builder.dynamic_update_slice(operand, update, 4, 1);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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

    let builder = Builder::new();
    let operand = builder.constant("operand", TensorType::new(vec![0, 5], GDt::E4M3FN));
    let update = builder.constant("update", TensorType::new(vec![0, 5], GDt::E4M3FN));
    let output = builder.dynamic_update_slice(operand, update, 0, 0);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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

    let builder = Builder::new();
    let operand = builder.constant("operand", TensorType::f32(vec![4, 5]));
    let update = builder.constant("update", TensorType::f32(vec![2, 5]));
    let index = builder.constant("index", TensorType::scalar(GDt::I32));
    let output = builder.dynamic_update_slice_dyn(operand, update, index, 0);
    let graph = builder.finish(output);
    let eqn = graph
        .eqns
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
    let builder = Builder::new();
    let p0 = builder.constant("p0", TensorType::new(vec![2, 2, 3], GDt::E4M3FN));
    let p1 = builder.constant("p1", TensorType::new(vec![2, 1, 3], GDt::E4M3FN));
    let p2 = builder.constant("p2", TensorType::new(vec![2, 3, 3], GDt::E4M3FN));
    let output = builder.concat(1, &[p0, p1, p2]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
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
            &poot_test_util::graph_fixtures::roomy_body_limits()
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN Concat on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, backend);

    let builder = Builder::new();
    let a = builder.constant("a", TensorType::new(vec![2, 3], GDt::E4M3FN));
    let b = builder.constant("b", TensorType::new(vec![2, 2], GDt::E4M3FN));
    let output = builder.concat(1, &[a, b]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let shape = graph.aval(eqn.out).shape.clone();
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap();
    let Dispatch {
        launch: Launch { grid, .. },
        ..
    } = into_plain(planned_plan);
    assert_eq!(shape, [2, 5]);
    assert_eq!(grid, [4, 1, 1]);

    let builder = Builder::new();
    let only = builder.constant("only", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let output = builder.concat(1, &[only]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == only.id()
       ));
}

#[test]
fn e4m3fn_concat_rejects_zero_element_and_mixed_storage_by_name() {
    let builder = Builder::new();
    let empty = builder.constant("empty", TensorType::new(vec![0, 5], GDt::E4M3FN));
    let full = builder.constant("full", TensorType::new(vec![1, 5], GDt::E4M3FN));
    let output = builder.concat(0, &[empty, full]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "zero-element E4M3FN Concat",
    );
    assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    assert_eq!(refusal.op, eqn.op);

    let builder = Builder::new();
    let raw = builder.constant("raw", TensorType::new(vec![1, 5], GDt::E4M3FN));
    let dense = builder.constant("dense", TensorType::new(vec![1, 5], GDt::F32));
    let output = builder.concat(0, &[raw, dense]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN Concat over a dense operand",
    );
    assert_eq!(refusal.missing, Capability::DtypeLowering);
    assert_eq!(refusal.dtypes.operands, [GDt::E4M3FN, GDt::F32]);
}

#[test]
fn e4m3fn_broadcast_materializes_expansions_and_aliases_only_identity_on_wgpu() {
    for (input_shape, output_shape, expected_words) in [
        (vec![5], vec![3, 5], 6),
        (vec![2, 1, 5], vec![2, 3, 5], 12),
        (vec![1, 3], vec![2, 1, 3], 2),
    ] {
        let builder = Builder::new();
        let input = builder.constant("raw", TensorType::new(input_shape.clone(), GDt::E4M3FN));
        let output = builder.broadcast(input, output_shape.clone());
        let graph = builder.finish(output);
        let eqn = &graph.eqns[0];
        let shape = graph.aval(eqn.out).shape.clone();
        assert_eq!(shape, output_shape);
        let planned_plan = plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
                &poot_test_util::graph_fixtures::roomy_body_limits()
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
                &poot_test_util::graph_fixtures::roomy_body_limits(),
            ),
            "E4M3FN Broadcast on ROCm",
        );
        assert_eq!(refusal.missing, Capability::E4m3Storage);
        assert_eq!(refusal.op, eqn.op);
        assert_eq!(refusal.target, backend);
    }

    let builder = Builder::new();
    let input = builder.constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let output = builder.broadcast(input, vec![2, 5]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == input.id()
       ));
}

#[test]
fn e4m3fn_empty_output_broadcast_is_rejected_by_name() {
    let builder = Builder::new();
    let input = builder.constant("raw", TensorType::new(vec![1, 5], GDt::E4M3FN));
    let output = builder.broadcast(input, vec![0, 5]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN Broadcast to an empty output",
    );
    assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
    assert_eq!(refusal.op, eqn.op);
}

#[test]
fn e4m3fn_cast_and_reshape_plan_only_for_spirv_wgpu_value_executor() {
    let decode = {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![2, 5], GDt::E4M3FN));
        let y = b.cast(x, GDt::F32);
        b.finish(y)
    };
    let encode = {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![2, 5], GDt::F32));
        let y = b.cast(x, GDt::E4M3FN);
        b.finish(y)
    };
    let repack = {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![2, 5], GDt::E4M3FN));
        let y = b.reshape(x, vec![5, 2]);
        b.finish(y)
    };

    for graph in [&decode, &encode, &repack] {
        let eqn = &graph.eqns[0];
        let shape = graph.aval(eqn.out).shape.clone();
        let planned_plan = plan_eqn(
            graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
                &poot_test_util::graph_fixtures::roomy_body_limits()
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
                &poot_test_util::graph_fixtures::roomy_body_limits(),
            ),
            "E4M3FN Cast and Reshape on ROCm",
        );
        assert_eq!(refusal.missing, Capability::E4m3Storage);
        assert_eq!(refusal.target, backend);
    }
}

#[test]
fn e4m3fn_materializing_cast_and_reshape_reject_zero_element_outputs() {
    let encode = {
        let builder = Builder::new();
        let input = builder.constant("empty", TensorType::new(vec![0, 5], GDt::F32));
        let output = builder.cast(input, GDt::E4M3FN);
        builder.finish(output)
    };
    let decode = {
        let builder = Builder::new();
        let input = builder.constant("empty", TensorType::new(vec![0, 5], GDt::E4M3FN));
        let output = builder.cast(input, GDt::F32);
        builder.finish(output)
    };
    let repack = {
        let builder = Builder::new();
        let input = builder.constant("empty", TensorType::new(vec![0, 5], GDt::E4M3FN));
        let output = builder.reshape(input, vec![0, 7]);
        builder.finish(output)
    };

    for graph in [&encode, &decode, &repack] {
        let eqn = &graph.eqns[0];
        let refusal = refusal_of(
            plan_eqn(
                graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        let builder = Builder::new();
        let input = builder.constant("large", TensorType::new(vec![rows, 1], GDt::F32));
        let output = builder.cast(input, GDt::E4M3FN);
        builder.finish(output)
    };
    let decode = {
        let builder = Builder::new();
        let input = builder.constant("large", TensorType::new(vec![rows, 1], GDt::E4M3FN));
        let output = builder.cast(input, GDt::F32);
        builder.finish(output)
    };

    for graph in [&encode, &decode] {
        let eqn = &graph.eqns[0];
        let refusal = refusal_of(
            plan_eqn(
                graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::f32(vec![out_numel]));
    let b = bld.constant("b", TensorType::f32(vec![out_numel]));
    let y = bld.binary(GBinOp::Add, a, b);
    let g = bld.finish(y);
    let eqn = &g.eqns[0];
    let plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &small_caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::f32(vec![out_numel]));
    let b = bld.constant("b", TensorType::f32(vec![out_numel]));
    let y = bld.binary(GBinOp::Add, a, b);
    let g = bld.finish(y);
    let eqn = &g.eqns[0];
    let plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &small_caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    // (`poot-runtime`'s `workgroup_count`), never a floor division that would silently truncate.
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
    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let same_dtype = builder.cast(x, GDt::E4M3FN);
    let graph = builder.finish(same_dtype);
    let eqn = &graph.eqns[0];

    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::Nvptx,
       &default_caps_for(Backend::Nvptx),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            backend,
            &default_caps_for(backend),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "same-dtype E4M3FN Cast on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.target, backend);
}

#[test]
fn e4m3fn_transpose_materializes_nonidentity_and_aliases_only_identity_on_wgpu() {
    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let transposed = builder.transpose(x, vec![1, 0]);
    let graph = builder.finish(transposed);
    let eqn = &graph.eqns[0];
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
            &poot_test_util::graph_fixtures::roomy_body_limits()
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN Transpose on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, backend);

    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let identity = builder.transpose(x, vec![0, 1]);
    let graph = builder.finish(identity);
    let eqn = &graph.eqns[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));

    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 1, 5], GDt::E4M3FN));
    let moved_unit_axis = builder.transpose(x, vec![1, 0, 2]);
    let graph = builder.finish(moved_unit_axis);
    let eqn = &graph.eqns[0];
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap()
    ));
}

#[test]
fn e4m3fn_zero_element_transpose_is_rejected_by_name() {
    for perm in [vec![1, 0], vec![0, 1]] {
        let builder = Builder::new();
        let x = builder.constant("raw", TensorType::new(vec![0, 5], GDt::E4M3FN));
        let transposed = builder.transpose(x, perm);
        let graph = builder.finish(transposed);
        let eqn = &graph.eqns[0];
        let refusal = refusal_of(
            plan_eqn(
                &graph,
                eqn,
                Backend::SpirvVulkan,
                &default_caps_for(Backend::SpirvVulkan),
                &poot_test_util::graph_fixtures::roomy_body_limits(),
            ),
            "zero-element E4M3FN Transpose",
        );
        assert_eq!(refusal.missing, Capability::ZeroElementE4m3);
        assert_eq!(refusal.op, eqn.op);
    }
}

#[test]
fn e4m3fn_slice_materializes_proper_ranges_and_aliases_only_full_range_on_wgpu() {
    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let sliced = builder.slice(x, 1, 1, 4);
    let graph = builder.finish(sliced);
    let eqn = &graph.eqns[0];
    let shape = graph.aval(eqn.out).shape.clone();
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
            &poot_test_util::graph_fixtures::roomy_body_limits()
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN Slice on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, backend);

    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let identity = builder.slice(x, 1, 0, 5);
    let graph = builder.finish(identity);
    let eqn = &graph.eqns[0];
    assert!(matches!(
           plan_eqn(&graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id()
       ));

    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![3, 5], GDt::E4M3FN));
    let contiguous_rows = builder.slice(x, 0, 1, 3);
    let graph = builder.finish(contiguous_rows);
    let eqn = &graph.eqns[0];
    let planned_plan = plan_eqn(
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let empty = builder.slice(x, 1, 2, 2);
    let graph = builder.finish(empty);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        let builder = Builder::new();
        let data = builder.constant("raw", TensorType::new(data_shape, GDt::E4M3FN));
        let index = builder.constant("index", TensorType::new(index_shape, GDt::I32));
        let gathered = builder.gather(data, axis, index);
        let graph = builder.finish(gathered);
        let eqn = &graph.eqns[0];
        let shape = graph.aval(eqn.out).shape.clone();
        assert_eq!(shape, expected_shape);
        let planned_plan = plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap();
        let Dispatch {
            launch: Launch { grid, .. },
            ..
        } = into_plain(planned_plan);
        assert_eq!(grid, [expected_words, 1, 1]);
    }

    let builder = Builder::new();
    let data = builder.constant("raw", TensorType::new(vec![4, 5], GDt::E4M3FN));
    let index = builder.constant("index", TensorType::new(vec![4], GDt::I32));
    let gathered = builder.gather(data, 0, index);
    let graph = builder.finish(gathered);
    let eqn = &graph.eqns[0];
    assert!(is_plain(
        &plan_eqn(
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN Gather on ROCm",
    );
    assert_eq!(refusal.missing, Capability::E4m3Storage);
    assert_eq!(refusal.op, eqn.op);
    assert_eq!(refusal.target, backend);
}

#[test]
fn e4m3fn_gather_rejects_empty_output_and_non_i32_indices_by_name() {
    let builder = Builder::new();
    let data = builder.constant("raw", TensorType::new(vec![4, 5], GDt::E4M3FN));
    let index = builder.constant("index", TensorType::new(vec![0], GDt::I32));
    let gathered = builder.gather(data, 0, index);
    let graph = builder.finish(gathered);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn(
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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

#[test]
fn e4m3fn_zero_eqn_identity_graph_is_rejected_before_executor_binding() {
    let constant_graph = {
        let b = Builder::new();
        let x = b.constant("raw", TensorType::new(vec![3, 5], GDt::E4M3FN));
        b.finish(x)
    };
    let slot_graph = {
        let b = Builder::new();
        let x = b.slot(
            poot_graph_ir::Slot::TokenEmbed,
            TensorType::new(vec![3, 5], GDt::E4M3FN),
        );
        b.finish(x)
    };

    for graph in [&constant_graph, &slot_graph] {
        assert!(
            graph.eqns.is_empty(),
            "regression requires an identity graph"
        );
        let err = validate_graph_execution(graph).unwrap_err();
        assert!(
            matches!(
                err,
                PlanError::UnrepresentableValue {
                    value,
                    dtype: GDt::E4M3FN,
                    gap: StorageGap::E4m3RawBytes,
                } if value == graph.output
            ),
            "{err:?}"
        );
    }
}

#[test]
fn tensorcore_selected_only_for_nvptx_bf16_aligned() {
    // NVPTX + bf16 + all dims 16-aligned -> tensor cores (the `tc:` tag).
    let g = matmul_graph(16, 32, 16, GDt::BF16);
    assert!(is_tensor_core(&matmul_choice(&g, Backend::Nvptx)));
    // same graph on SPIR-V -> the serial kernel (WMMA is NVPTX-only).
    assert!(!is_tensor_core(&matmul_choice(&g, Backend::SpirvVulkan)));
    // f32 on NVPTX -> serial (tensor cores are the bf16 path).
    let gf = matmul_graph(16, 32, 16, GDt::F32);
    assert!(!is_tensor_core(&matmul_choice(&gf, Backend::Nvptx)));
    // bf16 but a non-16-aligned N -> serial fallback.
    let gn = matmul_graph(16, 32, 24, GDt::BF16);
    assert!(!is_tensor_core(&matmul_choice(&gn, Backend::Nvptx)));
    // bf16 but a non-16-aligned K -> serial fallback.
    let gk = matmul_graph(16, 24, 16, GDt::BF16);
    assert!(!is_tensor_core(&matmul_choice(&gk, Backend::Nvptx)));
}

/// Build a 16-aligned matmul `[1,16,32] @ [32,16]` with bf16/f16 operands and an explicit output dtype,
/// to exercise the mixed-precision (bf16-in/f32-out) case that reaches the AMD WMMA path. `MatMul.infer`
/// sets the output dtype to the operand dtype, so the mixed case overrides the output aval (white-box,
/// plan-only test).
fn mixed_matmul_graph(op_dt: GDt, out_dt: GDt) -> Graph {
    mixed_matmul_graph_shape(16, 32, 16, op_dt, out_dt)
}

/// Like [`mixed_matmul_graph`] but with an explicit M/K/N shape.
fn mixed_matmul_graph_shape(m: usize, k: usize, n: usize, op_dt: GDt, out_dt: GDt) -> Graph {
    let mut g = matmul_graph(m, k, n, op_dt);
    let out = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn")
        .out;
    g.values[out].aval.dtype = out_dt;
    g
}

/// spec 130 FR-004 / card 145: what reaches the AMD WMMA tensor-core path (the `matmul:tc:` tag) on
/// AmdGcn (RDNA3 gfx1151), and the spec-135 F16 boundary.
///
/// The AMD WMMA gate is on the operand dtype (bf16), but the tensor-core kernel also needs an F32 output:
/// `matmul_tensorcore` writes f32 via plain `WmmaStore`, whereas a bf16 output needs `WmmaStoreLds`
/// (f32->bf16 LDS narrowing), which is not wired for AMDGPU. So a pure bf16->bf16 matmul falls to the
/// serial `matmul_batched_dt`; only a mixed bf16-in/f32-out matmul dispatches AMD WMMA. F16 operands never
/// take this path (the WMMA fragment loads hardcode the bf16 bit layout and would reinterpret f16 bits).
/// CPU-safe: inspects the selected plan key only.
#[test]
fn amd_wmma_reached_only_by_mixed_bf16_in_f32_out() {
    let amd = Backend::AmdGcn(AmdArch::gfx1151());
    // Mixed bf16 operands -> f32 output: AMD WMMA tensor cores.
    let gmix = mixed_matmul_graph(GDt::BF16, GDt::F32);
    assert!(
        is_tensor_core(&matmul_choice(&gmix, amd)),
        "mixed bf16-in/f32-out 16-aligned matmul on gfx1151 should select WMMA"
    );
    // Pure bf16 -> bf16: serial fallback (WmmaStoreLds narrowing not wired for AMDGPU).
    let gbf = matmul_graph(16, 32, 16, GDt::BF16);
    assert!(
        !is_tensor_core(&matmul_choice(&gbf, amd)),
        "pure bf16->bf16 on AMD takes the serial fallback, not WMMA"
    );
    // F16 operands, even with an f32 output, must not route through the bf16 WMMA kernel. There is no
    // serial route for the mix either (the fallback's single element type would read the 2-byte f16
    // operands as f32 words), so the planner rejects the pair outright.
    let gf16_mix = mixed_matmul_graph(GDt::F16, GDt::F32);
    assert!(
        matches!(
            try_matmul_plan(&gf16_mix, amd),
            Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering
        ),
        "f16 operands must never reach the bf16-hardcoded WMMA kernel (spec 135)"
    );
    // Pure f16 -> the dtype-generic serial fallback.
    let gf16 = matmul_graph(16, 32, 16, GDt::F16);
    let f16_choice = matmul_choice(&gf16, amd);
    assert_eq!(
        serial_matmul_dtype(&f16_choice),
        Some(&Ty::F16),
        "f16 matmul should take the dtype-generic serial fallback: {f16_choice:?}"
    );
    // f32 on AmdGcn -> serial (tensor cores are the bf16 path).
    let gf32 = matmul_graph(16, 32, 16, GDt::F32);
    assert!(!is_tensor_core(&matmul_choice(&gf32, amd)));
    // Mixed bf16-in/f32-out but a non-16-aligned K: WMMA needs aligned M/N/K, and the serial fallback
    // cannot serve a mixed matmul (its single element type would read the bf16 operands as f32 words),
    // so the pair is rejected.
    let mut gk = matmul_graph(16, 24, 16, GDt::BF16);
    let gk_out = gk
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap()
        .out;
    gk.values[gk_out].aval.dtype = GDt::F32;
    assert!(matches!(
        try_matmul_plan(&gk, amd),
        Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering
    ));
}

#[test]
fn attention_rank4_matmul_routes_through_batched_tiled_gemm() {
    // A prefill-attention-shaped matmul [B,H,M,K] @ [B,H,K,N] -> [B,H,M,N] (M>1) under the 2^15 workgroup
    // cap routes through the batched tiled GEMM (b_count = B*H), not the naive batched kernel.
    let bld = Builder::new();
    let q = bld.constant("q", TensorType::new(vec![1, 14, 64, 64], GDt::F32));
    let kt = bld.constant("kt", TensorType::new(vec![1, 14, 64, 64], GDt::F32));
    let mm = bld.matmul(q, kt);
    let g = bld.finish(mm);
    let choice = matmul_choice(&g, Backend::SpirvVulkan);
    assert!(
        matches!(
            choice,
            KernelChoice::Imported {
                kernel: ImportedKernel::TiledGemmBatched,
                ..
            }
        ),
        "attention matmul should take the tiled path: {choice:?}"
    );
    // Above the 2^15 workgroup cap (very long context) it falls back to the naive batched kernel (card 095).
    let bld2 = Builder::new();
    let q2 = bld2.constant("q", TensorType::new(vec![1, 14, 2048, 2048], GDt::F32));
    let k2 = bld2.constant("k", TensorType::new(vec![1, 14, 2048, 2048], GDt::F32));
    let mm2 = bld2.matmul(q2, k2);
    let g2 = bld2.finish(mm2);
    let choice2 = matmul_choice(&g2, Backend::SpirvVulkan);
    assert!(
        serial_matmul_dtype(&choice2).is_some(),
        "over-cap attention matmul should fall back to naive: {choice2:?}"
    );
}

#[test]
fn decode_attn_scores_v_matmul_lowers_to_the_lds_gemv_kernel() {
    // Card 143 Lever 1b: decode `scores @ V` (M=1, Hq rows already GQA-repeat_kv-expanded on both
    // operands, cap=300 not a multiple of GEMV_WIDTH=128) takes the LDS-parallel attn-scores-v GEMV on
    // every backend that lowers WorkgroupLocalWrite/Barrier, like is_decode_gemv's M=1 gate.
    let bld = Builder::new();
    let probs = bld.constant("probs", TensorType::new(vec![1, 8, 1, 300], GDt::F32));
    let v = bld.constant("v", TensorType::new(vec![1, 8, 300, 64], GDt::F32));
    let mm = bld.matmul(probs, v);
    let g = bld.finish(mm);
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
    let bldq = Builder::new();
    let q = bldq.constant("q", TensorType::new(vec![1, 8, 1, 64], GDt::F32));
    let k = bldq.constant("k", TensorType::new(vec![1, 8, 300, 64], GDt::F32));
    let kt = bldq.transpose(k, vec![0, 1, 3, 2]);
    let scores = bldq.matmul(q, kt);
    let gq = bldq.finish(scores);
    let choice_q = matmul_choice(&gq, Backend::SpirvVulkan);
    assert!(
        !matches!(
            contraction(&choice_q),
            Some(ContractionSpec::AttnScoresVGemv { .. })
        ),
        "Q@K^T must not take the scores@V attn GEMV path: got {choice_q:?}"
    );

    // M>1 (prefill) stays off this M=1-only path (is_batched_tiled_gemm handles it).
    let bldp = Builder::new();
    let probs_p = bldp.constant("probs", TensorType::new(vec![1, 8, 4, 300], GDt::F32));
    let v_p = bldp.constant("v", TensorType::new(vec![1, 8, 300, 64], GDt::F32));
    let mm_p = bldp.matmul(probs_p, v_p);
    let gp = bldp.finish(mm_p);
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
    //   - `outer = k_col @ delta` [1,32,128,1] @ [1,32,1,128] -> [1,32,128,128] (M=128) never takes
    //     `is_batched_tiled_gemm` on AmdGcn: that gate is `SpirvVulkan | Nvptx` only (the imported
    //     batched-tiled body miscompiles on AMDGCN, LDS zeroinitializer in addrspace(3); see the gate's
    //     doc). It falls through to the naive `matmul_batched_dt_grid` one-thread-per-output kernel, a
    //     documented backend gap.
    let amd = Backend::AmdGcn(AmdArch::gfx1151());
    const H: usize = 32;
    const D: usize = 128;

    // kv = k[1,H,1,D] @ s_decayed[1,H,D,D] -> [1,H,1,D]
    let bld_kv = Builder::new();
    let k = bld_kv.constant("k", TensorType::new(vec![1, H, 1, D], GDt::F32));
    let s_decayed = bld_kv.constant("s_decayed", TensorType::new(vec![1, H, D, D], GDt::F32));
    let kv = bld_kv.matmul(k, s_decayed);
    let g_kv = bld_kv.finish(kv);
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
    let bld_o = Builder::new();
    let q_scaled = bld_o.constant("q_scaled", TensorType::new(vec![1, H, 1, D], GDt::F32));
    let s_out = bld_o.constant("s_out", TensorType::new(vec![1, H, D, D], GDt::F32));
    let o = bld_o.matmul(q_scaled, s_out);
    let g_o = bld_o.finish(o);
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
    let bld_outer = Builder::new();
    let k_col = bld_outer.constant("k_col", TensorType::new(vec![1, H, D, 1], GDt::F32));
    let delta = bld_outer.constant("delta", TensorType::new(vec![1, H, 1, D], GDt::F32));
    let outer = bld_outer.matmul(k_col, delta);
    let g_outer = bld_outer.finish(outer);
    let choice_outer = matmul_choice(&g_outer, amd);
    assert!(
        matches!(
            contraction(&choice_outer),
            Some(ContractionSpec::Serial { dt: Ty::F32, epilogue: None, shapes, .. })
                if shapes.out_shape == [1, 32, 128, 128]
                    && shapes.a_shape == [1, 32, 128, 1]
                    && shapes.b_shape == [1, 32, 1, 128]
        ),
        "GDN outer=k_col@delta should fall through to the naive batched matmul on AmdGcn \
         (is_batched_tiled_gemm is SpirvVulkan|Nvptx-only): got {choice_outer:?}"
    );
}

// --- spec 130 / FR-001 / FR-004 / FR-005 / SC-004 ---

#[test]
fn tensorcore_amdgcn_selected_for_bf16_aligned_rdna3() {
    // Backend::AmdGcn(gfx1151) + bf16 + 16-aligned unbatched: plan_eqn routes through the amd_tc arm
    // (FR-003: selection in plan_eqn, not the executor). The WMMA Body (matmul_tensorcore) is a
    // placeholder pending WmmaStoreLds AmdGcn codegen; the arm uses matmul_batched_dt(BF16) as a serial fallback.
    let g = matmul_graph(16, 32, 16, GDt::BF16);
    let choice = matmul_choice(&g, Backend::AmdGcn(AmdArch::gfx1151()));
    assert_eq!(
        serial_matmul_dtype(&choice),
        Some(&Ty::BF16),
        "gfx1151 bf16 aligned should use serial bf16 fallback (amd_tc arm): got {choice:?}"
    );
}

#[test]
fn mixed_bf16_tc_key_encodes_operand_dtype_not_just_output() {
    // Card 045: the mixed-precision WMMA matmul (bf16 operands -> f32 output) must put the operand dtype
    // in its cache key, not only the output dtype; otherwise the `matmul:tc:` key reads as an f32 key
    // (dtag(F32) == "") and could share a disk artifact with an f32-operand kernel of the same shapes.
    // The WMMA tensor-core shape: bf16 operands, f32-accumulate output (the deleted `to_mixed_bf16`
    // pass used to produce this by inserting Cast(F32->BF16) on both operands; built directly here).
    let gf = matmul_graph(16, 32, 16, GDt::F32);
    let mut gm = matmul_graph(16, 32, 16, GDt::BF16);
    let mm_out = gm.output;
    gm.values[mm_out].aval.dtype = GDt::F32;
    // sanity: the matmul reads bf16 operands and writes f32.
    let mm = gm
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    assert_eq!(gm.aval(mm.out).dtype, GDt::F32, "output stays f32");
    for op in &mm.inputs {
        if let poot_graph_ir::Operand::Value(v) = op {
            assert_eq!(gm.aval(*v).dtype, GDt::BF16, "operands are bf16");
        }
    }
    // On gfx1151 the mixed matmul takes the amd_tc arm. Its key must start with `matmul:tc:` and carry
    // the bf16 operand tag, so it is distinct from any f32-operand kernel.
    let gfx1151 = Backend::AmdGcn(AmdArch::gfx1151());
    let choice = matmul_choice(&gm, gfx1151);
    assert!(
        matches!(
            contraction(&choice),
            Some(ContractionSpec::TensorCore { dt: Ty::F32, .. })
        ),
        "mixed matmul should take the WMMA tc arm: got {choice:?}"
    );
    // The equivalent f32-operand matmul (no cast) is a different request (the serial kernel), so the two
    // never share a plan key: the tensor-core request exists only for bf16 operands.
    let choice_f32 = matmul_choice(&gf, gfx1151);
    assert!(
        !is_tensor_core(&choice_f32),
        "the f32-operand matmul must not be the WMMA request: {choice_f32:?}"
    );
    let plan_key = |g: &Graph| {
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .unwrap();
        into_plain(planned_with_choice(g, eqn, gfx1151).unwrap().0)
            .key()
            .to_string()
    };
    assert_ne!(
        plan_key(&gm),
        plan_key(&gf),
        "bf16-operand and f32-operand plan keys must differ"
    );
}

#[test]
fn tensorcore_amdgcn_not_selected_for_non_rdna3() {
    // Backend::AmdGcn(gfx942) = CDNA3/CdnaMfma: no WMMA, must not take the tc path.
    let g = matmul_graph(16, 32, 16, GDt::BF16);
    let cdna = AmdArch::new("gfx942", 64);
    assert_eq!(
        cdna.tensor_core,
        TensorCoreSupport::CdnaMfma,
        "sanity: gfx942 = CDNA"
    );
    let choice = matmul_choice(&g, Backend::AmdGcn(cdna));
    assert!(
        !is_tensor_core(&choice),
        "gfx942 (CdnaMfma) must not take the RDNA3 WMMA path: got {choice:?}"
    );
}

#[test]
fn mixed_dtype_matmul_rejected_when_no_tensor_core_arm_fires() {
    // A mixed matmul (bf16 operands, f32 output; `to_mixed_bf16`'s shape) on an AMD arch without RDNA3
    // WMMA (CDNA gfx942, RDNA4) would fall to the serial fallback planned at Ty::F32: an f32 kernel
    // reading 2-byte bf16 operand buffers as f32 words (silent garbage), under a cache key colliding with
    // the f32 matmul. The planner must reject the mix by name on every route where
    // no tensor-core arm fires.
    let gm = mixed_matmul_graph(GDt::BF16, GDt::F32);
    let cdna = AmdArch::new("gfx942", 64);
    let r = try_matmul_plan(&gm, Backend::AmdGcn(cdna));
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "mixed bf16->f32 matmul on CDNA must be rejected, not silently planned at f32: got {r:?}"
    );
    // SPIR-V has no tensor-core arm either: same rejection.
    let r = try_matmul_plan(&gm, Backend::SpirvVulkan);
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "mixed bf16->f32 matmul on SPIR-V must be rejected: got {r:?}"
    );
    // The tensor-core arms are unaffected: RDNA3 and NVPTX still plan the WMMA kernel.
    assert!(is_tensor_core(&matmul_choice(
        &gm,
        Backend::AmdGcn(AmdArch::gfx1151())
    )));
    assert!(is_tensor_core(&matmul_choice(&gm, Backend::Nvptx)));
}

// --- card 235: the decode-time mixed-operand-dtype MatMul widen-cast pass ---

/// A matmul `[1,m,k] @ [k,n]` with a realistic operand-dtype mismatch: an F32 activation against a BF16
/// weight (the loader types every native-bf16 checkpoint's projection weights as bf16, independent of
/// the activation). `MatMul::infer` takes the output dtype from the first operand, so the output is F32
/// without a white-box override. This is the shape `plan_eqn`'s mixed-operand-dtype guard sees on a real
/// decode graph.
fn mixed_operand_matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::new(vec![1, m, k], GDt::F32));
    let w = bld.constant("w", TensorType::new(vec![k, n], GDt::BF16));
    let mm = bld.matmul(a, w);
    bld.finish(mm)
}

#[test]
fn widen_inserts_cast_for_decode_shaped_mixed_matmul() {
    // Card 235: decode is always M=1, so `mm.is_multiple_of(16)` fails and neither the amd_tc nor the tc
    // arm can fire, regardless of N/K alignment. Without the widen pass every backend that is not the
    // refused decode projection (NVPTX, and any other shape) rejects this graph.
    let g = mixed_operand_matmul_graph(1, 32, 16);
    // NVPTX has no refused column Gemv (PTX has its own prepare rule), so it still needs the widening Cast.
    let backend = Backend::Nvptx;
    let r = try_matmul_plan(&g, backend);
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "unwidened decode-shaped mixed matmul should be rejected on {backend:?}: got {r:?}"
    );

    let gw = widen_mismatched_matmul_dtypes(&g, backend);
    let mm = gw
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
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
        .eqns
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
        let gw = widen_mismatched_matmul_dtypes(&g, backend);
        assert!(
            !gw.eqns.iter().any(|e| matches!(e.op, OpKind::Cast { .. })),
            "the refused decode projection must not gain a widening Cast on {backend:?}"
        );
        assert_eq!(
            gw.aval(const_id(&gw, "w")).dtype,
            GDt::BF16,
            "the weight must stay BF16 on {backend:?}"
        );
    }
}

#[test]
fn widen_is_noop_when_tensor_core_arm_fires() {
    // A 16-aligned, no-batch-dims mixed matmul (bf16 operands -> f32 output, `to_mixed_bf16`'s shape)
    // takes the amd_tc/tc arm directly. The widen pass must leave it untouched (no Cast, operands stay
    // bf16) so `matmul_tensorcore`'s bf16 fragment loads see bf16 buffers and the `matmul:tc:` key is unchanged.
    let gm = mixed_matmul_graph(GDt::BF16, GDt::F32);
    for backend in [Backend::AmdGcn(AmdArch::gfx1151()), Backend::Nvptx] {
        let gw = widen_mismatched_matmul_dtypes(&gm, backend);
        assert_eq!(
            gw.eqns.len(),
            gm.eqns.len(),
            "no eqns should be added when the tensor-core arm already fires on {backend:?}"
        );
        assert!(
            !gw.eqns.iter().any(|e| matches!(e.op, OpKind::Cast { .. })),
            "no cast should be inserted when the tensor-core arm already fires on {backend:?}"
        );
        assert!(
            is_tensor_core(&matmul_choice(&gw, backend)),
            "tensor-core-eligible matmul should still take the tc arm after widen on {backend:?}"
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
    let (m, k, n) = (1, 37, 41); // deliberately non-16-aligned: decode-realistic, not tensor-core-shaped
    let g = mixed_operand_matmul_graph(m, k, n);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.037).sin() * 3.0).collect();
    // Round through bf16 up front, as a bf16 safetensors checkpoint already is (the loader
    // reads the raw bf16 bytes, widened to f32 by the loader).
    let w_data: Vec<f32> = (0..k * n)
        .map(|i| round_bf16((i as f32 * 0.019).cos() * 7.0))
        .collect();

    let mut inputs: HashMap<ValueId, poot_eval::Value> = HashMap::new();
    inputs.insert(
        g.inputs[0],
        poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![1, m, k], a_data.clone())),
    );
    inputs.insert(
        g.inputs[1],
        // `w_data` is already bf16-exact, so its words are the high halves of the f32 bit patterns.
        poot_eval::Value::from(poot_tensor::HostTensor::bf16(
            vec![k, n],
            w_data.iter().map(|v| (v.to_bits() >> 16) as u16).collect(),
        )),
    );
    let unwidened = poot_eval::eval(
        &g,
        &inputs,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval (unwidened)")
    .output
    .into_host()
    .expect("dense unwidened output");

    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let gw = widen_mismatched_matmul_dtypes(&g, backend);
        let widened = poot_eval::eval(
            &gw,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
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
        gf.inputs[0],
        poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![1, m, k], a_data)),
    );
    inputs_f32.insert(
        gf.inputs[1],
        poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![k, n], w_data)),
    );
    let f32_ref = poot_eval::eval(
        &gf,
        &inputs_f32,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
    )
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
    let bld = Builder::new();
    let x = bld.constant("x", TensorType::new(vec![8], in_dt));
    let c = bld.cast(x, to);
    let g = bld.finish(c);
    let eqn = g
        .eqns
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
        match cast_planned(in_dt, to, Backend::SpirvVulkan) {
            Ok((
                compute_plan,
                KernelChoice::Generated(KernelRequest::Elementwise(ElementwiseSpec::Cast(got))),
            )) if is_plain(&compute_plan) => {
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{in_dt:?}->{to:?}")
            }
            other => panic!("cast {in_dt:?}->{to:?} should plan a Compute kernel: {other:?}"),
        }
    }
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
        let bld = Builder::new();
        let x = bld.slot(
            poot_graph_ir::Slot::Activation,
            TensorType::new(vec![8], GDt::BF16),
        );
        let widened = bld.cast(x, GDt::F32);
        let g = bld.finish(widened);
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::Cast { .. }))
            .unwrap();
        match planned_with_choice(&g, eqn, Backend::Nvptx) {
            Ok((
                compute_plan,
                KernelChoice::Generated(KernelRequest::Elementwise(ElementwiseSpec::Cast(got))),
            )) if is_plain(&compute_plan) => assert_eq!(
                format!("{got:?}"),
                format!("{:?}", CastSpec::Bf16ToF32 { numel: 8 })
            ),
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
        let bld = Builder::new();
        let x = bld.constant("x", TensorType::new(vec![8], GDt::BF16));
        let c = bld.cast(x, GDt::F16);
        let g = bld.finish(c);
        let eqn = &g.eqns[0];
        let refusal = refusal_of(
            plan_eqn(
                &g,
                eqn,
                backend,
                &default_caps_for(backend),
                &poot_test_util::graph_fixtures::roomy_body_limits(),
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
                target: backend,
                missing: Capability::DtypeLowering,
            },
            "on {backend:?}"
        );
    }
}

/// An I32->F32 cast of an exact-I32 source (a `GeU` output, as card 372c's guarded selector Cast reads).
fn exact_i32_cast_planned(backend: Backend) -> Result<(Plan, KernelChoice), PlanError> {
    let bld = Builder::new();
    let x = bld.constant("x", TensorType::new(vec![8], GDt::I32));
    let flag = bld.binary_scalar(GBinOp::GeU, x, Scalar::I32(0));
    let c = bld.cast(flag, GDt::F32);
    let g = bld.finish(c);
    let eqn = g
        .eqns
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
            KernelChoice::Generated(KernelRequest::Elementwise(ElementwiseSpec::Cast(
                CastSpec::I32ToF32 { numel: 8 }
            )))
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
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    plan_eqn(
        g,
        eqn,
        backend,
        &default_caps_for(backend),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    // plans. A rank-3 weight on AmdGcn steers past every fast path (decode-GEMV and the generated tiled
    // GEMM need a rank-2 weight; the imported batched-tiled GEMM is SpirvVulkan/Nvptx-only for a plain
    // MatMul) to the naive serial fallback.
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![2, 32, 32], GDt::F32));
    let w = b.constant("w", TensorType::new(vec![2, 32, 16], GDt::F32));
    let out = b.matmul(x, w);
    let g = b.finish(out);
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![4, 8], GDt::F16));
    let w = b.constant("w", TensorType::new(vec![2, 8, 16], GDt::F16));
    let idx = b.constant("idx", TensorType::new(vec![4], GDt::I32));
    let out = b.indexed_matmul(x, w, idx);
    let g = b.finish(out);
    let eqn = g
        .eqns
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
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    );
    assert!(
        res.is_err(),
        "an F16-typed non-wired op must be rejected, not silently planned: got {res:?}"
    );
}

/// Card 557: the contraction choice (`is_tileable_matmul_in`) takes the generated tiled GEMM only for
/// an M>1 matmul over F32 operands and a rank-2 weight, on every backend. A BF16 matmul must reach the
/// WMMA / tensor-core arms, and an F16 one the dtype-generic serial kernel (the generated body is
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
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .expect("a matmul eqn");
        let planned = plan_eqn_choice(
            &g,
            eqn,
            backend,
            &HashMap::new(),
            &default_caps_for(backend),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap_or_else(|e| panic!("{backend:?} {dtype:?} M={m}: {e:?}"));
        assert_eq!(
            is_tiled_region(&planned.choice),
            tiled,
            "{backend:?} {dtype:?} M={m}: {:?}",
            planned.choice
        );
    }
}

#[test]
fn f16_matmul_never_reaches_tiled_region() {
    // FR-012: an F16 whole-graph matmul with M>1 (the tiled-GEMM shape) must not take the f32-only
    // generated tiled GEMM (`is_tileable_matmul_in`, the contraction choice); that kernel reads all
    // buffers as f32, so an F16 operand would be read as 4-byte f32 over 2-byte storage (a fault). M=32
    // (>1) is the shape the tiled choice targets for f32.
    let g = matmul_graph(32, 32, 16, GDt::F16);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    let planned = plan_eqn_choice(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &HashMap::new(),
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .expect("the F16 matmul plans");
    assert!(
        !matches!(
            planned.choice,
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::TiledRegion { .. }
            ))
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

/// Card 154 end-to-end: trace a small F16-in/F32-out matmul graph, plan it through the SpirvVulkan
/// coopmat selection arm, emit it via `poot_codegen`, and confirm the SPIR-V (after `fix_coopmat_calls`,
/// wired into `compile()`) is `spirv-val`-clean, through the real graph-trace -> plan -> emit pipeline.
/// Skips if `llc`/`spirv-val` are absent, like every other real-toolchain test in this workspace.
///
/// Card 525 SC-001 / R469-003: the grid assertion below closes the drift the pre-card-525
/// `dispatch_grid` had (no coopmat arm, so it fell through to the generic `out_numel` default).
/// MUTATION (recorded here, never left in the tree): in `planner/matmul.rs`'s `coopmat` arm, `grid:
/// [(num_tiles * 32) as u32, 1, 1]` was changed to `grid: [out_numel as u32, 1, 1]` (the pre-card-525
/// value). Rerunning this test then panicked: `assertion `left == right` failed: planned grid must
/// be num_tiles*32 (1 tile here), not the generic out_numel default (256) left: [256, 1, 1] right:
/// [32, 1, 1]`. Restoring the real formula made it green again.
#[test]
fn coopmat_matmul_traces_plans_and_validates() {
    // 16x16x16: F16 operands, F32 output (the only shape/dtype combination
    // `matmul_spirv_coopmat_eligible` accepts, see its doc).
    let g = mixed_matmul_graph_shape(16, 16, 16, GDt::F16, GDt::F32);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    let (plan, choice) = planned_with_choice(&g, eqn, Backend::SpirvVulkan)
        .expect("a 16x16x16 f16-in/f32-out matmul must plan on SpirvVulkan");
    let Dispatch {
        body,
        launch: Launch { grid, .. },
        ..
    } = into_plain(plan);
    assert!(
        matches!(contraction(&choice), Some(ContractionSpec::Coopmat { .. })),
        "expected the coopmat arm to fire: got {choice:?}"
    );
    // card 525 R469-003: `matmul_tensorcore_coopmat` bakes one 32-lane subgroup per 16x16 output tile
    // (`body.workgroup_size == [32,1,1]`, `block = global_thread_id / 32`); at 16x16 output that is
    // exactly 1 tile, so the plan's own grid must be [32,1,1] threads, not out_numel (16*16 = 256) -
    // the wrong value the pre-card-525 `dispatch_grid` fell through to (no coopmat arm).
    assert_eq!(
        body.workgroup_size,
        [32, 1, 1],
        "the coopmat body's baked workgroup size must stay one subgroup"
    );
    assert_eq!(
        grid,
        [32, 1, 1],
        "planned grid must be num_tiles*32 (1 tile here), not the generic out_numel default (256)"
    );

    fn have(bin: &str) -> bool {
        std::process::Command::new(bin)
            .arg("--version")
            .output()
            .is_ok()
    }
    if !have("llc") || !have("spirv-val") {
        eprintln!("llc/spirv-val not on PATH; skipping coopmat graph-plan round-trip validation");
        return;
    }
    let dir = std::env::temp_dir().join("poot-graph-plan-coopmat-test");
    std::fs::create_dir_all(&dir).unwrap();
    let out =
        poot_codegen::artifact_path(&dir, "coopmat_matmul", poot_codegen::Target::SpirvVulkan);
    poot_codegen::compile(&body, poot_codegen::Target::SpirvVulkan, &out)
        .expect("the planned coopmat body must compile");
    let v = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        v.status.success(),
        "spirv-val failed on the planned coopmat matmul:\n{}",
        String::from_utf8_lossy(&v.stderr)
    );
}

/// Card 525: the NVPTX `tc` arm's grid dropped the batch factor.
/// `matmul_nvptx_tc_eligible` deliberately allows batch dims ("batch dims are fine on NVPTX, unlike
/// the AMD arm"), and `kg::matmul_tensorcore` (wmma.rs) launches one 256-thread block per
/// `batch_count * tiles_m * tiles_n`, decoding `bidx = tid/256`. A `[2,16,16] @ [16,16] -> [2,16,16]`
/// bf16 matmul is 1 tile (16x16) per batch element, so the plan must launch `2*1*256 = 512` threads,
/// not the pre-fix `256` (which dispatches only batch 0; batch 1's output buffer would keep its
/// stale/zero contents on a real device).
///
/// MUTATION (recorded here, never left in the tree): in `planner/matmul.rs`'s `tc` arm, `grid:
/// [(batch_dims_product * num_tiles * 256) as u32, 1, 1]` was changed to `grid: [(num_tiles * 256) as
/// u32, 1, 1]` (dropping the batch factor, the pre-fix bug). Rerunning this test then panicked:
/// `assertion `left == right` failed: planned grid must include the batch factor (2 batches * 1 tile *
/// 256 threads) left: [256, 1, 1] right: [512, 1, 1]`. Restoring the real formula made it green again.
#[test]
fn nvptx_tc_batched_bf16_matmul_grid_includes_batch_factor() {
    let g = batched_matmul_graph(2, 16, 16, 16, GDt::BF16);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    let (plan, choice) = planned_with_choice(&g, eqn, Backend::Nvptx)
        .expect("a batched 16x16x16 bf16 matmul must plan on Nvptx");
    let grid = into_plain(plan).launch().grid;
    assert!(
        is_tensor_core(&choice),
        "expected the NVPTX tc arm to fire: got {choice:?}"
    );
    assert_eq!(
        grid,
        [512, 1, 1],
        "planned grid must include the batch factor (2 batches * 1 tile * 256 threads)"
    );
}

/// `batched_matmul_graph` with the output retagged to the f32 accumulate type (the mixed shape every tensor-core arm
/// admits), planned for `backend`: the plan and its kernel choice.
fn mixed_batched_matmul_plan(
    batch: usize,
    (m, k, n): (usize, usize, usize),
    operand: GDt,
    backend: Backend,
) -> Result<(Plan, KernelChoice), PlanError> {
    let mut g = batched_matmul_graph(batch, m, k, n, operand);
    let out = g.output;
    g.values[out].aval.dtype = GDt::F32;
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    planned_with_choice(&g, eqn, backend)
}

/// Card 663 SC-001/SC-003, planner side: the cooperative-matrix arm admits a multi-block (`K = 48`) batched
/// (`[2, 32, 48] @ [48, 32]`) matmul, launching one subgroup per output tile of every batch element
/// (`2 * 2 * 2 = 8` tiles of 32 lanes); a ragged `K` (not a multiple of 16) is not eligible, is not padded, and
/// refuses the mixed-dtype lowering (the f16-in/f32-out shape has no other kernel).
///
/// Mutation: restore the single K tile in `matmul_spirv_coopmat_eligible` (`k == 16`); the multi-block shape
/// stops selecting the coopmat arm and the first assertion goes red.
#[test]
fn coopmat_arm_admits_multi_block_batched_shapes_and_refuses_a_ragged_k() {
    let (plan, choice) = mixed_batched_matmul_plan(2, (32, 48, 32), GDt::F16, Backend::SpirvVulkan)
        .expect("a 16-aligned batched f16-in/f32-out matmul plans on SpirvVulkan");
    assert!(
        matches!(contraction(&choice), Some(ContractionSpec::Coopmat { .. })),
        "expected the coopmat arm for K = 48: got {choice:?}"
    );
    assert_eq!(
        into_plain(plan).launch().grid,
        [8 * 32, 1, 1],
        "one 32-lane subgroup per output tile of every batch element"
    );
    let ragged = mixed_batched_matmul_plan(2, (32, 40, 32), GDt::F16, Backend::SpirvVulkan);
    assert!(
        matches!(&ragged, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "a ragged K must refuse rather than pad: got {ragged:?}"
    );
}

/// Card 663 SC-001, planner side for the AMD arm: a batched multi-block bf16 matmul on gfx1151 selects the WMMA
/// tensor-core kernel (it used to fall to the serial kernel for any batch dim) and launches one 256-thread block per
/// output tile of every batch element.
///
/// Mutation: restore the single-batch requirement in `matmul_amd_tc_eligible`; the choice is no longer the
/// tensor-core kernel and the row goes red.
#[test]
fn amd_tensor_core_arm_admits_multi_block_batched_shapes() {
    let (plan, choice) = mixed_batched_matmul_plan(
        2,
        (32, 48, 32),
        GDt::BF16,
        Backend::AmdGcn(AmdArch::gfx1151()),
    )
    .expect("a 16-aligned batched bf16-in/f32-out matmul plans on gfx1151");
    assert!(
        is_tensor_core(&choice),
        "expected the WMMA tensor-core arm for a batched K = 48: got {choice:?}"
    );
    assert_eq!(into_plain(plan).launch().grid, [8 * 256, 1, 1]);
}

#[test]
fn tensorcore_not_selected_for_spirv() {
    // SpirvVulkan + bf16 + 16-aligned -> the serial path (WMMA is not wgpu).
    let g = matmul_graph(16, 32, 16, GDt::BF16);
    let choice = matmul_choice(&g, Backend::SpirvVulkan);
    assert!(
        !is_tensor_core(&choice),
        "SpirvVulkan must never select the tc path: got {choice:?}"
    );
}

/// `g` compiled for `backend` through the full pipeline.
fn compiled(g: &Graph, backend: Backend) -> Program {
    let target = Target {
        backend,
        caps: default_caps_for(backend),
    };
    let options = CompileOptions::new(Submission::Replay, crate::CompileLimits::STANDARD);
    compile(g, &target, &options).expect("the matmul compiles")
}

/// The compiled MatMul's plan and kernel choice.
fn compiled_matmul(program: &Program) -> (&Plan, &KernelChoice) {
    let (eqn, plan) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::MatMul))
        .expect("a matmul eqn");
    (plan, program.kernel_choice(eqn))
}

fn is_tiled_region(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Generated(KernelRequest::Contraction(
            ContractionSpec::TiledRegion { .. }
        ))
    )
}

#[test]
fn tiled_matmul_gets_the_tiled_grid_not_the_naive_fallthrough() {
    // Card 099b: the generated tiled GEMM (the contraction choice for an M>1 F32 shared-weight
    // MatMul, Card 557) gets the tiled-GEMM launch grid (`groups * GEMM_TILE^2` threads), not the
    // one-thread-per-output default (a ~2x over-launch).
    let (m, k, n) = (64usize, 32usize, 48usize);
    let g = matmul_graph(m, k, n, GDt::F32);
    let out_numel = m * n;

    let program = compiled(&g, Backend::SpirvVulkan);
    let (plan, choice) = compiled_matmul(&program);
    assert!(
        is_tiled_region(choice),
        "the compile chooses the tiled GEMM: {choice:?}"
    );
    let tiled = plain(plan).launch().grid;

    let groups = m.div_ceil(2 * GEMM_TILE) * n.div_ceil(GEMM_TILE);
    assert_eq!(
        tiled,
        [(groups * GEMM_TILE * GEMM_TILE) as u32, 1, 1],
        "the generated tiled GEMM must get the tiled-GEMM grid"
    );
    assert_ne!(
        tiled,
        [out_numel as u32, 1, 1],
        "must NOT be the naive one-thread-per-output fallthrough"
    );
}

/// The generated tiled GEMM over the RADV tile ceiling (card 096) splits into sub-cap dispatches whose grids
/// are `groups * GEMM_TILE^2` threads each and whose groups sum to the whole tile range. Mutation: build a
/// split dispatch's grid as `groups + workgroup` instead of `groups * workgroup`; the grid assertion goes red.
#[test]
fn a_tiled_gemm_over_the_tile_ceiling_splits_into_disjoint_launches() {
    let (m, k, n) = (4096usize, 8usize, 4096usize);
    let g = matmul_graph(m, k, n, GDt::F32);
    let program = compiled(&g, Backend::SpirvVulkan);
    let (plan, choice) = compiled_matmul(&program);
    assert!(
        is_tiled_region(choice),
        "the compile chooses the tiled GEMM: {choice:?}"
    );
    let ceiling = default_caps_for(Backend::SpirvVulkan)
        .known_miscompiles
        .tiled_gemm_max_workgroups
        .expect("wgpu on RADV has a tile ceiling") as usize;
    let groups = m.div_ceil(2 * GEMM_TILE) * n.div_ceil(GEMM_TILE);
    assert!(groups >= ceiling, "the fixture must reach the ceiling");
    let dispatches = plan.dispatches();
    assert!(dispatches.len() > 1, "{plan:?}");
    let lanes = GEMM_TILE * GEMM_TILE;
    let mut covered = 0usize;
    for dispatch in dispatches {
        let launch = dispatch.launch();
        let chunk_groups = launch.grid[0] as usize / lanes;
        assert_eq!(
            launch.grid,
            [(chunk_groups * lanes) as u32, 1, 1],
            "a split dispatch launches whole tile workgroups"
        );
        assert!(
            chunk_groups < ceiling,
            "each launch stays under the ceiling"
        );
        covered += chunk_groups;
    }
    assert_eq!(covered, groups, "the launches tile the whole output");
}

/// Card 557: the generated tiled GEMM owns its launch (`GEMM_TILE^2`-lane workgroups over a 64-lane LDS
/// tile), and the op alone no longer says so: `finalize` reads the recorded choice. On AMDGCN (where
/// the imported tiled GEMM, and so `is_tiled_gemm_in`, does not apply) a MatMul whose output is past the
/// grid cap at 64 lanes must keep its 64-lane workgroups. Mutation: drop the choice from `finalize`'s
/// custom-grid test; the wg-bump widens the body to 256 lanes and this row goes red.
#[test]
fn a_large_tiled_matmul_keeps_its_tile_workgroup_on_amdgcn() {
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let caps = default_caps_for(backend);
    let (k, n) = (8usize, 2048usize);
    let m = (caps.max_grid[0] as usize * GEMM_TILE * GEMM_TILE).div_ceil(n) + 1;
    let g = matmul_graph(m, k, n, GDt::F32);
    let program = compiled(&g, backend);
    let (plan, choice) = compiled_matmul(&program);
    let bodies: Vec<&Body> = plan
        .dispatches()
        .iter()
        .map(|dispatch| dispatch.body())
        .collect();
    assert!(
        !bodies.is_empty(),
        "the tiled GEMM plans a dispatch: {plan:?}"
    );
    assert!(
        is_tiled_region(choice),
        "an M>1 F32 matmul takes the generated tiled GEMM: {choice:?}"
    );
    for body in bodies {
        assert_eq!(
            body.workgroup_size[0] as usize,
            GEMM_TILE * GEMM_TILE,
            "the tiled GEMM's workgroup must stay one tile wide"
        );
    }
}

// --- a collective is a cut between stages, never an equation a stage program plans ---

/// POOT-747 SC-005: a collective built into one stage graph (uncut) is refused by `compile` on every
/// backend. The baseline lowered it to `Plan::Alias` and `compile` returned `Ok`, silently dropping the
/// cross-device exchange the node stands for.
#[test]
fn a_collective_inside_a_stage_graph_is_refused_by_compile() {
    let world = std::num::NonZeroU32::new(2).unwrap();
    type Collective<'a> = &'a dyn Fn(&Builder, poot_graph_ir::Traced) -> poot_graph_ir::Traced;
    let all_reduce = |bld: &Builder, y| bld.all_reduce(y, RedOp::Sum, 1);
    let all_gather = |bld: &Builder, y| bld.all_gather(y, 1, world);
    let cases: [(&str, Collective); 2] = [("all_reduce", &all_reduce), ("all_gather", &all_gather)];
    for backend in [Backend::Nvptx, Backend::SpirvVulkan] {
        for (name, collective) in cases {
            let bld = Builder::new();
            let x = bld.constant("x", TensorType::f32(vec![2, 8]));
            let w = bld.constant("w", TensorType::f32(vec![8, 4]));
            let y = collective(&bld, bld.matmul(x, w));
            let reduced = y.id();
            let g = bld.finish(y);
            let target = Target {
                backend,
                caps: default_caps_for(backend),
            };
            let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
            match compile(&g, &target, &options) {
                Err(CompileError::Plan(error)) => match *error {
                    PlanError::Refused(refusal) => {
                        assert_eq!(
                            refusal.missing,
                            Capability::CollectiveInStage,
                            "{backend:?} {name}"
                        );
                        assert_eq!(refusal.eqn, reduced, "{backend:?} {name}");
                        assert_eq!(refusal.target, backend);
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
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::f32(a_shape));
    let t = movement(&bld, a);
    let y = bld.binary(GBinOp::Add, t, t);
    let g = bld.finish(y);
    let t_id = t.id();
    let move_eqn = g
        .eqns
        .iter()
        .position(|e| e.out == t_id)
        .expect("the movement eqn produces t");
    (g, t_id, move_eqn)
}

#[test]
fn compute_views_never_promotes_validation_values() {
    let (g, t_id, _) = view_test_graph(vec![3, 2, 4], |bld, a| bld.transpose(a, vec![2, 0, 1]));
    assert!(
        compute_views(&g, Backend::SpirvVulkan).contains_key(&t_id),
        "control: without a witness the transpose is a view"
    );
    let observed = g.with_validations(vec![poot_graph_ir::ValidationOutput {
        id: ValidationId(3),
        name: "transposed".into(),
        value: t_id,
    }]);
    observed.validate().unwrap();
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
    let (g, t_id, move_eqn) =
        view_test_graph(vec![3, 2, 4], |bld, a| bld.transpose(a, vec![2, 0, 1]));
    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        views.contains_key(&t_id),
        "a Transpose feeding only a Binary(Add) must be promoted to a view"
    );
    let eqn = &g.eqns[move_eqn];
    match plan_eqn_views(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &views,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::f32(vec![4, 3]));
    let t = bld.transpose(a, vec![1, 0]); // [3,4]
    let w = bld.constant("w", TensorType::f32(vec![4, 5]));
    let mm = bld.matmul(t, w); // [3,5]
    let g = bld.finish(mm);
    let t_id = t.id();

    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        !views.contains_key(&t_id),
        "a Transpose feeding a MatMul must NOT be promoted to a view (FR-004)"
    );
    let move_eqn = g.eqns.iter().find(|e| e.out == t_id).unwrap();
    match plan_eqn_views(
        &g,
        move_eqn,
        Backend::SpirvVulkan,
        &views,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::f32(vec![3, 2])); // pre-transpose source
    let t = bld.transpose(a, vec![1, 0]); // view candidate, shape [2,3]
    let b = bld.constant("b", TensorType::f32(vec![2, 3])); // shared operand
    let y_view = bld.binary(GBinOp::Add, t, b); // t (a view) + b
    let c = bld.constant("c", TensorType::f32(vec![2, 3])); // a PLAIN buffer, same shape as t
    let y_plain = bld.binary(GBinOp::Add, c, b); // c (plain) + b - same shapes as y_view's eqn
    let out = bld.binary(GBinOp::Add, y_view, y_plain);
    let g = bld.finish(out);

    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        views.contains_key(&t.id()),
        "t's only consumer (y_view's Binary) is strided-capable"
    );

    let eqn_view = g.eqns.iter().find(|e| e.out == y_view.id()).unwrap();
    let eqn_plain = g.eqns.iter().find(|e| e.out == y_plain.id()).unwrap();
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        let (g, t_id, move_eqn) =
            view_test_graph(a_shape.clone(), |bld, a| bld.transpose(a, perm.clone()));
        let views = compute_views(&g, Backend::SpirvVulkan);
        let layout = views
            .get(&t_id)
            .cloned()
            .expect("a Transpose into Binary(Add) must be a view");

        let a_numel = numel(&a_shape);
        let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32 + 0.5).collect();
        let a_tensor = poot_tensor::HostTensor::f32(a_shape.clone(), a_data.clone());
        let oracle = poot_eval::apply_movement(&g.eqns[move_eqn].op, &[a_tensor]).unwrap();

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
        let (g, t_id, move_eqn) =
            view_test_graph(a_shape.clone(), |bld, a| bld.slice(a, axis, start, end));
        let views = compute_views(&g, Backend::SpirvVulkan);
        let layout = views
            .get(&t_id)
            .cloned()
            .expect("a Slice into Binary(Add) must be a view");

        let a_numel = numel(&a_shape);
        let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32 + 0.5).collect();
        let a_tensor = poot_tensor::HostTensor::f32(a_shape.clone(), a_data.clone());
        let oracle = poot_eval::apply_movement(&g.eqns[move_eqn].op, &[a_tensor]).unwrap();

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
            bld.broadcast(a, out_shape.clone())
        });
        let views = compute_views(&g, Backend::SpirvVulkan);
        let layout = views
            .get(&t_id)
            .cloned()
            .expect("a Broadcast into Binary(Add) must be a view");

        let a_numel = numel(&a_shape);
        let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32 + 0.5).collect();
        let a_tensor = poot_tensor::HostTensor::f32(a_shape.clone(), a_data.clone());
        let oracle = poot_eval::apply_movement(&g.eqns[move_eqn].op, &[a_tensor]).unwrap();

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

/// Dispatches per token of the 24-layer Qwen2.5-0.5B real-config decode graph (the legacy tracer planned 415).
/// POOT-725 made the `Cast(Slot::Pos -> F32)` of the position a kernel (368); POOT-1108 folds it into the
/// mask region that reads it, so the launch is gone again.
const QWEN2_0_5B_DISPATCHES: usize = 367;

/// Dispatches per token of the 28-layer Qwen3-0.6B real-config decode graph (the legacy tracer planned 539).
/// POOT-725: 484; POOT-1108: 483, the position cast folded as for Qwen2.5 above.
const QWEN3_0_6B_DISPATCHES: usize = 483;

/// Dispatches `program` runs per step: every planned equation but the free ones (an alias or a strided view).
fn dispatches(program: &Program) -> usize {
    program
        .planned()
        .filter(|(_, plan)| !matches!(plan, Plan::Alias(_) | Plan::View { .. }))
        .count()
}

#[test]
fn dispatch_count_planned_regression_on_real_qwen2_decode_configs() {
    // Pins the dispatches per token of the real Qwen2.5-0.5B / Qwen3-0.6B decode graphs (`Model::trace`,
    // one token) through `compile`, the production pipeline, on both planner families (SPIR-V and PTX).
    // The pins are exact: a bound above the real count would still pass with a fusion pass deleted, so any
    // change in either direction fails here and the pin is updated deliberately with the change that moved
    // it. Weights-free: the count depends only on graph structure.
    let cap = 64usize;
    for (name, dense, pin) in [
        ("qwen2_0_5b", qwen2_0_5b(), QWEN2_0_5B_DISPATCHES),
        ("qwen3_0_6b", qwen3_0_6b(), QWEN3_0_6B_DISPATCHES),
    ] {
        let g = real_decode_graph(dense, cap);
        for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
            let program = compiled(&g, backend);
            assert_eq!(
                dispatches(&program),
                pin,
                "{name} {backend:?} dispatches per token moved off its pin"
            );
            // Every norm scale and bias is read narrow by the region that uses it: no cast of a weight runs.
            let weight_casts = program
                .graph()
                .eqns
                .iter()
                .filter(|eqn| {
                    matches!(eqn.op, OpKind::Cast { .. })
                        && matches!(
                            eqn.inputs.as_slice(),
                            [Operand::Value(src)] if program.graph().meta(*src).storage == Storage::Const
                        )
                })
                .count();
            assert_eq!(
                weight_casts, 0,
                "{name} {backend:?}: a weight cast still dispatches"
            );
        }
    }
}

/// A copy of `g` with every named value renamed by `rename(value_index, old_name)`. Ids, shapes,
/// storage and equations are untouched, so only the strings differ.
fn rename_values(g: &Graph, rename: impl Fn(usize, &str) -> String) -> Graph {
    let mut renamed = g.clone();
    for (index, value) in renamed.values.iter_mut().enumerate() {
        if let Some(name) = &value.name {
            value.name = Some(rename(index, name));
        }
    }
    renamed
}

/// Everything the per-equation planner decides for `g` on `backend`, as comparable text: the strided
/// views, then the plan (or typed refusal) of every equation. `Plan` carries kernel bodies and cache
/// keys, so a different body, key, grid metadata or refusal shows up as a different string.
fn planned_graph_summary(g: &Graph, backend: Backend) -> Vec<String> {
    let views = compute_views(g, backend);
    let mut view_ids: Vec<ValueId> = views.keys().copied().collect();
    view_ids.sort();
    let mut summary: Vec<String> = view_ids
        .into_iter()
        .map(|id| format!("view {id}: {:?}", views[&id]))
        .collect();
    for eqn in &g.eqns {
        let plan = plan_eqn_views(
            g,
            eqn,
            backend,
            &views,
            &default_caps_for(backend),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        );
        summary.push(format!("{}: {plan:?}", eqn.out));
    }
    summary
}

/// SC-002 (card 520a): planning is a function of the graph's structure, never of a name. The same
/// dense decode and prefill graphs (raw and through `optimize`) are planned on every planned backend
/// under their tracer names and under names of other model families and role words; every equation
/// must plan identically. Mutation observed red: the per-equation planner refuses (or plans
/// differently) when an operand name contains "qwen".
#[test]
fn planning_a_graph_does_not_depend_on_its_value_names() {
    use poot_executor_parity::dense::{Dense, Family};
    use poot_models::model::{LogitRows, Phase};

    let mut graphs = Vec::new();
    for family in [Family::Qwen2, Family::Qwen3] {
        let dense = Dense::new(family)
            .vocab(32)
            .dims(16, 32, 2)
            .heads(4, 2)
            .head_dim(4)
            .max_positions(16);
        let decode = dense_graph(dense.clone(), Phase::Decode, 1, 16, LogitRows::Last);
        let prefill = dense_graph(dense, Phase::Prefill, 8, 16, LogitRows::Last);
        graphs.push(optimize(&decode));
        graphs.push(decode);
        graphs.push(optimize(&prefill));
        graphs.push(prefill);
    }

    type Naming = (&'static str, fn(usize, &str) -> String);
    let namings: [Naming; 4] = [
        ("qwen", |index, _| format!("qwen2.layers.{index}.weight")),
        ("deepseek", |index, _| {
            format!("deepseek.blocks.{index}.expert_{index}")
        }),
        ("role words", |index, _| {
            format!("layer_idx.{index}.tensor_name.expert_role")
        }),
        ("opaque", |index, _| format!("v{index}")),
    ];
    let backends = [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ];

    for (graph_index, graph) in graphs.iter().enumerate() {
        assert!(
            graph.values.iter().any(|value| value.name.is_some()),
            "graph {graph_index} has no named value, so renaming would prove nothing"
        );
        for backend in backends {
            let original = planned_graph_summary(graph, backend);
            assert!(
                original.iter().filter(|line| line.contains("Ok(")).count() > graph.eqns.len() / 2,
                "graph {graph_index} on {backend:?} must mostly plan, or the comparison is vacuous"
            );
            for (family, rename) in namings {
                let renamed = planned_graph_summary(&rename_values(graph, rename), backend);
                assert_eq!(
                    renamed.len(),
                    original.len(),
                    "graph {graph_index} on {backend:?} under {family} names: entry count"
                );
                // Bodies are large, so name the first differing equation and clip both sides.
                if let Some((was, now)) = original.iter().zip(&renamed).find(|(a, b)| a != b) {
                    let clip = |text: &str| text.chars().take(160).collect::<String>();
                    panic!(
                        "graph {graph_index} on {backend:?} planned differently under {family} \
                         names:\n  tracer names: {}\n  {family} names: {}",
                        clip(was),
                        clip(now)
                    );
                }
            }
        }
    }
}

#[test]
fn amdgcn_view_promotion_matches_spirv_on_real_qwen2_decode_configs() {
    // Card 137: `compute_views`'s gate is `SpirvVulkan | AmdGcn(_)` (spec 132 phase 1). Pins that ROCm
    // agrees exactly with wgpu on the real Qwen2.5-0.5B and Qwen3-0.6B decode graphs (same fixtures as
    // `dispatch_count_planned_regression_on_real_qwen2_decode_configs` above): identical promoted
    // value-id set and identical composed Layout (strides+offset) per promoted value. Also asserts the
    // planned dispatch counts agree between backends and are strictly below the plan-independent
    // `transform::dispatch_count`, so ROCm gets the same free-view win as wgpu.
    //
    // Uses the raw (pre-`transform::optimize`) traced graph: optimize's fusion passes absorb every
    // Transpose/Slice/Broadcast before `compute_views` sees them (the fully optimized graph
    // promotes 0 views on both configs), so only the raw graph, which keeps the movement ops as distinct
    // eqns, exercises the promotion path.
    use poot_graph_ir::analysis::dispatch_count;
    let cap = 64usize;
    let amd = Backend::AmdGcn(AmdArch::gfx1151());

    for (name, cfg) in [("qwen2_0_5b", qwen2_0_5b()), ("qwen3_0_6b", qwen3_0_6b())] {
        let g = real_decode_graph(cfg, cap);

        let views_spirv = compute_views(&g, Backend::SpirvVulkan);
        let views_amd = compute_views(&g, amd);

        let mut spirv_ids: Vec<ValueId> = views_spirv.keys().copied().collect();
        let mut amd_ids: Vec<ValueId> = views_amd.keys().copied().collect();
        spirv_ids.sort();
        amd_ids.sort();
        assert_eq!(
            spirv_ids, amd_ids,
            "{name}: AmdGcn must promote the IDENTICAL value-id set as SpirvVulkan"
        );
        assert!(
            !spirv_ids.is_empty(),
            "{name}: this real decode graph is known to have promotable movement ops on the raw \
             (pre-fusion) trace - an empty set would mean the test fixture stopped exercising the \
             promotion path"
        );
        for id in &spirv_ids {
            let l_spirv = &views_spirv[id];
            let l_amd = &views_amd[id];
            assert_eq!(
                l_spirv.strides, l_amd.strides,
                "{name}: value {id} strides diverge between SpirvVulkan and AmdGcn"
            );
            assert_eq!(
                l_spirv.offset, l_amd.offset,
                "{name}: value {id} offset diverges between SpirvVulkan and AmdGcn"
            );
        }

        let n_spirv = dispatch_count_planned(&g, Backend::SpirvVulkan);
        let n_amd = dispatch_count_planned(&g, amd);
        assert_eq!(
            n_amd, n_spirv,
            "{name}: dispatch_count_planned must agree between AmdGcn and SpirvVulkan"
        );
        let n_unpromoted = dispatch_count(&g);
        assert!(
            n_amd < n_unpromoted,
            "{name}: AmdGcn's planned dispatch count ({n_amd}) must be strictly below the \
             unpromoted dispatch_count ({n_unpromoted}) - else the ROCm gate relaxation bought \
             nothing on this graph"
        );
    }
}

/// Card 549 SC-003 (R-546-9): the same fixture plans the same number of views on Nvptx as on
/// AmdGcn, now that `poot_codegen::capability(Nvptx).strided_views` is `true` - every
/// `is_strided_capable` kernel body PTX dispatches is the same backend-neutral `poot-kernelgen`
/// body wgpu/ROCm already read strided operands through, so there was never a codegen reason for
/// PTX alone to pay the extra materialize copy. GPU-free (planning only). Mirrors
/// `amdgcn_view_promotion_matches_spirv_on_real_qwen2_decode_configs` above, against Nvptx
/// instead of SpirvVulkan.
///
/// Mutation: restore `strided_views: false` for `Target::Nvptx` in
/// `poot-codegen/src/lib.rs::capability`; `compute_views(&g, Nvptx)` goes back to returning an
/// empty map (its `if !codegen_capability(backend).strided_views { return views; }` early-out),
/// so `nvptx_ids` is empty while `amd_ids` is not, and this row goes red.
#[test]
fn nvptx_view_promotion_matches_amdgcn_on_real_qwen2_decode_configs() {
    let cap = 64usize;
    let amd = Backend::AmdGcn(AmdArch::gfx1151());

    for (name, cfg) in [("qwen2_0_5b", qwen2_0_5b()), ("qwen3_0_6b", qwen3_0_6b())] {
        let g = real_decode_graph(cfg, cap);

        let views_nvptx = compute_views(&g, Backend::Nvptx);
        let views_amd = compute_views(&g, amd);

        let mut nvptx_ids: Vec<ValueId> = views_nvptx.keys().copied().collect();
        let mut amd_ids: Vec<ValueId> = views_amd.keys().copied().collect();
        nvptx_ids.sort();
        amd_ids.sort();
        assert_eq!(
            nvptx_ids, amd_ids,
            "{name}: Nvptx must promote the IDENTICAL value-id set as AmdGcn"
        );
        assert!(
            !nvptx_ids.is_empty(),
            "{name}: this real decode graph is known to have promotable movement ops on the raw \
             (pre-fusion) trace - an empty set would mean Nvptx fell back to the pre-549 \
             materialize-everything behavior"
        );
        for id in &nvptx_ids {
            let l_nvptx = &views_nvptx[id];
            let l_amd = &views_amd[id];
            assert_eq!(
                l_nvptx.strides, l_amd.strides,
                "{name}: value {id} strides diverge between Nvptx and AmdGcn"
            );
            assert_eq!(
                l_nvptx.offset, l_amd.offset,
                "{name}: value {id} offset diverges between Nvptx and AmdGcn"
            );
        }

        let n_nvptx = dispatch_count_planned(&g, Backend::Nvptx);
        let n_amd = dispatch_count_planned(&g, amd);
        assert_eq!(
            n_nvptx, n_amd,
            "{name}: dispatch_count_planned must agree between Nvptx and AmdGcn"
        );
    }
}

/// Card 549 SC-004 (F1): the strided-view fixture plans the same copy
/// (dispatch) count on Nvptx as on AmdGcn - a GPU-free plan-level restatement of SC-004's acceptance
/// text ("the strided-view fixture runs on PTX without the extra copy ... the counter reads one more
/// copy per view") as a planning-time assertion instead of a pod-measured native copy count: every
/// view the planner promotes (`Plan::View`, no dispatch) is one fewer
/// materialize-copy dispatch than an unpromoted plan would need, so an equal promoted-dispatch count
/// between backends is exactly "PTX pays no extra copy for the views it shares with ROCm". Separate
/// from `nvptx_view_promotion_matches_amdgcn_on_real_qwen2_decode_configs` above (which checks the
/// promoted value-id SET and each one's Layout): this row isolates the one property SC-004 itself
/// names, with its own mutation record.
///
/// Mutation: restore `strided_views: false` for `Target::Nvptx` in
/// `poot-codegen/src/lib.rs::capability`. `compute_views(&g, Nvptx)` goes back to its early-out empty
/// map while `compute_views(&g, amd)` still promotes every view, so every one of those views costs
/// Nvptx one extra materialize-copy dispatch it does not pay today:
/// `dispatch_count_planned(&g, Nvptx)` reads higher than `dispatch_count_planned(&g, amd)` by exactly
/// the view count, and this row goes red. Reverted: GREEN.
#[test]
fn sc004_nvptx_strided_view_fixture_pays_no_extra_copy_vs_amdgcn() {
    let cap = 64usize;
    let amd = Backend::AmdGcn(AmdArch::gfx1151());

    for (name, cfg) in [("qwen2_0_5b", qwen2_0_5b()), ("qwen3_0_6b", qwen3_0_6b())] {
        let g = real_decode_graph(cfg, cap);

        let views_nvptx = compute_views(&g, Backend::Nvptx);
        assert!(
            !views_nvptx.is_empty(),
            "{name}: this fixture is known to have promotable views on Nvptx - an empty set means \
             the mutation (or a real regression) already dropped Nvptx's promotion, which is exactly \
             the failure this row exists to catch"
        );

        let n_nvptx = dispatch_count_planned(&g, Backend::Nvptx);
        let n_amd = dispatch_count_planned(&g, amd);
        assert_eq!(
            n_nvptx,
            n_amd,
            "{name}: Nvptx's planned copy (dispatch) count must equal AmdGcn's - a gap here is \
             exactly {} extra materialize-copy dispatch(es) PTX would pay per decode that ROCm does \
             not",
            views_nvptx.len()
        );
    }
}

// --- spec 132 phase 2 (view-chain composition) ---

#[test]
fn transpose_then_slice_into_binary_add_composes_both_into_views() {
    // A Transpose feeding a Slice feeding only a Binary(Add). Phase 1's one-hop restriction promoted at
    // most the Slice; the Transpose stayed a materialize copy because its consumer, the Slice, is not
    // `is_strided_capable`. Phase 2 must promote both: the Slice is a view of a view, and the chain
    // composes into one Layout per link, both relative to the original contiguous buffer.
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::f32(vec![3, 2, 4]));
    let t = bld.transpose(a, vec![2, 0, 1]); // [3,2,4] -> [4,3,2]
    let s = bld.slice(t, 1, 1, 3); // axis 1 (len 3) [1,3) -> [4,2,2]
    let y = bld.binary(GBinOp::Add, s, s);
    let g = bld.finish(y);
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

    let bld = Builder::new();
    let a = bld.constant("a", TensorType::f32(vec![3, 2, 4]));
    let t = bld.transpose(a, vec![2, 0, 1]);
    let s = bld.slice(t, 1, 1, 3);
    let y = bld.binary(GBinOp::Add, s, s);
    let g = bld.finish(y);

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
        let bld = Builder::new();
        let mut shape: Vec<usize> = (0..3)
            .map(|_| 2 + (xorshift(&mut seed) % 3) as usize)
            .collect();
        let a = bld.constant("a", TensorType::f32(shape.clone()));
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
                    cur = bld.transpose(cur, perm.clone());
                    shape = perm.iter().map(|&p| shape[p]).collect();
                }
                1 => {
                    // Slice: a random axis/range of the current shape.
                    let rank = shape.len();
                    let axis = (xorshift(&mut seed) as usize) % rank;
                    let len = shape[axis];
                    let start = (xorshift(&mut seed) as usize) % len;
                    let end = start + 1 + (xorshift(&mut seed) as usize) % (len - start);
                    cur = bld.slice(cur, axis, start, end);
                    shape[axis] = end - start;
                }
                _ => {
                    // Broadcast: prepend one new leading dim (always valid for the current shape).
                    let lead = 2 + (xorshift(&mut seed) % 3) as usize;
                    let mut new_shape = vec![lead];
                    new_shape.extend_from_slice(&shape);
                    cur = bld.broadcast(cur, new_shape.clone());
                    shape = new_shape;
                }
            }
            link_ids.push(cur.id());
        }
        let y = bld.binary(GBinOp::Add, cur, cur);
        let g = bld.finish(y);

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
                .eqns
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

/// Card 159 Inc3: at gemma4-dense's batched-shared-pool-prefill local-layer scatter shape
/// (`pool_slots=1025, Hkv=16, D=256` -> `out_numel=4,198,400`, over `SCATTER_UPDATE_CHUNK_WORK`), the
/// wgpu `ScatterUpdate` plan must split into several dispatches (kernelgen
/// `scatter_update_chunked_dt`, not the fixed-signature imported kernel) covering disjoint,
/// full-coverage element ranges, avoiding the flush_work-watchdog single-dispatch thrashing (see
/// `gemma4_prefill_kv_trace_shared_pool` in poot-models). NVPTX (no display watchdog) stays the single
/// unchunked imported dispatch at the same shape; chunking is SpirvVulkan-only.
#[test]
fn scatter_update_chunks_on_wgpu_at_gemma4_local_layer_pool_shape() {
    let pool_slots = 1025;
    let hkv = 16;
    let d = 256;
    let l = 49; // a real prompt length (card 159's repro)
    let out_numel = pool_slots * hkv * d;

    let b = Builder::new();
    let base = b.constant("base", TensorType::f32(vec![pool_slots, hkv, d]));
    let src = b.constant("src", TensorType::f32(vec![l, hkv, d]));
    let inv = b.constant("inv", TensorType::new(vec![pool_slots], GDt::I32));
    let out = b.scatter_update(base, src, inv);
    let g = b.finish(out);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::ScatterUpdate))
        .unwrap();
    let out_shape = g.aval(eqn.out).shape.clone();
    assert_eq!(numel(&out_shape), out_numel);

    let plan = plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap();
    let chunks = plan.dispatches();
    assert!(
        chunks.len() > 1,
        "large wgpu scatter should split into >1 dispatch: {plan:?}"
    );
    // dispatches must be disjoint and fully cover out_numel (the split-plan contract).
    let covered: usize = chunks
        .iter()
        .map(|c| {
            c.launch().grid[0] as usize / c.body().workgroup_size[0] as usize * SCATTER_UPDATE_WG
        })
        .sum::<usize>()
        .min(out_numel + (chunks.len() - 1) * SCATTER_UPDATE_WG); // last chunk may over-launch to a wg boundary
    assert!(covered >= out_numel, "chunks must cover the full output");
    // A small pool (well under the chunk budget) stays the single dispatch. card 525: plan_eqn
    // can no longer be handed a shape that disagrees with the graph, so this needs its own small
    // ScatterUpdate graph rather than replaying the large one under a fake `[4, 3]` shape.
    let small_b = Builder::new();
    let small_base = small_b.constant("base", TensorType::f32(vec![4, 3]));
    let small_src = small_b.constant("src", TensorType::f32(vec![2, 3]));
    let small_inv = small_b.constant("inv", TensorType::new(vec![4], GDt::I32));
    let small_out = small_b.scatter_update(small_base, small_src, small_inv);
    let small_g = small_b.finish(small_out);
    let small_eqn = small_g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::ScatterUpdate))
        .unwrap();
    assert_eq!(numel(&small_g.aval(small_eqn.out).shape), 4 * 3);
    let whole_scatter = |choice: &KernelChoice| {
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Movement(kg::MovementSpec {
                op: kg::MovementOp::ScatterRows { window: None, .. },
                ..
            }))
        )
    };
    match planned_with_choice(&small_g, small_eqn, Backend::SpirvVulkan) {
        Ok((compute_plan, choice)) if is_plain(&compute_plan) => {
            assert!(whole_scatter(&choice), "{choice:?}")
        }
        other => {
            panic!("small wgpu ScatterUpdate should be the single whole dispatch: {other:?}")
        }
    }
    // NVPTX has no display watchdog: stays the single whole dispatch even at the large shape.
    match planned_with_choice(&g, eqn, Backend::Nvptx).unwrap() {
        (compute_plan, choice) if is_plain(&compute_plan) => {
            assert!(whole_scatter(&choice), "{choice:?}")
        }
        _ => panic!("NVPTX ScatterUpdate should stay a single dispatch (no chunking)"),
    }
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
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
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
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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

    // gpt-oss's scale (at/under the trigger) stays the single unchunked imported dispatch.
    let (k2, n2) = (32usize, 201_088usize);
    let g2 = matmul_graph(1, k2, n2, GDt::F32);
    let eqn2 = g2
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    match planned_with_choice(&g2, eqn2, Backend::SpirvVulkan).unwrap() {
        (compute_plan, choice) if is_plain(&compute_plan) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "{choice:?}"
            )
        }
        other => panic!("gpt-oss-scale wgpu decode-GEMV should stay one dispatch: {other:?}"),
    }

    // AmdGcn has no display watchdog: stays the single unchunked imported dispatch even at BLOOM's
    // large shape (chunking is SpirvVulkan-only, as in card 163).
    let amdgcn = Backend::AmdGcn(AmdArch::gfx1151());
    match planned_with_choice(&g, eqn, amdgcn).unwrap() {
        (compute_plan, choice) if is_plain(&compute_plan) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "{choice:?}"
            )
        }
        other => panic!("AmdGcn decode-GEMV should stay one dispatch (no chunking): {other:?}"),
    }
}

/// R486-013 / SC-006: the NVPTX decode-GEMV is kept only for small output dims; `is_decode_gemv`'s
/// `out_shape[last] > 256` gate sends a larger decode projection to the imported one-thread-per-output
/// fallback. The fixture prints the branch value (`N` and the 256 threshold). Raising the threshold
/// past 300 makes `N = 300` select the coalesced GEMV and the assertion fails; lowering it makes the
/// `N = 256` boundary case fail.
#[test]
fn nvptx_decode_gemv_excludes_output_dims_above_256() {
    for (n, expect_gemv) in [(256usize, true), (300usize, false)] {
        let g = matmul_graph(1, 64, n, GDt::F32);
        let choice = matmul_choice(&g, Backend::Nvptx);
        eprintln!(
            "nvptx decode matmul: N={n}, gemv_n_threshold=256, gemv_eligible={expect_gemv}, choice={choice:?}"
        );
        assert_eq!(
            is_imported(&choice, ImportedKernel::GemvCoalesced),
            expect_gemv,
            "N={n} on NVPTX: decode-GEMV eligibility must be {expect_gemv}, got {choice:?}"
        );
    }
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
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
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
        &poot_test_util::graph_fixtures::roomy_body_limits(),
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
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "{choice:?}"
            )
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
        let bld = Builder::new();
        let operand =
            |name: &str, shape: Vec<usize>| bld.constant(name, TensorType::new(shape, GDt::F32));
        let q = operand("q", vec![self.batch, self.hq, self.lq, self.d]);
        let k = operand("k", vec![self.batch, hkv, self.lk, self.d]);
        let v = operand("v", vec![self.batch, hkv, self.lk, self.d]);
        let mask = operand("mask", vec![self.batch, mask_heads, self.lq, self.lk]);
        let mut plan = bld.append_plan(0);
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
        bld.finish(committed)
    }
}

fn attention_planned(
    fixture: AttentionFixture,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<(Plan, KernelChoice), PlanError> {
    let g = fixture.graph();
    let eqn = g
        .eqns
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
/// memory: `4 * (lanes * dv + 2 * lanes)` bytes. The same decode plans 32, 16, 8 and 4 lanes at value dim 64,
/// 128, 256 and 512 on the wgpu cap (16352 bytes), and more on the larger ROCm and PTX caps. A fixture
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
        let kg::AttentionSchedule::Streaming { lanes } = attention_spec(&choice).schedule;
        lanes
    };
    let wgpu = Backend::SpirvVulkan;
    assert_eq!(
        [64, 128, 256, 512].map(|d| lanes(wgpu, d)),
        [32, 16, 8, 4],
        "16352 bytes of workgroup memory"
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

/// The plan records the implemented streaming schedule and the exact resources it needs, over a fixed bank:
/// short and long prefill, low- and high-head decode, and a D = 256 softcapped prefill. Per row the request
/// names `Streaming { lanes }`, the dispatch body states `lanes * (dv + 2) * 4` bytes of workgroup memory
/// and a `[lanes, 1, 1]` workgroup, and the kernel declares the serial work one thread does (its share of
/// the keys, twice over the head width). Finalization enforces the declaration: a device whose dispatch
/// budget is exactly that work admits the equation, one step less refuses with `DispatchWork` naming it.
/// Mutation: drop the attention arm of `KernelRequest::requirements` (an under-declaration), or bypass
/// resource finalization in the planner; the one-step-short row is admitted and goes red.
#[test]
fn the_streaming_schedule_records_its_exact_resources_and_finalization_enforces_them() {
    // (fixture, backend, lanes, workgroup-memory bytes, threads, steps per thread), from the shapes by hand.
    let wgpu = Backend::SpirvVulkan;
    let rows = [
        (
            AttentionFixture::prefill("short prefill", 8, 4, 16, 64),
            wgpu,
            32usize,
            8448usize,
            8 * 16 * 32,
            16u64.div_ceil(32) * 2 * 64,
        ),
        (
            AttentionFixture::prefill("long prefill", 8, 4, 512, 128),
            wgpu,
            16,
            8320,
            8 * 512 * 16,
            512u64.div_ceil(16) * 2 * 128,
        ),
        (
            AttentionFixture::decode("low-head decode", 1, 2, 1, 1024, 128),
            wgpu,
            16,
            8320,
            2 * 16,
            1024u64.div_ceil(16) * 2 * 128,
        ),
        (
            AttentionFixture::decode("high-head decode", 4, 32, 4, 2048, 128),
            Backend::AmdGcn(AmdArch::gfx1151()),
            64,
            4 * (64 * 128 + 2 * 64),
            4 * 32 * 64,
            2048u64.div_ceil(64) * 2 * 128,
        ),
        (
            AttentionFixture::prefill("softcap prefill D=256", 4, 2, 24, 256).softcapped(50.0),
            Backend::Nvptx,
            32,
            4 * (32 * 256 + 2 * 32),
            4 * 24 * 32,
            24u64.div_ceil(32) * 2 * 256,
        ),
    ];
    for (fixture, backend, lanes, lds_bytes, threads, steps) in rows {
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
        let KernelChoice::Generated(request) = &choice else {
            unreachable!("asserted generated above")
        };
        let work = kg::SerialWork {
            threads: threads as u64,
            steps_per_thread: steps,
        };
        assert_eq!(
            request.requirements().serial_work,
            Some(work),
            "{}: declared serial work",
            fixture.name
        );

        let budgeted = |budget: u64| DeviceCaps {
            max_dispatch_work: Queried::Known(budget),
            ..caps
        };
        attention_planned(fixture, backend, &budgeted(work.total()))
            .unwrap_or_else(|e| panic!("{}: the exact budget admits: {e:?}", fixture.name));
        let refusal = refusal_of(
            attention_planned(fixture, backend, &budgeted(work.total() - 1)).map(|(plan, _)| plan),
            fixture.name,
        );
        assert!(
            matches!(
                refusal.missing,
                Capability::KernelResources(ResourceRefusal::DispatchWork { work: w, .. })
                    if w == work.total()
            ),
            "{}: one step short must refuse naming {} steps, got {:?}",
            fixture.name,
            work.total(),
            refusal.missing
        );
    }
}

/// Defense in depth: `Builder::arg_top_k`'s shape inference (`OpKind::ArgTopK::infer`)
/// rejects a 0-D rank operand at graph construction, so the normal Builder API cannot reach this. But
/// `plan_eqn` trusts the graph's declared avals, so a hand-rolled or mis-transformed `Graph` could reach
/// this arm with a 0-D rank operand; it must return a typed `PlanError`, not panic.
#[test]
fn arg_top_k_plan_rejects_0d_rank_operand_as_typed_error_not_panic() {
    let bld = Builder::new();
    let rank = bld.constant("rank", TensorType::f32(vec![4, 8]));
    let idx = bld.arg_top_k(rank, 2);
    let mut g = bld.finish(idx);
    g.values[rank.id()].aval = TensorType::f32(vec![]); // corrupt the declared shape to 0-D
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::ArgTopK { .. }))
        .expect("an ArgTopK eqn");
    match plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    ) {
        Err(PlanError::BadShape(msg)) => assert!(msg.contains("ArgTopK"), "{msg}"),
        other => panic!("expected PlanError::BadShape, got {other:?}"),
    }
}

/// Same defense in depth as the `ArgTopK` test above, for `PackI8`'s 0-D input guard.
#[test]
fn pack_i8_plan_rejects_0d_input_as_typed_error_not_panic() {
    let bld = Builder::new();
    let codes = bld.constant("codes", TensorType::f32(vec![4, 8]));
    let packed = bld.pack_i8(codes);
    let mut g = bld.finish(packed);
    g.values[codes.id()].aval = TensorType::f32(vec![]); // corrupt the declared shape to 0-D
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::PackI8))
        .expect("a PackI8 eqn");
    match plan_eqn(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    ) {
        Err(PlanError::BadShape(msg)) => assert!(msg.contains("PackI8"), "{msg}"),
        other => panic!("expected PlanError::BadShape, got {other:?}"),
    }
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
    use poot_graph_ir::op::UnOp as GUnOp;
    let bld = Builder::new();
    let over_ceiling = 65_535 * 256 + 256; // one workgroup past the ceiling
    let x = bld.constant("x", TensorType::new(vec![over_ceiling], GDt::BF16));
    let g = bld.finish(x);
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

    // Every data-movement body is dtype-generic, so a BF16 ScatterUpdate past the ceiling folds like any
    // other movement, and one below it stays on the flat launch.
    let scatter_eqn = Eqn {
        op: OpKind::ScatterUpdate,
        inputs: vec![],
        out: x.id(),
        layer: None,
    };
    assert!(
        is_elementwise_2d(
            &g,
            Backend::SpirvVulkan,
            &scatter_eqn,
            &out_shape,
            over_ceiling
        ),
        "a BF16 ScatterUpdate past the ceiling must 2-D-fold"
    );
    assert!(!is_elementwise_2d(
        &g,
        Backend::SpirvVulkan,
        &scatter_eqn,
        &small_shape,
        256
    ));
}

/// Find the named const's value id in `g`.
fn const_id(g: &Graph, name: &str) -> ValueId {
    g.values
        .ids()
        .find(|&id| g.meta(id).name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("const {name} not found"))
}

/// Card 298, Card 1011: the operand shape a natively-bf16 safetensors checkpoint traces for every projection
/// matmul: an F32 activation against a BF16 weight const, F32 output. `plan_eqn` rejects it on wgpu and
/// ROCm with "has no non-tensor-core kernel"; the 16-alignment named in that message is misleading: olmo2-1b's
/// prefill shape (M=128, K=2048, N=8192, every dim 16-aligned) is still rejected, because both
/// tensor-core gates require both operands bf16 and the activation is F32.
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

        let gp = widen_mismatched_matmul_dtypes(&g, backend);
        assert_eq!(
            gp.aval(const_id(&gp, "w")).dtype,
            GDt::BF16,
            "the BF16 weight const keeps its stored dtype on {backend:?}"
        );
        assert_eq!(
            gp.eqns
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
        let gp = widen_mismatched_matmul_dtypes(&g, backend);
        let r = try_matmul_plan(&gp, backend);
        assert!(
            matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
            "decode-shaped mixed matmul must be refused on {backend:?}: got {r:?}"
        );
        assert!(
            !gp.eqns.iter().any(|e| matches!(e.op, OpKind::Cast { .. })),
            "decode-shaped mixed matmul must not gain a Cast on {backend:?}"
        );
        assert_eq!(
            gp.aval(const_id(&gp, "w")).dtype,
            GDt::BF16,
            "the refused decode weight must stay BF16 on {backend:?}"
        );
    }
}

/// Card 298: the widening must not touch an AMD WMMA operand pair. `to_mixed_bf16` emits bf16 x
/// bf16 -> f32 at 16-aligned dims, ROCm's binder uploads those consts as two-byte bf16, and the
/// tensor-core arm reads them that way.
#[test]
fn card298_widen_keeps_amd_wmma_operands_narrow() {
    let gm = mixed_matmul_graph(GDt::BF16, GDt::F32);
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let gp = widen_mismatched_matmul_dtypes(&gm, backend);
    for name in ["a", "w"] {
        assert_eq!(
            gp.aval(const_id(&gp, name)).dtype,
            GDt::BF16,
            "WMMA operand {name} must stay BF16 on {backend:?}"
        );
    }
    assert_eq!(gp.eqns.len(), gm.eqns.len(), "no eqns added");
    assert!(
        is_tensor_core(&matmul_choice(&gp, backend)),
        "a widened WMMA-eligible matmul must still take the tensor-core arm"
    );
}

/// Card 298, Card 1011: SPIR-V has no bf16 compute (RADV has no `SPV_KHR_bfloat16`), so even a bf16 x bf16
/// pair no tensor-core gate admits is widened by explicit casts that read the packed lanes. The consts keep
/// their BF16 dtype; nothing reaches codegen as a BF16 matmul.
#[test]
fn card298_spirv_widens_bf16_pairs_with_casts_not_retypes() {
    let gm = mixed_matmul_graph(GDt::BF16, GDt::F32);
    let gp = widen_mismatched_matmul_dtypes(&gm, Backend::SpirvVulkan);
    for name in ["a", "w"] {
        assert_eq!(
            gp.aval(const_id(&gp, name)).dtype,
            GDt::BF16,
            "const {name} keeps its stored dtype"
        );
    }
    assert!(
        gp.eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Cast { to: GDt::F32 })),
        "the matmul reads F32 through the widening casts"
    );
    assert!(try_matmul_plan(&gp, Backend::SpirvVulkan).is_ok());
}

/// The Qwen3.8-27B LM head shape after `fold_dense_contractions`: a BF16 `[n, k]` checkpoint const
/// contracted directly, with no transpose or widening cast.
fn dense_contraction_graph(m: usize, k: usize, n: usize) -> Graph {
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::new(vec![1, m, k], GDt::F32));
    let w = bld.constant("w", TensorType::new(vec![n, k], GDt::BF16));
    let mm = bld.matmul(a, bld.transpose(w, vec![1, 0]));
    crate::fold_dense_contractions(&bld.finish(mm))
}

fn try_dense_contraction_plan_planned(
    g: &Graph,
    backend: Backend,
) -> Result<(Plan, KernelChoice), PlanError> {
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::DenseContraction { .. }))
        .expect("a dense contraction eqn");
    planned_with_choice(g, eqn, backend)
}

/// Card 380 FR-009, narrowed by cards 450 ROCm and 450 PTX; the body is Card 1024's. The lowering is scoped by
/// name to the backends that have a typed packed walk: wgpu's (card 380), ROCm's (card 450 ROCm) and PTX's
/// (card 450 PTX). Above M == 1 all three plan the same generated `[N, K]` Tiled body over the packed-BF16 lanes.
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
        let KernelChoice::Generated(KernelRequest::Contraction(
            request @ ContractionSpec::DenseTiled { weight, .. },
        )) = choice
        else {
            panic!("{backend:?}: expected the generated dense Tiled body, got {choice:?}");
        };
        assert_eq!(weight, poot_target::BufferStorage::bf16_packed());
        format!("{request:?}")
    })
    .collect();
    assert!(
        requests.windows(2).all(|pair| pair[0] == pair[1]),
        "{requests:?}"
    );
}

/// Card 727: at M == 1 every backend plans the same generated dense Gemv request over the packed-BF16
/// `[N, K]` weight: one selection and one identical request on every backend. Host-side plan
/// evidence only; the device rows are the dense-Gemv parity rows in `poot-gpu` and `poot-rocm-gpu`.
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
        let KernelChoice::Generated(KernelRequest::Contraction(
            request @ ContractionSpec::DenseGemv { weight, .. },
        )) = choice
        else {
            panic!("{backend:?}: expected the generated dense Gemv, got {choice:?}");
        };
        assert_eq!(
            weight,
            poot_target::BufferStorage::bf16_packed(),
            "{backend:?}"
        );
        format!("{request:?}")
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
    let prepared = widen_mismatched_matmul_dtypes(&folded, Backend::SpirvVulkan);
    let weight = const_id(&prepared, "w");
    assert_eq!(
        prepared.aval(weight).dtype,
        GDt::BF16,
        "a contraction weight must stay BF16 on wgpu: the typed walk uploads it as packed u32 lanes"
    );
    assert!(
        !prepared
            .eqns
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
    let widened = widen_mismatched_matmul_dtypes(&mixed, Backend::SpirvVulkan);
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
        widen_mismatched_matmul_dtypes(&decode_mixed, Backend::SpirvVulkan,)
            .aval(const_id(&decode_mixed, "w"))
            .dtype,
        GDt::BF16
    );

    // A widening `Cast` is a packed reader (Card 1011); a second consumer that is not one disqualifies the
    // const: that consumer would read f32 lanes.
    let mut shared = folded.clone();
    let widened = shared.values.push(poot_graph_ir::graph::ValueMeta::new(
        TensorType::new(vec![16, 32], GDt::F32),
        Storage::Device,
        None,
    ));
    shared.eqns.push(poot_graph_ir::Eqn {
        op: OpKind::Cast { to: GDt::F32 },
        inputs: vec![poot_graph_ir::Operand::Value(weight)],
        out: widened,
        layer: None,
    });
    assert!(bf16_const_feeds_only_packed_readers(
        &shared,
        &Default::default(),
        weight
    ));
    let negated = shared.values.push(poot_graph_ir::graph::ValueMeta::new(
        TensorType::new(vec![16, 32], GDt::BF16),
        Storage::Device,
        None,
    ));
    shared.eqns.push(poot_graph_ir::Eqn {
        op: OpKind::Unary(poot_graph_ir::UnOp::Neg),
        inputs: vec![poot_graph_ir::Operand::Value(weight)],
        out: negated,
        layer: None,
    });
    assert!(!bf16_const_feeds_only_packed_readers(
        &shared,
        &Default::default(),
        weight
    ));

    // Only the weight operand is admitted. A const read as a contraction's activation is an f32 operand of
    // that kernel, so keeping it narrow would feed two-byte words to an `&[f32]` param.
    let mut as_activation = folded.clone();
    let out = as_activation
        .values
        .push(poot_graph_ir::graph::ValueMeta::new(
            TensorType::new(vec![32, 16], GDt::F32),
            Storage::Device,
            None,
        ));
    as_activation.eqns.push(poot_graph_ir::Eqn {
        op: OpKind::DenseContraction { weight: GDt::BF16 },
        inputs: vec![
            poot_graph_ir::Operand::Value(weight),
            poot_graph_ir::Operand::Value(weight),
        ],
        out,
        layer: None,
    });
    assert!(!bf16_const_feeds_only_packed_readers(
        &as_activation,
        &Default::default(),
        weight
    ));

    // A const nothing reads is not eligible: the binder would upload two-byte words no kernel can
    // interpret.
    let mut orphaned = folded.clone();
    let orphan = orphaned.values.push(poot_graph_ir::graph::ValueMeta::new(
        TensorType::new(vec![4, 4], GDt::BF16),
        Storage::Const,
        Some("orphan".into()),
    ));
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
    let bld = Builder::new();
    let table = bld.constant("embed", TensorType::new(vec![v, r], source));
    let token = bld.slot(
        poot_graph_ir::Slot::Token,
        TensorType::new(vec![rows], GDt::I32),
    );
    let widened = bld.cast(bld.gather(table, 0, token), GDt::F32);
    crate::fold_dense_bf16_row_gathers(&bld.finish(widened))
}

fn try_dense_row_gather_plan_planned(
    g: &Graph,
    backend: Backend,
) -> Result<(Plan, KernelChoice), PlanError> {
    let eqn = g
        .eqns
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
        g.eqns.iter().any(|e| matches!(
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
    let prepared = widen_mismatched_matmul_dtypes(&folded, Backend::SpirvVulkan);
    assert_eq!(
        prepared.aval(table).dtype,
        GDt::BF16,
        "an embedding table must stay BF16 on wgpu: the typed walk uploads it as packed u32 lanes"
    );
    assert!(
        !prepared
            .eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Cast { .. })),
        "the fold plus the narrowed retype leave no widening cast"
    );

    // The un-folded traced shape (a plain `Gather` of the BF16 table): the table is not a packed-lane
    // reader's operand, and it is never retyped to F32 (Card 1011). The storage plan refuses the gather's
    // BF16-element body against the packed lane (`bf16_const_reads_refuse_a_reader_without_the_lane`).
    let bld = Builder::new();
    let unfolded_table = bld.constant("embed", TensorType::new(vec![8, 4], GDt::BF16));
    let token = bld.slot(
        poot_graph_ir::Slot::Token,
        TensorType::new(vec![3], GDt::I32),
    );
    let unfolded_widened = bld.cast(bld.gather(unfolded_table, 0, token), GDt::F32);
    let unfolded = bld.finish(unfolded_widened);
    let id = const_id(&unfolded, "embed");
    assert!(!bf16_const_feeds_only_packed_readers(
        &unfolded,
        &Default::default(),
        id
    ));
    let prepared = widen_mismatched_matmul_dtypes(&unfolded, Backend::SpirvVulkan);
    assert_eq!(prepared.aval(id).dtype, GDt::BF16);
    // Only the table operand is admitted for this operation.
    let mut as_index = folded.clone();
    let out = as_index.values.push(poot_graph_ir::graph::ValueMeta::new(
        TensorType::f32(vec![8, 4]),
        Storage::Device,
        None,
    ));
    as_index.eqns.push(poot_graph_ir::Eqn {
        op: OpKind::DenseRowGather { source: GDt::BF16 },
        inputs: vec![
            poot_graph_ir::Operand::Value(table),
            poot_graph_ir::Operand::Value(table),
        ],
        out,
        layer: None,
    });
    assert!(!bf16_const_feeds_only_packed_readers(
        &as_index,
        &Default::default(),
        table
    ));
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
        let bld = Builder::new();
        let table = bld.constant("embed", TensorType::f32(vec![8, 4]));
        let token = bld.slot(
            poot_graph_ir::Slot::Token,
            TensorType::new(vec![3], GDt::I32),
        );
        let gathered = bld.gather(table, 0, token);
        bld.finish(gathered)
    };
    let eqn = f32_gather
        .eqns
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
/// or disk artifact with an f32-operand kernel of the same shapes (the card-128 Bug-1 class the
/// tensor-core keys guard against).
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
        let bld = Builder::new();
        let a = bld.constant("a", TensorType::new(vec![1, 1, 32], GDt::F32));
        let w = bld.constant("w", TensorType::new(vec![32, 16], GDt::F32));
        let mm = bld.matmul(a, w);
        bld.finish(mm)
    };
    let f32_key = one_dispatch(try_matmul_plan(&f32_matmul, Backend::SpirvVulkan).unwrap())
        .key()
        .to_string();
    assert_ne!(key, f32_key);
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
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::DenseContraction { .. }))
            .expect("a dense contraction eqn");
        let out_shape = g.aval(eqn.out).shape.clone();
        let out_numel = numel(&out_shape);
        assert_eq!(out_numel, n, "M == 1: one row of {n} outputs");
        let schedule =
            dense_decode_gemv_in(eqn, &out_shape, out_numel, &caps).unwrap_or_else(|| {
                panic!("{backend:?}: the dense Gemv must take the M == 1 contraction")
            });
        let DenseGemvLaunch {
            width,
            cols,
            unroll,
        } = dense_gemv_launch(n, caps.compute_units);
        assert_eq!(
            schedule,
            poot_kernelgen::Schedule::Gemv {
                width,
                cols,
                unroll
            }
        );

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

    // M > 1 runs the generated `[N, K]` Tiled body on every backend: no Gemv, no imported kernel.
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let g = dense_contraction_graph(2, k, n);
        let eqn = g
            .eqns
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
                KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled {
                    m: 2,
                    weight,
                    ..
                })) if *weight == poot_target::BufferStorage::bf16_packed()
            ),
            "{backend:?}: M > 1 must plan the generated dense Tiled body: {choice:?}"
        );
    }
}

/// Card 380 P1. `validate_wgpu_tensor_execution` is the fail-closed half of the packed-BF16 contract: the
/// typed value walk stores BF16 as packed u32 lanes and checks every plan against that, and every other
/// wgpu entry point (which binds `Arc<[f32]>` tensors) rejects a BF16 value by name instead of uploading
/// f32 words for a kernel that reads packed lanes.
///
/// The device test `wgpu_resident_kv_rejects_a_bf16_contraction_weight` drives the same rejection through
/// the executor contract; this one covers it without a device.
#[test]
fn card380_wgpu_tensor_execution_rejects_every_bf16_value() {
    // The card 380 shape: the contraction weight survives preparation as BF16, safe only on the typed walk.
    let prepared =
        widen_mismatched_matmul_dtypes(&dense_contraction_graph(1, 32, 16), Backend::SpirvVulkan);
    let weight = const_id(&prepared, "w");
    assert_eq!(prepared.aval(weight).dtype, GDt::BF16);
    let error = validate_wgpu_tensor_execution(&prepared)
        .expect_err("a BF16 value must be rejected on the wgpu tensor lane");
    assert!(
        matches!(
            error,
            PlanError::UnrepresentableValue {
                value,
                dtype: GDt::BF16,
                gap: StorageGap::Bf16PackedLanes,
            } if value == weight
        ),
        "the rejection must name bf16 and the value: {error:?}"
    );

    // The other way a BF16 value reaches a prepared wgpu graph: a plain gather of a BF16 table, which would
    // otherwise fail in SPIR-V codegen with an opaque message.
    let bld = Builder::new();
    let table = bld.constant("table", TensorType::new(vec![4, 3], GDt::BF16));
    let index = bld.constant("index", TensorType::new(vec![2], GDt::I32));
    let gathered = bld.gather(table, 0, index);
    let widened = widen_mismatched_matmul_dtypes(&bld.finish(gathered), Backend::SpirvVulkan);
    assert!(
        !widened
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::Cast { to: GDt::BF16 })),
        "the table is not retyped, so nothing casts it back to BF16"
    );
    assert!(
        matches!(
            validate_wgpu_tensor_execution(&widened)
                .expect_err("a BF16 table must be rejected too"),
            PlanError::UnrepresentableValue {
                dtype: GDt::BF16,
                gap: StorageGap::Bf16PackedLanes,
                ..
            }
        ),
        "the BF16 table must be rejected for being BF16, not for some other reason"
    );

    // An all-F32 graph is untouched.
    let f32_only =
        widen_mismatched_matmul_dtypes(&matmul_graph(1, 32, 16, GDt::F32), Backend::SpirvVulkan);
    validate_wgpu_tensor_execution(&f32_only).expect("an all-F32 prepared graph still executes");

    // Scoped to wgpu: the shared admission path PTX and ROCm also call still accepts the same BF16
    // graph, because both have narrow BF16 storage.
    validate_graph_execution(&prepared)
        .expect("PTX and ROCm must not be gated by a wgpu limitation");
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
        .eqns
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
    let bld = Builder::new();
    let scores = bld.constant("scores", TensorType::f32(vec![1, hq, 1, cap]));
    let v = bld.constant("v", TensorType::f32(vec![1, hq, cap, d]));
    let out = bld.matmul(scores, v);
    bld.finish(out)
}

/// Card 727 SC-005, plan side: a contraction plan whose body sizes its LDS reduction by its baked
/// width keeps that width through the planner's launch shaping, by its request type. The fixture is the decode
/// attention GEMV (`GEMV_WIDTH` lanes) on a device whose X grid holds a single workgroup, so the generic bump
/// (raise a one-thread-per-output body to 256 lanes when `out_numel / width` passes the X cap) would fire; the
/// plan keeps `GEMV_WIDTH` and an LDS array of exactly that many lanes. The device half (the result equals the
/// oracle) is `a_contraction_keeps_its_lds_width_through_the_planner_*` in `poot-gpu` and `poot-rocm-gpu`.
///
/// Mutation: in `kernel_mapping::shape_launch` drop the request-type exemption (`contraction = false`); the
/// workgroup becomes 256 and this row goes red.
#[test]
fn a_contraction_plan_keeps_its_lds_width_through_the_planner() {
    let g = attn_scores_v_graph(4, 64, 64);
    let eqn = &g.eqns[0];
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1, 1, k], GDt::F32));
    let out = if transposed {
        let w = b.constant("w", TensorType::new(vec![n, k], GDt::BF16));
        b.matmul(x, b.transpose(w, vec![1, 0]))
    } else {
        let w = b.constant("w", TensorType::new(vec![k, n], GDt::BF16));
        b.matmul(x, w)
    };
    let graph = b.finish(out);
    let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    let target = Target {
        backend,
        caps: default_caps_for(backend),
    };
    let program = compile(&graph, &target, &options)?;
    let mut contractions = program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::MatMul | OpKind::DenseContraction { .. }));
    let (eqn, _) = contractions
        .next()
        .unwrap_or_else(|| panic!("{backend:?} transposed={transposed}: no contraction planned"));
    assert!(contractions.next().is_none(), "one contraction");
    Ok(program.kernel_choice(eqn).clone())
}

/// POOT-739 SC-006, at planning: through `compile`, a BF16 decode projection (M == 1) whose weight is held in
/// checkpoint `[N, K]` order plans the generated dense Gemv on every backend. The same projection over a
/// `[K, N]` weight is a typed `DtypeLowering` refusal on SPIR-V and ROCm (the retired column-tile body was its
/// only lowering; a per-column Gemv would read `[K, N]` at a stride of `N` per lane), and on NVPTX it plans a
/// non-Gemv body. Mutation: in `dtype_widen::matmul_bf16_column_gemv` return `false`; the weight is widened by a
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
                !matches!(
                    kn,
                    KernelChoice::Generated(KernelRequest::Contraction(
                        ContractionSpec::DenseGemv { .. } | ContractionSpec::GemvChunk { .. }
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
    let b = Builder::new();
    let a = b.slot_named(
        poot_graph_ir::Slot::Token,
        "a",
        TensorType::new(vec![2, 4], GDt::I32),
    );
    let c = b.slot_named(
        poot_graph_ir::Slot::Token,
        "c",
        TensorType::new(vec![2, 4], GDt::I32),
    );
    let w = b.constant("w", TensorType::f32(vec![2, 4]));
    let hit = b.binary(GBinOp::Ge, a, c);
    let out = b.binary(GBinOp::Mul, b.cast(hit, GDt::F32), w);
    let g = b.finish(out);
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
    Plan::Dispatches(DispatchPlan::new(rebuilt, 0).expect("the rebound plan validates"))
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

    let b = Builder::new();
    let table = b.constant("table", TensorType::new(vec![8, 3], GDt::F32));
    let index = b.constant("index", TensorType::new(vec![2], GDt::I32));
    let gathered = b.gather(table, 0, index);
    let g = b.finish(gathered);
    let backend = Backend::SpirvVulkan;
    let caps = default_caps_for(backend);
    let eqn = g
        .eqns
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

/// POOT-1115 decision 3, a deliberate tightening: an imported kernel's serial work is apportioned over
/// the dispatches of an unequal split by each launch's share of the plan, not divided evenly by the
/// dispatch count. A 4096 x 4096 F32 imported tiled GEMM on RADV splits its tile groups at the device's
/// tile ceiling into full dispatches and a one-group remainder; a full dispatch carries more than an even
/// share of the output, so its serial work is the bound.
///
/// The plan is `tiled_gemm_plan`'s (the matmul route's own constructor) through `finalize`: the compile
/// pipeline prefers the generated tiled region for this shape, so the imported route is not reached by
/// `compile` today.
///
/// Mutation: apportion by `(1, dispatches.len())` in `validate_planned_resources` (the old even split);
/// one step short of the largest dispatch's work is then admitted and the refusal below goes red.
#[test]
fn an_unequal_tile_split_apportions_serial_work_by_launch_share() {
    use crate::device_validation::ResourceRefusal;
    use crate::kernel_mapping::finalize;

    let (m, k, n) = (4096usize, 8usize, 4096usize);
    let g = matmul_graph(m, k, n, GDt::F32);
    let backend = Backend::SpirvVulkan;
    let caps = default_caps_for(backend);
    let eqn = g
        .eqns
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::MatMul))
        .expect("a matmul eqn");
    let shape = g.aval(eqn.out).shape.clone();
    let out_numel = numel(&shape);
    let split = |caps: &DeviceCaps| {
        let mut planned = crate::predicates::tiled_gemm_plan(backend, m, k, n, caps)?;
        finalize(
            &mut planned,
            GraphTables::from(&g),
            eqn,
            &shape,
            out_numel,
            backend,
            GDt::F32,
            caps,
        )
        .map(|()| planned)
    };
    let planned = split(&caps).expect("an unbudgeted device admits the split");
    let launches: Vec<u128> = planned
        .plan
        .dispatches()
        .iter()
        .map(|d| d.launch().grid.iter().map(|&e| u128::from(e)).product())
        .collect();
    let last = launches.len() - 1;
    assert!(launches.len() > 1, "the GEMM splits: {launches:?}");
    assert!(
        launches[0] > launches[last],
        "the split is unequal: {launches:?}"
    );
    let total: u128 = launches.iter().sum();
    let (out, steps) = (out_numel as u128, k as u128);
    let first_work = (out * launches[0]).div_ceil(total) * steps;
    let even_work = out.div_ceil(launches.len() as u128) * steps;
    assert!(
        first_work > even_work,
        "the tightening: {first_work} vs {even_work}"
    );

    let budgeted = |budget: u128| DeviceCaps {
        max_dispatch_work: Queried::Known(budget as u64),
        ..caps
    };
    split(&budgeted(first_work)).expect("the exact budget admits the largest dispatch");
    match split(&budgeted(first_work - 1)) {
        Err(PlanError::Refused(refusal)) => assert!(
            matches!(
                refusal.missing,
                Capability::KernelResources(ResourceRefusal::DispatchWork { work, .. })
                    if u128::from(work) == first_work
            ),
            "got {:?}",
            refusal.missing
        ),
        other => panic!(
            "one step short must refuse, got {:?}",
            other.map(|p| p.plan)
        ),
    }
}
