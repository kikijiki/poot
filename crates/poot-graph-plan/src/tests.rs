//! Unit tests for the planner: dispatch-predicate gating, cache-key stability, strided-view promotion,
//! real-config dispatch-count regressions.

use poot_graph_ir::{Builder, ShapeError, StateRole, Storage, TensorType, ValidationId};
use poot_target::{
    AmdArch, Backend, DECODE_GEMV_CHUNK_TRIGGER, DeviceCaps, FLASH_LDS_CAP, TensorCoreSupport,
    WatchdogBudget,
};
use poot_tensor::DType as GDt;
use poot_test_util::device_caps::default_caps_for;

use super::*;

/// Whole-graph exact-I32 storage query, as the planner and executors ask it.
fn value_requires_exact_i32_storage(g: &Graph, id: ValueId) -> bool {
    ExactI32StorageAnalysis::new(g).required(id)
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
                value: first.id,
                first_lane: 0,
                lane_count: 2,
            },
            ValidationPacketSource {
                value: second.id,
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

fn prior_exact_i32_storage_query(graph: &Graph, id: ValueId) -> bool {
    if graph
        .values
        .get(id)
        .is_none_or(|value| value.aval.dtype != GDt::I32)
    {
        return false;
    }
    let mut adjacency: std::collections::HashMap<ValueId, Vec<ValueId>> =
        std::collections::HashMap::new();
    for eqn in &graph.eqns {
        if graph
            .values
            .get(eqn.out)
            .is_none_or(|value| value.aval.dtype != GDt::I32)
            || !matches!(
                eqn.op,
                OpKind::Binary(_)
                    | OpKind::Unary(_)
                    | OpKind::Select
                    | OpKind::Fused(_)
                    | OpKind::Reshape { .. }
                    | OpKind::Transpose { .. }
                    | OpKind::Slice { .. }
                    | OpKind::Broadcast { .. }
                    | OpKind::Concat { .. }
            )
        {
            continue;
        }
        for input in &eqn.inputs {
            if let Operand::Value(input) = input
                && graph
                    .values
                    .get(*input)
                    .is_some_and(|value| value.aval.dtype == GDt::I32)
            {
                adjacency.entry(eqn.out).or_default().push(*input);
                adjacency.entry(*input).or_default().push(eqn.out);
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut pending = std::collections::VecDeque::from([id]);
    while let Some(value) = pending.pop_front() {
        if !seen.insert(value) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&value) {
            pending.extend(neighbors.iter().copied());
        }
    }
    let has_binary = graph.eqns.iter().any(|eqn| {
        matches!(
            eqn.op,
            OpKind::Binary(_) | OpKind::Unary(_) | OpKind::Select | OpKind::Fused(_)
        ) && (seen.contains(&eqn.out)
            || eqn
                .inputs
                .iter()
                .any(|input| matches!(input, Operand::Value(input) if seen.contains(input))))
    });
    has_binary
        && (seen.contains(&graph.output)
            || graph
                .state
                .iter()
                .any(|&(_, output)| seen.contains(&output))
            || graph.eqns.iter().any(|eqn| {
                let observed = match &eqn.op {
                    OpKind::Binary(
                        GBinOp::GeU
                        | GBinOp::RemU
                        | GBinOp::And
                        | GBinOp::Or
                        | GBinOp::Xor
                        | GBinOp::Shl
                        | GBinOp::Shr,
                    )
                    | OpKind::Unary(GUnOp::Not | GUnOp::Clz)
                    | OpKind::Select => true,
                    OpKind::Fused(region) => region.steps.iter().any(|step| {
                        matches!(
                            step.op,
                            FusedOp::Binary(
                                GBinOp::GeU
                                    | GBinOp::RemU
                                    | GBinOp::And
                                    | GBinOp::Or
                                    | GBinOp::Xor
                                    | GBinOp::Shl
                                    | GBinOp::Shr
                            ) | FusedOp::Unary(GUnOp::Not | GUnOp::Clz)
                                | FusedOp::Select
                        )
                    }),
                    _ => false,
                };
                observed
                    && (seen.contains(&eqn.out)
                        || eqn.inputs.iter().any(
                            |input| matches!(input, Operand::Value(input) if seen.contains(input)),
                        ))
            }))
}

fn assert_analysis_matches_prior(graph: &Graph, label: &str) {
    let analysis = ExactI32StorageAnalysis::new(graph);
    assert!(
        std::ptr::eq(analysis.graph(), graph),
        "{label}: analysis must borrow the supplied graph"
    );
    for id in (0..=graph.values.len()).chain([usize::MAX]) {
        let analyzed = analysis.required(id);
        assert_eq!(
            analyzed,
            prior_exact_i32_storage_query(graph, id),
            "{label}: v{id}"
        );
        if id >= graph.values.len() {
            assert!(
                !value_requires_exact_i32_storage(graph, id),
                "{label}: v{id}"
            );
        }
    }
}

fn exact_i32_component_fixture() -> Graph {
    let b = Builder::new();
    let exact_input = b.constant("exact", TensorType::new(vec![2], GDt::I32));
    let exact_moved = b.broadcast(exact_input, vec![2, 2]);
    let exact_moved = b.transpose(exact_moved, vec![1, 0]);
    let exact_moved = b.slice(exact_moved, 0, 0, 1);
    let exact_moved = b.reshape(exact_moved, vec![2]);
    let exact_moved = b.concat(0, &[exact_moved, exact_moved]);
    let exact = b.binary_scalar(GBinOp::GeU, exact_moved, Scalar::I32(i32::MIN));

    let legacy_input = b.constant("legacy", TensorType::new(vec![2], GDt::I32));
    let legacy = b.binary_scalar(GBinOp::Add, legacy_input, Scalar::I32(1));
    let table = b.constant("table", TensorType::f32(vec![4, 1]));
    let _legacy_output = b.gather(table, 0, legacy);

    let disconnected_input = b.constant("disconnected", TensorType::new(vec![1], GDt::I32));
    let _disconnected = b.binary_scalar(GBinOp::Add, disconnected_input, Scalar::I32(i32::MAX));
    b.finish(exact)
}

#[test]
fn reusable_exact_i32_analysis_matches_the_prior_component_query() {
    assert_analysis_matches_prior(&Graph::default(), "empty graph");

    let b = Builder::new();
    let identity = b.constant("identity", TensorType::new(vec![1], GDt::I32));
    assert_analysis_matches_prior(&b.finish(identity), "i32 constant identity");

    let graph = exact_i32_component_fixture();
    assert_analysis_matches_prior(&graph, "movement/GeU/legacy/disconnected");

    let b = Builder::new();
    let output_input = b.constant("output", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, output_input, Scalar::I32(1));
    let output_graph = b.finish(output);
    assert_analysis_matches_prior(&output_graph, "primary output");

    let b = Builder::new();
    let state = b.state_input(
        "state",
        TensorType::new(vec![1], GDt::I32),
        StateRole::Recurrent,
    );
    let state_output = b.binary_scalar(GBinOp::Add, state, Scalar::I32(1));
    let visible = b.constant("visible", TensorType::f32(vec![1]));
    let state_graph = b.finish_with_state(visible, &[(state, state_output)]);
    assert_analysis_matches_prior(&state_graph, "state output");

    let fixtures = [graph, output_graph, state_graph];
    for (fixture_index, fixture) in fixtures.into_iter().enumerate() {
        for bad in [fixture.values.len(), usize::MAX] {
            let mut malformed = fixture.clone();
            malformed.output = bad;
            assert_analysis_matches_prior(
                &malformed,
                &format!("fixture {fixture_index} malformed graph output {bad}"),
            );

            let mut malformed = fixture.clone();
            malformed.state.push((bad, bad));
            assert_analysis_matches_prior(
                &malformed,
                &format!("fixture {fixture_index} malformed state pair {bad}"),
            );

            for equation in 0..fixture.eqns.len() {
                let mut malformed = fixture.clone();
                malformed.eqns[equation].out = bad;
                assert_analysis_matches_prior(
                    &malformed,
                    &format!("fixture {fixture_index} malformed eqn {equation} output {bad}"),
                );

                for operand in 0..fixture.eqns[equation].inputs.len() {
                    if !matches!(fixture.eqns[equation].inputs[operand], Operand::Value(_)) {
                        continue;
                    }
                    let mut malformed = fixture.clone();
                    malformed.eqns[equation].inputs[operand] = Operand::Value(bad);
                    assert_analysis_matches_prior(
                        &malformed,
                        &format!(
                            "fixture {fixture_index} malformed eqn {equation} operand {operand} {bad}"
                        ),
                    );
                }
            }
        }

        for value in 0..fixture.values.len() {
            let mut malformed = fixture.clone();
            malformed.values[value].aval.dtype = if malformed.values[value].aval.dtype == GDt::I32 {
                GDt::F32
            } else {
                GDt::I32
            };
            assert_analysis_matches_prior(
                &malformed,
                &format!("fixture {fixture_index} malformed dtype v{value}"),
            );
        }
    }
}

/// Card 671: `plan_eqn`, the legacy single-equation wrapper this test used to contrast against, is
/// deleted (it called the per-eqn body directly, skipping `ensure_eqn`'s pointer-identity check -
/// `poot_test_util::graph_fixtures::plan_eqn` reconstructs that call shape for cross-crate test
/// callers, but every in-crate caller, this test included, now goes straight through
/// `plan_eqn_analyzed`). `ensure_eqn` rejects a foreign or cloned `Eqn` unconditionally - it checks
/// that `eqn`'s address falls inside `analysis`'s own graph's `eqns` backing storage, not whether
/// the analysis happens to be freshly built from a graph with equivalent contents - so there is no
/// longer a lenient entry point to assert compatibility with a detached equation through.
#[test]
fn analyzed_planner_rejects_a_different_graph_or_foreign_equation() {
    let graph = exact_i32_component_fixture();
    let foreign = exact_i32_component_fixture();
    let analysis = ExactI32StorageAnalysis::new(&graph);

    let foreign_eqn = &foreign.eqns[0];
    assert!(matches!(
        plan_eqn_analyzed(
            &analysis,
            &foreign,
            foreign_eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        ),
        Err(PlanError::AnalysisContextMismatch(_))
    ));

    assert!(matches!(
        plan_eqn_analyzed(
            &analysis,
            &graph,
            foreign_eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        ),
        Err(PlanError::AnalysisContextMismatch(_))
    ));

    let cloned = graph.eqns[0].clone();
    assert!(
        matches!(
            plan_eqn_analyzed(
                &analysis,
                &graph,
                &cloned,
                Backend::Nvptx,
                &default_caps_for(Backend::Nvptx),
                &poot_test_util::graph_fixtures::roomy_body_limits()
            ),
            Err(PlanError::AnalysisContextMismatch(_))
        ),
        "analyzed planning requires the stored equation, not a clone"
    );
}

#[test]
fn fresh_and_reused_storage_analysis_plan_identically() {
    let graph = exact_i32_component_fixture();
    let analysis = ExactI32StorageAnalysis::new(&graph);
    let views = compute_views(&graph, Backend::SpirvVulkan);
    for eqn in &graph.eqns {
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            assert_eq!(
                format!(
                    "{:?}",
                    plan_eqn_analyzed(
                        &ExactI32StorageAnalysis::new(&graph),
                        &graph,
                        eqn,
                        backend,
                        &default_caps_for(backend),
                        &poot_test_util::graph_fixtures::roomy_body_limits()
                    )
                ),
                format!(
                    "{:?}",
                    plan_eqn_analyzed(
                        &analysis,
                        &graph,
                        eqn,
                        backend,
                        &default_caps_for(backend),
                        &poot_test_util::graph_fixtures::roomy_body_limits()
                    )
                ),
                "{:?} on {backend:?}",
                eqn.op
            );
        }
        assert_eq!(
            format!(
                "{:?}",
                plan_eqn_views_analyzed(
                    &ExactI32StorageAnalysis::new(&graph),
                    &graph,
                    eqn,
                    Backend::SpirvVulkan,
                    1,
                    &views,
                    &default_caps_for(Backend::SpirvVulkan),
                    &poot_test_util::graph_fixtures::roomy_body_limits()
                )
            ),
            format!(
                "{:?}",
                plan_eqn_views_analyzed(
                    &analysis,
                    &graph,
                    eqn,
                    Backend::SpirvVulkan,
                    1,
                    &views,
                    &default_caps_for(Backend::SpirvVulkan),
                    &poot_test_util::graph_fixtures::roomy_body_limits()
                )
            ),
            "view plan for {:?}",
            eqn.op
        );
    }
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
    plan_eqn_choice_analyzed(
        &ExactI32StorageAnalysis::new(g),
        g,
        eqn,
        backend,
        1,
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
            dt, bias: false, ..
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
            let (Plan::Compute { body, .. }, choice) =
                planned_with_choice(&g, eqn, backend).unwrap()
            else {
                panic!("exact I32 Binary must plan as Compute");
            };
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
            let (Plan::Compute { body, .. }, choice) =
                planned_with_choice(&g, eqn, backend).unwrap()
            else {
                panic!("{:?} must plan as Compute", eqn.op);
            };
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
    let (Plan::Compute { body, .. }, choice) =
        planned_with_choice(&fused, eqn, Backend::Nvptx).unwrap()
    else {
        panic!("fused I32 DAG must plan as Compute");
    };
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
    let (Plan::Compute { body, .. }, choice) =
        planned_with_choice(&g, eqn, Backend::SpirvVulkan).unwrap()
    else {
        panic!("large I32 Select must plan as Compute");
    };
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

#[test]
fn bounded_i32_planning_remains_compatible_outside_exact_paths() {
    // The legacy helper is unchanged for index-consuming kernels. Exact integer selection is limited to
    // Binary and bit-preserving movement, so bounded position/index consumers are not globally retyped.
    assert_eq!(fty_ty(GDt::I32), Ty::F32);

    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![2], GDt::I32));
    let y = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let g = b.finish(y);
    let eqn = &g.eqns[0];
    let Plan::Compute { body, .. } = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::Nvptx,
        &default_caps_for(Backend::Nvptx),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap() else {
        panic!("I32 scalar add must compute");
    };
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

#[test]
fn i32_movement_storage_is_graph_connected_and_cache_distinct() {
    let movement_plan = |exact: bool| {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
        let moved = b.broadcast(x, vec![2]);
        let output = if exact {
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
        let (Plan::Compute { body, key, .. }, choice) =
            planned_with_choice(&g, eqn, Backend::Nvptx).unwrap()
        else {
            panic!("broadcast must compute");
        };
        (body, key, choice)
    };

    let (legacy_body, legacy_key, legacy_choice) = movement_plan(false);
    let (exact_body, exact_key, exact_choice) = movement_plan(true);
    assert_eq!(slice_element_ty(&legacy_body, 1), &Ty::F32);
    assert_eq!(slice_element_ty(&exact_body, 1), &Ty::I32);
    let broadcast_dtype = |choice: &KernelChoice| match choice {
        KernelChoice::Generated(KernelRequest::Movement(MovementSpec::Broadcast {
            dt, ..
        })) => dt.clone(),
        other => panic!("a non-f32 broadcast must be generated: {other:?}"),
    };
    assert_eq!(broadcast_dtype(&legacy_choice), Ty::F32);
    assert_eq!(broadcast_dtype(&exact_choice), Ty::I32);
    assert_ne!(legacy_key, exact_key);
}

#[test]
fn exact_and_legacy_i32_binary_cache_keys_cannot_collide() {
    let binary_plan = |exact: bool| {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![2], GDt::I32));
        let y = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
        let output = if exact {
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
        let (Plan::Compute { body, key, .. }, choice) =
            planned_with_choice(&g, eqn, Backend::Nvptx).unwrap()
        else {
            panic!("Binary must compute");
        };
        (body, key, choice)
    };
    let (exact_body, exact_key, exact_choice) = binary_plan(true);
    let (legacy_body, legacy_key, legacy_choice) = binary_plan(false);
    assert_eq!(slice_element_ty(&exact_body, 1), &Ty::I32);
    assert_eq!(slice_element_ty(&legacy_body, 1), &Ty::F32);
    assert!(
        matches!(
            &exact_choice,
            KernelChoice::Generated(KernelRequest::Elementwise(
                ElementwiseSpec::ScalarI32 { .. }
            ))
        ),
        "{exact_choice:?}"
    );
    assert!(
        matches!(
            &legacy_choice,
            KernelChoice::Generated(KernelRequest::Elementwise(
                ElementwiseSpec::ScalarFloat { .. }
            ))
        ),
        "{legacy_choice:?}"
    );
    assert_ne!(exact_key, legacy_key);
}

#[test]
fn gather_reads_exact_i32_index_chains_without_retyping_legacy_indices() {
    let gather_plan = |exact: bool, axis: usize| {
        let b = Builder::new();
        let table = if axis == 0 {
            b.constant("table", TensorType::f32(vec![4, 2]))
        } else {
            b.constant("table", TensorType::f32(vec![2, 4, 2]))
        };
        let index = b.constant("index", TensorType::new(vec![2], GDt::I32));
        let index = if exact {
            b.binary_scalar(GBinOp::Add, index, Scalar::I32(1))
        } else {
            b.broadcast(index, vec![2])
        };
        let gathered = b.gather(table, axis, index);
        // An early exact-I32 primary output keeps the later Gather in the graph and makes the shared index
        // component externally observed. The legacy case observes only the gathered float result.
        let g = b.finish(if exact { index } else { gathered });
        let eqn = g
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .unwrap();
        planned_with_choice(&g, eqn, Backend::Nvptx).unwrap()
    };

    // The index element type the request names.
    let generated_index = |choice: &KernelChoice| match choice {
        KernelChoice::Generated(KernelRequest::Movement(
            MovementSpec::GatherAxis0 { index_dt, .. } | MovementSpec::GatherAxis { index_dt, .. },
        )) => index_dt.clone(),
        other => panic!("expected a generated gather: {other:?}"),
    };
    for axis in [0, 1] {
        let (Plan::Compute { body, .. }, choice) = gather_plan(true, axis) else {
            panic!("exact-index gather must use a generated Compute body");
        };
        assert_eq!(slice_element_ty(&body, 1), &Ty::F32);
        assert_eq!(slice_element_ty(&body, 2), &Ty::I32);
        assert_eq!(generated_index(&choice), Ty::I32);

        let (legacy, choice) = gather_plan(false, axis);
        match legacy {
            Plan::Compute { body, .. } => {
                assert_eq!(slice_element_ty(&body, 2), &Ty::F32);
                // The f32-index gather is the shipped axis-0 kernel, or a generated one on the f32 lane.
                if !is_imported(&choice, ImportedKernel::GatherAxis0) {
                    assert_eq!(generated_index(&choice), Ty::F32);
                }
            }
            Plan::ComputeMeta { .. } => {
                assert!(
                    matches!(choice, KernelChoice::Imported { .. }),
                    "{choice:?}"
                );
            }
            other => panic!("legacy gather planned unexpectedly: {other:?}"),
        }
    }
}

#[test]
fn gather_over_exact_i32_data_with_no_binary_component_keeps_the_f32_lane() {
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
        let analysis = ExactI32StorageAnalysis::new(&g);
        let (body, _) = match plan_eqn_analyzed(
            &analysis,
            &g,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap()
        {
            Plan::Compute { body, key, .. } => (body, key),
            other => panic!("axis {axis}: exact I32 Gather planned as {other:?}"),
        };
        // An isolated I32 Gather has no Binary component, so the component rule keeps it on the f32 lane.
        assert_eq!(slice_element_ty(&body, 1), &Ty::F32, "axis {axis}");
    }
}

#[test]
fn legacy_i32_gather_plans_are_unchanged_and_mixed_exact_storage_rejects() {
    // With the component rule an isolated I32 Gather keeps the master f32-lane kernel and cache key.
    for (axis, table_shape) in [(0, vec![4, 2]), (1, vec![2, 4, 2])] {
        let b = Builder::new();
        let table = b.constant("table", TensorType::new(table_shape, GDt::I32));
        let index = b.constant("index", TensorType::new(vec![3], GDt::I32));
        let gathered = b.gather(table, axis, index);
        let g = b.finish(gathered);
        let eqn = &g.eqns[0];
        let (Plan::Compute { body, .. }, choice) =
            planned_with_choice(&g, eqn, Backend::SpirvVulkan).unwrap()
        else {
            panic!("legacy I32 Gather must stay a generated Compute body");
        };
        // The f32 lane: the legacy element and index types, with the table's own geometry.
        match (axis, &choice) {
            (
                0,
                KernelChoice::Generated(KernelRequest::Movement(MovementSpec::GatherAxis0 {
                    dt: Ty::F32,
                    index_dt: Ty::F32,
                    rest: 2,
                    numel: 6,
                })),
            ) => {}
            (
                1,
                KernelChoice::Generated(KernelRequest::Movement(MovementSpec::GatherAxis {
                    dt: Ty::F32,
                    index_dt: Ty::F32,
                    inner: 2,
                    axis_len: 4,
                    idx_numel: 3,
                    numel: 12,
                })),
            ) => {}
            _ => panic!("axis {axis}: {choice:?}"),
        }
        assert_eq!(slice_element_ty(&body, 1), &Ty::F32, "axis {axis} data");
        assert_eq!(slice_element_ty(&body, 2), &Ty::F32, "axis {axis} index");
    }

    // An observed Binary makes the Gather output exact while the table stays on the f32 lane.
    let b = Builder::new();
    let table = b.constant("table", TensorType::new(vec![4], GDt::I32));
    let index = b.constant("index", TensorType::new(vec![3], GDt::I32));
    let addend = b.constant("addend", TensorType::new(vec![3], GDt::I32));
    let gathered = b.gather(table, 0, index);
    let output = b.binary(GBinOp::Add, gathered, addend);
    let g = b.finish(output);
    let analysis = ExactI32StorageAnalysis::new(&g);
    assert!(analysis.required(gathered.id));
    assert!(!analysis.required(table.id));
    let eqn = g
        .eqns
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
        .unwrap();
    let error = plan_eqn_analyzed(
        &analysis,
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            PlanError::ExactI32GatherStorage {
                gather,
                data,
                data_exact: false,
                output_exact: true,
            } if gather == gathered.id && data == table.id
        ),
        "{error:?}"
    );
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&g),
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
fn exact_i32_runtime_is_gated_without_blocking_bounded_index_graphs() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![2], GDt::I32));
    let exact = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let exact_graph = b.finish(exact);
    let error = validate_graph_execution(&exact_graph).unwrap_err();
    assert!(
        matches!(
            error,
            PlanError::UnrepresentableValue {
                value,
                dtype: GDt::I32,
                gap: StorageGap::ExactI32BindReplay,
            } if value == x.id
        ),
        "{error:?}"
    );

    let b = Builder::new();
    let table = b.constant("table", TensorType::f32(vec![4, 1]));
    let pos = b.constant("pos", TensorType::new(vec![2], GDt::I32));
    let index = b.binary_scalar(GBinOp::Add, pos, Scalar::I32(1));
    let gathered = b.gather(table, 0, index);
    let legacy_graph = b.finish(gathered);
    validate_graph_execution(&legacy_graph)
        .expect("bounded I32 arithmetic used only as an index retains legacy execution");
    assert!(!value_requires_exact_i32_storage(&legacy_graph, pos.id));
}

#[test]
fn exact_i32_runtime_gate_precedes_binder_payload_choice() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let g = b.finish(output);
    for tensor in [
        poot_tensor::HostTensor::i32(vec![1], vec![7]),
        poot_tensor::HostTensor::f32(vec![1], vec![7.0]),
    ] {
        let inputs = std::collections::HashMap::from([(x.id, tensor)]);
        assert!(inputs.contains_key(&x.id));
        let error = validate_graph_execution(&g).unwrap_err();
        assert!(
            matches!(
                error,
                PlanError::UnrepresentableValue {
                    gap: StorageGap::ExactI32BindReplay,
                    ..
                }
            ),
            "{error:?}"
        );
    }
}

#[test]
fn planner_structural_boundary_rejects_invalid_binding_table_first() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let mut graph = b.finish(output);
    graph.consts.push(usize::MAX);

    assert!(matches!(
        validate_graph_execution(&graph),
        Err(PlanError::InvalidGraph(
            poot_graph_ir::GraphValidationError::ConstOutOfRange {
                value: usize::MAX,
                value_count,
            }
        )) if value_count == graph.values.len()
    ));
}

#[test]
fn exact_i32_runtime_inspection_is_total_for_a_malformed_graph() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let mut g = b.finish(output);
    g.eqns[0].inputs[0] = Operand::Value(g.values.len());

    assert!(!value_requires_exact_i32_storage(&g, g.values.len()));
    assert!(matches!(
        validate_graph_execution(&g),
        Err(PlanError::InvalidGraph(
            poot_graph_ir::GraphValidationError::EquationOperandNotDefined {
                equation: 0,
                value,
            }
        )) if value == g.values.len()
    ));
}

#[test]
fn exact_i32_planner_rejects_static_numel_overflow() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], GDt::I32));
    let output = b.binary_scalar(GBinOp::Add, x, Scalar::I32(1));
    let mut g = b.finish(output);
    g.values[output.id].aval.shape = vec![usize::MAX, 2];
    // card 525: plan_eqn now derives out_shape/out_numel from this same mutated aval itself (it can no
    // longer be told a different shape than the graph's own), so the overflow it must survive without
    // panicking is now computed inside plan_eqn, not passed in by the caller.
    let error = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
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
    let Plan::Compute { grid, .. } = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap() else {
        panic!("Binary Add must plan as Plan::Compute");
    };
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

/// Card 253: `OpKind::MatMulBias` (unlike `OpKind::MatMul`) has no mismatched-operand-dtype guard;
/// `operand_dts` appears once in `lib.rs`, inside the `MatMul` arm. Real qwen2/llama q/k/v decode
/// projections go through `MatMulBias`, so a BF16 weight against an F32 activation plans at
/// `fty(odt) = F32` with no rejection and no inserted Cast. Executors must run
/// `widen_mismatched_matmul_dtypes` before planning a mixed MatMulBias (required for PTX correctness
/// because eligible BF16 weights have two-byte backing storage). If this raw plan begins rejecting the
/// mismatch, re-audit the widening pass and all executor call sites together.
#[test]
fn matmul_bias_mismatched_dtype_is_unguarded_by_design() {
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::new(vec![1, 1, 32], GDt::F32));
    let w = bld.constant("w", TensorType::new(vec![32, 16], GDt::BF16));
    let bias = bld.constant("bias", TensorType::new(vec![16], GDt::F32));
    let mm = poot_graph_ir::ops::linear(&bld, a, w, Some(bias));
    let g = crate::passes::fuse_bias_epilogues(&bld.finish(mm));
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMulBias))
        .expect("a matmul_bias eqn");
    let r = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::Nvptx,
        &default_caps_for(Backend::Nvptx),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    );
    assert!(
        r.is_ok(),
        "MatMulBias with a mismatched BF16 weight operand now errors - re-audit every eager decode \
         call site's widening contract (see this test's doc comment): {r:?}"
    );
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
        let inverse = builder.constant("inverse", TensorType::f32(vec![4]));
        let output = builder.scatter_update(base, src, inverse);
        builder.finish(output)
    };
    let graph = build(3);
    let eqn = &graph.eqns[0];
    let (
        Plan::Compute {
            body, key, grid, ..
        },
        choice,
    ) = planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap()
    else {
        panic!("e4m3fn ScatterUpdate must materialize conservatively")
    };
    assert_eq!(body.param_count, 4);
    let KernelChoice::Generated(KernelRequest::Movement(MovementSpec::E4m3 {
        op: E4m3Movement::ScatterUpdate {
            base_shape,
            src_shape,
        },
        ..
    })) = &choice
    else {
        panic!("expected the packed E4M3FN scatter-update request: {choice:?}");
    };
    assert_eq!(base_shape, &[4, 2, 5]);
    assert_eq!(src_shape, &[3, 2, 5]);
    assert_eq!(grid, [16, 1, 1]);

    let other = build(2);
    let other_eqn = &other.eqns[0];
    let Plan::Compute { key: other_key, .. } = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&other),
        &other,
        other_eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap() else {
        panic!("e4m3fn ScatterUpdate must materialize")
    };
    assert_ne!(key, other_key, "source row count must separate plan keys");

    assert!(matches!(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap(),
        Plan::Compute { .. }
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
    let inverse = builder.constant("inverse_rank1", TensorType::f32(vec![5]));
    let output = builder.scatter_update(base, src, inverse);
    let rank1 = builder.finish(output);
    let rank1_eqn = &rank1.eqns[0];
    let (Plan::Compute { body, grid, .. }, choice) =
        planned_with_choice(&rank1, rank1_eqn, Backend::SpirvVulkan).unwrap()
    else {
        panic!("rank-1 e4m3fn ScatterUpdate must materialize")
    };
    assert_eq!(body.param_count, 4);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(MovementSpec::E4m3 {
                op: E4m3Movement::ScatterUpdate { base_shape, src_shape },
                ..
            })) if base_shape == &[5] && src_shape == &[3]
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
    let (Plan::Compute { body, grid, .. }, choice) =
        planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap()
    else {
        panic!("static e4m3fn DynamicUpdateSlice must materialize")
    };
    assert_eq!(body.param_count, 3);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(MovementSpec::E4m3 {
                op: E4m3Movement::DynamicUpdateSlice {
                    operand_shape, update_shape, axis: 1, index: 1,
                },
                ..
            })) if operand_shape == &[2, 4, 5] && update_shape == &[2, 2, 5]
        ),
        "{choice:?}"
    );
    assert_eq!(grid, [16, 1, 1]);

    let builder = Builder::new();
    let operand = builder.constant("operand", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let update = builder.constant("update", TensorType::new(vec![2, 2], GDt::E4M3FN));
    let index = builder.constant("index", TensorType::f32(vec![]));
    let output = builder.dynamic_update_slice_dyn(operand, update, index, 1);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let (Plan::Compute { body, grid, .. }, choice) =
        planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap()
    else {
        panic!("runtime e4m3fn DynamicUpdateSlice must materialize")
    };
    assert_eq!(body.param_count, 4);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(MovementSpec::E4m3 {
                op: E4m3Movement::DynamicUpdateSliceRuntime {
                    operand_shape, update_shape, axis: 1,
                },
                ..
            })) if operand_shape == &[2, 5] && update_shape == &[2, 2]
        ),
        "{choice:?}"
    );
    assert_eq!(grid, [4, 1, 1]);
    assert!(matches!(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap(),
        Plan::Compute { .. }
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
        let inverse = builder.constant("inverse", TensorType::f32(vec![pool]));
        let output = builder.scatter_update(base, src, inverse);
        let graph = builder.finish(output);
        let eqn = &graph.eqns[0];
        let refusal = refusal_of(
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(&graph),
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
    let inverse = builder.constant("inverse", TensorType::f32(vec![2]));
    let output = builder.scatter_update(base, src, inverse);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
        [GDt::E4M3FN, GDt::F32, GDt::F32],
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
            let index = builder.constant("index", TensorType::f32(vec![]));
            builder.dynamic_update_slice_dyn(operand, update, index, axis)
        };
        let graph = builder.finish(output);
        let eqn = &graph.eqns[0];
        let Plan::Compute { key, .. } = plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap() else {
            panic!("e4m3fn DynamicUpdateSlice must materialize")
        };
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
            value: operand.id,
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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

    let builder = Builder::new();
    let operand = builder.constant("operand", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let update = builder.constant("update", TensorType::new(vec![2, 2], GDt::E4M3FN));
    let index = builder.constant("index", TensorType::scalar(GDt::I32));
    let output = builder.dynamic_update_slice_dyn(operand, update, index, 1);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        ),
        "E4M3FN DynamicUpdateSlice with an I32 runtime index",
    );
    assert_eq!(refusal.missing, Capability::DtypeLowering);
    assert_eq!(
        refusal.dtypes.operands,
        [GDt::E4M3FN, GDt::E4M3FN, GDt::I32],
        "the refusal must carry the I32 index dtype"
    );
}

/// Card 529 (SC-004): a runtime `DynamicUpdateSlice` index's OWN dtype is part of the plan
/// cache key, not just the data dtype. The baseline shared one key (`dus_dyn:imported`) for an F32-index
/// and an I32-index DUS over the same F32 data, so the imported body's binding assumption for one dtype
/// could silently apply to the other.
#[test]
fn dynamic_update_slice_runtime_index_dtype_separates_the_plan_key() {
    let key_for = |index_dtype: GDt| {
        let builder = Builder::new();
        let operand = builder.constant("operand", TensorType::f32(vec![4, 5]));
        let update = builder.constant("update", TensorType::f32(vec![2, 5]));
        let index = builder.constant("index", TensorType::scalar(index_dtype));
        let output = builder.dynamic_update_slice_dyn(operand, update, index, 0);
        let graph = builder.finish(output);
        let eqn = &graph.eqns[0];
        let (Plan::ComputeMeta { key, .. }, choice) =
            planned_with_choice(&graph, eqn, Backend::SpirvVulkan).unwrap()
        else {
            panic!("F32 DynamicUpdateSlice with a runtime index must plan as ComputeMeta")
        };
        (key, choice)
    };

    let (f32_key, f32_choice) = key_for(GDt::F32);
    let (i32_key, i32_choice) = key_for(GDt::I32);
    assert_ne!(
        f32_key, i32_key,
        "an F32-index and an I32-index DUS over F32 data must not share a cache key"
    );
    // The choice records the index element type the equation binds.
    let index_of = |choice: KernelChoice| match choice {
        KernelChoice::Imported {
            kernel: ImportedKernel::DynUpdateSlice,
            index,
            ..
        } => index,
        other => panic!("expected the imported dynamic-update-slice: {other:?}"),
    };
    assert_eq!(index_of(f32_choice), Some(GDt::F32));
    assert_eq!(index_of(i32_choice), Some(GDt::I32));
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
    let Plan::Compute { body, grid, .. } = plan else {
        panic!("three-input e4m3fn Concat must materialize")
    };
    assert_eq!(shape, [2, 6, 3]);
    assert_eq!(body.param_count, 4);
    assert!(
        matches!(
            &choice,
            KernelChoice::Generated(KernelRequest::Movement(MovementSpec::E4m3 {
                op: E4m3Movement::Concat { axis: 1, in_shapes },
                ..
            })) if in_shapes.len() == 3
        ),
        "{choice:?}"
    );
    assert_eq!(grid, [12, 1, 1]);
    assert!(matches!(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap(),
        Plan::Compute { .. }
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
    let Plan::Compute { grid, .. } = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&graph),
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap() else {
        panic!("two-input e4m3fn Concat must materialize")
    };
    assert_eq!(shape, [2, 5]);
    assert_eq!(grid, [4, 1, 1]);

    let builder = Builder::new();
    let only = builder.constant("only", TensorType::new(vec![2, 5], GDt::E4M3FN));
    let output = builder.concat(1, &[only]);
    let graph = builder.finish(output);
    let eqn = &graph.eqns[0];
    assert!(matches!(
           plan_eqn_analyzed(
       &ExactI32StorageAnalysis::new(&graph),
       &graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == only.id
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
        let Plan::Compute { grid, .. } = plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap() else {
            panic!("E4M3FN Broadcast must materialize")
        };
        assert_eq!(grid, [expected_words, 1, 1]);
        assert!(matches!(
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(&graph),
                &graph,
                eqn,
                Backend::Nvptx,
                &default_caps_for(Backend::Nvptx),
                &poot_test_util::graph_fixtures::roomy_body_limits()
            )
            .unwrap(),
            Plan::Compute { .. }
        ));
        let backend = Backend::AmdGcn(AmdArch::gfx1151());
        let refusal = refusal_of(
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(&graph),
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
           plan_eqn_analyzed(
       &ExactI32StorageAnalysis::new(&graph),
       &graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == input.id
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
        let Plan::Compute { grid, .. } = plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(graph),
            graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap() else {
            panic!("E4M3FN Cast/Reshape must plan as Compute on wgpu")
        };
        if graph.aval(eqn.out).dtype == GDt::E4M3FN {
            let layout_words = shape[..shape.len() - 1].iter().product::<usize>()
                * shape.last().unwrap().div_ceil(4);
            assert_eq!(grid, [layout_words as u32, 1, 1]);
        }

        assert!(matches!(
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(graph),
                graph,
                eqn,
                Backend::Nvptx,
                &default_caps_for(Backend::Nvptx),
                &poot_test_util::graph_fixtures::roomy_body_limits()
            )
            .unwrap(),
            Plan::Compute { .. } | Plan::Alias(_)
        ));
        let backend = Backend::AmdGcn(AmdArch::gfx1151());
        let refusal = refusal_of(
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(graph),
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
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(graph),
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
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(graph),
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
    let plan = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        &small_caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap();
    let Plan::Compute { body, .. } = plan else {
        panic!("elementwise add must plan as Plan::Compute");
    };
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
    let plan = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        &small_caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap();
    let Plan::Compute { body, grid, .. } = plan else {
        panic!("elementwise add must plan as Plan::Compute");
    };
    let wg0 = body.workgroup_size[0].max(1) as usize;
    let wg1 = body.workgroup_size[1].max(1) as usize;
    // `grid` is the total thread count (card 525: `types.rs`'s `Plan::Compute` doc); the executor
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
           plan_eqn_analyzed(
       &ExactI32StorageAnalysis::new(&graph),
       &graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id
       ));
    assert!(matches!(
           plan_eqn_analyzed(
       &ExactI32StorageAnalysis::new(&graph),
       &graph,
       eqn,
       Backend::Nvptx,
       &default_caps_for(Backend::Nvptx),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id
       ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
    let Plan::Compute { grid, .. } = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&graph),
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap() else {
        panic!("nonidentity E4M3FN Transpose must materialize")
    };
    assert_eq!(grid, [5, 1, 1]);
    assert!(matches!(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap(),
        Plan::Compute { .. }
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
           plan_eqn_analyzed(
       &ExactI32StorageAnalysis::new(&graph),
       &graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id
       ));

    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![2, 1, 5], GDt::E4M3FN));
    let moved_unit_axis = builder.transpose(x, vec![1, 0, 2]);
    let graph = builder.finish(moved_unit_axis);
    let eqn = &graph.eqns[0];
    assert!(matches!(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap(),
        Plan::Compute { .. }
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
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(&graph),
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
    let Plan::Compute { grid, .. } = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&graph),
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap() else {
        panic!("E4M3FN Slice must materialize")
    };
    assert_eq!(shape, [2, 3]);
    assert_eq!(grid, [2, 1, 1]);
    assert!(matches!(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap(),
        Plan::Compute { .. }
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
           plan_eqn_analyzed(
       &ExactI32StorageAnalysis::new(&graph),
       &graph,
       eqn,
       Backend::SpirvVulkan,
       &default_caps_for(Backend::SpirvVulkan),
    &poot_test_util::graph_fixtures::roomy_body_limits()).unwrap(),
           Plan::Alias(id) if id == x.id
       ));

    let builder = Builder::new();
    let x = builder.constant("raw", TensorType::new(vec![3, 5], GDt::E4M3FN));
    let contiguous_rows = builder.slice(x, 0, 1, 3);
    let graph = builder.finish(contiguous_rows);
    let eqn = &graph.eqns[0];
    let Plan::Compute { grid, .. } = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&graph),
        &graph,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap() else {
        panic!("contiguous-row E4M3FN Slice must materialize")
    };
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
        let index = builder.constant("index", TensorType::f32(index_shape));
        let gathered = builder.gather(data, axis, index);
        let graph = builder.finish(gathered);
        let eqn = &graph.eqns[0];
        let shape = graph.aval(eqn.out).shape.clone();
        assert_eq!(shape, expected_shape);
        let Plan::Compute { grid, .. } = plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap() else {
            panic!("E4M3FN Gather must materialize on wgpu")
        };
        assert_eq!(grid, [expected_words, 1, 1]);
    }

    let builder = Builder::new();
    let data = builder.constant("raw", TensorType::new(vec![4, 5], GDt::E4M3FN));
    let index = builder.constant("index", TensorType::f32(vec![4]));
    let gathered = builder.gather(data, 0, index);
    let graph = builder.finish(gathered);
    let eqn = &graph.eqns[0];
    assert!(matches!(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::Nvptx,
            &default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits()
        )
        .unwrap(),
        Plan::Compute { .. }
    ));
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
fn e4m3fn_gather_rejects_empty_output_and_non_f32_indices_by_name() {
    let builder = Builder::new();
    let data = builder.constant("raw", TensorType::new(vec![4, 5], GDt::E4M3FN));
    let index = builder.constant("index", TensorType::f32(vec![0]));
    let gathered = builder.gather(data, 0, index);
    let graph = builder.finish(gathered);
    let eqn = &graph.eqns[0];
    let refusal = refusal_of(
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&graph),
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
                    value: 0,
                    dtype: GDt::E4M3FN,
                    gap: StorageGap::E4m3RawBytes,
                }
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

/// Like [`mixed_matmul_graph`] but with an explicit M/K/N shape (card 154's coopmat gate needs `K == 16`,
/// which the fixed K=32 shape cannot satisfy).
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
            Some(ContractionSpec::Serial { dt: Ty::F32, bias: false, shapes, .. })
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
        match planned_with_choice(g, eqn, gfx1151).unwrap().0 {
            Plan::Compute { key, .. } => key,
            other => panic!("matmul must be a plain compute: {other:?}"),
        }
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
/// weight (`Runner::load` types every native-bf16 checkpoint's projection weights as bf16, independent of
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
    // decode bf16 GEMV (NVPTX, and any non-eligible shape) rejects this graph.
    let g = mixed_operand_matmul_graph(1, 32, 16);
    // NVPTX is out of scope for the decode bf16 GEMV body (task is wgpu + ROCm; PTX has its own
    // prepare rule), so it still needs the widening Cast.
    let backend = Backend::Nvptx;
    let r = try_matmul_plan(&g, backend);
    assert!(
        matches!(&r, Err(PlanError::Refused(r)) if r.missing == Capability::DtypeLowering),
        "unwidened decode-shaped mixed matmul should be rejected on {backend:?}: got {r:?}"
    );

    let gw = widen_mismatched_matmul_dtypes(&g, backend, &default_caps_for(backend));
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

    // SpirvVulkan / AmdGcn: the decode bf16 GEMV eligibility arm accepts the mixed dtype and plans
    // the packed-u32 body with no Cast (the residency path this change adds).
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::AmdGcn(AmdArch::new("gfx942", 64)),
    ] {
        let (plan, choice) = try_matmul_plan_planned(&g, backend).unwrap_or_else(|e| {
            panic!("decode-shaped mixed matmul must plan as bf16 GEMV on {backend:?}: {e}")
        });
        assert!(
            matches!(plan, Plan::Compute { .. } | Plan::ComputeMeta { .. }),
            "expected a Compute/ComputeMeta plan, got {plan:?}"
        );
        assert!(
            is_imported(&choice, ImportedKernel::GemvCoalescedBf16),
            "decode bf16 GEMV on {backend:?}: {choice:?}"
        );
        let gw = widen_mismatched_matmul_dtypes(&g, backend, &default_caps_for(backend));
        assert!(
            !gw.eqns.iter().any(|e| matches!(e.op, OpKind::Cast { .. })),
            "bf16 GEMV-eligible matmul must not gain a widening Cast on {backend:?}"
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
        let gw = widen_mismatched_matmul_dtypes(&gm, backend, &default_caps_for(backend));
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
    // Round through bf16 up front, as a bf16 safetensors checkpoint already is (Runner::load
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
        let gw = widen_mismatched_matmul_dtypes(&g, backend, &default_caps_for(backend));
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
                Plan::Compute { .. },
                KernelChoice::Generated(KernelRequest::Elementwise(ElementwiseSpec::Cast(got))),
            )) => assert_eq!(format!("{got:?}"), format!("{want:?}"), "{in_dt:?}->{to:?}"),
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
            Ok((Plan::Compute { .. }, choice)) => assert!(
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
                Plan::Compute { .. },
                KernelChoice::Generated(KernelRequest::Elementwise(ElementwiseSpec::Cast(got))),
            )) => assert_eq!(
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
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(&g),
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
                eqn: c.id,
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
        matches!(plan, Plan::Compute { .. }),
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
    let body = match &plan {
        Plan::Compute { body, .. } | Plan::ComputeMeta { body, .. } => body,
        _ => unreachable!(),
    };
    assert_eq!(
        body.locals[2].ty,
        Ty::Ref {
            mutable: true,
            pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
        },
        "the destination must be a mutable f32 slice"
    );
}

/// Spike 562 F-9: the source of an I32->F32 cast outside an exact-I32 component lives on the f32-mirror
/// lane, whose bytes already are the F32 result, so the cast aliases it (like every other reader of that
/// component, it reads the representation `ExactI32StorageAnalysis` chose).
#[test]
fn cast_i32_to_f32_of_a_mirror_source_aliases_it() {
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
    ] {
        let plan = cast_plan(GDt::I32, GDt::F32, backend);
        assert!(
            matches!(plan, Ok(Plan::Alias(_))),
            "{backend:?}: a mirror-lane i32->f32 cast must alias, got {plan:?}"
        );
    }
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
    plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(g),
        g,
        eqn,
        backend,
        &default_caps_for(backend),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
}

/// The `Ty` of a `Plan::Compute`/`Plan::ComputeMeta` body's first non-return parameter local (`_1`, the
/// `a` operand). Asserts the actual kernel element type the `fty` bridge chose, which the cache-key
/// string alone would not catch (a bug mapped `DType::F16 -> Ty::F32` while the key still said "f16:").
fn body_param1_ty(plan: &Plan) -> Ty {
    let body = match plan {
        Plan::Compute { body, .. } | Plan::ComputeMeta { body, .. } => body,
        other => panic!("expected a Compute/ComputeMeta plan, got {other:?}"),
    };
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
    // plans. M=32 (>1) + AmdGcn under the MoE-hang guard steers past every fast path (decode-GEMV needs
    // M==1; the guard keeps the generated tiled GEMM off it, Card 557; the imported tiled and
    // batched-tiled GEMMs are SpirvVulkan/Nvptx-only for a plain MatMul) to the naive serial fallback.
    let g = matmul_graph(32, 32, 16, GDt::F32);
    let program = compiled(
        &g,
        Backend::AmdGcn(AmdArch::gfx1151()),
        FusionPolicy::MoeHangGuard,
    );
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
    // is not in the BF16/F16 wired list (MatMul/MatMulBias/Reshape/Transpose/Slice/Broadcast/Binary/
    // Unary/Gather/Concat/DynamicUpdateSlice/Reduce/Cast).
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![4, 8], GDt::F16));
    let w = b.constant("w", TensorType::new(vec![2, 8, 16], GDt::F16));
    let idx = b.constant("idx", TensorType::new(vec![4], GDt::F32));
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
    let res = plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
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
        let planned = plan_eqn_choice_analyzed(
            &ExactI32StorageAnalysis::new(&g),
            &g,
            eqn,
            backend,
            1,
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
    let planned = plan_eqn_choice_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        1,
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
    let (body, grid) = match plan {
        Plan::Compute { body, grid, .. } => (body, grid),
        other => panic!("expected Plan::Compute, got {other:?}"),
    };
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
    let grid = match plan {
        Plan::Compute { grid, .. } => grid,
        other => panic!("expected Plan::Compute, got {other:?}"),
    };
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

/// `g` compiled for `backend` under `fusion`.
fn compiled(g: &Graph, backend: Backend, fusion: FusionPolicy) -> Program {
    let target = Target {
        backend,
        caps: default_caps_for(backend),
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion,
        limits: crate::CompileLimits::STANDARD,
    };
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
    // MatMul, Card 557) gets the tiled-GEMM launch grid (`groups * GEMM_TILE^2` threads), the same grid
    // the imported tiled GEMM plans for the guarded MatMul, not the one-thread-per-output default (a ~2x
    // over-launch).
    let (m, k, n) = (64usize, 32usize, 48usize);
    let g = matmul_graph(m, k, n, GDt::F32);
    let out_numel = m * n;

    let full = compiled(&g, Backend::SpirvVulkan, FusionPolicy::Full);
    let (plan, choice) = compiled_matmul(&full);
    assert!(
        is_tiled_region(choice),
        "Full chooses the tiled GEMM: {choice:?}"
    );
    let &Plan::Compute { grid: tiled, .. } = plan else {
        panic!("the generated tiled GEMM plans as Compute, got {plan:?}")
    };
    // Under the guard the MatMul keeps `is_tiled_gemm_in`'s imported tiled-GEMM body
    // (`tiled_gemm_plan`, a `Plan::ComputeMeta`: its dims ride in a metadata buffer).
    let guarded = compiled(&g, Backend::SpirvVulkan, FusionPolicy::MoeHangGuard);
    let (plan, choice) = compiled_matmul(&guarded);
    assert!(
        !is_tiled_region(choice),
        "the guard keeps the imported GEMM: {choice:?}"
    );
    let &Plan::ComputeMeta { grid: imported, .. } = plan else {
        panic!("the imported tiled GEMM plans as ComputeMeta, got {plan:?}")
    };

    let groups = m.div_ceil(2 * GEMM_TILE) * n.div_ceil(GEMM_TILE);
    assert_eq!(
        tiled,
        [(groups * GEMM_TILE * GEMM_TILE) as u32, 1, 1],
        "the generated tiled GEMM must get the tiled-GEMM grid"
    );
    assert_eq!(
        tiled, imported,
        "and it must match the imported tiled GEMM's grid"
    );
    assert_ne!(
        tiled,
        [out_numel as u32, 1, 1],
        "must NOT be the naive one-thread-per-output fallthrough"
    );
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
    let program = compiled(&g, backend, FusionPolicy::Full);
    let (plan, choice) = compiled_matmul(&program);
    let bodies: Vec<&Body> = match plan {
        Plan::Compute { body, .. } => vec![body],
        Plan::ComputeChunks(chunks) => chunks.iter().map(|chunk| &chunk.body).collect(),
        other => panic!("the tiled GEMM plans as Compute or ComputeChunks, got {other:?}"),
    };
    assert!(
        is_tiled_region(choice)
            || matches!(choice, KernelChoice::Chunked(chunks) if chunks.iter().all(is_tiled_region)),
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

// --- Plan::Collective marker for the collective graph ops ---

/// Build a matmul graph ending in `AllReduce(Sum)` over the last axis and return that eqn's plan at the
/// given `world_size`.
fn all_reduce_plan(world_size: usize) -> Plan {
    let bld = Builder::new();
    let x = bld.constant("x", TensorType::f32(vec![2, 8])); // [M, K]
    let w = bld.constant("w", TensorType::f32(vec![8, 4])); // [K, N]
    let y = bld.all_reduce(bld.matmul(x, w), RedOp::Sum, 1);
    let g = bld.finish(y);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::AllReduce { .. }))
        .expect("the graph ends in an AllReduce eqn");
    plan_eqn_views_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::Nvptx,
        world_size,
        &HashMap::new(),
        &default_caps_for(Backend::Nvptx),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
}

/// Build a matmul graph ending in `AllGather` over the last axis and return that eqn's plan at `world_size`.
fn all_gather_plan(world_size: usize) -> Plan {
    let bld = Builder::new();
    let x = bld.constant("x", TensorType::f32(vec![2, 8])); // [M, K]
    let w = bld.constant("w", TensorType::f32(vec![8, 4])); // [K, N]
    let y = bld.all_gather(bld.matmul(x, w), 1);
    let g = bld.finish(y);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::AllGather { .. }))
        .expect("the graph ends in an AllGather eqn");
    plan_eqn_views_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::Nvptx,
        world_size,
        &HashMap::new(),
        &default_caps_for(Backend::Nvptx),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
}

/// At world_size=1 the collectives are the identity: they lower to `Plan::Alias` (FR-007), as the plain
/// `plan_eqn` entry point does (the single-GPU path).
#[test]
fn collective_world_size_1_lowers_to_alias() {
    assert!(
        matches!(all_reduce_plan(1), Plan::Alias(_)),
        "ws=1 AllReduce must stay Plan::Alias (identity)"
    );
    assert!(
        matches!(all_gather_plan(1), Plan::Alias(_)),
        "ws=1 AllGather must stay Plan::Alias (identity)"
    );
    // the default entry point (plan_eqn) is ws=1, so it must agree with plan_eqn_ws(.., 1).
    let bld = Builder::new();
    let x = bld.constant("x", TensorType::f32(vec![2, 8]));
    let w = bld.constant("w", TensorType::f32(vec![8, 4]));
    let y = bld.all_reduce(bld.matmul(x, w), RedOp::Sum, 1);
    let g = bld.finish(y);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::AllReduce { .. }))
        .unwrap();
    assert!(
        matches!(
            plan_eqn_analyzed(
                &ExactI32StorageAnalysis::new(&g),
                &g,
                eqn,
                Backend::Nvptx,
                &default_caps_for(Backend::Nvptx),
                &poot_test_util::graph_fixtures::roomy_body_limits()
            )
            .unwrap(),
            Plan::Alias(_)
        ),
        "plan_eqn (the ws=1 wrapper) must lower the collective to Plan::Alias"
    );
}

/// At world_size=2 the collectives lower to `Plan::Collective`: AllReduce carries `{AllReduce, Sum, axis}`,
/// AllGather carries `{AllGather, .., axis}`.
#[test]
fn collective_world_size_2_lowers_to_collective_marker() {
    // The AllReduce(Sum) is over the last axis of the [M, N] output (axis 1).
    match all_reduce_plan(2) {
        Plan::Collective { kind, op, axis } => {
            assert_eq!(kind, CollectiveKind::AllReduce);
            assert_eq!(op, RedOp::Sum, "row-parallel AllReduce combines with Sum");
            assert_eq!(
                axis, 1,
                "AllReduce axis is the [M,N] output's last (N) axis"
            );
        }
        _ => panic!("ws=2 AllReduce must be Plan::Collective, got a different plan"),
    }
    // The AllGather concatenates along the output N axis = the last axis.
    match all_gather_plan(2) {
        Plan::Collective { kind, axis, .. } => {
            assert_eq!(kind, CollectiveKind::AllGather);
            assert_eq!(
                axis, 1,
                "AllGather axis is the [M,N] output's last (N) axis"
            );
        }
        _ => panic!("ws=2 AllGather must be Plan::Collective, got a different plan"),
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
    let t_id = t.id;
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
    match plan_eqn_views_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        1,
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
    match plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
    {
        Plan::ComputeMeta { .. } => {}
        other => panic!("expected the unchanged ComputeMeta copy plan, got {other:?}"),
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
    let t_id = t.id;

    let views = compute_views(&g, Backend::SpirvVulkan);
    assert!(
        !views.contains_key(&t_id),
        "a Transpose feeding a MatMul must NOT be promoted to a view (FR-004)"
    );
    let move_eqn = g.eqns.iter().find(|e| e.out == t_id).unwrap();
    match plan_eqn_views_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        move_eqn,
        Backend::SpirvVulkan,
        1,
        &views,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
    {
        // exactly one Cont (the `index_remap_copy`-backed ComputeMeta).
        Plan::ComputeMeta { .. } => {}
        other => panic!("expected the unchanged materialize (ComputeMeta) plan, got {other:?}"),
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
        views.contains_key(&t.id),
        "t's only consumer (y_view's Binary) is strided-capable"
    );

    let eqn_view = g.eqns.iter().find(|e| e.out == y_view.id).unwrap();
    let eqn_plain = g.eqns.iter().find(|e| e.out == y_plain.id).unwrap();
    let (key_view, body_view) = match plan_eqn_views_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn_view,
        Backend::SpirvVulkan,
        1,
        &views,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
    {
        Plan::Compute { key, body, .. } => (key, body),
        other => panic!("expected Plan::Compute, got {other:?}"),
    };
    let (key_plain, body_plain) = match plan_eqn_views_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn_plain,
        Backend::SpirvVulkan,
        1,
        &views,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
    {
        Plan::Compute { key, body, .. } => (key, body),
        other => panic!("expected Plan::Compute, got {other:?}"),
    };
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

#[test]
fn dispatch_count_planned_regression_on_real_qwen2_decode_configs() {
    // Planner-level sibling of poot-eval's `dispatch_count_regression_on_real_qwen2_decode_configs`:
    // pins `dispatch_count_planned(_, Backend::SpirvVulkan)` (spec 132 phase 1; an eqn `compute_views`
    // promoted to a free `Plan::View` counts as 0 dispatches) on the real Qwen2.5-0.5B / Qwen3-0.6B
    // decode graphs through [`optimize`] (the pre-`compile` IR passes alone, the former
    // `transform::optimize()` pipeline). The pins are exact: a bound that sits above the real count
    // would still pass with a fusion pass (rope fusion) deleted, so any change in either direction
    // fails here and the pin is updated deliberately with the change that moved it. Weights-free: the
    // count depends only on graph structure.
    let cap = 64usize;

    // 24-layer Qwen2.5-0.5B real-config decode graph (`Model::trace`, one token), full optimize():
    // 825 planned dispatches. The dense decode body plans about twice the dispatches per layer it
    // should: POOT-1022 brings it down, and this pin moves down with that card.
    let og = optimize(&real_decode_graph(qwen2_0_5b(), cap));
    assert_eq!(
        dispatch_count_planned(&og, Backend::SpirvVulkan),
        825,
        "qwen2_0_5b real-config dispatch_count_planned moved off its pin"
    );

    // 28-layer Qwen3-0.6B real-config decode graph, full optimize(): 905 planned dispatches; see the
    // qwen2_0_5b comment above (POOT-1022 moves this pin down too).
    let og3 = optimize(&real_decode_graph(qwen3_0_6b(), cap));
    assert_eq!(
        dispatch_count_planned(&og3, Backend::SpirvVulkan),
        905,
        "qwen3_0_6b real-config dispatch_count_planned moved off its pin"
    );
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
        let plan = plan_eqn_views_analyzed(
            &ExactI32StorageAnalysis::new(g),
            g,
            eqn,
            backend,
            1,
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
/// view the planner promotes (`Plan::View`, no dispatch) is one fewer `Plan::Compute`/`ComputeMeta`
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
    let (t_id, s_id) = (t.id, s.id);

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
            link_ids.push(cur.id);
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

        let a_shape = g.aval(a.id).shape.clone();
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
/// wgpu `ScatterUpdate` plan must split into several `ComputeChunks` (kernelgen
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
    let inv = b.constant("inv", TensorType::f32(vec![pool_slots]));
    let out = b.scatter_update(base, src, inv);
    let g = b.finish(out);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::ScatterUpdate))
        .unwrap();
    let out_shape = g.aval(eqn.out).shape.clone();
    assert_eq!(numel(&out_shape), out_numel);

    match plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
    {
        Plan::ComputeChunks(chunks) => {
            assert!(
                chunks.len() > 1,
                "large wgpu scatter should split into >1 chunk"
            );
            // chunks must be disjoint and fully cover out_numel (the ComputeChunks contract).
            let covered: usize = chunks
                .iter()
                .map(|c| c.groups * SCATTER_UPDATE_WG)
                .sum::<usize>()
                .min(out_numel + (chunks.len() - 1) * SCATTER_UPDATE_WG); // last chunk may over-launch to a wg boundary
            assert!(covered >= out_numel, "chunks must cover the full output");
        }
        _ => panic!("large wgpu ScatterUpdate should be Plan::ComputeChunks"),
    }
    // A small pool (well under the chunk budget) stays the single imported dispatch. card 525: plan_eqn
    // can no longer be handed a shape that disagrees with the graph, so this needs its own small
    // ScatterUpdate graph rather than replaying the large one under a fake `[4, 3]` shape.
    let small_b = Builder::new();
    let small_base = small_b.constant("base", TensorType::f32(vec![4, 3]));
    let small_src = small_b.constant("src", TensorType::f32(vec![2, 3]));
    let small_inv = small_b.constant("inv", TensorType::f32(vec![4]));
    let small_out = small_b.scatter_update(small_base, small_src, small_inv);
    let small_g = small_b.finish(small_out);
    let small_eqn = small_g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::ScatterUpdate))
        .unwrap();
    assert_eq!(numel(&small_g.aval(small_eqn.out).shape), 4 * 3);
    match planned_with_choice(&small_g, small_eqn, Backend::SpirvVulkan) {
        Ok((Plan::Compute { .. }, choice)) => {
            assert!(
                is_imported(&choice, ImportedKernel::ScatterUpdate),
                "{choice:?}"
            )
        }
        other => {
            panic!("small wgpu ScatterUpdate should be the single imported dispatch: {other:?}")
        }
    }
    // NVPTX has no display watchdog: stays the single imported dispatch even at the large shape.
    match planned_with_choice(&g, eqn, Backend::Nvptx).unwrap() {
        (Plan::Compute { .. }, choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::ScatterUpdate),
                "{choice:?}"
            )
        }
        _ => panic!("NVPTX ScatterUpdate should stay Plan::Compute (no chunking)"),
    }
}

/// Card 258: at BLOOM's tied-`lm_head` decode-GEMV scale (`M=1, K=1024, N=250880`, the shape that hung
/// real hardware, over
/// `DECODE_GEMV_CHUNK_TRIGGER`), the wgpu decode-GEMV plan must split into several `Plan::ComputeChunks`
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

    match plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
    {
        Plan::ComputeChunks(chunks) => {
            assert!(
                chunks.len() > 1,
                "BLOOM-scale wgpu decode-GEMV should split into >1 chunk"
            );
            // every chunk's workgroup count must clear the wgpu per-dim grid cap (chunk size is capped at
            // caps.max_grid[0]; each chunk is a plain 1-D ComputeChunks grid).
            let grid_cap = default_caps_for(Backend::SpirvVulkan).max_grid[0] as usize;
            for c in &chunks {
                assert!(c.groups <= grid_cap, "chunk grid must clear the grid cap");
                assert!(
                    c.meta.is_empty(),
                    "a kernelgen chunk bakes its offset as a const, no meta buffer"
                );
            }
            // chunks must be disjoint and fully cover out_numel (one workgroup per output element, so
            // `groups` sums exactly to out_numel, with no workgroup-size rounding slack).
            let covered: usize = chunks.iter().map(|c| c.groups).sum();
            assert_eq!(
                covered, out_numel,
                "chunks must exactly cover the full output"
            );
        }
        other => panic!("BLOOM-scale wgpu decode-GEMV should be Plan::ComputeChunks: {other:?}"),
    }

    // gpt-oss's scale (at/under the trigger) stays the single unchunked imported dispatch.
    let (k2, n2) = (32usize, 201_088usize);
    let g2 = matmul_graph(1, k2, n2, GDt::F32);
    let eqn2 = g2
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    match planned_with_choice(&g2, eqn2, Backend::SpirvVulkan).unwrap() {
        (Plan::Compute { .. }, choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "{choice:?}"
            )
        }
        other => panic!("gpt-oss-scale wgpu decode-GEMV should stay Plan::Compute: {other:?}"),
    }

    // AmdGcn has no display watchdog: stays the single unchunked imported dispatch even at BLOOM's
    // large shape (chunking is SpirvVulkan-only, as in card 163).
    let amdgcn = Backend::AmdGcn(AmdArch::gfx1151());
    match planned_with_choice(&g, eqn, amdgcn).unwrap() {
        (Plan::Compute { .. }, choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "{choice:?}"
            )
        }
        other => panic!("AmdGcn decode-GEMV should stay Plan::Compute (no chunking): {other:?}"),
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
    match plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        Backend::SpirvVulkan,
        &low,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap()
    {
        Plan::ComputeChunks(chunks) => assert!(chunks.len() > 1),
        other => panic!("low watchdog trigger should chunk this shape: {other:?}"),
    }

    // High trigger (over this shape's 230_000 elements): a single unchunked dispatch, even though this
    // is the exact same backend and the exact same shape that chunked above.
    let high = caps_for(1_000_000);
    match planned_with_choice_caps(&g, eqn, Backend::SpirvVulkan, &high).unwrap() {
        (Plan::Compute { .. }, choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "{choice:?}"
            )
        }
        other => panic!("high watchdog trigger should stay a single dispatch: {other:?}"),
    }
}

/// Card 522 SC-003, Card 557 SC-001: the flash decode's kernel choice reads one shared
/// `poot_target::FLASH_LDS_CAP`. A head dim at the cap plans the synthesized region decode on every
/// target with no cap error (no plan-summary fixture has D = 256, so this row pins the boundary); one
/// above the cap, planned on `Backend::SpirvVulkan` (which has no D>cap fallback), refuses with
/// `Capability::HeadDimExceedsLdsCap`. Mutation: change the region-decode choice to
/// `d < FLASH_LDS_CAP` (`planner/attention.rs`); the D = 256 decode leaves the region decode and this
/// row goes red.
#[test]
fn flash_decode_lds_cap_boundary_is_one_planner_choice() {
    let at_cap = flash_decode_graph(poot_target::FLASH_LDS_CAP);
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let (_, choice) = flash_decode_planned(&at_cap, backend)
            .unwrap_or_else(|e| panic!("{backend:?}: head_dim == FLASH_LDS_CAP must plan: {e:?}"));
        assert!(
            matches!(
                choice,
                KernelChoice::Generated(KernelRequest::Attention(AttentionSpec::RegionDecode {
                    d,
                    ..
                })) if d == poot_target::FLASH_LDS_CAP
            ),
            "{backend:?}: head_dim == FLASH_LDS_CAP must take the region decode, got {choice:?}"
        );
    }

    let err = flash_decode_planned(
        &flash_decode_graph(poot_target::FLASH_LDS_CAP + 1),
        Backend::SpirvVulkan,
    )
    .expect_err("head_dim == FLASH_LDS_CAP + 1 must refuse on SpirvVulkan");
    assert!(
        matches!(
            &err,
            PlanError::Refused(refusal) if refusal.missing == Capability::HeadDimExceedsLdsCap {
                head_dim: poot_target::FLASH_LDS_CAP + 1,
                lds_cap: FLASH_LDS_CAP,
            }
        ),
        "expected a HeadDimExceedsLdsCap refusal, got {err:?}"
    );
}

/// A single-sequence (B=1) `FlashAttentionDecode` with head dim `d`, to exercise both the
/// `d <= FLASH_LDS_CAP` (region decode) and `d > FLASH_LDS_CAP` (kernelgen fallback) choices.
/// `q[1,Hq,1,d]`, `k`/`v[1,Hq,cap,d]` (n_rep=1, so Hkv==Hq), `mask[1,1,1,cap]`. The composite is staged
/// directly: `flash_attention_capped` (what `compile` runs) declines a decode wider than the cap,
/// and the planner still answers for any valid equation.
fn flash_decode_graph(d: usize) -> Graph {
    let hq = 2;
    let cap = 8;
    let bld = Builder::new();
    let q = bld.constant("q", TensorType::new(vec![1, hq, 1, d], GDt::F32));
    let k = bld.constant("k", TensorType::new(vec![1, hq, cap, d], GDt::F32));
    let v = bld.constant("v", TensorType::new(vec![1, hq, cap, d], GDt::F32));
    let mask = bld.constant("mask", TensorType::new(vec![1, 1, 1, cap], GDt::F32));
    let mut plan = bld.append_plan(0);
    let out = plan
        .equation(
            OpKind::FlashAttentionDecode {
                n_rep: 1,
                scale: 1.0 / (d as f32).sqrt(),
            },
            [q, k, v, mask]
                .map(|operand| Operand::Value(operand.id))
                .to_vec(),
        )
        .expect("stage the flash decode");
    plan.declare_result(out).expect("declare the result");
    let mut prepared = bld.preflight_append(plan).expect("preflight");
    let id = bld.commit_append(&mut prepared).expect("commit");
    bld.finish(poot_graph_ir::Traced { id })
}

fn flash_decode_planned(g: &Graph, backend: Backend) -> Result<(Plan, KernelChoice), PlanError> {
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::FlashAttentionDecode { .. }))
        .expect("a FlashAttentionDecode eqn");
    planned_with_choice(g, eqn, backend)
}

fn flash_decode_plan(g: &Graph, backend: Backend) -> Result<Plan, PlanError> {
    flash_decode_planned(g, backend).map(|(plan, _)| plan)
}

/// Flashdecode SPIR-V guard: the single-sequence (B=1), D > FLASH_LDS_CAP fallback arm emits
/// `kg::flash_attention_decode`, which uses a private array that is NVPTX-only and crashes SPIR-V codegen
/// (lib.rs:1009-1016). `Backend::SpirvVulkan` must be refused with a typed [`Refusal`] at plan time,
/// while Nvptx is unaffected, and the `d <= FLASH_LDS_CAP` imported path is unaffected on either backend.
#[test]
fn flash_decode_fallback_rejects_spirv_but_keeps_nvptx() {
    let g = flash_decode_graph(512); // D=512 > FLASH_LDS_CAP (256)

    // SpirvVulkan: a clean refusal naming the head dim and the cap, not a crashing Plan::Compute.
    let refusal = refusal_of(
        flash_decode_plan(&g, Backend::SpirvVulkan),
        "flash decode D > LDS cap on SpirvVulkan",
    );
    assert_eq!(refusal.target, Backend::SpirvVulkan);
    assert_eq!(
        refusal.missing,
        Capability::HeadDimExceedsLdsCap {
            head_dim: 512,
            lds_cap: FLASH_LDS_CAP,
        }
    );

    // Nvptx: still Plan::Compute with the kernelgen fallback body/key.
    match flash_decode_planned(&g, Backend::Nvptx).expect("nvptx flash decode should still plan") {
        (Plan::Compute { .. }, choice) => {
            assert!(
                matches!(
                    &choice,
                    KernelChoice::Generated(KernelRequest::Attention(
                        AttentionSpec::DecodeSingle { d: 512, .. }
                    ))
                ),
                "nvptx D>{FLASH_LDS_CAP} should select the kernelgen fallback: {choice:?}"
            );
        }
        other => panic!("expected Plan::Compute on Nvptx, got {other:?}"),
    }
}

/// The `d <= FLASH_LDS_CAP` choice is unaffected by the SpirvVulkan guard above: it plans the
/// synthesized region decode, a plain `Plan::Compute`, on SpirvVulkan.
#[test]
fn flash_decode_within_the_cap_takes_the_region_decode_on_spirv() {
    let g = flash_decode_graph(128); // D=128 <= FLASH_LDS_CAP (256)

    match flash_decode_planned(&g, Backend::SpirvVulkan).expect("the region decode should plan") {
        (Plan::Compute { .. }, choice) => assert!(
            matches!(
                choice,
                KernelChoice::Generated(KernelRequest::Attention(AttentionSpec::RegionDecode {
                    d: 128,
                    ..
                }))
            ),
            "{choice:?}"
        ),
        other => panic!("expected the region decode's Plan::Compute, got {other:?}"),
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
    g.values[rank.id].aval = TensorType::f32(vec![]); // corrupt the declared shape to 0-D
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::ArgTopK { .. }))
        .expect("an ArgTopK eqn");
    match plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
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
    g.values[codes.id].aval = TensorType::f32(vec![]); // corrupt the declared shape to 0-D
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::PackI8))
        .expect("a PackI8 eqn");
    match plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
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
        out: x.id,
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
        out: x.id,
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

    // ScatterUpdate's materializing path is F32-only (its imported kernel has no BF16 variant), so a
    // BF16 output must not take the 2-D-fold path meant for that imported body.
    let scatter_eqn = Eqn {
        op: OpKind::ScatterUpdate,
        inputs: vec![],
        out: x.id,
        layer: None,
    };
    assert!(
        !is_elementwise_2d(
            &g,
            Backend::SpirvVulkan,
            &scatter_eqn,
            &out_shape,
            over_ceiling
        ),
        "ScatterUpdate's imported body is F32-only - a BF16 output must not take this fold"
    );
}

/// Find the named const's value id in `g`.
fn const_id(g: &Graph, name: &str) -> ValueId {
    (0..g.values.len())
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

        let gp = widen_mismatched_matmul_dtypes(&g, backend, &default_caps_for(backend));
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

/// Card 298 + the decode bf16 GEMV lane: the same shape at decode's M=1, which 16-alignment can never
/// admit. On SpirvVulkan/AmdGcn the bf16 GEMV eligibility arm plans the packed-u32 body directly and
/// `widen_mismatched_matmul_dtypes` keeps the weight narrow (no Cast).
#[test]
fn card298_decode_shaped_mixed_matmul_plans_after_widen() {
    let g = mixed_operand_matmul_graph(1, 2048, 2048);
    for backend in [Backend::SpirvVulkan, Backend::AmdGcn(AmdArch::gfx1151())] {
        let (plan, choice) = try_matmul_plan_planned(&g, backend).unwrap_or_else(|e| {
            panic!("decode-shaped mixed matmul must plan as bf16 GEMV on {backend:?}: {e}")
        });
        assert!(
            matches!(plan, Plan::Compute { .. } | Plan::ComputeMeta { .. }),
            "expected a Compute/ComputeMeta plan, got {plan:?}"
        );
        assert!(
            is_imported(&choice, ImportedKernel::GemvCoalescedBf16),
            "on {backend:?}: {choice:?}"
        );
        let gp = widen_mismatched_matmul_dtypes(&g, backend, &default_caps_for(backend));
        assert!(
            try_matmul_plan(&gp, backend).is_ok(),
            "widened decode-shaped mixed matmul must plan on {backend:?}"
        );
        // Reached by keeping the weight BF16 for the packed-lane GEMV, not by a widening Cast.
        assert!(
            !gp.eqns.iter().any(|e| matches!(e.op, OpKind::Cast { .. })),
            "decode-shaped mixed matmul must not gain a Cast on {backend:?}"
        );
        assert_eq!(
            gp.aval(const_id(&gp, "w")).dtype,
            GDt::BF16,
            "the decode GEMV weight must stay BF16 on {backend:?}"
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
    let gp = widen_mismatched_matmul_dtypes(&gm, backend, &default_caps_for(backend));
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
    let gp = widen_mismatched_matmul_dtypes(
        &gm,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
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

/// Card 380 FR-009, narrowed by cards 450 ROCm and 450 PTX. The lowering is scoped by name to the
/// backends that have a kernel: wgpu's typed walk landed it (card 380), the ROCm typed walk the same
/// lowering on AmdGcn (card 450 ROCm), and the PTX typed walk the same lowering on Nvptx (card 450
/// PTX). Above M == 1 all three dispatch the identical imported body and key.
#[test]
fn card450_dense_contraction_plans_on_nvptx_with_the_wgpu_key() {
    let g = dense_contraction_graph(2, 32, 16);
    let Ok((Plan::ComputeMeta { meta, .. }, choice)) =
        try_dense_contraction_plan_planned(&g, Backend::Nvptx)
    else {
        panic!("the packed-BF16 contraction must plan on Nvptx as ComputeMeta");
    };
    assert!(
        matches!(
            choice,
            KernelChoice::Imported {
                kernel: ImportedKernel::DenseBf16Contraction,
                meta_schema: 3,
                ..
            }
        ),
        "{choice:?}"
    );
    assert_eq!(meta, vec![2, 32, 16]);
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
        let Ok((Plan::ComputeMeta { meta, .. }, choice)) =
            try_dense_contraction_plan_planned(&g, backend)
        else {
            panic!(
                "the M == 1 contraction must plan as the Gemv's ComputeMeta step on {backend:?}"
            );
        };
        assert_eq!(meta, vec![32, 16], "{backend:?}: metadata is [K, N]");
        let KernelChoice::Generated(KernelRequest::Contraction(
            request @ ContractionSpec::DenseGemv { weight, layout, .. },
        )) = choice
        else {
            panic!("{backend:?}: expected the generated dense Gemv, got {choice:?}");
        };
        assert_eq!(
            weight,
            poot_target::BufferStorage::bf16_packed(),
            "{backend:?}"
        );
        assert_eq!(layout, WeightLayout::Nk, "{backend:?}");
        format!("{request:?}")
    })
    .collect();
    assert!(
        requests.windows(2).all(|pair| pair[0] == pair[1]),
        "{requests:?}"
    );
}

/// Card 380 FR-010 + the decode bf16 GEMV lane. A BF16 const is packed-lane when every consumer is a
/// `DenseContraction`/`DenseRowGather`/`Cast` packed-lane reader (typed walk), or every consumer is a decode
/// bf16 GEMV weight (resident packed-u32 lane). No const is retyped (Card 1011): one with any other
/// consumer stays BF16 and that consumer is planned or refused on its own.
#[test]
fn card380_bf16_const_stays_narrow_only_for_a_contraction_weight() {
    let folded = dense_contraction_graph(1, 32, 16);
    let prepared = widen_mismatched_matmul_dtypes(
        &folded,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
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
    assert!(bf16_const_feeds_only_packed_readers(&folded, weight));

    // An ordinary mixed matmul at M>1 (prefill, not a decode GEMV) has no packed-lane consumer until the
    // widening pass puts a `Cast` in front of it; the weight is never retyped.
    let mixed = mixed_operand_matmul_graph(128, 32, 16);
    assert!(!bf16_const_feeds_only_packed_readers(
        &mixed,
        const_id(&mixed, "w")
    ));
    assert!(!bf16_const_feeds_decode_bf16_gemv(
        &mixed,
        const_id(&mixed, "w"),
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan)
    ));
    let widened = widen_mismatched_matmul_dtypes(
        &mixed,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
    assert_eq!(widened.aval(const_id(&mixed, "w")).dtype, GDt::BF16);
    assert!(bf16_const_feeds_only_packed_readers(
        &widened,
        const_id(&widened, "w")
    ));

    // The decode-shaped mixed matmul is the third narrow case: stays BF16 for the packed-u32 GEMV.
    let decode_mixed = mixed_operand_matmul_graph(1, 32, 16);
    assert!(bf16_const_feeds_decode_bf16_gemv(
        &decode_mixed,
        const_id(&decode_mixed, "w"),
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan)
    ));
    assert_eq!(
        widen_mismatched_matmul_dtypes(
            &decode_mixed,
            Backend::SpirvVulkan,
            &default_caps_for(Backend::SpirvVulkan)
        )
        .aval(const_id(&decode_mixed, "w"))
        .dtype,
        GDt::BF16
    );

    // A widening `Cast` is a packed reader (Card 1011); a second consumer that is not one disqualifies the
    // const: that consumer would read f32 lanes.
    let mut shared = folded.clone();
    let widened = shared.values.len();
    shared.values.push(poot_graph_ir::graph::ValueMeta::new(
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
    assert!(bf16_const_feeds_only_packed_readers(&shared, weight));
    let negated = shared.values.len();
    shared.values.push(poot_graph_ir::graph::ValueMeta::new(
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
    assert!(!bf16_const_feeds_only_packed_readers(&shared, weight));

    // Only the weight operand is admitted. A const read as a contraction's activation is an f32 operand of
    // that kernel, so keeping it narrow would feed two-byte words to an `&[f32]` param.
    let mut as_activation = folded.clone();
    let out = as_activation.values.len();
    as_activation
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
        weight
    ));

    // A const nothing reads is not eligible: the binder would upload two-byte words no kernel can
    // interpret.
    let mut orphaned = folded.clone();
    let orphan = orphaned.values.len();
    orphaned.values.push(poot_graph_ir::graph::ValueMeta::new(
        TensorType::new(vec![4, 4], GDt::BF16),
        Storage::Const,
        Some("orphan".into()),
    ));
    assert!(!bf16_const_feeds_only_packed_readers(&orphaned, orphan));
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
    let Ok((Plan::Compute { .. }, choice)) = try_dense_row_gather_plan_planned(&g, Backend::Nvptx)
    else {
        panic!("the row gather must plan on Nvptx as a plain Compute");
    };
    assert!(
        is_imported(&choice, ImportedKernel::PackedBf16RowGather),
        "{choice:?}"
    );
}

/// Card 381 FR-009's named fallback, narrowed by Card 449 D1. `E4M3FN` is the second admitted
/// row-gather source dtype, and Card 449 D1 gave it a packed-lane kernel on SpirvVulkan - the only
/// typed packed walk with a Qwen4Exp route today - so it plans there under its own key. Nvptx and
/// AmdGcn have no receipt for that body, so they keep a refusal by name with the op and the source
/// dtype: Nvptx at the row-gather arm, AmdGcn one gate earlier where every e4m3fn
/// equation outside the SPIR-V/wgpu value executor is refused. Neither may fall back to a gather
/// kernel that would read the table at the wrong element width.
#[test]
fn card449_d1_dense_row_gather_e4m3_plans_on_spirv_and_stays_named_elsewhere() {
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
    let Ok((Plan::Compute { .. }, choice)) =
        try_dense_row_gather_plan_planned(&g, Backend::SpirvVulkan)
    else {
        panic!("the E4M3FN row gather must plan on SpirvVulkan, with no metadata buffer");
    };
    assert!(
        is_imported(&choice, ImportedKernel::PackedE4m3RowGather),
        "the choice must name the E4M3FN source dtype's kernel: {choice:?}"
    );
    // PTX has E4M3FN typed storage but no packed-E4M3FN row-gather kernel; ROCm has no E4M3FN storage
    // lowering at all. The refusal carries the op (with its source dtype) and which of the two is missing.
    for (backend, missing) in [
        (Backend::Nvptx, Capability::DtypeLowering),
        (Backend::AmdGcn(AmdArch::gfx1151()), Capability::E4m3Storage),
    ] {
        let refusal = refusal_of(
            try_dense_row_gather_plan(&g, backend),
            "DenseRowGather(E4M3FN) must be rejected",
        );
        assert_eq!(
            refusal.op,
            OpKind::DenseRowGather {
                source: GDt::E4M3FN
            },
            "the refusal must name the op and the source dtype on {backend:?}"
        );
        assert_eq!(refusal.target, backend);
        assert_eq!(refusal.missing, missing, "on {backend:?}");
    }
}

/// Card 450 ROCm: the very same imported packed-BF16 row-gather body plans on AmdGcn, with the key
/// unchanged (the key names the source dtype, not the backend, as the wgpu row pins), so the ROCm
/// typed walk dispatches the identical kernel the wgpu walk does. Host-side plan evidence only; the
/// device receipt is `card450rocm_exact_decode_two_steps_match_cpu_oracle`.
#[test]
fn card450_dense_row_gather_plans_on_amdgcn_with_the_wgpu_key() {
    let g = dense_row_gather_graph(8, 4, 3, GDt::BF16);
    let Ok((Plan::Compute { .. }, choice)) =
        try_dense_row_gather_plan_planned(&g, Backend::AmdGcn(AmdArch::gfx1151()))
    else {
        panic!("the row gather must plan on AmdGcn as a plain Compute");
    };
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
    assert!(bf16_const_feeds_only_packed_readers(&folded, table));
    let prepared = widen_mismatched_matmul_dtypes(
        &folded,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
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
    assert!(!bf16_const_feeds_only_packed_readers(&unfolded, id));
    let prepared = widen_mismatched_matmul_dtypes(
        &unfolded,
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
    assert_eq!(prepared.aval(id).dtype, GDt::BF16);
    // Only the table operand is admitted for this operation.
    let mut as_index = folded.clone();
    let out = as_index.values.len();
    as_index.values.push(poot_graph_ir::graph::ValueMeta::new(
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
    assert!(!bf16_const_feeds_only_packed_readers(&as_index, table));
}

/// Card 381. On wgpu the row gather plans to the imported packed kernel as a plain `Plan::Compute` with no
/// metadata buffer (the row width is derived in-kernel). The key must name the source dtype, so a
/// packed-BF16 gather never shares a cache entry or disk artifact with the f32 table gather of the same
/// shapes (the card-128 Bug-1 class) - nor, since Card 449 D1, with the packed-E4M3FN gather of the
/// same shapes, which decodes a different byte layout.
#[test]
fn card381_row_gather_plan_key_names_the_source_dtype() {
    let folded = dense_row_gather_graph(8, 4, 3, GDt::BF16);
    let Ok((Plan::Compute { key, .. }, choice)) =
        try_dense_row_gather_plan_planned(&folded, Backend::SpirvVulkan)
    else {
        panic!("the row gather must plan on SpirvVulkan, with no metadata buffer");
    };
    assert!(
        is_imported(&choice, ImportedKernel::PackedBf16RowGather),
        "the choice must name the source dtype's kernel: {choice:?}"
    );

    // The packed-E4M3FN gather of the same shapes is a different body and must not collide either.
    let e4m3_folded = dense_row_gather_graph(8, 4, 3, GDt::E4M3FN);
    let Ok(Plan::Compute { key: e4m3_key, .. }) =
        try_dense_row_gather_plan(&e4m3_folded, Backend::SpirvVulkan)
    else {
        panic!("the E4M3FN row gather must plan on SpirvVulkan, with no metadata buffer");
    };
    assert_ne!(
        key, e4m3_key,
        "the BF16 and E4M3FN row gathers of the same shapes must not share a cache entry"
    );

    // The f32 table gather of the same shapes uses a different kernel and must not collide.
    let f32_gather = {
        let bld = Builder::new();
        let table = bld.constant("embed", TensorType::f32(vec![8, 4]));
        let token = bld.slot(poot_graph_ir::Slot::Token, TensorType::f32(vec![3]));
        let gathered = bld.gather(table, 0, token);
        bld.finish(gathered)
    };
    let eqn = f32_gather
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::Gather { .. }))
        .expect("an f32 gather eqn");
    let f32_key = match planned_with_choice(&f32_gather, eqn, Backend::SpirvVulkan).unwrap() {
        (Plan::Compute { key, .. } | Plan::ComputeMeta { key, .. }, choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GatherAxis0),
                "{choice:?}"
            );
            key
        }
        other => panic!("an f32 gather plans to a compute kernel: {other:?}"),
    };
    assert_ne!(key, f32_key);
}

/// Card 380. The plan key must name the operand dtype, so a packed-BF16 kernel never shares a cache entry
/// or disk artifact with an f32-operand kernel of the same shapes (the card-128 Bug-1 class the
/// tensor-core keys guard against).
///
/// At M == 1 the plan is the generated dense Gemv (see [`dense_bf16_decode_gemv_selection_rows`]), whose
/// request names the packed-BF16 weight storage, so the key this row pins is that body's; the imported
/// body's `[rows, K, N]` metadata row is the M > 1 half of the selection test.
#[test]
fn card380_dense_contraction_plan_key_names_the_operand_dtype() {
    let contraction = dense_contraction_graph(1, 32, 16);
    let Ok((Plan::ComputeMeta { key, .. }, choice)) =
        try_dense_contraction_plan_planned(&contraction, Backend::SpirvVulkan)
    else {
        panic!("the M == 1 contraction must plan as the Gemv's ComputeMeta step on SpirvVulkan");
    };
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
    let f32_key = match try_matmul_plan(&f32_matmul, Backend::SpirvVulkan).unwrap() {
        Plan::Compute { key, .. } | Plan::ComputeMeta { key, .. } => key,
        other => panic!("an f32 matmul plans to a compute kernel: {other:?}"),
    };
    assert_ne!(key, f32_key);
}

/// Body, launch and selection for one packed-BF16 `DenseContraction` are chosen together (Card 727): at
/// M == 1, on every backend, the generated dense Gemv takes it with the launch `dense_gemv_launch` picks,
/// its workgroup the schedule's `width` and its grid the schedule's own `grid_threads`, `[K, N]` riding in
/// metadata; above M == 1 the shape-generic imported body keeps its `[rows, K, N]` metadata and the
/// one-thread-per-element launch.
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

        let (
            Plan::ComputeMeta {
                body, grid, meta, ..
            },
            choice,
        ) = try_dense_contraction_plan_planned(&g, backend).unwrap()
        else {
            panic!("M == 1 must plan as the Gemv's ComputeMeta step on {backend:?}");
        };
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

    // M > 1 keeps the imported body on every backend: no Gemv, one thread per output element.
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
        let (Plan::ComputeMeta { meta, grid, .. }, choice) =
            try_dense_contraction_plan_planned(&g, backend).unwrap()
        else {
            panic!("M > 1 must plan as the imported ComputeMeta step on {backend:?}");
        };
        assert!(
            is_imported(&choice, ImportedKernel::DenseBf16Contraction),
            "{backend:?}: {choice:?}"
        );
        assert_eq!(
            meta,
            vec![2, k as u32, n as u32],
            "{backend:?}: metadata is [rows, K, N]"
        );
        assert_eq!(
            grid,
            [out_numel as u32, 1, 1],
            "{backend:?}: above M == 1 the launch stays one thread per output element"
        );
    }
}

/// Card 380 P1. `validate_wgpu_tensor_execution` is the fail-closed half of the packed-BF16 contract: the
/// typed value walk stores BF16 as packed u32 lanes and checks every plan against that, and every other
/// wgpu entry point (which binds `Arc<[f32]>` tensors) rejects a BF16 value by name instead of uploading
/// f32 words for a kernel that reads packed lanes.
///
/// The device test `wgpu_resident_kv_rejects_a_bf16_contraction_weight` drives the same rejection through
/// `run_resident_kv`; this one covers it without a device.
#[test]
fn card380_wgpu_tensor_execution_rejects_every_bf16_value() {
    // The card 380 shape: the contraction weight survives preparation as BF16, safe only on the typed walk.
    let prepared = widen_mismatched_matmul_dtypes(
        &dense_contraction_graph(1, 32, 16),
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
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
    let widened = widen_mismatched_matmul_dtypes(
        &bld.finish(gathered),
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
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
    let f32_only = widen_mismatched_matmul_dtypes(
        &mixed_operand_matmul_graph(1, 32, 16),
        Backend::SpirvVulkan,
        &default_caps_for(Backend::SpirvVulkan),
    );
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
    let (Plan::ComputeMeta { body, .. }, _) = planned_with_choice(&g, eqn, backend).unwrap() else {
        panic!("the roomy device plans the Gemv");
    };
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
    let (Plan::Compute { body, .. } | Plan::ComputeMeta { body, .. }) = plan else {
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

/// The single contraction choice `compile` makes for a decode projection `x [1, 1, k] f32 @ w bf16` on
/// `backend`, with `w` held `[k, n]` (`transposed: false`) or in checkpoint `[n, k]` order read through a
/// transpose (`transposed: true`).
fn bf16_projection_contraction_choice(
    backend: Backend,
    transposed: bool,
    k: usize,
    n: usize,
) -> KernelChoice {
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
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: CompileLimits::STANDARD,
    };
    let target = Target {
        backend,
        caps: default_caps_for(backend),
    };
    let program = compile(&graph, &target, &options)
        .unwrap_or_else(|e| panic!("{backend:?} transposed={transposed}: {e}"));
    let mut contractions = program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::MatMul | OpKind::DenseContraction { .. }));
    let (eqn, _) = contractions
        .next()
        .unwrap_or_else(|| panic!("{backend:?} transposed={transposed}: no contraction planned"));
    assert!(contractions.next().is_none(), "one contraction");
    program.kernel_choice(eqn).clone()
}

/// Card 727 SC-004, at planning: through `compile`, a BF16 decode projection (M == 1) whose
/// weight is held in checkpoint `[N, K]` order plans the generated dense Gemv, and the same projection over a
/// `[K, N]` weight never plans a Gemv that gives each output column its own workgroup (the generated dense Gemv
/// or the `gemv_lds` chunk body, both of which would read `[K, N]` at a stride of `N` per lane): it takes the
/// imported column-tile body on SPIR-V and ROCm, and a non-Gemv body on NVPTX. Mutation: in
/// `planner/matmul.rs`'s decode-GEMV arm, request the dense Gemv (`ContractionSpec::DenseGemv` over
/// `WeightLayout::Kn`) for a BF16 weight; `generate` refuses it, `compile` fails and this row goes red.
#[test]
fn a_k_by_n_bf16_decode_projection_never_plans_a_per_column_gemv() {
    let (k, n) = (896, 256);
    let per_column_gemv = |choice: &KernelChoice| {
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::DenseGemv { .. } | ContractionSpec::GemvChunk { .. }
            ))
        )
    };
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let nk = bf16_projection_contraction_choice(backend, true, k, n);
        assert!(
            matches!(
                nk,
                KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
                    layout: WeightLayout::Nk,
                    ..
                }))
            ),
            "{backend:?} [N, K]: {nk:?}"
        );
        let kn = bf16_projection_contraction_choice(backend, false, k, n);
        assert!(!per_column_gemv(&kn), "{backend:?} [K, N]: {kn:?}");
        if backend != Backend::Nvptx {
            assert!(
                is_imported(&kn, ImportedKernel::GemvCoalescedBf16),
                "{backend:?} [K, N]: the column-tile body, got {kn:?}"
            );
        }
    }
}
