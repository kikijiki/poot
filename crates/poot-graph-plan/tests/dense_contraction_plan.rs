//! Card 645: a projection against an F32 weight held in checkpoint `[N, K]` orientation compiles to one
//! `DenseContraction` that reads the weight as stored, never a device `transpose` of it, and the fold refuses a
//! transposed value something else reads. Card 1007: the same holds for an F16 weight, stored packed two elements
//! per `u32` word, which folds only when the contraction is its sole reader.
//!
//! Every row drives the production `compile` for each backend and reads what `Program::planned()` holds.

use poot_graph_ir::{BinOp, Builder, Graph, OpKind, Operand, RedOp, StateRole, TensorType, Traced};
use poot_graph_plan::{
    CompileError, CompileOptions, KernelChoice, PlanError, Program, StorageGap, Target, compile,
};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_target::{AmdArch, Backend, BufferStorage, DeviceCaps};
use poot_tensor::DType;

fn backends() -> [Backend; 3] {
    [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ]
}

/// `K` past one 128-lane GEMV workgroup, `N` not a multiple of the 8-column tile.
const K: usize = 200;
const N: usize = 37;

fn target(backend: Backend) -> Target {
    Target::Gpu {
        backend,
        caps: match backend {
            Backend::SpirvVulkan => DeviceCaps::wgpu_rdna3_igpu(),
            Backend::Nvptx => DeviceCaps::ptx_default(),
            Backend::AmdGcn(_) => DeviceCaps::rocm_default(),
        },
    }
}

/// `g` compiled for `backend` through the full pipeline.
fn program(g: &Graph, backend: Backend) -> Program {
    let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    compile(g, &target(backend), &options).expect("the graph compiles")
}

/// `matmul(x, transpose(w))`, `w` an F32 `[N, K]` constant.
fn projection(m: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![N, K])).unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

fn op_names(program: &Program) -> Vec<String> {
    program.planned().map(|(eqn, _)| eqn.op.name()).collect()
}

fn count(program: &Program, want: impl Fn(&OpKind) -> bool) -> usize {
    program.planned().filter(|(eqn, _)| want(&eqn.op)).count()
}

/// The generated request behind one equation's plan.
fn request(program: &Program, want: impl Fn(&OpKind) -> bool) -> &KernelRequest {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| want(&eqn.op))
        .expect("the equation is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(request) => request,
        other => panic!("expected a generated kernel, got {other:?}"),
    }
}

fn is_dense_f32(op: &OpKind) -> bool {
    matches!(op, OpKind::DenseContraction { weight } if *weight == DType::F32)
}

/// SC-002: the program holds one `DenseContraction` equation for the projection and no `Transpose` dispatch,
/// at `M = 1`, `4` and `33`, on every backend. The kernel is a generated one reading the weight in `[N, K]`
/// order: the decode GEMV at `M = 1` and the tiled GEMM above it.
///
/// Mutation: make `fold_dense_contractions` skip an F32 weight; the fold skips, a `Transpose` and a
/// `MatMul` stay in the program and this row goes red.
#[test]
fn checkpoint_orientation_projection_plans_one_dense_contraction_and_no_transpose() {
    for backend in backends() {
        for m in [1, 4, 33] {
            let program = program(&projection(m), backend);
            let label = format!("{backend:?} M={m}: {:?}", op_names(&program));
            assert_eq!(count(&program, is_dense_f32), 1, "{label}");
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::Transpose { .. })),
                0,
                "{label}"
            );
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::MatMul { .. })),
                0,
                "{label}"
            );
            let request = request(&program, is_dense_f32);
            let gemv = m == 1;
            assert!(
                match request {
                    KernelRequest::Contraction(ContractionSpec::DenseGemv { .. }) => gemv,
                    KernelRequest::Contraction(ContractionSpec::DenseTiled { .. }) => !gemv,
                    _ => false,
                },
                "{label}: {request:?}"
            );
        }
    }
}

/// SC-003: a transposed weight something else also reads is a real materialization, so the fold leaves the
/// `Transpose` and the `MatMul` in the plan. The weight's transpose feeds the matmul and a reduction.
///
/// Mutation: drop the consumer-count check in `fold_dense_contractions`; the transpose is dropped under its
/// second reader, no `Transpose` is planned and this row goes red.
#[test]
fn a_transposed_weight_with_a_second_consumer_keeps_the_transpose_and_the_matmul() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, K])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![N, K])).unwrap();
    let wt = b.transpose(w, vec![1, 0]).unwrap();
    let y = b.matmul(x, wt).unwrap();
    let column_sums = b.reduce(RedOp::Sum, wt, 0, false).unwrap();
    let row_sums = b.reduce(RedOp::Sum, y, 0, false).unwrap();
    let total = b.binary(BinOp::Add, row_sums, column_sums).unwrap();
    let g = b.finish(total).unwrap();
    for backend in backends() {
        let program = program(&g, backend);
        let label = format!("{backend:?}: {:?}", op_names(&program));
        assert_eq!(count(&program, is_dense_f32), 0, "{label}");
        assert!(
            program
                .planned()
                .any(|(eqn, _)| matches!(eqn.op, OpKind::Transpose { .. })),
            "{label}"
        );
        assert_eq!(
            count(&program, |op| matches!(op, OpKind::MatMul { .. })),
            1,
            "{label}"
        );
    }
}

/// SC-003: a transposed weight that escapes the graph (here, written to carried state) is read outside the
/// matmul, so the fold leaves the `Transpose` and the `MatMul` in the plan.
///
/// Mutation: drop the graph-escape check in `fold_dense_contractions`; the transpose is dropped though state
/// reads it, no `Transpose` is planned and this row goes red.
#[test]
fn a_transposed_weight_that_escapes_the_graph_keeps_the_transpose_and_the_matmul() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, K])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![N, K])).unwrap();
    let kept = b
        .state_input("kept", TensorType::f32(vec![K, N]), StateRole::Recurrent)
        .unwrap();
    let wt = b.transpose(w, vec![1, 0]).unwrap();
    let y = b.matmul(x, wt).unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(y).with_state(&[(kept, wt)]))
        .unwrap();
    for backend in backends() {
        let program = program(&g, backend);
        let label = format!("{backend:?}: {:?}", op_names(&program));
        assert_eq!(count(&program, is_dense_f32), 0, "{label}");
        assert!(
            program
                .planned()
                .any(|(eqn, _)| matches!(eqn.op, OpKind::Transpose { .. })),
            "{label}"
        );
        assert_eq!(
            count(&program, |op| matches!(op, OpKind::MatMul { .. })),
            1,
            "{label}"
        );
    }
}

/// The scalar sum of a rank-2 value, so two differently shaped readers can meet in one output.
fn total_sum(b: &Builder, value: Traced) -> Traced {
    b.reduce(
        RedOp::Sum,
        b.reduce(RedOp::Sum, value, 1, false).unwrap(),
        0,
        false,
    )
    .unwrap()
}

/// `matmul(x, transpose(w))`, `w` an F16 `[N, K]` constant.
fn f16_projection(m: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::F16))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

fn is_dense_f16(op: &OpKind) -> bool {
    matches!(op, OpKind::DenseContraction { weight } if *weight == DType::F16)
}

/// The weight-operand storage every request of the contraction names, and the storage the program planned for
/// that weight value.
fn weight_storages(program: &Program) -> (BufferStorage, BufferStorage) {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| is_dense_f16(&eqn.op))
        .expect("the contraction is planned");
    let Operand::Value(weight) = eqn.inputs[1] else {
        panic!("the weight operand is a value");
    };
    let named = match request(program, is_dense_f16) {
        KernelRequest::Contraction(
            ContractionSpec::DenseGemv { weight, .. }
            | ContractionSpec::DenseTiled { weight, .. }
            | ContractionSpec::DenseSerial { weight, .. },
        ) => *weight,
        other => panic!("expected a dense-contraction request, got {other:?}"),
    };
    (named, program.storage().storage(weight).buffer_storage())
}

/// Card 1007 (acceptance row 1, plan side): an F16 `[N, K]` checkpoint weight compiles exactly like the F32 one:
/// one `DenseContraction` and no `Transpose` dispatch at `M = 1`, `4` and `33`, on every backend, the decode GEMV at
/// `M = 1` and the tiled GEMM above it. Every request names packed-F16 weight storage, and the weight value's
/// planned storage is that same packed record: one buffer of two-byte elements in `u32` words, neither widened to
/// f32 nor native F16.
///
/// Mutation: make `fold_dense_contractions` skip an F16 weight; the fold skips, a `Transpose` stays in the
/// program and this row goes red.
#[test]
fn an_f16_checkpoint_projection_plans_one_packed_dense_contraction_and_no_transpose() {
    for backend in backends() {
        for m in [1, 4, 33] {
            let program = program(&f16_projection(m), backend);
            let label = format!("{backend:?} M={m}: {:?}", op_names(&program));
            assert_eq!(count(&program, is_dense_f16), 1, "{label}");
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::Transpose { .. })),
                0,
                "{label}"
            );
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::MatMul { .. })),
                0,
                "{label}"
            );
            let gemv = m == 1;
            let request = request(&program, is_dense_f16);
            assert!(
                match request {
                    KernelRequest::Contraction(ContractionSpec::DenseGemv { .. }) => gemv,
                    KernelRequest::Contraction(ContractionSpec::DenseTiled { .. }) => !gemv,
                    _ => false,
                },
                "{label}: {request:?}"
            );
            let (named, planned) = weight_storages(&program);
            assert_eq!(planned, BufferStorage::f16_packed(), "{label}");
            assert_eq!(
                named, planned,
                "{label}: the request names {named:?}, the weight is planned {planned}"
            );
        }
    }
}

/// Card 1007: an F16 weight that another equation also reads (here, a tied embedding gathering its rows) needs
/// its native storage there, and a buffer has one storage, so the fold leaves the projection as traced rather
/// than packing a buffer a second reader cannot read.
///
/// Mutation: drop the F16 sole-reader gate in `fold_dense_contractions`; the projection folds, the storage
/// analysis refuses the shared weight (`F16PackedLanes`) and this row goes red.
#[test]
fn an_f16_weight_with_a_second_reader_is_not_folded() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::F16))
        .unwrap();
    let ids = b
        .constant("ids", TensorType::new(vec![3], DType::I32))
        .unwrap();
    let rows = b.cast(b.gather(w, 0, ids).unwrap(), DType::F32).unwrap();
    let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let total = b
        .binary(BinOp::Add, total_sum(&b, y), total_sum(&b, rows))
        .unwrap();
    let g = b.finish(total).unwrap();
    let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    for backend in backends() {
        match compile(&g, &target(backend), &options) {
            Ok(program) => assert_eq!(
                count(&program, |op| matches!(op, OpKind::DenseContraction { .. })),
                0,
                "{backend:?}: {:?}",
                op_names(&program)
            ),
            // The unfolded mixed F32 x F16 matmul may have no body on a backend; the shared-weight refusal must
            // never be why.
            Err(error) => assert!(
                !matches!(
                    &error,
                    CompileError::Plan(plan) if matches!(
                        **plan,
                        PlanError::UnrepresentableValue {
                            gap: StorageGap::F16PackedLanes,
                            ..
                        }
                    )
                ),
                "{backend:?}: {error}"
            ),
        }
    }
}

/// Card 1007: the storage analysis is the backstop the fold's gate mirrors. A graph that reaches `compile` already
/// holding an F16 `DenseContraction` whose weight a second equation reads natively is refused, naming the weight,
/// instead of packing a buffer the other reader would read at the wrong width.
///
/// Mutation: return `Ok` from `dense_contraction_weight_storages` without the sole-reader check; `compile`
/// accepts the graph and this row goes red.
#[test]
fn a_packed_f16_weight_with_a_native_reader_is_a_typed_refusal() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::F16))
        .unwrap();
    let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let widened = b.cast(w, DType::F32).unwrap();
    let total = b
        .binary(BinOp::Add, total_sum(&b, y), total_sum(&b, widened))
        .unwrap();
    let traced = b.finish(total).unwrap();
    // The contraction the fold would make of the projection, written into the graph directly: the fold itself
    // never makes this one, because `w` has a second reader. (The transpose stays, now unread.)
    let g = poot_graph_ir::test_support::graph_fixtures::derive(&traced, |draft| {
        let matmul = draft
            .eqns_mut()
            .iter_mut()
            .find(|eqn| matches!(eqn.op, OpKind::MatMul { .. }))
            .expect("the traced graph multiplies");
        matmul.op = OpKind::DenseContraction { weight: DType::F16 };
        matmul.inputs[1] = Operand::Value(w.id());
    })
    .unwrap();
    let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    for backend in backends() {
        match compile(&g, &target(backend), &options) {
            Err(CompileError::Plan(plan)) => match *plan {
                PlanError::UnrepresentableValue {
                    value,
                    dtype: DType::F16,
                    gap: StorageGap::F16PackedLanes,
                } => assert_eq!(value, w.id(), "{backend:?}"),
                other => panic!("{backend:?}: expected the packed-F16 refusal, got {other}"),
            },
            Err(other) => panic!("{backend:?}: expected the packed-F16 refusal, got {other}"),
            Ok(program) => panic!(
                "{backend:?}: compiled a shared packed weight: {:?}",
                op_names(&program)
            ),
        }
    }
}

/// `(x @ w^T + bias) + residual`, `w` a `[N, K]` constant of dtype `weight`: the shape of a biased projection
/// feeding a residual connection.
fn biased_residual_projection(m: usize, weight: DType) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], weight))
        .unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![N])).unwrap();
    let residual = b.constant("residual", TensorType::f32(vec![m, N])).unwrap();
    let projection = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let out = b
        .binary(
            BinOp::Add,
            b.binary(BinOp::Add, projection, bias).unwrap(),
            residual,
        )
        .unwrap();
    b.finish(out).unwrap()
}

/// POOT-1032: a dense head with a bias and a residual plans ONE `ContractionEpilogue` at every `M`, on every
/// backend and for an F32 and a BF16 weight: the decode Gemv at `M = 1`, the generated tiled GEMM above it (a
/// single partial row-tile at `M = 5`, many at `M = 160`). The epilogue carries both steps and both leaves in the
/// store, so no standalone add is dispatched and no head result round-trips through memory.
///
/// Mutation: make `epilogue_head_lowers` answer `dense_decode_gemv_in(..).is_some()` for a dense head (the
/// pre-1032 gate); at `M > 1` the rule leaves the head and both adds apart and this row goes red.
#[test]
fn a_biased_residual_dense_head_is_one_epilogue_contraction_at_every_m() {
    for backend in backends() {
        for weight in [DType::F32, DType::BF16] {
            for m in [1, 5, 160] {
                let program = program(&biased_residual_projection(m, weight), backend);
                let label = format!("{backend:?} {weight:?} M={m}: {:?}", op_names(&program));
                let is_head = |op: &OpKind| {
                    matches!(
                        op,
                        OpKind::ContractionEpilogue {
                            head: poot_graph_ir::op::ContractionHead::Dense { weight: stored },
                            ..
                        } if *stored == weight
                    )
                };
                assert_eq!(program.planned().count(), 1, "{label}");
                assert_eq!(count(&program, is_head), 1, "{label}");
                match request(&program, is_head) {
                    KernelRequest::Contraction(ContractionSpec::DenseGemv {
                        epilogue: Some(epilogue),
                        ..
                    }) if m == 1 => assert_eq!(epilogue.steps.len(), 2, "{label}"),
                    KernelRequest::Contraction(ContractionSpec::DenseTiled {
                        epilogue: Some(epilogue),
                        ..
                    }) if m > 1 => {
                        assert_eq!(epilogue.steps.len(), 2, "{label}");
                        assert_eq!(epilogue.leaves.len(), 2, "{label}");
                    }
                    other => panic!("{label}: {other:?}"),
                }
            }
        }
    }
}

/// POOT-1036: a plain `MatMul` head over an f32 `[K, N]` weight at `M > 1` (the non-checkpoint route, the
/// generated tiled GEMM) with a bias and a residual plans ONE `ContractionEpilogue { head: MatMul }` whose
/// `TiledRegion` carries both steps and both leaves, on every backend.
///
/// Mutation: make `epilogue_head_lowers` accept only `MatMulRoute::Serial` for a `MatMul` head (the pre-1036
/// gate); the head and both adds stay apart and this row goes red.
#[test]
fn a_biased_residual_tiled_matmul_head_is_one_epilogue_contraction() {
    for backend in backends() {
        for m in [5, 160] {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
            let w = b.constant("w", TensorType::f32(vec![K, N])).unwrap();
            let bias = b.constant("bias", TensorType::f32(vec![N])).unwrap();
            let residual = b.constant("residual", TensorType::f32(vec![m, N])).unwrap();
            let head = b.matmul(x, w).unwrap();
            let out = b
                .binary(
                    BinOp::Add,
                    b.binary(BinOp::Add, head, bias).unwrap(),
                    residual,
                )
                .unwrap();
            let g = b.finish(out).unwrap();
            let fused = program(&g, backend);
            let label = format!("{backend:?} M={m}: {:?}", op_names(&fused));
            let is_head = |op: &OpKind| {
                matches!(
                    op,
                    OpKind::ContractionEpilogue {
                        head: poot_graph_ir::op::ContractionHead::MatMul,
                        ..
                    }
                )
            };
            assert_eq!(fused.planned().count(), 1, "{label}");
            assert_eq!(count(&fused, is_head), 1, "{label}");
            match request(&fused, is_head) {
                KernelRequest::Contraction(ContractionSpec::TiledRegion {
                    epilogue: Some(epilogue),
                    ..
                }) => {
                    assert_eq!(epilogue.steps.len(), 2, "{label}");
                    assert_eq!(epilogue.leaves.len(), 2, "{label}");
                }
                other => panic!("{label}: {other:?}"),
            }
        }
    }
}

/// `(x @ w + bias) + residual` with `x: [xs]`, `w: [ws]` of `head_dtype` (bias and residual of the head's dtype
/// and the output's shape), the shape of a `MatMul` head the generated tiled GEMM must not take.
fn biased_residual_matmul(xs: Vec<usize>, ws: Vec<usize>, head_dtype: DType) -> Graph {
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(xs.clone(), head_dtype))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(ws.clone(), head_dtype))
        .unwrap();
    let n = *ws.last().unwrap();
    let mut out_shape = xs;
    *out_shape.last_mut().unwrap() = n;
    let bias = b
        .constant("bias", TensorType::new(vec![n], head_dtype))
        .unwrap();
    let residual = b
        .constant("residual", TensorType::new(out_shape, head_dtype))
        .unwrap();
    let out = b
        .binary(
            BinOp::Add,
            b.binary(BinOp::Add, b.matmul(x, w).unwrap(), bias).unwrap(),
            residual,
        )
        .unwrap();
    b.finish(out).unwrap()
}

/// POOT-1036: only an f32 `[K, N]` `MatMul` head above one row takes the generated tiled GEMM with an epilogue.
/// Every other `MatMul` head keeps the plan it would get without one, so no head loses its schedule to a fused store:
/// an F16 or BF16 head is not tileable (the body is f32-only) and fuses on its serial dtype-generic body (BF16 is
/// refused on SPIR-V, which has no scalar bf16); a rank-3 weight takes the imported batched tiled GEMM on SPIR-V and
/// PTX, which carries no epilogue, so its chain stays apart there (AMD plans it serially, fused); `M == 1` is the
/// decode GEMV and keeps its chain everywhere.
///
/// Mutation: drop the F32-operand clause, the rank-2-weight clause or the `M > 1` clause of
/// `is_tileable_matmul_in` (each lets the matcher admit the head onto the tiled GEMM); the matching case then plans
/// a `TiledRegion` or loses its kept chain and this row goes red.
#[test]
fn a_matmul_head_off_the_tiled_route_never_takes_the_tiled_epilogue() {
    /// What a case plans on a backend.
    enum Plan {
        /// Head and chain stay apart (`matmul` then `fused`).
        ChainKept,
        /// One `ContractionEpilogue` on the serial body.
        SerialEpilogue,
        /// The planner refuses the head by name.
        Refused,
    }
    use Plan::{ChainKept, Refused, SerialEpilogue};
    let cases = [
        (
            "f16 head",
            biased_residual_matmul(vec![5, K], vec![K, N], DType::F16),
            [SerialEpilogue, SerialEpilogue, SerialEpilogue],
        ),
        (
            "bf16 head",
            biased_residual_matmul(vec![5, K], vec![K, N], DType::BF16),
            [Refused, SerialEpilogue, SerialEpilogue],
        ),
        (
            "rank-3 weight",
            biased_residual_matmul(vec![2, 5, K], vec![2, K, N], DType::F32),
            [ChainKept, ChainKept, SerialEpilogue],
        ),
        (
            "M == 1",
            biased_residual_matmul(vec![1, K], vec![K, N], DType::F32),
            [ChainKept, ChainKept, ChainKept],
        ),
    ];
    let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    for (name, graph, expected) in &cases {
        for (backend, expected) in backends().into_iter().zip(expected) {
            let label = format!("{name} on {backend:?}");
            let program = match (compile(graph, &target(backend), &options), expected) {
                (Err(CompileError::Plan(_)), Refused) => continue,
                (Ok(program), ChainKept | SerialEpilogue) => program,
                (other, _) => panic!("{label}: unexpected compile result {:?}", other.err()),
            };
            for (eqn, _) in program.planned() {
                assert!(
                    !matches!(
                        program.kernel_choice(eqn),
                        KernelChoice::Generated(KernelRequest::Contraction(
                            ContractionSpec::TiledRegion { .. }
                        ))
                    ),
                    "{label}: {:?} took the generated tiled GEMM",
                    op_names(&program)
                );
            }
            let fused = count(&program, |op| {
                matches!(op, OpKind::ContractionEpilogue { .. })
            });
            match expected {
                ChainKept => {
                    assert_eq!(fused, 0, "{label}: {:?}", op_names(&program));
                    assert_eq!(program.planned().count(), 2, "{label}: head and chain");
                }
                SerialEpilogue => {
                    assert_eq!(fused, 1, "{label}: {:?}", op_names(&program));
                    assert!(
                        matches!(
                            request(&program, |op| matches!(
                                op,
                                OpKind::ContractionEpilogue { .. }
                            )),
                            KernelRequest::Contraction(ContractionSpec::Serial {
                                epilogue: Some(_),
                                ..
                            })
                        ),
                        "{label}: the serial body carries the epilogue"
                    );
                }
                Refused => unreachable!("a refusal continues above"),
            }
        }
    }
}

/// Groups of the grouped projection rows: `G * N` rows of `K` BF16 words, read as `G` blocks of `N` rows.
const GROUPS: usize = 3;

/// `matmul(x, transpose(reshape(w, [G, N, K]), [0, 2, 1]))`, `w` a BF16 `[G * N, K]` constant: a grouped
/// projection spelled the way the V4 tracer spells `wo_a`. `blocks` lets a row read the blocked weight twice.
fn grouped_projection(
    m: usize,
    blocks_reader: impl FnOnce(&Builder, Traced, Traced) -> Traced,
) -> Graph {
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::f32(vec![GROUPS, m, K]))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![GROUPS * N, K], DType::BF16))
        .unwrap();
    let blocks = b.reshape(w, vec![GROUPS, N, K]).unwrap();
    let y = b
        .matmul(x, b.transpose(blocks, vec![0, 2, 1]).unwrap())
        .unwrap();
    let out = blocks_reader(&b, blocks, y);
    b.finish(out).unwrap()
}

fn is_grouped_bf16(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::GroupedDenseContraction {
            weight: DType::BF16,
            groups: GROUPS
        }
    )
}

/// POOT-1048: a grouped projection over a BF16 `[G * N, K]` weight compiles to one `GroupedDenseContraction`
/// and no `Transpose`, `MatMul` or `Cast` at `M = 1`, `4` and `33`, on every backend, as the generated
/// one-thread-per-output dense body over a `[G, N, K]` weight in packed-BF16 storage; the weight value is planned
/// at that same storage (one resident buffer of two-byte elements, never widened).
///
/// Mutation: drop the `[0, 2, 1]` arm of `fold_dense_contractions`; the fold skips, the `Transpose` and `MatMul`
/// stay in the program and this row goes red.
#[test]
fn a_grouped_bf16_projection_plans_one_grouped_dense_contraction_and_no_transpose() {
    for backend in backends() {
        for m in [1, 4, 33] {
            let program = program(&grouped_projection(m, |_, _, y| y), backend);
            let label = format!("{backend:?} M={m}: {:?}", op_names(&program));
            assert_eq!(count(&program, is_grouped_bf16), 1, "{label}");
            for (name, op) in [
                ("Transpose", OpKind::Transpose { perm: vec![] }),
                ("MatMul", OpKind::matmul()),
                ("Cast", OpKind::Cast { to: DType::F32 }),
            ] {
                let same = |planned: &OpKind| {
                    std::mem::discriminant(planned) == std::mem::discriminant(&op)
                };
                assert_eq!(count(&program, same), 0, "{label}: {name}");
            }
            let (eqn, _) = program
                .planned()
                .find(|(eqn, _)| is_grouped_bf16(&eqn.op))
                .expect("the contraction is planned");
            let Operand::Value(weight) = eqn.inputs[1] else {
                panic!("the weight operand is a value");
            };
            match request(&program, is_grouped_bf16) {
                KernelRequest::Contraction(ContractionSpec::DenseSerial {
                    weight: named,
                    shapes,
                    ..
                }) => {
                    assert_eq!(*named, BufferStorage::bf16_packed(), "{label}");
                    assert_eq!(
                        *named,
                        program.storage().storage(weight).buffer_storage(),
                        "{label}: the request and the weight value share one storage"
                    );
                    assert_eq!(shapes.b_shape, [GROUPS, N, K], "{label}");
                    assert_eq!(shapes.out_shape, [GROUPS, m, N], "{label}");
                }
                other => panic!("{label}: expected the dense serial body, got {other:?}"),
            }
        }
    }
}

/// POOT-1048: the blocked weight is a real value when something else reads it (here a reduction), so the fold
/// leaves the `Reshape`, `Transpose` and `MatMul` as traced instead of dropping a reshape a second reader needs.
///
/// Mutation: drop the consumer-count check on the reshape's value in `fold_dense_contractions`; the reshape is
/// dropped under its second reader and this row goes red.
#[test]
fn a_blocked_weight_with_a_second_consumer_is_not_folded() {
    let g = grouped_projection(4, |b, blocks, y| {
        let sum_all = |v: Traced| {
            (0..3)
                .rev()
                .fold(v, |v, axis| b.reduce(RedOp::Sum, v, axis, false).unwrap())
        };
        b.binary(
            BinOp::Add,
            sum_all(y),
            sum_all(b.cast(blocks, DType::F32).unwrap()),
        )
        .unwrap()
    });
    let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    for backend in backends() {
        // Unfolded, the BF16 transposed view has no lowering on the packed-lane wgpu target (a typed refusal of
        // the transpose); the native-BF16 targets plan the chain as traced. Never a grouped contraction.
        match (backend, compile(&g, &target(backend), &options)) {
            (Backend::SpirvVulkan, Err(CompileError::Plan(plan))) => assert!(
                plan.to_string()
                    .contains("transpose [0, 2, 1] (bf16 -> bf16) is refused"),
                "{backend:?}: {plan}"
            ),
            (Backend::SpirvVulkan, other) => {
                panic!("{backend:?}: expected the transpose refusal, got {other:?}")
            }
            (_, Ok(program)) => {
                let label = format!("{backend:?}: {:?}", op_names(&program));
                assert_eq!(count(&program, is_grouped_bf16), 0, "{label}");
                assert_eq!(
                    count(&program, |op| matches!(op, OpKind::Reshape { .. })),
                    1,
                    "{label}"
                );
                assert_eq!(
                    count(&program, |op| matches!(op, OpKind::MatMul { .. })),
                    1,
                    "{label}"
                );
            }
            (_, Err(error)) => panic!("{backend:?}: {error}"),
        }
    }
}

/// POOT-1048: only a BF16 grouped weight folds; an F32 or F16 one has no grouped device row, so it keeps the
/// traced `Reshape`, `Transpose` and `MatMul` on every backend.
///
/// Mutation: drop the BF16 check in `fold_dense_contractions`' `[0, 2, 1]` arm; the F32 chain folds into an op the
/// planner refuses and this row goes red.
#[test]
fn a_grouped_weight_that_is_not_bf16_keeps_the_traced_chain() {
    for weight in [DType::F32, DType::F16] {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::f32(vec![GROUPS, 4, K]))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![GROUPS * N, K], weight))
            .unwrap();
        let blocks = b.reshape(w, vec![GROUPS, N, K]).unwrap();
        let out = b
            .matmul(x, b.transpose(blocks, vec![0, 2, 1]).unwrap())
            .unwrap();
        let g = b.finish(out).unwrap();
        let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
        for backend in backends() {
            // F32 plans the chain as traced; F16 has no lowering for the mixed matmul (a typed refusal).
            match compile(&g, &target(backend), &options) {
                Ok(program) => {
                    let label = format!("{weight:?} {backend:?}: {:?}", op_names(&program));
                    assert_eq!(count(&program, is_grouped_bf16), 0, "{label}");
                    assert_eq!(
                        count(&program, |op| matches!(
                            op,
                            OpKind::GroupedDenseContraction { .. }
                        )),
                        0,
                        "{label}"
                    );
                }
                Err(CompileError::Plan(_)) if weight == DType::F16 => {}
                Err(other) => panic!("{weight:?} {backend:?}: {other}"),
            }
        }
    }
}

/// POOT-1241: `infer` types a grouped contraction over any arithmetic float weight, so a target that has no
/// grouped body for the weight's dtype says so as a typed `DtypeLowering` refusal naming the equation (the
/// planner's capability to refuse), not by `infer` rejecting a graph that is well typed.
///
/// Mutation: make the planner's grouped arm plan any weight dtype as BF16; the F32 and F16 rows compile and
/// go red.
#[test]
fn a_grouped_contraction_over_a_weight_without_grouped_rows_is_a_typed_refusal() {
    for weight in [DType::F32, DType::F16] {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::f32(vec![GROUPS, 4, K]))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![GROUPS * N, K], weight))
            .unwrap();
        let blocks = b.reshape(w, vec![GROUPS, N, K]).unwrap();
        let out = b
            .matmul(x, b.transpose(blocks, vec![0, 2, 1]).unwrap())
            .unwrap();
        let traced = b.finish(out).unwrap();
        let matmul = traced
            .eqns()
            .iter()
            .position(|eqn| matches!(eqn.op, OpKind::MatMul { .. }))
            .expect("the projection has one matmul");
        let g = poot_graph_ir::test_support::graph_fixtures::derive(&traced, |draft| {
            draft.eqns_mut()[matmul].op = OpKind::GroupedDenseContraction {
                weight,
                groups: GROUPS,
            };
            draft.eqns_mut()[matmul].inputs = vec![Operand::Value(x.id()), Operand::Value(w.id())];
        })
        .expect("a grouped contraction over an arithmetic float weight is well typed");
        let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
        for backend in backends() {
            match compile(&g, &target(backend), &options) {
                Err(CompileError::Plan(error)) if matches!(*error, PlanError::Refused(_)) => {
                    let PlanError::Refused(refusal) = *error else {
                        unreachable!("matched above")
                    };
                    assert_eq!(
                        refusal.missing,
                        poot_graph_plan::Capability::DtypeLowering,
                        "{weight:?} {backend:?}"
                    );
                    assert_eq!(refusal.eqn, out.id(), "{weight:?} {backend:?}");
                }
                other => panic!(
                    "{weight:?} {backend:?}: expected a DtypeLowering refusal, got {other:?}"
                ),
            }
        }
    }
}
