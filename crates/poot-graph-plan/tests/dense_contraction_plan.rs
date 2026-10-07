//! Card 645: a projection against an F32 weight held in checkpoint `[N, K]` orientation compiles to one
//! `DenseContraction` that reads the weight as stored, never a device `transpose` of it, and the fold refuses a
//! transposed value something else reads. Card 1007: the same holds for an F16 weight, stored packed two elements
//! per `u32` word, which folds only when the contraction is its sole reader.
//!
//! Every row drives the production `compile` for each backend and reads what `Program::planned()` holds.

use poot_graph_ir::{BinOp, Builder, Graph, OpKind, Operand, RedOp, StateRole, TensorType, Traced};
use poot_graph_plan::{
    CompileError, CompileOptions, KernelChoice, PlanError, Program, StorageGap, Submission, Target,
    compile,
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
    Target {
        backend,
        caps: match backend {
            Backend::SpirvVulkan => DeviceCaps::wgpu_rdna3_igpu(),
            Backend::Nvptx => DeviceCaps::ptx_default(),
            Backend::AmdGcn(_) => DeviceCaps::rocm_default(),
        },
    }
}

/// `g` compiled for `backend`: the full pipeline, or with `guarded` the routed-expert hang guard (the graph
/// then carries a routed-expert site and the target has the hang).
fn program(g: &Graph, backend: Backend, guarded: bool) -> Program {
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let base = target(backend);
    let (g, caps) = poot_test_util::graph_fixtures::pipeline_input(g, base.caps, guarded);
    compile(&g, &Target { caps, ..base }, &options).expect("the graph compiles")
}

/// `matmul(x, transpose(w))`, `w` an F32 `[N, K]` constant.
fn projection(m: usize) -> Graph {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, K]));
    let w = b.constant("w", TensorType::f32(vec![N, K]));
    let out = b.matmul(x, b.transpose(w, vec![1, 0]));
    b.finish(out)
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
/// order: the decode GEMV at `M = 1`, the tiled GEMM above it, and the one-thread-per-output kernel where the
/// compile's fusion policy keeps a matmul off the tiled GEMM.
///
/// Mutation: remove `DType::F32` from `DENSE_CONTRACTION_WEIGHT_DTYPES`; the fold skips, a `Transpose` and a
/// `MatMul` stay in the program and this row goes red.
#[test]
fn checkpoint_orientation_projection_plans_one_dense_contraction_and_no_transpose() {
    for backend in backends() {
        for m in [1, 4, 33] {
            let program = program(&projection(m), backend, false);
            let label = format!("{backend:?} M={m}: {:?}", op_names(&program));
            assert_eq!(count(&program, is_dense_f32), 1, "{label}");
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::Transpose { .. })),
                0,
                "{label}"
            );
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::MatMul)),
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

/// The same projection under `true` keeps off the tiled GEMM (cards 186/192), so an
/// `M > 1` contraction takes the one-thread-per-output `[N, K]` kernel.
#[test]
fn checkpoint_orientation_projection_under_the_moe_hang_guard_is_serial() {
    for backend in backends() {
        let program = program(&projection(33), backend, true);
        assert_eq!(count(&program, is_dense_f32), 1, "{backend:?}");
        let request = request(&program, is_dense_f32);
        assert!(
            matches!(
                request,
                KernelRequest::Contraction(ContractionSpec::DenseSerial { .. })
            ),
            "{backend:?}: {request:?}"
        );
    }
}

/// SC-003: a transposed weight something else also reads is a real materialization, so the fold leaves the
/// `Transpose` and the `MatMul` in the plan. The weight's transpose feeds the matmul and a reduction.
///
/// Mutation: drop the consumer-count check in `fold_dense_contractions`; the transpose is dropped under its
/// second reader, no `Transpose` is planned and this row goes red.
#[test]
fn a_transposed_weight_with_a_second_consumer_keeps_the_transpose_and_the_matmul() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, K]));
    let w = b.constant("w", TensorType::f32(vec![N, K]));
    let wt = b.transpose(w, vec![1, 0]);
    let y = b.matmul(x, wt);
    let column_sums = b.reduce(RedOp::Sum, wt, 0, false);
    let row_sums = b.reduce(RedOp::Sum, y, 0, false);
    let total = b.binary(BinOp::Add, row_sums, column_sums);
    let g = b.finish(total);
    for backend in backends() {
        let program = program(&g, backend, false);
        let label = format!("{backend:?}: {:?}", op_names(&program));
        assert_eq!(count(&program, is_dense_f32), 0, "{label}");
        assert!(
            program
                .planned()
                .any(|(eqn, _)| matches!(eqn.op, OpKind::Transpose { .. })),
            "{label}"
        );
        assert_eq!(
            count(&program, |op| matches!(op, OpKind::MatMul)),
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, K]));
    let w = b.constant("w", TensorType::f32(vec![N, K]));
    let kept = b.state_input("kept", TensorType::f32(vec![K, N]), StateRole::Recurrent);
    let wt = b.transpose(w, vec![1, 0]);
    let y = b.matmul(x, wt);
    let g = b.finish_with_state(y, &[(kept, wt)]);
    for backend in backends() {
        let program = program(&g, backend, false);
        let label = format!("{backend:?}: {:?}", op_names(&program));
        assert_eq!(count(&program, is_dense_f32), 0, "{label}");
        assert!(
            program
                .planned()
                .any(|(eqn, _)| matches!(eqn.op, OpKind::Transpose { .. })),
            "{label}"
        );
        assert_eq!(
            count(&program, |op| matches!(op, OpKind::MatMul)),
            1,
            "{label}"
        );
    }
}

/// The scalar sum of a rank-2 value, so two differently shaped readers can meet in one output.
fn total_sum(b: &Builder, value: Traced) -> Traced {
    b.reduce(RedOp::Sum, b.reduce(RedOp::Sum, value, 1, false), 0, false)
}

/// `matmul(x, transpose(w))`, `w` an F16 `[N, K]` constant.
fn f16_projection(m: usize) -> Graph {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, K]));
    let w = b.constant("w", TensorType::new(vec![N, K], DType::F16));
    let out = b.matmul(x, b.transpose(w, vec![1, 0]));
    b.finish(out)
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
/// Mutation: remove `DType::F16` from `DENSE_CONTRACTION_WEIGHT_DTYPES`; the fold skips, a `Transpose` stays in the
/// program and this row goes red.
#[test]
fn an_f16_checkpoint_projection_plans_one_packed_dense_contraction_and_no_transpose() {
    for backend in backends() {
        for m in [1, 4, 33] {
            let program = program(&f16_projection(m), backend, false);
            let label = format!("{backend:?} M={m}: {:?}", op_names(&program));
            assert_eq!(count(&program, is_dense_f16), 1, "{label}");
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::Transpose { .. })),
                0,
                "{label}"
            );
            assert_eq!(
                count(&program, |op| matches!(op, OpKind::MatMul)),
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

/// Card 1007: under `true` the F16 contraction at `M > 1` takes the one-thread-per-output
/// `[N, K]` kernel, reading the same packed weight.
#[test]
fn an_f16_checkpoint_projection_under_the_moe_hang_guard_is_serial_over_packed_words() {
    for backend in backends() {
        let program = program(&f16_projection(33), backend, true);
        assert_eq!(count(&program, is_dense_f16), 1, "{backend:?}");
        let request = request(&program, is_dense_f16);
        assert!(
            matches!(
                request,
                KernelRequest::Contraction(ContractionSpec::DenseSerial { .. })
            ),
            "{backend:?}: {request:?}"
        );
        let (named, planned) = weight_storages(&program);
        assert_eq!(planned, BufferStorage::f16_packed(), "{backend:?}");
        assert_eq!(named, planned, "{backend:?}");
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, K]));
    let w = b.constant("w", TensorType::new(vec![N, K], DType::F16));
    let ids = b.constant("ids", TensorType::new(vec![3], DType::I32));
    let rows = b.cast(b.gather(w, 0, ids), DType::F32);
    let y = b.matmul(x, b.transpose(w, vec![1, 0]));
    let total = b.binary(BinOp::Add, total_sum(&b, y), total_sum(&b, rows));
    let g = b.finish(total);
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, K]));
    let w = b.constant("w", TensorType::new(vec![N, K], DType::F16));
    let y = b.matmul(x, b.transpose(w, vec![1, 0]));
    let widened = b.cast(w, DType::F32);
    let total = b.binary(BinOp::Add, total_sum(&b, y), total_sum(&b, widened));
    let mut g = b.finish(total);
    // The contraction the fold would make of the projection, written into the graph directly: the fold itself
    // never makes this one, because `w` has a second reader. (The transpose stays, now unread.)
    let matmul = g
        .eqns
        .iter_mut()
        .find(|eqn| matches!(eqn.op, OpKind::MatMul))
        .expect("the traced graph multiplies");
    matmul.op = OpKind::DenseContraction { weight: DType::F16 };
    matmul.inputs[1] = Operand::Value(w.id);
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    for backend in backends() {
        match compile(&g, &target(backend), &options) {
            Err(CompileError::Plan(plan)) => match *plan {
                PlanError::UnrepresentableValue {
                    value,
                    dtype: DType::F16,
                    gap: StorageGap::F16PackedLanes,
                } => assert_eq!(value, w.id, "{backend:?}"),
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
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, K]));
    let w = b.constant("w", TensorType::new(vec![N, K], weight));
    let bias = b.constant("bias", TensorType::f32(vec![N]));
    let residual = b.constant("residual", TensorType::f32(vec![m, N]));
    let projection = b.matmul(x, b.transpose(w, vec![1, 0]));
    let out = b.binary(BinOp::Add, b.binary(BinOp::Add, projection, bias), residual);
    b.finish(out)
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
                let program = program(&biased_residual_projection(m, weight), backend, false);
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
/// `TiledRegion` carries both steps and both leaves, on every backend; the `MoeHangGuard` policy keeps the
/// head and both adds apart.
///
/// Mutation: make `epilogue_head_lowers` accept only `MatMulRoute::Serial` for a `MatMul` head (the pre-1036
/// gate); the head and both adds stay apart and this row goes red.
#[test]
fn a_biased_residual_tiled_matmul_head_is_one_epilogue_contraction() {
    for backend in backends() {
        for m in [5, 160] {
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32(vec![m, K]));
            let w = b.constant("w", TensorType::f32(vec![K, N]));
            let bias = b.constant("bias", TensorType::f32(vec![N]));
            let residual = b.constant("residual", TensorType::f32(vec![m, N]));
            let head = b.matmul(x, w);
            let out = b.binary(BinOp::Add, b.binary(BinOp::Add, head, bias), residual);
            let g = b.finish(out);
            let fused = program(&g, backend, false);
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
            let unfused = program(&g, backend, true);
            assert_eq!(unfused.planned().count(), 3, "{label}: guard keeps 3 eqns");
        }
    }
}

/// `(x @ w + bias) + residual` with `x: [xs]`, `w: [ws]` of `head_dtype` (bias and residual of the head's dtype
/// and the output's shape), the shape of a `MatMul` head the generated tiled GEMM must not take.
fn biased_residual_matmul(xs: Vec<usize>, ws: Vec<usize>, head_dtype: DType) -> Graph {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(xs.clone(), head_dtype));
    let w = b.constant("w", TensorType::new(ws.clone(), head_dtype));
    let n = *ws.last().unwrap();
    let mut out_shape = xs;
    *out_shape.last_mut().unwrap() = n;
    let bias = b.constant("bias", TensorType::new(vec![n], head_dtype));
    let residual = b.constant("residual", TensorType::new(out_shape, head_dtype));
    let out = b.binary(
        BinOp::Add,
        b.binary(BinOp::Add, b.matmul(x, w), bias),
        residual,
    );
    b.finish(out)
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
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
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
