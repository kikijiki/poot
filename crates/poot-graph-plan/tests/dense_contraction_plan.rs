//! Card 645: a projection against an F32 weight held in checkpoint `[N, K]` orientation compiles to one
//! `DenseContraction` that reads the weight as stored, never a device `transpose` of it, and the fold refuses a
//! transposed value something else reads. Card 1007: the same holds for an F16 weight, stored packed two elements
//! per `u32` word, which folds only when the contraction is its sole reader.
//!
//! Every row drives the production `compile` for each backend and reads what `Program::planned()` holds.

use poot_graph_ir::{BinOp, Builder, Graph, OpKind, Operand, RedOp, StateRole, TensorType, Traced};
use poot_graph_plan::{
    CompileError, CompileOptions, FusionPolicy, KernelChoice, PlanError, Program, StorageGap,
    Submission, Target, compile,
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

fn program(g: &Graph, backend: Backend, fusion: FusionPolicy) -> Program {
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    compile(g, &target(backend), &options).expect("the graph compiles")
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

/// The generated requests behind one equation's plan, one per chunk.
fn requests(program: &Program, want: impl Fn(&OpKind) -> bool) -> Vec<&KernelRequest> {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| want(&eqn.op))
        .expect("the equation is planned");
    let leaves = match program.kernel_choice(eqn) {
        KernelChoice::Chunked(chunks) => chunks.iter().collect(),
        other => vec![other],
    };
    leaves
        .into_iter()
        .map(|choice| match choice {
            KernelChoice::Generated(request) => request,
            other => panic!("expected a generated kernel, got {other:?}"),
        })
        .collect()
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
            let program = program(&projection(m), backend, FusionPolicy::Full);
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
            let requests = requests(&program, is_dense_f32);
            let gemv = m == 1;
            assert!(
                requests.iter().all(|request| match request {
                    KernelRequest::Contraction(ContractionSpec::DenseGemv { .. }) => gemv,
                    KernelRequest::Contraction(ContractionSpec::DenseTiled { .. }) => !gemv,
                    _ => false,
                }),
                "{label}: {requests:?}"
            );
        }
    }
}

/// The same projection under `FusionPolicy::MoeHangGuard` keeps off the tiled GEMM (cards 186/192), so an
/// `M > 1` contraction takes the one-thread-per-output `[N, K]` kernel.
#[test]
fn checkpoint_orientation_projection_under_the_moe_hang_guard_is_serial() {
    for backend in backends() {
        let program = program(&projection(33), backend, FusionPolicy::MoeHangGuard);
        assert_eq!(count(&program, is_dense_f32), 1, "{backend:?}");
        let requests = requests(&program, is_dense_f32);
        assert!(
            requests.iter().all(|request| matches!(
                request,
                KernelRequest::Contraction(ContractionSpec::DenseSerial { .. })
            )),
            "{backend:?}: {requests:?}"
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
        let program = program(&g, backend, FusionPolicy::Full);
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
        let program = program(&g, backend, FusionPolicy::Full);
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
fn weight_storages(program: &Program) -> (Vec<BufferStorage>, BufferStorage) {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| is_dense_f16(&eqn.op))
        .expect("the contraction is planned");
    let Operand::Value(weight) = eqn.inputs[1] else {
        panic!("the weight operand is a value");
    };
    let named = requests(program, is_dense_f16)
        .into_iter()
        .map(|request| match request {
            KernelRequest::Contraction(
                ContractionSpec::DenseGemv { weight, .. }
                | ContractionSpec::DenseTiled { weight, .. }
                | ContractionSpec::DenseSerial { weight, .. },
            ) => *weight,
            other => panic!("expected a dense-contraction request, got {other:?}"),
        })
        .collect();
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
            let program = program(&f16_projection(m), backend, FusionPolicy::Full);
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
            let requests = requests(&program, is_dense_f16);
            assert!(
                requests.iter().all(|request| match request {
                    KernelRequest::Contraction(ContractionSpec::DenseGemv { .. }) => gemv,
                    KernelRequest::Contraction(ContractionSpec::DenseTiled { .. }) => !gemv,
                    _ => false,
                }),
                "{label}: {requests:?}"
            );
            let (named, planned) = weight_storages(&program);
            assert_eq!(planned, BufferStorage::f16_packed(), "{label}");
            assert!(
                named.iter().all(|&storage| storage == planned),
                "{label}: requests name {named:?}, the weight is planned {planned}"
            );
        }
    }
}

/// Card 1007: under `FusionPolicy::MoeHangGuard` the F16 contraction at `M > 1` takes the one-thread-per-output
/// `[N, K]` kernel, reading the same packed weight.
#[test]
fn an_f16_checkpoint_projection_under_the_moe_hang_guard_is_serial_over_packed_words() {
    for backend in backends() {
        let program = program(&f16_projection(33), backend, FusionPolicy::MoeHangGuard);
        assert_eq!(count(&program, is_dense_f16), 1, "{backend:?}");
        let requests = requests(&program, is_dense_f16);
        assert!(
            requests.iter().all(|request| matches!(
                request,
                KernelRequest::Contraction(ContractionSpec::DenseSerial { .. })
            )),
            "{backend:?}: {requests:?}"
        );
        let (named, planned) = weight_storages(&program);
        assert_eq!(planned, BufferStorage::f16_packed(), "{backend:?}");
        assert_eq!(named, vec![planned], "{backend:?}");
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
        fusion: FusionPolicy::Full,
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
        fusion: FusionPolicy::Full,
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
