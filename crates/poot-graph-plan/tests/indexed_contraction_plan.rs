//! POOT-761: an `IndexedMatMul` compiles to one generated indexed contraction on every backend, the Gemv at a decode
//! row count and the register-tiled expert walk above it, and a routed gate weighting fuses into it as one
//! `ContractionEpilogue { head: Indexed }`.
//!
//! Every row drives the production `compile` for each backend and reads what `Program::planned()` holds.

use poot_graph_ir::op::{BinOp, ContractionHead, RedOp};
use poot_graph_ir::{Builder, Graph, OpKind, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, KernelChoice, Plan, Program, Target, compile,
};
use poot_kernelgen::{ContractionSpec, IndexedContraction, KernelRequest, Schedule, WeightLayout};
use poot_target::{AmdArch, Backend, BufferStorage, DeviceCaps};
use poot_tensor::DType;

const EXPERTS: usize = 4;
const K: usize = 36;
const N: usize = 21;

fn backends() -> [Backend; 3] {
    [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ]
}

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

fn options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
}

/// `g` compiled for `backend` through the full pipeline.
fn program(g: &Graph, backend: Backend) -> Program {
    compile(g, &target(backend), &options()).expect("the graph compiles")
}

/// `IndexedMatMul(x [m, K], w [E, K, N], ids [m])`, all f32; with `gate`, the product times `gate[m]` (`[m, 1]`
/// after a reshape), the routed gate weighting.
fn indexed(m: usize, gate: bool) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, K, N]))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![m], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let mut out = b.indexed_matmul(x, w, ids).unwrap();
    if gate {
        let gate = b.constant("gate", TensorType::f32(vec![m])).unwrap();
        out = b
            .binary(BinOp::Mul, out, b.reshape(gate, vec![m, 1]).unwrap())
            .unwrap();
    }
    b.finish(out).unwrap()
}

fn is_indexed_head(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::IndexedMatMul
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Indexed,
                ..
            }
    )
}

fn count(program: &Program, want: impl Fn(&OpKind) -> bool) -> usize {
    program.planned().filter(|(eqn, _)| want(&eqn.op)).count()
}

/// The indexed contraction request behind the program's one indexed equation.
fn indexed_request(program: &Program) -> &IndexedContraction {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| is_indexed_head(&eqn.op))
        .expect("the indexed equation is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Indexed(request))) => {
            request
        }
        other => panic!("expected a generated indexed contraction, got {other:?}"),
    }
}

/// SC-003: the decode shape (`M = 5`) plans `Schedule::Gemv` on wgpu, ROCm and PTX, and the prefill shape
/// (`M = 160`) the register-tiled expert walk (`Schedule::RegisterTiled`).
///
/// Mutation: restore the `SpirvVulkan`-only condition on the Gemv choice (`indexed_gemv_in` gated on the backend);
/// ROCm and PTX plan the register-tiled schedule at `M = 5` and the decode row goes red.
#[test]
fn decode_shape_plans_the_gemv_on_every_backend_and_prefill_the_register_tiled_walk() {
    for backend in backends() {
        for (m, gemv) in [(5, true), (160, false)] {
            let label = format!("{backend:?} M={m}");
            let program = program(&indexed(m, false), backend);
            let request = indexed_request(&program);
            assert!(
                match request.schedule {
                    Schedule::Gemv { .. } => gemv,
                    Schedule::RegisterTiled { .. } => !gemv,
                },
                "{label}: {:?}",
                request.schedule
            );
            // `[K, N, M, x_groups, tile_offset, E]`: the shapes the shape-generic body reads.
            assert_eq!(
                [&request.meta[..3], &request.meta[5..6]].concat(),
                [K as u32, N as u32, m as u32, EXPERTS as u32]
            );
        }
    }
}

/// SC-005 (plan half): the routed gate weighting compiles to exactly one `ContractionEpilogue { head: Indexed }`,
/// with no standalone `IndexedMatMul` or `Mul` left, and its request carries the epilogue.
///
/// Mutation: remove the `IndexedMatMul` arm of `epilogue_head`; the fused count falls to zero and this row goes red.
#[test]
fn routed_gate_weighting_compiles_to_one_indexed_contraction_epilogue() {
    for backend in backends() {
        for m in [5, 160] {
            let label = format!("{backend:?} M={m}");
            let g = indexed(m, true);
            let fused = program(&g, backend);
            assert_eq!(
                count(&fused, |op| matches!(
                    op,
                    OpKind::ContractionEpilogue {
                        head: ContractionHead::Indexed,
                        ..
                    }
                )),
                1,
                "{label}"
            );
            assert_eq!(
                count(&fused, |op| matches!(op, OpKind::IndexedMatMul)),
                0,
                "{label}"
            );
            assert_eq!(
                count(&fused, |op| matches!(op, OpKind::Binary(BinOp::Mul))),
                0,
                "{label}"
            );
            assert!(indexed_request(&fused).epilogue.is_some(), "{label}");
        }
    }
}

/// An F32 id operand never reaches the planner: inference admits an I32 id only, with a typed `DtypeOp` error.
///
/// Mutation: admit `F32` in `check_index_dtype` (`op.rs`); the F32 id infers and this row goes red.
#[test]
fn an_f32_id_operand_is_refused_by_inference() {
    let error = OpKind::IndexedMatMul
        .infer(&[
            TensorType::f32(vec![5, K]),
            TensorType::f32(vec![EXPERTS, K, N]),
            TensorType::f32(vec![5]),
        ])
        .expect_err("an F32 id operand must be refused");
    assert!(
        matches!(
            error,
            poot_graph_ir::ShapeError::DtypeOp {
                op: "IndexedMatMul",
                dtype: DType::F32
            }
        ),
        "{error:?}"
    );
}

/// The checkpoint-order spelling: `indexed_matmul(x, transpose(w_enk, [0, 2, 1]), ids)` with `w_enk` `[E, N, K]`.
/// With `second_consumer`, the transposed value is also summed to a scalar (last-axis `Reduce`s, which no kernel
/// reads strided) and that scalar is added to the product.
fn checkpoint_order(m: usize, second_consumer: bool) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, N, K]))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![m], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let w_ekn = b.transpose(w, vec![0, 2, 1]).unwrap();
    let out = b.indexed_matmul(x, w_ekn, ids).unwrap();
    if second_consumer {
        let sum = |t, axis, keepdim| b.reduce(RedOp::Sum, t, axis, keepdim).unwrap();
        let per_expert = sum(sum(w_ekn, 2, false), 1, false);
        let total = b.reshape(sum(per_expert, 0, true), vec![1, 1]).unwrap();
        let joined = b.binary(BinOp::Add, out, total).unwrap();
        return b.finish(joined).unwrap();
    }
    b.finish(out).unwrap()
}

/// The `Transpose` equations that launch a kernel: one that planned as a view moves no data.
fn transposes(program: &Program) -> usize {
    program
        .planned()
        .filter(|(eqn, plan)| {
            matches!(eqn.op, OpKind::Transpose { .. }) && matches!(plan, Plan::Dispatches(_))
        })
        .count()
}

/// POOT-646 SC-002: the checkpoint-order program plans one indexed contraction reading the stored `[E, N, K]`
/// weight (`WeightLayout::Nk`) and no `transpose` dispatch, on every backend, at the Gemv and the register-tiled shapes.
///
/// Mutation: drop the indexed arm of `is_strided_capable` (`predicates.rs`); the `Transpose` is materialized, a
/// `transpose` dispatch appears, the request reads `Kn` and this row goes red.
#[test]
fn checkpoint_order_experts_plan_one_contraction_and_no_transpose() {
    for backend in backends() {
        for m in [1, 5, 160] {
            let label = format!("{backend:?} M={m}");
            let program = program(&checkpoint_order(m, false), backend);
            assert_eq!(count(&program, is_indexed_head), 1, "{label}");
            assert_eq!(transposes(&program), 0, "{label}: a transpose dispatch ran");
            assert_eq!(
                indexed_request(&program).layout,
                WeightLayout::Nk,
                "{label}"
            );
        }
    }
}

/// POOT-646 guard: a transposed weight another consumer also reads is a real value, so the `Transpose` stays
/// materialized and the contraction reads the plain `[E, K, N]` result of it.
///
/// Mutation: in `compute_views`, promote when `any` consumer is strided-capable instead of `all`; the `Transpose`
/// becomes a view the `Reduce` then reads as a plain buffer, the `Transpose` dispatch disappears and this row
/// goes red.
#[test]
fn a_transposed_weight_with_a_second_consumer_keeps_its_transpose() {
    for backend in backends() {
        let program = program(&checkpoint_order(5, true), backend);
        assert_eq!(transposes(&program), 1, "{backend:?}");
        assert_eq!(
            indexed_request(&program).layout,
            WeightLayout::Kn,
            "{backend:?}"
        );
    }
}

/// POOT-646 SC-003: the `[E, K, N]` orientation (no transpose, the gpt-oss layout) plans `WeightLayout::Kn`.
///
/// Mutation: force `WeightLayout::Nk` for every indexed contraction in `weight_layout` (`planner/moe.rs`).
#[test]
fn the_ekn_orientation_plans_the_plain_layout() {
    for backend in backends() {
        for m in [1, 5, 160] {
            let program = program(&indexed(m, false), backend);
            assert_eq!(
                indexed_request(&program).layout,
                WeightLayout::Kn,
                "{backend:?} M={m}"
            );
        }
    }
}

/// POOT-646: a weight view the indexed contraction cannot read is refused with the typed `OperandLayout`, never
/// read as if it were a plain or a transposed stack. A `slice` of the stored `[E, N, K + 3]` weight before the
/// transpose is a view with different strides than the exact `[E, N, K]` transpose.
///
/// Mutation: remove the `Err(OperandLayout)` arm of `weight_layout` (`planner/moe.rs`) by returning `Nk` for any
/// view; the program compiles and this row goes red.
#[test]
fn a_sliced_then_transposed_weight_is_refused_by_layout() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![5, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, N, K + 3]))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![5], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let sliced = b.slice(w, 2, 0, K).unwrap();
    let out = b
        .indexed_matmul(x, b.transpose(sliced, vec![0, 2, 1]).unwrap(), ids)
        .unwrap();
    let g = b.finish(out).unwrap();
    for backend in backends() {
        let error = compile(&g, &target(backend), &options())
            .err()
            .unwrap_or_else(|| panic!("{backend:?}: a strided weight view must be refused"));
        assert!(
            format!("{error:?}").contains("OperandLayout"),
            "{backend:?}: {error:?}"
        );
    }
}

/// POOT-646: the routed gate weighting fuses into one `ContractionEpilogue { head: Indexed }` that reads the
/// checkpoint-order `[E, N, K]` weight in place: no `transpose` dispatch, and the request's layout is `Nk`.
///
/// Mutation: drop the `ContractionEpilogue` arm of `is_strided_capable` (`predicates.rs`); the `Transpose` is
/// materialized and this row goes red.
#[test]
fn a_checkpoint_order_weight_under_the_gate_epilogue_plans_no_transpose() {
    for backend in backends() {
        for m in [5, 160] {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
            let w = b
                .constant("w", TensorType::f32(vec![EXPERTS, N, K]))
                .unwrap();
            let ids = b
                .constant_ranged(
                    "ids",
                    TensorType::new(vec![m], DType::I32),
                    poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
                )
                .unwrap();
            let gate = b.constant("gate", TensorType::f32(vec![m])).unwrap();
            let product = b
                .indexed_matmul(x, b.transpose(w, vec![0, 2, 1]).unwrap(), ids)
                .unwrap();
            let out = b
                .binary(BinOp::Mul, product, b.reshape(gate, vec![m, 1]).unwrap())
                .unwrap();
            let program = program(&b.finish(out).unwrap(), backend);
            let label = format!("{backend:?} M={m}");
            assert_eq!(
                count(&program, |op| matches!(
                    op,
                    OpKind::ContractionEpilogue {
                        head: ContractionHead::Indexed,
                        ..
                    }
                )),
                1,
                "{label}"
            );
            assert_eq!(transposes(&program), 0, "{label}: a transpose dispatch ran");
            let request = indexed_request(&program);
            assert_eq!(request.layout, WeightLayout::Nk, "{label}");
            assert!(request.epilogue.is_some(), "{label}");
        }
    }
}

/// POOT-1028: `IndexedMatMul(x, w, ids)` over a BF16 const expert stack (`[E, K, N]`, or the checkpoint's
/// `[E, N, K]` behind a `Transpose`), optionally under the routed gate weighting. The activation, ids and output
/// stay f32: only the stack is BF16, which is how a BF16 checkpoint loads.
fn bf16_experts(m: usize, checkpoint_order: bool, gate: bool) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![m], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let w = if checkpoint_order {
        let w = b
            .constant("w", TensorType::new(vec![EXPERTS, N, K], DType::BF16))
            .unwrap();
        b.transpose(w, vec![0, 2, 1]).unwrap()
    } else {
        b.constant("w", TensorType::new(vec![EXPERTS, K, N], DType::BF16))
            .unwrap()
    };
    let mut out = b.indexed_matmul(x, w, ids).unwrap();
    if gate {
        let gate = b.constant("gate", TensorType::f32(vec![m])).unwrap();
        out = b
            .binary(BinOp::Mul, out, b.reshape(gate, vec![m, 1]).unwrap())
            .unwrap();
    }
    b.finish(out).unwrap()
}

/// POOT-1028: a BF16 expert stack plans one indexed contraction that reads it as packed BF16 lanes, on every
/// backend, at the Gemv and the register-tiled shapes, in both orientations and under the gate epilogue, and the stack's
/// buffer is the packed lane the request reads: no f32 copy of the stack and no `Cast` over it exists. The
/// checkpoint-order spelling still plans no `transpose` dispatch (the const is packed through its view).
///
/// Mutations: restore the all-f32 operand check in `indexed_matmul_lowers` (`predicates.rs`) and every row is
/// refused (`Refused { op: IndexedMatMul, dtypes: [F32, BF16, I32], missing: DtypeLowering }`); drop the indexed
/// arm of `packed_bf16_reader` (`dtype_widen.rs`) and the AMD and NVPTX consts upload native BF16, which
/// `check_bf16_const_reads` refuses against the packed-lane body.
#[test]
fn a_bf16_expert_stack_plans_the_packed_lane_without_a_widened_copy() {
    for backend in backends() {
        for m in [1, 5, 160] {
            for (checkpoint_order, gate) in
                [(false, false), (true, false), (true, true), (false, true)]
            {
                let label = format!("{backend:?} M={m} nk={checkpoint_order} gate={gate}");
                let program = program(&bf16_experts(m, checkpoint_order, gate), backend);
                assert_eq!(count(&program, is_indexed_head), 1, "{label}");
                assert_eq!(transposes(&program), 0, "{label}: a transpose dispatch ran");
                assert_eq!(
                    count(&program, |op| matches!(op, OpKind::Cast { .. })),
                    0,
                    "{label}: the stack was widened by a cast"
                );
                let request = indexed_request(&program);
                assert_eq!(request.weight, BufferStorage::bf16_packed(), "{label}");
                assert_eq!(
                    request.layout,
                    if checkpoint_order {
                        WeightLayout::Nk
                    } else {
                        WeightLayout::Kn
                    },
                    "{label}"
                );
                assert_eq!(request.epilogue.is_some(), gate, "{label}");
                let graph = program.graph();
                let stack = graph
                    .consts()
                    .iter()
                    .copied()
                    .find(|&id| graph.meta(id).name().as_deref() == Some("w"))
                    .expect("the expert stack is a const");
                assert_eq!(
                    program.storage().storage(stack).buffer_storage(),
                    BufferStorage::bf16_packed(),
                    "{label}: the stack's buffer is not the packed BF16 lane"
                );
            }
        }
    }
}

/// POOT-1028: a BF16 stack that is not a const has no packed upload lane, so it is refused with the typed
/// `DtypeLowering` rather than read as packed words from a native buffer.
///
/// Mutation: accept any BF16 weight in `indexed_matmul_lowers` (drop the `Storage::Const` check); the program
/// compiles and this row goes red.
#[test]
fn a_computed_bf16_expert_stack_is_refused() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![5, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, K, N]))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![5], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let out = b
        .indexed_matmul(x, b.cast(w, DType::BF16).unwrap(), ids)
        .unwrap();
    let g = b.finish(out).unwrap();
    for backend in backends() {
        let error = compile(&g, &target(backend), &options())
            .err()
            .unwrap_or_else(|| panic!("{backend:?}: a computed BF16 stack must be refused"));
        assert!(
            format!("{error:?}").contains("DtypeLowering"),
            "{backend:?}: {error:?}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum SecondReader {
    Reduce,
    Cast,
    Binary,
    Fused,
}

/// POOT-1028: the BF16 `[E, N, K]` const behind a `Transpose` that the indexed contraction shares with a second
/// reader (a last-axis `Reduce`, or a `Cast` to f32 feeding one). That reader reads a plain buffer, so the
/// `Transpose` is not a view: the stack the contraction would read is a computed BF16 value with no packed upload
/// lane.
fn bf16_checkpoint_order_shared(reader: SecondReader) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![5, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![EXPERTS, N, K], DType::BF16))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![5], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let w_ekn = b.transpose(w, vec![0, 2, 1]).unwrap();
    let out = b.indexed_matmul(x, w_ekn, ids).unwrap();
    let shared = match reader {
        SecondReader::Reduce => w_ekn,
        SecondReader::Cast => b.cast(w_ekn, DType::F32).unwrap(),
        // A strided-capable elementwise reader: the view promotion would accept it were the dtype not a gate.
        // A cast and an elementwise step over it fuse into one region that widens its BF16 input in-register,
        // reading it flat: the one reader that would take a BF16 view without refusing it.
        SecondReader::Fused => {
            let other = b
                .constant("other", TensorType::f32(vec![EXPERTS, K, N]))
                .unwrap();
            b.binary(BinOp::Mul, b.cast(w_ekn, DType::F32).unwrap(), other)
                .unwrap()
        }
        SecondReader::Binary => {
            let other = b
                .constant("other", TensorType::new(vec![EXPERTS, K, N], DType::BF16))
                .unwrap();
            b.cast(b.binary(BinOp::Add, w_ekn, other).unwrap(), DType::F32)
                .unwrap()
        }
    };
    let sum = |t, axis, keepdim| b.reduce(RedOp::Sum, t, axis, keepdim).unwrap();
    let per_expert = sum(sum(shared, 2, false), 1, false);
    let total = b.reshape(sum(per_expert, 0, true), vec![1, 1]).unwrap();
    // The `Reduce` form sums BF16; widen its scalar for the join with the f32 product.
    let total = match reader {
        SecondReader::Reduce => b.cast(total, DType::F32).unwrap(),
        SecondReader::Cast | SecondReader::Binary | SecondReader::Fused => total,
    };
    let joined = b.binary(BinOp::Add, out, total).unwrap();
    b.finish(joined).unwrap()
}

/// POOT-1028: a BF16 checkpoint-order stack whose `Transpose` another reader also reads is refused with the typed
/// `DtypeLowering` at the `IndexedMatMul`, identically on wgpu, ROCm and PTX: the transposed value is a materialized BF16 buffer, never a
/// const or a promoted view of one.
///
/// Mutation: force the BF16 admission in `compute_views` (`predicates.rs`) to `true`; the `Binary` reader's
/// `Transpose` becomes a view, the contraction plans, and the refusal moves to the `Binary` (wgpu) or the fused
/// epilogue (ROCm, PTX): the `Binary` rows go red. The other readers are not strided-capable, so they never promote
/// the view and stay green.
#[test]
fn a_bf16_stack_transpose_with_a_second_reader_is_refused_on_every_backend() {
    for reader in [
        SecondReader::Reduce,
        SecondReader::Cast,
        SecondReader::Binary,
        SecondReader::Fused,
    ] {
        let g = bf16_checkpoint_order_shared(reader);
        for backend in backends() {
            let error = compile(&g, &target(backend), &options())
                .err()
                .unwrap_or_else(|| {
                    panic!("{backend:?} {reader:?}: a shared BF16 transpose must be refused")
                });
            // The contraction itself is the refused equation: its weight is neither a const nor a promoted view.
            let error = format!("{error:?}");
            assert!(
                error.contains("op: IndexedMatMul") && error.contains("DtypeLowering"),
                "{backend:?} {reader:?}: {error}"
            );
        }
    }
}

/// POOT-1028: an F16 expert stack has no indexed lowering and is refused `DtypeLowering` on every backend.
///
/// Mutation: admit `DType::F16` in `indexed_matmul_lowers` and map it to `BufferStorage::f16_packed()` in
/// `plan_indexed`; the generator refuses it with `Unsupported` instead, and the row goes red.
#[test]
fn an_f16_expert_stack_is_refused() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![5, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![EXPERTS, K, N], DType::F16))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![5], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let out = b.indexed_matmul(x, w, ids).unwrap();
    let g = b.finish(out).unwrap();
    for backend in backends() {
        let error = compile(&g, &target(backend), &options())
            .err()
            .unwrap_or_else(|| panic!("{backend:?}: an F16 stack must be refused"));
        assert!(
            format!("{error:?}").contains("DtypeLowering"),
            "{backend:?}: {error:?}"
        );
    }
}
