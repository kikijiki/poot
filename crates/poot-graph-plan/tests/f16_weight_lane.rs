//! POOT-1477: the driver stores its dense BF16 weights in one format (POOT-1471). Under the narrow policy on a unit that multiplies in f16, a BF16 weight that
//! only dense contractions and embedding row gathers read is declared F16 by `compile` (the bind converts the
//! checkpoint's bf16 words once, POOT-1458), whichever schedule the readers plan; any other BF16 weight keeps the
//! checkpoint's format. Every row drives the production `compile` on RADV caps (f16 fragments only).

use poot_graph_ir::op::BinOp;
use poot_graph_ir::{Builder, Graph, MatmulPrecision, OpKind, Operand, Slot, TensorType, ValueId};
use poot_graph_plan::{
    CompileLimits, CompileOptions, ImportedKernel, KernelChoice, Program, Target, compile,
};
use poot_kernelgen::{ContractionSpec, DenseTiledContraction, DenseTiledWeight, KernelRequest};
use poot_target::{Backend, BufferStorage};
use poot_tensor::DType;

const K: usize = 64;
const N: usize = 32;
const VOCAB: usize = 48;

fn radv() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

/// A tied-embedding LM step with `m` rows: `embed` gathered by token and widened, `h = x @ w^T` over a BF16
/// projection, `logits = h @ head^T` over the tied BF16 head (its own const of the embedding's values), and with
/// `cast_reader` also `x @ f32(w)^T`, so `w` has a reader that widens it.
fn lm_step(m: usize, cast_reader: bool) -> Graph {
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
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let head = b
        .constant("head", TensorType::new(vec![VOCAB, N], DType::BF16))
        .unwrap();
    let mut h = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    if cast_reader {
        let wide = b.cast(w, DType::F32).unwrap();
        let other = b.matmul(x, b.transpose(wide, vec![1, 0]).unwrap()).unwrap();
        h = b.binary(BinOp::Add, h, other).unwrap();
    }
    let logits = b.matmul(h, b.transpose(head, vec![1, 0]).unwrap()).unwrap();
    b.finish(logits).unwrap()
}

fn compiled(g: &Graph, policy: MatmulPrecision) -> Program {
    compile(
        g,
        &radv(),
        &CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy),
    )
    .expect("the step compiles")
}

/// The constant named `name` of the program's graph.
fn constant(program: &Program, name: &str) -> ValueId {
    let g = program.graph();
    g.values()
        .ids()
        .find(|&id| g.meta(id).name().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no constant {name}"))
}

/// `(declared dtype, checkpoint dtype, buffer storage)` of the program's constant `name`.
fn format_of(program: &Program, name: &str) -> (DType, DType, BufferStorage) {
    let id = constant(program, name);
    let meta = program.graph().meta(id);
    (
        meta.aval.dtype,
        meta.stored_dtype(),
        program.storage().storage(id).buffer_storage(),
    )
}

/// The weight storage of each request of the contractions that read `name` as their weight.
fn contraction_lanes(program: &Program, name: &str) -> Vec<BufferStorage> {
    let id = constant(program, name);
    program
        .planned()
        .filter(|(eqn, _)| {
            eqn.op.dense_weight().is_some()
                && matches!(eqn.inputs.get(1), Some(Operand::Value(read)) if *read == id)
        })
        .map(|(eqn, _)| match program.kernel_choice(eqn) {
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::DenseTiled(DenseTiledContraction {
                    weight: DenseTiledWeight::Nk(storage),
                    ..
                })
                | ContractionSpec::DenseGemv {
                    weight: storage, ..
                },
            )) => *storage,
            other => panic!("{}: {other:?}", eqn.op.kind_name()),
        })
        .collect()
}

const F16: (DType, DType, BufferStorage) = (DType::F16, DType::BF16, BufferStorage::f16_packed());
const BF16: (DType, DType, BufferStorage) =
    (DType::BF16, DType::BF16, BufferStorage::bf16_packed());

/// A BF16 weight only dense contractions read, and an embedding table only a row gather reads, are declared F16
/// over the BF16 checkpoint and held in the packed f16 lane, and the contraction's request reads that lane: the
/// buffer, the body and the graph's declaration are one decision. The same weights under `Exact`, or a weight with
/// a reader that widens it to f32, keep the checkpoint's bf16 format.
///
/// Mutation: in `passes/weight_format.rs` make `reads_as_dense_weight` always true; the weight with the cast
/// reader is converted and its widening cast reads an f16 const, so the row goes red.
#[test]
fn a_bf16_weight_of_dense_readers_is_declared_f16_under_narrow() {
    let narrow = compiled(&lm_step(8, false), MatmulPrecision::Narrow);
    assert_eq!(format_of(&narrow, "w"), F16);
    assert_eq!(format_of(&narrow, "head"), F16);
    assert_eq!(format_of(&narrow, "embed"), F16);
    assert_eq!(
        contraction_lanes(&narrow, "w"),
        vec![BufferStorage::f16_packed()]
    );
    assert!(
        narrow
            .planned()
            .any(|(eqn, _)| matches!(eqn.op, OpKind::DenseRowGather { source: DType::F16 })),
        "the embedding gathers from the f16 table"
    );

    let exact = compiled(&lm_step(8, false), MatmulPrecision::Exact);
    for name in ["w", "head", "embed"] {
        assert_eq!(format_of(&exact, name), BF16, "exact {name}");
    }

    let widened = compiled(&lm_step(8, true), MatmulPrecision::Narrow);
    assert_eq!(format_of(&widened, "w"), BF16, "a weight a cast reads");
    assert_eq!(format_of(&widened, "head"), F16);
}

/// The format is the driver's, not the program's: a prefill-shaped step (many rows, fragment tilings) and a
/// decode-shaped one (one row, a GEMV) of the same model declare every weight alike, so the two programs of a
/// driver upload one copy of each weight. The decision reads the policy, the units and the kinds of a weight's
/// readers, never a shape.
///
/// Mutation: in `store_weights_f16` skip a weight when its reader plans a GEMV (`m == 1`); the decode program
/// keeps bf16 weights and the row goes red.
#[test]
fn prefill_and_decode_programs_declare_every_weight_alike() {
    let prefill = compiled(&lm_step(64, false), MatmulPrecision::Narrow);
    let decode = compiled(&lm_step(1, false), MatmulPrecision::Narrow);
    for name in ["w", "head", "embed"] {
        assert_eq!(
            format_of(&prefill, name),
            format_of(&decode, name),
            "{name}"
        );
    }
    assert_eq!(format_of(&decode, "embed"), F16);
}

/// Two consts over one stored tensor (a tied embedding and head) get one format, so they share one upload: both
/// are the packed f16 lane of the same checkpoint dtype, the residency key's `storage`. Before the driver decided
/// once, the embedding kept bf16 words and the head took f16, so the aliased bytes uploaded twice (0.78 GB on
/// qwen3-4b). Mutation: in `reads_as_dense_weight` accept only contractions; the embedding stays bf16 and the
/// formats differ.
#[test]
fn a_tied_embedding_and_head_have_one_weight_format() {
    let program = compiled(&lm_step(8, false), MatmulPrecision::Narrow);
    assert_eq!(
        format_of(&program, "embed"),
        format_of(&program, "head"),
        "an embedding and its tied head upload one copy only if they are planned in one lane"
    );
}

/// The embedding gather of the f16 table is the packed-f16 row gather on every backend that has typed packed
/// walks (SPIR-V, ROCm, PTX), planned from the production `compile`: the table's buffer and the kernel that
/// reads it are one decision.
///
/// Mutation: in `planner/packed.rs` plan a `DenseRowGather { source: F16 }` with `PackedBf16RowGather`; the
/// kernel reads bf16 words from an f16 lane and this row goes red.
#[test]
fn an_f16_embedding_gathers_with_the_packed_f16_kernel_on_every_backend() {
    use poot_target::AmdArch;
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let target = Target::Gpu {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        let program = compile(
            &lm_step(8, false),
            &target,
            &CompileOptions::new(CompileLimits::STANDARD)
                .with_matmul_precision(MatmulPrecision::Narrow),
        )
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        let (gather, _) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::DenseRowGather { .. }))
            .expect("the embedding gathers");
        assert_eq!(
            gather.op,
            OpKind::DenseRowGather { source: DType::F16 },
            "{backend:?}"
        );
        assert!(
            matches!(
                program.kernel_choice(gather),
                KernelChoice::Imported {
                    kernel: ImportedKernel::PackedF16RowGather,
                    ..
                }
            ),
            "{backend:?}: {:?}",
            program.kernel_choice(gather)
        );
    }
}
