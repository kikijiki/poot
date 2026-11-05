//! POOT-1458: a BF16 weight whose every reader multiplies it in f16 is rounded to f16 once, at bind
//! (`receipts/1458-f16-rounding/design.md` section 4). The storage plan puts the const in the f16 lane and the
//! contraction's request reads that lane, so the kernel stages the weight as stored instead of rounding it per
//! staged value; any other reader keeps the bf16 lane. Every row drives the production `compile` on RADV caps
//! (f16 fragments only, so the narrow policy multiplies in f16).

use poot_graph_ir::op::BinOp;
use poot_graph_ir::{Builder, Graph, MatmulPrecision, OpKind, Operand, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{ContractionSpec, DenseTiledContraction, DenseTiledWeight, KernelRequest};
use poot_target::{Backend, BufferStorage, DeviceCaps};
use poot_tensor::DType;

const M: usize = 8;
const K: usize = 64;
const N: usize = 32;

fn radv() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
    }
}

/// `x [M, K] @ w^T` over a BF16 const `w [N, K]`; with `cast_reader`, plus `x @ f32(w)^T`, so `w` also has a
/// reader that widens it to f32.
fn projection(cast_reader: bool) -> Graph {
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![M, K], DType::F32))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let mut out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    if cast_reader {
        let wide = b.cast(w, DType::F32).unwrap();
        let other = b.matmul(x, b.transpose(wide, vec![1, 0]).unwrap()).unwrap();
        out = b.binary(BinOp::Add, out, other).unwrap();
    }
    b.finish(out).unwrap()
}

fn compiled(g: &Graph, policy: MatmulPrecision) -> Program {
    compile(
        g,
        &radv(),
        &CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy),
    )
    .expect("the projection compiles")
}

/// The buffer storage of the program's BF16 const, and the weight storage of each request of a contraction that
/// reads it.
fn weight_lanes(program: &Program) -> (BufferStorage, Vec<BufferStorage>) {
    let g = program.graph();
    let w = g
        .values()
        .ids()
        .find(|&id| g.meta(id).storage.holds_weight_bytes() && g.aval(id).dtype == DType::BF16)
        .expect("the BF16 weight");
    let requests = program
        .planned()
        .filter(|(eqn, _)| {
            eqn.op.dense_weight().is_some()
                && matches!(eqn.inputs.get(1), Some(Operand::Value(read)) if *read == w)
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
        .collect();
    (program.storage().storage(w).buffer_storage(), requests)
}

/// A BF16 weight read only by a contraction that multiplies in f16 is held in the f16 lane, and the contraction's
/// request reads that lane (so its body stages the weight unrounded): the buffer and the body are one decision.
/// The same weight under `Exact`, or with a reader that widens it to f32, keeps the packed bf16 lane in both.
/// Mutation: in `planner/packed.rs` make `bf16_weight_storage` always return `StorageKind::Bf16Packed`; the
/// request reads bf16 words from an f16 buffer and this row goes red.
#[test]
fn a_bf16_weight_of_f16_contractions_is_one_f16_lane_for_its_buffer_and_its_body() {
    let narrow = compiled(&projection(false), MatmulPrecision::Narrow);
    assert!(
        narrow
            .planned()
            .any(|(eqn, _)| matches!(eqn.op, OpKind::DenseContraction { .. })
                || matches!(eqn.op, OpKind::ContractionEpilogue { .. })),
        "the projection folds to a dense contraction"
    );
    assert_eq!(
        weight_lanes(&narrow),
        (
            BufferStorage::f16_packed(),
            vec![BufferStorage::f16_packed()]
        )
    );
    for (label, g, policy) in [
        ("exact", projection(false), MatmulPrecision::Exact),
        ("a cast reader", projection(true), MatmulPrecision::Narrow),
    ] {
        let (buffer, requests) = weight_lanes(&compiled(&g, policy));
        assert_eq!(buffer, BufferStorage::bf16_packed(), "{label}");
        assert_eq!(
            requests,
            vec![BufferStorage::bf16_packed()],
            "{label}: the contraction reads the bf16 lane"
        );
    }
}
