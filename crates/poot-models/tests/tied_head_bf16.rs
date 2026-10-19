//! POOT-651: the tied lm_head of a BF16 checkpoint plans on the dense BF16 Gemv over its BF16-resident weight,
//! on the real Qwen2.5-0.5B decode trace through `compile`. (Moved from `poot-graph-plan/src/dtype_widen.rs`:
//! a test that traces a model sits above the planner, POOT-1130.)

use poot_graph_ir::{OpKind, Operand, Storage};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::test_support::dense::{Dense, Family, step};
use poot_target::{AmdArch, Backend, BufferStorage};
use poot_tensor::DType;
use poot_test_util::device_caps::default_caps_for;

use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Target, compile, numel};

const VOCAB: usize = 151936;
const HIDDEN: usize = 896;

/// Qwen2.5-0.5B's dimensions over a natively BF16 checkpoint; the head is tied, so it is its own `w.head`
/// const of the embedding's values.
fn qwen2_5_0_5b_decode() -> poot_graph_ir::Graph {
    let dense = Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(HIDDEN, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(32768);
    dense
        .zeroed_model(DType::BF16)
        .model
        .trace(
            &StepSpec::step(step(1, 1, 64, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// POOT-651 SC-001, at planning, through `compile` on wgpu and ROCm: the head contraction is the dense Gemv over
/// the `[vocab, hidden]` weight in checkpoint order, reading packed BF16 storage, and the program's constants
/// hold no F32 buffer of the head's element count: the head's residency is `vocab * hidden * 2` bytes.
///
/// Mutation: in `fold_dense_contractions`, skip the fold for a BF16 weight (`continue` before the
/// `DENSE_CONTRACTION_WEIGHT_DTYPES` check); the head stays a `Transpose` and a mixed-dtype `MatMul` over the BF16
/// weight, which no backend lowers (no pass retypes a const), so `compile` refuses it and this row goes red.
#[test]
fn the_tied_head_decode_runs_the_bf16_gemv_over_bf16_resident_weight() {
    let graph = qwen2_5_0_5b_decode();
    let options = CompileOptions::new(CompileLimits::STANDARD);
    for backend in [Backend::SpirvVulkan, Backend::AmdGcn(AmdArch::gfx1151())] {
        let target = Target::Gpu {
            backend,
            caps: default_caps_for(backend),
        };
        let program = compile(&graph, &target, &options).expect("the decode graph compiles");
        let (eqn, _) = program
            .planned()
            .find(|(eqn, _)| program.graph().aval(eqn.out).shape.last() == Some(&VOCAB))
            .expect("the head contraction is planned");
        assert_eq!(
            eqn.op,
            OpKind::DenseContraction {
                weight: DType::BF16
            },
            "{backend:?}: the head reads the weight in checkpoint order"
        );
        let KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
            weight,
            n,
            ..
        })) = program.kernel_choice(eqn)
        else {
            panic!(
                "{backend:?}: the head is the dense Gemv, got {:?}",
                program.kernel_choice(eqn)
            );
        };
        assert_eq!(*n, VOCAB, "{backend:?}");
        assert_eq!(*weight, BufferStorage::bf16_packed(), "{backend:?}");

        let Operand::Value(head) = eqn.inputs[1] else {
            panic!("the head weight is a value");
        };
        assert_eq!(
            program.storage().storage(head).buffer_storage(),
            BufferStorage::bf16_packed(),
            "{backend:?}: the head weight is resident as packed BF16"
        );
        let g = program.graph();
        let head_elems = VOCAB * HIDDEN;
        assert!(
            g.values().iter_enumerated().all(|(id, meta)| {
                meta.storage != Storage::Const
                    || g.aval(id).dtype != DType::F32
                    || numel(&g.aval(id).shape) < head_elems
            }),
            "{backend:?}: no F32 constant is as large as the head weight"
        );
        assert_eq!(
            (g.aval(head).dtype, g.aval(head).shape.as_slice()),
            (DType::BF16, [VOCAB, HIDDEN].as_slice()),
            "{backend:?}: the head weight is the BF16 [vocab, hidden] const"
        );
    }
}
