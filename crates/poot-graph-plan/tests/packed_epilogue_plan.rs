//! POOT-1626: a contraction head takes an epilogue wherever its own generator carries one, so the packed prefill
//! contraction (the block-diagonal tiled schedule) fuses its bias and residual into the store like the packed Gemv
//! does, and a head whose generator carries none (the decode attention `scores @ V` reduction) keeps its chain apart.
//!
//! Every row drives the production `compile` for each backend and reads what `Program::planned()` holds.

use poot_graph_ir::op::ContractionHead;
use poot_graph_ir::test_support::packed_graphs::packed_linear;
use poot_graph_ir::{BinOp, Builder, Graph, MatmulPrecision, OpKind, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest, PackedContraction, PackedRows, Schedule};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::{AmdArch, Backend, DeviceCaps};

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
            Backend::SpirvVulkan => poot_test_util::device_caps::wgpu_rdna3_igpu(),
            Backend::Nvptx => DeviceCaps::ptx_default(),
            Backend::AmdGcn(_) => DeviceCaps::rocm_default(),
        },
    }
}

/// `g` compiled for `backend` through the full pipeline, under the exact policy (the plans below are the exact
/// lowering's).
fn program(g: &Graph, backend: Backend) -> Program {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact);
    compile(g, &target(backend), &options).expect("the graph compiles")
}

fn op_names(program: &Program) -> Vec<String> {
    program.planned().map(|(eqn, _)| eqn.op.name()).collect()
}

/// A packed `[N, K]` Q8_0 linear over `m` rows, its bias and a residual of the output's shape added after it.
fn packed_biased_residual(m: usize) -> Graph {
    const K: usize = 64;
    const N: usize = 40;
    let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, [N, K]).unwrap();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![N])).unwrap();
    let residual = b.constant("residual", TensorType::f32(vec![m, N])).unwrap();
    let linear = packed_linear(&b, x, "layer", descriptor, None, Some(bias)).unwrap();
    let out = b.binary(BinOp::Add, linear, residual).unwrap();
    b.finish(out).unwrap()
}

/// POOT-1626: a packed contraction with a bias and a residual plans ONE `ContractionEpilogue` over its packed head at
/// every `M`: the packed Gemv at `M = 1` and the block-diagonal tiled schedule above it (a single partial row tile
/// at `M = 5`, several at `M = 160`). The epilogue carries both steps and both leaves in the store, so the plan holds
/// one dispatch: no standalone `add` runs and the head result never round-trips through memory.
///
/// Mutation: restore the refusal in `packed_contraction`'s tiled arm (`Err(KernelGenError::Unsupported { reason: "a
/// packed tiled contraction has no epilogue" })` when `epilogue.is_some()`): the planner cannot plan the fused
/// head, reverts the fusion, and the `M > 1` rows go red on the dispatch count (`SpirvVulkan M=5:
/// ["packed_contraction q8_0", "fused n_in=3 steps=2"]: left 2, right 1`). Or make `ContractionSpec::accepts_epilogue`
/// answer `false` for `Packed`: every row goes red the same way, at `M = 1` first.
#[test]
fn a_packed_prefill_contraction_fuses_its_bias_and_residual_into_the_tiled_store() {
    for backend in backends() {
        for m in [1, 5, 160] {
            let program = program(&packed_biased_residual(m), backend);
            let label = format!("{backend:?} M={m}: {:?}", op_names(&program));
            assert_eq!(program.planned().count(), 1, "{label}");
            let (eqn, _) = program.planned().next().unwrap();
            assert!(
                matches!(
                    eqn.op,
                    OpKind::ContractionEpilogue {
                        head: ContractionHead::Packed { .. },
                        ..
                    }
                ),
                "{label}"
            );
            let KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(
                PackedContraction {
                    rows: PackedRows::Blocks,
                    schedule,
                    epilogue: Some(epilogue),
                    ..
                },
            ))) = program.kernel_choice(eqn)
            else {
                panic!("{label}: not a packed contraction with an epilogue");
            };
            assert_eq!(epilogue.steps.len(), 2, "{label}");
            assert_eq!(epilogue.leaves.len(), 2, "{label}");
            match (m, schedule) {
                (1, Schedule::Gemv { .. }) => {}
                (m, Schedule::RegisterTiled { .. } | Schedule::FragmentTiled { .. }) if m > 1 => {}
                (_, schedule) => panic!("{label}: unexpected schedule {schedule:?}"),
            }
        }
    }
}

/// `(scores @ v) + bias` over the decode attention shapes: `scores: [B, H, 1, CAP]`, `v: [B, H, CAP, D]`, the matmul
/// that plans the LDS-parallel `scores @ V` reduction.
fn attention_scores_v_plus_bias() -> Graph {
    const B: usize = 1;
    const H: usize = 4;
    const CAP: usize = 64;
    const D: usize = 32;
    let b = Builder::for_tests();
    let scores = b
        .constant("scores", TensorType::f32(vec![B, H, 1, CAP]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![B, H, CAP, D]))
        .unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![D])).unwrap();
    let out = b
        .binary(BinOp::Add, b.matmul(scores, v).unwrap(), bias)
        .unwrap();
    b.finish(out).unwrap()
}

/// POOT-1626: a head whose planned request carries no epilogue keeps its chain apart. The decode attention
/// `scores @ V` plans the LDS-parallel reduction ([`ContractionSpec::AttnScoresVGemv`]), which has no per-output
/// store to apply a pointwise step to; the fuse rule asks the planned request, finds none and leaves the head and
/// the bias add as two dispatches rather than trading that schedule for the serial body that would carry the store.
///
/// Mutation: make `ContractionSpec::accepts_epilogue` answer `true` for `AttnScoresVGemv`; the head fuses, the
/// planner plans it on the serial body and this row goes red with `the head keeps its LDS reduction`.
#[test]
fn an_attention_scores_v_head_keeps_its_chain_apart() {
    for backend in backends() {
        let program = program(&attention_scores_v_plus_bias(), backend);
        let label = format!("{backend:?}: {:?}", op_names(&program));
        assert_eq!(program.planned().count(), 2, "{label}: head and add");
        assert!(
            program
                .planned()
                .all(|(eqn, _)| !matches!(eqn.op, OpKind::ContractionEpilogue { .. })),
            "{label}"
        );
        let (head, _) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::MatMul { .. }))
            .expect("the head is planned");
        assert!(
            matches!(
                program.kernel_choice(head),
                KernelChoice::Generated(KernelRequest::Contraction(
                    ContractionSpec::AttnScoresVGemv { .. }
                ))
            ),
            "{label}: the head keeps its LDS reduction"
        );
    }
}
