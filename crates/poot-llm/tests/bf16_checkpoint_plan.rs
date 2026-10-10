//! Card 298 regression (CPU only, no GPU and no checkpoint needed): every equation of a
//! natively-bf16 checkpoint's prefill and decode graph must plan on wgpu and on ROCm with every BF16 const
//! kept BF16 (Card 1011: no pass retypes a const to F32).
//!
//! A natively BF16 checkpoint declares every weight const BF16 while the activation stays F32. The new invariant is narrower
//! than a blanket retype:
//!
//! - Every BF16 weight const stays BF16, so the raw graph's mixed-dtype matmuls are rejected by name
//!   before `prepare_target_graph`, and the fold then reads each projection's weight as stored through a
//!   `DenseContraction` (residency; no f32 device shadow). A BF16 `[K, N]` decode projection has no lowering
//!   (`matmul_bf16_column_gemv`) and is refused rather than widened.
//!
//! Synthetic configs reproduce the operand shapes without loading weights. Covers the tracer's
//! dtype choice, `prepare_target_graph`, and `plan_eqn`'s mixed-dtype rejection.

use poot_tensor::DType;
use std::collections::BTreeSet;

use poot_graph_plan::passes_without_target as optimize;

use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::{Eqn, Graph, OpKind, Operand, Storage, ValueId};
use poot_graph_plan::{PlanError, prepare_target_graph};
use poot_models::model::{LogitRows, Phase};
use poot_target::Backend;
use poot_test_util::graph_fixtures::plan_eqn;

/// olmo2-1b proportions with vocab/layers shrunk; hidden, heads, head_dim and inter are real. The
/// checkpoint is natively BF16 (zeros: only the traced graph is read).
fn olmo2_like(family: Family) -> Dense {
    Dense::new(family)
        .vocab(512)
        .dims(2048, 8192, 2)
        .heads(16, 16)
        .head_dim(128)
        .max_positions(4096)
}

/// `dense`'s BF16 checkpoint traced at `phase` (`tokens` new tokens over `cap` positions).
fn traced(dense: &Dense, phase: Phase, tokens: usize, cap: usize) -> Graph {
    let m = dense.zeroed_model(DType::BF16);
    plain(
        m.model
            .trace(phase, step(1, tokens, cap, LogitRows::Last))
            .unwrap(),
    )
}

fn backends() -> Vec<Backend> {
    vec![
        Backend::SpirvVulkan,
        Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
    ]
}

/// Every eqn's plan result, as `(index, error)` for the ones that fail.
fn plan_failures(g: &Graph, backend: Backend) -> Vec<(usize, PlanError)> {
    g.eqns
        .iter()
        .enumerate()
        .filter_map(|(i, eqn)| {
            plan_eqn(
                g,
                eqn,
                backend,
                &poot_test_util::device_caps::default_caps_for(backend),
            )
            .err()
            .map(|e| (i, e))
        })
        .collect()
}

/// A raw-graph BF16 consumer that production's mixed-dtype guard must reject: every `MatMul` that reads a BF16
/// operand. The raw graph holds no `DenseContraction` (the fold has not run), so a decode projection over a
/// transposed checkpoint weight is a mixed matmul too, and a BF16 `[K, N]` decode projection
/// ([`matmul_bf16_column_gemv`]) is refused rather than widened.
///
/// The guard lives only on [`OpKind::MatMul`]: a bias epilogue is never formed over a mixed-dtype head, so
/// it is not part of the rejected set.
fn bf16_matmul(g: &Graph, eqn: &Eqn) -> bool {
    eqn.op == OpKind::MatMul
        && eqn
            .inputs
            .iter()
            .any(|operand| matches!(operand, Operand::Value(v) if g.aval(*v).dtype == DType::BF16))
}

/// The BF16 consts of `g`.
fn bf16_consts(g: &Graph) -> BTreeSet<ValueId> {
    g.values
        .iter_enumerated()
        .filter(|(_, meta)| meta.storage == Storage::Const && meta.aval.dtype == DType::BF16)
        .map(|(id, _)| id)
        .collect()
}

/// The `Cast{BF16 -> F32}` equations of `g`.
fn bf16_widening_casts(g: &Graph) -> usize {
    g.eqns
        .iter()
        .filter(|e| {
            matches!(e.op, OpKind::Cast { to: DType::F32 })
                && e.inputs.first().is_some_and(|operand| {
                    let Operand::Value(v) = operand else {
                        return false;
                    };
                    g.aval(*v).dtype == DType::BF16
                })
        })
        .count()
}

/// Op, output shape, and operand shapes/dtypes of an eqn.
fn describe(g: &Graph, index: usize) -> String {
    let eqn = &g.eqns[index];
    let ins: Vec<String> = eqn
        .inputs
        .iter()
        .filter_map(|o| match o {
            Operand::Value(v) => Some(*v),
            _ => None,
        })
        .map(|v| format!("{:?}/{:?}", g.aval(v).shape, g.aval(v).dtype))
        .collect();
    format!(
        "eqn#{index} {:?} out={:?}/{:?} inputs=[{}]",
        eqn.op,
        g.aval(eqn.out).shape,
        g.aval(eqn.out).dtype,
        ins.join(", ")
    )
}

#[test]
fn bf16_prefill_and_decode_graphs_plan_on_wgpu_and_rocm() {
    let (n, cap) = (32usize, 64usize);
    let olmo2 = olmo2_like(Family::Olmo2);
    let graphs = [
        (
            "olmo2 prefill",
            optimize(&traced(&olmo2, Phase::Prefill, n, cap)),
        ),
        (
            "olmo2 decode",
            optimize(&traced(&olmo2, Phase::Decode, 1, cap)),
        ),
        (
            "qwen2 prefill",
            optimize(&traced(&olmo2_like(Family::Qwen2), Phase::Prefill, n, cap)),
        ),
    ];

    for (what, g) in &graphs {
        for backend in backends() {
            // The raw graph is rejected for exactly the BF16 `MatMul`s (mixed F32 x BF16 -> F32 with no
            // lowering): the fold into a `DenseContraction` has not run, and a BF16 `[K, N]` decode
            // projection has none. Every rejection must be a BF16 consumer, and no BF16 `MatMul` may be
            // missing from the rejected set.
            let expected_rejected: BTreeSet<usize> = g
                .eqns
                .iter()
                .enumerate()
                .filter(|(_, eqn)| bf16_matmul(g, eqn))
                .map(|(i, _)| i)
                .collect();
            let rejected: BTreeSet<usize> = plan_failures(g, backend)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                rejected,
                expected_rejected,
                "{what} on {backend:?}: raw rejections must be exactly the BF16 MatMul \
                 consumers.\n  unexpected: [{}]\n  missing: [{}]",
                rejected
                    .difference(&expected_rejected)
                    .map(|i| describe(g, *i))
                    .collect::<Vec<_>>()
                    .join("; "),
                expected_rejected
                    .difference(&rejected)
                    .map(|i| format!("{} (should have been rejected)", describe(g, *i)))
                    .collect::<Vec<_>>()
                    .join("; "),
            );

            let prepared = prepare_target_graph(g, backend);
            let failures = plan_failures(&prepared, backend);
            assert!(
                failures.is_empty(),
                "{what} on {backend:?}: {} eqn(s) still unplannable after the widening pass; first is {}\n  {}",
                failures.len(),
                describe(&prepared, failures[0].0),
                failures[0].1
            );

            // Card 1011: no BF16 const is retyped to F32 -- the BF16 const set is untouched. Every BF16
            // projection is folded into a `DenseContraction` that reads the checkpoint weight as stored
            // (decode as the generated `[N, K]` Gemv, prefill as the generated tiled body), so no BF16
            // `MatMul` is left and the pass adds no widening Cast for a weight.
            assert_eq!(
                bf16_consts(&prepared),
                bf16_consts(g),
                "{what} on {backend:?}: a BF16 const was retyped"
            );
            assert!(
                !bf16_consts(g).is_empty(),
                "{what} on {backend:?}: the checkpoint graph must hold BF16 consts"
            );
            assert!(
                !prepared.eqns.iter().any(|e| bf16_matmul(&prepared, e)),
                "{what} on {backend:?}: a BF16 MatMul survived the dense fold"
            );
            assert!(
                prepared
                    .eqns
                    .iter()
                    .any(|e| matches!(e.op, OpKind::DenseContraction { .. })),
                "{what} on {backend:?}: the BF16 projections must fold into DenseContractions"
            );
            assert!(
                bf16_widening_casts(&prepared) <= bf16_widening_casts(g),
                "{what} on {backend:?}: the pass adds no widening Cast for a weight"
            );
        }
    }
}
