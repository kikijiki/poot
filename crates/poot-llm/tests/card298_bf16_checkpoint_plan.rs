//! Card 298 regression (CPU only, no GPU and no checkpoint needed): every equation of a
//! natively-bf16 checkpoint's prefill and decode graph must plan on wgpu and on ROCm with every BF16 const
//! kept BF16 (Card 1011: no pass retypes a const to F32).
//!
//! A natively BF16 checkpoint declares every weight const BF16 while the activation stays F32. The new invariant is narrower
//! than a blanket retype:
//!
//! - A BF16 weight const whose only consumers are decode GEMVs that
//!   [`matmul_bf16_decode_gemv_eligible`] accepts stays BF16 after `widen_mismatched_matmul_dtypes`
//!   and is planned onto the packed-u32 bf16 GEMV body (residency; no f32 device shadow).
//! - Every other BF16 const stays BF16 too, so the raw graph's mixed-dtype matmuls remain rejected by
//!   name before the widening pass, and the pass reads each such weight through an explicit
//!   `Cast{BF16 -> F32}` equation.
//!
//! Synthetic configs reproduce the operand shapes without loading weights. Covers the tracer's
//! dtype choice, `widen_mismatched_matmul_dtypes`, and `plan_eqn`'s mixed-dtype rejection.

use poot_tensor::DType;
use std::collections::BTreeSet;

use poot_graph_plan::passes_without_target as optimize;

use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::{Eqn, Graph, OpKind, Operand, Storage, ValueId};
use poot_graph_plan::{
    PlanError, matmul_bf16_decode_gemv_eligible, numel, widen_mismatched_matmul_dtypes,
};
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

/// Shape/dtype projection onto the production predicate [`matmul_bf16_decode_gemv_eligible`]. This
/// only pulls the nine scalars out of the eqn; the eligibility rules themselves stay in production.
fn eligible_decode_gemv(
    g: &Graph,
    backend: Backend,
    eqn: &Eqn,
    caps: &poot_target::DeviceCaps,
) -> bool {
    if !matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias) || eqn.inputs.len() < 2 {
        return false;
    }
    let (Operand::Value(a), Operand::Value(b)) = (&eqn.inputs[0], &eqn.inputs[1]) else {
        return false;
    };
    let out_shape = &g.aval(eqn.out).shape;
    let r = out_shape.len();
    if r < 2 {
        return false;
    }
    let (m, n) = (out_shape[r - 2], out_shape[r - 1]);
    let k = g.aval(*a).shape.last().copied().unwrap_or(0);
    matmul_bf16_decode_gemv_eligible(
        backend,
        g.aval(*a).dtype,
        g.aval(*b).dtype,
        g.aval(eqn.out).dtype,
        m,
        n,
        k,
        g.aval(*b).shape.len(),
        numel(out_shape),
        caps,
    )
}

/// A non-GEMV BF16 consumer that production's raw-graph mixed-dtype guard must reject.
///
/// The guard lives only on [`OpKind::MatMul`]: `MatMulBias` has no mixed-dtype arm (prepare retypes
/// its weight before any entry point plans it), so a bias epilogue is not part of the rejected set.
fn non_gemv_bf16_matmul(
    g: &Graph,
    backend: Backend,
    eqn: &Eqn,
    caps: &poot_target::DeviceCaps,
) -> bool {
    if eqn.op != OpKind::MatMul {
        return false;
    }
    let consumes_bf16 = eqn
        .inputs
        .iter()
        .any(|operand| matches!(operand, Operand::Value(v) if g.aval(*v).dtype == DType::BF16));
    consumes_bf16 && !eligible_decode_gemv(g, backend, eqn, caps)
}

/// The BF16 consts of `g`.
fn bf16_consts(g: &Graph) -> BTreeSet<ValueId> {
    (0..g.values.len())
        .filter(|&id| {
            g.values[id].storage == Storage::Const && g.values[id].aval.dtype == DType::BF16
        })
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
fn card298_bf16_prefill_and_decode_graphs_plan_on_wgpu_and_rocm() {
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
        let is_decode = what.contains("decode");
        for backend in backends() {
            let caps = poot_test_util::device_caps::default_caps_for(backend);
            // Narrowed: the raw graph is rejected for exactly the non-GEMV BF16
            // `MatMul`s (mixed F32 x BF16 -> F32 with no eligible arm). Eligible decode GEMVs plan
            // raw; every other rejection must still be a BF16 consumer, and no guard-covered BF16
            // consumer may be missing from the rejected set.
            let expected_rejected: BTreeSet<usize> = g
                .eqns
                .iter()
                .enumerate()
                .filter(|(_, eqn)| non_gemv_bf16_matmul(g, backend, eqn, &caps))
                .map(|(i, _)| i)
                .collect();
            let rejected: BTreeSet<usize> = plan_failures(g, backend)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                rejected,
                expected_rejected,
                "{what} on {backend:?}: raw rejections must be exactly the non-GEMV BF16 MatMul \
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

            let prepared = widen_mismatched_matmul_dtypes(g, backend, &caps);
            let failures = plan_failures(&prepared, backend);
            assert!(
                failures.is_empty(),
                "{what} on {backend:?}: {} eqn(s) still unplannable after the widening pass; first is {}\n  {}",
                failures.len(),
                describe(&prepared, failures[0].0),
                failures[0].1
            );

            // Card 1011: no BF16 const is retyped to F32 -- the BF16 const set is untouched. The
            // decode-GEMV weights are read as packed lanes directly; every other BF16 weight use is
            // widened by one explicit `Cast{BF16 -> F32}` equation the planner lowers.
            assert_eq!(
                bf16_consts(&prepared),
                bf16_consts(g),
                "{what} on {backend:?}: a BF16 const was retyped"
            );
            assert!(
                !bf16_consts(g).is_empty(),
                "{what} on {backend:?}: the checkpoint graph must hold BF16 consts"
            );
            // The family already traces some explicit casts (a norm scale is stored BF16 and widened
            // in the graph); the pass's own contribution is what it adds to those.
            let widening_casts = bf16_widening_casts(&prepared) - bf16_widening_casts(g);
            if is_decode {
                assert_eq!(
                    widening_casts, 0,
                    "{what} on {backend:?}: decode GEMV weights are read packed, never cast"
                );
                assert_eq!(
                    prepared.eqns.len(),
                    g.eqns.len(),
                    "{what} on {backend:?}: the widening pass must not add eqns for a decode graph"
                );
            } else {
                assert!(
                    widening_casts > 0 && prepared.eqns.len() == g.eqns.len() + widening_casts,
                    "{what} on {backend:?}: prefill (M>1) widens each BF16 weight with one Cast \
                     ({widening_casts} casts, {} -> {} eqns)",
                    g.eqns.len(),
                    prepared.eqns.len()
                );
            }
        }
    }
}
