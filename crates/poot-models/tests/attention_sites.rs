//! Every attention root of every registered family is fused or declined for a named reason.
//!
//! The corpus is the plan-summary one (every registered family's fixture, at decode and prefill shapes, plus
//! the dense parity checkpoints). A root is a softmax-weighted matmul: a contraction whose left operand is a
//! `Div` of an `Exp`, however the family spelled the rest of the chain. After `compile`, each root is either
//! an [`OpKind::Attention`] (or the block-selected composite a pass forms from one) or a recorded attention
//! [`FusionDecline`](poot_graph_plan::FusionDecline), so a family whose attention stops fusing is a failing
//! row naming the check that failed, not a silent cliff of one dispatch chain per layer.
//!
//! The families that decline today are listed with the reason; every other family fuses all its roots on
//! every reference target. A sink family (gpt-oss, DeepSeek-V4) is in the second group: its roots are the
//! sink variant of the fused op.
use std::collections::{HashMap, HashSet};

use poot_graph_ir::{BinOp, Graph, MaskOperand, OpKind, Operand, UnOp};
use poot_graph_plan::{AttentionDecline, FusionDeclineReason, compile};
use poot_models::registry::Registry;

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/targets.rs"]
mod targets;

use corpus::corpus;

/// The contractions of `g` whose left operand is `Div(Exp(..), ..)`.
fn roots(g: &Graph) -> usize {
    let producers: HashMap<_, _> = g.eqns().iter().map(|e| (e.out, e)).collect();
    let producer = |operand: Option<&Operand>| match operand {
        Some(Operand::Value(value)) => producers.get(value).copied(),
        _ => None,
    };
    g.eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::MatMul { .. }))
        .filter_map(|e| producer(e.inputs.first()))
        .filter(|div| matches!(div.op, OpKind::Binary(BinOp::Div)))
        .filter_map(|div| producer(div.inputs.first()))
        .filter(|numerator| matches!(numerator.op, OpKind::Unary(UnOp::Exp)))
        .count()
}

/// The fused attention equations of a compiled graph.
fn fused(g: &Graph) -> usize {
    g.eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Attention(_) | OpKind::SelectedAttention(_)))
        .count()
}

/// The corpus rows whose attention is declined today, with the first failing check of each root. These are
/// recorded gaps of the encoders' attention, not decisions: a row not listed fuses every root.
const DECLINED: &[(&str, &[AttentionDecline])] = &[
    (
        "bert.encode",
        &[
            AttentionDecline::TypeMismatch,
            AttentionDecline::TypeMismatch,
        ],
    ),
    (
        "bert/cross_encoder.encode",
        &[
            AttentionDecline::TypeMismatch,
            AttentionDecline::TypeMismatch,
        ],
    ),
    (
        "idefics3/vision.encode",
        &[AttentionDecline::ScoreAdd, AttentionDecline::ScoreAdd],
    ),
];

#[test]
fn every_attention_root_is_fused_or_declined_for_a_named_reason() {
    let registry = Registry::builtin().unwrap();
    let declined: HashMap<&str, &[AttentionDecline]> = DECLINED.iter().copied().collect();
    let mut seen_declined = HashSet::new();
    let mut with_roots = 0;
    for row in corpus(&registry) {
        for reference in targets::reference_targets() {
            let traced = row.graph();
            let roots = roots(&traced);
            let Ok(program) = compile(&traced, &reference.target, &targets::options()) else {
                continue;
            };
            let reasons: Vec<AttentionDecline> = program
                .declines()
                .iter()
                .filter_map(|decline| match decline.reason {
                    FusionDeclineReason::Attention(reason) => Some(reason),
                    FusionDeclineReason::Rope(_) => None,
                })
                .collect();
            let fused = fused(program.graph());
            let at = format!("{} on {}", row.name, reference.id);
            assert_eq!(
                roots,
                fused + reasons.len(),
                "{at}: {roots} attention roots, {fused} fused, declined for {reasons:?}"
            );
            let expected = declined.get(row.name.as_str()).copied().unwrap_or(&[]);
            assert_eq!(reasons, expected, "{at}: the declined roots' reasons");
            if reasons.is_empty() {
                assert_eq!(fused, roots, "{at}: every root of a fusing family fuses");
            } else {
                seen_declined.insert(row.name.clone());
            }
            with_roots += usize::from(roots > 0);
        }
    }
    assert!(with_roots > 0, "the corpus has attention roots");
    for (name, _) in DECLINED {
        assert!(
            seen_declined.contains(*name),
            "{name} is listed as declining but fuses everything: delete its row"
        );
    }
}

/// The fused attentions of `g` whose mask is a materialized tensor rather than the positions of a causal mask.
fn additive(g: &Graph) -> usize {
    g.eqns()
        .iter()
        .filter(|e| matches!(&e.op, OpKind::Attention(a) if a.mask == MaskOperand::Additive))
        .count()
}

/// The families that keep a mask tensor for some fused attention, with how many per graph (the same at
/// every shape of the family): a mask that is no `causal_mask_from_pos` alone. ALiBi adds a per-head bias to
/// it, a sparse-attention family's indexer selects keys on top of it, and the vision encoder's window mask is
/// block-diagonal. Every other fused attention states its mask by positions, so the kernel visits only the
/// keys a row can see.
const ADDITIVE: &[(&str, usize)] = &[
    ("bloom", 2),
    ("mpt", 2),
    ("deepseek32", 2),
    ("deepseek4", 3),
    ("deepseek4/finetune", 2),
    ("glm53_flash", 2),
    ("glm53_flash/finetune", 1),
    ("qwen2_5_vl/vision", 4),
];

/// Every causal attention of every registered family is fused with a structural mask: the roots that still read
/// a mask tensor are exactly the listed ones, so a tracer or a matcher change that loses the structure is a
/// failing row naming the family, not a mask tensor quietly read per key.
#[test]
fn every_causal_attention_states_its_mask_by_positions() {
    let registry = Registry::builtin().unwrap();
    let expected: HashMap<&str, usize> = ADDITIVE.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut structural = 0;
    let mut wrong = Vec::new();
    for row in corpus(&registry) {
        for reference in targets::reference_targets() {
            let Ok(program) = compile(&row.graph(), &reference.target, &targets::options()) else {
                continue;
            };
            let graph = program.graph();
            let additive = additive(graph);
            let family = row
                .name
                .rsplit_once('.')
                .map_or(row.name.as_str(), |(f, _)| f);
            let want = expected.get(family).copied().unwrap_or(0);
            if additive != want {
                wrong.push(format!(
                    "{} on {}: {additive} fused attentions read a mask tensor, {want} listed",
                    row.name, reference.id
                ));
            }
            if additive > 0 {
                seen.insert(family.to_string());
            }
            structural += fused(graph) - additive;
        }
    }
    assert_eq!(wrong, Vec::<String>::new());
    assert!(
        structural > 0,
        "the corpus has structurally masked attentions"
    );
    for (name, _) in ADDITIVE {
        assert!(
            seen.contains(*name),
            "{name} is listed as reading a mask tensor but states every mask: delete its row"
        );
    }
}
