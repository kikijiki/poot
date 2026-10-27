//! A long prefill's arena fits one device buffer (POOT-1308).
//!
//! The decode-curve benchmark prefills an 8192-token context whole. Its traced attention is the
//! materialized `softmax(Q K^T) V` chain, so a plan that leaves it decomposed holds `[Hq, L, capacity]`
//! F32 scores, 3.8 GB for Qwen2.5-0.5B (14 heads, 8192 rows, 8320 positions): over every wgpu device's
//! buffer limit, and refused only when the executor allocated the arena. The flash fusion declined that
//! shape because its `Hq * L` rows passed wgpu's 65535 X grid cap, though the kernel folds its rows
//! onto a 2-D grid. This compiles the real-dimension models at that shape for every reference target
//! and holds the plan to the target's own limit.

use poot_graph_ir::OpKind;
use poot_graph_plan::{Target, compile};

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/real_dims.rs"]
mod real_dims;
#[path = "support/targets.rs"]
mod targets;

/// The decode-curve scenario's longest context and its output window (`benchmarks/manifest.toml`).
const LONGEST_ISL: usize = 8192;
const OSL: usize = 128;

#[test]
fn a_whole_longest_context_prefill_plans_arena_slots_within_the_buffer_limit() {
    let root = poot_test_util::repo_root();
    for model in real_dims::real_models(&root)
        .into_iter()
        .filter(|m| matches!(m.label.as_str(), "Qwen2.5-0.5B" | "Qwen3-0.6B"))
    {
        let graph = model.trace(corpus::Shape {
            name: "decode_curve_prefill",
            program: poot_models::io::ProgramId::STEP,
            tokens: LONGEST_ISL,
            capacity: LONGEST_ISL + OSL,
        });
        for reference in targets::reference_targets() {
            let Target::Gpu { caps, .. } = &reference.target else {
                unreachable!("every reference target is a GPU target");
            };
            let program = compile(&graph, &reference.target, &targets::options())
                .unwrap_or_else(|e| panic!("{} on {}: {e}", model.label, reference.id));
            let plan = program.buffer_plan();
            for (slot, storage, elems) in plan.slots() {
                let bytes = (elems * storage.element().byte_width()) as u64;
                assert!(
                    bytes <= caps.max_buffer_bytes,
                    "{} on {}: arena slot {slot:?} is {bytes} bytes, over the {}-byte buffer limit",
                    model.label,
                    reference.id,
                    caps.max_buffer_bytes
                );
            }
            assert!(
                program
                    .graph()
                    .eqns()
                    .iter()
                    .any(|e| matches!(e.op, OpKind::Attention(_))),
                "{} on {}: the prefill attention must fuse",
                model.label,
                reference.id
            );
        }
    }
}
