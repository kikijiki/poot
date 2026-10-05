//! Trace a tiny Qwen2 decode step into poot's backend-neutral graph IR, then run automatic fusion and show
//! the dispatch (eqn) collapse. A model's forward is traced to a fine-grained primitive graph (attention,
//! RMSNorm, RoPE, SwiGLU all decomposed), then fused by codegen into far fewer synthesized kernels, with no
//! hand-written fused kernels.
//!
//! Run: `cargo run -p poot-llm --example trace_and_fuse` (CPU only, no GPU or weights needed).

use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::OpKind;
use poot_graph_plan as transform;
use poot_models::model::{LogitRows, Phase};

fn main() {
    // Tiny dims, same structure as Qwen2.5.
    let model = Dense::new(Family::Qwen2)
        .vocab(256)
        .dims(64, 128, 4)
        .heads(8, 4)
        .head_dim(8)
        .max_positions(64)
        .model();
    let cap = 16;

    // trace one constant-shape masked decode step into the primitive graph (G2/G3d shape).
    let g = plain(
        model
            .model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .expect("trace the decode step"),
    );
    println!(
        "traced decode (cap={cap}): {} primitive eqns, {} inputs, {} carried KV-state pairs",
        g.eqns.len(),
        g.inputs.len(),
        g.state.len()
    );

    // the standard pipeline: cse to drop redundancy, then G5 automatic fusion.
    let fused = transform::fuse(&transform::cse(&g));
    let mut pointwise = 0usize;
    let mut row = 0usize;
    for e in &fused.eqns {
        match &e.op {
            OpKind::Fused(_) => pointwise += 1,
            OpKind::FusedRow(_) => row += 1,
            _ => {}
        }
    }
    println!(
        "after cse + fuse:  {} eqns ({pointwise} Fused pointwise + {row} FusedRow reduction-rooted regions)",
        fused.eqns.len()
    );
    let before = g.eqns.len();
    let after = fused.eqns.len();
    println!(
        "fusion collapsed {before} -> {after} eqns ({:.0}% fewer dispatches), each region one synthesized kernel",
        100.0 * (before - after) as f64 / before as f64
    );
}
