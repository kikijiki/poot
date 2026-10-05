//! R486-013 / SC-006: a GQA sliding-window prefill fixture (`n_heads=4`, `n_kv_heads=2`) must fuse to a
//! `FlashAttentionPrefill` that carries the grouped head count `n_rep = 2`. The branch value is printed.
//! Setting `n_kv_heads` back to `n_heads` makes `n_rep = 1` and the grouped-path assertion fails.
//!
//! Card 626: moved here from `poot-eval/src/tests/swa_gemma3.rs` with the pass itself - poot-eval
//! must never depend on poot-graph-plan (its own architecture test); this crate already dev-depends
//! on poot-eval, so an integration test here can drive both (though this particular test needs no
//! poot-eval API at all).

use poot_executor_parity::dense::{Dense, Family, step};
use poot_graph_ir::op::OpKind;
use poot_graph_plan::{cse, dce, flash_attention_capped};
use poot_models::model::{LogitRows, Phase};

#[test]
fn swa_gqa_prefill_reaches_the_grouped_flash_path() {
    let (heads, kv_heads) = (4, 2);
    let m = Dense::new(Family::Gemma3)
        .vocab(16)
        .dims(8, 16, 2)
        .heads(heads, kv_heads)
        .head_dim(4)
        .max_positions(16)
        .with("sliding_window", 3)
        .f32_model();
    let n_rep = heads / kv_heads;
    let g = m
        .model
        .trace(Phase::Prefill, step(1, 6, 6, LogitRows::Last))
        .unwrap();
    let fused = dce(&flash_attention_capped(&cse(&g), None));
    let reps: Vec<usize> = fused
        .eqns
        .iter()
        .filter_map(|e| match e.op {
            OpKind::FlashAttentionPrefill { n_rep, .. } => Some(n_rep),
            _ => None,
        })
        .collect();
    eprintln!(
        "swa gqa prefill: n_heads={heads} n_kv_heads={kv_heads} n_rep={n_rep} fused_flash={reps:?}"
    );
    assert!(
        n_rep > 1,
        "the fixture must have head grouping (n_rep > 1), got n_rep={n_rep}"
    );
    assert!(
        !reps.is_empty(),
        "the GQA windowed prefill must fuse to FlashAttentionPrefill"
    );
    assert!(
        reps.iter().all(|&r| r == n_rep),
        "the fused flash op must carry the grouped head count n_rep={n_rep}, got {reps:?}"
    );
}
