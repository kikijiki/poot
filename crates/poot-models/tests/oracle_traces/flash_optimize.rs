//! Masked decode decomposition-correctness checks (`Model::trace` graphs only, no poot-graph-plan pass
//! involved).
//!
//! Card 626: the `optimize`/dispatch-count/pass-fusion tests that used to live here moved to
//! `poot-graph-plan/tests/flash_optimize_pass_soundness.rs` with the passes themselves - poot-eval
//! must never depend on poot-graph-plan (its own architecture test). Card 557: tracers no longer
//! emit the flash ops, so the "flash trace equals decomposed trace" checks that were here are that
//! file's pass-soundness rows (the pass forms the op from the traced chain).

use poot_models::model::{LogitRows, Phase};
use poot_test_util::assert_close_rel;

use super::qwen2_pipeline::{eval_step, tiny_qwen2, trace_step};

/// G3d-3: the constant-shape masked decode (built once per capacity) does not depend on how many unwritten positions its
/// cache holds: the same weights and tokens give the same logits, step for step, over a cache sized to the sequence and
/// over one twice that size, because the mask hides every position the sequence has not reached.
#[test]
fn masked_decode_does_not_depend_on_capacity() {
    let model = tiny_qwen2(32, 16, 32, 16);
    let tokens: Vec<i32> = vec![5, 9, 2, 14, 7, 1];
    let (tight, wide) = (tokens.len(), 2 * tokens.len());

    let g_tight = trace_step(&*model, Phase::Decode, (1, 1, tight), None, LogitRows::Last);
    let g_wide = trace_step(&*model, Phase::Decode, (1, 1, wide), None, LogitRows::Last);
    let (mut tight_caches, mut wide_caches) = (None, None);
    for (pos, &token) in tokens.iter().enumerate() {
        let a = eval_step(
            &g_tight,
            &[token],
            &[pos as i32],
            None,
            tight_caches.as_deref(),
        );
        let b = eval_step(
            &g_wide,
            &[token],
            &[pos as i32],
            None,
            wide_caches.as_deref(),
        );
        assert_close_rel(a.logits.as_f32().unwrap(), b.logits.as_f32().unwrap(), 5e-3);
        (tight_caches, wide_caches) = (Some(a.state), Some(b.state));
    }
}
