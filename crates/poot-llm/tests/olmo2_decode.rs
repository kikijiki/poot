//! Checks the OLMo 2 fixed-KV decode step against the prefill step: decode-at-pos-0 logits
//! (empty-then-written KV cache, single token) must equal prefill-of-one-token logits, so the
//! post-norm + full-dim QK-norm block is composed identically. Tiny synthetic config, CPU eval:
//! no model, no GPU.

mod common;

use common::{step, token_pos, zero_state};
use poot_executor_parity::dense::{Dense, Family, step as step_shape};
use poot_models::model::{LogitRows, Phase};

#[test]
fn olmo2_decode_matches_prefill_first_token() {
    const VOCAB: usize = 48;
    let m = Dense::new(Family::Olmo2)
        .vocab(VOCAB)
        .dims(16, 24, 2)
        .heads(4, 4) // OLMo-2-1B is MHA (no GQA)
        .head_dim(4)
        .max_positions(32)
        .f32_model();
    let slots = token_pos(1, 1, &[7], &[0]);
    let cap = 1; // single position

    let pg = m
        .model
        .trace(Phase::Prefill, step_shape(1, 1, cap, LogitRows::Last))
        .unwrap();
    let dg = m
        .model
        .trace(Phase::Decode, step_shape(1, 1, cap, LogitRows::Last))
        .unwrap();
    let (logits_p, _) = step(&m, &pg, &slots, &zero_state(&pg));
    let (logits_d, _) = step(&m, &dg, &slots, &zero_state(&dg));

    assert_eq!(logits_p.shape(), vec![1, 1, VOCAB]);
    assert_eq!(logits_d.shape(), vec![1, 1, VOCAB]);
    // A panic here means OLMo 2 decode-at-pos-0 != prefill-n=1: the block composition differs.
    poot_test_util::assert_close(logits_p.as_f32().unwrap(), logits_d.as_f32().unwrap(), 1e-5);
}
