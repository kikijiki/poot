//! Spec 023 sub-step 1: the CPU batched-prefill correctness oracle.
//!
//! Builds the per-layer K/V cache two ways on a TINY synthetic config and asserts the two agree:
//!   (a) batched prefill (one multi-token forward, fills slots [0,N)),
//!   (b) token-by-token decode replayed for pos 0..N (the path the PTX capture replays).
//! The cache that (a) leaves must equal what (b) leaves, so a decode continuing from either is unchanged.
//! Deterministic synthetic weights (no real model); CPU eager executor only.

mod common;

use poot_executor_parity::dense::{Dense, Family};

#[test]
fn batched_prefill_fills_cache_identically_to_token_by_token() {
    // A tiny config exercising GQA (n_heads != n_kv_heads) and the qwen2 bias path.
    let m = Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(64)
        .f32_model();
    common::assert_prefill_fills_cache_like_token_by_token(&m, 2, 1e-5);
}
