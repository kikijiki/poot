//! Card 190: OLMo 2 CPU batched-prefill correctness oracle, the olmo2 analog of
//! `batched_prefill.rs`.
//!
//! Builds the per-layer K/V cache two ways on a tiny synthetic olmo2-shaped config and asserts
//! near-identity (both are f32 CPU eval of the same ops in a different composition order, so a
//! tight but non-zero tolerance is used, as in `batched_prefill.rs` and `olmo2_decode.rs`):
//!   (a) batched prefill (one multi-token forward, fills slots [0,N)),
//!   (b) token-by-token decode replayed for pos 0..N.
//!
//! (b) is a different graph (single-token masked decode, one cache write per token) from (a), so
//! this is an independently derived reference. A wrong post-norm placement or full-dimension
//! QK-norm in the olmo2 body (the two ways this arch differs from plain qwen2) would show as a
//! numeric divergence.

mod common;

use poot_executor_parity::dense::{Dense, Family};

/// Tiny GQA config (n_heads != n_kv_heads, unlike the MHA OLMo-2-1B), as in
/// `batched_prefill.rs`, to exercise the n_rep repeat path.
#[test]
fn olmo2_batched_prefill_fills_cache_identically_to_token_by_token() {
    let m = Dense::new(Family::Olmo2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(64)
        .with("rope_theta", 500_000.0)
        .f32_model();
    common::assert_prefill_fills_cache_like_token_by_token(&m, 2, 1e-4);
}
