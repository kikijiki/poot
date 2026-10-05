//! The gemma3 analog of `olmo2_batched_prefill.rs` (spec 023 sub-step 1 / card 190).
//!
//! Builds the per-layer K/V cache two ways on a tiny synthetic gemma3-shaped config and asserts
//! near-identity (both are f32 CPU eval of the same primitive ops in a different composition order, so
//! a tight but non-zero tolerance is used, as in `olmo2_batched_prefill.rs`):
//!   (a) batched prefill (one multi-token forward, fills slots [0,N)),
//!   (b) token-by-token decode replayed for pos 0..N.
//!
//! This is the independently derived reference check card 190 requires, aimed at gemma3's risk
//! surface: the 4-norm post-norm "sandwich", per-head q_norm/k_norm at `[head_dim]`, the embedding
//! scale, a scalar attention scale in place of `1/sqrt(head_dim)`, and the per-layer local-vs-global
//! RoPE base and attention window. The tiny config has `sliding_window_pattern: 2` over 2 layers so
//! layer 0 is local (window 3 over a 5-token prompt, so the window cuts keys) and layer 1 global, with
//! distinct rope bases so a table mixup shows as a numeric divergence, not just a shape match.

mod common;

use poot_executor_parity::dense::{Dense, Family};

#[test]
fn gemma3_batched_prefill_fills_cache_identically_to_token_by_token() {
    let m = Dense::new(Family::Gemma3)
        .vocab(40)
        .dims(32, 48, 2)
        .heads(4, 2)
        .head_dim(8)
        .max_positions(64)
        .with("rope_theta", 1_000_000.0)
        .with("rope_local_base_freq", 10_000.0)
        .f32_model();
    common::assert_prefill_fills_cache_like_token_by_token(&m, 2, 1e-4);
}
