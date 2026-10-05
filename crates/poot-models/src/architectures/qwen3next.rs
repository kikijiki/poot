//! Qwen3-Next (qwen35moe / Qwen3.6-35B-A3B) MoE FFN + shared-expert block.
//!
//! Every transformer block (GDN linear-attention or gated full-attention) is followed by the same FFN:
//!
//! ```text
//! moe_out   = moe(x, router, w_in, w_out, n_experts, top_k, inter)   -- softmax-routed top-k of n_experts
//! shexp_act = silu(shexp_gate @ x) * (shexp_up @ x)                   -- shared expert SwiGLU
//! shexp     = shexp_down @ shexp_act
//! g         = sigmoid( shexp_gate_inp @ x )                            -- scalar gate per token
//! out       = moe_out + g * shexp                                      -- ADD, sigmoid-gated
//! ```
//!
//! References: section 5 (`build_layer_ffn`).
//!
//! ## Router normalization
//!
//! The reference (`build_moe_ffn`, norm_w=true, SOFTMAX gating):
//! 1. `probs = softmax(router @ x)` over all `n_experts`.
//! 2. `idx = top_k(probs, top_k)` (pick highest-prob experts, same order as highest logit).
//! 3. `w = probs[idx] / sum(probs[idx])` (renormalize the selected weights).
//!
//! poot's existing `moe()` op implements an equivalent computation via a masked softmax:
//! mask non-top-k logits to -1e30, then softmax over all; the non-top-k entries
//! exponentiate to ~0 so only the top-k contribute to the denominator - numerically
//! identical to step 3.
//!
//! ## Tensor shapes (poot convention, `[in, out]` pre-transposed)
//!
//! ```text
//! router_w      [H, E]        router: hidden -> n_experts logits
//! w_in          [E, H, 2*I]   per-expert fused gate||up (gate first, up second over the I dim)
//! w_out         [E, I, H]     per-expert down proj
//! shexp_gate_w  [H, I]        shared expert gate proj
//! shexp_up_w    [H, I]        shared expert up proj
//! shexp_down_w  [I, H]        shared expert down proj
//! shexp_gin_w   [H, 1]        shared expert scalar gate (hidden -> 1 scalar per token)
//! ```
//!
//! The real checkpoint stores gate/up separately as `[H, I, E]` tensors (ggml ne order); the loader must
//! transpose and fuse them into `[E, H, 2*I]` for `w_in` before calling this composition.

use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_tensor::DType;

use poot_graph_ir::ops::{
    attention, attention_masked, attention_prefill, causal_conv1d_decode, causal_conv1d_prefill,
    gated_delta_net_decode, gdn_prefill_chunked, linear, moe, repeat_kv, repeat_kv_tiled, rmsnorm,
    rope, rope_batched, rope_prefill, sigmoid, softplus, swiglu,
};

use poot_graph_ir::types::Scalar;

use poot_graph_ir::{Builder, Graph, Slot, StateRole, TensorType, Traced};

use crate::components::{gather_shared_pool, scatter_shared_pool};

mod attn;
mod config;
mod decode;
mod gdn;
mod moe;

pub use attn::*;
pub use config::*;
pub use decode::*;
pub use gdn::*;
pub use moe::*;

#[cfg(test)]
mod tests;
