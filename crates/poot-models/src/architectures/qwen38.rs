//! Qwen3.8-Flash-Next Qwen Sparse Attention (QSA) CPU-oracle tracer (spec 282). Scoped like
//! `crate::deepseek32`'s DSA first slice: QSA's block-sparse indexer gets a config struct, a graph-IR
//! construction, and a bit-exact independent-reference differential test. The rest of the model (Gated
//! DeltaNet, Hyper-Connections via [`qwen38_gated_residual`] et al., the routed-MoE FFN via
//! [`qwen38_moe_ffn`], N-gram/PLE embedding) is composed from existing pieces or implemented below; the
//! vision tower is deferred. Card 362 factors the MoE routing, shared-expert, and combine semantics so the
//! synthetic dense and official split packed sources share one definition. See
//! `specs/282-qwen38-qsa/spec.md` for the real-source citations (`config.json` and the `transformers`
//! `qwen4_exp` modeling source) and the up-to-date out-of-scope list.
//!
//! # QSA
//!
//! QSA exists only on the `full_attention` layers of the hybrid schedule (12 of 48 layers in the real
//! config, 3 Gated-DeltaNet layers between each). A QSA layer is [`crate::qwen3next`]'s gated GQA attention
//! (interleaved query|gate `q_proj`, per-head RMSNorm before RoPE, output multiplied by `sigmoid(gate)`
//! before `o_proj`; identical to [`crate::qwen3next::qwen3next_gated_attention_prefill`]) plus an additive
//! block-sparse mask from a small indexer network, ANDed into the causal mask before the softmax. This is
//! the same "indexer computes an extra mask term" shape as `crate::deepseek32`'s DSA tracer, checked
//! against the real `Qwen4ExpTextQSAIndexer.forward` (`transformers` `modeling_qwen4_exp.py`).
//!
//! The indexer (`Qwen4ExpTextQSAIndexer.forward`):
//!
//! 1. Projects the layer's normed input directly (DSA's `wq_b` consumes MLA's low-rank `qr` instead) to a
//!    per-head query `q [index_n_heads, index_head_dim]` and a single shared raw key
//!    `raw_k [index_head_dim]` (`index_kv_heads == 1` in the real config, MQA-shaped).
//! 2. `q` gets a weight-only RMSNorm then partial RoPE at its own position, using the same `(cos, sin)`
//!    table as the main attention (rotary width `partial_rotary_factor * head_dim`, not `index_head_dim`;
//!    trailing dims pass through unrotated, as the `poot_graph_ir::ops::rope_prefill` partial-rope
//!    composition DSA's indexer reuses).
//! 3. `raw_k` is mean-pooled raw (no norm, no rope) into fixed, absolute-position-aligned micro-blocks of
//!    `index_compress_ratio` consecutive tokens. Only then does the pooled result get its own weight-only
//!    RMSNorm and partial RoPE at the block's first-token position: pool, then norm, then rope.
//! 4. Score for query `t`, block `b`: `sum_h ReLU(q_{t,h} . k_block_b) / sqrt(index_head_dim)`, an
//!    unweighted sum over heads (DSA's is per-head-weighted) with a `1/sqrt(d)` scale (DSA's is unscaled).
//! 5. Per query row, keep the top `min(block_topk, num_eligible_blocks)` blocks
//!    (`block_topk = index_budget / index_compress_ratio`); a block is eligible once fully causally
//!    complete (`block_last_token <= query_pos`). All member tokens of a kept block become visible.
//! 6. The query's own not-yet-complete block is always force-visible in full, unscored (the real
//!    source's unconditional `tail` slice).
//!
//! # Static-shape derivation of the ragged tail (spec 282 FR-002/FR-003)
//!
//! The PyTorch reference computes step 6 with a per-query Python loop because "how many blocks are
//! complete" and "how many tail tokens remain" vary by query position. Both are a pure function of
//! `(query_pos, block_size)`, not of any tensor value, so both become host-precomputed additive
//! `[1,1,L,*]` constants, like the plain causal mask:
//!
//! - `block_eligible[i,b] = 0.0` if `b*C + C - 1 <= i` (block `b` complete as of query `i`) else a large
//!   negative constant. Ranking `index_scores + block_eligible` before top-k (as
//!   `crate::deepseek32::dsa_combined_mask` ranks the sum) keeps an incomplete or future block from
//!   winning top-k.
//! - `tail_mask[i,j] = 0.0` if `j` is in query `i`'s own incomplete block (`floor(j/C) == floor(i/C)`,
//!   `j <= i`, and `(i+1)` is not a multiple of `C`) else large-negative.
//!
//! `final_mask = causal_mask AND (block_topk_selection OR tail_mask)`. OR is `BinOp::Max`: every mask
//! value is exactly `0.0` or a large negative constant, so `Max` picks "allowed" if either side allows,
//! which plain `Add` would get backwards. AND is plain `BinOp::Add`: both sides come from the same
//! `{0.0, MASK_NEG}` pair, so summing two `MASK_NEG` terms is still deeply negative and never overflows
//! `f32` (as `crate::deepseek3::deepseek3_router_gate` and `crate::deepseek32::dsa_combined_mask` rely on).
//!
//! # Reused pieces
//!
//! `crate::qwen3next::qwen3next_gated_attention_prefill` (called unchanged with a different `mask`
//! argument), `crate::qwen3next::qwen3next_gdn_prefill_block` (the Gated-DeltaNet layers the structural
//! test interleaves QSA with; numerics covered by `crate::qwen3next`'s tests),
//! `poot_graph_ir::ops::{stable_descending_rank, top_k_keep_mask}` (the stable top-k at block granularity,
//! exact on tied scores, lower block index first), and `crate::deepseek2::deepseek2_dense_ffn`
//! (a plain-SwiGLU stand-in for the MoE FFN in the structural test). No new `poot_graph_ir::OpKind`
//! (`ArgTopK`/`Gather`/`Reduce`/`Rope`/`MatMul`/`Binary::Max` already exist).
//!
//! # Out of scope
//!
//! Vision tower and sequence lengths not a multiple of `index_compress_ratio`. See
//! `specs/282-qwen38-qsa/spec.md`'s Out of scope section for the current list.

use crate::qwen3next::{
    GdnHeadOrder, qwen3next_gated_attention_prefill, qwen3next_gdn_block,
    qwen3next_gdn_prefill_block,
};
use poot_tensor::DType;

use poot_graph_ir::ops::{
    PackedLinearGraphRow, attention_masked, causal_conv1d_decode_dilated,
    causal_conv1d_prefill_dilated, linear, moe_grouped_prep, packed_grouped_linear,
    packed_indexed_linear, relu, repeat_kv, rmsnorm, rope, rope_prefill, sigmoid, swiglu,
    u64_mul_i32_const, u64_rem_u32, u64_xor,
};

use poot_graph_ir::{
    BinOp, Builder, BuilderAppendError, BuilderAppendPlan, Graph, RedOp, Scalar, Slot, StateRole,
    Storage, TensorType, Traced, UnOp, packed_source_constants,
};

mod config;
mod exact_trace;
mod gated_residual;
mod moe_packed;
mod ple;
mod qsa;
mod trace;

pub use config::*;
pub use exact_trace::*;
pub(crate) use gated_residual::*;
pub use moe_packed::*;
pub use ple::*;
pub use qsa::*;
pub use trace::*;

#[cfg(test)]
mod tests;
