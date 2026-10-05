//! DeepSeek-V4 CPU-oracle tracers (spec 281, arXiv 2606.19348). F32 only, no GPU path. See
//! `specs/281-deepseek-v4-csa-hca-mhc/spec.md` for the derivation, citations, and scope.
//!
//! # Coverage
//!
//! V4 has three per-layer attention types (`sliding_attention`, `compressed_sparse_attention` (CSA),
//! `heavily_compressed_attention` (HCA)) and a widened `[.., hc_mult, hidden]` residual stream mixed by
//! Manifold-Constrained Hyper-Connections (mHC) at every layer.
//!
//! - HCA: non-overlapping softmax-gated pooling plus dense (no top-k) attention over every causally valid
//!   compressed entry, concatenated onto the local sliding-window K==V branch
//!   (`trace_deepseek4_hca_decode`/`trace_deepseek4_hca_prefill`).
//! - CSA: overlapping-window pooling (`csa_overlap_pool`: an entry pools window `w-1`'s "Ca" half plus
//!   window `w`'s "Cb" half) and a Lightning Indexer (`csa_indexer_scores`/`csa_topk_mask`) that selects
//!   the top-`k` entries each query attends to
//!   (`trace_deepseek4_csa_decode`/`trace_deepseek4_csa_prefill`).
//! - FP4/FP8, DSpark and YaRN are out of scope (see the spec).
//!
//! # The attention core is not MLA
//!
//! Checked against the `huggingface/transformers` `deepseek_v4` module: `DeepseekV4Attention` has no
//! `kv_a_proj_with_mqa`/`kv_a_layernorm`/`kv_b_proj` latent compression. Instead it uses:
//!
//! - a low-rank query (`q_a_proj`, weighted RMSNorm, `q_b_proj`, then an unweighted per-head RMSNorm
//!   ([`poot_graph_ir::ops::rmsnorm_no_weight`]) before RoPE);
//! - a single shared KV head read as both K and V;
//! - per-query-head learned attention sinks (same math as `crate::gpt_oss`, shared via
//!   [`poot_graph_ir::ops::attention_masked_with_sink`]/[`poot_graph_ir::ops::attention_prefill_with_sink`]);
//! - partial interleaved RoPE on the trailing `qk_rope_head_dim` slice of each head, with an inverse rotation
//!   on the trailing slice of the attention output (undoing the rotation V picked up from sharing K's
//!   tensor);
//! - a grouped low-rank output projection (`o_groups` block-diagonal matmuls, then one linear).
//!
//! # mHC (Manifold-Constrained Hyper-Connections)
//!
//! The residual stream is `[1, S, hc_mult, hidden]` for the whole model. Each decoder layer has two
//! `hyper_connection`/`hyper_connection_combine` pairs (attention site, MLP site), and one final
//! `hyper_head` collapse precedes the closing RMSNorm. `comb` (the cross-stream residual mixer) is
//! projected onto the doubly stochastic matrices by Sinkhorn-Knopp: one column-normalize, then
//! `hc_sinkhorn_iters - 1` rounds of (row-normalize, column-normalize), matching the reference code. The
//! column-normalize reduces the penultimate `hc` axis directly (see `sum_penultimate_keepdim`; card 523b:
//! `compile` legalizes a non-last-axis reduce for every target).
//!
//! # Real config values (`deepseek-ai/DeepSeek-V4-Flash-0731/config.json`)
//!
//! `hidden_size: 4096`, `num_hidden_layers: 43`, `num_attention_heads: 64`, `head_dim: 512`,
//! `qk_rope_head_dim: 64`, `q_lora_rank: 1024`, `o_groups: 8`, `o_lora_rank: 1024`, `sliding_window: 128`,
//! `hc_mult: 4`, `hc_sinkhorn_iters: 20`, `hc_eps: 1e-6`, `rms_norm_eps: 1e-6`, `swiglu_limit: 10.0`. Test
//! fixtures use tiny analogs of these dims.
//!
//! # Reused pieces
//!
//! - `crate::deepseek2::{rope_interleave_apply, rope_interleaved_decode, rope_interleaved_prefill,
//!   deepseek2_rope_tables_interleaved}`: interleaved-pair partial RoPE, also used for the inverse rotation.
//! - `poot_graph_ir::ops::{attention_masked_with_sink, attention_prefill_with_sink, sigmoid, softmax, rmsnorm,
//!   rmsnorm_no_weight, linear, relu, repeat_kv}`.
//! - `crate::gemma4::gemma4_local_window_floor` for the decode-side sliding-window mask.
//! - `crate::deepseek3::{pairwise_rank, keep_top_k_mask}` for the indexer's top-`k` selection.
//!   `csa_topk_mask` has the same additive-mask recombination as `crate::deepseek32::dsa_combined_mask`, but
//!   V4's indexer scorer applies two extra scale multiplies (see `csa_indexer_scores`).
//! - `hca_window_rope`/`hca_decode_validity_mask` are shared by CSA. `softmax_gate_pool_core` is the
//!   softmax-then-pool math shared by `hca_softmax_gate_pool` and `csa_overlap_pool`.

#[cfg(test)]
use crate::deepseek2::{rope_interleave_apply, rope_interleaved_decode, rope_interleaved_prefill};
#[cfg(test)]
use poot_tensor::DType;

#[cfg(test)]
use crate::deepseek3::{keep_top_k_mask, pairwise_rank};

#[cfg(test)]
use crate::gemma4::gemma4_local_window_floor;

#[cfg(test)]
use poot_graph_ir::ops::{
    PackedLinearGraphRow, attention_masked_with_sink, attention_prefill_with_sink,
    canonical_router_scores, packed_block_diagonal_linear, packed_grouped_linear,
    packed_indexed_linear, packed_linear, relu, repeat_kv, rmsnorm, softplus,
    stable_descending_rank,
};

#[cfg(test)]
use poot_graph_ir::{
    ExecutionValidationFailure, Graph, Slot, StateRole, TensorType, UnOp, ValidationId,
    ValidationOutputs,
};

#[cfg(test)]
use poot_graph_ir::ops::{linear, rmsnorm_no_weight, sigmoid, softmax};
#[cfg(test)]
use poot_graph_ir::{
    BinOp, Builder, BuilderAppendError, GraphValidationError, IndexGuardError, RedOp, Scalar,
    Traced,
};
#[cfg(test)]
use sources::V4SourcePlanError;

/// Owner of every V4 graph source name and its role.
pub mod sources;

#[cfg(test)]
use sources::{
    DeepseekV4SourcePlan, V4_COMPRESS_ROPE_COS, V4_COMPRESS_ROPE_SIN, V4_CSA_BLOCK_BIAS,
    V4_CSA_WINDOW_POSITIONS, V4_HCA_BLOCK_BIAS, V4_HCA_WINDOW_POSITIONS, V4_ROPE_COS, V4_ROPE_SIN,
    V4CompressorDense, V4DenseSourceSpec, V4ExpertProjection, V4IndexerDense, V4LayerDense,
    V4PackedRole, V4TopDense,
};

mod config;
mod csa;
mod hca;
mod mhc;
mod moe;
mod sliding;
mod stack;

#[cfg(test)]
pub(crate) use config::*;
#[cfg(test)]
pub(crate) use csa::*;
#[cfg(test)]
pub(crate) use hca::*;
#[cfg(test)]
pub(crate) use mhc::*;
#[cfg(test)]
pub(crate) use moe::*;
#[cfg(test)]
pub(crate) use sliding::*;
#[cfg(test)]
pub(crate) use stack::*;

#[cfg(test)]
mod tests;
