use super::*;

/// QSA's indexer shape, additional to the GQA attention dims [`Qwen4ExpConfig`] carries. Real
/// `Qwen/Qwen3.8-Flash-Next` `config.json` values (spec 282): `indexer_n_heads: 4`,
/// `indexer_kv_heads: 1`, `indexer_head_dim: 128`, `indexer_budget: 2048`, `indexer_compress_ratio: 4`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsaConfig {
    /// Number of indexer query heads (`Hi`), independent of the main attention's head count (the indexer is
    /// its own small scoring network, as DSA's Lightning Indexer is).
    pub index_n_heads: usize,
    /// Number of indexer KV heads. The real value is always `1` (MQA-shaped: one shared raw key per token).
    /// Only that value is implemented.
    pub index_kv_heads: usize,
    /// Per-indexer-head query/key width (`Di`).
    pub index_head_dim: usize,
    /// Total selected-token budget (`indexer_budget`); [`QsaConfig::block_topk`] derives the per-query
    /// block count from it and `index_compress_ratio`.
    pub index_budget: usize,
    /// Tokens per micro-block (`indexer_compress_ratio`). [`Qwen4ExpConfig`]'s traced `seq_len` must be an
    /// exact multiple of this (a ragged final block is deferred, spec 282).
    pub index_compress_ratio: usize,
}

impl QsaConfig {
    /// `block_topk = index_budget / index_compress_ratio` ("512 blocks or 2048 tokens" is one budget, not
    /// two knobs; spec 282).
    pub fn block_topk(&self) -> usize {
        self.index_budget / self.index_compress_ratio
    }
}

/// Plain GQA attention shape shared by every QSA layer here. Real config: `num_attention_heads: 24`,
/// `num_key_value_heads: 2`, `head_dim: 256`, `partial_rotary_factor: 0.25` (so `rotary_dim = 64`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpConfig {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// RoPE rotary width shared by the main attention and the indexer (both consume the same
    /// `position_embeddings` in the real source).
    pub rotary_dim: usize,
    pub eps: f32,
}

/// GDN (Gated DeltaNet) shape for Qwen3.8-Flash-Next's `linear_attention` layers: the mechanism
/// [`qwen3next_gdn_prefill_block`] implements, with its own head/dim counts. The HF checkpoint keeps V
/// heads grouped by K head, so the traces expand q/k in grouped order (`GDN_HEAD_ORDER`) rather than the
/// GGUF tiled order. Real `Qwen/Qwen3.8-Flash-Next` `text_config` values: `linear_num_key_heads: 16`,
/// `linear_num_value_heads: 48`, `linear_key_head_dim: 128`, `linear_value_head_dim: 128` (equal, so this
/// struct carries one `head_dim`), `linear_conv_kernel_dim: 4`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpGdnConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_dim: usize,
    pub conv_k: usize,
}

/// V-head order of the HF `Qwen/Qwen3.8-Flash-Next` safetensors. `Qwen4ExpTextGatedDeltaNet.forward`
/// (`transformers` 5.16.1 `modeling_qwen4_exp.py`) expands q/k with `repeat_interleave(n_rep, dim=2)`, and
/// the GDN tensors are used without reordering V heads (`GDN_HEAD_ORDER`).
pub(crate) const GDN_HEAD_ORDER: GdnHeadOrder = GdnHeadOrder::Grouped;

/// The whole-model bundle [`trace_qwen38_prefill`] needs on top of [`Qwen4ExpConfig`]/[`QsaConfig`]: the GDN
/// shape, the per-layer `full_attention`/`linear_attention` schedule, and the pieces to close an
/// embed -> layer stack -> `lm_head` trace. Not `Copy` (`layer_is_full` is a `Vec<bool>`).
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpModelConfig {
    pub cfg: Qwen4ExpConfig,
    pub qcfg: QsaConfig,
    pub gdn: Qwen4ExpGdnConfig,
    /// Real `layer_types`: `true` for `full_attention` (QSA), `false` for `linear_attention` (GDN). Length
    /// is `layers`.
    pub layer_is_full: Vec<bool>,
    pub vocab: usize,
    pub max_pos: usize,
    /// Real `shared_expert_intermediate_size` (640): the `mlp.shared_expert` SwiGLU branch's width, gated
    /// by `mlp.shared_expert_gate` and added to the routed-MoE sum (see [`qwen38_moe_ffn`]).
    pub ffn_inter: usize,
    /// Real `num_experts` (512): total routed experts in `mlp.experts`.
    pub moe_n_experts: usize,
    /// Real `num_experts_per_tok` (10): routed experts selected per token.
    pub moe_top_k: usize,
    /// Real `moe_intermediate_size` (640; equal to `ffn_inter` here but a distinct config field): each
    /// routed expert's SwiGLU intermediate width.
    pub moe_inter: usize,
    pub eps: f32,
    /// GDN chunked-recurrence tile width: a trace-time performance/precision parameter, not a checkpoint
    /// field (16 in the deleted Qwen3-Next runtime). Not a length constraint:
    /// `qwen3next_gdn_prefill_block` zero-pads `seq_len` to whole `chunk` tiles and truncates back, so
    /// [`trace_qwen38_prefill`]'s `seq_len` must be a multiple of `qcfg.index_compress_ratio` only (QSA's
    /// block-pooling reshape).
    pub chunk: usize,
    /// Real `hc_count` (`4`): the number of parallel streams Hyper-Connections widens the residual into
    /// (see [`qwen38_gated_residual`]/[`qwen38_widen_embedding`]).
    pub hc_count: usize,
    /// Real `hc_lowrank` (`320`): the bottleneck width of Hyper-Connections'
    /// `input_mix_weight_down`/`input_mix_weight_up` gate projection.
    pub hc_lowrank: usize,
    /// The N-gram Embedding / PLE layer ([`Qwen4ExpPleConfig`]), or `None` for no PLE layer. `Some(..)` puts
    /// one [`qwen38_ple_prefill`]/[`qwen38_ple_decode`] block at the front of each decoder layer named by
    /// [`Qwen4ExpPleConfig::ple_layer_ids`] (real config: the one-indexed id `2`, i.e. 0-based decoder
    /// layer 1).
    ///
    /// # Why optional
    ///
    /// The real PLE embedding table is `[320001536, 160]` BF16 (about 102 GB across 128
    /// `ngram_embedding.shard_*.weight` tensors), so real checkpoint loading for it is a separate project.
    /// The CPU oracle's `Gather` also reads its index through the f32 mirror (`poot_eval`'s `gather`:
    /// `index.data[..].round() as usize`), exact only below `2^24 = 16777216`, so real-scale n-gram ids
    /// cannot be looked up on this path. `None` leaves the default composition and the synthetic
    /// end-to-end test unchanged; `Some(..)` exercises the composition
    /// at tiny synthetic scale.
    pub ple: Option<Qwen4ExpPleConfig>,
}
