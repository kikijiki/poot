/// Dims for a V4 model. Real reference values (`DeepSeek-V4-Flash-0731/config.json`) are noted per field;
/// test fixtures use tiny analogs.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DeepseekV4Config {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    /// `num_attention_heads` (real: 64 Flash / 128 Pro).
    pub num_heads: usize,
    /// `head_dim` (real: 512) - shared by Q, the single K/V head, and the sink's own per-head axis.
    pub head_dim: usize,
    /// `qk_rope_head_dim` (real: 64) - width of the TRAILING rotated slice of each `head_dim`-wide head;
    /// must be even and `<= head_dim` (the interleaved-pair convention's own constraint).
    pub qk_rope_head_dim: usize,
    /// `q_lora_rank` (real: 1024) - the low-rank query bottleneck width.
    pub q_lora_rank: usize,
    /// `o_groups` (real: 8 Flash / 16 Pro) - number of independent blocks the grouped output projection
    /// splits `num_heads*head_dim` into. Must divide `num_heads*head_dim`.
    pub o_groups: usize,
    /// `o_lora_rank` (real: 1024) - per-group intermediate width of the grouped output projection.
    pub o_lora_rank: usize,
    /// MLP intermediate width (`intermediate_size`).
    pub intermediate: usize,
    /// `swiglu_limit` (real: 10.0) - `gate_proj`/`up_proj` pre-activations are clamped to this before the
    /// SiLU-gate multiply.
    pub swiglu_limit: f32,
    /// `sliding_window` (real: 128) - local attention window of every `sliding_attention` layer, applied via
    /// [`gemma4_local_window_floor`] on decode.
    pub sliding_window: usize,
    /// `rms_norm_eps` (real: 1e-6) - shared by every weighted and unweighted RMSNorm (`q_b_norm` and mHC's
    /// `input_norm` are the unweighted ones).
    pub eps: f32,
    pub max_pos: usize,
    /// `rope_theta` (real: 10000) - RoPE base of `sliding_attention` layers (CSA/HCA layers use
    /// `compress_rope_theta`).
    pub rope_theta: f32,
    /// `hc_mult` (real: 4) - `n_hc`, the number of parallel residual streams mHC widens to.
    pub hc_mult: usize,
    /// `hc_sinkhorn_iters` (real: 20) - Sinkhorn-Knopp round count for `comb`'s doubly-stochastic
    /// projection (asymmetric round counting, see the module doc).
    pub hc_sinkhorn_iters: usize,
    /// `hc_eps` (real: 1e-6) - floor added after every Sinkhorn normalize division and after the `pre`/`comb`
    /// sigmoid/softmax outputs.
    pub hc_eps: f32,
    /// `compress_rates["heavily_compressed_attention"]` (real: 128, `m'`) - HCA's non-overlapping pooling-window
    /// width. [`trace_deepseek4_hca_decode`]'s `cap` must be an exact multiple of it. Prefill pads `seq_len`
    /// internally instead (see [`deepseek4_prefill_pad_len`]).
    pub hca_compress_rate: usize,
    /// `compress_rope_theta` (real: 160000) - RoPE base CSA/HCA layers use for everything on that layer (Q,
    /// local K/V, compressed entries, output derotation), replacing `rope_theta`.
    pub compress_rope_theta: f32,
    /// `compress_rates["compressed_sparse_attention"]` (real: 4, `m`) - CSA's overlapping-window pooling stride
    /// (see [`csa_overlap_pool`]).
    pub csa_compress_rate: usize,
    /// `index_n_heads` (real: 64) - Lightning Indexer head count (CSA only, independent of `num_heads`).
    pub index_n_heads: usize,
    /// `index_head_dim` (real: 128) - per-indexer-head query/key width. The indexer runs its own overlapping-window
    /// compressor at this width. Must be `>= qk_rope_head_dim` (the layer's trailing-slice RoPE is reused).
    pub index_head_dim: usize,
    /// `index_topk` (real: 512) - top-`k` compressed entries selected per query, clamped to
    /// `min(index_topk, n_windows)` at trace time.
    pub index_topk: usize,
    /// `n_routed_experts` (real: 256) - routed experts per layer. Every V4 layer is MoE-routed.
    pub routed_experts: usize,
    /// `num_experts_per_tok` (real: 6) - selected routed experts per token, both layer rows.
    pub experts_per_tok: usize,
    /// `moe_intermediate_size` (real: 2048) - per-expert SwiGLU width, shared by routed and shared experts.
    pub moe_intermediate: usize,
    /// Layers `0..hash_router_layers` (real: 3) take expert ids from `ffn.gate.tid2eid`; the rest use a
    /// `ffn.gate.bias`-selected top-k. The checkpoint manifest must yield the same partition.
    pub hash_router_layers: usize,
    /// Fixed multiplier applied to the normalized selected route weights (real: 1.5).
    pub route_scale: f32,
}

#[cfg(test)]
impl DeepseekV4Config {
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    #[cfg(test)]
    pub(crate) fn in_per_group(&self) -> usize {
        assert!(
            (self.num_heads * self.head_dim).is_multiple_of(self.o_groups),
            "o_groups must divide num_heads*head_dim"
        );
        (self.num_heads * self.head_dim) / self.o_groups
    }

    #[cfg(test)]
    pub(crate) fn hc_mix_dim(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }
}

/// One layer's `layer_types` entry, fixed at model-build time by `config.layer_types[layer_idx]`.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum V4LayerKind {
    /// `"sliding_attention"` - plain local window, no compressed branch.
    Sliding,
    /// `"compressed_sparse_attention"` (CSA) - overlapping-window pooling, Lightning-Indexer top-k selected.
    Csa,
    /// `"heavily_compressed_attention"` (HCA) - non-overlapping-window pooling, dense (no top-k).
    Hca,
}

/// Decode a per-layer `compress_ratios` schedule (`0`/`4`/`128` on `DeepSeek-V4-Flash-0731`) into one
/// [`V4LayerKind`] per layer. `cfg`'s `csa_compress_rate`/`hca_compress_rate` supply the 4/128 mapping, so toy
/// configs with other rates parse too. Panics on a ratio matching none of `0`, `csa_compress_rate`,
/// `hca_compress_rate`.
#[cfg(test)]
pub(crate) fn parse_compress_ratios(cfg: &DeepseekV4Config, ratios: &[usize]) -> Vec<V4LayerKind> {
    ratios
        .iter()
        .map(|&r| {
            if r == 0 {
                V4LayerKind::Sliding
            } else if r == cfg.csa_compress_rate {
                V4LayerKind::Csa
            } else if r == cfg.hca_compress_rate {
                V4LayerKind::Hca
            } else {
                panic!(
                    "parse_compress_ratios: ratio {r} matches neither 0, csa_compress_rate ({}), \
nor hca_compress_rate ({})",
                    cfg.csa_compress_rate, cfg.hca_compress_rate
                )
            }
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
pub(crate) fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

/// Smallest length `>= seq_len` that is a valid PREFILL length for a schedule whose active compressed-
/// attention kinds are `has_hca`/`has_csa`: `seq_len` rounded up to a multiple of `lcm` of the active compress
/// rates. The prefill tracers build the graph at this padded length and slice the final logits at
/// `seq_len - 1`.
///
/// # Why padding is inert for every real query position
///
/// 1. The local/sliding branch (the `mask.prefill` step input) is strictly causal, so a real query never sees an
///    appended trailing position.
/// 2. In the compressed branch, padding tokens only enter the last window, and that window is masked out for
///    every real query. Windows are aligned in blocks of `rate` from 0, and `padded_len < seq_len + rate`, so
///    every earlier window is built from real tokens only. The last window's causal-completeness threshold
///    (`[hca,csa].block_bias`) is its last source position `padded_len - 1 >= seq_len > i` for every real
///    query `i`. That `-1e30` bias is in the final combined mask of every layer kind (HCA's `combined_mask`,
///    CSA's `csa_topk_mask`, which re-adds the causal term after top-k, so a top-k tie cannot unmask it).
///    The last window's pooled value may be anything finite.
///
/// See `deepseek4_prefill_pad_len_is_inert_for_real_prefix`.
#[cfg(test)]
pub(crate) fn deepseek4_prefill_pad_len(
    seq_len: usize,
    cfg: &DeepseekV4Config,
    has_hca: bool,
    has_csa: bool,
) -> usize {
    let rate = match (has_hca, has_csa) {
        (true, true) => lcm(cfg.hca_compress_rate, cfg.csa_compress_rate),
        (true, false) => cfg.hca_compress_rate,
        (false, true) => cfg.csa_compress_rate,
        (false, false) => return seq_len,
    };
    assert!(
        rate > 0,
        "deepseek4_prefill_pad_len: an active compress rate must be nonzero"
    );
    seq_len.div_ceil(rate) * rate
}
