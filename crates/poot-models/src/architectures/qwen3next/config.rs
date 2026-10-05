use super::*;

/// How a Gated DeltaNet layer pairs its `H_k` query/key heads with its `H_v` value heads.
///
/// With `n_rep = H_v / H_k`, value head `j` reads key head `j % H_k` in tiled order and key head
/// `j / n_rep` in grouped order. The order belongs to the checkpoint source, not to the architecture:
/// HF safetensors store V heads grouped by K head and expand q/k with `repeat_interleave`, while
/// llama.cpp's converter (`_reorder_v_heads` in `conversion/qwen.py`) rewrites every per-V-head tensor
/// to tiled order so GGUF files expand q/k with a plain `ggml_repeat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnHeadOrder {
    /// GGUF files: V heads reordered by the converter, q/k tiled `[k0, k1, k0, k1, ...]`.
    Tiled,
    /// HF safetensors: V heads grouped by K head, q/k grouped `[k0, k0, k1, k1, ...]`.
    Grouped,
}

impl GdnHeadOrder {
    /// Expand per-head query or key values `[B, H_k, L, D]` to `[B, H_k * n_rep, L, D]`.
    pub(crate) fn expand(self, b: &Builder, x: Traced, n_rep: usize) -> Traced {
        match self {
            Self::Tiled => repeat_kv_tiled(b, x, n_rep),
            Self::Grouped => repeat_kv(b, x, n_rep),
        }
    }
}

/// Qwen3-Next (qwen35moe / Qwen3.6-35B-A3B) full-model decode config.
///
/// One config drives the whole hybrid stack. The per-layer type is decided by
/// [`Qwen3NextConfig::is_attn_layer`]: full-attention every `full_attention_interval`-th layer,
/// Gated DeltaNet linear attention elsewhere (section 0). Every
/// layer, of either type, is followed by the same MoE FFN + shared expert.
///
/// Real-model values (section 1): `hidden=2048`, `n_layers=40`, `full_attention_interval=4`,
/// `n_heads=16`, `n_kv_heads=2`, `head_dim=256`, `rotary_dim=64`, GDN `num_k_heads=16`,
/// `num_v_heads=32`, `gdn_head_dim=128`, `conv_k=4`, `n_experts=256`, `top_k=8`,
/// `expert_inter=512`, `shared_inter=512`, `eps=1e-6`.
#[derive(Clone, Debug)]
pub struct Qwen3NextConfig {
    pub vocab: usize,
    /// n_embd (hidden width).
    pub hidden: usize,
    pub n_layers: usize,
    /// Full-attention every `full_attention_interval`-th layer (=4 in the real model): layer `i` is
    /// full-attention iff `(i + 1) % full_attention_interval == 0`, else Gated DeltaNet.
    pub full_attention_interval: usize,
    pub eps: f32,
    pub max_pos: usize,
    /// Leading head dims that get RoPE (partial rotary: 64 of 256 in the real model). Full-attn only.
    pub rotary_dim: usize,

    // --- full-attention (gated GQA) dims ---
    /// Query heads.
    pub n_heads: usize,
    /// KV heads (GQA; `n_rep = n_heads / n_kv_heads`).
    pub n_kv_heads: usize,
    /// Per-head dim for the full-attention layers.
    pub head_dim: usize,

    // --- Gated DeltaNet (GDN) dims ---
    /// K/Q head count (`ssm.group_count`).
    pub gdn_num_k_heads: usize,
    /// V head count (`ssm.time_step_rank`); `gdn_num_v_heads / gdn_num_k_heads` GQA repeat.
    pub gdn_num_v_heads: usize,
    /// Per-head dim for GDN (`ssm.state_size`; head_k_dim == head_v_dim).
    pub gdn_head_dim: usize,
    /// Causal conv kernel width (`ssm.conv_kernel`).
    pub conv_k: usize,

    // --- MoE FFN dims (shared by both layer types) ---
    pub n_experts: usize,
    pub top_k: usize,
    /// Per-routed-expert intermediate width.
    pub expert_inter: usize,
    /// Shared-expert intermediate width.
    pub shared_inter: usize,
}

impl Qwen3NextConfig {
    /// True if layer `li` is a full-attention layer (else Gated DeltaNet). Matches the reference
    /// `is_recr(i) = (i + 1) % full_attention_interval != 0` (sec 0).
    pub fn is_attn_layer(&self, li: usize) -> bool {
        (li + 1).is_multiple_of(self.full_attention_interval)
    }
}
