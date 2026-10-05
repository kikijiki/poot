use super::*;

/// A Qwen2-family dense decoder config. Qwen2 has q/k/v bias and no QK-norm; Qwen3 drops the bias, adds
/// per-head QK-norm (RMSNorm over head_dim on q and k before RoPE), and uses an explicit head_dim (so
/// q_dim = n_heads*head_dim can exceed hidden). One trace function handles both.
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen2Config {
    pub vocab: usize,
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Number of leading head dims that get RoPE. Equals `head_dim` except for Phi-3/Phi-4 (partial rotary,
    /// e.g. 96 of 128); the trailing dims pass through unrotated.
    pub rotary_dim: usize,
    pub eps: f32,
    pub max_pos: usize,
    /// q/k/v projections have a bias (Qwen2: true, Qwen3: false).
    pub qkv_bias: bool,
    /// per-head RMSNorm on q and k before RoPE (Qwen2: false, Qwen3: true).
    pub qk_norm: bool,
    /// Sliding-window attention size. `None` = full causal; `Some(w)` restricts each query to the `w` most
    /// recent keys. Affects mask values only (filled at runtime via `prefill_causal_mask` /
    /// `decode_mask_row`). Default `None`.
    pub sliding_window: Option<usize>,
    /// Gemma2/Grok per-layer attention-logit softcap: `c * tanh(scores / c)` on scaled QK^T before softmax.
    /// `None` emits the plain attention path. Default `None`.
    pub attn_logit_softcap: Option<f32>,
    /// Gemma2/Grok final-logit softcap: `c * tanh(logits / c)` on the lm_head output before sampling.
    /// `None` skips the extra ops. Default `None`.
    pub final_logit_softcap: Option<f32>,
    /// ALiBi (Press et al., spec 254): a fixed per-head linear distance bias on the attention scores
    /// instead of RoPE (BLOOM/MPT-family). `false` (default) leaves the RoPE path unchanged. `true` makes
    /// `trace_decode_kv_masked` (and its paged/paged-quant siblings) skip `rope` and widen `Slot::Mask`
    /// from `[cap]` to `[n_heads, cap]`, filled by `poot_llm::graphs::alibi_decode_mask_row`.
    pub alibi: bool,
    /// Storage dtype for the 7 per-layer projection weights (q/k/v/o/gate/up/down). Default `DType::F32`.
    /// `DType::BF16` types those constants as bf16 so GPU backends can upload native bf16 bytes without an
    /// in-graph Cast. Norms, rope tables, and embeddings stay f32.
    pub proj_dtype: DType,
    /// Qwen2-VL/2.5-VL sectioned ("mRoPE") RoPE section widths `[temporal, height, width]`, e.g.
    /// `[16,24,24]` for Qwen2-VL-7B's 128-dim head (HF `Qwen2VLConfig.rope_scaling["mrope_section"]`).
    /// `Some(_)` routes [`trace_prefill_mrope`], [`trace_decode_mrope`], and the batched masked decode
    /// family through [`rope_sectioned`] instead of `rope_prefill`/`rope` (card 152 / spec 267). The
    /// Qwen2-VL text tower is plain Qwen2 (q/k/v bias, no o_proj bias, no QK-norm, SwiGLU, pre-norm
    /// RMSNorm), so it is `Qwen2Config` with `qkv_bias: true, qk_norm: false` plus this field. Widths
    /// must sum to `rotary_dim/2`. Default `None`.
    pub mrope_section: Option<[usize; 3]>,
}

impl Qwen2Config {
    /// Qwen2.5-0.5B-Instruct.
    pub fn qwen2_0_5b() -> Self {
        Qwen2Config {
            vocab: 151936,
            hidden: 896,
            inter: 4864,
            layers: 24,
            n_heads: 14,
            n_kv_heads: 2,
            head_dim: 64,
            rotary_dim: 64,
            eps: 1e-6,
            max_pos: 32768,
            qkv_bias: true,
            qk_norm: false,
            ..Default::default()
        }
    }

    /// Qwen3-0.6B.
    pub fn qwen3_0_6b() -> Self {
        Qwen2Config {
            vocab: 151936,
            hidden: 1024,
            inter: 3072,
            layers: 28,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            rotary_dim: 128,
            eps: 1e-6,
            max_pos: 40960,
            qkv_bias: false,
            qk_norm: true,
            ..Default::default()
        }
    }
}
