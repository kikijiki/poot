//! The GGUF name table: llama.cpp's tensor names for every dense decoder family, as [`NameRow`]s
//! (R-562-3). [`GGUF_BASE`] holds the rows the `blk.{l}` layout shares; a family adds the rows its
//! architecture carries beside it (a later table's row for a role replaces an earlier one).

use poot_quant::weights::{AttnRole, ExpertProj, FfnRole, MoeRole, NormRole, WeightRole};

use super::{FusedPart, NameRow, Presence, Source};

const fn layer_fused(role: WeightRole, stem: &'static str, part: FusedPart) -> NameRow {
    NameRow {
        role,
        source: Source::LayerFused {
            prefix: "blk.",
            stem,
            part,
        },
        presence: Presence::Required,
    }
}

/// Rows beside [`GGUF_BASE`] for a GGUF that fuses `q|k|v` into `attn_qkv` and `gate|up` into
/// `ffn_up` (phi3): they replace the separate projections' rows.
pub(crate) const GGUF_FUSED_QKV_GATE_UP: &[NameRow] = &[
    layer_fused(WeightRole::Attn(AttnRole::Q), ".attn_qkv", FusedPart::Q),
    layer_fused(WeightRole::Attn(AttnRole::K), ".attn_qkv", FusedPart::K),
    layer_fused(WeightRole::Attn(AttnRole::V), ".attn_qkv", FusedPart::V),
    layer_fused(WeightRole::Ffn(FfnRole::Gate), ".ffn_up", FusedPart::Gate),
    layer_fused(WeightRole::Ffn(FfnRole::Up), ".ffn_up", FusedPart::Up),
];

const fn layer_linear(role: WeightRole, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear {
            prefix: "blk.",
            stem,
        },
        presence: Presence::Required,
    }
}

const fn layer_tensor(role: WeightRole, suffix: &'static str, presence: Presence) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor {
            prefix: "blk.",
            suffix,
        },
        presence,
    }
}

/// The rows every GGUF decoder shares: embedding, output norm, the head (an absent `output.weight`
/// is a head tied to the embedding), and per layer the attention and SwiGLU projections, their
/// norms and the optional QKV biases. The Q/K rows here are the stored tensors; a llama-architecture
/// GGUF stores them permuted, which [`GGUF_PAIRED_QK`] undoes.
pub(crate) const GGUF_BASE: &[NameRow] = &[
    NameRow {
        role: WeightRole::Embed,
        source: Source::Tensor("token_embd.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNorm,
        source: Source::Tensor("output_norm.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("output"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    layer_linear(WeightRole::Attn(AttnRole::Q), ".attn_q"),
    layer_linear(WeightRole::Attn(AttnRole::K), ".attn_k"),
    layer_linear(WeightRole::Attn(AttnRole::V), ".attn_v"),
    layer_linear(WeightRole::Attn(AttnRole::O), ".attn_output"),
    layer_tensor(
        WeightRole::Attn(AttnRole::QBias),
        ".attn_q.bias",
        Presence::Optional,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::KBias),
        ".attn_k.bias",
        Presence::Optional,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::VBias),
        ".attn_v.bias",
        Presence::Optional,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Attn),
        ".attn_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".ffn_norm.weight",
        Presence::Required,
    ),
    layer_linear(WeightRole::Ffn(FfnRole::Gate), ".ffn_gate"),
    layer_linear(WeightRole::Ffn(FfnRole::Up), ".ffn_up"),
    layer_linear(WeightRole::Ffn(FfnRole::Down), ".ffn_down"),
];

/// Rows beside [`GGUF_BASE`] for a llama-architecture GGUF (llama, mistral, mixtral, smollm3): llama.cpp's
/// converter stores the Q and K rows (and their biases) permuted for its interleaved RoPE, each head's rows
/// as adjacent pairs, and these rows read them back in the half-split order the model traces, so the one
/// fused half-split rotation serves the GGUF and the HF checkpoint alike. They replace the base rows for Q
/// and K.
pub(crate) const GGUF_PAIRED_QK: &[NameRow] = &[
    NameRow {
        role: WeightRole::Attn(AttnRole::Q),
        source: Source::PairedHeads(&Source::LayerLinear {
            prefix: "blk.",
            stem: ".attn_q",
        }),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Attn(AttnRole::K),
        source: Source::PairedHeads(&Source::LayerLinear {
            prefix: "blk.",
            stem: ".attn_k",
        }),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Attn(AttnRole::QBias),
        source: Source::PairedHeads(&Source::LayerTensor {
            prefix: "blk.",
            suffix: ".attn_q.bias",
        }),
        presence: Presence::Optional,
    },
    NameRow {
        role: WeightRole::Attn(AttnRole::KBias),
        source: Source::PairedHeads(&Source::LayerTensor {
            prefix: "blk.",
            suffix: ".attn_k.bias",
        }),
        presence: Presence::Optional,
    },
];

/// The per-layer Q/K norm scales (qwen3 per head, OLMo 2 over the projection).
pub(crate) const GGUF_QK_NORM: &[NameRow] = &[
    layer_tensor(
        WeightRole::Attn(AttnRole::QNorm),
        ".attn_q_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::KNorm),
        ".attn_k_norm.weight",
        Presence::Required,
    ),
];

/// gpt-oss's rows beside [`GGUF_BASE`]: llama.cpp names the norm before the feed-forward block
/// `post_attention_norm` (HF's `post_attention_layernorm`), and the output projection has a bias
/// and every head a sink logit; every projection's bias is required.
pub(crate) const GGUF_SINK_ATTENTION: &[NameRow] = &[
    layer_tensor(
        WeightRole::Attn(AttnRole::QBias),
        ".attn_q.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::KBias),
        ".attn_k.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::VBias),
        ".attn_v.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".post_attention_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::OBias),
        ".attn_output.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::Sinks),
        ".attn_sinks.weight",
        Presence::Required,
    ),
];

/// OLMo 2's post-block norms in place of the input norms: the layer's attention and feed-forward
/// norm roles read the norms applied to each block's output.
pub(crate) const GGUF_POST_NORMS: &[NameRow] = &[
    layer_tensor(
        WeightRole::Norm(NormRole::Attn),
        ".post_attention_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".post_ffw_norm.weight",
        Presence::Required,
    ),
];

const fn mpt_qkv(role: AttnRole, part: FusedPart) -> NameRow {
    NameRow {
        role: WeightRole::Attn(role),
        source: Source::LayerFused {
            prefix: "blk.",
            stem: ".attn_qkv",
            part,
        },
        presence: Presence::Required,
    }
}

/// mpt's rows: the fused, block-concatenated `attn_qkv` (llama.cpp keeps MPT's `q | k | v` order)
/// split into Q/K/V parts, and a plain two-matrix MLP; the head is tied to the embedding.
pub(crate) const GGUF_MPT: &[NameRow] = &[
    NameRow {
        role: WeightRole::Embed,
        source: Source::Tensor("token_embd.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNorm,
        source: Source::Tensor("output_norm.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("output"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    mpt_qkv(AttnRole::Q, FusedPart::Q),
    mpt_qkv(AttnRole::K, FusedPart::K),
    mpt_qkv(AttnRole::V, FusedPart::V),
    layer_linear(WeightRole::Attn(AttnRole::O), ".attn_output"),
    layer_tensor(
        WeightRole::Norm(NormRole::Attn),
        ".attn_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".ffn_norm.weight",
        Presence::Required,
    ),
    layer_linear(WeightRole::Ffn(FfnRole::Up), ".ffn_up"),
    layer_linear(WeightRole::Ffn(FfnRole::Down), ".ffn_down"),
];

/// bloom's rows: llama.cpp re-packs BLOOM's per-head-interleaved `query_key_value` block-
/// concatenated (`q | k | v`) as `attn_qkv`, so each of Q/K/V (and its bias) reads a row range of
/// it; every norm has a bias, there is an embedding norm, and the head is tied to the embedding.
pub(crate) const GGUF_BLOOM: &[NameRow] = &[
    NameRow {
        role: WeightRole::Embed,
        source: Source::Tensor("token_embd.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::EmbedNorm,
        source: Source::Tensor("token_embd_norm.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::EmbedNormBias,
        source: Source::Tensor("token_embd_norm.bias"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNorm,
        source: Source::Tensor("output_norm.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNormBias,
        source: Source::Tensor("output_norm.bias"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("output"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    layer_fused(WeightRole::Attn(AttnRole::Q), ".attn_qkv", FusedPart::Q),
    layer_fused(WeightRole::Attn(AttnRole::K), ".attn_qkv", FusedPart::K),
    layer_fused(WeightRole::Attn(AttnRole::V), ".attn_qkv", FusedPart::V),
    layer_fused(
        WeightRole::Attn(AttnRole::QBias),
        ".attn_qkv.bias",
        FusedPart::Q,
    ),
    layer_fused(
        WeightRole::Attn(AttnRole::KBias),
        ".attn_qkv.bias",
        FusedPart::K,
    ),
    layer_fused(
        WeightRole::Attn(AttnRole::VBias),
        ".attn_qkv.bias",
        FusedPart::V,
    ),
    layer_linear(WeightRole::Attn(AttnRole::O), ".attn_output"),
    layer_tensor(
        WeightRole::Attn(AttnRole::OBias),
        ".attn_output.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Attn),
        ".attn_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::AttnBias),
        ".attn_norm.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".ffn_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::FfnBias),
        ".ffn_norm.bias",
        Presence::Required,
    ),
    layer_linear(WeightRole::Ffn(FfnRole::Up), ".ffn_up"),
    layer_tensor(
        WeightRole::Ffn(FfnRole::UpBias),
        ".ffn_up.bias",
        Presence::Required,
    ),
    layer_linear(WeightRole::Ffn(FfnRole::Down), ".ffn_down"),
    layer_tensor(
        WeightRole::Ffn(FfnRole::DownBias),
        ".ffn_down.bias",
        Presence::Required,
    ),
];

/// The sandwich norms of Gemma 2 and 3 beside the base rows (`attn_norm` and `ffn_norm` are the
/// pre-attention and pre-feed-forward norms): the norms applied to each block's output.
pub(crate) const GGUF_SANDWICH_NORMS: &[NameRow] = &[
    layer_tensor(
        WeightRole::Norm(NormRole::PostAttn),
        ".post_attention_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::PostFfn),
        ".post_ffw_norm.weight",
        Presence::Required,
    ),
];

/// Gemma 4's routed block beside [`GGUF_GEMMA4`]: the router, its input and gate scales, the fused
/// expert stacks and the three norms the routed and the dense branches add around the sum.
pub(crate) const GGUF_GEMMA4_MOE: &[NameRow] = &[
    layer_tensor(
        WeightRole::Norm(NormRole::PreRouted),
        ".ffn_pre_norm_2.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::PostDense),
        ".ffn_post_norm_1.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::PostRouted),
        ".ffn_post_norm_2.weight",
        Presence::Required,
    ),
    layer_linear(WeightRole::Moe(MoeRole::Router), ".ffn_gate_inp"),
    layer_tensor(
        WeightRole::Moe(MoeRole::RouterScale),
        ".ffn_gate_inp.scale",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Moe(MoeRole::ExpertScale),
        ".ffn_down_exps.scale",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Moe(MoeRole::Stack(ExpertProj::GateUp)),
        ".ffn_gate_up_exps.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Moe(MoeRole::Stack(ExpertProj::Down)),
        ".ffn_down_exps.weight",
        Presence::Required,
    ),
];

/// Gemma 4's rows beside [`GGUF_BASE`], [`GGUF_QK_NORM`] and [`GGUF_SANDWICH_NORMS`]: a global layer
/// that reads V from K has no `attn_v` (so that row is optional), and every layer ends in its
/// output scale.
pub(crate) const GGUF_GEMMA4: &[NameRow] = &[
    NameRow {
        role: WeightRole::Attn(AttnRole::V),
        source: Source::LayerLinear {
            prefix: "blk.",
            stem: ".attn_v",
        },
        presence: Presence::Optional,
    },
    layer_tensor(
        WeightRole::LayerScale,
        ".layer_output_scale.weight",
        Presence::Required,
    ),
];
