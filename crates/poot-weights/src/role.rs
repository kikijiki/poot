//! What a weight is for: the typed role vocabulary a model family maps its checkpoint's tensor names
//! onto, and the [`WeightId`] (layer and role) that names one mapped weight everywhere (graph constants,
//! the weight map, binders).

use std::borrow::Cow;
use std::fmt;

/// What a weight is for, independent of any checkpoint's tensor names. Families map
/// their checkpoint names onto these roles; a family card adds the sub-roles it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeightRole {
    /// The token embedding table.
    Embed,
    /// The output head: the projection from the final hidden state to the output logits.
    Head,
    /// The norm scale applied after the last layer, before the head.
    FinalNorm,
    /// A projection, bias or norm of a layer's attention.
    Attn(AttnRole),
    /// A projection or bias of a layer's dense feed-forward.
    Ffn(FfnRole),
    /// A per-layer norm scale or bias.
    Norm(NormRole),
    /// A weight of a layer's routed mixture-of-experts block.
    Moe(MoeRole),
    /// The multi-head latent attention projections beside [`AttnRole::Q`] and [`AttnRole::O`].
    Mla(MlaRole),
    /// A Mamba2 mixer's weights.
    Mamba2(Mamba2Role),
    /// A Gated DeltaNet mixer's weights.
    Gdn(GdnRole),
    /// A Kimi Delta Attention (KDA) mixer's weights.
    Kda(KdaRole),
    /// The DeepSeek sparse-attention lightning indexer's weights.
    Dsa(DsaRole),
    /// A multi-token-prediction layer's input fusion and closing norm (GLM's `eh_proj`, `enorm`,
    /// `hnorm` and `shared_head.norm`); the layer's own block reads the ordinary roles.
    Mtp(MtpRole),
    /// The block indexer of Qwen Sparse Attention.
    Qsa(QsaRole),
    /// One Hyper-Connections gate around a block (Qwen4Exp's `*_hyper_connection` modules), and the
    /// model-level mixer that collapses the streams before the head.
    Hyper(HyperSite, HyperRole),
    /// The per-layer-embedding (n-gram) block's weights.
    Ple(PleRole),
    /// DeepSeek-V4's manifold-constrained hyper-connection (mHC) mixing weights.
    Mhc(MhcRole),
    /// DeepSeek-V4's attention-window compressors (the layer's and its indexer's).
    Compress(CompressRole),
    /// The bias of a model-level LayerNorm (the final norm's; a layer norm's bias is
    /// [`NormRole::AttnBias`] or [`NormRole::FfnBias`]).
    FinalNormBias,
    /// A LayerNorm over the embedded tokens, before the first layer (BLOOM), and its bias.
    EmbedNorm,
    /// The bias of [`WeightRole::EmbedNorm`].
    EmbedNormBias,
    /// The scalar a layer multiplies its output by (Gemma 4's `layer_scalar`), `[1]`.
    LayerScale,
    /// A learned absolute position table `[positions, width]` (BERT, a vision tower), and BERT's
    /// token-type table `[types, width]`.
    PosEmbed,
    /// BERT's token-type table `[types, width]`.
    TypeEmbed,
    /// BERT's `[CLS]` pooler projection `[width, width]` and its bias.
    Pooler,
    /// The bias of [`WeightRole::Pooler`].
    PoolerBias,
    /// The bias of the output head (a classifier's), `[out]`.
    HeadBias,
    /// A vision tower's patch embedding, the stored convolution `[width, channels, patch, patch]`,
    /// and its bias.
    Patch,
    /// The bias of [`WeightRole::Patch`].
    PatchBias,
    /// A modality connector's projection `[text_width, vision_width * scale^2]`.
    Connector,
    /// A two-layer connector's first projection's bias, and its second projection and bias (a
    /// Qwen2.5-VL patch merger's `mlp.0` bias, `mlp.2` and its bias).
    ConnectorBias,
    /// The second projection of a two-layer connector.
    ConnectorOut,
    /// The bias of [`WeightRole::ConnectorOut`].
    ConnectorOutBias,
    /// The stacked LoRA adapter tensors the compile-time LoRA pass reads beside the base weights.
    Lora(LoraRole),
}

/// A projection a LoRA adapter can target: the four attention projections and the three gated-MLP ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoraTarget {
    /// The attention query projection ([`AttnRole::Q`]).
    AttnQ,
    /// The attention key projection ([`AttnRole::K`]).
    AttnK,
    /// The attention value projection ([`AttnRole::V`]).
    AttnV,
    /// The attention output projection ([`AttnRole::O`]).
    AttnO,
    /// The gated MLP's gate projection ([`FfnRole::Gate`]).
    FfnGate,
    /// The gated MLP's up projection ([`FfnRole::Up`]).
    FfnUp,
    /// The gated MLP's down projection ([`FfnRole::Down`]).
    FfnDown,
}

impl LoraTarget {
    /// Every targetable projection.
    pub const ALL: [LoraTarget; 7] = [
        LoraTarget::AttnQ,
        LoraTarget::AttnK,
        LoraTarget::AttnV,
        LoraTarget::AttnO,
        LoraTarget::FfnGate,
        LoraTarget::FfnUp,
        LoraTarget::FfnDown,
    ];

    /// The base weight role this projection is.
    pub const fn weight_role(self) -> WeightRole {
        match self {
            LoraTarget::AttnQ => WeightRole::Attn(AttnRole::Q),
            LoraTarget::AttnK => WeightRole::Attn(AttnRole::K),
            LoraTarget::AttnV => WeightRole::Attn(AttnRole::V),
            LoraTarget::AttnO => WeightRole::Attn(AttnRole::O),
            LoraTarget::FfnGate => WeightRole::Ffn(FfnRole::Gate),
            LoraTarget::FfnUp => WeightRole::Ffn(FfnRole::Up),
            LoraTarget::FfnDown => WeightRole::Ffn(FfnRole::Down),
        }
    }

    /// The projection `role` is, if it is one an adapter can target.
    pub fn of(role: WeightRole) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|target| target.weight_role() == role)
    }
}

/// Which stacked adapter tensor a [`WeightRole::Lora`] weight holds. `A` is `[adapters + 1, in, r]` and
/// `B` is `[adapters + 1, r, out]` for the projection `target` of a layer, both in `x @ W` orientation
/// and zero-padded to a common rank; row 0 of each is the no-adapter row. `Scale` is the model-level
/// `[adapters + 1]` per-adapter `lora_alpha / r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoraRole {
    /// The stacked `[adapters + 1, in, r]` factor of the adapter for this projection.
    A(LoraTarget),
    /// The stacked `[adapters + 1, r, out]` factor of the adapter for this projection.
    B(LoraTarget),
    /// The model-level `[adapters + 1]` per-adapter `lora_alpha / r`.
    Scale,
}

impl LoraRole {
    const fn spelling(self) -> &'static str {
        match self {
            LoraRole::A(LoraTarget::AttnQ) => "lora.a.attn.q",
            LoraRole::A(LoraTarget::AttnK) => "lora.a.attn.k",
            LoraRole::A(LoraTarget::AttnV) => "lora.a.attn.v",
            LoraRole::A(LoraTarget::AttnO) => "lora.a.attn.o",
            LoraRole::A(LoraTarget::FfnGate) => "lora.a.ffn.gate",
            LoraRole::A(LoraTarget::FfnUp) => "lora.a.ffn.up",
            LoraRole::A(LoraTarget::FfnDown) => "lora.a.ffn.down",
            LoraRole::B(LoraTarget::AttnQ) => "lora.b.attn.q",
            LoraRole::B(LoraTarget::AttnK) => "lora.b.attn.k",
            LoraRole::B(LoraTarget::AttnV) => "lora.b.attn.v",
            LoraRole::B(LoraTarget::AttnO) => "lora.b.attn.o",
            LoraRole::B(LoraTarget::FfnGate) => "lora.b.ffn.gate",
            LoraRole::B(LoraTarget::FfnUp) => "lora.b.ffn.up",
            LoraRole::B(LoraTarget::FfnDown) => "lora.b.ffn.down",
            LoraRole::Scale => "lora.scale",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// The weights of one layer's attention: its projections, biases and QK norms.
pub enum AttnRole {
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The output projection.
    O,
    /// The query projection's bias.
    QBias,
    /// The key projection's bias.
    KBias,
    /// The value projection's bias.
    VBias,
    /// The norm scale applied to the query (QK-norm).
    QNorm,
    /// The norm scale applied to the key (QK-norm).
    KNorm,
    /// The output projection's bias.
    OBias,
    /// A fused `[(heads + 2 * kv_heads) * head_dim, width]` QKV projection, and its bias. The row
    /// order is the family's: BLOOM interleaves per head (`q_h, k_h, v_h` for each head), a Qwen2.5-VL
    /// vision block stacks `q | k | v`. A family that has one reads it instead of `Q`/`K`/`V`.
    Qkv,
    /// The bias of the fused QKV projection ([`AttnRole::Qkv`]).
    QkvBias,
    /// The per-query-head attention sink logits (gpt-oss), `[heads]`.
    Sinks,
    /// The first stage of a grouped low-rank output projection (DeepSeek-V4's `wo_a`): the stored
    /// `[groups * rank, heads * head_dim / groups]` matrix, read as `groups` block-diagonal
    /// projections. The second stage is [`AttnRole::O`].
    OA,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// The projections and biases of one layer's dense feed-forward block.
pub enum FfnRole {
    /// The gate projection of a gated MLP.
    Gate,
    /// The up projection of a gated or plain MLP.
    Up,
    /// The down projection of a gated or plain MLP.
    Down,
    /// The biases of a plain MLP's two projections, and of a gated MLP's gate (a Qwen2.5-VL vision
    /// block's three).
    UpBias,
    /// The down projection's bias.
    DownBias,
    /// The gate projection's bias.
    GateBias,
}

/// The projection of a routed expert a weight holds. `GateUp` is the one fused gate and up projection
/// a checkpoint stores as a single tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertProj {
    /// The gate projection of one expert.
    Gate,
    /// The up projection of one expert.
    Up,
    /// The gate and up projections fused into one tensor.
    GateUp,
    /// The down projection of one expert.
    Down,
}

impl ExpertProj {
    const fn spelling(self) -> &'static str {
        match self {
            ExpertProj::Gate => "gate",
            ExpertProj::Up => "up",
            ExpertProj::GateUp => "gate_up",
            ExpertProj::Down => "down",
        }
    }
}

/// The weights of a routed mixture-of-experts block. A checkpoint stores one projection of every expert
/// as a stack ([`MoeRole::Stack`]), whether as one tensor or as per-expert tensors; per-expert packed
/// weights (each one `[out, K]` matrix) are viewed as one stack ([`WeightView::ExpertStack`]) without
/// being copied together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MoeRole {
    /// The router projection, `[experts, width]`.
    Router,
    /// The router's bias, `[experts]`.
    RouterBias,
    /// The scale on the router's normalized input, `[width]` (Gemma 4).
    RouterScale,
    /// The scale on each expert's routing weight, `[experts]` (Gemma 4's `per_expert_scale`).
    ExpertScale,
    /// One projection of every expert: the stored stack `[experts, out, K]` / `[experts, K, out]`, or
    /// per-expert tensors, dense or packed, stacked into `[experts, out, K]`
    /// ([`WeightView::ExpertStack`]).
    Stack(ExpertProj),
    /// The bias of every expert's projection, `[experts, out]`.
    StackBias(ExpertProj),
    /// One projection of the always-active shared expert, `[out, K]`.
    Shared(ExpertProj),
    /// A hash router's token-to-experts table, `[vocab, top_k]` I32 (DeepSeek-V4's `tid2eid`).
    HashTable,
    /// The shared expert's scalar gate: one logit per token, `[1, width]` (or `[width]`), through a
    /// sigmoid (Qwen3-Next).
    SharedGate,
}

/// The weights of multi-head latent attention (DeepSeek). A plain query is [`AttnRole::Q`]; the
/// output projection is [`AttnRole::O`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlaRole {
    /// The low-rank query's down projection `[q_lora_rank, width]`, its norm scale and its up
    /// projection `[heads * (nope + rope), q_lora_rank]`.
    QA,
    /// The norm scale over the query latent.
    QANorm,
    /// The query up projection out of the latent.
    QB,
    /// The shared KV latent's down projection `[kv_lora_rank + rope, width]` and its norm scale.
    KvA,
    /// The norm scale over the KV latent.
    KvANorm,
    /// HF's one up projection `[heads * (nope + v), kv_lora_rank]`, viewed per head in the graph.
    KvB,
    /// llama.cpp's split up projections, stored per head: the key's `[heads, kv_lora_rank, nope]`
    /// and the value's `[heads, v, kv_lora_rank]`.
    KB,
    /// llama.cpp's per-head value up projection.
    VB,
}

/// The weights of a Mamba2 mixer layer (Nemotron-H). The layer's pre-mixer norm is
/// [`NormRole::Attn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mamba2Role {
    /// The three row ranges of the fused input projection: the output gate `z`, the conv input
    /// `xBC` and the timestep `dt`.
    Z,
    /// The conv-input (`xBC`) rows of the fused input projection.
    Xbc,
    /// The timestep (`dt`) rows of the fused input projection.
    Dt,
    /// The bias added to `dt`.
    DtBias,
    /// The depthwise conv kernel `[K, channels]` and its bias.
    Conv,
    /// The depthwise conv's bias.
    ConvBias,
    /// `log(-A)` per head and the skip `D` per head.
    ALog,
    /// The per-head skip connection `D`.
    D,
    /// The gated RMSNorm scale before the output projection, and the output projection.
    GateNorm,
    /// The output projection.
    Out,
}

/// The weights of a Gated DeltaNet mixer layer (Qwen3-Next). The layer's pre-mixer norm is
/// [`NormRole::Attn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GdnRole {
    /// The mixed query, key and value projection `[2 * key_dim + value_dim, width]` and the output
    /// gate `z` projection `[value_dim, width]`, each separate (a GGUF; Qwen3.5).
    Qkv,
    /// The output gate `z` projection.
    Z,
    /// The per-value-head beta and decay (`alpha`) projections `[value_heads, width]`, separate.
    Beta,
    /// The per-value-head decay (`alpha`) projection.
    Alpha,
    /// The fused projections of an HF `qwen3_next` checkpoint, whose rows group per key head:
    /// `[key_heads * (2 * head_dim + 2 * value_heads / key_heads * head_dim), width]` for `q|k|v|z`
    /// and `[2 * value_heads, width]` for `b|a`.
    Qkvz,
    /// The fused `b` and `a` (beta and alpha) projection of an HF checkpoint.
    Ba,
    /// The bias added to `alpha` before its softplus, `[value_heads]`.
    DtBias,
    /// The depthwise conv kernel over the mixed channels, `[channels, 1, K]` (or `[channels, K]`).
    Conv,
    /// The per-head decay factor, `[value_heads]`: `A_log` (HF; the decay is `-exp(A_log)`) or the
    /// decay itself, already negative (GGUF's `ssm_a`).
    A,
    /// The gated RMSNorm scale over each value head's `head_dim`, and the output projection.
    GateNorm,
    /// The output projection.
    Out,
}

/// The weights of a Kimi Delta Attention mixer layer (GLM-5.3-Flash). The layer's pre-mixer norm is
/// [`NormRole::Attn`]. `heads * head_dim` is the width of each of the query, key and value
/// projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KdaRole {
    /// The query, key and value projections `[heads * head_dim, width]`.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The depthwise conv kernels over the query, key and value channels, `[heads * head_dim, 1, K]`.
    ConvQ,
    /// The depthwise conv kernel over the key channels.
    ConvK,
    /// The depthwise conv kernel over the value channels.
    ConvV,
    /// The low-rank forget-gate projections: `FA` `[gate_rank, width]` and `FB`
    /// `[heads * head_dim, gate_rank]`.
    FA,
    /// The forget gate's up projection out of the low-rank bottleneck.
    FB,
    /// The per-head beta projection `[heads, width]`.
    Beta,
    /// The low-rank output-gate projections, shaped as `FA` and `FB`.
    GA,
    /// The output gate's up projection out of the low-rank bottleneck.
    GB,
    /// The per-head log decay rate `[heads]` and the bias added to the forget gate's argument,
    /// `[heads * head_dim]`.
    ALog,
    /// The bias added to the forget gate's argument.
    DtBias,
    /// The RMSNorm scale over each head's `head_dim`, and the output projection `[width, heads *
    /// head_dim]`.
    ONorm,
    /// The output projection.
    Out,
}

/// The weights a multi-token-prediction layer adds around an ordinary block: its input fuses the
/// normalized embedding of the next token with the normalized hidden state of the position, and the
/// shared head reads the block's output through its own norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MtpRole {
    /// The fusing projection `[width, 2 * width]` over `concat(enorm(embed), hnorm(hidden))`.
    EhProj,
    /// The RMSNorm scale `[width]` of the next token's embedding (`enorm`).
    ENorm,
    /// The RMSNorm scale `[width]` of the position's hidden state (`hnorm`).
    HNorm,
    /// The RMSNorm scale `[width]` applied to the block's output before the shared head.
    HeadNorm,
}

/// The lightning indexer of DeepSeek sparse attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DsaRole {
    /// The query projection from the low-rank query `[heads * dim, q_lora_rank]`.
    Q,
    /// The key projection `[dim, width]`, its LayerNorm scale and bias.
    K,
    /// The LayerNorm scale over the key.
    KNorm,
    /// The LayerNorm bias over the key.
    KNormBias,
    /// The per-head weights projection `[heads, width]`.
    Weights,
}

/// The block indexer of Qwen Sparse Attention: its query and key projections (read from one fused
/// checkpoint tensor, `[(heads + 1) * dim, width]`, query rows first) and the scales of the two
/// norms over each head's `dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QsaRole {
    /// The query projection `[heads * dim, width]`.
    Q,
    /// The shared key projection `[dim, width]`.
    K,
    /// The scale of the norm over each query head's `dim`.
    QNorm,
    /// The scale of the norm over the key's `dim`.
    KNorm,
}

/// Where a Hyper-Connections gate sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HyperSite {
    /// Around a layer's mixer (attention or Gated DeltaNet).
    Attn,
    /// Around a layer's feed-forward.
    Mlp,
    /// The model-level mixer after the last layer (a model-level weight).
    Final,
}

/// One Hyper-Connections gate's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HyperRole {
    /// The grouped RMSNorm scale over the `streams * width` state, `[streams * width]`.
    Norm,
    /// The low-rank gate projections `[rank, streams * width]` and `[streams * width, rank]`.
    MixDown,
    /// The low-rank gate's up projection.
    MixUp,
    /// The per-stream injection weights `[streams, streams * width]` (not on the final mixer).
    Inject,
}

/// The weights of a per-layer-embedding block: the n-gram embedding table and the projections,
/// norms and short conv that gate it into the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PleRole {
    /// The n-gram embedding table `[rows, head_dim]`.
    Table,
    /// One row-range shard of the table, `[shard_rows, head_dim]`, in logical row order: shard `i`
    /// holds the rows after shards `0..i`. A table is stored as `Table` or as shards, never both.
    TableShard(u32),
    /// The scalar a sharded table's rows are multiplied by after the lookup, `[]`.
    TableScale,
    /// The key projection `[streams * width, embed_dim]` and value projection `[width, embed_dim]`.
    Key,
    /// The value projection.
    Value,
    /// The grouped norm scales over the key, the query (the stream) and the gated value.
    KeyNorm,
    /// The grouped norm scale over the query (the stream).
    QueryNorm,
    /// The grouped norm scale over the gated value.
    ConvNorm,
    /// The dilated depthwise conv kernel `[streams * width, 1, K]`.
    Conv,
}

/// The weights of one mHC site. A layer has an `Attn` and an `Ffn` site and the model one `Head`
/// collapse; each site is a projection `[mix, streams * width]`, a base `[mix]` and a scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MhcRole {
    /// The attention site's projection `[mix, streams * width]`.
    AttnFn,
    /// The attention site's base `[mix]`.
    AttnBase,
    /// The attention site's scale.
    AttnScale,
    /// The feed-forward site's projection `[mix, streams * width]`.
    FfnFn,
    /// The feed-forward site's base `[mix]`.
    FfnBase,
    /// The feed-forward site's scale.
    FfnScale,
    /// The model-level head collapse's projection `[mix, streams * width]`.
    HeadFn,
    /// The model-level head collapse's base `[mix]`.
    HeadBase,
    /// The model-level head collapse's scale.
    HeadScale,
}

/// The weights of an attention-window compressor: the key/value and gate projections of the layer's
/// compressed branch, the learned position bias added to the gate, and the norm of the pooled
/// entry. The `Indexer` rows are the same four for the lightning indexer's own compressor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompressRole {
    /// The key/value projection of the layer's compressed branch.
    Kv,
    /// The gate projection of the layer's compressed branch.
    Gate,
    /// The learned position bias added to the compressor's gate.
    PositionBias,
    /// The norm scale of the pooled entry.
    Norm,
    /// The key/value projection of the lightning indexer's compressor.
    IndexerKv,
    /// The gate projection of the lightning indexer's compressor.
    IndexerGate,
    /// The position bias added to the indexer compressor's gate.
    IndexerPositionBias,
    /// The norm scale of the indexer compressor's pooled entry.
    IndexerNorm,
}

/// The per-layer norms around a block (the model-level output norm is [`WeightRole::FinalNorm`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NormRole {
    /// The norm before the layer's mixer (attention or recurrent).
    Attn,
    /// The norm before the layer's feed-forward.
    Ffn,
    /// The norms applied to a block's output before it joins the residual (a sandwich layer's
    /// post-attention and post-feed-forward norms).
    PostAttn,
    /// The norm applied to the feed-forward's output before it joins the residual.
    PostFfn,
    /// The biases of a LayerNorm's [`NormRole::Attn`] and [`NormRole::Ffn`] scales.
    AttnBias,
    /// The bias of the [`NormRole::Ffn`] LayerNorm.
    FfnBias,
    /// The norm before a routed block that sits beside a dense feed-forward (Gemma 4's MoE
    /// layers): the dense block reads [`NormRole::Ffn`], the routed experts read this one.
    PreRouted,
    /// The norms applied to the dense and the routed block's outputs before they are summed.
    PostDense,
    /// The norm applied to the routed block's output before the sum.
    PostRouted,
}

impl WeightRole {
    /// The role's stable spelling inside [`WeightId::const_name`]. Written out per role, never
    /// derived from `Debug`: the name enters kernel caches and plan summaries, so renaming a variant
    /// must not silently rename a graph const.
    pub fn spelling(self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            WeightRole::Moe(role) => return role.spelling(),
            WeightRole::Mla(role) => match role {
                MlaRole::QA => "mla.q_a",
                MlaRole::QANorm => "mla.q_a_norm",
                MlaRole::QB => "mla.q_b",
                MlaRole::KvA => "mla.kv_a",
                MlaRole::KvANorm => "mla.kv_a_norm",
                MlaRole::KvB => "mla.kv_b",
                MlaRole::KB => "mla.k_b",
                MlaRole::VB => "mla.v_b",
            },
            WeightRole::Mamba2(role) => match role {
                Mamba2Role::Z => "mamba2.z",
                Mamba2Role::Xbc => "mamba2.xbc",
                Mamba2Role::Dt => "mamba2.dt",
                Mamba2Role::DtBias => "mamba2.dt_bias",
                Mamba2Role::Conv => "mamba2.conv",
                Mamba2Role::ConvBias => "mamba2.conv_bias",
                Mamba2Role::ALog => "mamba2.a_log",
                Mamba2Role::D => "mamba2.d",
                Mamba2Role::GateNorm => "mamba2.gate_norm",
                Mamba2Role::Out => "mamba2.out",
            },
            WeightRole::Gdn(role) => match role {
                GdnRole::Qkv => "gdn.qkv",
                GdnRole::Z => "gdn.z",
                GdnRole::Beta => "gdn.beta",
                GdnRole::Alpha => "gdn.alpha",
                GdnRole::Qkvz => "gdn.qkvz",
                GdnRole::Ba => "gdn.ba",
                GdnRole::DtBias => "gdn.dt_bias",
                GdnRole::Conv => "gdn.conv",
                GdnRole::A => "gdn.a",
                GdnRole::GateNorm => "gdn.gate_norm",
                GdnRole::Out => "gdn.out",
            },
            WeightRole::Kda(role) => match role {
                KdaRole::Q => "kda.q",
                KdaRole::K => "kda.k",
                KdaRole::V => "kda.v",
                KdaRole::ConvQ => "kda.conv_q",
                KdaRole::ConvK => "kda.conv_k",
                KdaRole::ConvV => "kda.conv_v",
                KdaRole::FA => "kda.f_a",
                KdaRole::FB => "kda.f_b",
                KdaRole::Beta => "kda.beta",
                KdaRole::GA => "kda.g_a",
                KdaRole::GB => "kda.g_b",
                KdaRole::ALog => "kda.a_log",
                KdaRole::DtBias => "kda.dt_bias",
                KdaRole::ONorm => "kda.o_norm",
                KdaRole::Out => "kda.out",
            },
            WeightRole::Mtp(role) => match role {
                MtpRole::EhProj => "mtp.eh_proj",
                MtpRole::ENorm => "mtp.enorm",
                MtpRole::HNorm => "mtp.hnorm",
                MtpRole::HeadNorm => "mtp.head_norm",
            },
            WeightRole::Dsa(role) => match role {
                DsaRole::Q => "dsa.q",
                DsaRole::K => "dsa.k",
                DsaRole::KNorm => "dsa.k_norm",
                DsaRole::KNormBias => "dsa.k_norm_bias",
                DsaRole::Weights => "dsa.weights",
            },
            WeightRole::Qsa(role) => match role {
                QsaRole::Q => "qsa.q",
                QsaRole::K => "qsa.k",
                QsaRole::QNorm => "qsa.q_norm",
                QsaRole::KNorm => "qsa.k_norm",
            },
            WeightRole::Hyper(site, role) => {
                return Cow::Owned(format!(
                    "hyper.{}.{}",
                    match site {
                        HyperSite::Attn => "attn",
                        HyperSite::Mlp => "mlp",
                        HyperSite::Final => "final",
                    },
                    match role {
                        HyperRole::Norm => "norm",
                        HyperRole::MixDown => "mix_down",
                        HyperRole::MixUp => "mix_up",
                        HyperRole::Inject => "inject",
                    }
                ));
            }
            WeightRole::Ple(role) => match role {
                PleRole::Table => "ple.table",
                PleRole::TableShard(shard) => {
                    return Cow::Owned(format!("ple.table.shard_{shard}"));
                }
                PleRole::TableScale => "ple.table_scale",
                PleRole::Key => "ple.key",
                PleRole::Value => "ple.value",
                PleRole::KeyNorm => "ple.key_norm",
                PleRole::QueryNorm => "ple.query_norm",
                PleRole::ConvNorm => "ple.conv_norm",
                PleRole::Conv => "ple.conv",
            },
            WeightRole::Mhc(role) => match role {
                MhcRole::AttnFn => "mhc.attn_fn",
                MhcRole::AttnBase => "mhc.attn_base",
                MhcRole::AttnScale => "mhc.attn_scale",
                MhcRole::FfnFn => "mhc.ffn_fn",
                MhcRole::FfnBase => "mhc.ffn_base",
                MhcRole::FfnScale => "mhc.ffn_scale",
                MhcRole::HeadFn => "mhc.head_fn",
                MhcRole::HeadBase => "mhc.head_base",
                MhcRole::HeadScale => "mhc.head_scale",
            },
            WeightRole::Compress(role) => match role {
                CompressRole::Kv => "compress.kv",
                CompressRole::Gate => "compress.gate",
                CompressRole::PositionBias => "compress.position_bias",
                CompressRole::Norm => "compress.norm",
                CompressRole::IndexerKv => "compress.indexer_kv",
                CompressRole::IndexerGate => "compress.indexer_gate",
                CompressRole::IndexerPositionBias => "compress.indexer_position_bias",
                CompressRole::IndexerNorm => "compress.indexer_norm",
            },
            WeightRole::Embed => "embed",
            WeightRole::Head => "head",
            WeightRole::FinalNorm => "final_norm",
            WeightRole::FinalNormBias => "final_norm_bias",
            WeightRole::EmbedNorm => "embed_norm",
            WeightRole::EmbedNormBias => "embed_norm_bias",
            WeightRole::Attn(role) => match role {
                AttnRole::Q => "attn.q",
                AttnRole::K => "attn.k",
                AttnRole::V => "attn.v",
                AttnRole::O => "attn.o",
                AttnRole::QBias => "attn.q_bias",
                AttnRole::KBias => "attn.k_bias",
                AttnRole::VBias => "attn.v_bias",
                AttnRole::QNorm => "attn.q_norm",
                AttnRole::KNorm => "attn.k_norm",
                AttnRole::OBias => "attn.o_bias",
                AttnRole::Qkv => "attn.qkv",
                AttnRole::QkvBias => "attn.qkv_bias",
                AttnRole::Sinks => "attn.sinks",
                AttnRole::OA => "attn.o_a",
            },
            WeightRole::Ffn(role) => match role {
                FfnRole::Gate => "ffn.gate",
                FfnRole::Up => "ffn.up",
                FfnRole::Down => "ffn.down",
                FfnRole::UpBias => "ffn.up_bias",
                FfnRole::DownBias => "ffn.down_bias",
                FfnRole::GateBias => "ffn.gate_bias",
            },
            WeightRole::Norm(role) => match role {
                NormRole::Attn => "norm.attn",
                NormRole::Ffn => "norm.ffn",
                NormRole::PostAttn => "norm.post_attn",
                NormRole::PostFfn => "norm.post_ffn",
                NormRole::AttnBias => "norm.attn_bias",
                NormRole::FfnBias => "norm.ffn_bias",
                NormRole::PreRouted => "norm.pre_routed",
                NormRole::PostDense => "norm.post_dense",
                NormRole::PostRouted => "norm.post_routed",
            },
            WeightRole::LayerScale => "layer_scale",
            WeightRole::PosEmbed => "pos_embed",
            WeightRole::TypeEmbed => "type_embed",
            WeightRole::Pooler => "pooler",
            WeightRole::PoolerBias => "pooler_bias",
            WeightRole::HeadBias => "head_bias",
            WeightRole::Patch => "patch",
            WeightRole::PatchBias => "patch_bias",
            WeightRole::Connector => "connector",
            WeightRole::ConnectorBias => "connector_bias",
            WeightRole::ConnectorOut => "connector_out",
            WeightRole::ConnectorOutBias => "connector_out_bias",
            WeightRole::Lora(role) => role.spelling(),
        })
    }
}

impl MoeRole {
    fn spelling(self) -> Cow<'static, str> {
        match self {
            MoeRole::Router => Cow::Borrowed("moe.router"),
            MoeRole::RouterBias => Cow::Borrowed("moe.router_bias"),
            MoeRole::RouterScale => Cow::Borrowed("moe.router_scale"),
            MoeRole::ExpertScale => Cow::Borrowed("moe.expert_scale"),
            MoeRole::Stack(proj) => Cow::Owned(format!("moe.stack.{}", proj.spelling())),
            MoeRole::StackBias(proj) => Cow::Owned(format!("moe.stack_bias.{}", proj.spelling())),
            MoeRole::Shared(proj) => Cow::Owned(format!("moe.shared.{}", proj.spelling())),
            MoeRole::HashTable => Cow::Borrowed("moe.hash_table"),
            MoeRole::SharedGate => Cow::Borrowed("moe.shared_gate"),
        }
    }
}

/// The typed identity of one model weight: its layer (`None` for a model-level weight) and role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightId {
    /// The layer index, or `None` for a model-level weight.
    pub layer: Option<usize>,
    /// What the weight is for.
    pub role: WeightRole,
}

impl WeightId {
    /// The identity of a model-level weight (no layer).
    pub const fn model(role: WeightRole) -> Self {
        Self { layer: None, role }
    }

    /// The identity of the `role` weight of layer `layer`.
    pub const fn layer(layer: usize, role: WeightRole) -> Self {
        Self {
            layer: Some(layer),
            role,
        }
    }

    /// The graph const name of this weight: `w.l3.attn.q` for a layer weight, `w.embed` for a
    /// model-level one. Every tracer and every binder spells a mapped weight this way.
    pub fn const_name(&self) -> String {
        match self.layer {
            Some(layer) => format!("w.l{layer}.{}", self.role.spelling()),
            None => format!("w.{}", self.role.spelling()),
        }
    }
}

impl fmt::Display for WeightId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.const_name())
    }
}
