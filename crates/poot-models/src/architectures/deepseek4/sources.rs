//! The one owner of every DeepSeek-V4 graph source name (spec 364b).
//!
//! A V4 graph constant is one of four things:
//!
//! - a **packed** Card 359 owner pair, staged as the two `I8` [`poot_graph_ir::PackedSourceName`] components of one
//!   linear and read through Card 369's packed helpers;
//! - an **exact dense** Card 359 BF16 or F32 owner, staged at the checkpoint's own shape and dtype;
//! - the **exact I32** table Card 371 derives from one I64 owner (`ffn.gate.tid2eid`); or
//! - a **derived** host table with no checkpoint row: RoPE tables, iotas, causal masks, window positions,
//!   block biases.
//!
//! The first three are plan rows. The fourth is a named family (`deepseek4_is_derived_constant`) rather than
//! a row, because a derived table's shape depends on the traced length, not the config. Naming it lets a test
//! prove the partition is total: every `Storage::Const` input of every V4 tracer resolves to exactly one of
//! the four.
//!
//! # Where the names come from
//!
//! The real-name table here, verified against
//! `deepseek-ai/DeepSeek-V4-Flash-0731`'s `model.safetensors.index.json` and shard headers, carries the
//! audited packed prefixes with their logical shapes (spec 348).
//!
//! # Which dense rows are F32
//!
//! 1. Card 348 pins two rows: `ffn.gate.tid2eid` is I64 and `ffn.gate.bias` is F32.
//! 2. The mHC rows follow `Glm5NextTextSourcePlan`: the projection is BF16, the `base` vector and the `scale`
//!    triple are F32.
//! 3. Every other dense row is BF16.
//!
//! This does not reproduce Card 348's audited F32 payload total (150,966,520 bytes), and the residual cannot
//! be attributed from this repository: the audit publishes per-dtype totals over all 72,317 header names but
//! no per-name dtype list, and the DSpark/MTP dense inventory those totals include is not enumerated here.
//! Card 364c reconciles the full manifest against the authenticated index; a row it finds to be F32 is one
//! cell of this table. The I64 total closes exactly: `3 * 129280 * 6 * 8 = 18,616,320` bytes is the audited
//! I64 figure, which also shows no `mtp.*` module carries a hash table.
//!
//! This is graph metadata. It owns no checkpoint bytes and makes no loading, binding, accounting, admission,
//! device, or publication claim.

#[cfg(test)]
use super::{DeepseekV4Config, V4LayerKind, V4RouterKind};
#[cfg(test)]
use poot_graph_ir::op::PackedWeight;
#[cfg(test)]
use poot_graph_ir::op::WeightFormat;
#[cfg(test)]
use poot_graph_ir::ops::PackedLinearGraphRow;
#[cfg(test)]
use poot_graph_ir::packed_source_constants;
#[cfg(test)]
use poot_graph_ir::{PackedSourceName, TensorType};
#[cfg(test)]
use poot_load::packed_safetensors::ExactSourceKind;
#[cfg(test)]
use poot_quant::format::ScaleEncoding;
#[cfg(test)]
use poot_tensor::DType;
#[cfg(test)]
use std::collections::{BTreeMap, BTreeSet};

/// Pinned main-decoder row counts for `deepseek-ai/DeepSeek-V4-Flash-0731` at the Card 348 revision.
///
/// The E4M3 count `43 * (5 + 3) + 21` is reached only when the five attention linears, the three shared-expert
/// linears, and the indexer query projection on exactly the 21 CSA layers all carry the packed role, so a wrong
/// role or CSA predicate falsifies it. Both packed counts are Card 348's audited figures.
#[cfg(test)]
pub(crate) const V4_EXACT_E4M3_PAIR_COUNT: usize = 365;
/// `43 * 256 * 3`, Card 348's audited routed-expert pair count.
#[cfg(test)]
pub(crate) const V4_EXACT_FP4_PAIR_COUNT: usize = 33_024;
/// `6 + 43 * 12 + 40 + 41 * 4 + 21 * 5`: top level, the twelve rows every layer carries, the 40
/// score-layer biases, the compressor on 41 HCA/CSA layers, and the indexer on 21 CSA layers.
#[cfg(test)]
pub(crate) const V4_EXACT_DENSE_SOURCE_COUNT: usize = 831;
/// One `ffn.gate.tid2eid` per hash-routed layer.
#[cfg(test)]
pub(crate) const V4_EXACT_I32_SOURCE_COUNT: usize = 3;

/// The main RoPE table pair (`rope_theta`), used by `sliding_attention` layers.
#[cfg(test)]
pub(crate) const V4_ROPE_COS: &str = "rope.cos";
#[cfg(test)]
pub(crate) const V4_ROPE_SIN: &str = "rope.sin";
/// The compress RoPE table pair (`compress_rope_theta`), used by every HCA and CSA layer for its whole block.
/// Named `hca.` for both kinds for historical reasons.
#[cfg(test)]
pub(crate) const V4_COMPRESS_ROPE_COS: &str = "hca.rope.cos";
#[cfg(test)]
pub(crate) const V4_COMPRESS_ROPE_SIN: &str = "hca.rope.sin";
/// HCA and CSA compressed-branch window positions and prefill block biases.
#[cfg(test)]
pub(crate) const V4_HCA_WINDOW_POSITIONS: &str = "hca.window_positions";
#[cfg(test)]
pub(crate) const V4_HCA_BLOCK_BIAS: &str = "hca.block_bias";
#[cfg(test)]
pub(crate) const V4_CSA_WINDOW_POSITIONS: &str = "csa.window_positions";
#[cfg(test)]
pub(crate) const V4_CSA_BLOCK_BIAS: &str = "csa.block_bias";

/// The graph inputs a V4 trace computes host-side instead of reading from the checkpoint: the two RoPE table
/// pairs and the HCA/CSA window positions and block biases (step inputs since card 550a; the mask became the
/// `mask.prefill` step input and the causal range an in-graph `iota`). They have no checkpoint
/// name. The array is built from the names the tracer stages, so the set and spellings are one table.
#[cfg(test)]
pub(crate) const V4_DERIVED_CONSTANTS: [&str; 8] = [
    V4_CSA_BLOCK_BIAS,
    V4_CSA_WINDOW_POSITIONS,
    V4_HCA_BLOCK_BIAS,
    V4_COMPRESS_ROPE_COS,
    V4_COMPRESS_ROPE_SIN,
    V4_HCA_WINDOW_POSITIONS,
    V4_ROPE_COS,
    V4_ROPE_SIN,
];

/// Whether `name` is one of the host tables a V4 trace computes rather than loads.
#[cfg(test)]
pub(crate) fn deepseek4_is_derived_constant(name: &str) -> bool {
    V4_DERIVED_CONSTANTS.contains(&name)
}

/// One expert or shared-expert projection. `w2` is the down projection; `w1` gates and `w3` is the up
/// projection, the standard SwiGLU convention recorded as not yet
/// value-verified against the real checkpoint.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum V4ExpertProjection {
    W1,
    W2,
    W3,
}

#[cfg(test)]
impl V4ExpertProjection {
    pub const ALL: [Self; 3] = [Self::W1, Self::W2, Self::W3];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::W1 => "w1",
            Self::W2 => "w2",
            Self::W3 => "w3",
        }
    }

    /// Logical `[out, in]` shape, in the checkpoint's own orientation.
    pub const fn logical_shape(self, hidden: usize, intermediate: usize) -> [usize; 2] {
        match self {
            Self::W1 | Self::W3 => [intermediate, hidden],
            Self::W2 => [hidden, intermediate],
        }
    }
}

/// Every packed linear one V4 layer can carry.
///
/// Ordering is the plan's iteration order, and the derived `Ord` puts routed experts in numeric expert
/// order before their projection - never map iteration order over names.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum V4PackedRole {
    /// Low-rank query down projection.
    WqA,
    /// Low-rank query up projection.
    WqB,
    /// The single shared K==V projection.
    Wkv,
    /// Grouped output projection, first stage. Its block contracts against the grouped
    /// `[o_groups, in_per_group, o_lora_rank]` view, reached by Card 385's
    /// `ops::packed_block_diagonal_linear` (`crate::deepseek4::deepseek4_grouped_out_a`).
    WoA,
    /// Grouped output projection, second stage.
    WoB,
    /// Lightning Indexer query up projection. CSA layers only.
    IndexerWqB,
    /// The always-active shared expert.
    SharedExpert(V4ExpertProjection),
    /// One of the `routed_experts` routed experts.
    RoutedExpert {
        expert: usize,
        projection: V4ExpertProjection,
    },
}

#[cfg(test)]
impl V4PackedRole {
    /// The five attention linears every layer carries, in graph order.
    #[cfg(test)]
    pub(crate) const ATTENTION: [Self; 5] = [Self::WqA, Self::WqB, Self::Wkv, Self::WoA, Self::WoB];

    /// The checkpoint suffix under `layers.{i}.`. This is the packed linear id, with no `.weight` or
    /// `.scale`: [`poot_graph_ir::PackedSourceName`] owns those two spellings.
    pub fn suffix(self) -> String {
        match self {
            Self::WqA => "attn.wq_a".to_string(),
            Self::WqB => "attn.wq_b".to_string(),
            Self::Wkv => "attn.wkv".to_string(),
            Self::WoA => "attn.wo_a".to_string(),
            Self::WoB => "attn.wo_b".to_string(),
            Self::IndexerWqB => "attn.indexer.wq_b".to_string(),
            Self::SharedExpert(projection) => {
                format!("ffn.shared_experts.{}", projection.suffix())
            }
            Self::RoutedExpert { expert, projection } => {
                format!("ffn.experts.{expert}.{}", projection.suffix())
            }
        }
    }

    /// The routed experts are adjacent-K E2M1 pairs with row-by-32 E8M0 scales; every other packed
    /// linear is E4M3 with a 128-by-128 E8M0 scale grid (spec 348's two packed layouts).
    pub const fn format(self) -> WeightFormat {
        match self {
            Self::RoutedExpert { .. } => WeightFormat::E2m1Row32,
            _ => WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0,
            },
        }
    }

    /// Logical `[out, in]` shape, equal to `DeepseekV4PackedContract::expected_e4m3_shape` and
    /// `expected_mxfp4_shape` for the same suffix.
    #[cfg(test)]
    fn logical_shape(self, cfg: &DeepseekV4Config) -> Result<[usize; 2], V4SourceError> {
        let mul = |a: usize, b: usize| {
            a.checked_mul(b).ok_or(V4SourceError::ShapeOverflow {
                field: "packed logical extent",
            })
        };
        Ok(match self {
            Self::WqA => [cfg.q_lora_rank, cfg.hidden],
            Self::WqB => [mul(cfg.num_heads, cfg.head_dim)?, cfg.q_lora_rank],
            Self::Wkv => [cfg.head_dim, cfg.hidden],
            Self::WoA => [mul(cfg.o_groups, cfg.o_lora_rank)?, v4_in_per_group(cfg)?],
            Self::WoB => [cfg.hidden, mul(cfg.o_groups, cfg.o_lora_rank)?],
            Self::IndexerWqB => [mul(cfg.index_n_heads, cfg.index_head_dim)?, cfg.q_lora_rank],
            Self::SharedExpert(projection) | Self::RoutedExpert { projection, .. } => {
                projection.logical_shape(cfg.hidden, cfg.moe_intermediate)
            }
        })
    }
}

#[cfg(test)]
fn v4_in_per_group(cfg: &DeepseekV4Config) -> Result<usize, V4SourceError> {
    let width = cfg
        .num_heads
        .checked_mul(cfg.head_dim)
        .ok_or(V4SourceError::ShapeOverflow {
            field: "num_heads * head_dim",
        })?;
    if cfg.o_groups == 0 || !width.is_multiple_of(cfg.o_groups) {
        return Err(V4SourceError::Config {
            field: "o_groups",
            requirement: "must be nonzero and divide num_heads * head_dim",
        });
    }
    Ok(width / cfg.o_groups)
}

/// Every dense source a V4 layer of any kind carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg(test)]
pub(crate) enum V4LayerDense {
    AttnNorm,
    FfnNorm,
    QNorm,
    KvNorm,
    AttnSink,
    HcAttnProjection,
    HcAttnBase,
    HcAttnScale,
    HcFfnProjection,
    HcFfnBase,
    HcFfnScale,
    RouterGate,
}

#[cfg(test)]
impl V4LayerDense {
    pub const ALL: [Self; 12] = [
        Self::AttnNorm,
        Self::FfnNorm,
        Self::QNorm,
        Self::KvNorm,
        Self::AttnSink,
        Self::HcAttnProjection,
        Self::HcAttnBase,
        Self::HcAttnScale,
        Self::HcFfnProjection,
        Self::HcFfnBase,
        Self::HcFfnScale,
        Self::RouterGate,
    ];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::AttnNorm => "attn_norm.weight",
            Self::FfnNorm => "ffn_norm.weight",
            Self::QNorm => "attn.q_norm.weight",
            Self::KvNorm => "attn.kv_norm.weight",
            Self::AttnSink => "attn.attn_sink",
            Self::HcAttnProjection => "hc_attn_fn",
            Self::HcAttnBase => "hc_attn_base",
            Self::HcAttnScale => "hc_attn_scale",
            Self::HcFfnProjection => "hc_ffn_fn",
            Self::HcFfnBase => "hc_ffn_base",
            Self::HcFfnScale => "hc_ffn_scale",
            Self::RouterGate => "ffn.gate.weight",
        }
    }

    /// The mHC `base` and `scale` rows are F32 and everything else here is BF16; see the module doc for
    /// where that split comes from.
    pub const fn kind(self) -> ExactSourceKind {
        match self {
            Self::HcAttnBase | Self::HcAttnScale | Self::HcFfnBase | Self::HcFfnScale => {
                ExactSourceKind::F32
            }
            _ => ExactSourceKind::Bf16,
        }
    }

    /// Shape in the checkpoint's own orientation. The mHC projection is `[mix, streams * hidden]`, which
    /// `hyper_connection` wants transposed; the graph writes that transpose.
    fn shape(self, cfg: &DeepseekV4Config) -> Result<Vec<usize>, V4SourceError> {
        let mix = v4_mhc_mix_dim(cfg)?;
        let streams = v4_mhc_input_width(cfg)?;
        Ok(match self {
            Self::AttnNorm | Self::FfnNorm => vec![cfg.hidden],
            Self::QNorm => vec![cfg.q_lora_rank],
            Self::KvNorm => vec![cfg.head_dim],
            Self::AttnSink => vec![cfg.num_heads],
            Self::HcAttnProjection | Self::HcFfnProjection => vec![mix, streams],
            Self::HcAttnBase | Self::HcFfnBase => vec![mix],
            Self::HcAttnScale | Self::HcFfnScale => vec![3],
            Self::RouterGate => vec![cfg.routed_experts, cfg.hidden],
        })
    }
}

/// The compressor rows an HCA or CSA layer carries. Both kinds read the same real prefix - which one a
/// layer is comes from the schedule, not from the tensor namespace - so a layer has one row per name and
/// the pooled width follows the kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg(test)]
pub(crate) enum V4CompressorDense {
    Kv,
    Gate,
    PositionBias,
    Norm,
}

#[cfg(test)]
impl V4CompressorDense {
    pub const ALL: [Self; 4] = [Self::Kv, Self::Gate, Self::PositionBias, Self::Norm];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Kv => "attn.compressor.wkv.weight",
            Self::Gate => "attn.compressor.wgate.weight",
            Self::PositionBias => "attn.compressor.ape",
            Self::Norm => "attn.compressor.norm.weight",
        }
    }

    fn shape(self, cfg: &DeepseekV4Config, kind: V4LayerKind) -> Result<Vec<usize>, V4SourceError> {
        let (rate, width) = match kind {
            V4LayerKind::Hca => (cfg.hca_compress_rate, cfg.head_dim),
            V4LayerKind::Csa => (
                cfg.csa_compress_rate,
                cfg.head_dim
                    .checked_mul(2)
                    .ok_or(V4SourceError::ShapeOverflow {
                        field: "2 * head_dim",
                    })?,
            ),
            V4LayerKind::Sliding => {
                return Err(V4SourceError::Config {
                    field: "compressor",
                    requirement: "only HCA and CSA layers carry a compressor",
                });
            }
        };
        Ok(match self {
            Self::Kv | Self::Gate => vec![width, cfg.hidden],
            Self::PositionBias => vec![rate, width],
            Self::Norm => vec![cfg.head_dim],
        })
    }
}

/// The Lightning Indexer's dense rows. CSA layers only; its query projection is packed
/// ([`V4PackedRole::IndexerWqB`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg(test)]
pub(crate) enum V4IndexerDense {
    Kv,
    Gate,
    PositionBias,
    Norm,
    Weights,
}

#[cfg(test)]
impl V4IndexerDense {
    pub const ALL: [Self; 5] = [
        Self::Kv,
        Self::Gate,
        Self::PositionBias,
        Self::Norm,
        Self::Weights,
    ];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Kv => "attn.indexer.compressor.wkv.weight",
            Self::Gate => "attn.indexer.compressor.wgate.weight",
            Self::PositionBias => "attn.indexer.compressor.ape",
            Self::Norm => "attn.indexer.compressor.norm.weight",
            Self::Weights => "attn.indexer.weights_proj.weight",
        }
    }

    fn shape(self, cfg: &DeepseekV4Config) -> Result<Vec<usize>, V4SourceError> {
        let width = cfg
            .index_head_dim
            .checked_mul(2)
            .ok_or(V4SourceError::ShapeOverflow {
                field: "2 * index_head_dim",
            })?;
        Ok(match self {
            Self::Kv | Self::Gate => vec![width, cfg.hidden],
            Self::PositionBias => vec![cfg.csa_compress_rate, width],
            Self::Norm => vec![cfg.index_head_dim],
            Self::Weights => vec![cfg.index_n_heads, cfg.hidden],
        })
    }
}

/// The six sources outside any layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg(test)]
pub(crate) enum V4TopDense {
    Embedding,
    FinalNorm,
    Head,
    HcHeadProjection,
    HcHeadBase,
    HcHeadScale,
}

#[cfg(test)]
impl V4TopDense {
    pub const ALL: [Self; 6] = [
        Self::Embedding,
        Self::FinalNorm,
        Self::Head,
        Self::HcHeadProjection,
        Self::HcHeadBase,
        Self::HcHeadScale,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embed.weight",
            Self::FinalNorm => "norm.weight",
            Self::Head => "head.weight",
            Self::HcHeadProjection => "hc_head_fn",
            Self::HcHeadBase => "hc_head_base",
            Self::HcHeadScale => "hc_head_scale",
        }
    }

    pub const fn kind(self) -> ExactSourceKind {
        match self {
            Self::HcHeadBase | Self::HcHeadScale => ExactSourceKind::F32,
            _ => ExactSourceKind::Bf16,
        }
    }

    /// The final `hyper_head` collapse mixes to `hc_mult` lanes, not to the per-layer
    /// `(2 + hc_mult) * hc_mult`, and its scale is a single lane.
    fn shape(self, cfg: &DeepseekV4Config) -> Result<Vec<usize>, V4SourceError> {
        Ok(match self {
            Self::Embedding | Self::Head => vec![cfg.vocab, cfg.hidden],
            Self::FinalNorm => vec![cfg.hidden],
            Self::HcHeadProjection => vec![cfg.hc_mult, v4_mhc_input_width(cfg)?],
            Self::HcHeadBase => vec![cfg.hc_mult],
            Self::HcHeadScale => vec![1],
        })
    }
}

#[cfg(test)]
fn v4_mhc_mix_dim(cfg: &DeepseekV4Config) -> Result<usize, V4SourceError> {
    cfg.hc_mult
        .checked_add(2)
        .and_then(|width| width.checked_mul(cfg.hc_mult))
        .ok_or(V4SourceError::ShapeOverflow {
            field: "(2 + hc_mult) * hc_mult",
        })
}

#[cfg(test)]
fn v4_mhc_input_width(cfg: &DeepseekV4Config) -> Result<usize, V4SourceError> {
    cfg.hc_mult
        .checked_mul(cfg.hidden)
        .ok_or(V4SourceError::ShapeOverflow {
            field: "hc_mult * hidden",
        })
}

/// The name a hash-routed layer's expert-id table takes, and the name a score-routed layer's correction
/// bias takes. Which one a layer carries is how Card 364c re-derives the router partition from the
/// manifest, so the two never appear on the same layer.
#[cfg(test)]
pub(crate) const V4_HASH_TABLE_SUFFIX: &str = "ffn.gate.tid2eid";
#[cfg(test)]
pub(crate) const V4_SCORE_BIAS_SUFFIX: &str = "ffn.gate.bias";

/// `layers.{layer}`, the real checkpoint prefix for every main-decoder row.
#[cfg(test)]
pub(crate) fn v4_layer_prefix(layer: usize) -> String {
    format!("layers.{layer}")
}

/// One exact dense or exact-I32 plan row.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct V4DenseSourceSpec {
    /// The real checkpoint name.
    pub name: String,
    /// Card 359 source kind. The loader rejects a row whose safetensors dtype differs from it.
    pub kind: ExactSourceKind,
    /// Graph dtype. Equal to `kind`'s own dtype for BF16 and F32 rows; `I32` for the I64 hash table,
    /// whose graph constant is the checked no-mirror view Card 371 derives from the I64 owner.
    pub dtype: DType,
    pub shape: Vec<usize>,
}

#[cfg(test)]
impl V4DenseSourceSpec {
    /// The constant type a tracer stages this row with.
    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(self.shape.clone(), self.dtype)
    }

    /// Whether this row's graph constant is Card 371's derived exact-I32 table rather than a Card 370
    /// dense owner view.
    #[cfg(test)]
    pub(crate) const fn is_exact_i32(&self) -> bool {
        matches!(self.kind, ExactSourceKind::I64)
    }
}

/// One packed plan row.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct V4PackedSourceSpec {
    pub layer: usize,
    pub role: V4PackedRole,
    /// The packed linear id: the checkpoint prefix whose `.weight` and `.scale` are the owner pair.
    pub linear_id: String,
    pub descriptor: PackedWeight,
}

#[cfg(test)]
impl V4PackedSourceSpec {
    /// The `I8` source constants this row stages, in `PackedWeight::sources` role order. Card 379
    /// owns the suffixes.
    #[cfg(test)]
    pub(crate) fn source_constants(&self) -> Vec<(PackedSourceName, TensorType)> {
        packed_source_constants(&self.linear_id, self.descriptor)
    }

    /// The Card 369 graph row for one ordinal of an ordered expert table.
    #[cfg(test)]
    pub(crate) fn graph_row(&self, ordinal: usize) -> PackedLinearGraphRow {
        PackedLinearGraphRow {
            ordinal,
            linear_id: self.linear_id.clone(),
            descriptor: self.descriptor,
        }
    }
}

/// What one graph constant name is, once the plan has classified it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum V4SourceClass<'a> {
    /// One component of a packed owner pair.
    Packed(&'a V4PackedSourceSpec),
    /// An exact dense BF16/F32 owner, or Card 371's derived exact-I32 table.
    Dense(&'a V4DenseSourceSpec),
    /// A host table with no checkpoint row.
    Derived,
}

/// Why a source plan could not be derived.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum V4SourceError {
    #[error("DeepSeek-V4 source plan: {field} {requirement}")]
    Config {
        field: &'static str,
        requirement: &'static str,
    },
    #[error("DeepSeek-V4 source plan: {field} overflows usize")]
    ShapeOverflow { field: &'static str },
    #[error("DeepSeek-V4 source plan row count {field} is {actual}, expected {expected}")]
    InventoryCount {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

/// Why a source plan could not be derived, for the errors that need to name a row.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum V4SourcePlanError {
    #[error(transparent)]
    Source(#[from] V4SourceError),
    #[error("DeepSeek-V4 source plan repeats source {name}")]
    DuplicateSource { name: String },
    #[error("DeepSeek-V4 source plan is missing dense source {name}")]
    MissingDenseSource { name: String },
    #[error("DeepSeek-V4 source plan is missing packed source {role:?} on layer {layer}")]
    MissingPackedSource { layer: usize, role: V4PackedRole },
    #[error(
        "DeepSeek-V4 packed linear {linear_id} logical shape {logical_shape:?} is not a valid descriptor"
    )]
    PackedDescriptor {
        linear_id: String,
        logical_shape: [usize; 2],
    },
    #[error(
        "DeepSeek-V4 dense source {name} with shape {shape:?} would mirror a packed logical weight"
    )]
    DecodedWeightMirror { name: String, shape: Vec<usize> },
}

/// Every graph source name of one V4 tower, with its role.
///
/// Built from a config and a per-layer schedule, so a schedule change cannot leave a stale row behind.
/// The tracer stages constants from it and, once Card 364c lands, the loader dispositions the same rows:
/// one table, two readers, no second copy.
#[derive(Clone, Debug)]
#[cfg(test)]
pub(crate) struct DeepseekV4SourcePlan {
    cfg: DeepseekV4Config,
    schedule: Vec<V4LayerKind>,
    dense: BTreeMap<String, V4DenseSourceSpec>,
    packed: BTreeMap<(usize, V4PackedRole), V4PackedSourceSpec>,
    /// Name index over `packed`, built in the same pass, so classifying a staged constant name does not
    /// scan 33,024 rows.
    packed_by_id: BTreeMap<String, (usize, V4PackedRole)>,
}

#[cfg(test)]
impl DeepseekV4SourcePlan {
    /// Derive the plan for `cfg` under `schedule`.
    pub fn new(
        cfg: &DeepseekV4Config,
        schedule: &[V4LayerKind],
    ) -> Result<Self, V4SourcePlanError> {
        if schedule.len() != cfg.layers {
            return Err(V4SourceError::Config {
                field: "schedule",
                requirement: "must have one entry per layer",
            }
            .into());
        }
        if cfg.hash_router_layers > cfg.layers {
            return Err(V4SourceError::Config {
                field: "hash_router_layers",
                requirement: "must be at most layers",
            }
            .into());
        }
        let mut rows = V4SourceRows::default();

        for top in V4TopDense::ALL {
            rows.dense(top.name(), top.kind(), top.shape(cfg)?)?;
        }

        for (layer, &kind) in schedule.iter().enumerate() {
            let prefix = v4_layer_prefix(layer);
            for role in V4PackedRole::ATTENTION {
                rows.packed(cfg, layer, role, &prefix)?;
            }
            for projection in V4ExpertProjection::ALL {
                rows.packed(cfg, layer, V4PackedRole::SharedExpert(projection), &prefix)?;
            }
            for expert in 0..cfg.routed_experts {
                for projection in V4ExpertProjection::ALL {
                    rows.packed(
                        cfg,
                        layer,
                        V4PackedRole::RoutedExpert { expert, projection },
                        &prefix,
                    )?;
                }
            }

            for dense in V4LayerDense::ALL {
                rows.dense(
                    &format!("{prefix}.{}", dense.suffix()),
                    dense.kind(),
                    dense.shape(cfg)?,
                )?;
            }
            match cfg.router_kind(layer) {
                V4RouterKind::Hash => rows.dense(
                    &format!("{prefix}.{V4_HASH_TABLE_SUFFIX}"),
                    ExactSourceKind::I64,
                    vec![cfg.vocab, cfg.experts_per_tok],
                )?,
                V4RouterKind::Score => rows.dense(
                    &format!("{prefix}.{V4_SCORE_BIAS_SUFFIX}"),
                    ExactSourceKind::F32,
                    vec![cfg.routed_experts],
                )?,
            }

            if kind != V4LayerKind::Sliding {
                for dense in V4CompressorDense::ALL {
                    rows.dense(
                        &format!("{prefix}.{}", dense.suffix()),
                        ExactSourceKind::Bf16,
                        dense.shape(cfg, kind)?,
                    )?;
                }
            }
            if kind == V4LayerKind::Csa {
                rows.packed(cfg, layer, V4PackedRole::IndexerWqB, &prefix)?;
                for dense in V4IndexerDense::ALL {
                    rows.dense(
                        &format!("{prefix}.{}", dense.suffix()),
                        ExactSourceKind::Bf16,
                        dense.shape(cfg)?,
                    )?;
                }
            }
        }

        rows.reject_decoded_weight_mirrors()?;
        Ok(Self {
            cfg: *cfg,
            schedule: schedule.to_vec(),
            dense: rows.dense,
            packed: rows.packed,
            packed_by_id: rows.packed_by_id,
        })
    }

    pub const fn config(&self) -> &DeepseekV4Config {
        &self.cfg
    }

    pub fn schedule(&self) -> &[V4LayerKind] {
        &self.schedule
    }

    /// Dense and exact-I32 rows in name order.
    #[cfg(test)]
    pub(crate) fn dense_sources(&self) -> impl ExactSizeIterator<Item = &V4DenseSourceSpec> {
        self.dense.values()
    }

    /// Packed rows in layer, then role order.
    #[cfg(test)]
    pub(crate) fn packed_sources(&self) -> impl ExactSizeIterator<Item = &V4PackedSourceSpec> {
        self.packed.values()
    }

    /// The row `name` names, or `None` when nothing in this model stages that constant.
    ///
    /// Total by construction: a staged constant is a packed component, a dense row, or a host table.
    /// Anything else - a leftover placeholder above all - is `None`.
    pub fn classify(&self, name: &str) -> Option<V4SourceClass<'_>> {
        if let Some(parsed) = PackedSourceName::parse(name) {
            let key = self.packed_by_id.get(parsed.linear_id())?;
            return self.packed.get(key).map(V4SourceClass::Packed);
        }
        if let Some(row) = self.dense.get(name) {
            return Some(V4SourceClass::Dense(row));
        }
        deepseek4_is_derived_constant(name).then_some(V4SourceClass::Derived)
    }

    pub fn dense(&self, name: &str) -> Result<&V4DenseSourceSpec, V4SourcePlanError> {
        self.dense
            .get(name)
            .ok_or_else(|| V4SourcePlanError::MissingDenseSource {
                name: name.to_string(),
            })
    }

    /// One layer's dense row, by suffix enum rather than by a spelled-out name.
    #[cfg(test)]
    pub(crate) fn layer_dense(
        &self,
        layer: usize,
        row: V4LayerDense,
    ) -> Result<&V4DenseSourceSpec, V4SourcePlanError> {
        self.dense(&format!("{}.{}", v4_layer_prefix(layer), row.suffix()))
    }

    #[cfg(test)]
    pub(crate) fn compressor_dense(
        &self,
        layer: usize,
        row: V4CompressorDense,
    ) -> Result<&V4DenseSourceSpec, V4SourcePlanError> {
        self.dense(&format!("{}.{}", v4_layer_prefix(layer), row.suffix()))
    }

    #[cfg(test)]
    pub(crate) fn indexer_dense(
        &self,
        layer: usize,
        row: V4IndexerDense,
    ) -> Result<&V4DenseSourceSpec, V4SourcePlanError> {
        self.dense(&format!("{}.{}", v4_layer_prefix(layer), row.suffix()))
    }

    #[cfg(test)]
    pub(crate) fn top_dense(
        &self,
        row: V4TopDense,
    ) -> Result<&V4DenseSourceSpec, V4SourcePlanError> {
        self.dense(row.name())
    }

    /// The hash table of a hash-routed layer, or the correction bias of a score-routed one.
    #[cfg(test)]
    pub(crate) fn router_selection(
        &self,
        layer: usize,
    ) -> Result<&V4DenseSourceSpec, V4SourcePlanError> {
        let suffix = match self.cfg.router_kind(layer) {
            V4RouterKind::Hash => V4_HASH_TABLE_SUFFIX,
            V4RouterKind::Score => V4_SCORE_BIAS_SUFFIX,
        };
        self.dense(&format!("{}.{suffix}", v4_layer_prefix(layer)))
    }

    pub fn packed(
        &self,
        layer: usize,
        role: V4PackedRole,
    ) -> Result<&V4PackedSourceSpec, V4SourcePlanError> {
        self.packed
            .get(&(layer, role))
            .ok_or(V4SourcePlanError::MissingPackedSource { layer, role })
    }

    /// The ordered Card 369 expert table for one routed projection of `layer`: expert 0 through `E-1` in
    /// numeric order, never map iteration order.
    #[cfg(test)]
    pub(crate) fn routed_rows(
        &self,
        layer: usize,
        projection: V4ExpertProjection,
    ) -> Result<Vec<PackedLinearGraphRow>, V4SourcePlanError> {
        (0..self.cfg.routed_experts)
            .map(|expert| {
                let row = self.packed(layer, V4PackedRole::RoutedExpert { expert, projection })?;
                Ok(row.graph_row(expert))
            })
            .collect()
    }

    /// Check the pinned main-decoder counts. Only the exact config and schedule satisfy them; a tiny
    /// fixture plan is built without this call.
    #[cfg(test)]
    pub(crate) fn check_exact_counts(&self) -> Result<(), V4SourceError> {
        let e4m3 = self
            .packed
            .values()
            .filter(|row| {
                row.descriptor.format()
                    == WeightFormat::E4m3Block128 {
                        scale: ScaleEncoding::E8m0,
                    }
            })
            .count();
        let fp4 = self.packed.len() - e4m3;
        let exact_i32 = self.dense.values().filter(|row| row.is_exact_i32()).count();
        for (field, expected, actual) in [
            ("E4M3 packed pairs", V4_EXACT_E4M3_PAIR_COUNT, e4m3),
            ("FP4 packed pairs", V4_EXACT_FP4_PAIR_COUNT, fp4),
            (
                "exact dense sources",
                V4_EXACT_DENSE_SOURCE_COUNT,
                self.dense.len() - exact_i32,
            ),
            ("exact I32 sources", V4_EXACT_I32_SOURCE_COUNT, exact_i32),
        ] {
            if actual != expected {
                return Err(V4SourceError::InventoryCount {
                    field,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// Rows collected while a plan is derived. Every insert rejects a repeated name.
#[derive(Default)]
#[cfg(test)]
struct V4SourceRows {
    dense: BTreeMap<String, V4DenseSourceSpec>,
    packed: BTreeMap<(usize, V4PackedRole), V4PackedSourceSpec>,
    packed_by_id: BTreeMap<String, (usize, V4PackedRole)>,
    packed_logical_shapes: BTreeSet<[usize; 2]>,
}

#[cfg(test)]
impl V4SourceRows {
    fn dense(
        &mut self,
        name: &str,
        kind: ExactSourceKind,
        shape: Vec<usize>,
    ) -> Result<(), V4SourcePlanError> {
        let dtype = match kind {
            ExactSourceKind::Bf16 => DType::BF16,
            ExactSourceKind::F32 => DType::F32,
            // The graph constant is Card 371's checked I32 view over the I64 owner, never an f32 mirror
            // of the table.
            ExactSourceKind::I64 => DType::I32,
            ExactSourceKind::E4m3 => {
                return Err(V4SourceError::Config {
                    field: "dense source kind",
                    requirement: "E4M3 sources are packed rows, not dense rows",
                }
                .into());
            }
        };
        let row = V4DenseSourceSpec {
            name: name.to_string(),
            kind,
            dtype,
            shape,
        };
        if self.dense.insert(name.to_string(), row).is_some() {
            return Err(V4SourcePlanError::DuplicateSource {
                name: name.to_string(),
            });
        }
        Ok(())
    }

    fn packed(
        &mut self,
        cfg: &DeepseekV4Config,
        layer: usize,
        role: V4PackedRole,
        prefix: &str,
    ) -> Result<(), V4SourcePlanError> {
        let linear_id = format!("{prefix}.{}", role.suffix());
        let logical_shape = role.logical_shape(cfg)?;
        let descriptor = PackedWeight::try_new(role.format(), logical_shape).map_err(|_| {
            V4SourcePlanError::PackedDescriptor {
                linear_id: linear_id.clone(),
                logical_shape,
            }
        })?;
        if self
            .packed_by_id
            .insert(linear_id.clone(), (layer, role))
            .is_some()
        {
            return Err(V4SourcePlanError::DuplicateSource { name: linear_id });
        }
        self.packed_logical_shapes.insert(logical_shape);
        let row = V4PackedSourceSpec {
            layer,
            role,
            linear_id: linear_id.clone(),
            descriptor,
        };
        if self.packed.insert((layer, role), row).is_some() {
            return Err(V4SourcePlanError::DuplicateSource { name: linear_id });
        }
        Ok(())
    }

    /// Spec 348 FR-010's mirror rule, applied to the plan instead of to a shard: no differently named
    /// F32 matrix may have an admitted logical weight's shape or its transpose.
    ///
    /// The rule covers F32 rows only, as the audit does. A BF16 row may share a packed row's shape (an HCA
    /// layer's `attn.compressor.wkv.weight` is `[head_dim, hidden]`, the logical shape of `attn.wkv`), and
    /// rejecting that would reject the real checkpoint.
    fn reject_decoded_weight_mirrors(&self) -> Result<(), V4SourcePlanError> {
        for row in self.dense.values() {
            if row.kind != ExactSourceKind::F32 {
                continue;
            }
            let Ok([out, k]) = <[usize; 2]>::try_from(row.shape.as_slice()) else {
                continue;
            };
            if self.packed_logical_shapes.contains(&[out, k])
                || self.packed_logical_shapes.contains(&[k, out])
            {
                return Err(V4SourcePlanError::DecodedWeightMirror {
                    name: row.name.clone(),
                    shape: row.shape.clone(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
