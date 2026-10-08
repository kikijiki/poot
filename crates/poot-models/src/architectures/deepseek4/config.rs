//! DeepSeek-V4's typed config: every value checked once, here, so tracing cannot fail on it. Every
//! dimension, count and the per-layer attention schedule comes from the checkpoint's `config.json`:
//! a fine-tune with fewer layers, other compression rates or other expert counts is the same family.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::rope_table::RopeFlavor;

use super::FAMILY;
use crate::components::dims::{Field, Fields};
use crate::components::ffn::Glu;
use crate::components::mhc::MhcParams;
use crate::components::moe::MoeParams;
use crate::components::norm::NormParams;
use crate::components::rope::RopeParams;
use crate::components::router::{RouterBias, RouterKind};
use crate::components::standard::ParamError;
use crate::model::{ConfigReason, ModelError};
use crate::registry::RawConfig;

/// One layer's attention, from its `compress_ratios` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Ratio `0`: the local sliding window only.
    Sliding,
    /// The compressed-sparse-attention ratio: overlapping-window compression of the keys and a
    /// lightning indexer that keeps the `index_topk` best compressed entries per query.
    Csa,
    /// The heavily-compressed-attention ratio: non-overlapping windows, every complete one attended.
    Hca,
}

/// The attention dimensions every layer shares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AttentionDims {
    pub heads: usize,
    pub head_dim: usize,
    /// The trailing `rope_dim` of each head rotate (interleaved pairs).
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub window: usize,
    /// `head_dim ** -0.5`.
    pub scale: f32,
}

impl AttentionDims {
    /// The width of one output group: `heads * head_dim / o_groups`.
    pub fn in_per_group(&self) -> usize {
        self.heads * self.head_dim / self.o_groups
    }
}

/// The lightning indexer's dimensions, present when any layer is [`LayerKind::Csa`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct IndexerDims {
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
}

/// DeepSeek-V4's config.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekV4Config {
    pub(super) vocab: usize,
    pub(super) width: usize,
    pub(super) max_positions: usize,
    pub(super) kinds: Vec<LayerKind>,
    pub(super) csa_rate: usize,
    pub(super) hca_rate: usize,
    pub(super) attn: AttentionDims,
    pub(super) indexer: Option<IndexerDims>,
    pub(super) norm: NormParams,
    /// The rotation of the sliding layers and of the compressed layers (their own base).
    pub(super) rope: RopeParams,
    pub(super) compress_rope: RopeParams,
    pub(super) mhc: MhcParams,
    /// The routed block of a score-routed layer, and of a hash-routed one (no selection bias).
    pub(super) score_moe: MoeParams,
    pub(super) hash_moe: MoeParams,
    /// Layers `0..hash_layers` take their experts from the token-to-experts table.
    pub(super) hash_layers: usize,
    pub(super) route_scale: f32,
    pub(super) granule: NonZeroUsize,
    pub(super) eos: BTreeSet<u32>,
    pub(super) bos: Option<u32>,
}

impl DeepseekV4Config {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        // The family is HF-only: a GGUF never names it.
        let RawConfig::HfJson { config, .. } = raw else {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "general.architecture",
                reason: ConfigReason::Unsupported,
            });
        };
        let fields = Fields::new(raw, FAMILY);
        let param = |e: ParamError| e.for_family(FAMILY);
        let count = |key: &'static str| fields.count(Field::hf(key), None);
        let float = |key: &'static str, default: f32| fields.float(Field::hf(key), default);

        // The router and the scaling this family traces: refuse a config that asks for another.
        let scoring = config.get("scoring_func").filter(|v| !v.is_null());
        if scoring.is_some_and(|v| v.as_str() != Some("sqrtsoftplus")) {
            return Err(fields.error("scoring_func", ConfigReason::Unsupported));
        }
        let max_positions = count("max_position_embeddings")?;
        if !matches!(fields.hf_rope_flavor(max_positions)?, RopeFlavor::Plain) {
            return Err(fields.error("rope_scaling", ConfigReason::Unsupported));
        }

        let width = count("hidden_size")?;
        let layers = count("num_hidden_layers")?;
        let heads = count("num_attention_heads")?;
        let head_dim = count("head_dim")?;
        let rope_dim = count("qk_rope_head_dim")?;
        if rope_dim > head_dim {
            return Err(fields.error("qk_rope_head_dim", ConfigReason::Exceeds { max: head_dim }));
        }
        let o_groups = count("o_groups")?;
        if !(heads * head_dim).is_multiple_of(o_groups) {
            return Err(fields.error("o_groups", ConfigReason::NotDivisible { by: o_groups }));
        }
        let attn = AttentionDims {
            heads,
            head_dim,
            rope_dim,
            q_lora: count("q_lora_rank")?,
            o_groups,
            o_lora: count("o_lora_rank")?,
            window: count("sliding_window")?,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };

        let rates = config.get("compress_rates").filter(|v| !v.is_null());
        let rate = |key: &'static str, default: usize| -> Result<usize, ModelError> {
            match rates.and_then(|r| r.get(key)) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n > 0)
                    .ok_or(fields.error("compress_rates", ConfigReason::WrongType)),
            }
        };
        let csa_rate = rate("compressed_sparse_attention", 4)?;
        let hca_rate = rate("heavily_compressed_attention", 128)?;
        if csa_rate == hca_rate {
            return Err(fields.error("compress_rates", ConfigReason::Unsupported));
        }
        let kinds = layer_kinds(&fields, layers, csa_rate, hca_rate)?;
        let granule = granule(&fields, &kinds, csa_rate, hca_rate)?;
        let indexer = kinds
            .contains(&LayerKind::Csa)
            .then(|| -> Result<IndexerDims, ModelError> {
                let dims = IndexerDims {
                    heads: count("index_n_heads")?,
                    head_dim: count("index_head_dim")?,
                    topk: count("index_topk")?,
                };
                if dims.head_dim < rope_dim {
                    return Err(
                        fields.error("index_head_dim", ConfigReason::Exceeds { max: rope_dim })
                    );
                }
                Ok(dims)
            })
            .transpose()?;

        let eps = float("rms_norm_eps", 1e-6)?;
        let norm = NormParams::new(eps).map_err(param)?;
        let rope = |theta: f32| {
            RopeParams::new(rope_dim, theta, &RopeFlavor::Plain).map(RopeParams::interleaved)
        };
        let mhc = MhcParams::new(
            count("hc_mult")?,
            eps,
            float("hc_eps", 1e-6)?,
            count("hc_sinkhorn_iters")?,
        )
        .map_err(param)?;

        let hash_layers = fields
            .opt_u64(Field::hf("num_hash_layers"))?
            .map_or(Ok(0), |n| {
                usize::try_from(n)
                    .ok()
                    .filter(|&n| n <= layers)
                    .ok_or(fields.error("num_hash_layers", ConfigReason::Exceeds { max: layers }))
            })?;
        let route_scale = float("routed_scaling_factor", 1.0)?;
        let moe_inter = count("moe_intermediate_size")?;
        let shared = fields.count(Field::hf("n_shared_experts"), Some(1))?;
        let glu = Glu::clamped_silu(float("swiglu_limit", 10.0)?).map_err(param)?;
        let moe = |selection_bias: bool| -> Result<MoeParams, ModelError> {
            let params = MoeParams::new(
                FAMILY,
                width,
                moe_inter,
                count("n_routed_experts")?,
                count("num_experts_per_tok")?,
                RouterKind::SqrtSoftplusBiasedTopK,
            )?
            .with_shared(moe_inter * shared)
            .map_err(param)?
            .with_routed_scale(route_scale)
            .map_err(param)?
            .with_glu(glu);
            if selection_bias {
                params
                    .with_router_bias(RouterBias::SelectionOnly)
                    .map_err(param)
            } else {
                Ok(params)
            }
        };

        Ok(Self {
            vocab: fields.vocab()?,
            width,
            max_positions,
            kinds,
            csa_rate,
            hca_rate,
            attn,
            indexer,
            norm,
            rope: rope(float("rope_theta", 10_000.0)?).map_err(param)?,
            compress_rope: rope(float("compress_rope_theta", 160_000.0)?).map_err(param)?,
            mhc,
            score_moe: moe(true)?,
            hash_moe: moe(false)?,
            hash_layers,
            route_scale,
            granule,
            eos: fields.eos(None)?,
            bos: fields.bos()?,
        })
    }

    /// Whether layer `layer` routes by the token-to-experts table.
    pub(super) fn is_hash_routed(&self, layer: usize) -> bool {
        layer < self.hash_layers
    }

    /// The compression rate of a layer of `kind`; `None` for a sliding layer.
    pub(super) fn rate(&self, kind: LayerKind) -> Option<usize> {
        match kind {
            LayerKind::Sliding => None,
            LayerKind::Csa => Some(self.csa_rate),
            LayerKind::Hca => Some(self.hca_rate),
        }
    }

    /// The rotation of a layer of `kind`: the compressed layers use their own base for everything
    /// on the layer.
    pub(super) fn rope_of(&self, kind: LayerKind) -> RopeParams {
        match kind {
            LayerKind::Sliding => self.rope,
            LayerKind::Csa | LayerKind::Hca => self.compress_rope,
        }
    }
}

/// The kind of each layer, from the first `layers` entries of `compress_ratios` (the rest of the
/// list covers the checkpoint's separate multi-token-prediction modules): `0` is a sliding layer, a
/// ratio equal to a compression rate that rate's kind. Any other ratio, or a list shorter than the
/// layer count, is refused.
fn layer_kinds(
    fields: &Fields<'_>,
    layers: usize,
    csa_rate: usize,
    hca_rate: usize,
) -> Result<Vec<LayerKind>, ModelError> {
    const KEY: &str = "compress_ratios";
    let RawConfig::HfJson { config, .. } = fields.raw() else {
        unreachable!("from_raw reads only an HF config")
    };
    let ratios = config
        .get(KEY)
        .filter(|v| !v.is_null())
        .ok_or(fields.error(KEY, ConfigReason::Missing))?
        .as_array()
        .ok_or(fields.error(KEY, ConfigReason::WrongType))?;
    if ratios.len() < layers {
        return Err(fields.error(KEY, ConfigReason::Length { expected: layers }));
    }
    ratios[..layers]
        .iter()
        .map(|ratio| {
            let ratio = ratio
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(fields.error(KEY, ConfigReason::WrongType))?;
            match ratio {
                0 => Ok(LayerKind::Sliding),
                r if r == csa_rate => Ok(LayerKind::Csa),
                r if r == hca_rate => Ok(LayerKind::Hca),
                _ => Err(fields.error(KEY, ConfigReason::Unsupported)),
            }
        })
        .collect()
}

/// The least common multiple of the compression rates a schedule uses (one when it has none): a
/// prefill step covers whole compression windows of every layer, and the tokens after the last
/// whole window of a prompt run as decode steps.
fn granule(
    fields: &Fields<'_>,
    kinds: &[LayerKind],
    csa_rate: usize,
    hca_rate: usize,
) -> Result<NonZeroUsize, ModelError> {
    let overflow = || fields.error("compress_rates", ConfigReason::Exceeds { max: usize::MAX });
    let mut granule = 1usize;
    for rate in [(LayerKind::Csa, csa_rate), (LayerKind::Hca, hca_rate)]
        .into_iter()
        .filter_map(|(kind, rate)| kinds.contains(&kind).then_some(rate))
    {
        let gcd = gcd(granule, rate);
        granule = (granule / gcd).checked_mul(rate).ok_or_else(overflow)?;
    }
    Ok(NonZeroUsize::new(granule).expect("a product of nonzero rates"))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
