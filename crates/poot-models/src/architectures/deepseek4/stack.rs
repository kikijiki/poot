//! DeepSeek-V4's layer and stack: the residual stream is `hc_mult` parallel streams per token, mixed
//! around the attention block and the feed-forward block of every layer by manifold-constrained
//! hyper-connections ([`crate::components::mhc`]), and collapsed to one stream before the closing
//! norm. This is the family's own topology; everything a block computes is a component.

use poot_graph_ir::{Builder, Traced, ValidationId};
use poot_weights::{NormRole, WeightId, WeightMap, WeightRole};

use super::config::{DeepseekV4Config, LayerKind};
use super::csa::{self, IndexerWeights};
use super::hca::{self, CompressorWeights, Owner, Pooling};
use super::moe::RoutedWeights;
use super::sliding::{self, Compressed, LocalWeights};
use crate::components::linear::{Weight, WeightError};
use crate::components::mhc::{MhcHeadWeights, MhcSite, MhcSiteWeights, mhc_post, widen};
use crate::components::norm::{norm, norm_with};
use crate::components::rope::rope_rows;
use crate::components::standard::{StackWeights, Step};
use crate::model::{LogitRows, TraceError};

/// The validation id of the token-id bound. The router witnesses of layer `l` are ids `3 * l` to
/// `3 * l + 2` ([`crate::components::witness::WitnessKind::id`]); this one is the family's own, at the
/// top of the range no layer count reaches.
const TOKEN_INDEX_IN_RANGE: ValidationId = ValidationId(u32::MAX);

/// One layer's weights.
#[derive(Debug)]
pub(super) struct Layer {
    kind: LayerKind,
    mhc_attn: MhcSiteWeights,
    mhc_ffn: MhcSiteWeights,
    attn_norm: Weight,
    ffn_norm: Weight,
    local: LocalWeights,
    compressor: Option<CompressorWeights>,
    indexer: Option<IndexerWeights>,
    routed: RoutedWeights,
}

impl Layer {
    fn new(
        map: &WeightMap,
        layer: usize,
        kind: LayerKind,
        cfg: &DeepseekV4Config,
    ) -> Result<Self, WeightError> {
        let norm_weight = |role| {
            Weight::new(
                map,
                WeightId::layer(layer, WeightRole::Norm(role)),
                &[cfg.width],
            )
        };
        let compressor = match (kind, cfg.rate(kind)) {
            (LayerKind::Sliding, _) | (_, None) => None,
            (kind, Some(rate)) => Some(CompressorWeights::new(
                map,
                layer,
                Owner::Layer,
                cfg.width,
                rate,
                cfg.attn.head_dim,
                match kind {
                    LayerKind::Csa => Pooling::Overlapping,
                    _ => Pooling::Disjoint,
                },
            )?),
        };
        Ok(Self {
            kind,
            mhc_attn: MhcSiteWeights::new(map, layer, MhcSite::Attn, cfg.width, &cfg.mhc)?,
            mhc_ffn: MhcSiteWeights::new(map, layer, MhcSite::Ffn, cfg.width, &cfg.mhc)?,
            attn_norm: norm_weight(NormRole::Attn)?,
            ffn_norm: norm_weight(NormRole::Ffn)?,
            local: LocalWeights::new(map, layer, cfg)?,
            compressor,
            indexer: (kind == LayerKind::Csa)
                .then(|| IndexerWeights::new(map, layer, cfg))
                .transpose()?,
            routed: RoutedWeights::new(map, layer, cfg)?,
        })
    }

    /// The layer over the widened streams `x[rows, tokens, streams, width]`: attention, then the
    /// routed feed-forward, each between a hyper-connection's collapse and its placement back.
    fn trace(
        &self,
        b: &Builder,
        step: &Step,
        index: usize,
        x: Traced,
        tokens: Traced,
        cfg: &DeepseekV4Config,
    ) -> Result<Traced, TraceError> {
        let mix = self.mhc_attn.pre(b, x, &cfg.mhc)?;
        let normed = norm(b, mix.collapsed, &self.attn_norm, cfg.norm)?;
        let attended = self.attention(b, step, index, normed, cfg)?;
        let x = mhc_post(b, mix.post, mix.comb, attended, x, &cfg.mhc)?;

        let mix = self.mhc_ffn.pre(b, x, &cfg.mhc)?;
        let normed = norm(b, mix.collapsed, &self.ffn_norm, cfg.norm)?;
        let routed = self.routed.trace(b, step, index, normed, tokens, cfg)?;
        mhc_post(b, mix.post, mix.comb, routed, x, &cfg.mhc)
    }

    /// The layer's attention block over the normalized input `normed`: `[rows, tokens, width]`.
    fn attention(
        &self,
        b: &Builder,
        step: &Step,
        index: usize,
        normed: Traced,
        cfg: &DeepseekV4Config,
    ) -> Result<Traced, TraceError> {
        let capacity = step.shape().capacity.get();
        let rows = rope_rows(b, cfg.rope_of(self.kind), step.pos(), capacity)?;
        let local = sliding::local(b, step, index, normed, &self.local, cfg, rows)?;
        let compressed: Option<Compressed> = match (self.kind, &self.compressor, &self.indexer) {
            (LayerKind::Hca, Some(compressor), _) => {
                hca::compressed(b, step, index, normed, &local.write_at, compressor, cfg)?
            }
            (LayerKind::Csa, Some(compressor), Some(indexer)) => {
                csa::compressed(b, step, index, normed, &local, compressor, indexer, cfg)?
            }
            _ => None,
        };
        sliding::attend(b, step, &local, compressed.as_ref(), &self.local, cfg)
    }
}

/// Every weight of the tower: the embedding, the layers, the closing collapse, norm and head.
#[derive(Debug)]
pub(super) struct Tower {
    stack: StackWeights,
    collapse: MhcHeadWeights,
    layers: Vec<Layer>,
}

impl Tower {
    pub(super) fn new(map: &WeightMap, cfg: &DeepseekV4Config) -> Result<Self, WeightError> {
        Ok(Self {
            stack: StackWeights::new(map, cfg.vocab, cfg.width)?,
            collapse: MhcHeadWeights::new(map, cfg.width, &cfg.mhc)?,
            layers: cfg
                .kinds
                .iter()
                .enumerate()
                .map(|(l, &kind)| Layer::new(map, l, kind, cfg))
                .collect::<Result<_, _>>()?,
        })
    }

    /// The step's logits: embed, widen, every layer inside its scope, collapse the streams, the
    /// final norm and the head. Under [`LogitRows::Last`] only the last token's logits.
    pub(super) fn trace(
        &self,
        b: &Builder,
        step: &Step,
        cfg: &DeepseekV4Config,
    ) -> Result<Traced, TraceError> {
        // The token ids, bounded by the vocabulary once: the embedding and every hash table index
        // read the bounded ids, and an id outside the vocabulary is the step's validation fault.
        let tokens = b.guard_index_bounds(step.tokens(), cfg.vocab)?;
        step.validate(
            TOKEN_INDEX_IN_RANGE,
            "deepseek4.token_index".to_string(),
            tokens.witness,
        );
        let embedded = self.stack.embed.embed(b, tokens.guarded)?;
        let mut x = widen(b, embedded, &cfg.mhc)?;
        for (l, layer) in self.layers.iter().enumerate() {
            let _scope = b.layer_scope(l);
            x = layer.trace(b, step, l, x, tokens.guarded, cfg)?;
        }
        let tokens = step.shape().tokens.get();
        if step.shape().logits == LogitRows::Last && tokens > 1 {
            x = b.slice(x, 1, tokens - 1, tokens)?;
        }
        let collapsed = self.collapse.collapse(b, x, &cfg.mhc)?;
        let normed = norm_with(b, collapsed, &self.stack.final_norm, cfg.norm)?;
        self.stack.head.logits(b, normed)
    }
}
