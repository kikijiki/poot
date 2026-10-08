//! GLM-5.3-Flash's layer and stack: the residual stream is `hc_mult` parallel streams per token,
//! mixed around the token mixer and the feed-forward block of every layer by manifold-constrained
//! hyper-connections ([`crate::components::mhc`]), and collapsed to one stream by their mean before
//! the closing norm. The mixer is a KDA layer or a pooled-indexer DSA layer, the feed-forward a dense
//! block or the routed experts. This is the family's own topology; everything a block computes is a
//! component.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced};
use poot_quant::weights::{NormRole, WeightId, WeightMap, WeightRole};

use super::config::{DenseDims, Ffn, Glm53Config, Mixer};
use crate::components::dsa::{PooledDsaWeights, pooled_dsa};
use crate::components::ffn::{GatedFfnWeights, glu};
use crate::components::kda::{KdaWeights, kda};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::mhc::{MhcSite, MhcSiteWeights, mhc_post, widen};
use crate::components::moe::{MoeWeights, moe};
use crate::components::norm::{norm, norm_with};
use crate::components::standard::{StackWeights, Step};
use crate::model::LogitRows;

#[derive(Debug)]
enum MixerWeights {
    Kda(Box<KdaWeights>),
    Dsa(Box<PooledDsaWeights>),
}

#[derive(Debug)]
enum FfnWeights {
    Dense(Box<GatedFfnWeights>),
    Sparse(Box<MoeWeights>),
}

/// One layer's weights.
#[derive(Debug)]
struct Layer {
    mhc_attn: MhcSiteWeights,
    mhc_ffn: MhcSiteWeights,
    attn_norm: Weight,
    ffn_norm: Weight,
    mixer: MixerWeights,
    ffn: FfnWeights,
}

impl Layer {
    fn new(
        map: &WeightMap,
        layer: usize,
        mixer: Mixer,
        ffn: Ffn,
        cfg: &Glm53Config,
    ) -> Result<Self, WeightError> {
        let norm_weight = |role| {
            Weight::new(
                map,
                WeightId::layer(layer, WeightRole::Norm(role)),
                &[cfg.width],
            )
        };
        // `Glm53Config::from_raw` reads the parameters of a kind exactly when some layer has it.
        Ok(Self {
            mhc_attn: MhcSiteWeights::new(map, layer, MhcSite::Attn, cfg.width, &cfg.mhc)?,
            mhc_ffn: MhcSiteWeights::new(map, layer, MhcSite::Ffn, cfg.width, &cfg.mhc)?,
            attn_norm: norm_weight(NormRole::Attn)?,
            ffn_norm: norm_weight(NormRole::Ffn)?,
            mixer: match mixer {
                Mixer::Kda => MixerWeights::Kda(Box::new(KdaWeights::new(
                    map,
                    layer,
                    cfg.width,
                    cfg.kda.as_ref().expect("a KDA layer has KDA parameters"),
                )?)),
                Mixer::Dsa => MixerWeights::Dsa(Box::new(PooledDsaWeights::new(
                    map,
                    layer,
                    cfg.width,
                    cfg.dsa.as_ref().expect("a DSA layer has DSA parameters"),
                )?)),
            },
            ffn: match ffn {
                Ffn::Dense => FfnWeights::Dense(Box::new(GatedFfnWeights::new(
                    map,
                    layer,
                    cfg.width,
                    cfg.dense
                        .as_ref()
                        .expect("a dense layer has dense dimensions")
                        .inter,
                )?)),
                Ffn::Sparse => FfnWeights::Sparse(Box::new(MoeWeights::new(
                    map,
                    layer,
                    cfg.moe.as_ref().expect("a sparse layer has MoE parameters"),
                )?)),
            },
        })
    }

    /// The layer over the widened streams `x[rows, tokens, streams, width]`: the mixer, then the
    /// feed-forward, each between a hyper-connection's collapse and its placement back.
    fn trace(
        &self,
        b: &Builder,
        step: &Step,
        index: usize,
        x: Traced,
        cfg: &Glm53Config,
    ) -> Traced {
        let mix = self.mhc_attn.pre(b, x, &cfg.mhc);
        let normed = norm(b, mix.collapsed, &self.attn_norm, cfg.norm);
        let mixed = match &self.mixer {
            MixerWeights::Kda(w) => kda(
                b,
                step,
                index,
                normed,
                w,
                cfg.kda.as_ref().expect("a KDA layer has KDA parameters"),
            ),
            MixerWeights::Dsa(w) => pooled_dsa(
                b,
                step,
                index,
                normed,
                w,
                cfg.dsa.as_ref().expect("a DSA layer has DSA parameters"),
            ),
        };
        let x = mhc_post(b, mix.post, mix.comb, mixed, x, &cfg.mhc);

        let mix = self.mhc_ffn.pre(b, x, &cfg.mhc);
        let normed = norm(b, mix.collapsed, &self.ffn_norm, cfg.norm);
        let out = match &self.ffn {
            FfnWeights::Dense(w) => dense_ffn(
                b,
                normed,
                w,
                cfg.dense
                    .as_ref()
                    .expect("a dense layer has dense dimensions"),
            ),
            FfnWeights::Sparse(w) => moe(
                b,
                step,
                index,
                normed,
                None,
                w,
                cfg.moe.as_ref().expect("a sparse layer has MoE parameters"),
            ),
        };
        mhc_post(b, mix.post, mix.comb, out, x, &cfg.mhc)
    }
}

/// The dense block: `down(glu(gate(x), up(x)))` with the family's clamped SwiGLU.
fn dense_ffn(b: &Builder, x: Traced, w: &GatedFfnWeights, dims: &DenseDims) -> Traced {
    let gate = linear(b, x, &w.gate, None);
    let up = linear(b, x, &w.up, None);
    linear(b, glu(b, gate, up, &dims.glu), &w.down, None)
}

/// Every weight of the tower: the embedding, the layers, the closing norm and head.
#[derive(Debug)]
pub(super) struct Tower {
    stack: StackWeights,
    layers: Vec<Layer>,
}

impl Tower {
    pub fn new(map: &WeightMap, cfg: &Glm53Config) -> Result<Self, WeightError> {
        Ok(Self {
            stack: StackWeights::new(map, cfg.vocab, cfg.width)?,
            layers: cfg
                .layers
                .iter()
                .enumerate()
                .map(|(l, kind)| Layer::new(map, l, kind.mixer, kind.ffn, cfg))
                .collect::<Result<_, _>>()?,
        })
    }

    /// The step's logits: embed, widen, every layer inside its scope, the mean of the streams, the
    /// final norm and the head. Under [`LogitRows::Last`] only the last token's logits.
    pub fn trace(&self, b: &Builder, step: &Step, cfg: &Glm53Config) -> Traced {
        let embedded = self.stack.embed.embed(b, step.tokens());
        let mut x = widen(b, embedded, &cfg.mhc);
        for (l, layer) in self.layers.iter().enumerate() {
            let _scope = b.layer_scope(l);
            x = layer.trace(b, step, l, x, cfg);
        }
        let tokens = step.shape().tokens.get();
        if step.shape().logits == LogitRows::Last && tokens > 1 {
            x = b.slice(x, 1, tokens - 1, tokens);
        }
        let streams = cfg.mhc.streams();
        let mean = b.binary_scalar(
            BinOp::Mul,
            b.reduce(RedOp::Sum, x, 2, false),
            Scalar::F32(1.0 / streams as f32),
        );
        let normed = norm_with(b, mean, &self.stack.final_norm, cfg.norm);
        self.stack.head.logits(b, normed)
    }
}
