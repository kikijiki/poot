//! GLM-5.3-Flash's weight map: how the checkpoint's tensor names map onto typed [`WeightId`]s.
//!
//! A linear projection is stored dense as `{stem}.weight` or packed under the bare linear id
//! `{stem}` (the loader joins a quantized `.weight`/`.scale_inv` pair into one packed entry); every
//! other tensor is stored under its own name. A layer's rows follow its mixer and feed-forward kind.
//! A required row whose tensor is absent is the typed missing-weight error naming its key; a stored
//! tensor no row claims is ignored (a fine-tune may add tensors). No count of tensors is checked.
//!
//! The multi-token-prediction layers the checkpoint stores after the text layers
//! (`num_nextn_predict_layers` of them, at the layer indices past the text stack) are claimed as
//! known and read by no weight: the map's [`LoadReport`](poot_quant::weights::LoadReport) lists them
//! as known, not unknown.

use poot_quant::weights::{
    AttnRole, CompressRole, DsaRole, ExpertProj, FfnRole, KdaRole, MhcRole, MlaRole, MoeRole,
    NormRole, WeightId, WeightKey, WeightMap, WeightMapBuilder, WeightMapError, WeightRole,
    WeightStore, WeightView,
};

use super::FAMILY;
use super::config::{Ffn, Glm53Config, Mixer};
use crate::model::ModelError;

const TOWER: &str = "model.language_model.";
const LAYERS: &str = "model.language_model.layers.";

/// The weight map of a checkpoint over `store`.
pub(super) fn weights_of(cfg: &Glm53Config, store: &WeightStore) -> Result<WeightMap, ModelError> {
    let mut rows = Rows {
        map: WeightMap::builder(store),
        store,
    };
    rows.model(cfg)?;
    for (layer, kind) in cfg.layers.iter().enumerate() {
        rows.layer(cfg, layer, kind.mixer, kind.ffn)?;
    }
    rows.known(cfg);
    Ok(rows.map.build())
}

/// The map builder and the store it reads, with one method per way a tensor is named.
struct Rows<'s> {
    map: WeightMapBuilder<'s>,
    store: &'s WeightStore,
}

fn weight_error(source: WeightMapError) -> ModelError {
    ModelError::Weight {
        family: FAMILY,
        source,
    }
}

impl Rows<'_> {
    fn add(&mut self, id: WeightId, view: WeightView) -> Result<(), ModelError> {
        self.map.map(id, view).map(|_| ()).map_err(weight_error)
    }

    fn missing(id: WeightId, key: String) -> ModelError {
        weight_error(WeightMapError::Missing {
            id,
            key: WeightKey::from(key),
        })
    }

    /// The tensor stored under exactly `key`.
    fn tensor(
        &mut self,
        role: WeightRole,
        layer: Option<usize>,
        key: String,
    ) -> Result<(), ModelError> {
        let id = WeightId { layer, role };
        if !self.store.contains(&key) {
            return Err(Self::missing(id, key));
        }
        self.add(id, WeightView::Stored(WeightKey::from(key)))
    }

    /// The linear projection with stem `stem`: `{stem}.weight` dense, else the packed `{stem}`.
    fn linear(
        &mut self,
        role: WeightRole,
        layer: Option<usize>,
        stem: &str,
    ) -> Result<(), ModelError> {
        let id = WeightId { layer, role };
        let key = self.linear_key(stem);
        if !self.store.contains(&key) {
            return Err(Self::missing(id, key));
        }
        self.add(id, WeightView::Stored(WeightKey::from(key)))
    }

    /// The key of linear `stem`: its dense spelling unless only the packed one is stored.
    fn linear_key(&self, stem: &str) -> String {
        let dense = format!("{stem}.weight");
        if self.store.contains(&dense) || !self.store.contains(stem) {
            dense
        } else {
            stem.to_string()
        }
    }

    fn model(&mut self, cfg: &Glm53Config) -> Result<(), ModelError> {
        let embed = format!("{TOWER}embed_tokens.weight");
        self.tensor(WeightRole::Embed, None, embed.clone())?;
        self.tensor(WeightRole::FinalNorm, None, format!("{TOWER}norm.weight"))?;
        if cfg.tied_head {
            // The head reads the embedding's tensor as its own map entry.
            self.add(
                WeightId::model(WeightRole::Head),
                WeightView::Stored(WeightKey::from(embed)),
            )
        } else {
            self.linear(WeightRole::Head, None, "lm_head")
        }
    }

    fn layer(
        &mut self,
        cfg: &Glm53Config,
        layer: usize,
        mixer: Mixer,
        ffn: Ffn,
    ) -> Result<(), ModelError> {
        let l = Some(layer);
        let at = |name: &str| format!("{LAYERS}{layer}.{name}");
        for (role, name) in [
            (WeightRole::Norm(NormRole::Attn), "input_layernorm.weight"),
            (
                WeightRole::Norm(NormRole::Ffn),
                "post_attention_layernorm.weight",
            ),
            (WeightRole::Mhc(MhcRole::AttnFn), "hc_attn_fn"),
            (WeightRole::Mhc(MhcRole::AttnBase), "hc_attn_base"),
            (WeightRole::Mhc(MhcRole::AttnScale), "hc_attn_scale"),
            (WeightRole::Mhc(MhcRole::FfnFn), "hc_ffn_fn"),
            (WeightRole::Mhc(MhcRole::FfnBase), "hc_ffn_base"),
            (WeightRole::Mhc(MhcRole::FfnScale), "hc_ffn_scale"),
        ] {
            self.tensor(role, l, at(name))?;
        }

        match mixer {
            Mixer::Kda => self.kda(layer, &at("self_attn."))?,
            Mixer::Dsa => self.dsa(layer, &at("self_attn."))?,
        }

        match ffn {
            Ffn::Dense => {
                for (role, stem) in [
                    (FfnRole::Gate, "mlp.gate_proj"),
                    (FfnRole::Up, "mlp.up_proj"),
                    (FfnRole::Down, "mlp.down_proj"),
                ] {
                    self.linear(WeightRole::Ffn(role), l, &at(stem))?;
                }
            }
            Ffn::Sparse => {
                self.linear(WeightRole::Moe(MoeRole::Router), l, &at("mlp.gate"))?;
                self.tensor(
                    WeightRole::Moe(MoeRole::RouterBias),
                    l,
                    at("mlp.gate.e_score_correction_bias"),
                )?;
                for (proj, name) in PROJECTIONS {
                    self.linear(
                        WeightRole::Moe(MoeRole::Shared(proj)),
                        l,
                        &at(&format!("mlp.shared_experts.{name}")),
                    )?;
                }
                let experts = cfg
                    .moe
                    .as_ref()
                    .map_or(0, crate::components::moe::MoeParams::experts);
                self.experts(layer, experts, &at("mlp.experts."))?;
            }
        }
        Ok(())
    }

    /// A KDA mixer's rows under `prefix` (`...self_attn.`).
    fn kda(&mut self, layer: usize, prefix: &str) -> Result<(), ModelError> {
        let l = Some(layer);
        for (role, name) in [
            (KdaRole::Q, "q_proj"),
            (KdaRole::K, "k_proj"),
            (KdaRole::V, "v_proj"),
            (KdaRole::FA, "f_a_proj"),
            (KdaRole::FB, "f_b_proj"),
            (KdaRole::Beta, "b_proj"),
            (KdaRole::GA, "g_a_proj"),
            (KdaRole::GB, "g_b_proj"),
            (KdaRole::Out, "o_proj"),
        ] {
            self.linear(WeightRole::Kda(role), l, &format!("{prefix}{name}"))?;
        }
        for (role, name) in [
            (KdaRole::ConvQ, "q_conv1d.weight"),
            (KdaRole::ConvK, "k_conv1d.weight"),
            (KdaRole::ConvV, "v_conv1d.weight"),
            (KdaRole::ALog, "A_log"),
            (KdaRole::DtBias, "dt_bias"),
            (KdaRole::ONorm, "o_norm.weight"),
        ] {
            self.tensor(WeightRole::Kda(role), l, format!("{prefix}{name}"))?;
        }
        Ok(())
    }

    /// A DSA mixer's rows under `prefix` (`...self_attn.`).
    fn dsa(&mut self, layer: usize, prefix: &str) -> Result<(), ModelError> {
        let l = Some(layer);
        for (role, stem) in [
            (WeightRole::Mla(MlaRole::QA), "q_a_proj"),
            (WeightRole::Mla(MlaRole::QB), "q_b_proj"),
            (WeightRole::Mla(MlaRole::KvA), "kv_a_proj_with_mqa"),
            (WeightRole::Mla(MlaRole::KvB), "kv_b_proj"),
            (WeightRole::Attn(AttnRole::O), "o_proj"),
            (WeightRole::Dsa(DsaRole::Q), "indexer.wq_b"),
            (WeightRole::Dsa(DsaRole::K), "indexer.wk"),
            (WeightRole::Dsa(DsaRole::Weights), "indexer.weights_proj"),
        ] {
            self.linear(role, l, &format!("{prefix}{stem}"))?;
        }
        for (role, name) in [
            (WeightRole::Mla(MlaRole::QANorm), "q_a_layernorm.weight"),
            (WeightRole::Mla(MlaRole::KvANorm), "kv_a_layernorm.weight"),
            (WeightRole::Dsa(DsaRole::KNorm), "indexer.k_norm.weight"),
            (WeightRole::Dsa(DsaRole::KNormBias), "indexer.k_norm.bias"),
            (
                WeightRole::Compress(CompressRole::IndexerGate),
                "indexer.index_kpool_compress_gate",
            ),
            (
                WeightRole::Compress(CompressRole::IndexerPositionBias),
                "indexer.index_kpool_compress_ape",
            ),
        ] {
            self.tensor(role, l, format!("{prefix}{name}"))?;
        }
        Ok(())
    }

    /// The three projections of every routed expert. Dense per-expert tensors stack into one
    /// `[experts, out, K]` weight; packed ones are one `[out, K]` weight per expert.
    fn experts(&mut self, layer: usize, experts: usize, prefix: &str) -> Result<(), ModelError> {
        for (proj, name) in PROJECTIONS {
            let stack = WeightId::layer(layer, WeightRole::Moe(MoeRole::Stack(proj)));
            // The spelling expert 0 is stored under (dense unless only the packed id exists).
            let packed_form = !self.store.contains(&format!("{prefix}0.{name}.weight"))
                && self.store.contains(&format!("{prefix}0.{name}"));
            let keys = (0..experts)
                .map(|e| {
                    let stem = format!("{prefix}{e}.{name}");
                    let key = if packed_form {
                        stem
                    } else {
                        format!("{stem}.weight")
                    };
                    if self.store.contains(&key) {
                        Ok(WeightKey::from(key))
                    } else {
                        Err(Self::missing(stack, key))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.add(stack, WeightView::ExpertStack(keys))?;
        }
        Ok(())
    }

    /// Claim every tensor of the multi-token-prediction layers (the layer indices past the text
    /// stack) as known: the family recognises them and traces none of them.
    fn known(&mut self, cfg: &Glm53Config) {
        let text = cfg.layers.len();
        for key in self.store.keys() {
            let layer = key
                .as_str()
                .strip_prefix(LAYERS)
                .and_then(|rest| rest.split_once('.'))
                .and_then(|(index, _)| index.parse::<usize>().ok());
            if layer.is_some_and(|l| (text..text + cfg.mtp_layers).contains(&l)) {
                self.map.known(key.clone());
            }
        }
    }
}

/// The three projections of an expert and their checkpoint stems.
const PROJECTIONS: [(ExpertProj, &str); 3] = [
    (ExpertProj::Gate, "gate_proj"),
    (ExpertProj::Up, "up_proj"),
    (ExpertProj::Down, "down_proj"),
];
