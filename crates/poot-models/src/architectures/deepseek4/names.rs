//! DeepSeek-V4's weight map: how the checkpoint's tensor names map onto typed [`WeightId`]s.
//!
//! A linear projection is stored dense as `{stem}.weight` or packed under the bare linear id
//! `{stem}` (the loader joins a quantized `.weight`/`.scale` pair into one packed entry); every
//! other tensor is stored under its own name. A layer's rows follow its [`LayerKind`] and routing:
//! only a compressed layer has a compressor, only a CSA layer an indexer, and a hash-routed layer a
//! token-to-experts table where a score-routed one has a selection bias. A required row whose tensor
//! is absent is the typed missing-weight error naming its key; a stored tensor no row claims is
//! ignored (a fine-tune may add tensors). No count of tensors is checked.

use poot_quant::weights::{
    AttnRole, CompressRole, DsaRole, ExpertProj, MhcRole, MlaRole, MoeRole, NormRole, WeightId,
    WeightKey, WeightMap, WeightRole, WeightStore, WeightView,
};

use super::FAMILY;
use super::config::{DeepseekV4Config, LayerKind};
use crate::model::ModelError;
use crate::names::TensorRows;

/// The weight map of a checkpoint over `store`.
pub(super) fn weights_of(
    cfg: &DeepseekV4Config,
    store: &WeightStore,
) -> Result<WeightMap, ModelError> {
    let mut rows = Rows(TensorRows::new(FAMILY, store));
    rows.model()?;
    for (layer, &kind) in cfg.kinds.iter().enumerate() {
        rows.layer(cfg, layer, kind)?;
    }
    Ok(rows.0.finish())
}

/// The shared tensor rows, with this family's way of reading each part of a checkpoint.
struct Rows<'s>(TensorRows<'s>);

impl<'s> std::ops::Deref for Rows<'s> {
    type Target = TensorRows<'s>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Rows<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Rows<'_> {
    fn model(&mut self) -> Result<(), ModelError> {
        self.tensor(WeightRole::Embed, None, "embed.weight".into())?;
        self.tensor(WeightRole::FinalNorm, None, "norm.weight".into())?;
        self.linear(WeightRole::Head, None, "head")?;
        for (role, key) in [
            (MhcRole::HeadFn, "hc_head_fn"),
            (MhcRole::HeadBase, "hc_head_base"),
            (MhcRole::HeadScale, "hc_head_scale"),
        ] {
            self.tensor(WeightRole::Mhc(role), None, key.into())?;
        }
        Ok(())
    }

    fn layer(
        &mut self,
        cfg: &DeepseekV4Config,
        layer: usize,
        kind: LayerKind,
    ) -> Result<(), ModelError> {
        let l = Some(layer);
        let at = |name: &str| format!("layers.{layer}.{name}");
        for (role, name) in [
            (WeightRole::Norm(NormRole::Attn), "attn_norm.weight"),
            (WeightRole::Norm(NormRole::Ffn), "ffn_norm.weight"),
            (WeightRole::Mla(MlaRole::QANorm), "attn.q_norm.weight"),
            (WeightRole::Attn(AttnRole::KNorm), "attn.kv_norm.weight"),
            (WeightRole::Attn(AttnRole::Sinks), "attn.attn_sink"),
            (WeightRole::Mhc(MhcRole::AttnFn), "hc_attn_fn"),
            (WeightRole::Mhc(MhcRole::AttnBase), "hc_attn_base"),
            (WeightRole::Mhc(MhcRole::AttnScale), "hc_attn_scale"),
            (WeightRole::Mhc(MhcRole::FfnFn), "hc_ffn_fn"),
            (WeightRole::Mhc(MhcRole::FfnBase), "hc_ffn_base"),
            (WeightRole::Mhc(MhcRole::FfnScale), "hc_ffn_scale"),
        ] {
            self.tensor(role, l, at(name))?;
        }
        for (role, stem) in [
            (WeightRole::Mla(MlaRole::QA), "attn.wq_a"),
            (WeightRole::Mla(MlaRole::QB), "attn.wq_b"),
            (WeightRole::Attn(AttnRole::K), "attn.wkv"),
            (WeightRole::Attn(AttnRole::OA), "attn.wo_a"),
            (WeightRole::Attn(AttnRole::O), "attn.wo_b"),
            (WeightRole::Moe(MoeRole::Router), "ffn.gate"),
            (
                WeightRole::Moe(MoeRole::Shared(ExpertProj::Gate)),
                "ffn.shared_experts.w1",
            ),
            (
                WeightRole::Moe(MoeRole::Shared(ExpertProj::Up)),
                "ffn.shared_experts.w3",
            ),
            (
                WeightRole::Moe(MoeRole::Shared(ExpertProj::Down)),
                "ffn.shared_experts.w2",
            ),
        ] {
            self.linear(role, l, &at(stem))?;
        }

        if cfg.is_hash_routed(layer) {
            let id = WeightId::layer(layer, WeightRole::Moe(MoeRole::HashTable));
            let key = at("ffn.gate.tid2eid");
            if !self.store.contains(&key) {
                return Err(self.missing(id, key));
            }
            self.add(id, WeightView::IndexNarrow(WeightKey::from(key)))?;
        } else {
            self.tensor(WeightRole::Moe(MoeRole::RouterBias), l, at("ffn.gate.bias"))?;
        }
        self.experts(layer, cfg.score_moe.experts(), &at("ffn.experts."))?;

        if kind != LayerKind::Sliding {
            for (role, name) in [
                (CompressRole::Kv, "attn.compressor.wkv.weight"),
                (CompressRole::Gate, "attn.compressor.wgate.weight"),
                (CompressRole::PositionBias, "attn.compressor.ape"),
                (CompressRole::Norm, "attn.compressor.norm.weight"),
            ] {
                self.tensor(WeightRole::Compress(role), l, at(name))?;
            }
        }
        if kind == LayerKind::Csa {
            self.linear(WeightRole::Dsa(DsaRole::Q), l, &at("attn.indexer.wq_b"))?;
            self.tensor(
                WeightRole::Dsa(DsaRole::Weights),
                l,
                at("attn.indexer.weights_proj.weight"),
            )?;
            for (role, name) in [
                (
                    CompressRole::IndexerKv,
                    "attn.indexer.compressor.wkv.weight",
                ),
                (
                    CompressRole::IndexerGate,
                    "attn.indexer.compressor.wgate.weight",
                ),
                (
                    CompressRole::IndexerPositionBias,
                    "attn.indexer.compressor.ape",
                ),
                (
                    CompressRole::IndexerNorm,
                    "attn.indexer.compressor.norm.weight",
                ),
            ] {
                self.tensor(WeightRole::Compress(role), l, at(name))?;
            }
        }
        Ok(())
    }

    /// The three projections of every routed expert: `w1` gates, `w3` is the up projection and `w2`
    /// the down projection. Dense per-expert tensors stack into one `[experts, out, K]` weight;
    /// packed ones are one `[out, K]` weight per expert.
    fn experts(&mut self, layer: usize, experts: usize, prefix: &str) -> Result<(), ModelError> {
        for (proj, name) in [
            (ExpertProj::Gate, "w1"),
            (ExpertProj::Up, "w3"),
            (ExpertProj::Down, "w2"),
        ] {
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
                        Err(self.missing(stack, key))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.add(stack, WeightView::ExpertStack(keys))?;
        }
        Ok(())
    }
}
