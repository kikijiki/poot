//! How a checkpoint spells a Gated DeltaNet layer, once for every family that has one (`qwen35`,
//! `qwen3next`, `qwen4exp`): the tensor names, the weight conventions those names carry, the weight-map
//! entries a GDN layer needs, the stack depth of a GGUF that appends multi-token-prediction blocks, and
//! (under test) the converter fixture that writes the llama.cpp spelling of an HF layer.
//!
//! The conventions belong to the checkpoint container, not to the architecture, so a [`GdnTable`] fixes
//! them together with the names and a family only picks the table: HF safetensors store the norms as
//! `(1 + w)`, the value heads grouped by key head and `A_log`; llama.cpp's converter folds the one in,
//! reorders every per-value-head tensor to tiled order (`_reorder_v_heads`) and stores `-exp(A_log)`.

use poot_quant::weights::{
    GdnRole, WeightKey, WeightMapBuilder, WeightMapError, WeightRole, WeightView,
};

use super::dims::{Field, Fields};
use super::gdn::{GdnHeadOrder, GdnParams};
use super::norm::{NormKind, NormParams};
use super::standard::ParamError;
use crate::model::{ConfigReason, ModelError};
use crate::registry::RawConfig;
use poot_quant::weights::WeightId;

/// How a checkpoint stores the `+ 1` of a zero-centered RMSNorm scale. The GDN's own gated norm is a
/// plain scale in every container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NormFold {
    /// The scale is stored as an offset from one (HF); the norm adds the one.
    PlusOne,
    /// The one is already folded into the stored scale (GGUF).
    Folded,
}

impl NormFold {
    /// The residual-stream and projection norm this storage needs.
    pub(crate) fn norm_kind(self) -> NormKind {
        match self {
            Self::PlusOne => NormKind::RmsPlusOne,
            Self::Folded => NormKind::Rms,
        }
    }
}

/// How a checkpoint stores the GDN decay `A`, which is negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecaySign {
    /// `A_log` is stored and the decay is `-exp(A_log)` (HF).
    NegExp,
    /// The decay itself is stored, already negative (GGUF `ssm_a`).
    Stored,
}

/// The stored conventions a container applies to a GDN layer and the norms around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GdnWeightConvention {
    pub(crate) norm_fold: NormFold,
    pub(crate) head_order: GdnHeadOrder,
    pub(crate) decay_sign: DecaySign,
}

/// The input projections: one per quantity, or the two fused ones of HF `qwen3_next`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnProjections {
    Fused {
        qkvz: &'static str,
        ba: &'static str,
    },
    Split {
        qkv: &'static str,
        z: &'static str,
        beta: &'static str,
        alpha: &'static str,
    },
}

/// A container's GDN tensor names (layer suffixes; a projection is a stem) and weight conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GdnTable {
    pub(crate) projections: GdnProjections,
    out: &'static str,
    dt_bias: &'static str,
    conv: &'static str,
    a: &'static str,
    norm: &'static str,
    pub(crate) convention: GdnWeightConvention,
}

impl GdnTable {
    /// HF `qwen3_next`: fused `in_proj_qkvz` and `in_proj_ba`.
    pub(crate) const HF_FUSED: Self = Self::hf(GdnProjections::Fused {
        qkvz: ".linear_attn.in_proj_qkvz",
        ba: ".linear_attn.in_proj_ba",
    });

    /// HF `qwen3_5` and `qwen4_exp`: one projection per quantity.
    pub(crate) const HF_SPLIT: Self = Self::hf(GdnProjections::Split {
        qkv: ".linear_attn.in_proj_qkv",
        z: ".linear_attn.in_proj_z",
        beta: ".linear_attn.in_proj_b",
        alpha: ".linear_attn.in_proj_a",
    });

    /// llama.cpp's `qwen35`, `qwen35moe` and `qwen4exp`.
    pub(crate) const GGUF: Self = Self {
        projections: GdnProjections::Split {
            qkv: ".attn_qkv",
            z: ".attn_gate",
            beta: ".ssm_beta",
            alpha: ".ssm_alpha",
        },
        out: ".ssm_out",
        dt_bias: ".ssm_dt.bias",
        conv: ".ssm_conv1d.weight",
        a: ".ssm_a",
        norm: ".ssm_norm.weight",
        convention: GdnWeightConvention {
            norm_fold: NormFold::Folded,
            head_order: GdnHeadOrder::Tiled,
            decay_sign: DecaySign::Stored,
        },
    };

    const fn hf(projections: GdnProjections) -> Self {
        Self {
            projections,
            out: ".linear_attn.out_proj",
            dt_bias: ".linear_attn.dt_bias",
            conv: ".linear_attn.conv1d.weight",
            a: ".linear_attn.A_log",
            norm: ".linear_attn.norm.weight",
            convention: GdnWeightConvention {
                norm_fold: NormFold::PlusOne,
                head_order: GdnHeadOrder::Grouped,
                decay_sign: DecaySign::NegExp,
            },
        }
    }

    /// The traced layer's parameters under this table's conventions.
    pub(crate) fn params(
        &self,
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
        kernel: usize,
        norm: NormParams,
    ) -> Result<GdnParams, ParamError> {
        let mut params = GdnParams::new(
            key_heads,
            value_heads,
            head_dim,
            kernel,
            norm,
            self.convention.head_order,
        )?;
        if matches!(self.projections, GdnProjections::Fused { .. }) {
            params = params.with_fused_projections();
        }
        if self.convention.decay_sign == DecaySign::NegExp {
            params = params.with_log_decay();
        }
        Ok(params)
    }

    /// Maps layer `layer`'s GDN weights. `key` joins a layer suffix to the layer's tensor name;
    /// `linear` is the stored key of a projection given its full stem. A tensor the store lacks is
    /// [`WeightMapError::Missing`] naming its key.
    pub(crate) fn map_layer(
        &self,
        map: &mut WeightMapBuilder<'_>,
        layer: usize,
        key: &dyn Fn(&str) -> String,
        linear: &dyn Fn(&str) -> String,
    ) -> Result<(), WeightMapError> {
        let mut put = |role: GdnRole, stored: String| {
            map.map(
                WeightId::layer(layer, WeightRole::Gdn(role)),
                WeightView::Stored(WeightKey::from(stored)),
            )
            .map(|_| ())
        };
        match self.projections {
            GdnProjections::Fused { qkvz, ba } => {
                put(GdnRole::Qkvz, linear(&key(qkvz)))?;
                put(GdnRole::Ba, linear(&key(ba)))?;
            }
            GdnProjections::Split {
                qkv,
                z,
                beta,
                alpha,
            } => {
                for (role, stem) in [
                    (GdnRole::Qkv, qkv),
                    (GdnRole::Z, z),
                    (GdnRole::Beta, beta),
                    (GdnRole::Alpha, alpha),
                ] {
                    put(role, linear(&key(stem)))?;
                }
            }
        }
        put(GdnRole::Out, linear(&key(self.out)))?;
        for (role, suffix) in [
            (GdnRole::DtBias, self.dt_bias),
            (GdnRole::Conv, self.conv),
            (GdnRole::A, self.a),
            (GdnRole::GateNorm, self.norm),
        ] {
            put(role, key(suffix))?;
        }
        Ok(())
    }
}

/// The GGUF key of the multi-token-prediction block count for architecture `$arch`.
macro_rules! nextn_field {
    ($arch:literal) => {
        $crate::components::dims::Field::new("", concat!($arch, ".nextn_predict_layers"))
    };
}
pub(crate) use nextn_field;

/// The depth of the traced stack. A GGUF appends its multi-token-prediction blocks after the last
/// block: they count in `block_count` and have their own tensors, and the stack does not run them
/// (`nextn` names how many). An HF config has none. A count that leaves no block is refused naming
/// the key.
pub(crate) fn trunk_layers(
    fields: &Fields<'_>,
    nextn: Field,
    layers: usize,
) -> Result<usize, ModelError> {
    match fields.raw() {
        RawConfig::HfJson { .. } => Ok(layers),
        RawConfig::Gguf(_) => {
            let nextn_layers = fields.opt_u64(nextn)?.unwrap_or(0);
            usize::try_from(nextn_layers)
                .ok()
                .and_then(|nextn_layers| layers.checked_sub(nextn_layers))
                .filter(|&trunk| trunk > 0)
                .ok_or(fields.error(fields.key(nextn), ConfigReason::Exceeds { max: layers }))
        }
    }
}

/// The converter fixture: what llama.cpp's converter writes for a GDN layer of an HF fixture, from
/// the converter's rules (`modify_tensors` and `_reorder_v_heads` in `conversion/qwen.py`), not from
/// a family's reader.
#[cfg(test)]
pub(crate) mod converter {
    use poot_load::gguf::GgufValue;

    /// A GDN layer's dimensions as the fixture's config states them.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct LayerDims {
        pub(crate) width: usize,
        pub(crate) key_heads: usize,
        pub(crate) value_heads: usize,
        pub(crate) head_dim: usize,
        pub(crate) kernel: usize,
    }

    /// How the HF fixture stores the projections.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum HfProjections {
        /// `in_proj_qkvz` and `in_proj_ba`, rows grouped per key head.
        Fused,
        /// `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`.
        Split,
    }

    impl LayerDims {
        fn rep(&self) -> usize {
            self.value_heads / self.key_heads
        }

        fn key_dim(&self) -> usize {
            self.key_heads * self.head_dim
        }

        fn value_dim(&self) -> usize {
            self.value_heads * self.head_dim
        }

        /// The tiled position `t` of a value head holds grouped head `(t % hk) * rep + t / hk`.
        fn grouped_of(&self, t: usize) -> usize {
            (t % self.key_heads) * self.rep() + t / self.key_heads
        }

        /// Rows of `data` (`width` values each) in runs of `block` rows, one run per value head,
        /// the runs reordered from grouped to tiled.
        pub(crate) fn reorder_blocks(&self, data: &[f32], width: usize, block: usize) -> Vec<f32> {
            (0..self.value_heads)
                .flat_map(|t| {
                    let g = self.grouped_of(t);
                    data[g * block * width..(g + 1) * block * width].to_vec()
                })
                .collect()
        }
    }

    /// The norms' `+ 1` folded into the stored scale.
    pub(crate) fn plus_one(v: Vec<f32>) -> Vec<f32> {
        v.into_iter().map(|x| x + 1.0).collect()
    }

    /// Writes the GGUF tensors of one GDN layer. `hf(suffix)` reads the HF fixture's tensor
    /// `linear_attn.{suffix}`; `gguf(suffix)` names the GGUF tensor `blk.{l}.{suffix}`; `put` takes
    /// the name, shape and values.
    pub(crate) fn write_layer(
        d: LayerDims,
        hf_projections: HfProjections,
        hf: &dyn Fn(&str) -> Vec<f32>,
        gguf: &dyn Fn(&str) -> String,
        put: &mut dyn FnMut(String, Vec<usize>, Vec<f32>),
    ) {
        let LayerDims {
            width: w,
            key_heads: hk,
            value_heads: hv,
            head_dim: hd,
            kernel,
        } = d;
        let (rep, key_dim, value_dim) = (d.rep(), d.key_dim(), d.value_dim());
        let conv = 2 * key_dim + value_dim;
        let (qkv, z, beta, alpha) = match hf_projections {
            HfProjections::Split => {
                // `in_proj_qkv` rows are `[q | k | v]`; only the value rows follow the value heads.
                let qkv = hf("in_proj_qkv.weight");
                let split = 2 * key_dim * w;
                (
                    [
                        qkv[..split].to_vec(),
                        d.reorder_blocks(&qkv[split..], w, hd),
                    ]
                    .concat(),
                    d.reorder_blocks(&hf("in_proj_z.weight"), w, hd),
                    d.reorder_blocks(&hf("in_proj_b.weight"), w, 1),
                    d.reorder_blocks(&hf("in_proj_a.weight"), w, 1),
                )
            }
            HfProjections::Fused => {
                // `in_proj_qkvz` rows group per key head:
                // `[q (hd) | k (hd) | v (rep * hd) | z (rep * hd)]`.
                let qkvz = hf("in_proj_qkvz.weight");
                let block = 2 * hd + 2 * rep * hd;
                let rows = |g: usize, from: usize, len: usize| {
                    qkvz[(g * block + from) * w..(g * block + from + len) * w].to_vec()
                };
                let q: Vec<f32> = (0..hk).flat_map(|g| rows(g, 0, hd)).collect();
                let k: Vec<f32> = (0..hk).flat_map(|g| rows(g, hd, hd)).collect();
                let v: Vec<f32> = (0..hk).flat_map(|g| rows(g, 2 * hd, rep * hd)).collect();
                let z: Vec<f32> = (0..hk)
                    .flat_map(|g| rows(g, 2 * hd + rep * hd, rep * hd))
                    .collect();
                // `in_proj_ba` rows group per key head: `[b (rep) | a (rep)]`.
                let ba = hf("in_proj_ba.weight");
                let part = |from: usize| -> Vec<f32> {
                    (0..hk)
                        .flat_map(|g| {
                            ba[(g * 2 * rep + from) * w..(g * 2 * rep + from + rep) * w].to_vec()
                        })
                        .collect()
                };
                (
                    [q, k, d.reorder_blocks(&v, w, hd)].concat(),
                    d.reorder_blocks(&z, w, hd),
                    d.reorder_blocks(&part(0), w, 1),
                    d.reorder_blocks(&part(rep), w, 1),
                )
            }
        };
        put(gguf("attn_qkv.weight"), vec![conv, w], qkv);
        put(gguf("attn_gate.weight"), vec![value_dim, w], z);
        put(gguf("ssm_beta.weight"), vec![hv, w], beta);
        put(gguf("ssm_alpha.weight"), vec![hv, w], alpha);
        put(
            gguf("ssm_dt.bias"),
            vec![hv],
            d.reorder_blocks(&hf("dt_bias"), 1, 1),
        );
        let a: Vec<f32> = hf("A_log").iter().map(|a| -a.exp()).collect();
        put(gguf("ssm_a"), vec![hv], d.reorder_blocks(&a, 1, 1));
        // The conv kernel `[conv, 1, K]` as `[conv, K]`, its value rows reordered.
        let kernel_rows = hf("conv1d.weight");
        let split = 2 * key_dim * kernel;
        put(
            gguf("ssm_conv1d.weight"),
            vec![conv, kernel],
            [
                kernel_rows[..split].to_vec(),
                d.reorder_blocks(&kernel_rows[split..], kernel, hd),
            ]
            .concat(),
        );
        put(gguf("ssm_norm.weight"), vec![hd], hf("norm.weight"));
        // `out_proj` `[w, value_dim]`: its columns follow the value heads.
        let out = hf("out_proj.weight");
        let mut reordered: Vec<f32> = Vec::with_capacity(out.len());
        for row in 0..w {
            for t in 0..hv {
                let g = d.grouped_of(t);
                reordered.extend(&out[row * value_dim + g * hd..row * value_dim + (g + 1) * hd]);
            }
        }
        put(gguf("ssm_out.weight"), vec![w, value_dim], reordered);
    }

    /// The multi-token-prediction block a GGUF appends after block `layer`: an attention layer's
    /// norms plus its own projection and norms. None are read.
    pub(crate) fn write_nextn_block(
        width: usize,
        layer: usize,
        put: &mut dyn FnMut(String, Vec<usize>, Vec<f32>),
    ) {
        let gg = |s: &str| format!("blk.{layer}.{s}");
        let ones = |len: usize| vec![1.0; len];
        put(gg("attn_norm.weight"), vec![width], ones(width));
        put(gg("post_attention_norm.weight"), vec![width], ones(width));
        put(
            gg("nextn.eh_proj.weight"),
            vec![width, 2 * width],
            ones(2 * width * width),
        );
        put(gg("nextn.enorm.weight"), vec![width], ones(width));
        put(gg("nextn.hnorm.weight"), vec![width], ones(width));
        put(
            gg("nextn.shared_head_norm.weight"),
            vec![width],
            ones(width),
        );
    }

    /// The GGUF key naming how many multi-token-prediction blocks architecture `arch` appends.
    pub(crate) fn nextn_key(arch: &str) -> String {
        format!("{arch}.nextn_predict_layers")
    }

    /// The GGUF metadata every GDN architecture `arch` states about its layer, and, when `nextn`,
    /// the one multi-token-prediction block the file appends.
    pub(crate) fn layer_metadata(
        arch: &str,
        d: LayerDims,
        nextn: bool,
    ) -> Vec<(String, GgufValue)> {
        let u = |v: usize| GgufValue::U32(v as u32);
        let mut kvs = vec![
            (format!("{arch}.ssm.inner_size"), u(d.value_dim())),
            (format!("{arch}.ssm.state_size"), u(d.head_dim)),
            (format!("{arch}.ssm.group_count"), u(d.key_heads)),
            (format!("{arch}.ssm.time_step_rank"), u(d.value_heads)),
            (format!("{arch}.ssm.conv_kernel"), u(d.kernel)),
            (format!("{arch}.full_attention_interval"), u(4)),
        ];
        if nextn {
            kvs.push((nextn_key(arch), u(1)));
        }
        kvs
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    use poot_quant::weights::{WeightMap, WeightStore};

    use super::*;
    use crate::components::testing::f32_entry;

    /// The stored keys of one GDN layer (layer 0) under `prefix`, for a table whose projections are
    /// `projections`, in the spelling each container's files use (written out here, not derived from
    /// the tables).
    fn layer_keys(prefix: &str, projections: &[&str], rest: &[&str]) -> BTreeSet<String> {
        projections
            .iter()
            .chain(rest)
            .map(|suffix| format!("{prefix}0.{suffix}"))
            .collect()
    }

    /// Maps layer 0 of `table` over a store holding exactly `keys`, reading a projection as
    /// `{stem}.weight`; the keys the map reads, or the error.
    fn mapped(
        table: &GdnTable,
        prefix: &str,
        keys: &BTreeSet<String>,
    ) -> Result<BTreeSet<String>, WeightMapError> {
        let mut store = WeightStore::builder();
        for key in keys {
            store
                .insert(key.as_str(), f32_entry(vec![2, 2], &[0.0; 4]))
                .unwrap();
        }
        let store = store.build();
        let mut map = WeightMap::builder(&store);
        table.map_layer(
            &mut map,
            0,
            &|suffix| format!("{prefix}0{suffix}"),
            &|stem| format!("{stem}.weight"),
        )?;
        Ok(map
            .build()
            .report(&store)
            .loaded
            .iter()
            .map(|key| key.as_str().to_string())
            .collect())
    }

    /// The HF safetensors names of `qwen3_next`, of `qwen3_5` and `qwen4_exp` (the same split
    /// spelling, under two prefixes) and the GGUF names all three architectures share each resolve to
    /// exactly the tensors the container stores for a GDN layer. Mutation: rename `.ssm_beta` in
    /// [`GdnTable::GGUF`]; the layer maps `Missing` naming the tensor.
    #[test]
    fn the_tables_resolve_the_tensors_each_family_stores() {
        let hf_rest = [
            "linear_attn.out_proj.weight",
            "linear_attn.dt_bias",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.norm.weight",
        ];
        let hf_split = [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight",
        ];
        let gguf_rest = [
            "ssm_out.weight",
            "ssm_dt.bias",
            "ssm_conv1d.weight",
            "ssm_a",
            "ssm_norm.weight",
        ];
        let gguf_projections = [
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
        ];
        let cases: [(&str, &GdnTable, &str, BTreeSet<String>); 4] = [
            (
                "qwen3next HF",
                &GdnTable::HF_FUSED,
                "model.layers.",
                layer_keys(
                    "model.layers.",
                    &[
                        "linear_attn.in_proj_qkvz.weight",
                        "linear_attn.in_proj_ba.weight",
                    ],
                    &hf_rest,
                ),
            ),
            (
                "qwen35 HF",
                &GdnTable::HF_SPLIT,
                "model.language_model.layers.",
                layer_keys("model.language_model.layers.", &hf_split, &hf_rest),
            ),
            (
                "qwen35 text-only HF",
                &GdnTable::HF_SPLIT,
                "model.layers.",
                layer_keys("model.layers.", &hf_split, &hf_rest),
            ),
            (
                "qwen35, qwen3next and qwen4exp GGUF",
                &GdnTable::GGUF,
                "blk.",
                layer_keys("blk.", &gguf_projections, &gguf_rest),
            ),
        ];
        for (name, table, prefix, keys) in cases {
            assert_eq!(mapped(table, prefix, &keys).unwrap(), keys, "{name}");
            let dropped = keys
                .iter()
                .find(|key| key.contains("ssm_beta") || key.contains("in_proj_b."));
            let Some(dropped) = dropped.or_else(|| keys.iter().next()) else {
                unreachable!()
            };
            let mut short = keys.clone();
            short.remove(dropped);
            let err = mapped(table, prefix, &short).unwrap_err();
            assert!(
                matches!(&err, WeightMapError::Missing { key, .. } if key.as_str() == dropped),
                "{name}: {err}"
            );
        }
    }

    /// The stored conventions of each container, as llama.cpp's converter and HF's modeling files
    /// state them: HF keeps `(1 + w)` norms, grouped value heads and `A_log`; a GGUF folds the one in,
    /// tiles the value heads and stores the negative decay. Mutation: flip `decay_sign` of
    /// [`GdnTable::GGUF`] to [`DecaySign::NegExp`]; this row and the real-file GDN row go red.
    #[test]
    fn the_containers_carry_their_own_conventions() {
        for table in [GdnTable::HF_FUSED, GdnTable::HF_SPLIT] {
            assert_eq!(
                table.convention,
                GdnWeightConvention {
                    norm_fold: NormFold::PlusOne,
                    head_order: GdnHeadOrder::Grouped,
                    decay_sign: DecaySign::NegExp,
                }
            );
        }
        assert_eq!(
            GdnTable::GGUF.convention,
            GdnWeightConvention {
                norm_fold: NormFold::Folded,
                head_order: GdnHeadOrder::Tiled,
                decay_sign: DecaySign::Stored,
            }
        );
        // The traced parameters follow the convention: only the fused HF table fuses projections.
        let norm = NormParams::new(1e-6).unwrap();
        let params = |table: &GdnTable| table.params(2, 4, 8, 4, norm).unwrap();
        let mut hf = GdnParams::new(2, 4, 8, 4, norm, GdnHeadOrder::Grouped).unwrap();
        hf = hf.with_log_decay();
        assert_eq!(params(&GdnTable::HF_SPLIT), hf);
        assert_eq!(params(&GdnTable::HF_FUSED), hf.with_fused_projections());
        assert_eq!(
            params(&GdnTable::GGUF),
            GdnParams::new(2, 4, 8, 4, norm, GdnHeadOrder::Tiled).unwrap()
        );
    }

    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    /// SC-004: no architecture owns a GDN name-stem table, a grouped-to-tiled reorder helper or the
    /// multi-token-prediction rule; they live here. A needle is a spelling that only such a copy
    /// carries (a stem written as a suffix, the GGUF key, the helper's name). Mutation: add a local
    /// `".ssm_beta"` stem table to `architectures/qwen3next.rs`; this test names the file and needle.
    #[test]
    fn no_architecture_keeps_its_own_gdn_names_reorder_or_nextn_rule() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/architectures");
        let needles = [
            "\".linear_attn.",
            "\".attn_qkv\"",
            "\".attn_gate\"",
            "\".ssm_",
            ".nextn_predict_layers\"",
            "fn reorder_blocks",
            "grouped_of",
        ];
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        assert!(files.len() > 20, "the scan found {} files", files.len());
        let found: Vec<String> = files
            .iter()
            .flat_map(|path| {
                let text = std::fs::read_to_string(path).unwrap();
                needles
                    .iter()
                    .filter(|needle| text.contains(**needle))
                    .map(|needle| {
                        format!(
                            "{} spells {needle}",
                            path.strip_prefix(&root).unwrap().display()
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(found.is_empty(), "{found:#?}");
    }
}
