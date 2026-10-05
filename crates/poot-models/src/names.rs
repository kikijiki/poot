//! Checkpoint-name tables: how a family's checkpoint tensor names map onto typed [`WeightId`]s.
//! A family lists [`NameRow`]s; [`build_weight_map`] turns them into the
//! model's [`WeightMap`] against the loaded store. [`HF_BASE`] holds the rows every HF decoder with
//! the `model.layers.{l}` layout shares; a family adds its own rows beside them.

use poot_quant::weights::{
    AttnRole, FfnRole, NormRole, WeightId, WeightKey, WeightMap, WeightMapError, WeightRole,
    WeightStore, WeightView,
};

use crate::model::{FamilyKey, ModelError};

pub(crate) mod gguf;

/// Where a role's tensor sits in the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A model-level tensor, by its checkpoint name.
    Tensor(&'static str),
    /// A model-level linear projection with stem `s`: stored dense as `{s}.weight`, or packed under
    /// the loader's linear id `{s}`.
    Linear(&'static str),
    /// A per-layer tensor named `{prefix}{layer}{suffix}`.
    LayerTensor {
        prefix: &'static str,
        suffix: &'static str,
    },
    /// A per-layer linear projection with stem `{prefix}{layer}{stem}`, dense or packed as
    /// [`Source::Linear`].
    LayerLinear {
        prefix: &'static str,
        stem: &'static str,
    },
    /// One part of a per-layer linear that fuses several roles' rows (phi3's `qkv_proj`, its
    /// `gate_up_proj`): the layer linear `{prefix}{layer}{stem}`, read as the rows of `part`.
    LayerFused {
        prefix: &'static str,
        stem: &'static str,
        part: FusedPart,
    },
}

/// Which rows of a fused projection a role reads, in the fused tensor's row order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedPart {
    /// `q | k | v` along the rows: the query, key or value rows.
    Q,
    K,
    V,
    /// `gate | up` along the rows.
    Gate,
    Up,
}

/// The row counts the fused parts of a config split into (see [`FusedPart`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedDims {
    /// Query rows: `heads * head_dim`.
    pub q: usize,
    /// Key (and value) rows: `kv_heads * head_dim`.
    pub kv: usize,
    /// Rows of the gate (and of the up) projection.
    pub inter: usize,
}

impl FusedDims {
    /// A family with no fused projection.
    pub const NONE: Self = Self {
        q: 0,
        kv: 0,
        inter: 0,
    };

    fn rows(self, part: FusedPart) -> std::ops::Range<usize> {
        match part {
            FusedPart::Q => 0..self.q,
            FusedPart::K => self.q..self.q + self.kv,
            FusedPart::V => self.q + self.kv..self.q + 2 * self.kv,
            FusedPart::Gate => 0..self.inter,
            FusedPart::Up => self.inter..2 * self.inter,
        }
    }
}

/// Whether a checkpoint must carry a row's tensor. An optional row that is absent is left out of
/// the map (a bias a family does not have); the family decides what an absent role means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presence {
    Required,
    Optional,
    /// When absent, the role reads the same stored tensor as the model-level role named here (a
    /// head tied to the embedding): its own map entry over the other row's entry, so each graph
    /// const keeps one reader and binding shares the stored bytes.
    TiedTo(WeightRole),
}

/// One role of a family's name table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NameRow {
    pub role: WeightRole,
    pub source: Source,
    pub presence: Presence,
}

impl NameRow {
    fn is_per_layer(&self) -> bool {
        matches!(
            self.source,
            Source::LayerTensor { .. } | Source::LayerLinear { .. } | Source::LayerFused { .. }
        )
    }

    /// The store keys this row can read at `layer`, in preference order.
    fn candidates(&self, layer: Option<usize>) -> Vec<String> {
        let l = layer.map(|l| l.to_string()).unwrap_or_default();
        match self.source {
            Source::Tensor(name) => vec![name.to_string()],
            Source::Linear(stem) => vec![format!("{stem}.weight"), stem.to_string()],
            Source::LayerTensor { prefix, suffix } => vec![format!("{prefix}{l}{suffix}")],
            Source::LayerLinear { prefix, stem } | Source::LayerFused { prefix, stem, .. } => {
                vec![
                    format!("{prefix}{l}{stem}.weight"),
                    format!("{prefix}{l}{stem}"),
                ]
            }
        }
    }
}

const LAYERS: &str = "model.layers.";

pub(crate) const fn layer_linear(role: WeightRole, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear {
            prefix: LAYERS,
            stem,
        },
        presence: Presence::Required,
    }
}

pub(crate) const fn layer_tensor(
    role: WeightRole,
    suffix: &'static str,
    presence: Presence,
) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor {
            prefix: LAYERS,
            suffix,
        },
        presence,
    }
}

/// The HF decoder rows qwen2/llama-style checkpoints share: embedding, output norm, the head (tied
/// to the embedding when the checkpoint has no `lm_head`), and per layer the attention and SwiGLU projections, their norms and the optional
/// QKV biases.
pub(crate) const HF_BASE: &[NameRow] = &[
    NameRow {
        role: WeightRole::Embed,
        source: Source::Tensor("model.embed_tokens.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNorm,
        source: Source::Tensor("model.norm.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("lm_head"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    layer_linear(WeightRole::Attn(AttnRole::Q), ".self_attn.q_proj"),
    layer_linear(WeightRole::Attn(AttnRole::K), ".self_attn.k_proj"),
    layer_linear(WeightRole::Attn(AttnRole::V), ".self_attn.v_proj"),
    layer_linear(WeightRole::Attn(AttnRole::O), ".self_attn.o_proj"),
    layer_tensor(
        WeightRole::Attn(AttnRole::QBias),
        ".self_attn.q_proj.bias",
        Presence::Optional,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::KBias),
        ".self_attn.k_proj.bias",
        Presence::Optional,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::VBias),
        ".self_attn.v_proj.bias",
        Presence::Optional,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Attn),
        ".input_layernorm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".post_attention_layernorm.weight",
        Presence::Required,
    ),
    layer_linear(WeightRole::Ffn(FfnRole::Gate), ".mlp.gate_proj"),
    layer_linear(WeightRole::Ffn(FfnRole::Up), ".mlp.up_proj"),
    layer_linear(WeightRole::Ffn(FfnRole::Down), ".mlp.down_proj"),
];

/// The HF sandwich norms (Gemma 2 and 3): `input_layernorm` before the attention block (the base
/// row), `post_attention_layernorm` after it, `pre_feedforward_layernorm` before the feed-forward
/// block and `post_feedforward_layernorm` after it.
pub(crate) const HF_SANDWICH_NORMS: &[NameRow] = &[
    layer_tensor(
        WeightRole::Norm(NormRole::PostAttn),
        ".post_attention_layernorm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".pre_feedforward_layernorm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::PostFfn),
        ".post_feedforward_layernorm.weight",
        Presence::Required,
    ),
];

const fn layer_fused(role: WeightRole, stem: &'static str, part: FusedPart) -> NameRow {
    NameRow {
        role,
        source: Source::LayerFused {
            prefix: LAYERS,
            stem,
            part,
        },
        presence: Presence::Required,
    }
}

/// HF rows beside [`HF_BASE`] for a checkpoint that fuses `q|k|v` into `self_attn.qkv_proj` and
/// `gate|up` into `mlp.gate_up_proj` (phi3): they replace the separate projections' rows.
pub(crate) const HF_FUSED_QKV_GATE_UP: &[NameRow] = &[
    layer_fused(
        WeightRole::Attn(AttnRole::Q),
        ".self_attn.qkv_proj",
        FusedPart::Q,
    ),
    layer_fused(
        WeightRole::Attn(AttnRole::K),
        ".self_attn.qkv_proj",
        FusedPart::K,
    ),
    layer_fused(
        WeightRole::Attn(AttnRole::V),
        ".self_attn.qkv_proj",
        FusedPart::V,
    ),
    layer_fused(
        WeightRole::Ffn(FfnRole::Gate),
        ".mlp.gate_up_proj",
        FusedPart::Gate,
    ),
    layer_fused(
        WeightRole::Ffn(FfnRole::Up),
        ".mlp.gate_up_proj",
        FusedPart::Up,
    ),
];

/// The rows of `tables` as one table: a later table's row for a role replaces an earlier table's
/// (a family's overrides beside [`HF_BASE`] or the GGUF base).
fn merged(tables: &[&[NameRow]]) -> Vec<NameRow> {
    let mut rows: Vec<NameRow> = Vec::new();
    for row in tables.iter().flat_map(|table| table.iter()) {
        match rows.iter_mut().find(|r| r.role == row.role) {
            Some(slot) => *slot = *row,
            None => rows.push(*row),
        }
    }
    rows
}

/// The per-layer Q/K norm scales (`self_attn.q_norm.weight`, `k_norm.weight`): qwen3 and OLMo 2
/// carry them, with different widths (the family's attention params say which).
pub(crate) const HF_QK_NORM: &[NameRow] = &[
    layer_tensor(
        WeightRole::Attn(AttnRole::QNorm),
        ".self_attn.q_norm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::KNorm),
        ".self_attn.k_norm.weight",
        Presence::Required,
    ),
];

/// The `[out, in]` (or `[n]`) shape of every [`HF_BASE`] tensor a checkpoint with `config`'s HF dims
/// carries (the QKV biases when `qkv_bias`), by its HF name: what a family's test fixture holds.
pub(crate) fn hf_base_shapes(
    config: &serde_json::Value,
    qkv_bias: bool,
) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (vocab, h, inter) = (n("vocab_size"), n("hidden_size"), n("intermediate_size"));
    let heads = n("num_attention_heads");
    let d = config["head_dim"]
        .as_u64()
        .map_or(h / heads, |d| d as usize);
    let (q, kv) = (heads * d, n("num_key_value_heads") * d);
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_string(), vec![vocab, h]),
        ("model.norm.weight".to_string(), vec![h]),
        ("lm_head.weight".to_string(), vec![vocab, h]),
    ];
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| format!("{LAYERS}{l}.{s}");
        shapes.extend([
            (k("input_layernorm.weight"), vec![h]),
            (k("post_attention_layernorm.weight"), vec![h]),
            (k("self_attn.q_proj.weight"), vec![q, h]),
            (k("self_attn.k_proj.weight"), vec![kv, h]),
            (k("self_attn.v_proj.weight"), vec![kv, h]),
            (k("self_attn.o_proj.weight"), vec![h, q]),
            (k("mlp.gate_proj.weight"), vec![inter, h]),
            (k("mlp.up_proj.weight"), vec![inter, h]),
            (k("mlp.down_proj.weight"), vec![h, inter]),
        ]);
        if qkv_bias {
            shapes.extend([
                (k("self_attn.q_proj.bias"), vec![q]),
                (k("self_attn.k_proj.bias"), vec![kv]),
                (k("self_attn.v_proj.bias"), vec![kv]),
            ]);
        }
    }
    shapes
}

/// A family's weight map over `store` (see [`build_weight_map`]), as the family's typed error. When
/// `head_required` (an HF `tie_word_embeddings: false` checkpoint), a head that fell back to the
/// embedding is a missing `lm_head.weight`, not a silently tied head.
pub(crate) fn family_weights(
    family: FamilyKey,
    store: &WeightStore,
    tables: &[&[NameRow]],
    layers: usize,
    head_required: bool,
) -> Result<WeightMap, ModelError> {
    family_weights_fused(
        family,
        store,
        tables,
        layers,
        head_required,
        FusedDims::NONE,
    )
}

/// [`family_weights`] for a family whose tables read fused projections split by `fused`.
pub(crate) fn family_weights_fused(
    family: FamilyKey,
    store: &WeightStore,
    tables: &[&[NameRow]],
    layers: usize,
    head_required: bool,
    fused: FusedDims,
) -> Result<WeightMap, ModelError> {
    let weights = build_map(store, tables, layers, fused)
        .map_err(|source| ModelError::Weight { family, source })?;
    let head = WeightId::model(WeightRole::Head);
    if head_required && weights.view(head) == weights.view(WeightId::model(WeightRole::Embed)) {
        return Err(ModelError::Weight {
            family,
            source: WeightMapError::Missing {
                id: head,
                key: WeightKey::from("lm_head.weight"),
            },
        });
    }
    Ok(weights)
}

/// The weight map of a `layers`-layer model whose checkpoint `tables` describe (later tables
/// override earlier ones by role): every model-level row once and every per-layer row at each layer,
/// each read as one stored entry. A required row whose tensor is absent is
/// [`WeightMapError::Missing`], naming its preferred key; a tied row that is absent reads its
/// model-level role's entry.
#[cfg(test)]
pub(crate) fn build_weight_map(
    store: &WeightStore,
    tables: &[&[NameRow]],
    layers: usize,
) -> Result<WeightMap, WeightMapError> {
    build_map(store, tables, layers, FusedDims::NONE)
}

fn build_map(
    store: &WeightStore,
    tables: &[&[NameRow]],
    layers: usize,
    fused: FusedDims,
) -> Result<WeightMap, WeightMapError> {
    let rows = &merged(tables);
    let mut map = WeightMap::builder(store);
    for row in rows {
        let ids: Vec<WeightId> = if row.is_per_layer() {
            (0..layers).map(|l| WeightId::layer(l, row.role)).collect()
        } else {
            vec![WeightId::model(row.role)]
        };
        for id in ids {
            let found = |row: &NameRow, layer| {
                let candidates = row.candidates(layer);
                match candidates.iter().find(|key| store.contains(key)) {
                    Some(key) => Ok(WeightKey::from(key.as_str())),
                    None => Err(WeightKey::from(candidates[0].as_str())),
                }
            };
            let key = match (found(row, id.layer), row.presence) {
                (Ok(key), _) => key,
                (Err(_), Presence::Optional) => continue,
                (Err(key), Presence::Required) => return Err(WeightMapError::Missing { id, key }),
                (Err(key), Presence::TiedTo(role)) => {
                    match rows.iter().find(|r| r.role == role && !r.is_per_layer()) {
                        Some(tied) => {
                            found(tied, None).map_err(|key| WeightMapError::Missing { id, key })?
                        }
                        None => return Err(WeightMapError::Missing { id, key }),
                    }
                }
            };
            let view = match row.source {
                Source::LayerFused { part, .. } => WeightView::RowRange {
                    key,
                    rows: fused.rows(part),
                },
                _ => WeightView::Stored(key),
            };
            map.map(id, view)?;
        }
    }
    Ok(map.build())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{DenseWeight, HandleFormat, WeightEntry};
    use poot_quant::{PackedPayload, PackedWeight, SourceRole};
    use poot_tensor::DType;

    use super::*;

    fn dense(shape: &[usize]) -> WeightEntry {
        let n: usize = shape.iter().product();
        WeightEntry::Dense(
            DenseWeight::try_new(DType::BF16, shape.to_vec(), vec![0u8; n * 2].into()).unwrap(),
        )
    }

    fn packed(out: usize) -> WeightEntry {
        let weight = PackedWeight::try_new(WeightFormat::Q8_0, [out, 32]).unwrap();
        let blocks = vec![0u8; weight.source_bytes(SourceRole::Blocks)];
        WeightEntry::Packed(Arc::new(
            PackedPayload::try_new(weight, [(SourceRole::Blocks, blocks.into())]).unwrap(),
        ))
    }

    /// A two-layer qwen2-shaped checkpoint: biases present, head tied (no `lm_head`), and layer 1's
    /// `down_proj` packed (stored under its linear id, not `.weight`).
    fn qwen2_store(skip: Option<&str>) -> WeightStore {
        let mut names: Vec<(String, WeightEntry)> = vec![
            ("model.embed_tokens.weight".into(), dense(&[8, 32])),
            ("model.norm.weight".into(), dense(&[32])),
        ];
        for l in 0..2 {
            let p = format!("model.layers.{l}");
            for proj in ["q", "k", "v", "o"] {
                names.push((
                    format!("{p}.self_attn.{proj}_proj.weight"),
                    dense(&[32, 32]),
                ));
            }
            for proj in ["q", "k", "v"] {
                names.push((format!("{p}.self_attn.{proj}_proj.bias"), dense(&[32])));
            }
            names.push((format!("{p}.input_layernorm.weight"), dense(&[32])));
            names.push((format!("{p}.post_attention_layernorm.weight"), dense(&[32])));
            names.push((format!("{p}.mlp.gate_proj.weight"), dense(&[64, 32])));
            names.push((format!("{p}.mlp.up_proj.weight"), dense(&[64, 32])));
            match l {
                0 => names.push((format!("{p}.mlp.down_proj.weight"), dense(&[32, 64]))),
                _ => names.push((format!("{p}.mlp.down_proj"), packed(32))),
            }
        }
        let mut store = WeightStore::builder();
        for (name, entry) in names {
            if Some(name.as_str()) != skip {
                store.insert(name, entry).unwrap();
            }
        }
        store.build()
    }

    #[test]
    fn hf_base_rows_map_every_layer_dense_or_packed() {
        let store = qwen2_store(None);
        let map = build_weight_map(&store, &[HF_BASE], 2).unwrap();
        // 3 model-level roles + 12 per layer (4 projections, 3 biases, 2 norms, 3 FFN). The head is
        // tied: its own entry over the embedding's stored tensor.
        assert_eq!(map.len(), 3 + 2 * 12);
        assert_eq!(
            map.view(WeightId::model(WeightRole::Head)),
            Some(&WeightView::Stored("model.embed_tokens.weight".into()))
        );

        let q1 = WeightId::layer(1, WeightRole::Attn(AttnRole::Q));
        assert_eq!(
            map.view(q1),
            Some(&WeightView::Stored(
                "model.layers.1.self_attn.q_proj.weight".into()
            ))
        );
        assert_eq!(
            map.handle(q1).unwrap().format,
            HandleFormat::Dense(DType::BF16)
        );

        let down1 = WeightId::layer(1, WeightRole::Ffn(FfnRole::Down));
        assert_eq!(
            map.view(down1),
            Some(&WeightView::Stored("model.layers.1.mlp.down_proj".into()))
        );
        assert!(matches!(
            map.handle(down1).unwrap().format,
            HandleFormat::Packed(_)
        ));
        let bias0 = WeightId::layer(0, WeightRole::Attn(AttnRole::VBias));
        assert_eq!(map.handle(bias0).unwrap().shape, vec![32]);
    }

    #[test]
    fn an_absent_optional_row_is_left_out_and_an_absent_required_row_is_named() {
        let store = qwen2_store(Some("model.layers.0.self_attn.k_proj.bias"));
        let map = build_weight_map(&store, &[HF_BASE], 2).unwrap();
        assert!(
            map.handle(WeightId::layer(0, WeightRole::Attn(AttnRole::KBias)))
                .is_none()
        );
        assert!(
            map.handle(WeightId::layer(1, WeightRole::Attn(AttnRole::KBias)))
                .is_some()
        );

        let store = qwen2_store(Some("model.layers.1.post_attention_layernorm.weight"));
        assert_eq!(
            build_weight_map(&store, &[HF_BASE], 2),
            Err(WeightMapError::Missing {
                id: WeightId::layer(1, WeightRole::Norm(NormRole::Ffn)),
                key: "model.layers.1.post_attention_layernorm.weight".into(),
            })
        );
        let store = qwen2_store(Some("model.layers.0.mlp.up_proj.weight"));
        assert_eq!(
            build_weight_map(&store, &[HF_BASE], 2),
            Err(WeightMapError::Missing {
                id: WeightId::layer(0, WeightRole::Ffn(FfnRole::Up)),
                key: "model.layers.0.mlp.up_proj.weight".into(),
            })
        );
    }

    /// A one-layer phi3-shaped checkpoint (width 32, 16 query rows, 8 key and 8 value rows,
    /// inter 12): fused `qkv_proj` and `gate_up_proj`, the second one packed.
    fn fused_store(skip: Option<&str>) -> WeightStore {
        let names: Vec<(&str, WeightEntry)> = vec![
            ("model.embed_tokens.weight", dense(&[8, 32])),
            ("model.norm.weight", dense(&[32])),
            ("model.layers.0.input_layernorm.weight", dense(&[32])),
            (
                "model.layers.0.post_attention_layernorm.weight",
                dense(&[32]),
            ),
            ("model.layers.0.self_attn.qkv_proj.weight", dense(&[32, 32])),
            ("model.layers.0.self_attn.o_proj.weight", dense(&[32, 16])),
            ("model.layers.0.mlp.gate_up_proj", packed(24)),
            ("model.layers.0.mlp.down_proj.weight", dense(&[32, 12])),
        ];
        let mut store = WeightStore::builder();
        for (name, entry) in names {
            if Some(name) != skip {
                store.insert(name, entry).unwrap();
            }
        }
        store.build()
    }

    /// Fused rows read row ranges of the one stored tensor, in `q | k | v` and `gate | up` order,
    /// dense or packed; a missing fused tensor is named.
    #[test]
    fn fused_rows_read_row_ranges_of_the_fused_tensor() {
        let dims = FusedDims {
            q: 16,
            kv: 8,
            inter: 12,
        };
        let tables: &[&[NameRow]] = &[HF_BASE, HF_FUSED_QKV_GATE_UP];
        let map = build_map(&fused_store(None), tables, 1, dims).unwrap();
        let view = |role| map.view(WeightId::layer(0, role)).cloned();
        let qkv = || WeightKey::from("model.layers.0.self_attn.qkv_proj.weight");
        for (role, rows) in [
            (AttnRole::Q, 0..16),
            (AttnRole::K, 16..24),
            (AttnRole::V, 24..32),
        ] {
            assert_eq!(
                view(WeightRole::Attn(role)),
                Some(WeightView::RowRange { key: qkv(), rows })
            );
        }
        let gate_up = || WeightKey::from("model.layers.0.mlp.gate_up_proj");
        assert_eq!(
            view(WeightRole::Ffn(FfnRole::Up)),
            Some(WeightView::RowRange {
                key: gate_up(),
                rows: 12..24
            })
        );
        let k = map
            .handle(WeightId::layer(0, WeightRole::Attn(AttnRole::K)))
            .unwrap();
        assert_eq!(k.shape, vec![8, 32]);
        let gate = map
            .handle(WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate)))
            .unwrap();
        assert_eq!(gate.shape, vec![12, 32]);
        assert!(matches!(gate.format, HandleFormat::Packed(_)));

        assert_eq!(
            build_map(
                &fused_store(Some("model.layers.0.self_attn.qkv_proj.weight")),
                tables,
                1,
                dims
            ),
            Err(WeightMapError::Missing {
                id: WeightId::layer(0, WeightRole::Attn(AttnRole::Q)),
                key: "model.layers.0.self_attn.qkv_proj.weight".into(),
            })
        );
    }
}
