//! Checkpoint-name tables: how a family's checkpoint tensor names map onto typed [`WeightId`]s.
//! A family lists [`NameRow`]s; [`build_weight_map`] turns them into the
//! model's [`WeightMap`] against the loaded store. [`HF_BASE`] holds the rows every HF decoder with
//! the `model.layers.{l}` layout shares; a family adds its own rows beside them.

use poot_quant::weights::{
    AttnRole, ExpertProj, FfnRole, MoeRole, NormRole, WeightId, WeightKey, WeightMap,
    WeightMapBuilder, WeightMapError, WeightRole, WeightStore, WeightView,
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
        /// The text before the layer index.
        prefix: &'static str,
        /// The text after the layer index.
        suffix: &'static str,
    },
    /// A per-layer linear projection with stem `{prefix}{layer}{stem}`, dense or packed as
    /// [`Source::Linear`].
    LayerLinear {
        /// The text before the layer index.
        prefix: &'static str,
        /// The text after the layer index; the projection's stem.
        stem: &'static str,
    },
    /// One part of a per-layer linear that fuses several roles' rows (phi3's `qkv_proj`, its
    /// `gate_up_proj`): the layer linear `{prefix}{layer}{stem}`, read as the rows of `part`.
    LayerFused {
        /// The text before the layer index.
        prefix: &'static str,
        /// The text after the layer index; the fused projection's stem.
        stem: &'static str,
        /// Which rows of the fused projection this role reads.
        part: FusedPart,
    },
    /// The tensor `inner` names, its rows stored per head as adjacent pairs `(j, j + head_dim / 2)` and read
    /// back in the half-split order (all first halves, then all second halves): llama.cpp's Q/K layout for
    /// its interleaved RoPE, which the model traces as the half-split rotation. Needs
    /// [`FusedDims::head_dim`].
    PairedHeads(&'static Source),
}

/// Which rows of a fused projection a role reads, in the fused tensor's row order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedPart {
    /// `q | k | v` along the rows: the query, key or value rows.
    Q,
    /// The key rows.
    K,
    /// The value rows.
    V,
    /// `gate | up` along the rows.
    Gate,
    /// The up rows.
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
    /// Rows per attention head, for the rows a [`Source::PairedHeads`] reads.
    pub head_dim: usize,
}

impl FusedDims {
    /// A family with no fused projection.
    pub const NONE: Self = Self {
        q: 0,
        kv: 0,
        inter: 0,
        head_dim: 0,
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
    /// The checkpoint must carry the tensor; its absence is a typed missing-weight error.
    Required,
    /// The tensor may be absent (a bias a family does not have).
    Optional,
    /// When absent, the role reads the same stored tensor as the model-level role named here (a
    /// head tied to the embedding): its own map entry over the other row's entry, so each graph
    /// const keeps one reader and binding shares the stored bytes.
    TiedTo(WeightRole),
}

/// One role of a family's name table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NameRow {
    /// The typed role this row maps.
    pub role: WeightRole,
    /// Where the role's tensor sits in the checkpoint.
    pub source: Source,
    /// Whether the checkpoint must carry it.
    pub presence: Presence,
}

impl NameRow {
    fn is_per_layer(&self) -> bool {
        match self.source {
            Source::LayerTensor { .. } | Source::LayerLinear { .. } | Source::LayerFused { .. } => {
                true
            }
            Source::PairedHeads(inner) => NameRow {
                source: *inner,
                ..*self
            }
            .is_per_layer(),
            Source::Tensor(_) | Source::Linear(_) => false,
        }
    }

    /// The store keys this row can read at `layer`, in preference order.
    fn candidates(&self, layer: Option<usize>) -> Vec<String> {
        let l = layer.map(|l| l.to_string()).unwrap_or_default();
        match self.source {
            Source::PairedHeads(inner) => NameRow {
                source: *inner,
                ..*self
            }
            .candidates(layer),
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

/// gpt-oss's HF rows beside [`HF_BASE`]: attention with a bias on every projection (so the Q/K/V
/// biases are required, not optional) and the per-head attention sink logits.
pub(crate) const HF_SINK_ATTENTION: &[NameRow] = &[
    layer_tensor(
        WeightRole::Attn(AttnRole::QBias),
        ".self_attn.q_proj.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::KBias),
        ".self_attn.k_proj.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::VBias),
        ".self_attn.v_proj.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::OBias),
        ".self_attn.o_proj.bias",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::Sinks),
        ".self_attn.sinks",
        Presence::Required,
    ),
];

/// The `[out, in]` (or `[n]`) shape of every [`HF_BASE`] tensor a checkpoint with `config`'s HF dims
/// carries (the QKV biases when `qkv_bias`), by its HF name: what a family's test fixture holds.
#[cfg(feature = "test-support")]
pub(crate) fn hf_base_shapes(
    config: &serde_json::Value,
    qkv_bias: bool,
) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| crate::registry::Fixture::count(config, f);
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
    let weights = build_map(store, tables, layers, fused, None)
        .map_err(|source| ModelError::Weight { family, source })?;
    check_head(family, weights, head_required)
}

/// [`family_weights`] for a model whose layers `moe.sparse` marks routed: a routed layer maps the
/// router and expert weights `moe.names` describes and no dense feed-forward, a dense layer the
/// opposite.
pub(crate) fn family_weights_routed(
    family: FamilyKey,
    store: &WeightStore,
    tables: &[&[NameRow]],
    layers: usize,
    head_required: bool,
    fused: FusedDims,
    moe: MoeLayers<'_>,
) -> Result<WeightMap, ModelError> {
    let weights = build_map(store, tables, layers, fused, Some(moe))
        .map_err(|source| ModelError::Weight { family, source })?;
    check_head(family, weights, head_required)
}

fn check_head(
    family: FamilyKey,
    weights: WeightMap,
    head_required: bool,
) -> Result<WeightMap, ModelError> {
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

/// Where the routed layers of a model keep their weights.
#[derive(Clone, Copy)]
pub(crate) struct MoeLayers<'a> {
    pub names: &'a MoeNames,
    /// Per layer: routed (router and experts) or dense (the plain gated feed-forward).
    pub sparse: &'a [bool],
    pub experts: usize,
}

/// A routed layer's checkpoint names, each relative to `prefix` and the layer index.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeNames {
    pub prefix: &'static str,
    /// The router's linear stem (`.mlp.gate`).
    pub router: &'static str,
    /// The router's bias tensor, when the family has one.
    pub router_bias: Option<&'static str>,
    /// The always-active shared expert's three linear stems, when the family has one.
    pub shared: Option<SharedNames>,
    pub experts: &'static [ExpertNames],
}

/// A shared expert's gate, up and down linear stems, relative to `prefix` and the layer index, and
/// the stem of its scalar gate (`[1, width]` or `[width]`) when the family gates it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedNames {
    pub gate: &'static str,
    pub up: &'static str,
    pub down: &'static str,
    pub scalar_gate: Option<&'static str>,
}

/// One expert projection's checkpoint names.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpertNames {
    pub proj: ExpertProj,
    /// One tensor holding every expert (`.ffn_gate_exps.weight`), or the per-expert owners a
    /// packed reader makes of it (`{stack}.{e}`).
    pub stack: Option<&'static str>,
    /// One tensor per expert, named `{before}{e}{after}` (`.mlp.experts.` and `.gate_proj.weight`).
    pub each: Option<(&'static str, &'static str)>,
    /// The bias stack `[experts, out]`, when the family has one.
    pub bias: Option<&'static str>,
}

/// Where one expert projection of one layer was found in the store.
enum ExpertKeys {
    Stack(WeightKey),
    PerExpert(Vec<WeightKey>),
}

impl ExpertNames {
    /// The store keys holding this projection of `layer`: the stored stack, the per-expert owners of
    /// a packed stack, or one tensor per expert (`.weight`-suffixed or a packed linear id). When
    /// none is complete, the key to report as missing: the stack, or the first absent expert.
    fn find(
        &self,
        store: &WeightStore,
        prefix: &str,
        layer: usize,
        experts: usize,
    ) -> Result<ExpertKeys, WeightKey> {
        let per_expert = |key: &dyn Fn(usize) -> String| -> Result<ExpertKeys, WeightKey> {
            (0..experts)
                .map(|e| {
                    let key = key(e);
                    if store.contains(&key) {
                        Ok(WeightKey::from(key))
                    } else {
                        Err(WeightKey::from(key))
                    }
                })
                .collect::<Result<_, _>>()
                .map(ExpertKeys::PerExpert)
        };
        let mut absent = None;
        if let Some(stack) = self.stack {
            let key = format!("{prefix}{layer}{stack}");
            if store.contains(&key) {
                return Ok(ExpertKeys::Stack(WeightKey::from(key)));
            }
            if store.contains(&format!("{key}.0")) {
                return per_expert(&|e| format!("{key}.{e}"));
            }
            absent = Some(WeightKey::from(key));
        }
        if let Some((before, after)) = self.each {
            let bare = after.strip_suffix(".weight").unwrap_or(after);
            let named =
                |after: &'static str| move |e: usize| format!("{prefix}{layer}{before}{e}{after}");
            // The `.weight` spelling, or the packed linear id: whichever holds expert 0.
            let form = [after, bare]
                .into_iter()
                .find(|&after| store.contains(&named(after)(0)))
                .unwrap_or(after);
            if store.contains(&named(form)(0)) {
                return per_expert(&named(form));
            }
            absent.get_or_insert_with(|| WeightKey::from(named(after)(0)));
        }
        Err(absent.unwrap_or_else(|| WeightKey::from(format!("{prefix}{layer}"))))
    }
}

pub(crate) fn map_moe_layer(
    map: &mut WeightMapBuilder<'_>,
    store: &WeightStore,
    names: &MoeNames,
    layer: usize,
    experts: usize,
) -> Result<(), WeightMapError> {
    let id = |role| WeightId::layer(layer, WeightRole::Moe(role));
    let prefix = names.prefix;
    let linear = |stem: &str| {
        let key = format!("{prefix}{layer}{stem}.weight");
        if store.contains(&key) {
            return key;
        }
        let bare = format!("{prefix}{layer}{stem}");
        if store.contains(&bare) { bare } else { key }
    };
    let required = |role, key: String| -> Result<WeightView, WeightMapError> {
        if store.contains(&key) {
            Ok(WeightView::Stored(WeightKey::from(key)))
        } else {
            Err(WeightMapError::Missing {
                id: id(role),
                key: WeightKey::from(key),
            })
        }
    };
    map.map(
        id(MoeRole::Router),
        required(MoeRole::Router, linear(names.router))?,
    )?;
    if let Some(bias) = names.router_bias {
        map.map(
            id(MoeRole::RouterBias),
            required(MoeRole::RouterBias, format!("{prefix}{layer}{bias}"))?,
        )?;
    }
    if let Some(shared) = names.shared {
        for (proj, stem) in [
            (ExpertProj::Gate, shared.gate),
            (ExpertProj::Up, shared.up),
            (ExpertProj::Down, shared.down),
        ] {
            let role = MoeRole::Shared(proj);
            map.map(id(role), required(role, linear(stem))?)?;
        }
        if let Some(stem) = shared.scalar_gate {
            let role = MoeRole::SharedGate;
            map.map(id(role), required(role, linear(stem))?)?;
        }
    }
    for expert in names.experts {
        let stack = id(MoeRole::Stack(expert.proj));
        let found = expert
            .find(store, prefix, layer, experts)
            .map_err(|key| WeightMapError::Missing { id: stack, key })?;
        match found {
            ExpertKeys::Stack(key) => {
                map.map(stack, WeightView::Stored(key))?;
            }
            ExpertKeys::PerExpert(keys) => {
                // Per-expert tensors, dense or packed, are one `[experts, ..]` stack weight.
                map.map(stack, WeightView::ExpertStack(keys))?;
            }
        }
        if let Some(bias) = expert.bias {
            let role = MoeRole::StackBias(expert.proj);
            map.map(id(role), required(role, format!("{prefix}{layer}{bias}"))?)?;
        }
    }
    Ok(())
}

/// The stored row each row of the half-split order reads, for `rows` rows of heads of `head_dim` rows
/// stored as adjacent pairs: head `h`'s row `j` is stored at `2j` and its row `head_dim / 2 + j` at
/// `2j + 1`, so reading the stored rows in this order undoes llama.cpp's `permute`. `None` when `rows` is
/// not a whole number of heads or `head_dim` is zero or odd.
fn half_split_order(rows: usize, head_dim: usize) -> Option<Vec<usize>> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || !rows.is_multiple_of(head_dim) {
        return None;
    }
    let half = head_dim / 2;
    Some(
        (0..rows)
            .map(|row| {
                let (head, within) = (row / head_dim, row % head_dim);
                let (pair, side) = if within < half {
                    (within, 0)
                } else {
                    (within - half, 1)
                };
                head * head_dim + 2 * pair + side
            })
            .collect(),
    )
}

fn build_map(
    store: &WeightStore,
    tables: &[&[NameRow]],
    layers: usize,
    fused: FusedDims,
    moe: Option<MoeLayers<'_>>,
) -> Result<WeightMap, WeightMapError> {
    let rows = &merged(tables);
    let mut map = WeightMap::builder(store);
    let is_sparse =
        |layer: usize| moe.is_some_and(|m| m.sparse.get(layer).copied().unwrap_or(false));
    for row in rows {
        let ids: Vec<WeightId> = if row.is_per_layer() {
            (0..layers)
                .filter(|&l| !(is_sparse(l) && matches!(row.role, WeightRole::Ffn(_))))
                .map(|l| WeightId::layer(l, row.role))
                .collect()
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
                Source::PairedHeads(_) => {
                    let rows = store.get(key.as_str()).map_or(0, |entry| entry.shape()[0]);
                    let order =
                        half_split_order(rows, fused.head_dim).ok_or(WeightMapError::HeadRows {
                            id,
                            rows,
                            head_dim: fused.head_dim,
                        })?;
                    WeightView::RowOrder {
                        key,
                        order: order.into(),
                    }
                }
                _ => WeightView::Stored(key),
            };
            map.map(id, view)?;
        }
    }
    if let Some(moe) = moe {
        for layer in (0..layers).filter(|&l| is_sparse(l)) {
            map_moe_layer(&mut map, store, moe.names, layer, moe.experts)?;
        }
    }
    Ok(map.build())
}

/// The key of linear `stem` in `store`: `{stem}.weight` unless only the packed `{stem}` is stored.
pub(crate) fn linear_key(store: &WeightStore, stem: &str) -> String {
    let dense = format!("{stem}.weight");
    if store.contains(&dense) || !store.contains(stem) {
        dense
    } else {
        stem.to_string()
    }
}

/// The weight map of a family that names its tensors in code (hybrid and latent-attention families whose
/// layers do not fit a [`NameRow`] table): the map builder, the store it reads, and one method per way a
/// tensor is named. Every such family builds its map through this, so the missing-tensor error, the packed
/// spelling of a linear and the tied head are written once.
pub(crate) struct TensorRows<'s> {
    family: FamilyKey,
    pub(crate) map: WeightMapBuilder<'s>,
    pub(crate) store: &'s WeightStore,
}

impl<'s> TensorRows<'s> {
    /// An empty map over `store`, whose errors name `family`.
    pub(crate) fn new(family: FamilyKey, store: &'s WeightStore) -> Self {
        Self {
            family,
            map: WeightMap::builder(store),
            store,
        }
    }

    /// The model error for a map failure: a value outside its declared range is the weight's own
    /// error, anything else the family's weight error.
    pub(crate) fn error(&self, source: WeightMapError) -> ModelError {
        match source {
            WeightMapError::IndexOutOfRange {
                id, index, value, ..
            } => ModelError::WeightValue {
                family: self.family,
                id,
                index,
                value,
            },
            source => ModelError::Weight {
                family: self.family,
                source,
            },
        }
    }

    /// The typed missing-weight error naming `id` and the `key` the store lacks.
    pub(crate) fn missing(&self, id: WeightId, key: String) -> ModelError {
        self.error(WeightMapError::Missing {
            id,
            key: WeightKey::from(key),
        })
    }

    /// Map `id` to `view`.
    pub(crate) fn add(&mut self, id: WeightId, view: WeightView) -> Result<(), ModelError> {
        self.map
            .map(id, view)
            .map(|_| ())
            .map_err(|source| self.error(source))
    }

    /// Map `id` onto the stored tensor `key`; a key the store lacks is the map's own typed refusal.
    pub(crate) fn put(&mut self, id: WeightId, key: String) -> Result<(), ModelError> {
        self.add(id, WeightView::Stored(WeightKey::from(key)))
    }

    /// The tensor stored under exactly `key`.
    pub(crate) fn tensor(
        &mut self,
        role: WeightRole,
        layer: Option<usize>,
        key: String,
    ) -> Result<(), ModelError> {
        let id = WeightId { layer, role };
        if !self.store.contains(&key) {
            return Err(self.missing(id, key));
        }
        self.add(id, WeightView::Stored(WeightKey::from(key)))
    }

    /// The linear projection with stem `stem`: `{stem}.weight` dense, else the packed `{stem}`.
    pub(crate) fn linear(
        &mut self,
        role: WeightRole,
        layer: Option<usize>,
        stem: &str,
    ) -> Result<(), ModelError> {
        let id = WeightId { layer, role };
        let key = self.linear_key(stem);
        if !self.store.contains(&key) {
            return Err(self.missing(id, key));
        }
        self.add(id, WeightView::Stored(WeightKey::from(key)))
    }

    /// The key of linear `stem`: its dense spelling unless only the packed one is stored.
    pub(crate) fn linear_key(&self, stem: &str) -> String {
        linear_key(self.store, stem)
    }

    /// The head: the linear `head_stem` when the store holds one, else (when the checkpoint does not
    /// require its own) the embedding's tensor `embed`, which a tied head reads as its own map entry.
    pub(crate) fn head(
        &mut self,
        embed: &str,
        head_stem: &str,
        required: bool,
    ) -> Result<(), ModelError> {
        let head = WeightId::model(WeightRole::Head);
        let key = self.linear_key(head_stem);
        if self.store.contains(&key) {
            self.add(head, WeightView::Stored(WeightKey::from(key)))
        } else if required {
            Err(self.missing(head, key))
        } else {
            self.add(head, WeightView::Stored(WeightKey::from(embed)))
        }
    }

    /// The finished map.
    pub(crate) fn finish(self) -> WeightMap {
        self.map.build()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{DenseWeight, HandleFormat, WeightEntry};
    use poot_quant::{PackedPayload, PackedWeight, SourceRole};
    use poot_tensor::DType;

    use super::test_only::build_weight_map;
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

    /// llama.cpp's `permute` (`reshape(heads, 2, d / 2, ..).swapaxes(1, 2)`) stores half-split row
    /// `(h, side, j)` at `h * d + 2 * j + side`; the order reads each half-split row from there, for two
    /// heads of eight rows, and refuses rows that are not whole even heads.
    /// Mutation: swap the two sides in `half_split_order`; every row of the second half reads the wrong
    /// stored row.
    #[test]
    fn the_half_split_order_inverts_llama_cpps_permute() {
        let (heads, d) = (2, 8);
        let order = half_split_order(heads * d, d).unwrap();
        let mut seen = vec![false; heads * d];
        for h in 0..heads {
            for side in 0..2 {
                for j in 0..d / 2 {
                    let stored = h * d + 2 * j + side;
                    assert_eq!(order[h * d + side * (d / 2) + j], stored);
                    seen[stored] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "the order is a permutation");
        assert_eq!(half_split_order(15, 8), None);
        assert_eq!(half_split_order(9, 3), None);
        assert_eq!(half_split_order(8, 0), None);
    }

    /// Fused rows read row ranges of the one stored tensor, in `q | k | v` and `gate | up` order,
    /// dense or packed; a missing fused tensor is named.
    #[test]
    fn fused_rows_read_row_ranges_of_the_fused_tensor() {
        let dims = FusedDims {
            q: 16,
            kv: 8,
            inter: 12,
            head_dim: 0,
        };
        let tables: &[&[NameRow]] = &[HF_BASE, HF_FUSED_QKV_GATE_UP];
        let map = build_map(&fused_store(None), tables, 1, dims, None).unwrap();
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
                dims,
                None
            ),
            Err(WeightMapError::Missing {
                id: WeightId::layer(0, WeightRole::Attn(AttnRole::Q)),
                key: "model.layers.0.self_attn.qkv_proj.weight".into(),
            })
        );
    }
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    /// The weight map of a `layers`-layer model whose checkpoint `tables` describe (later tables
    /// override earlier ones by role): every model-level row once and every per-layer row at each layer,
    /// each read as one stored entry. A required row whose tensor is absent is
    /// [`WeightMapError::Missing`], naming its preferred key; a tied row that is absent reads its
    /// model-level role's entry.
    pub(crate) fn build_weight_map(
        store: &WeightStore,
        tables: &[&[NameRow]],
        layers: usize,
    ) -> Result<WeightMap, WeightMapError> {
        build_map(store, tables, layers, FusedDims::NONE, None)
    }
}
