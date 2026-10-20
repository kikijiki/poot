//! BERT behind [`Model`], an `encode` model: the bidirectional post-norm encoder of the sentence
//! embedders (MiniLM, BGE, E5) and, with the classifier head option, the cross-encoder rerankers
//! (`BertForSequenceClassification`). Its step returns [`ModelOutput::Hidden`] (every position's final
//! hidden state, pooled on the host) or, with the classifier, [`ModelOutput::Scores`] (the
//! `[CLS]` position's logits).
//!
//! A step is `rows` right-padded sequences of `tokens` positions. Four inputs bind per step: the
//! token ids and absolute positions the standard step declares, the segment id of every token
//! ([`TOKEN_TYPE`], a request input the driver defaults to zeros) and which tokens are real
//! ([`StepInput::RealTokens`]); a padded token is never
//! attended to, so a row's output does not depend on how far its batch pads it.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{BinOp, Builder, Graph, IndexRange, Slot, SlotKey, TensorType, Traced, ops};
use poot_quant::weights::{
    AttnRole, FfnRole, NormRole, WeightId, WeightMap, WeightRole, WeightStore,
};
use poot_tensor::DType;

use crate::components::dims::{Field, Fields};
use crate::components::embed::Embedding;
use crate::components::ffn::{Gelu, PlainMlpWeights, plain_mlp};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::{NormKind, NormParams, NormWeights, norm_with};
use crate::components::standard::{LayerNorms, Step};
use crate::io::{
    Fill, InputId, Layout, ModelSchema, PadValue, Per, Presence as RequestPresence, ProgramId,
    ProgramSpec, REAL_TOKENS_TAG, RequestForm, RequestInput,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, ShapeReason, StepSpec,
    TraceError,
};
use crate::names::{NameRow, Presence, Source, family_weights};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig};
use poot_text::ChatFormat;

/// BERT's [`FamilyKey`]: the name its errors and receipts carry.
pub const FAMILY: FamilyKey = FamilyKey::new("bert");

/// The request input holding every token's segment id (0 for the first sentence, 1 for a
/// cross-encoder's second): dense I32, defaulted to zeros.
pub const TOKEN_TYPE: InputId = InputId::new("token_type");
/// The `Slot::Activation` tag of [`TOKEN_TYPE`], `[rows, tokens]` I32.
const TOKEN_TYPE_TAG: &str = "token_type";

/// The schema entry of [`TOKEN_TYPE`].
fn token_type_input() -> RequestInput {
    RequestInput::new(
        TOKEN_TYPE,
        SlotKey::new(Slot::Activation, Some(TOKEN_TYPE_TAG)),
        RequestForm::new(
            DType::I32,
            Per::Token {
                width: 1,
                layout: Layout::TokenMajor,
                presence: None,
            },
            PadValue::I32(0),
        ),
    )
}

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "bert")],
    build,
    fixture,
    variants: &[("cross_encoder", cross_encoder_fixture)],
};

const fn tensor(role: WeightRole, name: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::Tensor(name),
        presence: Presence::Required,
    }
}

const fn linear_row(role: WeightRole, prefix: &'static str, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear { prefix, stem },
        presence: Presence::Required,
    }
}

const fn layer_row(role: WeightRole, prefix: &'static str, suffix: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor { prefix, suffix },
        presence: Presence::Required,
    }
}

/// The HF rows of a BERT checkpoint whose encoder tensors start with `$p` (`"bert."` under a
/// sequence-classification head, `""` for a bare `BertModel`).
macro_rules! hf_rows {
    ($p:literal) => {
        &[
            tensor(
                WeightRole::Embed,
                concat!($p, "embeddings.word_embeddings.weight"),
            ),
            tensor(
                WeightRole::PosEmbed,
                concat!($p, "embeddings.position_embeddings.weight"),
            ),
            tensor(
                WeightRole::TypeEmbed,
                concat!($p, "embeddings.token_type_embeddings.weight"),
            ),
            tensor(
                WeightRole::EmbedNorm,
                concat!($p, "embeddings.LayerNorm.weight"),
            ),
            tensor(
                WeightRole::EmbedNormBias,
                concat!($p, "embeddings.LayerNorm.bias"),
            ),
            linear_row(
                WeightRole::Attn(AttnRole::Q),
                concat!($p, "encoder.layer."),
                ".attention.self.query",
            ),
            layer_row(
                WeightRole::Attn(AttnRole::QBias),
                concat!($p, "encoder.layer."),
                ".attention.self.query.bias",
            ),
            linear_row(
                WeightRole::Attn(AttnRole::K),
                concat!($p, "encoder.layer."),
                ".attention.self.key",
            ),
            layer_row(
                WeightRole::Attn(AttnRole::KBias),
                concat!($p, "encoder.layer."),
                ".attention.self.key.bias",
            ),
            linear_row(
                WeightRole::Attn(AttnRole::V),
                concat!($p, "encoder.layer."),
                ".attention.self.value",
            ),
            layer_row(
                WeightRole::Attn(AttnRole::VBias),
                concat!($p, "encoder.layer."),
                ".attention.self.value.bias",
            ),
            linear_row(
                WeightRole::Attn(AttnRole::O),
                concat!($p, "encoder.layer."),
                ".attention.output.dense",
            ),
            layer_row(
                WeightRole::Attn(AttnRole::OBias),
                concat!($p, "encoder.layer."),
                ".attention.output.dense.bias",
            ),
            layer_row(
                WeightRole::Norm(NormRole::Attn),
                concat!($p, "encoder.layer."),
                ".attention.output.LayerNorm.weight",
            ),
            layer_row(
                WeightRole::Norm(NormRole::AttnBias),
                concat!($p, "encoder.layer."),
                ".attention.output.LayerNorm.bias",
            ),
            linear_row(
                WeightRole::Ffn(FfnRole::Up),
                concat!($p, "encoder.layer."),
                ".intermediate.dense",
            ),
            layer_row(
                WeightRole::Ffn(FfnRole::UpBias),
                concat!($p, "encoder.layer."),
                ".intermediate.dense.bias",
            ),
            linear_row(
                WeightRole::Ffn(FfnRole::Down),
                concat!($p, "encoder.layer."),
                ".output.dense",
            ),
            layer_row(
                WeightRole::Ffn(FfnRole::DownBias),
                concat!($p, "encoder.layer."),
                ".output.dense.bias",
            ),
            layer_row(
                WeightRole::Norm(NormRole::Ffn),
                concat!($p, "encoder.layer."),
                ".output.LayerNorm.weight",
            ),
            layer_row(
                WeightRole::Norm(NormRole::FfnBias),
                concat!($p, "encoder.layer."),
                ".output.LayerNorm.bias",
            ),
        ]
    };
}

const HF_ROWS_BARE: &[NameRow] = hf_rows!("");
const HF_ROWS_PREFIXED: &[NameRow] = hf_rows!("bert.");

/// The `[CLS]` pooler and the classifier of a `BertForSequenceClassification`.
const HF_CLASSIFIER_ROWS: &[NameRow] = &[
    tensor(WeightRole::Pooler, "bert.pooler.dense.weight"),
    tensor(WeightRole::PoolerBias, "bert.pooler.dense.bias"),
    tensor(WeightRole::Head, "classifier.weight"),
    tensor(WeightRole::HeadBias, "classifier.bias"),
];

const WIDTH: Field = Field::hf("hidden_size");
const HEADS: Field = Field::hf("num_attention_heads");
const LAYERS: Field = Field::hf("num_hidden_layers");
const INTER: Field = Field::hf("intermediate_size");
const MAX_POSITIONS: Field = Field::hf("max_position_embeddings");
const TYPES: Field = Field::hf("type_vocab_size");
const EPS: Field = Field::hf("layer_norm_eps");
const NUM_LABELS: Field = Field::hf("num_labels");

/// What the encoder returns beside its hidden states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BertHead {
    /// The hidden states of every position.
    Embedding,
    /// A `BertForSequenceClassification`: the `[CLS]` pooler and a `labels`-wide classifier.
    Classifier { labels: usize },
}

/// bert's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct BertParams {
    vocab: usize,
    width: usize,
    heads: usize,
    head_dim: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    types: usize,
    norm: NormParams,
    gelu: Gelu,
    head: BertHead,
}

/// Whether the config's `architectures` names a sequence-classification head.
fn is_classifier(config: &serde_json::Value) -> bool {
    config
        .get("architectures")
        .and_then(|a| a.as_array())
        .is_some_and(|archs| {
            archs
                .iter()
                .any(|a| a.as_str() == Some("BertForSequenceClassification"))
        })
}

impl BertParams {
    /// Reads the typed config from an HF `config.json` (BERT has no GGUF form: a GGUF is refused as
    /// `Unsupported`); a width the head count does not divide, an unknown activation or a missing
    /// count is a typed [`ModelError::Config`].
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let RawConfig::HfJson { config, .. } = raw else {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "general.architecture",
                reason: ConfigReason::Unsupported,
            });
        };
        let fields = Fields::new(raw, FAMILY);
        let vocab = fields.vocab()?;
        let width = fields.count(WIDTH, None)?;
        let heads = fields.count(HEADS, None)?;
        if !width.is_multiple_of(heads) {
            return Err(fields.error(fields.key(HEADS), ConfigReason::NotDivisible { by: heads }));
        }
        let eps = fields.float(EPS, 1e-12)?;
        let norm = NormParams::of(NormKind::Layer, eps)
            .map_err(|_| fields.error(fields.key(EPS), ConfigReason::NotFinitePositive))?;
        // HF's `gelu` is the exact erf form; the tanh approximations have their own names.
        let gelu = match config.get("hidden_act").and_then(|a| a.as_str()) {
            None | Some("gelu") => Gelu::Erf,
            Some("gelu_pytorch_tanh" | "gelu_new") => Gelu::Tanh,
            Some(_) => return Err(fields.error("hidden_act", ConfigReason::Unsupported)),
        };
        let head = if is_classifier(config) {
            // `id2label` names one label per class; `num_labels` counts them (HF defaults to two).
            let labels = match config.get("id2label").and_then(|m| m.as_object()) {
                Some(map) if !map.is_empty() => map.len(),
                _ => fields.count(NUM_LABELS, Some(2))?,
            };
            BertHead::Classifier { labels }
        } else {
            BertHead::Embedding
        };
        Ok(Self {
            vocab,
            width,
            heads,
            head_dim: width / heads,
            inter: fields.count(INTER, None)?,
            layers: fields.count(LAYERS, None)?,
            max_positions: fields.count(MAX_POSITIONS, Some(512))?,
            types: fields.count(TYPES, Some(2))?,
            norm,
            gelu,
            head,
        })
    }
}

/// A layer's attention projections (each `[width, width]`, with a bias) and its feed-forward.
#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    ffn: PlainMlpWeights,
}

/// A `[width, width]` projection and its bias, read from a layer's attention weights.
#[derive(Debug)]
pub(crate) struct Projection {
    pub(crate) weight: Weight,
    pub(crate) bias: Weight,
}

impl Projection {
    pub(crate) fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        role: AttnRole,
        bias: AttnRole,
    ) -> Result<Self, WeightError> {
        Ok(Self {
            weight: Weight::new(
                map,
                WeightId::layer(layer, WeightRole::Attn(role)),
                &[width, width],
            )?,
            bias: Weight::new(
                map,
                WeightId::layer(layer, WeightRole::Attn(bias)),
                &[width],
            )?,
        })
    }

    pub(crate) fn apply(&self, b: &Builder, x: Traced) -> Result<Traced, TraceError> {
        linear(b, x, &self.weight, Some(&self.bias))
    }
}

/// Full bidirectional self-attention of `x` (`[rows, tokens, heads * head_dim]`) through the four
/// projections: every position attends to every key `mask` leaves visible (`[rows, 1, 1, tokens]`,
/// additive; `None` leaves them all). An encoder's attention: BERT's and a vision tower's.
pub(crate) fn bidirectional_attention(
    b: &Builder,
    x: Traced,
    [q, k, v, o]: [&Projection; 4],
    (heads, head_dim): (usize, usize),
    mask: Option<Traced>,
) -> Result<Traced, TraceError> {
    let shape = b.aval(x)?.shape;
    let (rows, tokens) = (shape[0], shape[1]);
    let split = |proj: &Projection| {
        let y = b.reshape(proj.apply(b, x)?, vec![rows, tokens, heads, head_dim])?;
        Ok::<_, TraceError>(b.transpose(y, vec![0, 2, 1, 3])?)
    };
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (q, k, v) = (split(q)?, split(k)?, split(v)?);
    let out = match mask {
        Some(mask) => ops::attention_masked(b, q, k, v, 1, scale, mask)?,
        None => ops::attention(b, q, k, v, 1, scale)?,
    };
    let merged = b.reshape(
        b.transpose(out, vec![0, 2, 1, 3])?,
        vec![rows, tokens, heads * head_dim],
    )?;
    o.apply(b, merged)
}

/// The `[CLS]` pooler and classifier.
#[derive(Debug)]
struct Classifier {
    pooler: Projection,
    head: Weight,
    head_bias: Weight,
}

/// The built BERT encoder: its config, the name-resolved weights and the per-layer projections,
/// ready to trace its `encode` program. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct Bert {
    params: BertParams,
    weights: WeightMap,
    word: Embedding,
    positions: Weight,
    types: Weight,
    embed_norm: NormWeights,
    layers: Vec<Layer>,
    classifier: Option<Classifier>,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let p = BertParams::from_raw(raw)?;
    let encoder = if store.contains("embeddings.word_embeddings.weight") {
        HF_ROWS_BARE
    } else {
        HF_ROWS_PREFIXED
    };
    let weights = match p.head {
        BertHead::Embedding => family_weights(FAMILY, store, &[encoder], p.layers, false)?,
        BertHead::Classifier { .. } => family_weights(
            FAMILY,
            store,
            &[encoder, HF_CLASSIFIER_ROWS],
            p.layers,
            false,
        )?,
    };
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let w = p.width;
    let model_weight = |role, shape: &[usize]| Weight::new(&weights, WeightId::model(role), shape);
    let norm = |role, bias| -> Result<NormWeights, WeightError> {
        Ok(NormWeights {
            scale: model_weight(role, &[w])?,
            bias: Some(model_weight(bias, &[w])?),
        })
    };
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::biased(&weights, l, w)?,
                q: Projection::new(&weights, l, w, AttnRole::Q, AttnRole::QBias)?,
                k: Projection::new(&weights, l, w, AttnRole::K, AttnRole::KBias)?,
                v: Projection::new(&weights, l, w, AttnRole::V, AttnRole::VBias)?,
                o: Projection::new(&weights, l, w, AttnRole::O, AttnRole::OBias)?,
                ffn: PlainMlpWeights::new(&weights, l, w, p.inter, true)?,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let classifier = match p.head {
        BertHead::Embedding => None,
        BertHead::Classifier { labels } => Some(
            (|| {
                Ok(Classifier {
                    pooler: Projection {
                        weight: model_weight(WeightRole::Pooler, &[w, w])?,
                        bias: model_weight(WeightRole::PoolerBias, &[w])?,
                    },
                    head: model_weight(WeightRole::Head, &[labels, w])?,
                    head_bias: model_weight(WeightRole::HeadBias, &[labels])?,
                })
            })()
            .map_err(weight_error)?,
        ),
    };
    let output = match p.head {
        BertHead::Embedding => ModelOutput::Hidden { width: p.width },
        BertHead::Classifier { labels } => ModelOutput::Scores { labels },
    };
    let schema = ModelSchema::builder()
        .input(token_type_input())
        .encode(
            ProgramSpec::new(ProgramId::ENCODE, output)
                .reading(TOKEN_TYPE, RequestPresence::Defaulted(Fill::Pad)),
        )
        .build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: BTreeSet::new(),
        bos: None,
        prompt_bos: None,
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
        tool_codec: None,
    };
    Ok(ModelParts::new(Box::new(Bert {
        word: Embedding::new(&weights, p.vocab, w).map_err(weight_error)?,
        positions: model_weight(WeightRole::PosEmbed, &[p.max_positions, w])
            .map_err(weight_error)?,
        types: model_weight(WeightRole::TypeEmbed, &[p.types, w]).map_err(weight_error)?,
        embed_norm: norm(WeightRole::EmbedNorm, WeightRole::EmbedNormBias).map_err(weight_error)?,
        layers,
        classifier,
        params: p,
        weights,
        config,
        schema,
    })))
}

/// The F32 rows of `table` (`[n, width]`) at `ids` (`[rows, tokens]` I32): `[rows, tokens, width]`.
fn table_rows(b: &Builder, table: &Weight, ids: Traced) -> Result<Traced, TraceError> {
    let rows = b.gather(table.declare(b)?, 0, ids)?;
    if b.aval(rows)?.dtype == DType::F32 {
        Ok(rows)
    } else {
        Ok(b.cast(rows, DType::F32)?)
    }
}

impl Model for Bert {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: Builder) -> Result<Graph, TraceError> {
        let p = &self.params;
        let shape = spec.shape;
        if shape.tokens.get() > p.max_positions {
            return Err(TraceError::ShapeUnsupported {
                family: FAMILY,
                shape,
                reason: ShapeReason::CapacityAboveMax {
                    max: p.max_positions,
                },
            });
        }
        let step = Step::new(&b, shape)?.returning_hidden();
        let (rows, tokens) = (shape.rows.get(), shape.tokens.get());
        let ids = TensorType::new(vec![rows, tokens], DType::I32);
        let types = b.slot_named(Slot::Activation, TOKEN_TYPE_TAG, ids.clone())?;
        // A key is real (1) or padding (0): declared so the mask's `I32 -> F32` cast is proven exact.
        let real = b.slot_named_ranged(Slot::Activation, REAL_TOKENS_TAG, ids, IndexRange::FLAG)?;
        // 0 for a real key, -1e9 for padding: the additive key mask every query shares.
        let mask = b.reshape(
            b.binary_scalar(
                BinOp::Mul,
                b.binary_scalar(BinOp::Sub, b.cast(real, DType::F32)?, Builder::f32(1.0))?,
                Builder::f32(1.0e9),
            )?,
            vec![rows, 1, 1, tokens],
        )?;

        let embedded = b.binary(
            BinOp::Add,
            b.binary(
                BinOp::Add,
                self.word.embed(&b, step.tokens())?,
                table_rows(&b, &self.positions, step.pos())?,
            )?,
            table_rows(&b, &self.types, types)?,
        )?;
        let mut x = norm_with(&b, embedded, &self.embed_norm, p.norm)?;
        for (l, layer) in self.layers.iter().enumerate() {
            let _scope = b.layer_scope(l);
            let attn = bidirectional_attention(
                &b,
                x,
                [&layer.q, &layer.k, &layer.v, &layer.o],
                (p.heads, p.head_dim),
                Some(mask),
            )?;
            x = norm_with(
                &b,
                b.binary(BinOp::Add, x, attn)?,
                &layer.norms.attn,
                p.norm,
            )?;
            let ffn = plain_mlp(&b, x, &layer.ffn, p.gelu)?;
            x = norm_with(&b, b.binary(BinOp::Add, x, ffn)?, &layer.norms.ffn, p.norm)?;
        }
        let out = match &self.classifier {
            None => x,
            Some(c) => {
                let cls = b.slice(x, 1, 0, 1)?;
                let pooled = ops::tanh(&b, c.pooler.apply(&b, cls)?)?;
                linear(&b, pooled, &c.head, Some(&c.head_bias))?
            }
        };
        Ok(step.finish(b, out)?)
    }
}

/// The fixture's config: a two-layer BERT with a 64-wide, four-head encoder.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "bert",
        "architectures": ["BertModel"],
        "vocab_size": 48,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 16,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
    })
}

/// The shape of every tensor of a BERT checkpoint with `config`'s dims whose encoder names start
/// with `prefix`.
pub(crate) fn encoder_shapes(
    config: &serde_json::Value,
    prefix: &str,
) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| crate::registry::Fixture::count(config, f);
    let (vocab, w, inter) = (n("vocab_size"), n("hidden_size"), n("intermediate_size"));
    let (positions, types) = (n("max_position_embeddings"), n("type_vocab_size"));
    let name = |s: &str| format!("{prefix}{s}");
    let mut shapes = vec![
        (name("embeddings.word_embeddings.weight"), vec![vocab, w]),
        (
            name("embeddings.position_embeddings.weight"),
            vec![positions, w],
        ),
        (
            name("embeddings.token_type_embeddings.weight"),
            vec![types, w],
        ),
        (name("embeddings.LayerNorm.weight"), vec![w]),
        (name("embeddings.LayerNorm.bias"), vec![w]),
    ];
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| name(&format!("encoder.layer.{l}.{s}"));
        for proj in ["query", "key", "value"] {
            shapes.push((k(&format!("attention.self.{proj}.weight")), vec![w, w]));
            shapes.push((k(&format!("attention.self.{proj}.bias")), vec![w]));
        }
        shapes.extend([
            (k("attention.output.dense.weight"), vec![w, w]),
            (k("attention.output.dense.bias"), vec![w]),
            (k("attention.output.LayerNorm.weight"), vec![w]),
            (k("attention.output.LayerNorm.bias"), vec![w]),
            (k("intermediate.dense.weight"), vec![inter, w]),
            (k("intermediate.dense.bias"), vec![inter]),
            (k("output.dense.weight"), vec![w, inter]),
            (k("output.dense.bias"), vec![w]),
            (k("output.LayerNorm.weight"), vec![w]),
            (k("output.LayerNorm.bias"), vec![w]),
        ]);
    }
    shapes
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = encoder_shapes(&config, "");
    Fixture::bf16(config, shapes)
}

/// The cross-encoder fixture: a `bert.`-prefixed encoder, the pooler and a one-logit classifier.
fn cross_encoder_fixture() -> Fixture {
    let mut config = fixture_config();
    config["architectures"] = serde_json::json!(["BertForSequenceClassification"]);
    config["id2label"] = serde_json::json!({"0": "LABEL_0"});
    let w = 32;
    let mut shapes = encoder_shapes(&config, "bert.");
    shapes.extend([
        ("bert.pooler.dense.weight".to_string(), vec![w, w]),
        ("bert.pooler.dense.bias".to_string(), vec![w]),
        ("classifier.weight".to_string(), vec![1, w]),
        ("classifier.bias".to_string(), vec![1]),
    ]);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
pub(crate) mod reference {
    //! Plain f64 building blocks the encoder families' tests write their references from.

    /// erf by its Maclaurin series, exact to f64 precision for the |x| < 4 the fixture reaches.
    pub(crate) fn erf(x: f64) -> f64 {
        let (mut term, mut sum) = (x, x);
        for n in 1..80 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }

    /// `x [t, k]` times `w^T` (`w` is `[n, k]`), plus `bias [n]`.
    pub(crate) fn dense(x: &[f64], t: usize, w: &[f64], bias: &[f64]) -> Vec<f64> {
        let n = bias.len();
        let k = w.len() / n;
        (0..t)
            .flat_map(|i| {
                (0..n).map(move |o| {
                    bias[o] + (0..k).map(|j| x[i * k + j] * w[o * k + j]).sum::<f64>()
                })
            })
            .collect()
    }

    pub(crate) fn layer_norm(
        x: &[f64],
        width: usize,
        scale: &[f64],
        bias: &[f64],
        eps: f64,
    ) -> Vec<f64> {
        x.chunks(width)
            .flat_map(|row| {
                let mean = row.iter().sum::<f64>() / width as f64;
                let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / width as f64;
                let inv = 1.0 / (var + eps).sqrt();
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (v - mean) * inv * scale[j] + bias[j])
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use poot_quant::weights::WeightStore;

    use super::reference::{dense, erf, layer_norm};
    use super::*;
    use crate::components::standard::oracle::{self, assert_matches_f64};
    use crate::components::testing::{build as built, values};
    use crate::io::ProgramId;
    use crate::model::{LogitRows, StepShape, StepSpec};

    /// A checkpoint tensor's values, by its HF name.
    fn tensor(store: &WeightStore, name: &str) -> Vec<f64> {
        values(store.get(name).unwrap_or_else(|| panic!("{name}")))
            .into_iter()
            .map(f64::from)
            .collect()
    }

    /// BERT's forward over one unpadded sequence, written from the model card in f64 from the
    /// checkpoint's own values: the final hidden state of every position `[t, width]`.
    fn reference(store: &WeightStore, prefix: &str, ids: &[i32], types: &[i32]) -> Vec<f64> {
        let config = fixture_config();
        let n = |f: &str| crate::registry::Fixture::count(&config, f);
        let (w, heads, layers, inter) = (
            n("hidden_size"),
            n("num_attention_heads"),
            n("num_hidden_layers"),
            n("intermediate_size"),
        );
        let (d, t) = (w / heads, ids.len());
        let get = |name: &str| tensor(store, &format!("{prefix}{name}"));
        let (word, pos, kind) = (
            get("embeddings.word_embeddings.weight"),
            get("embeddings.position_embeddings.weight"),
            get("embeddings.token_type_embeddings.weight"),
        );
        let mut x: Vec<f64> = (0..t)
            .flat_map(|i| {
                let (word, pos, kind) = (&word, &pos, &kind);
                (0..w).map(move |j| {
                    word[ids[i] as usize * w + j] + pos[i * w + j] + kind[types[i] as usize * w + j]
                })
            })
            .collect();
        x = layer_norm(
            &x,
            w,
            &get("embeddings.LayerNorm.weight"),
            &get("embeddings.LayerNorm.bias"),
            1e-12,
        );
        for l in 0..layers {
            let k = |s: &str| get(&format!("encoder.layer.{l}.{s}"));
            let proj = |name: &str| {
                dense(
                    &x,
                    t,
                    &k(&format!("attention.self.{name}.weight")),
                    &k(&format!("attention.self.{name}.bias")),
                )
            };
            let (q, key, value) = (proj("query"), proj("key"), proj("value"));
            let mut merged = vec![0.0; t * w];
            for h in 0..heads {
                for i in 0..t {
                    let scores: Vec<f64> = (0..t)
                        .map(|j| {
                            (0..d)
                                .map(|c| q[i * w + h * d + c] * key[j * w + h * d + c])
                                .sum::<f64>()
                                / (d as f64).sqrt()
                        })
                        .collect();
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = exp.iter().sum();
                    for c in 0..d {
                        merged[i * w + h * d + c] = (0..t)
                            .map(|j| exp[j] / sum * value[j * w + h * d + c])
                            .sum();
                    }
                }
            }
            let attn = dense(
                &merged,
                t,
                &k("attention.output.dense.weight"),
                &k("attention.output.dense.bias"),
            );
            let sum: Vec<f64> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
            x = layer_norm(
                &sum,
                w,
                &k("attention.output.LayerNorm.weight"),
                &k("attention.output.LayerNorm.bias"),
                1e-12,
            );
            let up = dense(
                &x,
                t,
                &k("intermediate.dense.weight"),
                &k("intermediate.dense.bias"),
            );
            let act: Vec<f64> = up
                .iter()
                .map(|&v| 0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2)))
                .collect();
            let down = dense(&act, t, &k("output.dense.weight"), &k("output.dense.bias"));
            assert_eq!(act.len(), t * inter);
            let sum: Vec<f64> = x.iter().zip(&down).map(|(a, b)| a + b).collect();
            x = layer_norm(
                &sum,
                w,
                &k("output.LayerNorm.weight"),
                &k("output.LayerNorm.bias"),
                1e-12,
            );
        }
        x
    }

    fn encode_shape(rows: usize, tokens: usize) -> StepShape {
        StepShape::new(
            NonZeroUsize::new(rows).unwrap(),
            NonZeroUsize::new(tokens).unwrap(),
            NonZeroUsize::new(tokens).unwrap(),
        )
        .with_logits(LogitRows::All)
    }

    /// The model's `Encode` graph over padded rows of `lens` real tokens each, on the oracle.
    fn encode(
        model: &dyn Model,
        store: &WeightStore,
        rows: &[(&[i32], &[i32])],
        tokens: usize,
    ) -> Vec<f32> {
        let pad = |ids: &[i32]| {
            let mut padded = ids.to_vec();
            padded.resize(tokens, 0);
            padded
        };
        let ids: Vec<i32> = rows.iter().flat_map(|(ids, _)| pad(ids)).collect();
        let types: Vec<i32> = rows.iter().flat_map(|(_, types)| pad(types)).collect();
        let keep: Vec<i32> = rows
            .iter()
            .flat_map(|(ids, _)| (0..tokens).map(|i| i32::from(i < ids.len())))
            .collect();
        let pos: Vec<i32> = rows.iter().flat_map(|_| 0..tokens as i32).collect();
        let g = model
            .trace(
                &StepSpec::encode(encode_shape(rows.len(), tokens)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        oracle::eval_named(
            &g,
            store,
            model.weights(),
            &[],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &[
                ("activation.token_type", &types),
                ("activation.attention_mask", &keep),
            ],
            &[],
        )
        .0
    }

    const SHORT: [i32; 3] = [5, 17, 3];
    const LONG: [i32; 5] = [9, 2, 31, 4, 12];

    /// SC-003 (model half): padded rows of different lengths in one `Encode` step equal an f64
    /// BERT forward of each row alone, at every real position (tier 2), and the padding changes
    /// nothing: with the mask, the short row's output is its own, not a function of the batch.
    /// Mutation: skip the attention mask (every key visible); the short row's rows leak the padding
    /// and this goes red.
    #[test]
    fn padded_rows_equal_an_f64_forward_of_each_row_alone() {
        let fx = fixture();
        let model = built(&ENTRY, &fx.raw(), &fx.store);
        let w = 32;
        let got = encode(
            &*model,
            &fx.store,
            &[(&SHORT, &[0, 0, 0]), (&LONG, &[0, 0, 1, 1, 1])],
            5,
        );
        let short = reference(&fx.store, "", &SHORT, &[0, 0, 0]);
        let long = reference(&fx.store, "", &LONG, &[0, 0, 1, 1, 1]);
        assert_eq!(got.len(), 2 * 5 * w);
        assert_matches_f64(&got[..3 * w], &short, 1e-4);
        assert_matches_f64(&got[5 * w..], &long, 1e-4);
    }

    /// The segment ids matter: the same tokens with a second segment change the hidden states.
    #[test]
    fn token_types_change_the_hidden_states() {
        let fx = fixture();
        let model = built(&ENTRY, &fx.raw(), &fx.store);
        let a = encode(&*model, &fx.store, &[(&LONG, &[0, 0, 0, 0, 0])], 5);
        let b = encode(&*model, &fx.store, &[(&LONG, &[0, 0, 1, 1, 1])], 5);
        assert_ne!(a, b);
    }

    /// The cross-encoder fixture's step returns the `[CLS]` pooler and classifier logit of each
    /// row (`[rows, 1, 1]`), equal to an f64 pooler `tanh` and classifier over the reference
    /// hidden state.
    #[test]
    fn the_classifier_head_scores_the_cls_position() {
        let fx = cross_encoder_fixture();
        let model = built(&ENTRY, &fx.raw(), &fx.store);
        assert_eq!(
            model.schema().encode().map(|p| p.output()),
            Some(ModelOutput::Scores { labels: 1 })
        );
        let got = encode(
            &*model,
            &fx.store,
            &[(&SHORT, &[0, 0, 1]), (&LONG, &[0, 0, 1, 1, 1])],
            5,
        );
        assert_eq!(got.len(), 2);
        let get = |name: &str| tensor(&fx.store, name);
        let w = 32;
        let score = |ids: &[i32], types: &[i32]| {
            let hidden = reference(&fx.store, "bert.", ids, types);
            let pooled: Vec<f64> = dense(
                &hidden[..w],
                1,
                &get("bert.pooler.dense.weight"),
                &get("bert.pooler.dense.bias"),
            )
            .into_iter()
            .map(f64::tanh)
            .collect();
            dense(
                &pooled,
                1,
                &get("classifier.weight"),
                &get("classifier.bias"),
            )[0]
        };
        assert_matches_f64(
            &got,
            &[score(&SHORT, &[0, 0, 1]), score(&LONG, &[0, 0, 1, 1, 1])],
            1e-4,
        );
    }

    /// A sequence longer than the position table is a typed refusal, and so is any program but
    /// `encode`.
    #[test]
    fn a_shape_or_program_it_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = built(&ENTRY, &fx.raw(), &fx.store);
        assert!(matches!(
            model.trace(
                &StepSpec::encode(encode_shape(1, 17)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported {
                reason: ShapeReason::CapacityAboveMax { max: 16 },
                ..
            })
        ));
        assert!(matches!(
            model.trace(
                &StepSpec::step(encode_shape(1, 4)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ProgramUnsupported {
                program: ProgramId::STEP,
                ..
            })
        ));
    }

    /// An activation the family has no composition for is refused at build, naming the field.
    #[test]
    fn an_unknown_activation_is_a_config_error() {
        let fx = fixture();
        let mut config = fixture_config();
        config["hidden_act"] = serde_json::json!("relu");
        let raw = RawConfig::HfJson {
            config: &config,
            generation: None,
            preprocessor: None,
        };
        let err = build(&raw, &fx.store).unwrap_err();
        assert!(err.to_string().contains("hidden_act"), "{err}");
    }
}
