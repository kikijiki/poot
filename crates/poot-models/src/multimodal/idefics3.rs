//! The idefics3 / SmolVLM vision tower behind [`Model`], a model whose only program is `encode`: a SigLIP encoder over image frames and the connector (pixel shuffle, then the
//! projection into the text tower's embedding width). One step is `rows` frames; each returns the
//! `patches / scale^2` embedding rows the text tower takes in place of the prompt's `<image>`
//! placeholders ([`EMBEDS`](crate::components::embed::EMBEDS)).
//!
//! The checkpoint holds the tower beside the text tower (`model.text_model.`), which a second registry
//! resolve builds as the `llama` family; this family reads its own rows (`model.vision_model.`,
//! `model.connector.`) and ignores the rest.
//!
//! Its config is the checkpoint's `vision_config`, with the two top-level facts it needs added by the
//! caller: `scale_factor` (the connector's pixel shuffle) and `text_hidden_size` (the projection's
//! output width).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{BinOp, Builder, Graph, Slot, SlotKey, TensorType, Traced};
use poot_quant::weights::{
    AttnRole, FfnRole, NormRole, WeightId, WeightMap, WeightRole, WeightStore,
};
use poot_tensor::DType;

use crate::architectures::bert::{Projection, bidirectional_attention};
use crate::components::dims::{Field, Fields};
use crate::components::ffn::{Gelu, PlainMlpWeights, plain_mlp};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::{NormKind, NormParams, NormWeights, norm_with};
use crate::components::standard::LayerNorms;
use crate::io::{
    InputId, ModelSchema, PadValue, Per, Presence as RequestPresence, ProgramId, ProgramSpec,
    RequestForm, RequestInput,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, ShapeReason, StepSpec,
    TraceError,
};
use crate::names::{NameRow, Presence, Source, family_weights};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_text::ChatFormat;

/// The idefics3 vision tower's [`FamilyKey`]: a family of its own beside the text tower's.
pub const FAMILY: FamilyKey = FamilyKey::new("idefics3_vision");

/// The request input holding a frame (or, for the windowed tower, a row of patches): normalized
/// pixels, channel first. One tensor per encode row.
pub const PIXELS: InputId = InputId::new("pixels");
/// The `Slot::Activation` tag of [`PIXELS`]: `[rows, channels, size, size]` F32 for this tower.
pub(super) const PIXELS_TAG: &str = "pixels";

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "idefics3_vision")],
    build,
    fixture,
    variants: &[],
};

const fn tensor(role: WeightRole, name: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::Tensor(name),
        presence: Presence::Required,
    }
}

const fn linear_row(role: WeightRole, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear {
            prefix: "model.vision_model.encoder.layers.",
            stem,
        },
        presence: Presence::Required,
    }
}

const fn layer_row(role: WeightRole, suffix: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor {
            prefix: "model.vision_model.encoder.layers.",
            suffix,
        },
        presence: Presence::Required,
    }
}

const HF_ROWS: &[NameRow] = &[
    tensor(
        WeightRole::Patch,
        "model.vision_model.embeddings.patch_embedding.weight",
    ),
    tensor(
        WeightRole::PatchBias,
        "model.vision_model.embeddings.patch_embedding.bias",
    ),
    tensor(
        WeightRole::PosEmbed,
        "model.vision_model.embeddings.position_embedding.weight",
    ),
    tensor(
        WeightRole::FinalNorm,
        "model.vision_model.post_layernorm.weight",
    ),
    tensor(
        WeightRole::FinalNormBias,
        "model.vision_model.post_layernorm.bias",
    ),
    tensor(
        WeightRole::Connector,
        "model.connector.modality_projection.proj.weight",
    ),
    linear_row(WeightRole::Attn(AttnRole::Q), ".self_attn.q_proj"),
    layer_row(WeightRole::Attn(AttnRole::QBias), ".self_attn.q_proj.bias"),
    linear_row(WeightRole::Attn(AttnRole::K), ".self_attn.k_proj"),
    layer_row(WeightRole::Attn(AttnRole::KBias), ".self_attn.k_proj.bias"),
    linear_row(WeightRole::Attn(AttnRole::V), ".self_attn.v_proj"),
    layer_row(WeightRole::Attn(AttnRole::VBias), ".self_attn.v_proj.bias"),
    linear_row(WeightRole::Attn(AttnRole::O), ".self_attn.out_proj"),
    layer_row(
        WeightRole::Attn(AttnRole::OBias),
        ".self_attn.out_proj.bias",
    ),
    layer_row(WeightRole::Norm(NormRole::Attn), ".layer_norm1.weight"),
    layer_row(WeightRole::Norm(NormRole::AttnBias), ".layer_norm1.bias"),
    layer_row(WeightRole::Norm(NormRole::Ffn), ".layer_norm2.weight"),
    layer_row(WeightRole::Norm(NormRole::FfnBias), ".layer_norm2.bias"),
    linear_row(WeightRole::Ffn(FfnRole::Up), ".mlp.fc1"),
    layer_row(WeightRole::Ffn(FfnRole::UpBias), ".mlp.fc1.bias"),
    linear_row(WeightRole::Ffn(FfnRole::Down), ".mlp.fc2"),
    layer_row(WeightRole::Ffn(FfnRole::DownBias), ".mlp.fc2.bias"),
];

const WIDTH: Field = Field::hf("hidden_size");
const HEADS: Field = Field::hf("num_attention_heads");
const LAYERS: Field = Field::hf("num_hidden_layers");
const INTER: Field = Field::hf("intermediate_size");
const IMAGE_SIZE: Field = Field::hf("image_size");
const PATCH_SIZE: Field = Field::hf("patch_size");
const CHANNELS: Field = Field::hf("num_channels");
const EPS: Field = Field::hf("layer_norm_eps");
const SCALE: Field = Field::hf("scale_factor");
const TEXT_WIDTH: Field = Field::hf("text_hidden_size");

/// The tower's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Idefics3Params {
    width: usize,
    heads: usize,
    head_dim: usize,
    inter: usize,
    layers: usize,
    channels: usize,
    image_size: usize,
    patch_size: usize,
    scale: usize,
    text_width: usize,
    norm: NormParams,
    gelu: Gelu,
}

impl Idefics3Params {
    /// Reads the typed config from the checkpoint's `vision_config` (plus the caller-added
    /// `scale_factor` and `text_hidden_size`); a missing or inconsistent field is a typed
    /// [`ModelError::Config`].
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let RawConfig::HfJson { config, .. } = raw else {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "general.architecture",
                reason: ConfigReason::Unsupported,
            });
        };
        let fields = Fields::new(raw, FAMILY);
        let width = fields.count(WIDTH, None)?;
        let heads = fields.count(HEADS, None)?;
        if !width.is_multiple_of(heads) {
            return Err(fields.error(fields.key(HEADS), ConfigReason::NotDivisible { by: heads }));
        }
        let (image_size, patch_size) = (
            fields.count(IMAGE_SIZE, None)?,
            fields.count(PATCH_SIZE, None)?,
        );
        if !image_size.is_multiple_of(patch_size) {
            return Err(fields.error(
                fields.key(IMAGE_SIZE),
                ConfigReason::NotDivisible { by: patch_size },
            ));
        }
        let scale = fields.count(SCALE, None)?;
        let grid = image_size / patch_size;
        if !grid.is_multiple_of(scale) {
            return Err(fields.error(fields.key(SCALE), ConfigReason::NotDivisible { by: scale }));
        }
        let eps = fields.float(EPS, 1e-6)?;
        let norm = NormParams::of(NormKind::Layer, eps)
            .map_err(|_| fields.error(fields.key(EPS), ConfigReason::NotFinitePositive))?;
        let gelu = match config.get("hidden_act").and_then(|a| a.as_str()) {
            None | Some("gelu_pytorch_tanh" | "gelu_new") => Gelu::Tanh,
            Some("gelu") => Gelu::Erf,
            Some(_) => return Err(fields.error("hidden_act", ConfigReason::Unsupported)),
        };
        Ok(Self {
            width,
            heads,
            head_dim: width / heads,
            inter: fields.count(INTER, None)?,
            layers: fields.count(LAYERS, None)?,
            channels: fields.count(CHANNELS, Some(3))?,
            image_size,
            patch_size,
            scale,
            text_width: fields.count(TEXT_WIDTH, None)?,
            norm,
            gelu,
        })
    }

    /// Patches per frame: the grid squared.
    pub fn patches(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }

    /// Embedding rows per frame the connector returns.
    pub fn visual_tokens(&self) -> usize {
        self.patches() / (self.scale * self.scale)
    }
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    ffn: PlainMlpWeights,
}

/// The built idefics3 vision tower: config, name-resolved weights, the SigLIP encoder layers and
/// the connector. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct Idefics3Vision {
    params: Idefics3Params,
    weights: WeightMap,
    patch: Weight,
    patch_bias: Weight,
    positions: Weight,
    post_norm: NormWeights,
    connector: Weight,
    layers: Vec<Layer>,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Idefics3Params::from_raw(raw)?;
    let weights = family_weights(FAMILY, store, &[HF_ROWS], p.layers, false)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let w = p.width;
    let model_weight = |role, shape: &[usize]| Weight::new(&weights, WeightId::model(role), shape);
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
    let build_rest = || -> Result<_, WeightError> {
        Ok((
            model_weight(
                WeightRole::Patch,
                &[w, p.channels, p.patch_size, p.patch_size],
            )?,
            model_weight(WeightRole::PatchBias, &[w])?,
            model_weight(WeightRole::PosEmbed, &[p.patches(), w])?,
            NormWeights {
                scale: model_weight(WeightRole::FinalNorm, &[w])?,
                bias: Some(model_weight(WeightRole::FinalNormBias, &[w])?),
            },
            model_weight(
                WeightRole::Connector,
                &[p.text_width, w * p.scale * p.scale],
            )?,
        ))
    };
    let (patch, patch_bias, positions, post_norm, connector) =
        build_rest().map_err(weight_error)?;
    let frame = RequestInput::new(
        PIXELS,
        SlotKey::new(Slot::Activation, Some(PIXELS_TAG)),
        RequestForm::new(
            DType::F32,
            Per::Row {
                trailing: vec![p.channels, p.image_size, p.image_size],
                padded: false,
                valid: None,
            },
            PadValue::F32(0.0),
        ),
    );
    let schema = ModelSchema::builder()
        .input(frame)
        .encode(
            ProgramSpec::new(
                ProgramId::ENCODE,
                ModelOutput::Hidden {
                    width: p.text_width,
                },
            )
            .reading(PIXELS, RequestPresence::Required)
            .with_positions(NonZeroUsize::new(p.patches()).expect("a tower has patches")),
        )
        .build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: 0,
        max_positions: p.patches(),
        eos: BTreeSet::new(),
        bos: None,
        prompt_bos: None,
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(Idefics3Vision {
        params: p,
        weights,
        patch,
        patch_bias,
        positions,
        post_norm,
        connector,
        layers,
        config,
        schema,
    }))
}

/// Pixel shuffle: `[rows, seq, embed]` over a square `grid x grid` patch grid to
/// `[rows, seq / scale^2, embed * scale^2]`, grouping each `scale x scale` block of patches into the
/// channel axis (idefics3's connector `pixel_shuffle`: view, permute, reshape).
fn pixel_shuffle(
    b: &Builder,
    x: Traced,
    grid: usize,
    embed: usize,
    scale: usize,
) -> Result<Traced, TraceError> {
    let rows = b.aval(x)?.shape[0];
    let s = scale;
    // Regroup each row of the grid, `[w, embed]`, into `[w / s, embed * s]`.
    let x = b.reshape(x, vec![rows, grid, grid / s, embed * s])?;
    let x = b.transpose(x, vec![0, 2, 1, 3])?;
    // Then regroup each column the same way and bring the grid axes back to reading order.
    let x = b.reshape(x, vec![rows, grid / s, grid / s, embed * s * s])?;
    let x = b.transpose(x, vec![0, 2, 1, 3])?;
    Ok(b.reshape(x, vec![rows, grid * grid / (s * s), embed * s * s])?)
}

impl Model for Idefics3Vision {
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
        if shape.tokens.get() != p.patches() {
            return Err(TraceError::ShapeUnsupported {
                family: FAMILY,
                shape,
                reason: ShapeReason::CapacityAboveMax { max: p.patches() },
            });
        }
        let rows = shape.rows.get();
        let (c, s, patch) = (p.channels, p.image_size, p.patch_size);
        let grid = s / patch;
        let pixels = b.slot_named(
            Slot::Activation,
            PIXELS_TAG,
            TensorType::f32(vec![rows, c, s, s]),
        )?;
        // The strided convolution as an im2col: each patch flattened in `[channel, i, j]` order, the
        // convolution weight's own layout, then one projection.
        let split = b.reshape(pixels, vec![rows, c, grid, patch, grid, patch])?;
        let patches = b.reshape(
            b.transpose(split, vec![0, 2, 4, 1, 3, 5])?,
            vec![rows, grid * grid, c * patch * patch],
        )?;
        let conv = b.reshape(
            self.patch.declare_f32(&b)?,
            vec![p.width, c * patch * patch],
        )?;
        let embedded = b.matmul(patches, b.transpose(conv, vec![1, 0])?)?;
        let embedded = b.binary(
            BinOp::Add,
            embedded,
            b.broadcast(
                self.patch_bias.declare_f32(&b)?,
                vec![rows, grid * grid, p.width],
            )?,
        )?;
        let positions = b.broadcast(
            self.positions.declare_f32(&b)?,
            vec![rows, grid * grid, p.width],
        )?;
        let mut x = b.binary(BinOp::Add, embedded, positions)?;
        for (l, layer) in self.layers.iter().enumerate() {
            let _scope = b.layer_scope(l);
            let attn = bidirectional_attention(
                &b,
                norm_with(&b, x, &layer.norms.attn, p.norm)?,
                [&layer.q, &layer.k, &layer.v, &layer.o],
                (p.heads, p.head_dim),
                None,
            )?;
            x = b.binary(BinOp::Add, x, attn)?;
            let ffn = plain_mlp(
                &b,
                norm_with(&b, x, &layer.norms.ffn, p.norm)?,
                &layer.ffn,
                p.gelu,
            )?;
            x = b.binary(BinOp::Add, x, ffn)?;
        }
        let features = norm_with(&b, x, &self.post_norm, p.norm)?;
        let shuffled = pixel_shuffle(&b, features, grid, p.width, p.scale)?;
        let visual = linear(&b, shuffled, &self.connector, None)?;
        debug_assert_eq!(
            b.aval(visual)?.shape,
            vec![rows, p.visual_tokens(), p.text_width]
        );
        // The tower is stateless: the graph carries nothing to the next step.
        Ok(b.finish(visual)?)
    }
}

/// The fixture's `vision_config` plus the two facts the family reads from the top level: 8-pixel
/// frames of 2-pixel patches (16 patches), a 2x2 pixel shuffle (4 visual tokens) and a 24-wide text tower.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "idefics3_vision",
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "image_size": 8,
        "patch_size": 2,
        "num_channels": 3,
        "layer_norm_eps": 1e-6,
        "hidden_act": "gelu_pytorch_tanh",
        "scale_factor": 2,
        "text_hidden_size": 24,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let n = |f: &str| crate::registry::Fixture::count(&config, f);
    let (w, inter, c, patch) = (
        n("hidden_size"),
        n("intermediate_size"),
        n("num_channels"),
        n("patch_size"),
    );
    let (patches, scale, text) = (
        (n("image_size") / patch).pow(2),
        n("scale_factor"),
        n("text_hidden_size"),
    );
    let mut shapes = vec![
        (
            "model.vision_model.embeddings.patch_embedding.weight".to_string(),
            vec![w, c, patch, patch],
        ),
        (
            "model.vision_model.embeddings.patch_embedding.bias".to_string(),
            vec![w],
        ),
        (
            "model.vision_model.embeddings.position_embedding.weight".to_string(),
            vec![patches, w],
        ),
        (
            "model.vision_model.post_layernorm.weight".to_string(),
            vec![w],
        ),
        (
            "model.vision_model.post_layernorm.bias".to_string(),
            vec![w],
        ),
        (
            "model.connector.modality_projection.proj.weight".to_string(),
            vec![text, w * scale * scale],
        ),
    ];
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| format!("model.vision_model.encoder.layers.{l}.{s}");
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            shapes.push((k(&format!("self_attn.{proj}.weight")), vec![w, w]));
            shapes.push((k(&format!("self_attn.{proj}.bias")), vec![w]));
        }
        shapes.extend([
            (k("layer_norm1.weight"), vec![w]),
            (k("layer_norm1.bias"), vec![w]),
            (k("layer_norm2.weight"), vec![w]),
            (k("layer_norm2.bias"), vec![w]),
            (k("mlp.fc1.weight"), vec![inter, w]),
            (k("mlp.fc1.bias"), vec![inter]),
            (k("mlp.fc2.weight"), vec![w, inter]),
            (k("mlp.fc2.bias"), vec![w]),
        ]);
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_quant::weights::WeightStore;

    use super::*;
    use crate::architectures::bert::reference::{dense, layer_norm};
    use crate::components::standard::oracle::{self, assert_matches_f64};
    use crate::components::testing::{build as built, values};
    use crate::io::ProgramId;
    use crate::model::{LogitRows, StepShape, StepSpec};

    fn get(store: &WeightStore, name: &str) -> Vec<f64> {
        values(store.get(name).unwrap_or_else(|| panic!("{name}")))
            .into_iter()
            .map(f64::from)
            .collect()
    }

    /// The tanh GELU (`gelu_pytorch_tanh`).
    fn gelu(x: f64) -> f64 {
        0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// One frame through the tower, written from the reference modeling code in f64 from the
    /// checkpoint's own values: `[visual_tokens, text_width]`.
    fn reference(store: &WeightStore, pixels: &[f32]) -> Vec<f64> {
        let p = Idefics3Params::from_raw(&fixture().raw()).unwrap();
        let (w, c, patch, grid) = (
            p.width,
            p.channels,
            p.patch_size,
            p.image_size / p.patch_size,
        );
        let np = grid * grid;
        let v = |s: &str| get(store, &format!("model.vision_model.{s}"));
        let (conv, conv_bias, pos) = (
            v("embeddings.patch_embedding.weight"),
            v("embeddings.patch_embedding.bias"),
            v("embeddings.position_embedding.weight"),
        );
        // Each patch flattened in `[channel, i, j]` order, the convolution weight's layout.
        let patches: Vec<f64> = (0..np)
            .flat_map(|n| {
                let (gy, gx) = (n / grid, n % grid);
                (0..c).flat_map(move |ch| {
                    (0..patch).flat_map(move |i| {
                        (0..patch).map(move |j| {
                            f64::from(
                                pixels[(ch * p.image_size + gy * patch + i) * p.image_size
                                    + gx * patch
                                    + j],
                            )
                        })
                    })
                })
            })
            .collect();
        let mut x = dense(&patches, np, &conv, &conv_bias);
        for (xi, pi) in x.iter_mut().zip(&pos) {
            *xi += pi;
        }
        let eps = 1e-6;
        for l in 0..p.layers {
            let k = |s: &str| v(&format!("encoder.layers.{l}.{s}"));
            let normed = layer_norm(&x, w, &k("layer_norm1.weight"), &k("layer_norm1.bias"), eps);
            let proj = |name: &str| {
                dense(
                    &normed,
                    np,
                    &k(&format!("self_attn.{name}.weight")),
                    &k(&format!("self_attn.{name}.bias")),
                )
            };
            let (q, key, value) = (proj("q_proj"), proj("k_proj"), proj("v_proj"));
            let mut merged = vec![0.0; np * w];
            for h in 0..p.heads {
                for i in 0..np {
                    let scores: Vec<f64> = (0..np)
                        .map(|j| {
                            (0..p.head_dim)
                                .map(|d| {
                                    q[i * w + h * p.head_dim + d] * key[j * w + h * p.head_dim + d]
                                })
                                .sum::<f64>()
                                / (p.head_dim as f64).sqrt()
                        })
                        .collect();
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = exp.iter().sum();
                    for d in 0..p.head_dim {
                        merged[i * w + h * p.head_dim + d] = (0..np)
                            .map(|j| exp[j] / sum * value[j * w + h * p.head_dim + d])
                            .sum();
                    }
                }
            }
            let attn = dense(
                &merged,
                np,
                &k("self_attn.out_proj.weight"),
                &k("self_attn.out_proj.bias"),
            );
            for (xi, a) in x.iter_mut().zip(&attn) {
                *xi += a;
            }
            let normed = layer_norm(&x, w, &k("layer_norm2.weight"), &k("layer_norm2.bias"), eps);
            let up = dense(&normed, np, &k("mlp.fc1.weight"), &k("mlp.fc1.bias"));
            let act: Vec<f64> = up.iter().map(|&u| gelu(u)).collect();
            let down = dense(&act, np, &k("mlp.fc2.weight"), &k("mlp.fc2.bias"));
            for (xi, d) in x.iter_mut().zip(&down) {
                *xi += d;
            }
        }
        let features = layer_norm(
            &x,
            w,
            &v("post_layernorm.weight"),
            &v("post_layernorm.bias"),
            eps,
        );
        // Pixel shuffle by definition: output token `(yb, wb)` concatenates the patches
        // `(yb * s + di, wb * s + dj)` in `(di, dj)` order.
        let s = p.scale;
        let tokens = grid / s;
        let shuffled: Vec<f64> = (0..tokens * tokens)
            .flat_map(|t| {
                let (yb, wb) = (t / tokens, t % tokens);
                let features = &features;
                (0..s * s).flat_map(move |d| {
                    let (di, dj) = (d / s, d % s);
                    let patch = (yb * s + di) * grid + wb * s + dj;
                    features[patch * w..(patch + 1) * w].to_vec()
                })
            })
            .collect();
        dense(
            &shuffled,
            tokens * tokens,
            &get(store, "model.connector.modality_projection.proj.weight"),
            &vec![0.0; p.text_width],
        )
    }

    fn encode_shape(rows: usize, patches: usize) -> StepShape {
        StepShape::new(
            NonZeroUsize::new(rows).unwrap(),
            NonZeroUsize::new(patches).unwrap(),
            NonZeroUsize::new(patches).unwrap(),
        )
        .with_logits(LogitRows::All)
    }

    fn frames(rows: usize, seed: u64) -> Vec<f32> {
        oracle::random(rows * 3 * 8 * 8, seed)
    }

    fn run(model: &dyn Model, store: &WeightStore, rows: usize, pixels: &[f32]) -> Vec<f32> {
        let g = model
            .trace(
                &StepSpec::encode(encode_shape(rows, 16)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        oracle::eval_slots(
            &g,
            store,
            model.weights(),
            &[],
            &[],
            (&[], &[("activation.pixels", pixels)]),
            &[],
        )
        .0
    }

    /// SC-002 (model half): two frames in one step equal an f64 forward of each frame alone, and the
    /// two frames' embedding rows differ: a frame's content is a bound input, not a constant the
    /// first frame fixed. Mutation: bind `pixels` as a const (the second frame reads the first's
    /// pixels); this goes red.
    #[test]
    fn each_frame_equals_an_f64_forward_of_that_frame_alone() {
        let fx = fixture();
        let model = built(&ENTRY, &fx.raw(), &fx.store);
        let pixels = frames(2, 7);
        let got = run(&*model, &fx.store, 2, &pixels);
        let per_frame = 3 * 8 * 8;
        let out = 4 * 24;
        assert_eq!(got.len(), 2 * out);
        for f in 0..2 {
            assert_matches_f64(
                &got[f * out..(f + 1) * out],
                &reference(&fx.store, &pixels[f * per_frame..(f + 1) * per_frame]),
                1e-4,
            );
        }
        assert_ne!(got[..out], got[out..], "two different frames embed alike");
    }

    /// Anything but a frame of the tower's own patch count, and any program but `encode`, is a typed
    /// refusal.
    #[test]
    fn a_shape_or_phase_it_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = built(&ENTRY, &fx.raw(), &fx.store);
        assert!(matches!(
            model.trace(
                &StepSpec::encode(encode_shape(1, 15)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported { .. })
        ));
        assert!(matches!(
            model.trace(
                &StepSpec::step(encode_shape(1, 16)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ProgramUnsupported {
                program: ProgramId::STEP,
                ..
            })
        ));
    }

    /// A scale factor that does not divide the patch grid is a config error naming the field.
    #[test]
    fn a_scale_that_does_not_divide_the_grid_is_a_config_error() {
        let fx = fixture();
        let mut config = fixture_config();
        config["scale_factor"] = serde_json::json!(3);
        let raw = RawConfig::HfJson {
            config: &config,
            generation: None,
        };
        let err = build(&raw, &fx.store).unwrap_err();
        assert!(err.to_string().contains("scale_factor"), "{err}");
    }
}
