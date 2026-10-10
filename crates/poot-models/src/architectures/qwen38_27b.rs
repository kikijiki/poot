//! Qwen3.5 (HF `qwen3_5` and `qwen3_5_text`; the exact family `Qwen/Qwen3.8-27B`) behind [`Model`]: a
//! hybrid stack whose mixer is a Gated DeltaNet linear-attention layer ([`gdn`]) on most layers and
//! gated full attention (`components::attention` with the output gate) on every
//! `full_attention_interval`-th, each followed by the same dense gated feed-forward.
//!
//! The body is `standard_stack` over `standard_layer`: `x += mixer(norm(x)); x += ffn(norm(x))`, the
//! Qwen3-Next body with a dense block where Qwen3-Next routes. The step carries the attention layers'
//! KV caches and, for each GDN layer, its conv window and its state matrices, so a prompt prefilled in
//! chunks continues exactly where the previous chunk stopped and one token is a decode step.
//!
//! The family's checkpoint spells the layer as `modeling_qwen3_5.py` does: one projection per GDN
//! quantity (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`), value heads grouped by key head,
//! `A_log` and zero-centered norms (`Qwen3_5RMSNorm` scales by `1 + weight`; only the GDN's gated norm
//! scales by its stored weight). The multimodal wrapper nests the text tower under `text_config` and
//! stores it under `model.language_model.`; a text-only `qwen3_5_text` checkpoint stores it under
//! `model.`. A projection is stored dense or packed (the FP8 checkpoint's `E4m3Block128` codes and
//! scales), and the one tracer serves both: storage is the weight handle's format descriptor.
//!
//! Every dimension is config data, so any checkpoint of the architecture loads (a finetune with
//! fewer layers, another width), with or without an integrity manifest.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use std::sync::Arc;

use poot_graph_ir::op::{PackedWeight, ScaleEncoding, WeightFormat};
use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{
    AttnRole, FfnRole, GdnRole, NormRole, WeightEntry, WeightId, WeightKey, WeightMap,
    WeightMapError, WeightRole, WeightStore, WeightView,
};
use poot_quant::{OperandRole, PackedPayload, SourceRole};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::gdn::{GdnHeadOrder, GdnParams, GdnWeights, gdn};
use crate::components::linear::WeightError;
use crate::components::norm::{NormKind, NormParams};
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step,
    standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, FixtureVariant, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("qwen35");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "qwen3_5"),
        (ConfigSource::HfModelType, "qwen3_5_text"),
    ],
    build,
    fixture,
    variants: &[VARIANT_TEXT, VARIANT_FP8, VARIANT_FINETUNE],
};

/// The multimodal wrapper's `model_type`: its text tower sits under `text_config`.
const WRAPPER_TYPE: &str = "qwen3_5";

/// Qwen3.5's config facts: `Qwen3_5TextConfig`'s defaults. The class declares no `eos_token_id`: a
/// checkpoint names it. The GGUF keys are unread: only an HF config names this family.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("qwen35"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000.0,
    eps: 1e-6,
    eos: None,
};

const KEY_HEADS: Field = Field::hf("linear_num_key_heads");
const VALUE_HEADS: Field = Field::hf("linear_num_value_heads");
const KEY_HEAD_DIM: Field = Field::hf("linear_key_head_dim");
const VALUE_HEAD_DIM: Field = Field::hf("linear_value_head_dim");
const CONV_KERNEL: Field = Field::hf("linear_conv_kernel_dim");
const INTERVAL: Field = Field::hf("full_attention_interval");

/// One layer's mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet linear attention.
    LinearAttention,
    /// Gated full attention.
    FullAttention,
}

/// Where a checkpoint stores the text tower: the layer prefix and the model-level tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Names {
    layers: &'static str,
    embed: &'static str,
    final_norm: &'static str,
}

const WRAPPED_NAMES: Names = Names {
    layers: "model.language_model.layers.",
    embed: "model.language_model.embed_tokens.weight",
    final_norm: "model.language_model.norm.weight",
};

const TEXT_NAMES: Names = Names {
    layers: "model.layers.",
    embed: "model.embed_tokens.weight",
    final_norm: "model.norm.weight",
};

/// Qwen3.5's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35Params {
    vocab: usize,
    width: usize,
    inter: usize,
    max_positions: usize,
    kinds: Vec<LayerKind>,
    layer: LayerParams,
    stack: StackParams,
    attention: AttentionParams,
    gdn: GdnParams,
    names: Names,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

/// The text tower's config of an HF config: the wrapper's `text_config` (with the token ids and the
/// head tie the wrapper carries), or the config itself of a text-only checkpoint.
fn text_config(config: &serde_json::Value) -> Result<(serde_json::Value, Names), ModelError> {
    let wrong = |reason| ModelError::Config {
        family: FAMILY,
        field: "text_config",
        reason,
    };
    if config.get("model_type").and_then(|v| v.as_str()) != Some(WRAPPER_TYPE) {
        return Ok((config.clone(), TEXT_NAMES));
    }
    let mut text = config
        .get("text_config")
        .filter(|v| !v.is_null())
        .ok_or(wrong(ConfigReason::Missing))?
        .clone();
    let object = text.as_object_mut().ok_or(wrong(ConfigReason::WrongType))?;
    for key in ["eos_token_id", "bos_token_id", "tie_word_embeddings"] {
        if let Some(value) = config.get(key).filter(|v| !v.is_null()) {
            object.insert(key.to_string(), value.clone());
        }
    }
    Ok((text, WRAPPED_NAMES))
}

/// The fraction of each head the rotation covers: `rope_parameters.partial_rotary_factor`, or the
/// config's own, default 0.25.
fn rotary_factor(fields: &Fields<'_>, text: &serde_json::Value) -> Result<f32, ModelError> {
    const KEY: &str = "partial_rotary_factor";
    let value = text
        .get("rope_parameters")
        .and_then(|p| p.get(KEY))
        .or_else(|| text.get(KEY))
        .filter(|v| !v.is_null());
    let factor = match value {
        None => 0.25,
        Some(v) => v
            .as_f64()
            .map(|v| v as f32)
            .ok_or(fields.error(KEY, ConfigReason::WrongType))?,
    };
    if !(factor.is_finite() && factor > 0.0 && factor <= 1.0) {
        return Err(fields.error(KEY, ConfigReason::NotFinitePositive));
    }
    Ok(factor)
}

/// Which layers are full attention: `layer_types` names them (`"full_attention"` or
/// `"linear_attention"` per layer), or `full_attention_interval` (default 4) does, where layer `i` is
/// full attention iff `(i + 1)` is a multiple of the interval.
fn layer_kinds(
    fields: &Fields<'_>,
    text: &serde_json::Value,
    layers: usize,
) -> Result<Vec<LayerKind>, ModelError> {
    if let Some(types) = text.get("layer_types").filter(|v| !v.is_null()) {
        let wrong = || fields.error("layer_types", ConfigReason::WrongType);
        let types = types.as_array().ok_or_else(wrong)?;
        if types.len() != layers {
            return Err(fields.error("layer_types", ConfigReason::Length { expected: layers }));
        }
        return types
            .iter()
            .map(|t| match t.as_str().ok_or_else(wrong)? {
                "full_attention" => Ok(LayerKind::FullAttention),
                "linear_attention" => Ok(LayerKind::LinearAttention),
                _ => Err(fields.error("layer_types", ConfigReason::Unsupported)),
            })
            .collect();
    }
    let interval = match fields.opt_u64(INTERVAL)? {
        None => 4,
        Some(0) => return Err(fields.error(fields.key(INTERVAL), ConfigReason::Zero)),
        Some(n) => usize::try_from(n)
            .map_err(|_| fields.error(fields.key(INTERVAL), ConfigReason::WrongType))?,
    };
    Ok((0..layers)
        .map(|l| {
            if (l + 1).is_multiple_of(interval) {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            }
        })
        .collect())
}

impl Qwen35Params {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let RawConfig::HfJson { config, generation } = raw else {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "general.architecture",
                reason: ConfigReason::Unsupported,
            });
        };
        let (text, names) = text_config(config)?;
        let raw = RawConfig::HfJson {
            config: &text,
            generation: *generation,
        };
        let fields = Fields::new(&raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        for unsupported in ["attention_bias", "use_sliding_window"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        // The full-attention layers' output gate is the architecture: a config without it is another
        // model.
        if !fields.flag(Field::hf("attn_output_gate"), true)? {
            return Err(fields.error("attn_output_gate", ConfigReason::Unsupported));
        }
        let eps = dims.eps;
        let eps_error = |_| fields.error("rms_norm_eps", ConfigReason::NotFinitePositive);
        let norm = NormParams::of(NormKind::RmsPlusOne, eps).map_err(eps_error)?;
        let rotary = ((dims.head_dim as f32 * rotary_factor(&fields, &text)?) as usize).max(1);
        let attention = AttentionParams::new(
            dims.heads,
            dims.kv_heads,
            dims.head_dim,
            false,
            dims.rope(&fields, &SPEC, rotary)?,
        )
        .and_then(|a| a.with_output_gate())
        .map_err(|e| e.for_family(FAMILY))?
        .with_qk_norm(QkNorm::PerHead(norm));

        // The GDN's key and value heads share one width.
        let head_dim = fields.count(KEY_HEAD_DIM, None)?;
        if fields.count(VALUE_HEAD_DIM, Some(head_dim))? != head_dim {
            return Err(fields.error("linear_value_head_dim", ConfigReason::Unsupported));
        }
        let gdn = GdnParams::new(
            fields.count(KEY_HEADS, None)?,
            fields.count(VALUE_HEADS, None)?,
            head_dim,
            fields.count(CONV_KERNEL, None)?,
            NormParams::new(eps).map_err(eps_error)?,
            GdnHeadOrder::Grouped,
        )
        .map_err(|e| e.for_family(FAMILY))?
        .with_log_decay();

        Ok(Self {
            kinds: layer_kinds(&fields, &text, dims.layers)?,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            attention,
            gdn,
            names,
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            max_positions: dims.max_positions,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
        })
    }
}

/// The weight map of a checkpoint: the model-level weights and, per layer, the two norms, the
/// mixer's weights its kind names and the dense feed-forward.
fn weights_of(p: &Qwen35Params, store: &WeightStore) -> Result<WeightMap, ModelError> {
    let names = &p.names;
    let mut map = WeightMap::builder(store);
    let missing = |source: WeightMapError| ModelError::Weight {
        family: FAMILY,
        source,
    };
    // The stored key of a linear: `{stem}.weight`, dense or the packed payload the loader publishes
    // under it.
    let linear = |stem: &str| format!("{stem}.weight");
    let mut put = |id: WeightId, key: String| -> Result<(), ModelError> {
        map.map(id, WeightView::Stored(WeightKey::from(key)))
            .map_err(missing)?;
        Ok(())
    };

    let embed = WeightId::model(WeightRole::Embed);
    put(embed, names.embed.to_string())?;
    put(
        WeightId::model(WeightRole::FinalNorm),
        names.final_norm.to_string(),
    )?;
    let head = WeightId::model(WeightRole::Head);
    let head_key = linear("lm_head");
    if store.contains(&head_key) {
        put(head, head_key)?;
    } else if p.head_required {
        return Err(missing(WeightMapError::Missing {
            id: head,
            key: WeightKey::from(head_key),
        }));
    } else {
        put(head, names.embed.to_string())?;
    }

    for (l, kind) in p.kinds.iter().enumerate() {
        let layer = |suffix: &str| format!("{}{l}{suffix}", names.layers);
        let id = |role| WeightId::layer(l, role);
        put(
            id(WeightRole::Norm(NormRole::Attn)),
            layer(".input_layernorm.weight"),
        )?;
        put(
            id(WeightRole::Norm(NormRole::Ffn)),
            layer(".post_attention_layernorm.weight"),
        )?;
        match kind {
            LayerKind::FullAttention => {
                for (role, stem) in [
                    (AttnRole::Q, ".self_attn.q_proj"),
                    (AttnRole::K, ".self_attn.k_proj"),
                    (AttnRole::V, ".self_attn.v_proj"),
                    (AttnRole::O, ".self_attn.o_proj"),
                ] {
                    put(id(WeightRole::Attn(role)), linear(&layer(stem)))?;
                }
                for (role, suffix) in [
                    (AttnRole::QNorm, ".self_attn.q_norm.weight"),
                    (AttnRole::KNorm, ".self_attn.k_norm.weight"),
                ] {
                    put(id(WeightRole::Attn(role)), layer(suffix))?;
                }
            }
            LayerKind::LinearAttention => {
                for (role, stem) in [
                    (GdnRole::Qkv, ".linear_attn.in_proj_qkv"),
                    (GdnRole::Z, ".linear_attn.in_proj_z"),
                    (GdnRole::Beta, ".linear_attn.in_proj_b"),
                    (GdnRole::Alpha, ".linear_attn.in_proj_a"),
                    (GdnRole::Out, ".linear_attn.out_proj"),
                ] {
                    put(id(WeightRole::Gdn(role)), linear(&layer(stem)))?;
                }
                for (role, suffix) in [
                    (GdnRole::DtBias, ".linear_attn.dt_bias"),
                    (GdnRole::Conv, ".linear_attn.conv1d.weight"),
                    (GdnRole::A, ".linear_attn.A_log"),
                    (GdnRole::GateNorm, ".linear_attn.norm.weight"),
                ] {
                    put(id(WeightRole::Gdn(role)), layer(suffix))?;
                }
            }
        }
        for (role, stem) in [
            (FfnRole::Gate, ".mlp.gate_proj"),
            (FfnRole::Up, ".mlp.up_proj"),
            (FfnRole::Down, ".mlp.down_proj"),
        ] {
            put(id(WeightRole::Ffn(role)), linear(&layer(stem)))?;
        }
    }
    Ok(map.build())
}

#[derive(Debug)]
enum Mixer {
    Gdn(Box<GdnWeights>),
    Attention(Box<AttentionWeights>),
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    mixer: Mixer,
    ffn: GatedFfnWeights,
}

#[derive(Debug)]
pub struct Qwen35 {
    params: Qwen35Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Qwen35Params::from_raw(raw)?;
    let weights = weights_of(&p, store)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers =
        p.kinds
            .iter()
            .enumerate()
            .map(|(l, kind)| {
                let mixer =
                    match kind {
                        LayerKind::LinearAttention => {
                            Mixer::Gdn(Box::new(GdnWeights::new(&weights, l, p.width, &p.gdn)?))
                        }
                        LayerKind::FullAttention => Mixer::Attention(Box::new(
                            AttentionWeights::new(&weights, l, p.width, &p.attention)?,
                        )),
                    };
                Ok(Layer {
                    norms: LayerNorms::new(&weights, l, p.width)?,
                    mixer,
                    ffn: GatedFfnWeights::new(&weights, l, p.width, p.inter)?,
                })
            })
            .collect::<Result<Vec<_>, WeightError>>()
            .map_err(weight_error)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(Qwen35 {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Qwen35 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace(
        &self,
        phase: Phase,
        shape: StepShape,
    ) -> Result<Graph<ValidationOutputs>, TraceError> {
        let p = &self.params;
        check_step(FAMILY, phase, shape, p.max_positions)?;
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let logits = standard_stack(
            &b,
            &step,
            &self.stack,
            &p.stack,
            p.kinds.len(),
            |b, l, x| {
                let layer = &self.layers[l];
                standard_layer(
                    b,
                    x,
                    &layer.norms,
                    &p.layer,
                    |b, h| match &layer.mixer {
                        Mixer::Gdn(w) => gdn(b, &step, l, h, w, &p.gdn),
                        Mixer::Attention(w) => attention(b, &step, l, h, w, &p.attention),
                    },
                    |b, BlockInput { block: h, .. }| gated_ffn(b, h, &layer.ffn),
                )
            },
        );
        Ok(step.finish(b, logits))
    }
}

/// The fixture's text config: a four-layer hybrid (three GDN layers, then gated attention) over a
/// dense feed-forward, GQA on both mixers (`value / key` heads 2 for the GDN, `heads / kv_heads` 2 for
/// the attention), a partial rotation and the driver fixtures' 48-word vocabulary. Projection widths are multiples of 32 where a Q8_0 copy needs them.
pub(crate) fn fixture_text_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5_text",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "attn_output_gate": true,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5,
        },
        "tie_word_embeddings": false,
        "eos_token_id": 47,
    })
}

/// The multimodal wrapper of [`fixture_text_config`]: the text tower under `text_config`, the token ids
/// and the head tie on the wrapper.
pub(crate) fn fixture_config() -> serde_json::Value {
    let mut text = fixture_text_config();
    let object = text.as_object_mut().unwrap();
    object.insert("model_type".into(), WRAPPER_TYPE.into());
    let mut wrapper = serde_json::Map::new();
    for key in ["eos_token_id", "tie_word_embeddings"] {
        wrapper.insert(key.into(), object.remove(key).unwrap());
    }
    wrapper.insert("model_type".into(), WRAPPER_TYPE.into());
    wrapper.insert("text_config".into(), text);
    serde_json::Value::Object(wrapper)
}

/// The tensors of a Qwen3.5 text tower of `text`'s dims under `root` (`model.language_model` or
/// `model`), by HF name, and its `lm_head.weight`.
pub(crate) fn hf_shapes(text: &serde_json::Value, root: &str) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| text[f].as_u64().unwrap() as usize;
    let (vocab, w, inter) = (n("vocab_size"), n("hidden_size"), n("intermediate_size"));
    let (heads, kv, hd) = (
        n("num_attention_heads"),
        n("num_key_value_heads"),
        n("head_dim"),
    );
    let (hk, hv, d, kernel) = (
        n("linear_num_key_heads"),
        n("linear_num_value_heads"),
        n("linear_key_head_dim"),
        n("linear_conv_kernel_dim"),
    );
    let (key_dim, value_dim) = (hk * d, hv * d);
    let conv = 2 * key_dim + value_dim;
    let mut shapes = vec![
        (format!("{root}.embed_tokens.weight"), vec![vocab, w]),
        (format!("{root}.norm.weight"), vec![w]),
        ("lm_head.weight".to_string(), vec![vocab, w]),
    ];
    for (l, kind) in text["layer_types"].as_array().unwrap().iter().enumerate() {
        let k = |s: &str| format!("{root}.layers.{l}.{s}");
        shapes.push((k("input_layernorm.weight"), vec![w]));
        shapes.push((k("post_attention_layernorm.weight"), vec![w]));
        if kind == "full_attention" {
            shapes.extend([
                (k("self_attn.q_proj.weight"), vec![2 * heads * hd, w]),
                (k("self_attn.k_proj.weight"), vec![kv * hd, w]),
                (k("self_attn.v_proj.weight"), vec![kv * hd, w]),
                (k("self_attn.o_proj.weight"), vec![w, heads * hd]),
                (k("self_attn.q_norm.weight"), vec![hd]),
                (k("self_attn.k_norm.weight"), vec![hd]),
            ]);
        } else {
            shapes.extend([
                (k("linear_attn.in_proj_qkv.weight"), vec![conv, w]),
                (k("linear_attn.in_proj_z.weight"), vec![value_dim, w]),
                (k("linear_attn.in_proj_b.weight"), vec![hv, w]),
                (k("linear_attn.in_proj_a.weight"), vec![hv, w]),
                (k("linear_attn.conv1d.weight"), vec![conv, 1, kernel]),
                (k("linear_attn.dt_bias"), vec![hv]),
                (k("linear_attn.A_log"), vec![hv]),
                (k("linear_attn.norm.weight"), vec![d]),
                (k("linear_attn.out_proj.weight"), vec![w, value_dim]),
            ]);
        }
        shapes.extend([
            (k("mlp.gate_proj.weight"), vec![inter, w]),
            (k("mlp.up_proj.weight"), vec![inter, w]),
            (k("mlp.down_proj.weight"), vec![w, inter]),
        ]);
    }
    shapes
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_shapes(&config["text_config"], "model.language_model");
    Fixture::bf16(config, shapes)
}

/// The text-only checkpoint (`qwen3_5_text`): the same tower stored under `model.`.
fn fixture_text() -> Fixture {
    let config = fixture_text_config();
    let shapes = hf_shapes(&config, "model");
    Fixture::bf16(config, shapes)
}

// ---- the FP8 checkpoint -------------------------------------------------------------------------

/// Finite E4M3FN codes with mixed signs and magnitudes.
const E4M3_CODES: [u8; 9] = [0x00, 0x20, 0x28, 0x2c, 0x30, 0x34, 0xa4, 0xa8, 0xb0];

/// SplitMix64 finalizer: spreads a seed so neighboring inputs give unrelated outputs.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A packed `E4m3Block128` payload of `shape` whose codes and scale come from `stem`: seeding by
/// the checkpoint name gives same-shaped projections (`k_proj`/`v_proj`, `gate_proj`/`up_proj`)
/// different weights, so a projection bound under the wrong name changes the logits.
fn fp8_payload(stem: &str, shape: [usize; 2]) -> PackedPayload {
    let seed = stem.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    let codes = (0..shape[0] * shape[1])
        .map(|element| E4M3_CODES[(mix64(seed ^ element as u64) % 9) as usize])
        .collect::<Vec<u8>>();
    let scale = [0.75f32, 1.25][(mix64(seed) % 2) as usize];
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        shape,
    )
    .unwrap();
    let bf16 = ((scale.to_bits() >> 16) as u16).to_le_bytes();
    let scale_bytes = descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale));
    let scales = bf16
        .iter()
        .copied()
        .cycle()
        .take(scale_bytes)
        .collect::<Vec<u8>>();
    PackedPayload::try_new(
        descriptor,
        [
            (SourceRole::Planar(OperandRole::Codes), codes.into()),
            (SourceRole::Planar(OperandRole::Scale), scales.into()),
        ],
    )
    .unwrap()
}

/// Whether `key` is one of the checkpoint's quantized linears: the FP8 checkpoint quantizes every
/// text projection and keeps the small GDN gates (`in_proj_a`, `in_proj_b`), the norms, the
/// embedding and the head dense.
fn is_fp8_linear(key: &str) -> bool {
    [
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    .iter()
    .any(|stem| key.ends_with(&format!(".{stem}.weight")))
}

/// `fx`'s store with its quantized linears replaced by packed FP8 payloads under their `.weight` keys, as
/// the loader publishes them.
fn fp8_store(fx: &Fixture) -> WeightStore {
    let mut out = WeightStore::builder();
    for (key, entry) in fx.store.iter() {
        if is_fp8_linear(key.as_str()) {
            let stem = key.as_str().strip_suffix(".weight").unwrap();
            let [rows, k] = entry.shape()[..] else {
                panic!("{key} is not a matrix")
            };
            out.insert(
                key.clone(),
                WeightEntry::Packed(Arc::new(fp8_payload(stem, [rows, k]))),
            )
            .unwrap();
        } else {
            out.insert(key.clone(), entry.clone()).unwrap();
        }
    }
    out.build()
}

/// The fixture as the FP8 checkpoint stores it (`Qwen/Qwen3.8-27B-FP8`): every text projection a packed
/// `E4m3Block128` payload under its `.weight` key, the rest BF16.
fn fixture_fp8() -> Fixture {
    let mut fx = fixture();
    fx.store = fp8_store(&fx);
    fx
}

const VARIANT_FP8: FixtureVariant = ("fp8", fixture_fp8);

const VARIANT_TEXT: FixtureVariant = ("text", fixture_text);

/// A fine-tune of the fixture's architecture with other counts than any pinned release: two layers
/// (one GDN, one attention), other widths and head counts. It ships no manifest and is the same family.
fn finetune_fixture() -> Fixture {
    let mut config = fixture_config();
    let text = config["text_config"].as_object_mut().unwrap();
    for (key, value) in [
        ("num_hidden_layers", serde_json::json!(2)),
        (
            "layer_types",
            serde_json::json!(["linear_attention", "full_attention"]),
        ),
        ("hidden_size", serde_json::json!(48)),
        ("intermediate_size", serde_json::json!(80)),
        ("num_attention_heads", serde_json::json!(6)),
        ("num_key_value_heads", serde_json::json!(3)),
        ("linear_num_key_heads", serde_json::json!(1)),
        ("linear_num_value_heads", serde_json::json!(3)),
    ] {
        text.insert(key.to_string(), value);
    }
    let shapes = hf_shapes(&config["text_config"], "model.language_model");
    Fixture::bf16(config, shapes)
}

const VARIANT_FINETUNE: FixtureVariant = ("finetune", finetune_fixture);

#[cfg(test)]
mod tests {
    use poot_quant::StoredBytes;
    use poot_quant::weights::HandleFormat;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build as built,
        close, close_all, decoded, greedy_tokens, q8_0_store, step_shape, values,
    };
    use crate::model::LogitRows;
    use crate::registry::Registry;

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// The fixture's text tower and where its checkpoint stores it.
    fn text_of(fx: &Fixture) -> (&serde_json::Value, &'static str) {
        match fx.config.get("text_config") {
            Some(text) => (text, "model.language_model"),
            None => (&fx.config, "model"),
        }
    }

    /// The two checkpoint layouts of the family: the multimodal wrapper and the text-only model.
    fn fixtures() -> [(&'static str, Fixture); 2] {
        [("wrapped", fixture()), ("text", fixture_text())]
    }

    // ---- the independent reference -------------------------------------------------------------------

    /// The checkpoint's dimensions, read from the config JSON by their HF names (no `Qwen35Params`).
    struct Dims {
        hidden: usize,
        full: Vec<bool>,
        eps: f32,
        rotary: usize,
        theta: f32,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_heads: usize,
        value_heads: usize,
        gdn_dim: usize,
        kernel: usize,
    }

    impl Dims {
        fn of(text: &serde_json::Value) -> Self {
            let n = |f: &str| text[f].as_u64().unwrap() as usize;
            let rope = &text["rope_parameters"];
            let head_dim = n("head_dim");
            Self {
                hidden: n("hidden_size"),
                full: text["layer_types"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|k| k == "full_attention")
                    .collect(),
                eps: text["rms_norm_eps"].as_f64().unwrap() as f32,
                rotary: (head_dim as f64 * rope["partial_rotary_factor"].as_f64().unwrap())
                    as usize,
                theta: rope["rope_theta"].as_f64().unwrap() as f32,
                n_heads: n("num_attention_heads"),
                n_kv_heads: n("num_key_value_heads"),
                head_dim,
                key_heads: n("linear_num_key_heads"),
                value_heads: n("linear_num_value_heads"),
                gdn_dim: n("linear_key_head_dim"),
                kernel: n("linear_conv_kernel_dim"),
            }
        }
    }

    fn matvec(weight: &[f32], x: &[f32]) -> Vec<f32> {
        weight
            .chunks_exact(x.len())
            .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
            .collect()
    }

    fn rms_normalize(x: &[f32], eps: f32) -> Vec<f32> {
        let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inverse = (mean_square + eps).sqrt().recip();
        x.iter().map(|v| v * inverse).collect()
    }

    /// `Qwen3_5RMSNorm`: the stored weight is zero-centered, the scale is `1 + weight`.
    fn zero_centered_rms_norm(x: &[f32], stored: &[f32], eps: f32) -> Vec<f32> {
        rms_normalize(x, eps)
            .iter()
            .zip(stored)
            .map(|(v, w)| v * (1.0 + w))
            .collect()
    }

    fn l2_normalize(x: &[f32], eps: f32) -> Vec<f32> {
        let inverse = (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt().recip();
        x.iter().map(|v| v * inverse).collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }

    fn add_assign(target: &mut [f32], update: &[f32]) {
        target.iter_mut().zip(update).for_each(|(t, u)| *t += u);
    }

    enum LayerState {
        /// `conv`: the last `K - 1` raw QKV inputs, oldest first. `recurrent`: `[H_v, D_k, D_v]`.
        Gdn { conv: Vec<f32>, recurrent: Vec<f32> },
        /// `[n_kv_heads, capacity, head_dim]` post-RoPE keys and values.
        Attention { keys: Vec<f32>, values: Vec<f32> },
    }

    /// Token-by-token Qwen3.5 text model written from HF `modeling_qwen3_5.py`
    /// (`Qwen3_5GatedDeltaNet`, `torch_recurrent_gated_delta_rule`, `Qwen3_5RMSNormGated`,
    /// `Qwen3_5Attention`, `apply_rotary_pos_emb`, `Qwen3_5MLP`, `Qwen3_5RMSNorm`). It shares no math
    /// and no name table with the model: tensors are addressed by literal HF names over the checkpoint
    /// store (a packed projection dequantized from its codes and scales), and every operation is a
    /// plain f32 loop.
    struct HfReference<'a> {
        store: &'a WeightStore,
        root: &'a str,
        dims: Dims,
        capacity: usize,
        layers: Vec<LayerState>,
    }

    impl<'a> HfReference<'a> {
        fn new(store: &'a WeightStore, root: &'a str, text: &serde_json::Value) -> Self {
            let dims = Dims::of(text);
            let conv_dim = (2 * dims.key_heads + dims.value_heads) * dims.gdn_dim;
            let capacity = CAP;
            let layers = dims
                .full
                .iter()
                .map(|&full| {
                    if full {
                        let cache = vec![0.0; dims.n_kv_heads * capacity * dims.head_dim];
                        LayerState::Attention {
                            keys: cache.clone(),
                            values: cache,
                        }
                    } else {
                        LayerState::Gdn {
                            conv: vec![0.0; (dims.kernel - 1) * conv_dim],
                            recurrent: vec![0.0; dims.value_heads * dims.gdn_dim * dims.gdn_dim],
                        }
                    }
                })
                .collect();
            Self {
                store,
                root,
                dims,
                capacity,
                layers,
            }
        }

        /// A tensor of the text tower by HF name: dense, or a packed linear under its `.weight` key.
        fn tensor(&self, name: &str) -> Vec<f32> {
            let key = if name == "lm_head" {
                name.to_string()
            } else {
                format!("{}.{name}", self.root)
            };
            let entry = self
                .store
                .get(&format!("{key}.weight"))
                .or_else(|| self.store.get(&key))
                .unwrap_or_else(|| panic!("the store has no {key}"));
            values(entry)
        }

        fn project(&self, name: &str, x: &[f32]) -> Vec<f32> {
            matvec(&self.tensor(name), x)
        }

        /// Run one token and return its logits.
        fn step(&mut self, token: i32, position: usize) -> Vec<f32> {
            let hidden = self.dims.hidden;
            let eps = self.dims.eps;
            let token = usize::try_from(token).expect("token id");
            let embedding = self.tensor("embed_tokens");
            let mut h = embedding[token * hidden..(token + 1) * hidden].to_vec();
            for layer in 0..self.dims.full.len() {
                let root = format!("layers.{layer}");
                let input_norm = self.tensor(&format!("{root}.input_layernorm"));
                let normed = zero_centered_rms_norm(&h, &input_norm, eps);
                let mixer = if self.dims.full[layer] {
                    self.attention(layer, &root, &normed, position)
                } else {
                    self.gated_delta_net(layer, &root, &normed)
                };
                add_assign(&mut h, &mixer);

                let post_norm = self.tensor(&format!("{root}.post_attention_layernorm"));
                let mlp_input = zero_centered_rms_norm(&h, &post_norm, eps);
                let gate = self.project(&format!("{root}.mlp.gate_proj"), &mlp_input);
                let up = self.project(&format!("{root}.mlp.up_proj"), &mlp_input);
                let activated = gate
                    .iter()
                    .zip(&up)
                    .map(|(&gate, &up)| silu(gate) * up)
                    .collect::<Vec<_>>();
                add_assign(
                    &mut h,
                    &self.project(&format!("{root}.mlp.down_proj"), &activated),
                );
            }
            let final_norm = self.tensor("norm");
            let normed = zero_centered_rms_norm(&h, &final_norm, eps);
            self.project("lm_head", &normed)
        }

        fn gated_delta_net(&mut self, layer: usize, root: &str, x: &[f32]) -> Vec<f32> {
            let (key_heads, value_heads) = (self.dims.key_heads, self.dims.value_heads);
            let (d, kernel, eps) = (self.dims.gdn_dim, self.dims.kernel, self.dims.eps);
            let key_dim = key_heads * d;
            let conv_dim = 2 * key_dim + value_heads * d;
            let linear_attn = |suffix: &str| format!("{root}.linear_attn.{suffix}");
            let b = self.project(&linear_attn("in_proj_b"), x);
            let a = self.project(&linear_attn("in_proj_a"), x);
            let dt_bias = self.tensor(&linear_attn("dt_bias"));
            let a_log = self.tensor(&linear_attn("A_log"));
            let conv_weight = self.tensor(&linear_attn("conv1d"));
            let norm_weight = self.tensor(&linear_attn("norm"));
            let qkv = self.project(&linear_attn("in_proj_qkv"), x);
            let z = self.project(&linear_attn("in_proj_z"), x);

            let LayerState::Gdn { conv, recurrent } = &mut self.layers[layer] else {
                panic!("layer {layer} is not a GDN layer");
            };
            // Depthwise causal conv over the last `kernel` inputs, oldest first, then SiLU.
            let mut window = conv.clone();
            window.extend_from_slice(&qkv);
            let mixed = (0..conv_dim)
                .map(|channel| {
                    silu(
                        (0..kernel)
                            .map(|tap| {
                                conv_weight[channel * kernel + tap]
                                    * window[tap * conv_dim + channel]
                            })
                            .sum(),
                    )
                })
                .collect::<Vec<_>>();
            *conv = window[conv_dim..].to_vec();
            let (query, rest) = mixed.split_at(key_dim);
            let (key, value) = rest.split_at(key_dim);

            let n_rep = value_heads / key_heads;
            let scale = (d as f32).sqrt().recip();
            let mut core = Vec::with_capacity(value_heads * d);
            for head in 0..value_heads {
                // `repeat_interleave(n_rep, dim=2)`: value head `head` reads key head `head / n_rep`.
                let source = head / n_rep;
                let q = l2_normalize(&query[source * d..(source + 1) * d], eps);
                let k = l2_normalize(&key[source * d..(source + 1) * d], eps);
                let v = &value[head * d..(head + 1) * d];
                let beta = sigmoid(b[head]);
                let g = -a_log[head].exp() * (a[head] + dt_bias[head]).exp().ln_1p();

                let state = &mut recurrent[head * d * d..(head + 1) * d * d];
                let decay = g.exp();
                state.iter_mut().for_each(|s| *s *= decay);
                let delta = (0..d)
                    .map(|j| (v[j] - (0..d).map(|i| state[i * d + j] * k[i]).sum::<f32>()) * beta)
                    .collect::<Vec<_>>();
                for (row, &k_i) in state.chunks_exact_mut(d).zip(&k) {
                    row.iter_mut()
                        .zip(&delta)
                        .for_each(|(s, delta)| *s += k_i * delta);
                }
                let out = (0..d)
                    .map(|j| (0..d).map(|i| state[i * d + j] * q[i] * scale).sum())
                    .collect::<Vec<f32>>();

                // `Qwen3_5RMSNormGated`: stored weight, then the SiLU(z) gate.
                let normed = rms_normalize(&out, eps);
                core.extend((0..d).map(|j| normed[j] * norm_weight[j] * silu(z[head * d + j])));
            }
            self.project(&linear_attn("out_proj"), &core)
        }

        fn attention(&mut self, layer: usize, root: &str, x: &[f32], position: usize) -> Vec<f32> {
            let Dims {
                n_heads,
                n_kv_heads,
                head_dim,
                rotary,
                eps,
                theta,
                ..
            } = self.dims;
            let capacity = self.capacity;
            let self_attn = |suffix: &str| format!("{root}.self_attn.{suffix}");
            let q_norm = self.tensor(&self_attn("q_norm"));
            let k_norm = self.tensor(&self_attn("k_norm"));
            // `apply_rotary_pos_emb` at `position`: rotate the leading `rotary` dims, pass the rest
            // through. Each inverse frequency covers both halves.
            let half = rotary / 2;
            let angle = |column: usize| {
                position as f32 * theta.powf(-2.0 * (column % half) as f32 / rotary as f32)
            };
            let rope = |x: &mut [f32]| {
                let rotated = (0..rotary)
                    .map(|i| {
                        let rotate_half = if i < half { -x[i + half] } else { x[i - half] };
                        x[i] * angle(i).cos() + rotate_half * angle(i).sin()
                    })
                    .collect::<Vec<_>>();
                x[..rotary].copy_from_slice(&rotated);
            };
            let query_gate = self.project(&self_attn("q_proj"), x);
            let key = self.project(&self_attn("k_proj"), x);
            let value = self.project(&self_attn("v_proj"), x);

            let LayerState::Attention { keys, values } = &mut self.layers[layer] else {
                panic!("layer {layer} is not an attention layer");
            };
            for kv_head in 0..n_kv_heads {
                let range = kv_head * head_dim..(kv_head + 1) * head_dim;
                let mut k = zero_centered_rms_norm(&key[range.clone()], &k_norm, eps);
                rope(&mut k);
                let slot = (kv_head * capacity + position) * head_dim;
                keys[slot..slot + head_dim].copy_from_slice(&k);
                values[slot..slot + head_dim].copy_from_slice(&value[range]);
            }
            let n_rep = n_heads / n_kv_heads;
            let mut gated = Vec::with_capacity(n_heads * head_dim);
            for head in 0..n_heads {
                // Each head's projection is `[query, gate]`.
                let offset = head * 2 * head_dim;
                let mut q =
                    zero_centered_rms_norm(&query_gate[offset..offset + head_dim], &q_norm, eps);
                rope(&mut q);
                let gate_logits = &query_gate[offset + head_dim..offset + 2 * head_dim];
                // `repeat_kv`: query head `head` reads KV head `head / n_rep`.
                let kv_head = head / n_rep;
                let cached = |cache: &[f32], p: usize| {
                    let slot = (kv_head * capacity + p) * head_dim;
                    cache[slot..slot + head_dim].to_vec()
                };
                let scores = (0..=position)
                    .map(|p| {
                        let k = cached(keys.as_slice(), p);
                        q.iter().zip(&k).map(|(q, k)| q * k).sum::<f32>() / (head_dim as f32).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let total = exp.iter().sum::<f32>();
                for (i, &gate) in gate_logits.iter().enumerate() {
                    let mixed = (0..=position)
                        .map(|p| exp[p] / total * cached(values.as_slice(), p)[i])
                        .sum::<f32>();
                    gated.push(mixed * sigmoid(gate));
                }
            }
            self.project(&self_attn("o_proj"), &gated)
        }

        /// State tensors in `Graph::state` order.
        fn state(&self) -> Vec<Vec<f32>> {
            self.layers
                .iter()
                .flat_map(|layer| match layer {
                    LayerState::Gdn { conv, recurrent } => [conv.clone(), recurrent.clone()],
                    LayerState::Attention { keys, values } => [keys.clone(), values.clone()],
                })
                .collect()
        }
    }

    /// The reference's logits at every position of `tokens` from `start`, and its state after them.
    fn reference_steps(
        reference: &mut HfReference<'_>,
        tokens: &[i32],
        start: usize,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let logits = tokens
            .iter()
            .enumerate()
            .flat_map(|(i, &token)| reference.step(token, start + i))
            .collect();
        (logits, reference.state())
    }

    // ---- the rows ------------------------------------------------------------------------------------

    /// The prompt and the decode tokens of the reference rows.
    const PROMPT: [i32; 3] = [3, 40, 41];
    const DECODE: [i32; 2] = [17, 46];

    /// `store` through `model` on the CPU oracle against the reference over the same store: the
    /// prompt's logits at every position and every carried state element, then the decode steps'.
    fn assert_matches_reference(
        config: &serde_json::Value,
        text: &serde_json::Value,
        root: &str,
        store: &WeightStore,
    ) {
        let model = model_of(config, store);
        let mut reference = HfReference::new(store, root, text);
        let (want, want_state) = reference_steps(&mut reference, &PROMPT, 0);
        let (got, state) = run_step(&*model, store, Phase::Prefill, CAP, &PROMPT, 0, &[]);
        close(&got, &want);
        close_all(&state, &want_state);
        let mut state = state;
        for (i, &token) in DECODE.iter().enumerate() {
            let position = PROMPT.len() + i;
            let (want, want_state) = reference_steps(&mut reference, &[token], position);
            let (got, next) = run_step(
                &*model,
                store,
                Phase::Decode,
                CAP,
                &[token],
                position as i32,
                &state,
            );
            close(&got, &want);
            close_all(&next, &want_state);
            state = next;
        }
    }

    /// SC-001: the fixture compiles and matches the independent reference (a plain-loop f32 model
    /// written from HF `modeling_qwen3_5.py`) at every position and every carried state element on the
    /// CPU oracle (tier 2, NaN fails), in both checkpoint layouts and over both the BF16 store and the
    /// FP8-packed one. Mutation: swap two GDN layers' `A_log` rows in the name table
    /// (`weights_of` maps layer 0's `A` to layer 1's key); the first GDN layer's decay differs and the
    /// prefill logits go red.
    #[test]
    fn the_fixture_matches_the_independent_reference() {
        for (label, fx) in fixtures() {
            let (text, root) = text_of(&fx);
            assert_matches_reference(&fx.config, text, root, &fx.store);
            let packed = fp8_store(&fx);
            let model = model_of(&fx.config, &packed);
            let handles = model
                .weights()
                .iter()
                .filter(|(_, _, h)| matches!(h.format, HandleFormat::Packed(_)))
                .count();
            // 4 mixers' projections (3 GDN layers x 3 + attention x 4) and the 4 layers' 3 gates.
            assert_eq!(handles, 3 * 3 + 4 + 4 * 3, "{label}: packed projections");
            assert_matches_reference(&fx.config, text, root, &packed);
        }
    }

    /// SC-002: a finetune-like fixture (the same architecture with two layers, other head counts and
    /// widths, none of which the old pinned config admitted) and no manifest builds through the
    /// registry, traces and matches its reference. Mutation: reinstate a pinned count in `from_raw`
    /// (refuse any `num_hidden_layers` but the fixture's four); the build is refused with the typed
    /// config error and this row goes red.
    #[test]
    fn a_finetune_shaped_config_builds_and_matches_its_reference() {
        let fx = finetune_fixture();
        let (text, root) = text_of(&fx);
        assert_eq!(text["num_hidden_layers"], 2);
        assert_matches_reference(&fx.config, text, root, &fx.store);
    }

    /// SC-003: a store missing one required tensor, of any layer kind and of every role a layer reads,
    /// fails `Registry::build` with the typed missing-weight error naming it, before any trace. The
    /// two checkpoint layouts name their own tensors. Mutation: discard the error of `map.map` in
    /// `weights_of`; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        for (_, fx) in fixtures() {
            let (_, root) = text_of(&fx);
            for removed in [
                "embed_tokens.weight",
                "norm.weight",
                "layers.0.input_layernorm.weight",
                "layers.1.post_attention_layernorm.weight",
                "layers.0.linear_attn.in_proj_qkv.weight",
                "layers.1.linear_attn.in_proj_z.weight",
                "layers.2.linear_attn.in_proj_b.weight",
                "layers.0.linear_attn.in_proj_a.weight",
                "layers.2.linear_attn.conv1d.weight",
                "layers.0.linear_attn.dt_bias",
                "layers.1.linear_attn.A_log",
                "layers.2.linear_attn.norm.weight",
                "layers.0.linear_attn.out_proj.weight",
                "layers.3.self_attn.q_proj.weight",
                "layers.3.self_attn.k_proj.weight",
                "layers.3.self_attn.v_proj.weight",
                "layers.3.self_attn.o_proj.weight",
                "layers.3.self_attn.q_norm.weight",
                "layers.3.self_attn.k_norm.weight",
                "layers.1.mlp.gate_proj.weight",
                "layers.2.mlp.up_proj.weight",
                "layers.3.mlp.down_proj.weight",
            ] {
                assert_missing_weight_is_named(&registry, &fx, &format!("{root}.{removed}"));
            }
            assert_missing_weight_is_named(&registry, &fx, "lm_head.weight");
        }
    }

    /// SC-005 (dfamily F2): prefill of six tokens equals two prefills of three equals prefill of three
    /// then three decode steps equals six decode steps, at every logit and every carried state element
    /// (the attention cache, every GDN layer's conv window and state matrices). Mutation: ignore the
    /// carried recurrent state in a continued chunk (`gdn` seeds its scan from zeros); the second
    /// chunk goes red.
    #[test]
    fn chunked_prefill_continuing_from_the_carried_state_equals_token_by_token_decode() {
        for (_, fx) in fixtures() {
            let model = model_of(&fx.config, &fx.store);
            assert_chunked_prefill_equals_decode(&*model, &fx.store);
            let prompt = [3, 44, 41, 8, 25, 9];
            let (whole, whole_state) =
                run_step(&*model, &fx.store, Phase::Prefill, CAP, &prompt, 0, &[]);
            let (mut got, mut state) = run_step(
                &*model,
                &fx.store,
                Phase::Prefill,
                CAP,
                &prompt[..3],
                0,
                &[],
            );
            for (i, &token) in prompt[3..].iter().enumerate() {
                let (logits, next) = run_step(
                    &*model,
                    &fx.store,
                    Phase::Decode,
                    CAP,
                    &[token],
                    (3 + i) as i32,
                    &state,
                );
                got.extend(logits);
                state = next;
            }
            close(&got, &whole);
            close_all(&state, &whole_state);
        }
    }

    /// The step carries the two caches of the attention layer and the conv window and state matrices
    /// of each GDN layer: three GDN layers then one attention layer are eight values, in layer order.
    #[test]
    fn the_step_carries_each_layers_own_state() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let g = model
            .trace(Phase::Decode, step_shape(1, CAP, LogitRows::Last))
            .unwrap();
        let names: Vec<String> = g
            .state
            .iter()
            .map(|&(input, _)| g.meta(input).name.clone().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "gdn.l0.conv",
                "gdn.l0.ssm",
                "gdn.l1.conv",
                "gdn.l1.ssm",
                "gdn.l2.conv",
                "gdn.l2.ssm",
                "kv.l3.k",
                "kv.l3.v",
            ]
        );
    }

    /// The BF16 store and its Q8_0 copy trace through the one body and equal the same body over the
    /// store decoded to F32 (prefill and a decode step). The GDN projections stay dense: a random
    /// Q8_0 payload has wild block scales, which overflow the recurrence's decay in both the packed
    /// and the decoded run.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body() {
        let fx = fixture();
        let packable = |key: &str| {
            (key.contains("self_attn.") || key.contains("mlp."))
                && key.ends_with("_proj.weight")
                && fx
                    .store
                    .get(key)
                    .is_some_and(|e| e.shape().len() == 2 && e.shape()[1] % 32 == 0)
        };
        let prompt = [3, 17, 40, 8];
        for (label, store) in [
            ("bf16", fx.store.clone()),
            ("q8_0", q8_0_store(&fx.store, packable)),
        ] {
            let model = model_of(&fx.config, &store);
            let packed = model
                .weights()
                .iter()
                .filter(|(_, _, h)| matches!(h.format, HandleFormat::Packed(_)))
                .count();
            assert_eq!(packed > 0, label == "q8_0", "{label}: {packed} packed");
            let reference_store = decoded(&store);
            let reference = model_of(&fx.config, &reference_store);
            let (got, state) = run_step(&*model, &store, Phase::Prefill, CAP, &prompt, 0, &[]);
            let (want, want_state) = run_step(
                &*reference,
                &reference_store,
                Phase::Prefill,
                CAP,
                &prompt,
                0,
                &[],
            );
            close(&got, &want);
            close_all(&state, &want_state);
            let (got, _) = run_step(&*model, &store, Phase::Decode, CAP, &[5], 4, &state);
            let (want, _) = run_step(
                &*reference,
                &reference_store,
                Phase::Decode,
                CAP,
                &[5],
                4,
                &want_state,
            );
            close(&got, &want);
        }
    }

    /// The fixture's 24 greedy tokens after the 7-token prompt, recorded from the CPU oracle over
    /// the BF16 store (the reference row above pins the logits). Mutation: swap the carried conv
    /// window for zeros in a continued step (the decode steps); the tokens change.
    #[test]
    fn the_fixture_generates_the_recorded_tokens() {
        const RECORDED: [u32; 24] = [
            4, 11, 45, 13, 11, 14, 33, 6, 28, 2, 39, 25, 1, 2, 35, 35, 29, 37, 26, 40, 23, 10, 27,
            24,
        ];
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let got = greedy_tokens(&*model, &fx.store, &[3, 17, 40, 8, 25, 11, 2], 24, 32);
        assert_eq!(got, RECORDED);
    }

    /// The FP8 fixture's 24 greedy tokens after the same prompt, recorded from the CPU oracle over the
    /// packed store (the reference row pins its logits). Mutation: halve every block scale in
    /// `fp8_payload`; the tokens change.
    #[test]
    fn the_fp8_fixture_generates_the_recorded_tokens() {
        const RECORDED: [u32; 24] = [
            31, 6, 9, 22, 7, 12, 6, 6, 30, 9, 12, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        ];
        let fx = fixture_fp8();
        let model = model_of(&fx.config, &fx.store);
        let got = greedy_tokens(&*model, &fx.store, &[3, 17, 40, 8, 25, 11, 2], 24, 32);
        assert_eq!(got, RECORDED);
    }

    fn config_error_of(config: &serde_json::Value) -> (&'static str, ConfigReason) {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        match (ENTRY.build)(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn a_config_qwen35_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "use_sliding_window",
                json!(true),
                "use_sliding_window",
                ConfigReason::Unsupported,
            ),
            (
                "attn_output_gate",
                json!(false),
                "attn_output_gate",
                ConfigReason::Unsupported,
            ),
            (
                "linear_value_head_dim",
                json!(16),
                "linear_value_head_dim",
                ConfigReason::Unsupported,
            ),
            (
                "linear_num_value_heads",
                json!(3),
                "linear_num_value_heads",
                ConfigReason::NotDivisible { by: 2 },
            ),
            (
                "layer_types",
                json!(["linear_attention"]),
                "layer_types",
                ConfigReason::Length { expected: 4 },
            ),
            (
                "layer_types",
                json!([
                    "linear_attention",
                    "linear_attention",
                    "mamba",
                    "full_attention"
                ]),
                "layer_types",
                ConfigReason::Unsupported,
            ),
            ("head_dim", json!(0), "head_dim", ConfigReason::Zero),
            (
                "rms_norm_eps",
                json!(0.0),
                "rms_norm_eps",
                ConfigReason::NotFinitePositive,
            ),
        ];
        for (key, value, field, reason) in cases {
            let mut config = fixture_config();
            config["text_config"][*key] = value.clone();
            assert_eq!(
                config_error_of(&config),
                (*field, *reason),
                "{key} = {value}"
            );
        }
        // A wrapper without its text tower, and a rotation fraction outside (0, 1].
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("text_config");
        assert_eq!(
            config_error_of(&config),
            ("text_config", ConfigReason::Missing)
        );
        let mut config = fixture_config();
        config["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(1.5);
        assert_eq!(
            config_error_of(&config),
            ("partial_rotary_factor", ConfigReason::NotFinitePositive)
        );
    }

    /// The real `Qwen/Qwen3.8-27B` `config.json` (the multimodal wrapper, its text tower, the
    /// vision tower ignored) is read into the pinned dimensions as data: 64 layers, every fourth one
    /// gated attention, the `model.language_model.` names.
    #[test]
    fn the_real_checkpoint_config_reads_into_the_pinned_dimensions() {
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../../../poot-llm/tests/fixtures/qwen38_27b_config.json"
        ))
        .unwrap();
        let p = Qwen35Params::from_raw(&RawConfig::HfJson {
            config: &config,
            generation: None,
        })
        .unwrap();
        assert_eq!(
            (p.vocab, p.width, p.inter, p.max_positions),
            (248_320, 5_120, 17_408, 262_144)
        );
        assert_eq!(p.kinds.len(), 64);
        assert_eq!(
            p.kinds
                .iter()
                .filter(|k| **k == LayerKind::FullAttention)
                .count(),
            16
        );
        assert!(
            p.kinds
                .iter()
                .enumerate()
                .all(|(l, k)| (*k == LayerKind::FullAttention) == (l % 4 == 3))
        );
        assert_eq!(p.names, WRAPPED_NAMES);
        assert_eq!(p.eos, BTreeSet::from([248_044]));
        assert!(p.head_required);
    }

    /// A GGUF names no Qwen3.5 here: the config reader refuses it typed.
    #[test]
    fn a_gguf_config_is_refused_typed() {
        use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};
        let index = GgufIndex::parse(&StoredBytes::from(write_gguf(
            &[("general.architecture", GgufValue::Str("qwen35".into()))],
            &[],
        )))
        .unwrap();
        let err = Qwen35Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "general.architecture",
                    reason: ConfigReason::Unsupported,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn a_shape_the_body_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let n = |v| NonZeroUsize::new(v).unwrap();
        let shape = |tokens, capacity| StepShape {
            tokens: n(tokens),
            capacity: n(capacity),
            ..step_shape(1, 1, LogitRows::Last)
        };
        assert!(matches!(
            model.trace(Phase::Decode, shape(2, 8)),
            Err(TraceError::ShapeUnsupported { .. })
        ));
        assert!(matches!(
            model.trace(Phase::Prefill, shape(1, 65)),
            Err(TraceError::ShapeUnsupported { .. })
        ));
    }
}
