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
//! A GGUF (`general.architecture = qwen35`) is llama.cpp's spelling of the same model: one tensor per
//! GDN quantity (`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`), the value heads reordered from
//! grouped to tiled, `ssm_a` already negative, every norm with its one folded in (the GDN's gated norm
//! has none), and the multi-token-prediction layer the file appends after the last block
//! (`nextn_predict_layers`) left out of the stack. Projections stay packed under their GGUF names.
//!
//! Every dimension is config data, so any checkpoint of the architecture loads (a finetune with
//! fewer layers, another width), with or without an integrity manifest.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

#[cfg(feature = "test-support")]
use std::sync::Arc;

use poot_graph_ir::{Builder, Graph};
#[cfg(feature = "test-support")]
use poot_quant::OperandRole;
#[cfg(feature = "test-support")]
use poot_quant::PackedPayload;
#[cfg(feature = "test-support")]
use poot_quant::PackedWeight;
#[cfg(feature = "test-support")]
use poot_quant::SourceRole;
#[cfg(feature = "test-support")]
use poot_quant::format::ScaleEncoding;
#[cfg(feature = "test-support")]
use poot_quant::format::WeightFormat;
#[cfg(feature = "test-support")]
use poot_quant::weights::WeightEntry;
use poot_quant::weights::{AttnRole, FfnRole, NormRole, WeightMap, WeightRole, WeightStore};

use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::gdn::{GdnParams, GdnWeights, gdn};
use crate::components::gdn_names::{GdnTable, nextn_field, trunk_layers};
use crate::components::linear::WeightError;
use crate::components::norm::NormParams;
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, decoder_program,
    standard_layer, standard_stack,
};
use crate::io::{ModelSchema, StateSpec};
use crate::model::{ConfigReason, FamilyKey, Model, ModelConfig, ModelError, StepSpec, TraceError};
use crate::names::{TensorRows, linear_key};
#[cfg(feature = "test-support")]
use crate::registry::Fixture;
use crate::registry::{ConfigSource, FamilyEntry, ModelParts, RawConfig};
use poot_text::{ChatFormat, HermesToolCodec};

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("qwen35");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "qwen3_5"),
        (ConfigSource::HfModelType, "qwen3_5_text"),
        (ConfigSource::GgufArchitecture, "qwen35"),
    ],
    build,
};

/// The multimodal wrapper's `model_type`: its text tower sits under `text_config`.
const WRAPPER_TYPE: &str = "qwen3_5";

/// Qwen3.5's config facts: `Qwen3_5TextConfig`'s defaults. The class declares no `eos_token_id`: a
/// checkpoint names it. A GGUF must name the base (`gguf_theta: None`).
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("qwen35"),
    hf_theta: 10_000.0,
    gguf_theta: None,
    eps: 1e-6,
    eos: None,
};

const KEY_HEADS: Field = Field::new("linear_num_key_heads", "qwen35.ssm.group_count");
const VALUE_HEADS: Field = Field::new("linear_num_value_heads", "qwen35.ssm.time_step_rank");
const KEY_HEAD_DIM: Field = Field::new("linear_key_head_dim", "qwen35.ssm.state_size");
const VALUE_HEAD_DIM: Field = Field::hf("linear_value_head_dim");
const CONV_KERNEL: Field = Field::new("linear_conv_kernel_dim", "qwen35.ssm.conv_kernel");
const INNER_SIZE: Field = Field::new("", "qwen35.ssm.inner_size");
const INTERVAL: Field = Field::new("full_attention_interval", "qwen35.full_attention_interval");
const ROTARY_DIM: Field = Field::new("", "qwen35.rope.dimension_count");
/// Layers a GGUF appends after the stack for multi-token prediction: they count in `block_count`
/// and have their own tensors, and the stack does not run them.
const NEXTN_LAYERS: Field = nextn_field!("qwen35");

/// One layer's mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerKind {
    /// Gated DeltaNet linear attention.
    LinearAttention,
    /// Gated full attention.
    FullAttention,
}

/// How a checkpoint spells the text tower: the layer prefix, the model-level tensors and each
/// layer's tensor suffixes. A projection is a stem; its stored key is `{stem}.weight`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Names {
    layers: &'static str,
    embed: &'static str,
    final_norm: &'static str,
    head: &'static str,
    attn_norm: &'static str,
    ffn_norm: &'static str,
    attention: AttentionNames,
    gdn: GdnTable,
    ffn: [&'static str; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttentionNames {
    projections: [&'static str; 4],
    q_norm: &'static str,
    k_norm: &'static str,
}

const TEXT_NAMES: Names = Names {
    layers: "model.layers.",
    embed: "model.embed_tokens.weight",
    final_norm: "model.norm.weight",
    head: "lm_head",
    attn_norm: ".input_layernorm.weight",
    ffn_norm: ".post_attention_layernorm.weight",
    attention: AttentionNames {
        projections: [
            ".self_attn.q_proj",
            ".self_attn.k_proj",
            ".self_attn.v_proj",
            ".self_attn.o_proj",
        ],
        q_norm: ".self_attn.q_norm.weight",
        k_norm: ".self_attn.k_norm.weight",
    },
    gdn: GdnTable::HF_SPLIT,
    ffn: [".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"],
};

/// The multimodal wrapper stores the text tower under `model.language_model.`.
const WRAPPED_NAMES: Names = Names {
    layers: "model.language_model.layers.",
    embed: "model.language_model.embed_tokens.weight",
    final_norm: "model.language_model.norm.weight",
    ..TEXT_NAMES
};

/// llama.cpp's `qwen35` tensors.
const GGUF_NAMES: Names = Names {
    layers: "blk.",
    embed: "token_embd.weight",
    final_norm: "output_norm.weight",
    head: "output",
    attn_norm: ".attn_norm.weight",
    ffn_norm: ".post_attention_norm.weight",
    attention: AttentionNames {
        projections: [".attn_q", ".attn_k", ".attn_v", ".attn_output"],
        q_norm: ".attn_q_norm.weight",
        k_norm: ".attn_k_norm.weight",
    },
    gdn: GdnTable::GGUF,
    ffn: [".ffn_gate", ".ffn_up", ".ffn_down"],
};

/// Qwen3.5's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen35Params {
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
/// `"linear_attention"` per layer; HF only), or `full_attention_interval` (default 4) does, where layer
/// `i` is full attention iff `(i + 1)` is a multiple of the interval.
fn layer_kinds(
    fields: &Fields<'_>,
    text: Option<&serde_json::Value>,
    layers: usize,
) -> Result<Vec<LayerKind>, ModelError> {
    if let Some(types) = text
        .and_then(|t| t.get("layer_types"))
        .filter(|v| !v.is_null())
    {
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
    /// Reads the typed config from an HF `config.json` (the multimodal wrapper or a text-only
    /// config) or a GGUF index (`qwen35`); a field the family cannot serve is a typed
    /// [`ModelError::Config`] naming it.
    pub(crate) fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let text;
        let (raw, names) = match raw {
            RawConfig::HfJson {
                config, generation, ..
            } => {
                let (config, names) = text_config(config)?;
                text = config;
                (
                    RawConfig::HfJson {
                        config: &text,
                        generation: *generation,
                        preprocessor: None,
                    },
                    names,
                )
            }
            RawConfig::Gguf(index) => (RawConfig::Gguf(index), GGUF_NAMES),
        };
        let hf_text = match &raw {
            RawConfig::HfJson { config, .. } => Some(*config),
            RawConfig::Gguf(_) => None,
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
        let layers = trunk_layers(&fields, NEXTN_LAYERS, dims.layers)?;
        // The residual-stream and projection norms: HF stores `(1 + w)`, a GGUF folds the one in.
        let eps = dims.eps;
        let eps_error = |_| fields.error("rms_norm_eps", ConfigReason::NotFinitePositive);
        let norm =
            NormParams::of(names.gdn.convention.norm_fold.norm_kind(), eps).map_err(eps_error)?;
        let rotary = match hf_text {
            Some(text) => ((dims.head_dim as f32 * rotary_factor(&fields, text)?) as usize).max(1),
            None => match fields.opt_u64(ROTARY_DIM)? {
                Some(n) => usize::try_from(n)
                    .map_err(|_| fields.error(fields.key(ROTARY_DIM), ConfigReason::WrongType))?,
                None => dims.head_dim,
            },
        };
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
        let (key_heads, value_heads) = (
            fields.count(KEY_HEADS, None)?,
            fields.count(VALUE_HEADS, None)?,
        );
        if let Some(inner) = fields.opt_u64(INNER_SIZE)?
            && inner as usize != value_heads * head_dim
        {
            return Err(fields.error(fields.key(INNER_SIZE), ConfigReason::Unsupported));
        }
        let gdn = names
            .gdn
            .params(
                key_heads,
                value_heads,
                head_dim,
                fields.count(CONV_KERNEL, None)?,
                NormParams::new(eps).map_err(eps_error)?,
            )
            .map_err(|e| e.for_family(FAMILY))?;

        Ok(Self {
            kinds: layer_kinds(&fields, hf_text, layers)?,
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
    let mut rows = TensorRows::new(FAMILY, store);
    rows.tensor(WeightRole::Embed, None, names.embed.to_string())?;
    rows.tensor(WeightRole::FinalNorm, None, names.final_norm.to_string())?;
    rows.head(names.embed, names.head, p.head_required)?;

    // The stored key of a linear: `{stem}.weight`, dense or the packed payload the loader publishes
    // under it.
    let linear = |stem: &str| linear_key(store, stem);
    for (l, kind) in p.kinds.iter().enumerate() {
        let layer = |suffix: &str| format!("{}{l}{suffix}", names.layers);
        let at = Some(l);
        rows.tensor(WeightRole::Norm(NormRole::Attn), at, layer(names.attn_norm))?;
        rows.tensor(WeightRole::Norm(NormRole::Ffn), at, layer(names.ffn_norm))?;
        match kind {
            LayerKind::FullAttention => {
                let a = &names.attention;
                let [q, k, v, o] = a.projections;
                for (role, stem) in [
                    (AttnRole::Q, q),
                    (AttnRole::K, k),
                    (AttnRole::V, v),
                    (AttnRole::O, o),
                ] {
                    rows.linear(WeightRole::Attn(role), at, &layer(stem))?;
                }
                rows.tensor(WeightRole::Attn(AttnRole::QNorm), at, layer(a.q_norm))?;
                rows.tensor(WeightRole::Attn(AttnRole::KNorm), at, layer(a.k_norm))?;
            }
            LayerKind::LinearAttention => {
                let mapped = names.gdn.map_layer(&mut rows.map, l, &layer, &linear);
                mapped.map_err(|source| rows.error(source))?;
            }
        }
        let [gate, up, down] = names.ffn;
        for (role, stem) in [
            (FfnRole::Gate, gate),
            (FfnRole::Up, up),
            (FfnRole::Down, down),
        ] {
            rows.linear(WeightRole::Ffn(role), at, &layer(stem))?;
        }
    }
    Ok(rows.finish())
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
pub(crate) struct Qwen35 {
    params: Qwen35Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    assemble(Qwen35Params::from_raw(raw)?, store)
}

/// The model of `p` over the weights `store` holds.
fn assemble(p: Qwen35Params, store: &WeightStore) -> Result<ModelParts, ModelError> {
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
    let mut step = decoder_program(p.vocab);
    if p.kinds.contains(&LayerKind::LinearAttention) {
        step = step.carrying(StateSpec::recurrent("gdn"));
    }
    let schema = ModelSchema::builder().step(step).build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
        tool_codec: Some(Arc::new(HermesToolCodec)),
    };
    Ok(ModelParts::new(Box::new(Qwen35 {
        params: p,
        weights,
        stack,
        layers,
        config,
        schema,
    })))
}

impl Model for Qwen35 {
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
        let step = Step::new(&b, shape)?;
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
        )?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's text config: a four-layer hybrid (three GDN layers, then gated attention) over a
/// dense feed-forward, GQA on both mixers (`value / key` heads 2 for the GDN, `heads / kv_heads` 2 for
/// the attention), a partial rotation and the driver fixtures' 48-word vocabulary. Projection widths are multiples of 32 where a Q8_0 copy needs them.
#[cfg(feature = "test-support")]
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
#[cfg(feature = "test-support")]
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
#[cfg(feature = "test-support")]
pub(crate) fn hf_shapes(text: &serde_json::Value, root: &str) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| crate::registry::Fixture::count(text, f);
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

#[cfg(feature = "test-support")]
pub(crate) fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_shapes(&config["text_config"], "model.language_model");
    Fixture::bf16(config, shapes)
}

/// The text-only checkpoint (`qwen3_5_text`): the same tower stored under `model.`.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_text() -> Fixture {
    let config = fixture_text_config();
    let shapes = hf_shapes(&config, "model");
    Fixture::bf16(config, shapes)
}

// ---- the FP8 checkpoint -------------------------------------------------------------------------

/// Finite E4M3FN codes with mixed signs and magnitudes.
#[cfg(feature = "test-support")]
const E4M3_CODES: [u8; 9] = [0x00, 0x20, 0x28, 0x2c, 0x30, 0x34, 0xa4, 0xa8, 0xb0];

/// SplitMix64 finalizer: spreads a seed so neighboring inputs give unrelated outputs.
#[cfg(feature = "test-support")]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A packed `E4m3Block128` payload of `shape` whose codes and scale come from `stem`: seeding by
/// the checkpoint name gives same-shaped projections (`k_proj`/`v_proj`, `gate_proj`/`up_proj`)
/// different weights, so a projection bound under the wrong name changes the logits.
#[cfg(feature = "test-support")]
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
#[cfg(feature = "test-support")]
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
#[cfg(feature = "test-support")]
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
#[cfg(feature = "test-support")]
pub(crate) fn fixture_fp8() -> Fixture {
    let mut fx = fixture();
    fx.store = fp8_store(&fx);
    fx
}

/// A fine-tune of the fixture's architecture with other counts than any pinned release: two layers
/// (one GDN, one attention), other widths and head counts. It ships no manifest and is the same family.
#[cfg(feature = "test-support")]
pub(crate) fn finetune_fixture() -> Fixture {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use poot_graph_ir::{Slot, TensorType};
    use poot_load::gguf::{
        GgufIndex, GgufValue, IdentityNames, TensorNameMap, gguf_plan, read_gguf,
    };
    use poot_load::plan::default_workers;
    use poot_quant::StoredBytes;
    use poot_quant::weights::HandleFormat;
    use serde_json::json;

    use super::*;
    use crate::components::gdn_names::converter::{
        HfProjections, LayerDims, layer_metadata, nextn_key, plus_one, write_layer,
        write_nextn_block,
    };
    use crate::components::standard::oracle::{eval, run_step};
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build as built,
        close, close_all, decoded, f32_entry, gguf_tensor, greedy_tokens, q8_0_store,
        read_back_owned, step_shape, values,
    };
    use crate::model::{LogitRows, StepShape};
    use crate::registry::Registry;
    use poot_quant::weights::{WeightKey, WeightMapError};

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
            preprocessor: None,
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
            let n = |f: &str| crate::registry::Fixture::count(text, f);
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
        let (got, state) = run_step(&*model, store, CAP, &PROMPT, 0, &[]);
        close(&got, &want);
        close_all(&state, &want_state);
        let mut state = state;
        for (i, &token) in DECODE.iter().enumerate() {
            let position = PROMPT.len() + i;
            let (want, want_state) = reference_steps(&mut reference, &[token], position);
            let (got, next) = run_step(&*model, store, CAP, &[token], position as i32, &state);
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
            let (whole, whole_state) = run_step(&*model, &fx.store, CAP, &prompt, 0, &[]);
            let (mut got, mut state) = run_step(&*model, &fx.store, CAP, &prompt[..3], 0, &[]);
            for (i, &token) in prompt[3..].iter().enumerate() {
                let (logits, next) =
                    run_step(&*model, &fx.store, CAP, &[token], (3 + i) as i32, &state);
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
            .trace(
                &StepSpec::step(step_shape(1, CAP, LogitRows::Last)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let names: Vec<String> = g
            .state()
            .map(|(input, _)| g.meta(input).name.clone().unwrap())
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
            let (got, state) = run_step(&*model, &store, CAP, &prompt, 0, &[]);
            let (want, want_state) = run_step(&*reference, &reference_store, CAP, &prompt, 0, &[]);
            close(&got, &want);
            close_all(&state, &want_state);
            let (got, _) = run_step(&*model, &store, CAP, &[5], 4, &state);
            let (want, _) = run_step(&*reference, &reference_store, CAP, &[5], 4, &want_state);
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
            preprocessor: None,
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
            preprocessor: None,
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

    #[test]
    fn a_shape_the_body_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let n = |v| NonZeroUsize::new(v).unwrap();
        let shape = |tokens, capacity| StepShape::new(n(1), n(tokens), n(capacity));
        assert!(matches!(
            model.trace(
                &StepSpec::step(shape(9, 8)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported { .. })
        ));
        assert!(matches!(
            model.trace(
                &StepSpec::step(shape(1, 65)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported { .. })
        ));
    }

    // ---- the GGUF spelling ---------------------------------------------------------------------

    /// A GGUF tensor as the reader takes it: name, dims, GGML type and bytes.
    type GgufTensor = (String, Vec<u64>, u32, Vec<u8>);

    /// The GGUF checkpoint llama.cpp's converter writes for the wrapped HF `fixture`: every
    /// per-value-head tensor reordered from HF's grouped order to tiled, `ssm_a = -exp(A_log)`, the
    /// norms with their one folded in (the GDN's gated norm has none), and, when `nextn` is set, the
    /// extra block the file appends for multi-token prediction. Written here from the converter's
    /// rules, independently of the family's own GGUF reader.
    fn gguf_of(fx: &Fixture, nextn: bool) -> (Vec<(String, GgufValue)>, Vec<GgufTensor>) {
        let (text, root) = text_of(fx);
        let n = |f: &str| Fixture::count(text, f);
        let gdn = LayerDims {
            width: n("hidden_size"),
            key_heads: n("linear_num_key_heads"),
            value_heads: n("linear_num_value_heads"),
            head_dim: n("linear_key_head_dim"),
            kernel: n("linear_conv_kernel_dim"),
        };
        let w = gdn.width;
        let layers = n("num_hidden_layers");
        let (heads, kv, d) = (
            n("num_attention_heads"),
            n("num_key_value_heads"),
            n("head_dim"),
        );
        let (vocab, inter) = (n("vocab_size"), n("intermediate_size"));
        let vals = |key: &str| values(fx.store.get(key).unwrap_or_else(|| panic!("{key}")));
        let mut tensors: Vec<GgufTensor> = Vec::new();
        let mut put = |name: String, shape: Vec<usize>, data: Vec<f32>| {
            assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
            tensors.push(gguf_tensor(&name, &f32_entry(shape, &data)));
        };
        put(
            "token_embd.weight".into(),
            vec![vocab, w],
            vals(&format!("{root}.embed_tokens.weight")),
        );
        put(
            "output_norm.weight".into(),
            vec![w],
            plus_one(vals(&format!("{root}.norm.weight"))),
        );
        put(
            "output.weight".into(),
            vec![vocab, w],
            vals("lm_head.weight"),
        );
        for l in 0..layers {
            let hf = |s: &str| format!("{root}.layers.{l}.{s}");
            let gg = |s: &str| format!("blk.{l}.{s}");
            put(
                gg("attn_norm.weight"),
                vec![w],
                plus_one(vals(&hf("input_layernorm.weight"))),
            );
            put(
                gg("post_attention_norm.weight"),
                vec![w],
                plus_one(vals(&hf("post_attention_layernorm.weight"))),
            );
            if fx.store.contains(&hf("self_attn.q_proj.weight")) {
                for (gguf, hf_name, rows, cols) in [
                    ("attn_q", "self_attn.q_proj", 2 * heads * d, w),
                    ("attn_k", "self_attn.k_proj", kv * d, w),
                    ("attn_v", "self_attn.v_proj", kv * d, w),
                    ("attn_output", "self_attn.o_proj", w, heads * d),
                ] {
                    put(
                        gg(&format!("{gguf}.weight")),
                        vec![rows, cols],
                        vals(&hf(&format!("{hf_name}.weight"))),
                    );
                }
                for (gguf, hf_name) in [
                    ("attn_q_norm", "self_attn.q_norm"),
                    ("attn_k_norm", "self_attn.k_norm"),
                ] {
                    put(
                        gg(&format!("{gguf}.weight")),
                        vec![d],
                        plus_one(vals(&hf(&format!("{hf_name}.weight")))),
                    );
                }
            } else {
                write_layer(
                    gdn,
                    HfProjections::Split,
                    &|suffix| vals(&hf(&format!("linear_attn.{suffix}"))),
                    &gg,
                    &mut put,
                );
            }
            for (gguf, hf_name, rows, cols) in [
                ("ffn_gate", "mlp.gate_proj", inter, w),
                ("ffn_up", "mlp.up_proj", inter, w),
                ("ffn_down", "mlp.down_proj", w, inter),
            ] {
                put(
                    gg(&format!("{gguf}.weight")),
                    vec![rows, cols],
                    vals(&hf(&format!("{hf_name}.weight"))),
                );
            }
        }
        if nextn {
            write_nextn_block(w, layers, &mut put);
        }
        let u = |v: usize| GgufValue::U32(v as u32);
        let mut kvs = vec![
            (
                "general.architecture".to_string(),
                GgufValue::Str("qwen35".into()),
            ),
            ("qwen35.embedding_length".to_string(), u(w)),
            ("qwen35.feed_forward_length".to_string(), u(inter)),
            (
                "qwen35.block_count".to_string(),
                u(layers + usize::from(nextn)),
            ),
            ("qwen35.attention.head_count".to_string(), u(heads)),
            ("qwen35.attention.head_count_kv".to_string(), u(kv)),
            ("qwen35.attention.key_length".to_string(), u(d)),
            (
                "qwen35.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::F32(text["rms_norm_eps"].as_f64().unwrap() as f32),
            ),
            (
                "qwen35.context_length".to_string(),
                u(n("max_position_embeddings")),
            ),
            (
                "qwen35.rope.freq_base".to_string(),
                GgufValue::F32(text["rope_parameters"]["rope_theta"].as_f64().unwrap() as f32),
            ),
            (
                "qwen35.rope.dimension_count".to_string(),
                u((d as f64
                    * text["rope_parameters"]["partial_rotary_factor"]
                        .as_f64()
                        .unwrap()) as usize),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(
                    (0..vocab)
                        .map(|t| GgufValue::Str(format!("t{t}")))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_string(),
                u(fx.config["eos_token_id"].as_u64().unwrap() as usize),
            ),
        ];
        kvs.extend(layer_metadata("qwen35", gdn, nextn));
        (kvs, tensors)
    }

    /// The GGUF the converter writes for the HF fixture resolves through the registry to this family
    /// and gives the same logits and attention cache as the HF model, prefill and decode, with and
    /// without the appended multi-token-prediction block. Mutations: read the GGUF with
    /// `GdnHeadOrder::Grouped` (the value heads pair with the wrong key heads); take `block_count`
    /// as the stack's depth (the nextn block is traced as a fifth layer and the weight map names
    /// tensors the file lacks).
    #[test]
    fn the_gguf_spelling_of_the_fixture_equals_the_hf_model() {
        let fx = fixture();
        let hf = model_of(&fx.config, &fx.store);
        let prompt = [3, 17, 40, 8, 25, 9];
        let (want, want_state) = run_step(&*hf, &fx.store, CAP, &prompt, 0, &[]);
        let (want_decode, _) = run_step(&*hf, &fx.store, CAP, &[5], 6, &want_state);
        for nextn in [false, true] {
            let (kvs, tensors) = gguf_of(&fx, nextn);
            let (index, store) = read_back_owned(&kvs, &tensors);
            let registry = Registry::builtin().unwrap();
            let raw = RawConfig::Gguf(&index);
            assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
            let gguf = registry.build(&raw, &store).unwrap();
            let (got, state) = run_step(&*gguf.model, &store, CAP, &prompt, 0, &[]);
            close(&got, &want);
            // The carried tensors differ in value-head order, so only the logits and the attention
            // cache (the last two state values) are comparable.
            let kv = want_state.len() - 2;
            close_all(&state[kv..], &want_state[kv..]);
            let (got, _) = run_step(&*gguf.model, &store, CAP, &[5], 6, &state);
            close(&got, &want_decode);
        }
    }

    /// The GGUF's config is read from its own keys: the stack's depth excludes the appended
    /// multi-token-prediction blocks, the layer kinds follow `full_attention_interval`, and the norms
    /// are the folded kind.
    #[test]
    fn a_gguf_config_excludes_the_nextn_block_and_reads_the_interval() {
        let fx = fixture();
        let (kvs, tensors) = gguf_of(&fx, true);
        let (index, _) = read_back_owned(&kvs, &tensors);
        let p = Qwen35Params::from_raw(&RawConfig::Gguf(&index)).unwrap();
        assert_eq!(
            p.kinds,
            [
                LayerKind::LinearAttention,
                LayerKind::LinearAttention,
                LayerKind::LinearAttention,
                LayerKind::FullAttention,
            ]
        );
        assert_eq!(p.names, GGUF_NAMES);
        assert!(!p.head_required);
        assert_eq!(p.eos, BTreeSet::from([47]));
    }

    /// More multi-token-prediction blocks than blocks is a typed config refusal naming the key.
    #[test]
    fn a_gguf_with_more_nextn_blocks_than_blocks_is_refused_typed() {
        let fx = fixture();
        let (mut kvs, tensors) = gguf_of(&fx, true);
        for (key, value) in &mut kvs {
            if *key == nextn_key("qwen35") {
                *value = GgufValue::U32(5);
            }
        }
        let (index, _) = read_back_owned(&kvs, &tensors);
        let err = Qwen35Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field,
                    reason: ConfigReason::Exceeds { max: 5 },
                    ..
                } if field == nextn_key("qwen35")
            ),
            "{err}"
        );
    }

    /// SC-003: a GGUF header without `rope.freq_base` is refused as `Missing`, naming the key; it
    /// has no default base. Mutation: set `SPEC.gguf_theta` to `Some(10_000_000.0)`.
    #[test]
    fn a_gguf_without_a_rope_base_is_refused() {
        let fx = fixture();
        let (mut kvs, tensors) = gguf_of(&fx, false);
        kvs.retain(|(key, _)| key != "qwen35.rope.freq_base");
        let (index, _) = read_back_owned(&kvs, &tensors);
        let err = Qwen35Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "qwen35.rope.freq_base",
                    reason: ConfigReason::Missing,
                    ..
                }
            ),
            "{err}"
        );
    }

    // ---- the real checkpoint ---------------------------------------------------------------------

    const REAL_GGUF: poot_test_util::Checkpoint =
        poot_test_util::checkpoint!("Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf");

    /// Keeps the model-level tensors and the first `blocks` blocks of a GGUF (the real file is 18 GB:
    /// the rows below read only these, mapped, and decode single rows).
    struct FirstBlocks {
        blocks: usize,
    }

    impl TensorNameMap for FirstBlocks {
        fn weight_key(&self, name: &str) -> Option<WeightKey> {
            let keep = match name.strip_prefix("blk.") {
                None => true,
                Some(rest) => rest
                    .split_once('.')
                    .and_then(|(block, _)| block.parse::<usize>().ok())
                    .is_some_and(|block| block < self.blocks),
            };
            keep.then(|| WeightKey::from(name))
        }
    }

    /// Keeps exactly one tensor.
    struct Only<'a>(&'a str);

    impl TensorNameMap for Only<'_> {
        fn weight_key(&self, name: &str) -> Option<WeightKey> {
            (name == self.0).then(|| WeightKey::from(name))
        }
    }

    fn real_gguf() -> Option<GgufIndex> {
        let path = poot_test_util::model_path(REAL_GGUF)?;
        let file = StoredBytes::map(std::fs::File::open(path).unwrap()).unwrap();
        let index = GgufIndex::parse(&file).unwrap();
        Some(index)
    }

    /// The real `Qwen3.8-27B` GGUF (UD-Q4_K_XL) reads into the pinned dimensions: the 65th block is the
    /// multi-token-prediction layer, the stack is 64 layers with full attention on every fourth, the
    /// GDN's heads are the file's (16 key, 48 value, 128 wide), and its token ids and RoPE base are
    /// the file's. Expected values are the header's, read by an independent parser.
    #[test]
    fn the_real_gguf_config_reads_into_the_pinned_dimensions() {
        let Some(index) = real_gguf() else {
            return;
        };
        let raw = RawConfig::Gguf(&index);
        assert_eq!(
            Registry::builtin().unwrap().resolve(&raw).unwrap().family,
            FAMILY
        );
        let p = Qwen35Params::from_raw(&raw).unwrap();
        assert_eq!((p.vocab, p.width, p.inter), (248_320, 5_120, 17_408));
        assert_eq!(p.kinds.len(), 64);
        for (l, kind) in p.kinds.iter().enumerate() {
            let full = (l + 1) % 4 == 0;
            assert_eq!(*kind == LayerKind::FullAttention, full, "layer {l}");
        }
        assert_eq!(p.max_positions, 262_144);
        assert_eq!(p.eos, BTreeSet::from([248_046]));
        assert_eq!(p.bos, Some(248_044));
        assert_eq!(p.names, GGUF_NAMES);
        assert!(!p.head_required);
    }

    /// The first four blocks of the real file (three GDN layers, then gated attention) bind through
    /// the family's names with the quantization the file stores, kept packed: the weight map resolves
    /// every tensor the stack reads, and the model traces a prefill step over them. The expected
    /// formats are the header's, read by an independent parser. Mutation: map `attn_gate` to the
    /// `ssm_alpha` row; the GDN's `z` projection binds a `[48, 5120]` tensor and the build is refused.
    #[test]
    fn the_real_gguf_first_blocks_bind_packed_and_trace() {
        let Some(index) = real_gguf() else {
            return;
        };
        let raw = RawConfig::Gguf(&index);
        let mut p = Qwen35Params::from_raw(&raw).unwrap();
        p.kinds.truncate(4);
        let store = read_gguf(&index, &FirstBlocks { blocks: 4 }, default_workers()).unwrap();
        let format = |key: &str| match store.get(key).unwrap_or_else(|| panic!("{key}")) {
            WeightEntry::Packed(payload) => payload.weight().format(),
            WeightEntry::Dense(_) => panic!("{key} was widened to a dense tensor"),
        };
        for (key, want) in [
            ("token_embd.weight", WeightFormat::Q4_K),
            ("output.weight", WeightFormat::Q6_K),
            ("blk.0.attn_qkv.weight", WeightFormat::Q5_K),
            ("blk.0.attn_gate.weight", WeightFormat::Q5_K),
            ("blk.0.ssm_alpha.weight", WeightFormat::Q8_0),
            ("blk.0.ssm_out.weight", WeightFormat::Q8_0),
            ("blk.0.ffn_gate.weight", WeightFormat::Iq4_Xs),
            ("blk.0.ffn_up.weight", WeightFormat::Q3_K),
            ("blk.1.ffn_down.weight", WeightFormat::Iq4_Nl),
            ("blk.3.attn_q.weight", WeightFormat::Q5_K),
            ("blk.3.attn_k.weight", WeightFormat::Q4_K),
            ("blk.3.attn_output.weight", WeightFormat::Q6_K),
        ] {
            assert_eq!(format(key), want, "{key}");
        }
        let model = assemble(p, &store).unwrap();
        model
            .model
            .trace(
                &StepSpec::step(step_shape(2, 8, LogitRows::Last)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
    }

    /// The committed llama.cpp activation `name` (`attn_norm-0` or `linear_attn_out-0`) of the real
    /// file's first block: eight tokens by `width` f32, token-major.
    fn llama_cpp_activation(name: &str) -> Vec<f32> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/llamacpp-dump/reference")
            .join(format!("qwen35_gdn_layer0_{name}.f32"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let (words, rest) = bytes.as_chunks::<4>();
        assert!(rest.is_empty(), "{} is not whole f32 words", path.display());
        words.iter().map(|w| f32::from_le_bytes(*w)).collect()
    }

    /// SC-001: block 0 of the real file is a Gated DeltaNet layer; run on the CPU oracle over its
    /// packed weights, it maps llama.cpp's own input to the activation llama.cpp computed. The input
    /// (`attn_norm-0`, the layer's normed residual) and the output (`linear_attn_out-0`, after the
    /// output projection, before the residual add) are what llama.cpp (release b9190, built by
    /// `scripts/llamacpp-dump/build.sh` from the source `flake.lock` pins) computes for the eight
    /// tokens in `scripts/llamacpp-dump/reference/tokens.txt` on the CPU; `reference.sh` regenerates
    /// them from the real file. The oracle evaluates the same Q5_K / Q8_0 weights in f32, so the two
    /// differ by summation order only (llama.cpp ran block 0 with its tensors widened to f32 by its own
    /// `to_float`, so it quantizes no activations): observed 3.6e-6 of the largest activation. The
    /// bound is [`BOUND`], a fixed figure until the numerics class of POOT-1214 publishes tier 2.
    /// Mutation: flip the decay convention of
    /// [`GdnTable::GGUF`] to `DecaySign::NegExp` (the layer exponentiates the stored decay): the
    /// output departs from llama.cpp far beyond the bound.
    #[test]
    fn the_real_gguf_gdn_layer_matches_the_llama_cpp_activation() {
        /// The largest deviation allowed, as a fraction of the reference's largest magnitude.
        const BOUND: f32 = 1e-4;
        let Some(index) = real_gguf() else {
            return;
        };
        let (input, want) = (
            llama_cpp_activation("attn_norm-0"),
            llama_cpp_activation("linear_attn_out-0"),
        );
        let mut p = Qwen35Params::from_raw(&RawConfig::Gguf(&index)).unwrap();
        p.kinds.truncate(1);
        assert_eq!(p.kinds, [LayerKind::LinearAttention]);
        let store = read_gguf(&index, &FirstBlocks { blocks: 1 }, default_workers()).unwrap();
        let weights = weights_of(&p, &store).unwrap();
        let w = GdnWeights::new(&weights, 0, p.width, &p.gdn).unwrap();
        let tokens = input.len() / p.width;
        assert_eq!((tokens, input.len(), want.len()), (8, 8 * 5120, 8 * 5120));

        let b = Builder::for_tests();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All)).unwrap();
        let x = b
            .constant("x", TensorType::f32(vec![1, tokens, p.width]))
            .unwrap();
        let y = gdn(&b, &step, 0, x, &w, &p.gdn).unwrap();
        let g = step.finish(b, y).unwrap();
        let formats = poot_graph_plan::WeightFormats::from_weight_map(&weights);
        let g = poot_graph_plan::bind_packed_weights(&g, &formats).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let zeros: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).collect();
        let (ids, pos): (Vec<i32>, Vec<i32>) = (vec![0; tokens], (0..tokens as i32).collect());
        let (got, _) = eval(
            &g,
            &store,
            &weights,
            &[("x", &input)],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &zeros,
        );
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = poot_test_util::max_abs_error(&got, &want);
        assert!(
            worst <= BOUND * scale,
            "worst deviation {worst} against a largest activation of {scale}"
        );
    }

    /// Every tensor of the real file's header plans through the production name map (`IdentityNames`,
    /// what `poot-llm`'s driver loads a GGUF with): no tensor is refused for its type, including the
    /// file's one `IQ3_S` tensor (`blk.14.ffn_down.weight`, ggml type 21), and each of the 866 tensors is
    /// one owner of the plan. No first-blocks or single-tensor map narrows the file. Mutation: map ggml
    /// type 21 to `None` in `weight_format_of`; the plan is refused naming that tensor.
    #[test]
    fn every_tensor_of_the_real_gguf_plans_through_the_production_name_map() {
        let Some(index) = real_gguf() else {
            return;
        };
        assert_eq!(index.tensors.len(), 866);
        assert_eq!(index.tensors["blk.14.ffn_down.weight"].ggml_type, 21);
        let plan = gguf_plan(&index, &IdentityNames).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.len(), index.tensors.len());
    }

    /// Per-row dequant parity of the real file's tensors, one pair of rows per quantization the file
    /// uses, against llama.cpp's own `to_float` (release b9190, `scripts/llamacpp-dump/dequant_rows.cpp`;
    /// `reference.sh` regenerates `qwen35_dequant_rows.tsv` from the file). Each row is `(tensor, ggml
    /// type, row, first four values, sum, sum of absolute values, position-weighted sum)`; the weighted
    /// sum fails a decoder that permutes values within a block. Mutation: swap two entries of the IQ4_NL
    /// codebook; its rows go red.
    #[test]
    fn the_real_gguf_rows_dequantize_to_the_llama_cpp_reference() {
        let Some(index) = real_gguf() else {
            return;
        };
        type Row = (String, u32, usize, [f64; 4], f64, f64, f64);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/llamacpp-dump/reference/qwen35_dequant_rows.tsv");
        let table =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let rows: Vec<Row> = table
            .lines()
            .map(|line| {
                let cells: Vec<&str> = line.split('\t').collect();
                assert_eq!(cells.len(), 10, "{line}");
                let float = |i: usize| cells[i].parse::<f64>().unwrap();
                (
                    cells[0].to_string(),
                    cells[1].parse().unwrap(),
                    cells[2].parse().unwrap(),
                    [float(3), float(4), float(5), float(6)],
                    float(7),
                    float(8),
                    float(9),
                )
            })
            .collect();
        assert_eq!(rows.len(), 18);
        let mut tensors: BTreeMap<String, WeightStore> = BTreeMap::new();
        for (name, ggml_type, row, head, sum, abs_sum, weighted) in rows {
            let name = name.as_str();
            assert_eq!(index.tensors[name].ggml_type, ggml_type, "{name}");
            let store = tensors
                .entry(name.to_string())
                .or_insert_with(|| read_gguf(&index, &Only(name), default_workers()).unwrap());
            let WeightEntry::Packed(payload) = store.get(name).unwrap() else {
                panic!("{name} was widened to a dense tensor");
            };
            let mut out = vec![0.0f32; payload.weight().shape()[1]];
            payload.decode_row(row, &mut out).unwrap();
            let near = |got: f64, want: f64, what: &str| {
                let bound = 1e-4 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= bound,
                    "{name} row {row} {what}: {got} vs {want}"
                );
            };
            for (i, want) in head.into_iter().enumerate() {
                near(f64::from(out[i]), want, &format!("value {i}"));
            }
            near(out.iter().map(|&v| f64::from(v)).sum(), sum, "sum");
            near(
                out.iter().map(|&v| f64::from(v).abs()).sum(),
                abs_sum,
                "sum of absolute values",
            );
            near(
                out.iter()
                    .enumerate()
                    .map(|(i, &v)| (i + 1) as f64 * f64::from(v))
                    .sum(),
                weighted,
                "weighted sum",
            );
        }
    }

    /// A GGUF without a mapped tensor fails the build with the weight map's typed missing-weight
    /// error naming the GGUF tensor.
    #[test]
    fn a_gguf_missing_a_mapped_tensor_names_it() {
        let fx = fixture();
        let (kvs, tensors) = gguf_of(&fx, false);
        let tensors: Vec<GgufTensor> = tensors
            .into_iter()
            .filter(|(name, ..)| name != "blk.0.ssm_alpha.weight")
            .collect();
        let (index, store) = read_back_owned(&kvs, &tensors);
        let err = match Registry::builtin()
            .unwrap()
            .build(&RawConfig::Gguf(&index), &store)
        {
            Err(e) => e,
            Ok(_) => panic!("a store without blk.0.ssm_alpha.weight built"),
        };
        let crate::registry::LoadModelError::Model(ModelError::Weight {
            source: WeightMapError::Missing { key, .. },
            ..
        }) = &err
        else {
            panic!("expected a missing-weight error, got {err}");
        };
        assert_eq!(key.as_str(), "blk.0.ssm_alpha.weight", "{err}");
    }
}
