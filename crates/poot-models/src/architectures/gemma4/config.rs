//! Gemma 4's typed config (HF `gemma4`, which nests the text tower under `text_config`): every
//! value checked once here, so tracing cannot fail on it.
//!
//! The two layer kinds differ in head width, KV heads and RoPE: a local layer is sliding-window
//! attention over `head_dim`-wide heads with plain RoPE, a global layer is full attention over
//! `global_head_dim`-wide heads with proportional RoPE (a fraction of the frequency pairs rotate,
//! the rest are NoPE) and, when `attention_k_eq_v`, reads V from its K projection. Both scale
//! scores by one and normalize V per head with no learned scale.

use std::collections::BTreeSet;

use poot_graph_ir::rope_table::RopeFlavor;

use crate::components::attention::{AttentionParams, QkNorm};
use crate::components::dims::{Field, Fields};
use crate::components::ffn::Glu;
use crate::components::moe::{GateUpForm, MoeParams};
use crate::components::norm::{NormKind, NormParams};
use crate::components::rope::{GGUF_ROPE_FREQS, RopeParams, rope_freq_factors};
use crate::components::router::RouterKind;
use crate::components::standard::{LayerParams, NormPlacement, ParamError, StackParams};
use crate::model::{ConfigReason, FamilyKey, ModelError};
use crate::registry::RawConfig;

pub(super) const FAMILY: FamilyKey = FamilyKey::new("gemma4");

/// The activation HF names for Gemma's tanh-approximated GELU, the only one the body traces.
const ACTIVATION: &str = "gelu_pytorch_tanh";
/// RoPE base a layer kind's `rope_parameters` block omits.
const DEFAULT_LOCAL_THETA: f32 = 10_000.0;
const DEFAULT_GLOBAL_THETA: f32 = 1_000_000.0;

/// Gemma 4's config facts.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Params {
    pub(super) vocab: usize,
    pub(super) width: usize,
    pub(super) inter: usize,
    pub(super) layers: usize,
    pub(super) max_positions: usize,
    /// One per layer: the local or the global attention.
    pub(super) attention: Vec<AttentionParams>,
    /// Whether each layer is global.
    pub(super) global: Vec<bool>,
    /// The routed block of every layer, summed with the dense feed-forward.
    pub(super) moe: Option<MoeParams>,
    pub(super) layer: LayerParams,
    pub(super) stack: StackParams,
    pub(super) head_required: bool,
    pub(super) eos: BTreeSet<u32>,
    pub(super) bos: Option<u32>,
}

/// The text tower's config of an HF `gemma4` config: `text_config`, with the token ids the wrapper
/// config carries (it names the end-of-turn id the text config omits).
fn text_config(config: &serde_json::Value) -> Result<serde_json::Value, ModelError> {
    let wrong = |reason| ModelError::Config {
        family: FAMILY,
        field: "text_config",
        reason,
    };
    let mut text = config
        .get("text_config")
        .filter(|v| !v.is_null())
        .ok_or(wrong(ConfigReason::Missing))?
        .clone();
    let object = text.as_object_mut().ok_or(wrong(ConfigReason::WrongType))?;
    for key in ["eos_token_id", "bos_token_id"] {
        if let Some(value) = config.get(key).filter(|v| !v.is_null()) {
            object.insert(key.to_string(), value.clone());
        }
    }
    Ok(text)
}

/// One layer kind's RoPE block: `rope_parameters.{full_attention, sliding_attention}`.
fn rope_block<'a>(
    fields: &Fields<'_>,
    text: &'a serde_json::Value,
    kind: &'static str,
) -> Result<Option<&'a serde_json::Map<String, serde_json::Value>>, ModelError> {
    let Some(parameters) = text.get("rope_parameters").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let wrong = || fields.error("rope_parameters", ConfigReason::WrongType);
    match parameters.get(kind).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(block) => block.as_object().map(Some).ok_or_else(wrong),
    }
}

fn block_float(
    fields: &Fields<'_>,
    block: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &'static str,
) -> Result<Option<f32>, ModelError> {
    block
        .and_then(|b| b.get(key))
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| fields.error(key, ConfigReason::WrongType))
        })
        .transpose()
}

/// The size of the routed block.
#[derive(Clone, Copy)]
struct MoeShape {
    experts: usize,
    top_k: usize,
    inter: usize,
}

/// The facts of a text tower either checkpoint format names, before they become attention
/// parameters.
struct Tower {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    heads: usize,
    window: usize,
    /// Per layer: global (full attention) or local (sliding window).
    global: Vec<bool>,
    /// Per layer: the KV head count.
    kv_heads: Vec<usize>,
    /// The routed experts beside every layer's dense feed-forward, when the config has them.
    moe: Option<MoeShape>,
    head_dim: usize,
    global_head_dim: usize,
    local_rope: RopeParams,
    global_rope: RopeParams,
    /// Global layers read V from their K projection (an HF flag; a GGUF says so by omitting
    /// `attn_v`, which only the store shows).
    k_eq_v: bool,
    softcap: Option<f32>,
    eps: f32,
    tied: bool,
}

impl Gemma4Params {
    /// What a GGUF's tensors decide. llama.cpp's converter writes no `attn_v` for a global layer
    /// that reads V from K (checked against the real 31B file: `blk.5` has no `attn_v`, `blk.0`
    /// has one), so a global layer without it reads V from K; a local layer without it is a
    /// missing tensor, reported by the build. `rope_freqs.weight` is the global RoPE's
    /// per-frequency divisors, which carry the NoPE tail (ones, then `1e30`); a global layer
    /// without it would rotate every pair, so its absence is an error.
    pub(super) fn read_gguf_tensors(
        &mut self,
        store: &poot_quant::weights::WeightStore,
    ) -> Result<(), ModelError> {
        let param_error = |e: ParamError| e.for_family(FAMILY);
        let factors = store
            .get(GGUF_ROPE_FREQS)
            .map(|entry| rope_freq_factors(FAMILY, entry))
            .transpose()?;
        if factors.is_none() && self.global.iter().any(|&g| g) {
            return Err(ModelError::Config {
                family: FAMILY,
                field: GGUF_ROPE_FREQS,
                reason: ConfigReason::Missing,
            });
        }
        for l in (0..self.layers).filter(|&l| self.global[l]) {
            let mut a = self.attention[l];
            if !store.contains(&format!("blk.{l}.attn_v.weight")) {
                a = a.with_value_from_key().map_err(param_error)?;
            }
            if let Some(factors) = &factors {
                a = a.with_rope_freq_factors(factors).map_err(param_error)?;
            }
            self.attention[l] = a;
        }
        Ok(())
    }

    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        match raw {
            RawConfig::HfJson { config, generation } => {
                let text = text_config(config)?;
                let raw = RawConfig::HfJson {
                    config: &text,
                    generation: *generation,
                };
                let fields = Fields::new(&raw, FAMILY);
                let tower = hf_tower(&fields, &text, config)?;
                Self::assemble(&fields, tower)
            }
            RawConfig::Gguf(gguf) => {
                let fields = Fields::new(raw, FAMILY);
                let tower = gguf_tower(&fields, gguf)?;
                Self::assemble(&fields, tower)
            }
        }
    }

    fn assemble(fields: &Fields<'_>, tower: Tower) -> Result<Self, ModelError> {
        let param_error = |e: ParamError| e.for_family(FAMILY);
        let norm = NormParams::of(NormKind::Rms, tower.eps)
            .map_err(|_| fields.error("rms_norm_eps", ConfigReason::NotFinitePositive))?;
        let attention = (0..tower.layers)
            .map(|l| {
                let global = tower.global[l];
                let (dim, rope) = if global {
                    (tower.global_head_dim, tower.global_rope)
                } else {
                    (tower.head_dim, tower.local_rope)
                };
                let mut a = AttentionParams::new(tower.heads, tower.kv_heads[l], dim, false, rope)
                    .map_err(param_error)?
                    .with_qk_norm(QkNorm::PerHead(norm))
                    .with_value_norm(norm)
                    .with_scale(1.0)
                    .map_err(param_error)?;
                if !global {
                    a = a.with_window(tower.window).map_err(param_error)?;
                } else if tower.k_eq_v {
                    a = a.with_value_from_key().map_err(param_error)?;
                }
                Ok(a)
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        let mut stack = StackParams::new(norm)
            .with_embed_scale((tower.width as f32).sqrt())
            .map_err(param_error)?;
        if let Some(cap) = tower.softcap {
            stack = stack.with_softcap(cap).map_err(param_error)?;
        }
        let moe = tower
            .moe
            .map(|m| {
                MoeParams::new(
                    FAMILY,
                    tower.width,
                    m.inter,
                    m.experts,
                    m.top_k,
                    RouterKind::SoftmaxTopK { renormalize: true },
                )?
                .with_glu(Glu::Gelu)
                .with_gate_up(GateUpForm::FusedHalves)
                .with_scaled_router_input(norm)
                .with_expert_scale()
                .map_err(|e| e.for_family(FAMILY))
            })
            .transpose()?;
        Ok(Self {
            attention,
            global: tower.global,
            moe,
            layer: LayerParams::new(norm).with_placement(NormPlacement::Sandwich),
            stack,
            vocab: tower.vocab,
            width: tower.width,
            inter: tower.inter,
            layers: tower.layers,
            max_positions: tower.max_positions,
            head_required: !tower.tied,
            eos: fields.eos(Some(1))?,
            bos: fields.bos()?,
        })
    }
}

/// Refuse the features the body does not trace: per-layer input embeddings and cross-layer KV
/// sharing.
fn refuse_untraced(
    fields: &Fields<'_>,
    per_layer_input: Field,
    shared_kv: Field,
) -> Result<(), ModelError> {
    for field in [per_layer_input, shared_kv] {
        if fields.opt_u64(field)?.unwrap_or(0) != 0 {
            return Err(fields.error(fields.key(field), ConfigReason::Unsupported));
        }
    }
    Ok(())
}

fn hf_tower(
    fields: &Fields<'_>,
    text: &serde_json::Value,
    wrapper: &serde_json::Value,
) -> Result<Tower, ModelError> {
    let count = |key: &'static str, default: Option<usize>| fields.count(Field::hf(key), default);
    let heads = count("num_attention_heads", None)?;
    let head_dim = count("head_dim", None)?;
    let kv_heads = count("num_key_value_heads", Some(heads))?;
    let global_head_dim = count("global_head_dim", Some(head_dim))?;
    let global_kv_heads = count("num_global_key_value_heads", Some(kv_heads))?;
    let layers = count("num_hidden_layers", None)?;
    refuse_untraced(
        fields,
        Field::hf("hidden_size_per_layer_input"),
        Field::hf("num_kv_shared_layers"),
    )?;
    // The double-wide MLP widens the shared-KV layers' feed-forward, which this body does not trace.
    if fields.flag(Field::hf("use_double_wide_mlp"), false)? {
        return Err(fields.error("use_double_wide_mlp", ConfigReason::Unsupported));
    }
    let moe = if fields.flag(Field::hf("enable_moe_block"), false)? {
        Some(MoeShape {
            experts: count("num_experts", None)?,
            top_k: count("top_k_experts", None)?,
            inter: count("moe_intermediate_size", None)?,
        })
    } else {
        None
    };
    if let Some(activation) = text.get("hidden_activation").filter(|v| !v.is_null())
        && activation.as_str() != Some(ACTIVATION)
    {
        return Err(fields.error("hidden_activation", ConfigReason::Unsupported));
    }
    let global = layer_kinds(fields, text, layers)?;
    let tied = fields.flag(Field::hf("tie_word_embeddings"), true)?
        && wrapper
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
    Ok(Tower {
        vocab: fields.vocab()?,
        width: count("hidden_size", None)?,
        inter: count("intermediate_size", None)?,
        layers,
        max_positions: count("max_position_embeddings", None)?,
        heads,
        window: count("sliding_window", None)?,
        moe,
        kv_heads: global
            .iter()
            .map(|&g| if g { global_kv_heads } else { kv_heads })
            .collect(),
        global,
        head_dim,
        global_head_dim,
        local_rope: kind_rope(fields, text, "sliding_attention", head_dim)?,
        global_rope: kind_rope(fields, text, "full_attention", global_head_dim)?,
        k_eq_v: fields.flag(Field::hf("attention_k_eq_v"), false)?,
        softcap: fields.opt_f32(Field::hf("final_logit_softcapping"))?,
        eps: fields.float(Field::hf("rms_norm_eps"), 1e-6)?,
        tied,
    })
}

/// llama.cpp's `gemma4` metadata: `key_length` is the global head width and `key_length_swa` the
/// local one, `head_count_kv` and `sliding_window_pattern` are per-layer arrays (`true`: local),
/// and the global RoPE's NoPE tail lives in the `rope_freqs.weight` tensor, which the build reads.
fn gguf_tower(fields: &Fields<'_>, gguf: &poot_load::gguf::GgufIndex) -> Result<Tower, ModelError> {
    let count = |key: &'static str| fields.count(Field::new("", key), None);
    let layers = count("gemma4.block_count")?;
    let per_layer = |key: &'static str| -> Result<&[poot_load::gguf::GgufValue], ModelError> {
        let array = gguf
            .get(key)
            .ok_or_else(|| fields.error(key, ConfigReason::Missing))?
            .as_array()
            .ok_or_else(|| fields.error(key, ConfigReason::WrongType))?;
        if array.len() != layers {
            return Err(fields.error(key, ConfigReason::WrongType));
        }
        Ok(array)
    };
    let global = per_layer("gemma4.attention.sliding_window_pattern")?
        .iter()
        .map(|v| {
            v.as_bool().map(|local| !local).ok_or_else(|| {
                fields.error(
                    "gemma4.attention.sliding_window_pattern",
                    ConfigReason::WrongType,
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let kv_heads = per_layer("gemma4.attention.head_count_kv")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| {
                    fields.error("gemma4.attention.head_count_kv", ConfigReason::WrongType)
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    refuse_untraced(
        fields,
        Field::new("", "gemma4.embedding_length_per_layer_input"),
        Field::new("", "gemma4.attention.shared_kv_layers"),
    )?;
    let moe = if gguf.get("gemma4.expert_count").is_some() {
        Some(MoeShape {
            experts: count("gemma4.expert_count")?,
            top_k: count("gemma4.expert_used_count")?,
            inter: count("gemma4.expert_feed_forward_length")?,
        })
    } else {
        None
    };
    let float = |key: &'static str, default: f32| fields.float(Field::new("", key), default);
    let (head_dim, global_head_dim) = (
        count("gemma4.attention.key_length_swa")?,
        count("gemma4.attention.key_length")?,
    );
    let rope = |dim: usize, key: &'static str, default: f32| {
        RopeParams::new(dim, float(key, default)?, &RopeFlavor::Plain)
            .map_err(|e| e.for_family(FAMILY))
    };
    Ok(Tower {
        vocab: fields.vocab()?,
        width: count("gemma4.embedding_length")?,
        inter: count("gemma4.feed_forward_length")?,
        layers,
        max_positions: count("gemma4.context_length")?,
        heads: count("gemma4.attention.head_count")?,
        window: count("gemma4.attention.sliding_window")?,
        global,
        kv_heads,
        moe,
        head_dim,
        global_head_dim,
        local_rope: rope(head_dim, "gemma4.rope.freq_base_swa", DEFAULT_LOCAL_THETA)?,
        global_rope: rope(
            global_head_dim,
            "gemma4.rope.freq_base",
            DEFAULT_GLOBAL_THETA,
        )?,
        k_eq_v: false,
        softcap: fields.opt_f32(Field::new("", "gemma4.final_logit_softcapping"))?,
        eps: float("gemma4.attention.layer_norm_rms_epsilon", 1e-6)?,
        // A GGUF without `output.weight` ties its head: the name table says so.
        tied: true,
    })
}

/// Whether each layer is global, from `layer_types`.
fn layer_kinds(
    fields: &Fields<'_>,
    text: &serde_json::Value,
    layers: usize,
) -> Result<Vec<bool>, ModelError> {
    let wrong = || fields.error("layer_types", ConfigReason::WrongType);
    let types = text
        .get("layer_types")
        .filter(|v| !v.is_null())
        .ok_or_else(|| fields.error("layer_types", ConfigReason::Missing))?
        .as_array()
        .ok_or_else(wrong)?;
    if types.len() != layers {
        return Err(wrong());
    }
    types
        .iter()
        .map(|t| match t.as_str() {
            Some("full_attention") => Ok(true),
            Some("sliding_attention") => Ok(false),
            _ => Err(wrong()),
        })
        .collect()
}

/// The RoPE of one layer kind over `head_dim`-wide heads: `rope_type` `default` rotates every
/// pair, `proportional` only the first `partial_rotary_factor` of them.
fn kind_rope(
    fields: &Fields<'_>,
    text: &serde_json::Value,
    kind: &'static str,
    head_dim: usize,
) -> Result<RopeParams, ModelError> {
    let block = rope_block(fields, text, kind)?;
    let default_theta = if kind == "full_attention" {
        DEFAULT_GLOBAL_THETA
    } else {
        DEFAULT_LOCAL_THETA
    };
    let theta = block_float(fields, block, "rope_theta")?.unwrap_or(default_theta);
    let rope_type = match block.and_then(|b| b.get("rope_type")) {
        None => "default",
        Some(t) => t
            .as_str()
            .ok_or_else(|| fields.error("rope_parameters", ConfigReason::WrongType))?,
    };
    let flavor = match rope_type {
        "default" => RopeFlavor::Plain,
        "proportional" => {
            if block_float(fields, block, "factor")?.is_some_and(|f| f != 1.0) {
                return Err(fields.error("rope_parameters", ConfigReason::Unsupported));
            }
            let fraction = block_float(fields, block, "partial_rotary_factor")?.unwrap_or(1.0);
            if !(fraction.is_finite() && fraction > 0.0 && fraction <= 1.0) {
                return Err(fields.error("partial_rotary_factor", ConfigReason::WrongType));
            }
            RopeFlavor::Proportional {
                rotated: (fraction * head_dim as f32 / 2.0).floor() as usize,
            }
        }
        _ => return Err(fields.error("rope_parameters", ConfigReason::Unsupported)),
    };
    RopeParams::new(head_dim, theta, &flavor).map_err(|e| e.for_family(FAMILY))
}
