//! Qwen3-Next (HF `qwen3_next`; GGUF `qwen35moe`) behind [`Model`]: a hybrid stack whose mixer is a
//! Gated DeltaNet linear-attention layer ([`gdn`]) on most layers and gated full attention
//! (`components::attention` with the output gate) on every `full_attention_interval`-th, each followed
//! by the same routed mixture of experts with a sigmoid-gated shared expert ([`moe`]).
//!
//! The body is `standard_stack` over `standard_layer`: `x += mixer(norm(x)); x += moe(norm(x))`. The
//! step carries the attention layers' KV caches and, for each GDN layer, its conv window and its
//! state matrices, so a prompt prefilled in chunks continues exactly where the previous chunk stopped
//! and one token is a decode step.
//!
//! The two checkpoint formats spell the layer differently (see `components::gdn`): an HF checkpoint
//! stores the fused `in_proj_qkvz`/`in_proj_ba`, `A_log`, value heads grouped by key head and
//! zero-centered norms (`1 + w`); a GGUF stores one projection per quantity, `ssm_a` already negative,
//! tiled value heads and norms with the one folded in. Both build the same graph.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{AttnRole, ExpertProj, NormRole, WeightMap, WeightRole, WeightStore};

use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::gdn::{GdnParams, GdnWeights, gdn};
use crate::components::gdn_names::{GdnTable, nextn_field, trunk_layers};
use crate::components::linear::WeightError;
use crate::components::moe::{self, MoeParams, MoeWeights};
use crate::components::norm::NormParams;
use crate::components::router::RouterKind;
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, decoder_program,
    standard_layer, standard_stack,
};
use crate::io::{ModelSchema, StateSpec};
use crate::model::{ConfigReason, FamilyKey, Model, ModelConfig, ModelError, StepSpec, TraceError};
use crate::names::{ExpertNames, MoeNames, SharedNames, TensorRows, linear_key, map_moe_layer};
#[cfg(feature = "test-support")]
use crate::registry::Fixture;
use crate::registry::{ConfigSource, FamilyEntry, ModelParts, RawConfig};
use poot_text::{ChatFormat, HermesToolCodec};

/// Qwen3-Next's [`FamilyKey`].
pub const FAMILY: FamilyKey = FamilyKey::new("qwen3next");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "qwen3_next"),
        (ConfigSource::GgufArchitecture, "qwen35moe"),
    ],
    build,
};

/// Qwen3-Next's config facts: `Qwen3NextConfig`'s defaults. The class declares no `eos_token_id`: a
/// checkpoint names it. A GGUF must name its RoPE base (`gguf_theta: None`).
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("qwen35moe"),
    hf_theta: 10_000.0,
    gguf_theta: None,
    eps: 1e-6,
    eos: None,
};

/// The per-expert FFN width, not the (unused) dense layers' `intermediate_size`.
const EXPERT_INTER: Field = Field::new(
    "moe_intermediate_size",
    "qwen35moe.expert_feed_forward_length",
);
const SHARED_INTER: Field = Field::new(
    "shared_expert_intermediate_size",
    "qwen35moe.expert_shared_feed_forward_length",
);
const EXPERTS: Field = Field::new("num_experts", "qwen35moe.expert_count");
const TOP_K: Field = Field::new("num_experts_per_tok", "qwen35moe.expert_used_count");
const NORM_TOPK_PROB: Field = Field::new("norm_topk_prob", "qwen35moe.expert_weights_norm");
const KEY_HEADS: Field = Field::new("linear_num_key_heads", "qwen35moe.ssm.group_count");
const VALUE_HEADS: Field = Field::new("linear_num_value_heads", "qwen35moe.ssm.time_step_rank");
const KEY_HEAD_DIM: Field = Field::new("linear_key_head_dim", "qwen35moe.ssm.state_size");
const VALUE_HEAD_DIM: Field = Field::hf("linear_value_head_dim");
const CONV_KERNEL: Field = Field::new("linear_conv_kernel_dim", "qwen35moe.ssm.conv_kernel");
const INNER_SIZE: Field = Field::new("", "qwen35moe.ssm.inner_size");
const INTERVAL: Field = Field::new(
    "full_attention_interval",
    "qwen35moe.full_attention_interval",
);
const ROTARY_DIM: Field = Field::new("", "qwen35moe.rope.dimension_count");
const NEXTN_LAYERS: Field = nextn_field!("qwen35moe");

/// One layer's mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet linear attention.
    LinearAttention,
    /// Gated full attention.
    FullAttention,
}

/// Qwen3-Next's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3NextParams {
    vocab: usize,
    width: usize,
    max_positions: usize,
    kinds: Vec<LayerKind>,
    layer: LayerParams,
    stack: StackParams,
    attention: AttentionParams,
    gdn: GdnParams,
    moe: MoeParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl Qwen3NextParams {
    /// Reads the typed config from an HF `config.json` or a GGUF index (`qwen35moe`); a field the family
    /// cannot serve is a typed [`ModelError::Config`] naming it.
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read_with_inter(&fields, &SPEC, EXPERT_INTER)?;
        let layers = trunk_layers(&fields, NEXTN_LAYERS, dims.layers)?;
        let table = names_of(raw).gdn;
        let hf = matches!(raw, RawConfig::HfJson { .. });
        if hf {
            for unsupported in ["attention_bias", "use_sliding_window"] {
                if fields.flag(Field::hf(unsupported), false)? {
                    return Err(fields.error(unsupported, ConfigReason::Unsupported));
                }
            }
            // Every layer routes: a dense fallback layer has no tensors this family maps.
            if fields
                .opt_u64(Field::hf("decoder_sparse_step"))?
                .unwrap_or(1)
                != 1
            {
                return Err(fields.error("decoder_sparse_step", ConfigReason::Unsupported));
            }
            if let RawConfig::HfJson { config, .. } = raw {
                match config.get("mlp_only_layers").filter(|v| !v.is_null()) {
                    None => {}
                    Some(layers) => match layers.as_array() {
                        Some(layers) if layers.is_empty() => {}
                        Some(_) => {
                            return Err(fields.error("mlp_only_layers", ConfigReason::Unsupported));
                        }
                        None => {
                            return Err(fields.error("mlp_only_layers", ConfigReason::WrongType));
                        }
                    },
                }
            }
        }
        // The residual-stream and projection norms: HF stores `(1 + w)`, a GGUF folds the one in.
        let eps = dims.eps;
        let norm = NormParams::of(table.convention.norm_fold.norm_kind(), eps)
            .map_err(|e| e.for_family(FAMILY))?;
        let rotary = match raw {
            RawConfig::HfJson { .. } => {
                let factor = fields.float(Field::hf("partial_rotary_factor"), 0.25)?;
                if !(factor.is_finite() && factor > 0.0 && factor <= 1.0) {
                    return Err(
                        fields.error("partial_rotary_factor", ConfigReason::NotFinitePositive)
                    );
                }
                ((dims.head_dim as f32 * factor) as usize).max(1)
            }
            RawConfig::Gguf(_) => match fields.opt_u64(ROTARY_DIM)? {
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
        if let Some(value_dim) = fields.opt_u64(VALUE_HEAD_DIM)?
            && value_dim as usize != head_dim
        {
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
        let gdn = table
            .params(
                key_heads,
                value_heads,
                head_dim,
                fields.count(CONV_KERNEL, None)?,
                NormParams::new(eps).map_err(|e| e.for_family(FAMILY))?,
            )
            .map_err(|e| e.for_family(FAMILY))?;

        // HF's class default renormalizes the selected routes; llama.cpp's graph always does.
        let renormalize = fields.flag(NORM_TOPK_PROB, true)?;
        let moe = MoeParams::new(
            FAMILY,
            dims.width,
            dims.inter,
            fields.count(EXPERTS, None)?,
            fields.count(TOP_K, None)?,
            RouterKind::SoftmaxTopK { renormalize },
        )?
        .with_shared(fields.count(SHARED_INTER, None)?)
        .and_then(|m| m.with_shared_gate())
        .map_err(|e| e.for_family(FAMILY))?;

        Ok(Self {
            kinds: layer_kinds(&fields, layers)?,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            attention,
            gdn,
            moe,
            vocab: dims.vocab,
            width: dims.width,
            max_positions: dims.max_positions,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
        })
    }
}

/// Which layers are full attention: HF names them in `layer_types` (`"full_attention"` or
/// `"linear_attention"` per layer) or by `full_attention_interval` (default 4), where layer `i` is full
/// attention iff `(i + 1)` is a multiple of the interval; a GGUF states the interval.
fn layer_kinds(fields: &Fields<'_>, layers: usize) -> Result<Vec<LayerKind>, ModelError> {
    if let RawConfig::HfJson { config, .. } = fields.raw()
        && let Some(types) = config.get("layer_types").filter(|v| !v.is_null())
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

/// A checkpoint format's tensor names: the layer prefix and, per tensor, its suffix (`linear`
/// stems also read the packed loader's bare linear id, see [`Names::linear`]).
struct Names {
    prefix: &'static str,
    embed: &'static str,
    final_norm: &'static str,
    head: &'static str,
    attn_norm: &'static str,
    ffn_norm: &'static str,
    attention: AttentionNames,
    gdn: GdnTable,
    moe: MoeNames,
}

struct AttentionNames {
    q: &'static str,
    k: &'static str,
    v: &'static str,
    o: &'static str,
    q_norm: &'static str,
    k_norm: &'static str,
}

const HF_NAMES: Names = Names {
    prefix: "model.layers.",
    embed: "model.embed_tokens.weight",
    final_norm: "model.norm.weight",
    head: "lm_head",
    attn_norm: ".input_layernorm.weight",
    ffn_norm: ".post_attention_layernorm.weight",
    attention: AttentionNames {
        q: ".self_attn.q_proj",
        k: ".self_attn.k_proj",
        v: ".self_attn.v_proj",
        o: ".self_attn.o_proj",
        q_norm: ".self_attn.q_norm.weight",
        k_norm: ".self_attn.k_norm.weight",
    },
    gdn: GdnTable::HF_FUSED,
    moe: MoeNames {
        prefix: "model.layers.",
        router: ".mlp.gate",
        router_bias: None,
        shared: Some(SharedNames {
            gate: ".mlp.shared_expert.gate_proj",
            up: ".mlp.shared_expert.up_proj",
            down: ".mlp.shared_expert.down_proj",
            scalar_gate: Some(".mlp.shared_expert_gate"),
        }),
        experts: &[
            ExpertNames {
                proj: ExpertProj::Gate,
                stack: None,
                each: Some((".mlp.experts.", ".gate_proj.weight")),
                bias: None,
            },
            ExpertNames {
                proj: ExpertProj::Up,
                stack: None,
                each: Some((".mlp.experts.", ".up_proj.weight")),
                bias: None,
            },
            ExpertNames {
                proj: ExpertProj::Down,
                stack: None,
                each: Some((".mlp.experts.", ".down_proj.weight")),
                bias: None,
            },
        ],
    },
};

const GGUF_NAMES: Names = Names {
    prefix: "blk.",
    embed: "token_embd.weight",
    final_norm: "output_norm.weight",
    head: "output",
    attn_norm: ".attn_norm.weight",
    ffn_norm: ".post_attention_norm.weight",
    attention: AttentionNames {
        q: ".attn_q",
        k: ".attn_k",
        v: ".attn_v",
        o: ".attn_output",
        q_norm: ".attn_q_norm.weight",
        k_norm: ".attn_k_norm.weight",
    },
    gdn: GdnTable::GGUF,
    moe: MoeNames {
        prefix: "blk.",
        router: ".ffn_gate_inp",
        router_bias: None,
        shared: Some(SharedNames {
            gate: ".ffn_gate_shexp",
            up: ".ffn_up_shexp",
            down: ".ffn_down_shexp",
            scalar_gate: Some(".ffn_gate_inp_shexp"),
        }),
        experts: &[
            ExpertNames {
                proj: ExpertProj::Gate,
                stack: Some(".ffn_gate_exps.weight"),
                each: None,
                bias: None,
            },
            ExpertNames {
                proj: ExpertProj::Up,
                stack: Some(".ffn_up_exps.weight"),
                each: None,
                bias: None,
            },
            ExpertNames {
                proj: ExpertProj::Down,
                stack: Some(".ffn_down_exps.weight"),
                each: None,
                bias: None,
            },
        ],
    },
};

/// The weight map of a checkpoint: the model-level weights and, per layer, the two norms, the
/// mixer's weights its kind names and the routed block.
fn weights_of(
    p: &Qwen3NextParams,
    store: &WeightStore,
    names: &Names,
) -> Result<WeightMap, ModelError> {
    let mut rows = TensorRows::new(FAMILY, store);
    rows.tensor(WeightRole::Embed, None, names.embed.to_string())?;
    rows.tensor(WeightRole::FinalNorm, None, names.final_norm.to_string())?;
    rows.head(names.embed, names.head, p.head_required)?;

    // The stored key of a linear: `{stem}.weight`, or the bare stem a packed loader names it.
    let linear = |stem: &str| linear_key(store, stem);
    for (l, kind) in p.kinds.iter().enumerate() {
        let layer = |suffix: &str| format!("{}{l}{suffix}", names.prefix);
        let at = Some(l);
        rows.tensor(WeightRole::Norm(NormRole::Attn), at, layer(names.attn_norm))?;
        rows.tensor(WeightRole::Norm(NormRole::Ffn), at, layer(names.ffn_norm))?;
        match kind {
            LayerKind::FullAttention => {
                let a = &names.attention;
                for (role, stem) in [
                    (AttnRole::Q, a.q),
                    (AttnRole::K, a.k),
                    (AttnRole::V, a.v),
                    (AttnRole::O, a.o),
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
    }
    for l in 0..p.kinds.len() {
        let mapped = map_moe_layer(&mut rows.map, store, &names.moe, l, p.moe.experts());
        mapped.map_err(|source| rows.error(source))?;
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
    moe: MoeWeights,
}

/// The built Qwen3-Next model: config, name-resolved weights and the hybrid GDN / full-attention
/// layer stack with its routed experts. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct Qwen3Next {
    params: Qwen3NextParams,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
    schema: ModelSchema,
}

/// The tensor names of the checkpoint format `raw` comes from.
fn names_of(raw: &RawConfig<'_>) -> &'static Names {
    match raw.source() {
        ConfigSource::HfModelType => &HF_NAMES,
        ConfigSource::GgufArchitecture => &GGUF_NAMES,
    }
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let p = Qwen3NextParams::from_raw(raw)?;
    let weights = weights_of(&p, store, names_of(raw))?;
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
                    moe: MoeWeights::new(&weights, l, &p.moe)?,
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
    Ok(ModelParts::new(Box::new(Qwen3Next {
        params: p,
        weights,
        stack,
        layers,
        config,
        schema,
    })))
}

impl Model for Qwen3Next {
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
                    |b, BlockInput { block: h, .. }| {
                        moe::moe(b, &step, l, h, None, &layer.moe, &p.moe)
                    },
                )
            },
        )?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a four-layer hybrid (three GDN layers, then gated attention) over four
/// routed experts and a gated shared expert. Projection widths are multiples of 32 where a Q8_0 copy
/// needs them.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_next",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "tie_word_embeddings": false,
        "eos_token_id": 47,
    })
}

/// The tensors of a Qwen3-Next checkpoint of `config`, by HF name.
#[cfg(feature = "test-support")]
pub(crate) fn hf_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| crate::registry::Fixture::count(config, f);
    let (vocab, w) = (n("vocab_size"), n("hidden_size"));
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
    let (experts, inter, shared) = (
        n("num_experts"),
        n("moe_intermediate_size"),
        n("shared_expert_intermediate_size"),
    );
    let layers = n("num_hidden_layers");
    let interval = config["full_attention_interval"].as_u64().unwrap_or(4) as usize;
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_string(), vec![vocab, w]),
        ("model.norm.weight".to_string(), vec![w]),
        ("lm_head.weight".to_string(), vec![vocab, w]),
    ];
    for l in 0..layers {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        shapes.push((k("input_layernorm.weight"), vec![w]));
        shapes.push((k("post_attention_layernorm.weight"), vec![w]));
        if (l + 1).is_multiple_of(interval) {
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
                (
                    k("linear_attn.in_proj_qkvz.weight"),
                    vec![conv + value_dim, w],
                ),
                (k("linear_attn.in_proj_ba.weight"), vec![2 * hv, w]),
                (k("linear_attn.conv1d.weight"), vec![conv, 1, kernel]),
                (k("linear_attn.dt_bias"), vec![hv]),
                (k("linear_attn.A_log"), vec![hv]),
                (k("linear_attn.norm.weight"), vec![d]),
                (k("linear_attn.out_proj.weight"), vec![w, value_dim]),
            ]);
        }
        shapes.push((k("mlp.gate.weight"), vec![experts, w]));
        for e in 0..experts {
            shapes.extend([
                (
                    k(&format!("mlp.experts.{e}.gate_proj.weight")),
                    vec![inter, w],
                ),
                (
                    k(&format!("mlp.experts.{e}.up_proj.weight")),
                    vec![inter, w],
                ),
                (
                    k(&format!("mlp.experts.{e}.down_proj.weight")),
                    vec![w, inter],
                ),
            ]);
        }
        shapes.extend([
            (k("mlp.shared_expert.gate_proj.weight"), vec![shared, w]),
            (k("mlp.shared_expert.up_proj.weight"), vec![shared, w]),
            (k("mlp.shared_expert.down_proj.weight"), vec![w, shared]),
            (k("mlp.shared_expert_gate.weight"), vec![1, w]),
        ]);
    }
    shapes
}

#[cfg(feature = "test-support")]
pub(crate) fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_shapes(&config);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_load::gguf::GgufValue;
    use poot_quant::weights::HandleFormat;
    use serde_json::json;

    use super::*;
    use crate::components::gdn_names::converter::{
        HfProjections, LayerDims, layer_metadata, nextn_key, plus_one, write_layer,
        write_nextn_block,
    };
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, GGML_F32, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named,
        build as built, close, close_all, decoded, f32_entry, gguf_tensor, greedy_tokens,
        q8_0_store, read_back_owned, step_shape, values,
    };
    use crate::model::LogitRows;
    use crate::registry::Registry;

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
            preprocessor: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// SC-006 (F2): prefill of six tokens equals two prefills of three equals prefill of three then
    /// three decode steps equals six decode steps, at every logit and every carried state element
    /// (the attention cache, every GDN layer's conv window and state matrices). Mutation: ignore the
    /// carried conv window in a continued chunk (today's zero pad); the second chunk goes red at the
    /// first GDN layer.
    #[test]
    fn chunked_prefill_continuing_from_the_carried_state_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// The step carries the two caches of the attention layer and the conv window and state
    /// matrices of each GDN layer: the state of three GDN layers then one attention layer is eight
    /// values, in layer order.
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

    /// SC-007 (F6): a store missing one required tensor, of any layer kind and of every role a layer
    /// reads, fails `Registry::build` with the typed missing-weight error naming it, before any
    /// trace. Mutation: discard the error of `map.map` in `weights_of`; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            "model.layers.1.linear_attn.in_proj_ba.weight",
            "model.layers.2.linear_attn.conv1d.weight",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.1.linear_attn.A_log",
            "model.layers.2.linear_attn.norm.weight",
            "model.layers.0.linear_attn.out_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.3.self_attn.o_proj.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.2.mlp.experts.3.down_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.3.mlp.shared_expert_gate.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
        }
    }

    /// The BF16 store and its Q8_0 copy trace through the one body and equal the same body over the
    /// store decoded to F32 (prefill and a decode step). The fused GDN projections and the
    /// experts stay dense: a random Q8_0 payload has wild block scales, which overflow the
    /// recurrence's decay in both the packed and the decoded run.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body() {
        let fx = fixture();
        let packable = |key: &str| {
            (key.contains("self_attn.") || key.contains("shared_expert."))
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
    /// the BF16 store (the GDN and attention components are pinned against f64 references in their
    /// own modules). Mutation: swap the carried conv window for zeros in a continued step (the
    /// decode steps); the tokens change.
    #[test]
    fn the_fixture_generates_the_recorded_tokens() {
        const RECORDED: [u32; 24] = [
            20, 2, 20, 3, 6, 27, 25, 17, 18, 24, 20, 18, 45, 34, 45, 5, 45, 12, 35, 40, 5, 46, 35,
            29,
        ];
        let fx = fixture();
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

    /// A config the family cannot trace is a typed error naming the field, never a panic.
    #[test]
    fn a_config_qwen3next_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "linear_num_value_heads",
                json!(3),
                "linear_num_value_heads",
                ConfigReason::NotDivisible { by: 2 },
            ),
            (
                "linear_num_key_heads",
                json!(0),
                "linear_num_key_heads",
                ConfigReason::Zero,
            ),
            (
                "linear_conv_kernel_dim",
                json!(1),
                "conv_kernel",
                ConfigReason::Zero,
            ),
            (
                "linear_value_head_dim",
                json!(4),
                "linear_value_head_dim",
                ConfigReason::Unsupported,
            ),
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "decoder_sparse_step",
                json!(2),
                "decoder_sparse_step",
                ConfigReason::Unsupported,
            ),
            (
                "mlp_only_layers",
                json!([0]),
                "mlp_only_layers",
                ConfigReason::Unsupported,
            ),
            (
                "layer_types",
                json!(["full_attention"]),
                "layer_types",
                ConfigReason::Length { expected: 4 },
            ),
            (
                "layer_types",
                json!([
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "sliding"
                ]),
                "layer_types",
                ConfigReason::Unsupported,
            ),
            (
                "full_attention_interval",
                json!(0),
                "full_attention_interval",
                ConfigReason::Zero,
            ),
            (
                "num_experts_per_tok",
                json!(5),
                "top_k",
                ConfigReason::Exceeds { max: 4 },
            ),
            (
                "partial_rotary_factor",
                json!(1.5),
                "partial_rotary_factor",
                ConfigReason::NotFinitePositive,
            ),
        ];
        for (key, value, field, reason) in cases {
            let mut config = fixture_config();
            config[*key] = value.clone();
            assert_eq!(
                config_error_of(&config),
                (*field, *reason),
                "{key} = {value}"
            );
        }
    }

    /// An explicit `layer_types` list and the interval name the same schedule when they agree, and
    /// the list wins when they differ: three linear layers then one full here, one full first there.
    #[test]
    fn layer_types_name_the_schedule_and_the_interval_is_its_default() {
        let fx = fixture();
        let kinds = |config: &serde_json::Value| {
            Qwen3NextParams::from_raw(&RawConfig::HfJson {
                config,
                generation: None,
                preprocessor: None,
            })
            .unwrap()
            .kinds
        };
        use LayerKind::{FullAttention as F, LinearAttention as L};
        assert_eq!(kinds(&fx.config), [L, L, L, F]);
        let mut config = fx.config.clone();
        config["full_attention_interval"] = json!(2);
        assert_eq!(kinds(&config), [L, F, L, F]);
        config["layer_types"] = json!([
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]);
        assert_eq!(kinds(&config), [F, L, L, F]);
    }

    /// The registry resolves both spellings to this family.
    #[test]
    fn the_family_is_keyed_by_the_hf_model_type_and_the_gguf_architecture() {
        assert_eq!(
            ENTRY.keys,
            &[
                (ConfigSource::HfModelType, "qwen3_next"),
                (ConfigSource::GgufArchitecture, "qwen35moe"),
            ]
        );
    }

    // ---- the GGUF spelling ---------------------------------------------------------------------

    /// The GGUF checkpoint llama.cpp's converter writes for the HF `fixture`: separate projections
    /// (the fused `in_proj_qkvz` and `in_proj_ba` taken apart), every per-value-head tensor
    /// reordered from HF's grouped order to tiled, `ssm_a = -exp(A_log)`, the norms with their one
    /// folded in (the GDN's gated norm has none), the experts stacked and, when `nextn` is set, the
    /// extra block the file appends for multi-token prediction. Written from the converter's rules
    /// (`components::gdn_names::converter`), independently of the family's own GGUF reader.
    /// A GGUF tensor as the reader takes it: name, dims, GGML type and bytes.
    type GgufTensor = (String, Vec<u64>, u32, Vec<u8>);

    fn gguf_of(fx: &Fixture, nextn: bool) -> (Vec<(String, GgufValue)>, Vec<GgufTensor>) {
        let n = |f: &str| crate::registry::Fixture::count(&fx.config, f);
        let gdn = LayerDims {
            width: n("hidden_size"),
            key_heads: n("linear_num_key_heads"),
            value_heads: n("linear_num_value_heads"),
            head_dim: n("linear_key_head_dim"),
            kernel: n("linear_conv_kernel_dim"),
        };
        let w = gdn.width;
        let experts = n("num_experts");
        let layers = n("num_hidden_layers");
        let vals = |key: &str| values(fx.store.get(key).unwrap_or_else(|| panic!("{key}")));
        let mut tensors: Vec<GgufTensor> = Vec::new();
        let mut put = |name: String, shape: Vec<usize>, data: Vec<f32>| {
            assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
            tensors.push(gguf_tensor(&name, &f32_entry(shape, &data)));
        };
        let vocab = n("vocab_size");
        put(
            "token_embd.weight".into(),
            vec![vocab, w],
            vals("model.embed_tokens.weight"),
        );
        put(
            "output_norm.weight".into(),
            vec![w],
            plus_one(vals("model.norm.weight")),
        );
        put(
            "output.weight".into(),
            vec![vocab, w],
            vals("lm_head.weight"),
        );
        for l in 0..layers {
            let hf = |s: &str| format!("model.layers.{l}.{s}");
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
                let (heads, kv, d) = (
                    n("num_attention_heads"),
                    n("num_key_value_heads"),
                    n("head_dim"),
                );
                put(
                    gg("attn_q.weight"),
                    vec![2 * heads * d, w],
                    vals(&hf("self_attn.q_proj.weight")),
                );
                put(
                    gg("attn_k.weight"),
                    vec![kv * d, w],
                    vals(&hf("self_attn.k_proj.weight")),
                );
                put(
                    gg("attn_v.weight"),
                    vec![kv * d, w],
                    vals(&hf("self_attn.v_proj.weight")),
                );
                put(
                    gg("attn_output.weight"),
                    vec![w, heads * d],
                    vals(&hf("self_attn.o_proj.weight")),
                );
                put(
                    gg("attn_q_norm.weight"),
                    vec![d],
                    plus_one(vals(&hf("self_attn.q_norm.weight"))),
                );
                put(
                    gg("attn_k_norm.weight"),
                    vec![d],
                    plus_one(vals(&hf("self_attn.k_norm.weight"))),
                );
            } else {
                write_layer(
                    gdn,
                    HfProjections::Fused,
                    &|suffix| vals(&hf(&format!("linear_attn.{suffix}"))),
                    &gg,
                    &mut put,
                );
            }
            let (inter, shared) = (
                n("moe_intermediate_size"),
                n("shared_expert_intermediate_size"),
            );
            put(
                gg("ffn_gate_inp.weight"),
                vec![experts, w],
                vals(&hf("mlp.gate.weight")),
            );
            for (proj, name, out, k) in [
                ("gate_proj", "ffn_gate_exps.weight", inter, w),
                ("up_proj", "ffn_up_exps.weight", inter, w),
                ("down_proj", "ffn_down_exps.weight", w, inter),
            ] {
                let stack: Vec<f32> = (0..experts)
                    .flat_map(|e| vals(&hf(&format!("mlp.experts.{e}.{proj}.weight"))))
                    .collect();
                put(gg(name), vec![experts, out, k], stack);
            }
            put(
                gg("ffn_gate_shexp.weight"),
                vec![shared, w],
                vals(&hf("mlp.shared_expert.gate_proj.weight")),
            );
            put(
                gg("ffn_up_shexp.weight"),
                vec![shared, w],
                vals(&hf("mlp.shared_expert.up_proj.weight")),
            );
            put(
                gg("ffn_down_shexp.weight"),
                vec![w, shared],
                vals(&hf("mlp.shared_expert.down_proj.weight")),
            );
            put(
                gg("ffn_gate_inp_shexp.weight"),
                vec![w],
                vals(&hf("mlp.shared_expert_gate.weight")),
            );
        }
        if nextn {
            write_nextn_block(w, layers, &mut put);
        }
        let u = |v: usize| GgufValue::U32(v as u32);
        let mut kvs = vec![
            (
                "general.architecture".to_string(),
                GgufValue::Str("qwen35moe".into()),
            ),
            ("qwen35moe.embedding_length".to_string(), u(w)),
            (
                "qwen35moe.block_count".to_string(),
                u(layers + usize::from(nextn)),
            ),
            (
                "qwen35moe.attention.head_count".to_string(),
                u(n("num_attention_heads")),
            ),
            (
                "qwen35moe.attention.head_count_kv".to_string(),
                u(n("num_key_value_heads")),
            ),
            (
                "qwen35moe.attention.key_length".to_string(),
                u(n("head_dim")),
            ),
            (
                "qwen35moe.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::F32(fx.config["rms_norm_eps"].as_f64().unwrap() as f32),
            ),
            (
                "qwen35moe.context_length".to_string(),
                u(n("max_position_embeddings")),
            ),
            (
                "qwen35moe.rope.freq_base".to_string(),
                GgufValue::F32(fx.config["rope_theta"].as_f64().unwrap() as f32),
            ),
            (
                "qwen35moe.rope.dimension_count".to_string(),
                u(
                    (n("head_dim") as f64 * fx.config["partial_rotary_factor"].as_f64().unwrap())
                        as usize,
                ),
            ),
            ("qwen35moe.expert_count".to_string(), u(experts)),
            (
                "qwen35moe.expert_used_count".to_string(),
                u(n("num_experts_per_tok")),
            ),
            (
                "qwen35moe.expert_feed_forward_length".to_string(),
                u(n("moe_intermediate_size")),
            ),
            (
                "qwen35moe.expert_shared_feed_forward_length".to_string(),
                u(n("shared_expert_intermediate_size")),
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
                u(n("eos_token_id")),
            ),
        ];
        kvs.extend(layer_metadata("qwen35moe", gdn, nextn));
        (kvs, tensors)
    }

    /// SC-005's checkpoint half: the GGUF the converter writes for the HF fixture (tiled value
    /// heads, separate projections, `ssm_a`, folded norms, stacked experts, a flat shared gate)
    /// resolves through the registry to this family and gives the same logits and carried state as
    /// the HF model, prefill and decode, at every position, with and without the appended
    /// multi-token-prediction block. Mutations: read the GGUF with `GdnHeadOrder::Grouped`, the value
    /// heads pair with the wrong key heads and the logits differ; drop the `trunk_layers` call from
    /// `Qwen3NextParams::from_raw`, the nextn block is traced as a fifth layer and the weight map
    /// names tensors the file lacks.
    #[test]
    fn the_gguf_spelling_of_the_fixture_equals_the_hf_model() {
        let fx = fixture();
        let hf = model_of(&fx.config, &fx.store);
        let prompt = [3, 17, 40, 8, 25, 9];
        let (want, want_state) = run_step(&*hf, &fx.store, CAP, &prompt, 0, &[]);
        let (want_decode, _) = run_step(&*hf, &fx.store, CAP, &[5], 6, &want_state);
        for nextn in [false, true] {
            let (kvs, tensors) = gguf_of(&fx, nextn);
            assert!(tensors.iter().all(|(_, _, ty, _)| *ty == GGML_F32));
            let (index, store) = read_back_owned(&kvs, &tensors);
            let registry = Registry::builtin().unwrap();
            let raw = RawConfig::Gguf(&index);
            assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
            let gguf = registry.build(&raw, &store).unwrap();
            let (got, state) = run_step(&*gguf.model, &store, CAP, &prompt, 0, &[]);
            close(&got, &want);
            // The carried tensors differ in value-head order, so only the logits and the attention
            // cache are comparable: the cache is the last two state values.
            let kv = want_state.len() - 2;
            close_all(&state[kv..], &want_state[kv..]);
            let (got, _) = run_step(&*gguf.model, &store, CAP, &[5], 6, &state);
            close(&got, &want_decode);
        }
    }

    /// SC-002: a GGUF that appends a multi-token-prediction block (`nextn_predict_layers = 1`) has
    /// that block left out of the trunk: the stack is the fixture's four layers, not the five
    /// `block_count` states, and a count that leaves no layer is a typed refusal naming the key.
    /// Mutation: take `dims.layers` as the stack depth in `Qwen3NextParams::from_raw`.
    #[test]
    fn a_gguf_nextn_block_is_left_out_of_the_trunk() {
        let fx = fixture();
        let (kvs, tensors) = gguf_of(&fx, true);
        let (index, _) = read_back_owned(&kvs, &tensors);
        let p = Qwen3NextParams::from_raw(&RawConfig::Gguf(&index)).unwrap();
        assert_eq!(p.kinds.len(), 4);
        assert_eq!(p.kinds.last(), Some(&LayerKind::FullAttention));

        let (mut kvs, tensors) = gguf_of(&fx, true);
        for (key, value) in &mut kvs {
            if *key == nextn_key("qwen35moe") {
                *value = GgufValue::U32(5);
            }
        }
        let (index, _) = read_back_owned(&kvs, &tensors);
        let err = Qwen3NextParams::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field,
                    reason: ConfigReason::Exceeds { max: 5 },
                    ..
                } if field == nextn_key("qwen35moe")
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
        kvs.retain(|(key, _)| key != "qwen35moe.rope.freq_base");
        let (index, _) = read_back_owned(&kvs, &tensors);
        let err = Qwen3NextParams::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "qwen35moe.rope.freq_base",
                    reason: ConfigReason::Missing,
                    ..
                }
            ),
            "{err}"
        );
    }
}
