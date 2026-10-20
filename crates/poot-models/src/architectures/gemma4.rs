//! Gemma 4 text behind [`Model`] (HF `gemma4`): a sandwich-norm block (Gemma 3's) whose attention
//! layers come in two kinds. A local layer is sliding-window attention with plain RoPE; a global
//! layer is full attention over wider heads with proportional RoPE and, in checkpoints that tie
//! them, reads V from its K projection. Every layer ends in a learned scalar, scores are not
//! scaled, V is normalized per head with no learned scale, and the head's logits are softcapped.
//!
//! An HF checkpoint is the multimodal wrapper: the text tower lives under `text_config` and
//! `model.language_model.`, the vision and audio tower tensors stay unclaimed.

use std::num::NonZeroUsize;

use poot_graph_ir::{BinOp, Builder, Graph};
#[cfg(feature = "test-support")]
use poot_quant::weights::DenseWeight;
#[cfg(feature = "test-support")]
use poot_quant::weights::WeightEntry;
use poot_quant::weights::{WeightKey, WeightStore};
#[cfg(feature = "test-support")]
use poot_tensor::DType;
use poot_weights::{
    AttnRole, ExpertProj, FfnRole, MoeRole, NormRole, WeightId, WeightMap, WeightMapError,
    WeightRole,
};

use crate::components::attention::{AttentionWeights, attention};
use crate::components::ffn::{GatedFfnWeights, geglu_ffn};
use crate::components::linear::{Weight, WeightError};
use crate::components::moe::{MoeWeights, moe};
use crate::components::norm::norm;
use crate::components::standard::{
    LayerNorms, LayerScale, StackWeights, Step, decoder_program, standard_layer, standard_stack,
};
use crate::io::ModelSchema;
use crate::model::{FamilyKey, Model, ModelConfig, ModelError, StepSpec, TraceError};
use crate::names::gguf::{
    GGUF_BASE, GGUF_GEMMA4, GGUF_GEMMA4_MOE, GGUF_QK_NORM, GGUF_SANDWICH_NORMS,
};
use crate::names::{NameRow, Presence, Source, family_weights};
#[cfg(feature = "test-support")]
use crate::registry::Fixture;
use crate::registry::{ConfigSource, FamilyEntry, ModelParts, RawConfig};
use poot_text::ChatFormat;

mod config;

pub use config::Gemma4Params;

/// The Gemma vocabulary's `<bos>` id, for a config that names none.
const GEMMA_BOS: u32 = 2;

/// Gemma 4's [`FamilyKey`], shared with its config reader.
pub const FAMILY: FamilyKey = config::FAMILY;

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "gemma4"),
        (ConfigSource::GgufArchitecture, "gemma4"),
    ],
    build,
};

/// The text tower's tensor prefix in an HF `gemma4` checkpoint.
#[cfg(feature = "test-support")]
const TOWER: &str = "model.language_model.";
const LAYERS: &str = "model.language_model.layers.";

const fn tensor(role: WeightRole, name: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::Tensor(name),
        presence: Presence::Required,
    }
}

const fn layer_linear(role: WeightRole, stem: &'static str, presence: Presence) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear {
            prefix: LAYERS,
            stem,
        },
        presence,
    }
}

const fn layer_tensor(role: WeightRole, suffix: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor {
            prefix: LAYERS,
            suffix,
        },
        presence: Presence::Required,
    }
}

/// The checkpoint name of layer `l`'s V projection.
fn v_key(source: ConfigSource, l: usize) -> String {
    match source {
        ConfigSource::HfModelType => format!("{LAYERS}{l}.self_attn.v_proj.weight"),
        ConfigSource::GgufArchitecture => format!("blk.{l}.attn_v.weight"),
    }
}

/// Gemma 4's HF rows. A global layer that reads V from K has no `v_proj`, so that row is optional
/// and the layer's attention weights ask for it only when its parameters do.
const HF_ROWS: &[NameRow] = &[
    tensor(
        WeightRole::Embed,
        "model.language_model.embed_tokens.weight",
    ),
    tensor(WeightRole::FinalNorm, "model.language_model.norm.weight"),
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("lm_head"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    layer_linear(
        WeightRole::Attn(AttnRole::Q),
        ".self_attn.q_proj",
        Presence::Required,
    ),
    layer_linear(
        WeightRole::Attn(AttnRole::K),
        ".self_attn.k_proj",
        Presence::Required,
    ),
    layer_linear(
        WeightRole::Attn(AttnRole::V),
        ".self_attn.v_proj",
        Presence::Optional,
    ),
    layer_linear(
        WeightRole::Attn(AttnRole::O),
        ".self_attn.o_proj",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::QNorm),
        ".self_attn.q_norm.weight",
    ),
    layer_tensor(
        WeightRole::Attn(AttnRole::KNorm),
        ".self_attn.k_norm.weight",
    ),
    layer_tensor(WeightRole::Norm(NormRole::Attn), ".input_layernorm.weight"),
    layer_tensor(
        WeightRole::Norm(NormRole::PostAttn),
        ".post_attention_layernorm.weight",
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".pre_feedforward_layernorm.weight",
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::PostFfn),
        ".post_feedforward_layernorm.weight",
    ),
    layer_linear(
        WeightRole::Ffn(FfnRole::Gate),
        ".mlp.gate_proj",
        Presence::Required,
    ),
    layer_linear(
        WeightRole::Ffn(FfnRole::Up),
        ".mlp.up_proj",
        Presence::Required,
    ),
    layer_linear(
        WeightRole::Ffn(FfnRole::Down),
        ".mlp.down_proj",
        Presence::Required,
    ),
    layer_tensor(WeightRole::LayerScale, ".layer_scalar"),
];

/// The routed block's HF rows beside [`HF_ROWS`]: the router, the experts as two stacks and the
/// three norms around the routed and the dense branches.
const HF_MOE_ROWS: &[NameRow] = &[
    layer_tensor(
        WeightRole::Norm(NormRole::PreRouted),
        ".pre_feedforward_layernorm_2.weight",
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::PostDense),
        ".post_feedforward_layernorm_1.weight",
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::PostRouted),
        ".post_feedforward_layernorm_2.weight",
    ),
    layer_linear(
        WeightRole::Moe(MoeRole::Router),
        ".router.proj",
        Presence::Required,
    ),
    layer_tensor(WeightRole::Moe(MoeRole::RouterScale), ".router.scale"),
    layer_tensor(
        WeightRole::Moe(MoeRole::ExpertScale),
        ".router.per_expert_scale",
    ),
    layer_tensor(
        WeightRole::Moe(MoeRole::Stack(ExpertProj::GateUp)),
        ".experts.gate_up_proj",
    ),
    layer_tensor(
        WeightRole::Moe(MoeRole::Stack(ExpertProj::Down)),
        ".experts.down_proj",
    ),
];

/// The routed experts of a layer that has them, and the norms that fence them and the dense
/// feed-forward beside them.
#[derive(Debug)]
struct Routed {
    /// Normalizes the residual the experts read.
    pre: Weight,
    /// Normalize the dense and the routed branch before they are summed.
    post_dense: Weight,
    post_routed: Weight,
    experts: MoeWeights,
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    ffn: GatedFfnWeights,
    routed: Option<Routed>,
    scale: LayerScale,
}

/// The built Gemma 4 text model: config, name-resolved weights, the stack-level tensors and one
/// weight bundle per layer. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct Gemma4 {
    params: Gemma4Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let mut p = Gemma4Params::from_raw(raw)?;
    let mut tables: Vec<&[NameRow]> = match raw.source() {
        ConfigSource::HfModelType => vec![HF_ROWS],
        ConfigSource::GgufArchitecture => {
            p.read_gguf_tensors(store)?;
            vec![GGUF_BASE, GGUF_QK_NORM, GGUF_SANDWICH_NORMS, GGUF_GEMMA4]
        }
    };
    if p.moe.is_some() {
        tables.push(match raw.source() {
            ConfigSource::HfModelType => HF_MOE_ROWS,
            ConfigSource::GgufArchitecture => GGUF_GEMMA4_MOE,
        });
    }
    let weights = family_weights(FAMILY, store, &tables, p.layers, p.head_required)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    // The V row is optional (a global layer that reads V from K has none): a layer that needs
    // one and lacks it is a missing tensor, named as the other required rows are.
    for (l, attention) in p.attention.iter().enumerate() {
        let id = WeightId::layer(l, WeightRole::Attn(AttnRole::V));
        if !attention.value_from_key() && weights.handle(id).is_none() {
            return Err(ModelError::Weight {
                family: FAMILY,
                source: WeightMapError::Missing {
                    id,
                    key: WeightKey::from(v_key(raw.source(), l)),
                },
            });
        }
    }
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::sandwich(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention[l])?,
                ffn: GatedFfnWeights::new(&weights, l, p.width, p.inter)?,
                routed: p
                    .moe
                    .as_ref()
                    .map(|moe| {
                        let norm = |role| {
                            Weight::new(
                                &weights,
                                WeightId::layer(l, WeightRole::Norm(role)),
                                &[p.width],
                            )
                        };
                        Ok::<_, WeightError>(Routed {
                            pre: norm(NormRole::PreRouted)?,
                            post_dense: norm(NormRole::PostDense)?,
                            post_routed: norm(NormRole::PostRouted)?,
                            experts: MoeWeights::new(&weights, l, moe)?,
                        })
                    })
                    .transpose()?,
                scale: LayerScale::new(&weights, l)?,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let schema = ModelSchema::builder()
        .step(decoder_program(p.vocab))
        .build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        // Gemma is trained with a mandatory BOS.
        prompt_bos: Some(p.bos.unwrap_or(GEMMA_BOS)),
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::Gemma,
        tool_codec: None,
    };
    Ok(ModelParts::new(Box::new(Gemma4 {
        params: p,
        weights,
        stack,
        layers,
        config,
        schema,
    })))
}

impl Model for Gemma4 {
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
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            let x = standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention[l]),
                |b, input| {
                    let dense = geglu_ffn(b, input.block, &layer.ffn)?;
                    let (Some(routed), Some(moe_params)) = (&layer.routed, &p.moe) else {
                        return Ok::<_, TraceError>(dense);
                    };
                    let norm = |x, w: &Weight| norm(b, x, w, p.layer.norm());
                    // The router reads the residual stream; the experts read it normalized by
                    // their own norm.
                    let experts = moe(
                        b,
                        &step,
                        l,
                        norm(input.residual, &routed.pre)?,
                        Some(input.residual),
                        &routed.experts,
                        moe_params,
                    )?;
                    Ok::<_, TraceError>(b.binary(
                        BinOp::Add,
                        norm(dense, &routed.post_dense)?,
                        norm(experts, &routed.post_routed)?,
                    )?)
                },
            )?;
            layer.scale.apply(b, x)
        })?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny dense Gemma 4 with four layers (local, global, local, global): a
/// 3-key window, global heads twice as wide with one KV head that is also V, a quarter of the
/// global head's frequency pairs rotating, and the final softcap.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_config(moe: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "eos_token_id": [47, 46],
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "gemma4_text",
            "vocab_size": 48,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 32,
            "attention_k_eq_v": true,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "sliding_window": 3,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "enable_moe_block": false,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 100.0,
                    "rope_type": "default"
                }
            }
        }
    });
    if moe {
        config["text_config"]["enable_moe_block"] = true.into();
        config["text_config"]["num_experts"] = 8.into();
        config["text_config"]["top_k_experts"] = 2.into();
        config["text_config"]["moe_intermediate_size"] = 16.into();
    }
    config
}

/// The shape of every tensor of a Gemma 4 text tower with `config`'s dims by its HF name.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let text = &config["text_config"];
    let n = |f: &str| crate::registry::Fixture::count(text, f);
    let (vocab, h, inter, heads) = (
        n("vocab_size"),
        n("hidden_size"),
        n("intermediate_size"),
        n("num_attention_heads"),
    );
    let mut shapes = vec![
        (format!("{TOWER}embed_tokens.weight"), vec![vocab, h]),
        (format!("{TOWER}norm.weight"), vec![h]),
        ("lm_head.weight".to_string(), vec![vocab, h]),
    ];
    for (l, kind) in text["layer_types"].as_array().unwrap().iter().enumerate() {
        let global = kind == "full_attention";
        let (d, kv) = if global {
            (n("global_head_dim"), n("num_global_key_value_heads"))
        } else {
            (n("head_dim"), n("num_key_value_heads"))
        };
        let k = |s: &str| format!("{LAYERS}{l}.{s}");
        shapes.extend([
            (k("input_layernorm.weight"), vec![h]),
            (k("post_attention_layernorm.weight"), vec![h]),
            (k("pre_feedforward_layernorm.weight"), vec![h]),
            (k("post_feedforward_layernorm.weight"), vec![h]),
            (k("layer_scalar"), vec![1]),
            (k("self_attn.q_proj.weight"), vec![heads * d, h]),
            (k("self_attn.k_proj.weight"), vec![kv * d, h]),
            (k("self_attn.o_proj.weight"), vec![h, heads * d]),
            (k("self_attn.q_norm.weight"), vec![d]),
            (k("self_attn.k_norm.weight"), vec![d]),
            (k("mlp.gate_proj.weight"), vec![inter, h]),
            (k("mlp.up_proj.weight"), vec![inter, h]),
            (k("mlp.down_proj.weight"), vec![h, inter]),
        ]);
        if !(global && text["attention_k_eq_v"] == true) {
            shapes.push((k("self_attn.v_proj.weight"), vec![kv * d, h]));
        }
        if text["enable_moe_block"] == true {
            let (experts, inter) = (n("num_experts"), n("moe_intermediate_size"));
            shapes.extend([
                (k("pre_feedforward_layernorm_2.weight"), vec![h]),
                (k("post_feedforward_layernorm_1.weight"), vec![h]),
                (k("post_feedforward_layernorm_2.weight"), vec![h]),
                (k("router.proj.weight"), vec![experts, h]),
                (k("router.scale"), vec![h]),
                (k("router.per_expert_scale"), vec![experts]),
                (k("experts.gate_up_proj"), vec![experts, 2 * inter, h]),
                (k("experts.down_proj"), vec![experts, h, inter]),
            ]);
        }
    }
    shapes
}

/// The fixture of `config`. The scales a layer multiplies by (the layer scalar, the router's
/// input scale, the per-expert scale) hold values near one: random values centred on zero would
/// silence their layer or flip its sign.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_of(config: serde_json::Value) -> Fixture {
    let shapes = fixture_shapes(&config);
    let fx = Fixture::bf16(config, shapes);
    let mut store = WeightStore::builder();
    for (key, entry) in fx.store.iter() {
        let name = key.as_str();
        let is_scale = [".layer_scalar", ".router.scale", ".router.per_expert_scale"]
            .iter()
            .any(|suffix| name.ends_with(suffix));
        let entry = if is_scale {
            let layer: usize = name[LAYERS.len()..]
                .split('.')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let n: usize = entry.shape().iter().product();
            let bytes: Vec<u8> = (0..n)
                .flat_map(|i| {
                    let value = 0.8 + 0.4 * ((i * 7 + layer * 3) % 11) as f32 / 11.0;
                    ((value.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            WeightEntry::Dense(
                DenseWeight::try_new(DType::BF16, entry.shape(), bytes.into())
                    .expect("bf16 values of the entry's shape"),
            )
        } else {
            entry.clone()
        };
        store
            .insert(key.clone(), entry)
            .expect("fixture keys are unique");
    }
    Fixture {
        store: store.build(),
        ..fx
    }
}

#[cfg(feature = "test-support")]
pub(crate) fn fixture() -> Fixture {
    fixture_of(fixture_config(false))
}

/// The same tower with eight routed experts beside every layer's dense feed-forward.
#[cfg(feature = "test-support")]
pub(crate) fn moe_fixture() -> Fixture {
    fixture_of(fixture_config(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::standard::oracle::{assert_matches_f64, run_step};
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build as built,
        values,
    };
    use crate::registry::{RawConfig, Registry};

    fn model_of(fx: &Fixture, store: &WeightStore) -> Box<dyn Model> {
        built(&ENTRY, &fx.raw(), store)
    }

    /// An independent f64 forward pass of the fixture over `tokens` (one prefill from position 0):
    /// the logits of every position, written from the HF reference's formulas over the stored
    /// BF16 values, with none of the components under test.
    struct Reference<'a> {
        store: &'a WeightStore,
        text: &'a serde_json::Value,
    }

    impl Reference<'_> {
        fn tensor(&self, name: &str) -> Vec<f64> {
            let entry = self.store.get(name).unwrap_or_else(|| panic!("{name}"));
            values(entry).into_iter().map(f64::from).collect()
        }

        fn rms(x: &[f64], scale: Option<&[f64]>) -> Vec<f64> {
            let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-6).sqrt();
            x.iter()
                .enumerate()
                .map(|(i, v)| v * inv * scale.map_or(1.0, |s| s[i]))
                .collect()
        }

        /// `w[out, in] x`.
        fn matvec(w: &[f64], x: &[f64]) -> Vec<f64> {
            w.chunks(x.len())
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        }

        fn gelu_tanh(x: f64) -> f64 {
            0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
        }

        /// RoPE on one head at `pos`: pairs `(j, j + d/2)`, frequency `theta^(-2j/d)` for the first
        /// `rotated` pairs and zero after.
        fn rope(head: &mut [f64], pos: usize, theta: f64, rotated: usize) {
            let d = head.len();
            let half = d / 2;
            for j in 0..rotated.min(half) {
                let angle = pos as f64 * theta.powf(-((2 * j) as f64) / d as f64);
                let (s, c) = angle.sin_cos();
                let (a, b) = (head[j], head[j + half]);
                head[j] = a * c - b * s;
                head[j + half] = b * c + a * s;
            }
        }

        /// The routed branch of layer `l` over the residual `x1`: the router on its normalized,
        /// scaled input, the top experts' gates renormalized then scaled per expert, GeGLU
        /// experts over the residual normalized by the routed norm, and the routed output norm.
        fn routed(&self, l: usize, x1: &[f64]) -> Vec<f64> {
            let n = |k: &str| crate::registry::Fixture::count(self.text, k);
            let (h, experts, top_k, inter) = (
                n("hidden_size"),
                n("num_experts"),
                n("top_k_experts"),
                n("moe_intermediate_size"),
            );
            let t = |s: &str| self.tensor(&format!("{LAYERS}{l}.{s}"));
            let router_in: Vec<f64> = Self::rms(x1, None)
                .iter()
                .zip(t("router.scale"))
                .map(|(v, s)| v * s / (h as f64).sqrt())
                .collect();
            let logits = Self::matvec(&t("router.proj.weight"), &router_in);
            let max = logits.iter().copied().fold(f64::MIN, f64::max);
            let exp: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let probs: Vec<f64> = exp.iter().map(|e| e / sum).collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            let chosen = &order[..top_k];
            let kept: f64 = chosen.iter().map(|&e| probs[e]).sum();
            let (per_expert, gate_up, down) = (
                t("router.per_expert_scale"),
                t("experts.gate_up_proj"),
                t("experts.down_proj"),
            );
            let expert_in = Self::rms(x1, Some(&t("pre_feedforward_layernorm_2.weight")));
            let mut out = vec![0.0; h];
            for &e in chosen {
                let weight = probs[e] / kept * per_expert[e];
                let both = Self::matvec(
                    &gate_up[e * 2 * inter * h..(e + 1) * 2 * inter * h],
                    &expert_in,
                );
                let act: Vec<f64> = (0..inter)
                    .map(|i| Self::gelu_tanh(both[i]) * both[inter + i])
                    .collect();
                let y = Self::matvec(&down[e * h * inter..(e + 1) * h * inter], &act);
                for (o, v) in out.iter_mut().zip(y) {
                    *o += weight * v;
                }
            }
            Self::rms(&out, Some(&t("post_feedforward_layernorm_2.weight")))
        }

        fn logits(&self, tokens: &[usize]) -> Vec<f64> {
            let n = |k: &str| crate::registry::Fixture::count(self.text, k);
            let (h, heads, window) = (
                n("hidden_size"),
                n("num_attention_heads"),
                n("sliding_window"),
            );
            let embed = self.tensor(&format!("{TOWER}embed_tokens.weight"));
            let mut xs: Vec<Vec<f64>> = tokens
                .iter()
                .map(|&t| {
                    embed[t * h..(t + 1) * h]
                        .iter()
                        .map(|v| v * (h as f64).sqrt())
                        .collect()
                })
                .collect();
            for (l, kind) in self.text["layer_types"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let global = kind == "full_attention";
                let (d, kv, theta, rotated) = if global {
                    let d = n("global_head_dim");
                    // proportional: a quarter of the pairs rotate.
                    (d, n("num_global_key_value_heads"), 1e6, d / 8)
                } else {
                    (
                        n("head_dim"),
                        n("num_key_value_heads"),
                        100.0,
                        n("head_dim") / 2,
                    )
                };
                let t = |s: &str| self.tensor(&format!("{LAYERS}{l}.{s}"));
                let k_eq_v = global && self.text["attention_k_eq_v"] == true;
                let (q_norm, k_norm) = (t("self_attn.q_norm.weight"), t("self_attn.k_norm.weight"));
                let mut ks: Vec<Vec<Vec<f64>>> = Vec::new();
                let mut vs: Vec<Vec<Vec<f64>>> = Vec::new();
                let mut qs: Vec<Vec<Vec<f64>>> = Vec::new();
                for (pos, x) in xs.iter().enumerate() {
                    let normed = Self::rms(x, Some(&t("input_layernorm.weight")));
                    let q = Self::matvec(&t("self_attn.q_proj.weight"), &normed);
                    let k_raw = Self::matvec(&t("self_attn.k_proj.weight"), &normed);
                    let v_raw = if k_eq_v {
                        k_raw.clone()
                    } else {
                        Self::matvec(&t("self_attn.v_proj.weight"), &normed)
                    };
                    let rotate = |all: &[f64], norm: &[f64]| -> Vec<Vec<f64>> {
                        all.chunks(d)
                            .map(|head| {
                                let mut head = Self::rms(head, Some(norm));
                                Self::rope(&mut head, pos, theta, rotated);
                                head
                            })
                            .collect()
                    };
                    qs.push(rotate(&q, &q_norm));
                    ks.push(rotate(&k_raw, &k_norm));
                    vs.push(v_raw.chunks(d).map(|head| Self::rms(head, None)).collect());
                }
                let mut next = Vec::new();
                for (i, x) in xs.iter().enumerate() {
                    let mut ctx = Vec::new();
                    for (head, q_head) in qs[i].iter().enumerate() {
                        let group = head / (heads / kv);
                        let lo = if global {
                            0
                        } else {
                            (i + 1).saturating_sub(window)
                        };
                        let scores: Vec<f64> = (lo..=i)
                            .map(|j| q_head.iter().zip(&ks[j][group]).map(|(a, b)| a * b).sum())
                            .collect();
                        let max = scores.iter().copied().fold(f64::MIN, f64::max);
                        let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                        let sum: f64 = exp.iter().sum();
                        ctx.extend((0..d).map(|c| {
                            exp.iter()
                                .enumerate()
                                .map(|(j, e)| e / sum * vs[lo + j][group][c])
                                .sum::<f64>()
                        }));
                    }
                    let attn = Self::matvec(&t("self_attn.o_proj.weight"), &ctx);
                    let attn = Self::rms(&attn, Some(&t("post_attention_layernorm.weight")));
                    let x1: Vec<f64> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
                    let normed = Self::rms(&x1, Some(&t("pre_feedforward_layernorm.weight")));
                    let gate = Self::matvec(&t("mlp.gate_proj.weight"), &normed);
                    let up = Self::matvec(&t("mlp.up_proj.weight"), &normed);
                    let act: Vec<f64> = gate
                        .iter()
                        .zip(&up)
                        .map(|(g, u)| Self::gelu_tanh(*g) * u)
                        .collect();
                    let dense = Self::matvec(&t("mlp.down_proj.weight"), &act);
                    let combined = if self.text["enable_moe_block"] == true {
                        let routed = self.routed(l, &x1);
                        let dense =
                            Self::rms(&dense, Some(&t("post_feedforward_layernorm_1.weight")));
                        dense.iter().zip(&routed).map(|(a, b)| a + b).collect()
                    } else {
                        dense
                    };
                    let down = Self::rms(&combined, Some(&t("post_feedforward_layernorm.weight")));
                    let scalar = t("layer_scalar")[0];
                    next.push(
                        x1.iter()
                            .zip(&down)
                            .map(|(a, b)| (a + b) * scalar)
                            .collect(),
                    );
                }
                xs = next;
            }
            let final_norm = self.tensor(&format!("{TOWER}norm.weight"));
            let head = self.tensor("lm_head.weight");
            let cap = self.text["final_logit_softcapping"].as_f64().unwrap();
            xs.iter()
                .flat_map(|x| {
                    let normed = Self::rms(x, Some(&final_norm));
                    Self::matvec(&head, &normed)
                        .into_iter()
                        .map(|l| cap * (l / cap).tanh())
                })
                .collect()
        }
    }

    /// SC-001 (model half): the fixture's prefill logits equal the independent f64 forward pass
    /// at every position (ADR-0101 tier 2), over a prompt longer than the local window so the
    /// window cuts keys. Covers local and global layers, proportional RoPE, K read as V, the V
    /// norm, the layer scalar and the softcap.
    #[test]
    fn gemma4_prefill_matches_the_f64_reference() {
        assert_prefill_matches_the_reference(&fixture());
    }

    /// SC-001 (routed half): the same prefill over the fixture with routed experts beside every
    /// layer's dense feed-forward: the router on the residual, the per-expert gate scale, GeGLU
    /// experts and the three norms around the branches. Mutation: route on the experts' normalized
    /// input instead of the residual (pass `norm(input.residual, ..)` as the router input).
    #[test]
    fn gemma4_moe_prefill_matches_the_f64_reference() {
        assert_prefill_matches_the_reference(&moe_fixture());
    }

    fn assert_prefill_matches_the_reference(fx: &Fixture) {
        let model = model_of(fx, &fx.store);
        let prompt = [3usize, 17, 40, 8, 25, 9, 11];
        let tokens: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let (got, _) = run_step(&*model, &fx.store, CAP, &tokens, 0, &[]);
        let reference = Reference {
            store: &fx.store,
            text: &fx.config["text_config"],
        };
        assert_matches_f64(&got, &reference.logits(&prompt), 1e-4);
    }

    /// SC-004 (dfamily F15): a global layer reads V from the K projection, a local layer from its
    /// `v_proj`. The store here also holds a `v_proj` for each global layer, unlike its K, so a body
    /// that read it would differ from the reference, which takes V from K. Mutation: read
    /// `v_proj` on a global layer (drop `with_value_from_key` in `config.rs`).
    #[test]
    fn global_layers_read_v_from_the_k_projection_and_local_layers_from_v_proj() {
        let fx = fixture();
        let mut store = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            store.insert(key.clone(), entry.clone()).unwrap();
        }
        for l in [1usize, 3] {
            let k = fx
                .store
                .get(&format!("{LAYERS}{l}.self_attn.q_proj.weight"))
                .unwrap();
            // A decoy `v_proj` of the K shape `[32, 32]`, holding Q's first rows.
            let decoy: Vec<f32> = values(k).into_iter().take(32 * 32).collect();
            store
                .insert(
                    format!("{LAYERS}{l}.self_attn.v_proj.weight"),
                    crate::components::testing::f32_entry(vec![32, 32], &decoy),
                )
                .unwrap();
        }
        let store = store.build();
        let model = model_of(&fx, &store);
        let prompt = [3usize, 17, 40, 8, 25];
        let tokens: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let (got, _) = run_step(&*model, &store, CAP, &tokens, 0, &[]);
        let reference = Reference {
            store: &fx.store,
            text: &fx.config["text_config"],
        };
        assert_matches_f64(&got, &reference.logits(&prompt), 1e-4);
    }

    /// Chunked prefill from absolute positions equals token-by-token decode (the carried caches
    /// of both layer kinds, the windowed local layers included).
    #[test]
    fn gemma4_chunked_prefill_equals_decode() {
        for fx in [fixture(), moe_fixture()] {
            let model = model_of(&fx, &fx.store);
            assert_chunked_prefill_equals_decode(&*model, &fx.store);
        }
    }

    /// SC-005 (dfamily F6): a store missing one required tensor fails `Registry::build` with the
    /// typed missing-weight error naming it, for a per-layer tensor of each kind. Mutation: skip
    /// the row check (make the row optional); the build succeeds or fails later, and the row is red.
    #[test]
    fn a_store_missing_a_required_tensor_fails_registry_build() {
        let fx = fixture();
        let registry = Registry::builtin().unwrap();
        let moe = moe_fixture();
        for removed in [
            format!("{LAYERS}2.router.scale"),
            format!("{LAYERS}0.experts.down_proj"),
            format!("{LAYERS}3.post_feedforward_layernorm_2.weight"),
        ] {
            assert_missing_weight_is_named(&registry, &moe, &removed);
        }
        for removed in [
            format!("{LAYERS}1.layer_scalar"),
            format!("{LAYERS}2.self_attn.q_norm.weight"),
            format!("{LAYERS}0.self_attn.v_proj.weight"),
            format!("{TOWER}norm.weight"),
        ] {
            assert_missing_weight_is_named(&registry, &fx, &removed);
        }
    }

    /// llama.cpp's name for an HF `gemma4` text-tower tensor.
    fn gguf_name(hf: &str) -> String {
        let name = hf.strip_prefix(TOWER).unwrap_or(hf);
        match name {
            "embed_tokens.weight" => return "token_embd.weight".into(),
            "norm.weight" => return "output_norm.weight".into(),
            "lm_head.weight" => return "output.weight".into(),
            _ => {}
        }
        let rest = name.strip_prefix("layers.").unwrap();
        let (layer, tensor) = rest.split_once('.').unwrap();
        let mapped = match tensor {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
            "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
            "layer_scalar" => "layer_output_scale.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "pre_feedforward_layernorm_2.weight" => "ffn_pre_norm_2.weight",
            "post_feedforward_layernorm_1.weight" => "ffn_post_norm_1.weight",
            "post_feedforward_layernorm_2.weight" => "ffn_post_norm_2.weight",
            "router.proj.weight" => "ffn_gate_inp.weight",
            "router.scale" => "ffn_gate_inp.scale",
            "router.per_expert_scale" => "ffn_down_exps.scale",
            "experts.gate_up_proj" => "ffn_gate_up_exps.weight",
            "experts.down_proj" => "ffn_down_exps.weight",
            other => panic!("no GGUF name for {other}"),
        };
        format!("blk.{layer}.{mapped}")
    }

    /// The fixture as a GGUF: llama.cpp's metadata (per-layer KV counts and local flags, the two
    /// head widths), its tensor names, no `attn_v` on the global layers and the global RoPE's NoPE
    /// tail as `rope_freqs.weight` (ones, then `1e30`).
    fn gguf_case(fx: &Fixture, omit: &[&str]) -> (poot_load::gguf::GgufIndex, WeightStore) {
        use crate::components::testing::{f32_entry, gguf_tensor, read_back_owned};
        use poot_load::gguf::GgufValue;

        let text = &fx.config["text_config"];
        let u = |key: &str| GgufValue::U32(text[key].as_u64().unwrap() as u32);
        let kinds = text["layer_types"].as_array().unwrap();
        let local = |kind: &serde_json::Value| kind == "sliding_attention";
        let kvs: Vec<(String, GgufValue)> = vec![
            (
                "general.architecture".into(),
                GgufValue::Str("gemma4".into()),
            ),
            ("gemma4.embedding_length".into(), u("hidden_size")),
            ("gemma4.feed_forward_length".into(), u("intermediate_size")),
            ("gemma4.block_count".into(), u("num_hidden_layers")),
            ("gemma4.context_length".into(), u("max_position_embeddings")),
            (
                "gemma4.attention.head_count".into(),
                u("num_attention_heads"),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                GgufValue::Array(
                    kinds
                        .iter()
                        .map(|k| {
                            u(if local(k) {
                                "num_key_value_heads"
                            } else {
                                "num_global_key_value_heads"
                            })
                        })
                        .collect(),
                ),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                GgufValue::Array(kinds.iter().map(|k| GgufValue::Bool(local(k))).collect()),
            ),
            (
                "gemma4.attention.sliding_window".into(),
                u("sliding_window"),
            ),
            ("gemma4.attention.key_length".into(), u("global_head_dim")),
            ("gemma4.attention.key_length_swa".into(), u("head_dim")),
            ("gemma4.rope.freq_base".into(), GgufValue::F32(1e6)),
            ("gemma4.rope.freq_base_swa".into(), GgufValue::F32(100.0)),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                GgufValue::F32(1e-6),
            ),
            (
                "gemma4.final_logit_softcapping".into(),
                GgufValue::F32(30.0),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array((0..48).map(|t| GgufValue::Str(format!("t{t}"))).collect()),
            ),
            ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(47)),
        ];
        let mut kvs = kvs;
        if text["enable_moe_block"] == true {
            kvs.extend([
                ("gemma4.expert_count".into(), u("num_experts")),
                ("gemma4.expert_used_count".into(), u("top_k_experts")),
                (
                    "gemma4.expert_feed_forward_length".into(),
                    u("moe_intermediate_size"),
                ),
            ]);
        }
        let global_pairs = text["global_head_dim"].as_u64().unwrap() as usize / 2;
        let rotated = global_pairs / 4;
        let freqs: Vec<f32> = (0..global_pairs)
            .map(|j| if j < rotated { 1.0 } else { 1e30 })
            .collect();
        let mut tensors: Vec<_> = fx
            .store
            .iter()
            .filter(|(key, _)| {
                !key.as_str().contains("layers.1.self_attn.v_proj")
                    && !key.as_str().contains("layers.3.self_attn.v_proj")
            })
            .map(|(key, entry)| gguf_tensor(&gguf_name(key.as_str()), entry))
            .collect();
        tensors.push(gguf_tensor(
            "rope_freqs.weight",
            &f32_entry(vec![freqs.len()], &freqs),
        ));
        tensors.retain(|(name, ..)| !omit.contains(&name.as_str()));
        read_back_owned(&kvs, &tensors)
    }

    /// A GGUF of the fixture resolves to gemma4 by architecture and equals the HF body: the
    /// per-layer metadata, `attn_v` absent on global layers (V from K) and `rope_freqs.weight`'s
    /// NoPE tail all reach the graph. Mutation: ignore `rope_freqs.weight` in
    /// `read_gguf_tensors`; the global layers rotate every pair and the logits differ.
    #[test]
    fn a_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        for fx in [fixture(), moe_fixture()] {
            assert_gguf_equals_hf(&fx);
        }
    }

    fn assert_gguf_equals_hf(fx: &Fixture) {
        let hf = model_of(fx, &fx.store);
        let (index, store) = gguf_case(fx, &[]);
        let registry = Registry::builtin().unwrap();
        let gguf = registry
            .build(&RawConfig::Gguf(&index), &store)
            .unwrap_or_else(|e| panic!("{e}"));
        let tokens = [3, 17, 40, 8, 25, 9, 11];
        let (want, _) = run_step(&*hf, &fx.store, CAP, &tokens, 0, &[]);
        let (got, _) = run_step(&*gguf.model, &store, CAP, &tokens, 0, &[]);
        crate::components::testing::close(&got, &want);
    }

    /// A config the body cannot trace, or that is malformed, is a typed error naming its field
    /// before any weight is read.
    #[test]
    fn an_untraceable_or_malformed_config_is_a_typed_error_naming_its_field() {
        use crate::model::ConfigReason;
        let fx = fixture();
        let cases: [(&str, serde_json::Value, &str, ConfigReason); 5] = [
            (
                "text_config",
                serde_json::Value::Null,
                "text_config",
                ConfigReason::Missing,
            ),
            (
                "hidden_size_per_layer_input",
                serde_json::json!(8),
                "hidden_size_per_layer_input",
                ConfigReason::Unsupported,
            ),
            (
                "num_kv_shared_layers",
                serde_json::json!(2),
                "num_kv_shared_layers",
                ConfigReason::Unsupported,
            ),
            (
                "hidden_activation",
                serde_json::json!("relu"),
                "hidden_activation",
                ConfigReason::Unsupported,
            ),
            (
                "layer_types",
                serde_json::json!(["full_attention"]),
                "layer_types",
                ConfigReason::WrongType,
            ),
        ];
        for (key, value, field, reason) in cases {
            let mut config = fx.config.clone();
            if key == "text_config" {
                config.as_object_mut().unwrap().remove(key);
            } else {
                config["text_config"][key] = value;
            }
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
                preprocessor: None,
            };
            match build(&raw, &fx.store) {
                Err(ModelError::Config {
                    field: got,
                    reason: why,
                    ..
                }) => assert_eq!((got, why), (field, reason), "{key}"),
                other => panic!("{key}: expected a config error, got {other:?}"),
            }
        }
    }

    /// A GGUF that loses a tensor fails with the typed missing-weight error naming it, as the HF
    /// path does (SC-005): a local layer's `attn_v` is required (only a global layer may omit it,
    /// as llama.cpp writes it for K = V), and a GGUF with global layers needs `rope_freqs.weight`,
    /// whose absence would silently rotate every pair. Mutation: treat every missing `attn_v` as
    /// K = V in `read_gguf_tensors`; the first case builds and the row is red.
    #[test]
    fn a_gguf_missing_a_local_attn_v_or_the_rope_freqs_is_refused() {
        use crate::model::ConfigReason;
        let fx = fixture();
        let registry = Registry::builtin().unwrap();
        for dropped in ["blk.0.attn_v.weight", "blk.2.attn_q.weight"] {
            let (index, store) = gguf_case(&fx, &[dropped]);
            match registry.build(&RawConfig::Gguf(&index), &store) {
                Err(crate::registry::LoadModelError::Model(ModelError::Weight {
                    source: WeightMapError::Missing { key, .. },
                    ..
                })) => assert_eq!(key.as_str(), dropped),
                other => panic!("{dropped}: expected a missing-weight error, got {other:?}"),
            }
        }
        let (index, store) = gguf_case(&fx, &["rope_freqs.weight"]);
        match registry.build(&RawConfig::Gguf(&index), &store) {
            Err(crate::registry::LoadModelError::Model(ModelError::Config {
                field: "rope_freqs.weight",
                reason: ConfigReason::Missing,
                ..
            })) => {}
            other => panic!("expected a missing rope_freqs error, got {other:?}"),
        }
    }
}
