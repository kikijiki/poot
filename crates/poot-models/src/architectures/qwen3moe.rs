//! qwen3-MoE behind [`Model`]: qwen3's body (per-head Q/K RMSNorm before RoPE, an explicit
//! `head_dim`) whose feed-forward is a routed mixture of experts on some layers and the plain gated
//! FFN on the others.
//!
//! The body is `standard_stack` over `standard_layer` with the shared attention and, per layer,
//! [`moe`] or [`gated_ffn`]. HF `qwen3_moe` decides which layers route from the config
//! (`decoder_sparse_step`, `mlp_only_layers`); a GGUF has no key for it, so a layer routes there
//! exactly when it holds a router tensor.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{ExpertProj, WeightMap, WeightStore};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::linear::WeightError;
use crate::components::moe::{self, MoeParams, MoeWeights};
use crate::components::router::RouterKind;
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step,
    standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::{GGUF_BASE, GGUF_QK_NORM};
use crate::names::{
    ExpertNames, HF_BASE, HF_QK_NORM, MoeLayers, MoeNames, family_weights_routed, hf_base_shapes,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("qwen3moe");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "qwen3_moe"),
        (ConfigSource::GgufArchitecture, "qwen3moe"),
    ],
    build,
    fixture,
    variants: &[],
};

/// qwen3-MoE's config facts: `Qwen3MoeConfig`'s defaults (the class declares no `eos_token_id`: a
/// checkpoint names it) and llama.cpp's GGUF base default.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("qwen3moe"),
    hf_theta: 10_000.0,
    gguf_theta: 1_000_000.0,
    eps: 1e-6,
    eos: None,
};

/// The per-expert FFN width: `moe_intermediate_size`, which is not the dense layers' width.
const EXPERT_INTER: Field = Field::new(
    "moe_intermediate_size",
    "qwen3moe.expert_feed_forward_length",
);
const DENSE_INTER: Field = Field::new("intermediate_size", "qwen3moe.feed_forward_length");
const EXPERTS: Field = Field::new("num_experts", "qwen3moe.expert_count");
const TOP_K: Field = Field::new("num_experts_per_tok", "qwen3moe.expert_used_count");
const NORM_TOPK_PROB: Field = Field::new("norm_topk_prob", "qwen3moe.expert_weights_norm");

const HF_MOE: MoeNames = MoeNames {
    prefix: "model.layers.",
    router: ".mlp.gate",
    router_bias: None,
    shared: None,
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
};

const GGUF_MOE: MoeNames = MoeNames {
    prefix: "blk.",
    router: ".ffn_gate_inp",
    router_bias: None,
    shared: None,
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
};

/// qwen3-MoE's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3MoeParams {
    vocab: usize,
    width: usize,
    /// The dense layers' FFN width; `None` when the config does not state one.
    dense_inter: Option<usize>,
    layers: usize,
    max_positions: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    moe: MoeParams,
    /// Per layer: routed (`true`) or the plain gated FFN.
    sparse: Vec<bool>,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl Qwen3MoeParams {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read_with_inter(&fields, &SPEC, EXPERT_INTER)?;
        for unsupported in ["attention_bias", "use_sliding_window"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let norm = dims.norm(&fields, &SPEC)?;
        let dense_inter = fields
            .opt_u64(DENSE_INTER)?
            .map(|n| {
                usize::try_from(n)
                    .map_err(|_| fields.error(fields.key(DENSE_INTER), ConfigReason::WrongType))
            })
            .transpose()?;
        // HF's class default is no renormalization; llama.cpp's graph always renormalizes.
        let renormalize = fields.flag(NORM_TOPK_PROB, matches!(raw, RawConfig::Gguf(_)))?;
        let moe = MoeParams::new(
            FAMILY,
            dims.width,
            dims.inter,
            fields.count(EXPERTS, None)?,
            fields.count(TOP_K, None)?,
            RouterKind::SoftmaxTopK { renormalize },
        )?;
        let sparse = sparse_layers(&fields, dims.layers)?;
        if dense_inter.is_none() && sparse.contains(&false) {
            return Err(fields.error(fields.key(DENSE_INTER), ConfigReason::Missing));
        }
        Ok(Self {
            attention: dims
                .attention(&fields, &SPEC, false)?
                .with_qk_norm(QkNorm::PerHead(norm)),
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            dense_inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            moe,
            sparse,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
        })
    }
}

/// Which layers route. HF: `Qwen3MoeDecoderLayer` routes layer `l` unless it is in
/// `mlp_only_layers` or `(l + 1)` is not a multiple of `decoder_sparse_step`. A GGUF has no such
/// key (llama.cpp only runs all-routed qwen3-MoE, but its converter can write a dense fallback
/// layer), so a layer routes there when its router tensor exists.
fn sparse_layers(fields: &Fields<'_>, layers: usize) -> Result<Vec<bool>, ModelError> {
    match fields.raw() {
        RawConfig::Gguf(gguf) => Ok((0..layers)
            .map(|l| {
                gguf.tensors
                    .contains_key(&format!("blk.{l}.ffn_gate_inp.weight"))
            })
            .collect()),
        RawConfig::HfJson { config, .. } => {
            let step = match fields.opt_u64(Field::hf("decoder_sparse_step"))? {
                None => 1,
                Some(0) => {
                    return Err(fields.error("decoder_sparse_step", ConfigReason::Zero));
                }
                Some(n) => usize::try_from(n)
                    .map_err(|_| fields.error("decoder_sparse_step", ConfigReason::WrongType))?,
            };
            let wrong = || fields.error("mlp_only_layers", ConfigReason::WrongType);
            let mlp_only: Vec<usize> = match config.get("mlp_only_layers") {
                None | Some(serde_json::Value::Null) => Vec::new(),
                Some(value) => value
                    .as_array()
                    .ok_or_else(wrong)?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or_else(wrong)
                    })
                    .collect::<Result<_, _>>()?,
            };
            Ok((0..layers)
                .map(|l| !mlp_only.contains(&l) && (l + 1).is_multiple_of(step))
                .collect())
        }
    }
}

#[derive(Debug)]
enum Ffn {
    Dense(Box<GatedFfnWeights>),
    Routed(Box<MoeWeights>),
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    ffn: Ffn,
}

#[derive(Debug)]
pub struct Qwen3Moe {
    params: Qwen3MoeParams,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Qwen3MoeParams::from_raw(raw)?;
    let (tables, names): ([&[_]; 2], &MoeNames) = match raw.source() {
        ConfigSource::HfModelType => ([HF_BASE, HF_QK_NORM], &HF_MOE),
        ConfigSource::GgufArchitecture => ([GGUF_BASE, GGUF_QK_NORM], &GGUF_MOE),
    };
    let weights = family_weights_routed(
        FAMILY,
        store,
        &tables,
        p.layers,
        p.head_required,
        MoeLayers {
            names,
            sparse: &p.sparse,
            experts: p.moe.experts(),
        },
    )?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            let ffn = match (p.sparse[l], p.dense_inter) {
                (true, _) => Ffn::Routed(Box::new(MoeWeights::new(&weights, l, &p.moe)?)),
                (false, Some(inter)) => {
                    Ffn::Dense(Box::new(GatedFfnWeights::new(&weights, l, p.width, inter)?))
                }
                (false, None) => unreachable!("from_raw requires a dense width for a dense layer"),
            };
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention)?,
                ffn,
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
    Ok(Box::new(Qwen3Moe {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Qwen3Moe {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace(&self, phase: Phase, shape: StepShape) -> Result<Graph, TraceError> {
        let p = &self.params;
        check_step(FAMILY, phase, shape, p.max_positions)?;
        moe::check_step(FAMILY, shape, &p.moe)?;
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention),
                |b, BlockInput { block: h, .. }| match &layer.ffn {
                    Ffn::Dense(w) => gated_ffn(b, h, w),
                    Ffn::Routed(w) => moe::moe(b, &step, l, h, None, w, &p.moe),
                },
            )
        });
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny untied qwen3-MoE of two layers, the first dense and the second
/// routed (`decoder_sparse_step` 2), with `head_dim` 32 and projection widths that are multiples
/// of 32 (so a Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_moe",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "moe_intermediate_size": 64,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [],
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (h, d) = (n("hidden_size"), n("head_dim"));
    let (experts, inter) = (n("num_experts"), n("moe_intermediate_size"));
    let sparse = sparse_flags(&config);
    let mut shapes: Vec<_> = hf_base_shapes(&config, false)
        .into_iter()
        .filter(|(name, _)| {
            let layer = name
                .strip_prefix("model.layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|l| l.parse::<usize>().ok());
            !(name.contains(".mlp.") && layer.is_some_and(|l| sparse[l]))
        })
        .collect();
    for (l, &routed) in sparse.iter().enumerate() {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        shapes.push((k("self_attn.q_norm.weight"), vec![d]));
        shapes.push((k("self_attn.k_norm.weight"), vec![d]));
        if routed {
            shapes.push((k("mlp.gate.weight"), vec![experts, h]));
            for e in 0..experts {
                shapes.push((
                    k(&format!("mlp.experts.{e}.gate_proj.weight")),
                    vec![inter, h],
                ));
                shapes.push((
                    k(&format!("mlp.experts.{e}.up_proj.weight")),
                    vec![inter, h],
                ));
                shapes.push((
                    k(&format!("mlp.experts.{e}.down_proj.weight")),
                    vec![h, inter],
                ));
            }
        }
    }
    Fixture::bf16(config, shapes)
}

/// Which layers of a fixture `config` route, by the HF rule.
fn sparse_flags(config: &serde_json::Value) -> Vec<bool> {
    let step = config["decoder_sparse_step"].as_u64().unwrap() as usize;
    (0..config["num_hidden_layers"].as_u64().unwrap() as usize)
        .map(|l| (l + 1).is_multiple_of(step))
        .collect()
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::OpKind;
    use poot_load::gguf::{GgufIndex, GgufValue};
    use poot_quant::weights::{MoeRole, WeightId, WeightMapError, WeightRole};
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, GGML_Q8_0, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named,
        build as built, close, close_all, decoded, f32_entry, gguf_dense_kvs, gguf_name_with_head,
        gguf_tensor, q8_0_store, read_back_owned, step_shape, values,
    };
    use crate::components::witness::WitnessKind;
    use crate::model::LogitRows;
    use crate::registry::Registry;

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// A projection or expert weight a Q8_0 copy of a store packs (`K` a multiple of 32).
    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// The fixture routes layer 1 only (`decoder_sparse_step` 2): the map holds a router and the
    /// expert stacks there and a dense FFN on layer 0, never both on one layer.
    #[test]
    fn the_routed_layer_maps_experts_and_the_dense_layer_a_plain_ffn() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let has = |layer, role| {
            model
                .weights()
                .handle(WeightId::layer(layer, role))
                .is_some()
        };
        let router = WeightRole::Moe(MoeRole::Router);
        let gate_stack = WeightRole::Moe(MoeRole::Stack(ExpertProj::Gate));
        let dense_gate = WeightRole::Ffn(poot_quant::weights::FfnRole::Gate);
        assert!(!has(0, router) && !has(0, gate_stack) && has(0, dense_gate));
        assert!(has(1, router) && has(1, gate_stack) && !has(1, dense_gate));
        // Per-expert HF tensors stack along a new leading expert axis.
        let stack = model
            .weights()
            .handle(WeightId::layer(1, gate_stack))
            .unwrap();
        assert_eq!(stack.shape, vec![8, 64, 64]);
    }

    /// Each routed layer declares the three router witnesses under ids that name its layer; a dense
    /// layer declares none.
    #[test]
    fn a_routed_layer_declares_its_three_witnesses() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let g = model
            .trace(Phase::Decode, step_shape(1, CAP, LogitRows::Last))
            .unwrap();
        let declared: Vec<_> = g
            .witnesses()
            .map(|v| WitnessKind::decode(v.id).unwrap())
            .collect();
        assert_eq!(
            declared,
            [
                (1, WitnessKind::RouterLogitsFinite),
                (1, WitnessKind::SelectorInRange),
                (1, WitnessKind::SelectedSumUsable),
            ]
        );
    }

    /// SC-004's component half: the oracle checks the declared validation roots like an executor
    /// does, so a NaN in the router of the routed layer stops the step with a validation failure
    /// naming `router_logits_finite` and layer 1 (the logits witness of that layer), where the sound
    /// store runs (every other row of this module evaluates it).
    #[test]
    #[should_panic(
        expected = "validation router_logits_finite.l1 (ValidationId(3)) lane 0 failed with f32 bits 0x3f800000"
    )]
    fn a_nan_router_weight_stops_the_step_with_the_logits_witness() {
        let fx = fixture();
        let mut poisoned = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            let entry = if key.as_str() == "model.layers.1.mlp.gate.weight" {
                let mut v = values(entry);
                v[0] = f32::NAN;
                f32_entry(entry.shape(), &v)
            } else {
                entry.clone()
            };
            poisoned.insert(key.clone(), entry).unwrap();
        }
        let poisoned = poisoned.build();
        let model = model_of(&fx.config, &poisoned);
        run_step(&*model, &poisoned, Phase::Decode, CAP, &[3], 0, &[]);
    }

    /// SC-005: a store missing one required tensor (a router, any expert's projection, a dense
    /// layer's FFN, a Q/K norm) fails `Registry::build` with the typed missing-weight error naming
    /// it, before any trace. Mutation: skip the row check; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.experts.3.up_proj.weight",
            "model.layers.1.mlp.experts.7.down_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "lm_head.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
        }
    }

    /// BF16 and Q8_0 stores trace through the one body: the Q8_0 copy packs the attention and dense
    /// projections and every expert's three projections (one packed weight per expert), each
    /// equals (tier 2) the same body over its store decoded to F32, prefill and a decode step.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body() {
        let fx = fixture();
        let prompt = [3, 17, 40, 8];
        for (label, store) in [
            ("bf16", fx.store.clone()),
            ("q8_0", q8_0_store(&fx.store, is_projection)),
        ] {
            let model = model_of(&fx.config, &store);
            let g = model
                .trace(
                    Phase::Prefill,
                    step_shape(prompt.len(), CAP, LogitRows::All),
                )
                .unwrap();
            let formats = poot_graph_plan::WeightFormats::from_weight_map(model.weights());
            let bound = poot_graph_plan::bind_packed_weights(&g, &formats).unwrap();
            let dequants = bound
                .eqns()
                .iter()
                .filter(|e| matches!(e.op, OpKind::PackedDequant { .. }))
                .count();
            let indexed = bound
                .eqns()
                .iter()
                .filter(|e| {
                    matches!(
                        e.op,
                        OpKind::PackedContraction {
                            rows: poot_graph_ir::ContractionRows::Indexed { .. },
                            ..
                        }
                    )
                })
                .count();
            // 4 attention projections per layer and a dense FFN's 3 on layer 0 decode as projections;
            // the 3 expert-projection stacks of layer 1 are one indexed contraction each.
            let (packed, stacks) = if label == "q8_0" {
                (2 * 4 + 3, 3)
            } else {
                (0, 0)
            };
            assert_eq!((dequants, indexed), (packed, stacks), "{label}");

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

    fn config_error_of(config: &serde_json::Value) -> (&'static str, ConfigReason) {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        match build(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    /// A config qwen3-MoE cannot trace is a typed error naming the field: the `sparse_layer`
    /// asserts of the old tracer are config errors now.
    #[test]
    fn a_config_qwen3_moe_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "num_experts_per_tok",
                json!(9),
                "top_k",
                ConfigReason::Exceeds { max: 8 },
            ),
            ("num_experts", json!(0), "num_experts", ConfigReason::Zero),
            (
                "moe_intermediate_size",
                json!(0),
                "moe_intermediate_size",
                ConfigReason::Zero,
            ),
            (
                "decoder_sparse_step",
                json!(0),
                "decoder_sparse_step",
                ConfigReason::Zero,
            ),
            (
                "mlp_only_layers",
                json!("all"),
                "mlp_only_layers",
                ConfigReason::WrongType,
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
        // A dense layer needs the dense width.
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("intermediate_size");
        assert_eq!(
            config_error_of(&config),
            ("intermediate_size", ConfigReason::Missing)
        );
        // Without a dense layer the dense width is not needed.
        let mut config = fixture_config();
        config["decoder_sparse_step"] = json!(1);
        config.as_object_mut().unwrap().remove("intermediate_size");
        let raw = RawConfig::HfJson {
            config: &config,
            generation: None,
        };
        assert_eq!(Qwen3MoeParams::from_raw(&raw).unwrap().sparse, [true, true]);
    }

    /// HF's `Qwen3MoeDecoderLayer` rule: `mlp_only_layers` forces a layer dense and
    /// `decoder_sparse_step` routes every step-th layer.
    #[test]
    fn the_sparse_layer_rule_follows_the_hf_config() {
        let sparse = |step: u64, mlp_only: serde_json::Value| {
            let mut config = fixture_config();
            config["num_hidden_layers"] = json!(4);
            config["decoder_sparse_step"] = json!(step);
            config["mlp_only_layers"] = mlp_only;
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
            };
            Qwen3MoeParams::from_raw(&raw).unwrap().sparse
        };
        assert_eq!(sparse(1, json!([])), [true; 4]);
        assert_eq!(sparse(2, json!([])), [false, true, false, true]);
        assert_eq!(sparse(1, json!([2])), [true, true, false, true]);
        assert_eq!(sparse(3, json!([])), [false, false, true, false]);
    }

    /// The fixture as a Q8_0 qwen3moe GGUF: projections Q8_0 (each projection's experts one rank-3
    /// tensor), the rest F32, layer 0 with a dense FFN and layer 1 routed; and its HF reference
    /// store holding the same values.
    fn gguf_case() -> (GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let n = |f: &str| fx.config[f].as_u64().unwrap() as u32;
        let experts = n("num_experts") as usize;
        let (mut tensors, mut reference) = (Vec::new(), WeightStore::builder());
        for (key, entry) in q8.iter() {
            reference
                .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                .unwrap();
            let name = key.as_str();
            if name.contains(".mlp.experts.") {
                continue;
            }
            let gguf = match name.strip_suffix(".mlp.gate.weight") {
                Some(prefix) => format!(
                    "blk.{}.ffn_gate_inp.weight",
                    &prefix["model.layers.".len()..]
                ),
                None => gguf_name_with_head(name),
            };
            tensors.push(gguf_tensor(&gguf, entry));
        }
        for layer in [1usize] {
            for (hf, gguf) in [
                ("gate_proj", "ffn_gate_exps"),
                ("up_proj", "ffn_up_exps"),
                ("down_proj", "ffn_down_exps"),
            ] {
                let mut bytes = Vec::new();
                let mut dims = Vec::new();
                for e in 0..experts {
                    let key = format!("model.layers.{layer}.mlp.experts.{e}.{hf}.weight");
                    let (_, d, ty, b) = gguf_tensor("x", q8.get(&key).unwrap());
                    assert_eq!(ty, GGML_Q8_0);
                    bytes.extend(b);
                    dims = d;
                }
                dims.push(experts as u64);
                tensors.push((format!("blk.{layer}.{gguf}.weight"), dims, GGML_Q8_0, bytes));
            }
        }
        let mut kvs = gguf_dense_kvs("qwen3moe", &fx.config);
        kvs.extend([
            (
                "qwen3moe.expert_count".to_string(),
                GgufValue::U32(n("num_experts")),
            ),
            (
                "qwen3moe.expert_used_count".to_string(),
                GgufValue::U32(n("num_experts_per_tok")),
            ),
            (
                "qwen3moe.expert_feed_forward_length".to_string(),
                GgufValue::U32(n("moe_intermediate_size")),
            ),
        ]);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-007's qwen3moe half: a Q8_0 GGUF resolves by its `qwen3moe` architecture key, routes the
    /// layer that holds a router (and only that one), packs every expert per expert, and computes
    /// what the HF body computes over the same values.
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let raw = RawConfig::Gguf(&index);
        assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
        let model = registry.build(&raw, &store).unwrap();
        assert_eq!(
            Qwen3MoeParams::from_raw(&raw).unwrap().sparse,
            [false, true]
        );
        let experts = WeightId::layer(1, WeightRole::Moe(MoeRole::Stack(ExpertProj::Gate)));
        let handle = model.weights().handle(experts).unwrap();
        assert!(matches!(
            handle.format,
            poot_quant::weights::HandleFormat::Packed(_)
        ));
        assert_eq!(
            handle.shape.len(),
            3,
            "the experts are one stack: {handle:?}"
        );
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        let reference = model_of(&fixture_config(), &reference_store);
        let prompt = [3, 17, 40, 8];
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

    /// A GGUF layer without its router is dense, so a missing expert stack on a layer that has a
    /// router is named, not skipped.
    #[test]
    fn a_gguf_routed_layer_missing_an_expert_stack_is_named() {
        let (index, store, _) = gguf_case();
        let mut without = WeightStore::builder();
        for (key, entry) in store.iter() {
            if !key.as_str().starts_with("blk.1.ffn_up_exps.weight") {
                without.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        let err = build(&RawConfig::Gguf(&index), &without.build()).unwrap_err();
        assert!(
            matches!(
                &err,
                ModelError::Weight {
                    source: WeightMapError::Missing { key, .. },
                    ..
                } if key.as_str() == "blk.1.ffn_up_exps.weight"
            ),
            "{err}"
        );
    }

    #[test]
    fn a_shape_the_body_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let n = |v| NonZeroUsize::new(v).unwrap();
        let shape = |rows, tokens, capacity| StepShape {
            rows: n(rows),
            tokens: n(tokens),
            capacity: n(capacity),
            ..step_shape(1, 1, LogitRows::Last)
        };
        assert!(matches!(
            model.trace(Phase::Decode, shape(1, 2, 8)),
            Err(TraceError::ShapeUnsupported { .. })
        ));
        assert!(matches!(
            model.trace(Phase::Prefill, shape(1, 1, 65)),
            Err(TraceError::ShapeUnsupported { .. })
        ));
    }
}
