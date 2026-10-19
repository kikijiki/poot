//! Granite behind [`Model`]. A pre-norm transformer block (GQA, RMSNorm, RoPE) plus Granite's four
//! scalar multipliers (embedding, attention, residual, logits). The MLP is a dense swiglu
//! (`model_type: granite`) or a top-k expert mixture (`model_type: granitemoe`).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{ExpertProj, WeightMap, WeightStore};

use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::linear::WeightError;
use crate::components::moe::{self, GateUpForm, MoeParams, MoeWeights};
use crate::components::router::RouterKind;
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step,
    standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::GGUF_BASE;
use crate::names::{
    ExpertNames, HF_BASE, MoeLayers, MoeNames, family_weights, family_weights_routed,
    hf_base_shapes,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_text::ChatFormat;

// The `granite` family behind [`Model`]: the standard stack and layer with Granite's four scalar
// multipliers as options of the shared components, and `granitemoe`'s experts as the feed-forward
// when the config states an expert count. The legacy dense tracers at the end of this file are the
// tests' independent reference.

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("granite");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "granite"),
        (ConfigSource::HfModelType, "granitemoe"),
        (ConfigSource::GgufArchitecture, "granite"),
        (ConfigSource::GgufArchitecture, "granitemoe"),
    ],
    build,
    fixture,
    variants: &[("granitemoe", moe_fixture)],
};

/// The config facts of one granite architecture: `GraniteConfig`'s defaults (`rms_norm_eps` 1e-6,
/// `rope_theta` 1e4, eos 2), llama.cpp's GGUF defaults and the GGUF keys under the architecture's
/// prefix (`granite.*` or `granitemoe.*`).
#[derive(Clone, Copy)]
struct Keys {
    spec: DenseSpec,
    embedding: Field,
    attention: Field,
    residual: Field,
    logits: Field,
    experts: Field,
    top_k: Field,
}

macro_rules! granite_keys {
    ($arch:literal) => {
        Keys {
            spec: DenseSpec {
                family: FAMILY,
                gguf: dense_keys!($arch),
                hf_theta: 10_000.0,
                gguf_theta: Some(10_000.0),
                eps: 1e-6,
                eos: Some(2),
            },
            embedding: Field::new("embedding_multiplier", concat!($arch, ".embedding_scale")),
            attention: Field::new("attention_multiplier", concat!($arch, ".attention.scale")),
            residual: Field::new("residual_multiplier", concat!($arch, ".residual_scale")),
            logits: Field::new("logits_scaling", concat!($arch, ".logit_scale")),
            experts: Field::new("num_local_experts", concat!($arch, ".expert_count")),
            top_k: Field::new("num_experts_per_tok", concat!($arch, ".expert_used_count")),
        }
    };
}

const DENSE_KEYS: Keys = granite_keys!("granite");
const MOE_KEYS: Keys = granite_keys!("granitemoe");

/// The keys a config is read under: a GGUF of architecture `granitemoe` writes its facts under
/// that prefix, every other checkpoint under `granite`'s. Only the GGUF key prefix differs; the
/// feed-forward routes exactly when the config states an expert count.
fn keys_of(raw: &RawConfig<'_>) -> &'static Keys {
    match raw {
        RawConfig::Gguf(gguf) if gguf.architecture() == Some("granitemoe") => &MOE_KEYS,
        RawConfig::Gguf(_) | RawConfig::HfJson { .. } => &DENSE_KEYS,
    }
}

/// HF `granitemoe` stores every layer's experts fused: `input_linear` is `[experts, 2 * inter,
/// width]` (gate rows then up rows of each expert), `output_linear` `[experts, width, inter]`.
const HF_MOE: MoeNames = MoeNames {
    prefix: "model.layers.",
    router: ".block_sparse_moe.router.layer",
    router_bias: None,
    shared: None,
    experts: &[
        ExpertNames {
            proj: ExpertProj::GateUp,
            stack: Some(".block_sparse_moe.input_linear.weight"),
            each: None,
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Down,
            stack: Some(".block_sparse_moe.output_linear.weight"),
            each: None,
            bias: None,
        },
    ],
};

/// A `granitemoe` GGUF splits them: one stacked tensor per projection.
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

/// A Granite scalar: HF's config class defaults it to 1; a GGUF must carry it.
fn scalar(fields: &Fields<'_>, field: Field) -> Result<f32, ModelError> {
    match (fields.opt_f32(field)?, fields.raw()) {
        (Some(v), _) => Ok(v),
        (None, RawConfig::HfJson { .. }) => Ok(1.0),
        (None, RawConfig::Gguf(_)) => Err(fields.error(fields.key(field), ConfigReason::Missing)),
    }
}

/// granite's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraniteConfig {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    /// The experts of every layer's feed-forward when the config names an expert count (`inter`
    /// is then each expert's width).
    moe: Option<MoeParams>,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl GraniteConfig {
    pub(crate) fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let keys = keys_of(raw);
        let dims = DenseDims::read(&fields, &keys.spec)?;
        for unsupported in ["attention_bias", "mlp_bias"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        // GraniteConfig ties the head to the embedding unless told otherwise.
        let head_required = match raw {
            RawConfig::HfJson { .. } => !fields.flag(Field::hf("tie_word_embeddings"), true)?,
            RawConfig::Gguf(_) => false,
        };
        let param = |e: crate::components::standard::ParamError| e.for_family(FAMILY);
        let norm = dims.norm(&fields, &keys.spec)?;
        let attention = dims
            .attention(&fields, &keys.spec, false)?
            .with_scale(scalar(&fields, keys.attention)?)
            .map_err(param)?;
        let layer = LayerParams::new(norm)
            .with_residual_scale(scalar(&fields, keys.residual)?)
            .map_err(param)?;
        let stack = StackParams::new(norm)
            .with_embed_scale(scalar(&fields, keys.embedding)?)
            .map_err(param)?
            .with_logit_divisor(scalar(&fields, keys.logits)?)
            .map_err(param)?;
        let moe = match fields.opt_u64(keys.experts)? {
            None | Some(0) => None,
            Some(_) => {
                let gate_up = match raw {
                    RawConfig::HfJson { .. } => GateUpForm::FusedHalves,
                    RawConfig::Gguf(_) => GateUpForm::Split,
                };
                Some(
                    MoeParams::new(
                        FAMILY,
                        dims.width,
                        dims.inter,
                        fields.count(keys.experts, None)?,
                        fields.count(keys.top_k, None)?,
                        RouterKind::TopKSoftmax,
                    )?
                    .with_gate_up(gate_up),
                )
            }
        };
        Ok(Self {
            moe,
            attention,
            layer,
            stack,
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            head_required,
            eos: dims.eos,
            bos: dims.bos,
        })
    }
}

#[derive(Debug)]
enum Ffn {
    Dense(GatedFfnWeights),
    Routed(Box<MoeWeights>, MoeParams),
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    ffn: Ffn,
}

#[derive(Debug)]
pub(crate) struct Granite {
    params: GraniteConfig,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = GraniteConfig::from_raw(raw)?;
    let (rows, moe_names) = match raw.source() {
        ConfigSource::HfModelType => (HF_BASE, &HF_MOE),
        ConfigSource::GgufArchitecture => (GGUF_BASE, &GGUF_MOE),
    };
    let weights = match &p.moe {
        None => family_weights(FAMILY, store, &[rows], p.layers, p.head_required)?,
        Some(moe) => family_weights_routed(
            FAMILY,
            store,
            &[rows],
            p.layers,
            p.head_required,
            MoeLayers {
                names: moe_names,
                sparse: &vec![true; p.layers],
                experts: moe.experts(),
            },
        )?,
    };
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            let ffn = match &p.moe {
                None => Ffn::Dense(GatedFfnWeights::new(&weights, l, p.width, p.inter)?),
                Some(moe) => Ffn::Routed(Box::new(MoeWeights::new(&weights, l, moe)?), *moe),
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
        chat: ChatFormat::Granite,
    };
    Ok(Box::new(Granite {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Granite {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, phase: Phase, shape: StepShape, b: Builder) -> Result<Graph, TraceError> {
        let p = &self.params;
        check_step(FAMILY, phase, shape, p.max_positions)?;
        if let Some(params) = &p.moe {
            moe::check_step(FAMILY, shape, params)?;
        }
        let step = Step::new(&b, shape)?;
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
                    Ffn::Routed(w, params) => moe::moe(b, &step, l, h, None, w, params),
                },
            )
        })?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny untied granite with non-trivial multipliers, whose projection widths
/// are multiples of 32 (so a Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "granite",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "embedding_multiplier": 2.0,
        "attention_multiplier": 0.2,
        "residual_multiplier": 0.5,
        "logits_scaling": 2.0,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_base_shapes(&config, false);
    Fixture::bf16(config, shapes)
}

/// The shape of the `granitemoe` fixture.
const FIXTURE_EXPERTS: usize = 4;
const FIXTURE_WIDTH: usize = 64;
const FIXTURE_INTER: usize = 64;

fn moe_config() -> serde_json::Value {
    let mut config = fixture_config();
    config["model_type"] = serde_json::json!("granitemoe");
    config["intermediate_size"] = serde_json::json!(FIXTURE_INTER);
    config["num_local_experts"] = serde_json::json!(FIXTURE_EXPERTS);
    config["num_experts_per_tok"] = serde_json::json!(2);
    config
}

/// A tiny untied granitemoe: every layer routes; HF stores the experts fused, `input_linear`
/// `[experts, 2 * inter, width]` (gate rows then up rows) and `output_linear` `[experts, width,
/// inter]`.
fn moe_fixture() -> Fixture {
    let config = moe_config();
    let mut shapes: Vec<_> = hf_base_shapes(&config, false)
        .into_iter()
        .filter(|(name, _)| !name.contains(".mlp."))
        .collect();
    for l in 0..2 {
        let k = |s: &str| format!("model.layers.{l}.block_sparse_moe.{s}");
        shapes.push((
            k("router.layer.weight"),
            vec![FIXTURE_EXPERTS, FIXTURE_WIDTH],
        ));
        shapes.push((
            k("input_linear.weight"),
            vec![FIXTURE_EXPERTS, 2 * FIXTURE_INTER, FIXTURE_WIDTH],
        ));
        shapes.push((
            k("output_linear.weight"),
            vec![FIXTURE_EXPERTS, FIXTURE_WIDTH, FIXTURE_INTER],
        ));
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod family_tests {
    use poot_load::gguf::GgufValue;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, Recorded, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_recorded, assert_missing_weight_is_named, build as built, close, close_all,
        f32_entry, gguf_dense_kvs, gguf_name_with_head, gguf_tensor, q8_0_store, read_back_owned,
        values,
    };
    use crate::registry::Registry;

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// SC-002 (ADR-0101 tiers 2 and 3): the standard-stack body gives the top-five logits recorded
    /// from the per-mode granite tracers it replaced, over the fixture (token ids exact, logits
    /// tier 2). Mutation: change the residual multiplier the family passes
    /// (`with_residual_scale`); red.
    #[test]
    fn granite_matches_the_recorded_legacy_logits() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        assert_matches_recorded(
            &*model,
            &fx.store,
            &Recorded {
                prefill: [
                    (13, 0.88173276),
                    (46, 0.8056431),
                    (26, 0.6313378),
                    (37, 0.5623233),
                    (33, 0.4459815),
                ],
                decode: [
                    [
                        (47, 1.1704509),
                        (38, 1.0387655),
                        (45, 0.81752884),
                        (29, 0.6982854),
                        (46, 0.67362595),
                    ],
                    [
                        (24, 1.3668864),
                        (46, 0.88934255),
                        (0, 0.8467298),
                        (47, 0.7711986),
                        (1, 0.63357425),
                    ],
                    [
                        (5, 1.0899231),
                        (24, 1.0546522),
                        (12, 0.83868855),
                        (20, 0.83128154),
                        (25, 0.7864492),
                    ],
                ],
            },
        );
    }

    /// Each of the four multipliers is a different model: changing it moves the logits.
    #[test]
    fn each_multiplier_changes_the_logits() {
        let fx = fixture();
        let run = |config: &serde_json::Value| {
            let model = model_of(config, &fx.store);
            run_step(
                &*model,
                &fx.store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40],
                0,
                &[],
            )
            .0
        };
        let base = run(&fx.config);
        for (key, value) in [
            ("embedding_multiplier", 3.0),
            ("attention_multiplier", 0.5),
            ("residual_multiplier", 0.9),
            ("logits_scaling", 4.0),
        ] {
            let mut config = fx.config.clone();
            config[key] = json!(value);
            assert_ne!(run(&config), base, "{key}");
        }
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// SC-004 (HF half): BF16 and Q8_0 stores trace through the one body.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body_and_the_packed_transform() {
        assert_bf16_and_q8_0_trace_through_the_packed_transform(&ENTRY, &fixture(), is_projection);
    }

    /// SC-003: a store missing one required tensor fails `Registry::build` with the typed
    /// missing-weight error naming it, before any trace. Mutation: skip the row check; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.norm.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
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

    #[test]
    fn a_config_granite_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "logits_scaling",
                json!(0.0),
                "logit_divisor",
                ConfigReason::NotFinitePositive,
            ),
            (
                "residual_multiplier",
                json!(-1.0),
                "residual_scale",
                ConfigReason::NotFinitePositive,
            ),
            (
                "attention_multiplier",
                json!("x"),
                "attention_multiplier",
                ConfigReason::WrongType,
            ),
            (
                "num_attention_heads",
                json!(0),
                "num_attention_heads",
                ConfigReason::Zero,
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

    /// GraniteConfig's defaults: a missing eos is 2, the scalars default to 1, and the head is
    /// tied unless the config says otherwise (no `lm_head.weight` needed).
    #[test]
    fn omitted_fields_take_the_reference_defaults_and_the_head_is_tied() {
        let fx = fixture();
        let mut config = fixture_config();
        for key in [
            "eos_token_id",
            "tie_word_embeddings",
            "embedding_multiplier",
            "attention_multiplier",
            "residual_multiplier",
            "logits_scaling",
        ] {
            config.as_object_mut().unwrap().remove(key);
        }
        let mut store = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            if key.as_str() != "lm_head.weight" {
                store.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        let model = model_of(&config, &store.build());
        assert_eq!(model.config().eos, BTreeSet::from([2]));
        assert_eq!(model.config().chat, ChatFormat::Granite);
    }

    /// A Q8_0 granite GGUF (projections Q8_0, the rest F32, scalars in `granite.*` keys).
    fn gguf_case() -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            tensors.push(gguf_tensor(&gguf_name_with_head(key.as_str()), entry));
            reference
                .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                .unwrap();
        }
        let mut kvs = gguf_dense_kvs("granite", &fx.config);
        kvs.extend([
            ("granite.embedding_scale".to_string(), GgufValue::F32(2.0)),
            ("granite.attention.scale".to_string(), GgufValue::F32(0.2)),
            ("granite.residual_scale".to_string(), GgufValue::F32(0.5)),
            ("granite.logit_scale".to_string(), GgufValue::F32(2.0)),
        ]);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004 (R-562-3): a Q8_0 granite GGUF resolves by its architecture key, builds packed
    /// projections traced as `PackedDequant`, and computes what the HF body computes over the same
    /// values. Mutation: drop the GGUF key from `ENTRY.keys` (`Unregistered`).
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 14);
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
        close_all(&state, &want_state);
    }

    /// A GGUF without one of the Granite scalars is a typed error naming its key.
    #[test]
    fn a_gguf_missing_a_scalar_names_its_key() {
        let (index, ..) = gguf_case();
        let mut kvs = gguf_dense_kvs("granite", &fixture_config());
        kvs.push(("granite.embedding_scale".to_string(), GgufValue::F32(2.0)));
        let (partial, _) = read_back_owned(&kvs, &[]);
        drop(index);
        let err = GraniteConfig::from_raw(&RawConfig::Gguf(&partial)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "granite.attention.scale",
                    reason: ConfigReason::Missing,
                    ..
                }
            ),
            "{err}"
        );
    }
}

/// granitemoe: the granite body whose feed-forward routes every layer through experts.
#[cfg(test)]
mod moe_tests {
    use std::sync::Arc;

    use poot_load::gguf::GgufValue;
    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{MoeRole, WeightEntry, WeightKey, WeightMapError, WeightRole};
    use serde_json::json;

    use super::*;
    use crate::architectures::olmoe::moe_support::{
        GgufName, assert_bf16_and_q8_0_experts_trace_and_match, gguf_checkpoint,
    };
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build as built,
        close, f32_entry, gguf_dense_kvs, gguf_name_with_head, values,
    };
    use crate::model::Phase;
    use crate::registry::{LoadModelError, Registry};

    use super::{FIXTURE_EXPERTS as EXPERTS, FIXTURE_INTER as INTER, FIXTURE_WIDTH as WIDTH};

    fn is_attention(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn has_router(model: &dyn Model) -> bool {
        model
            .weights()
            .iter()
            .any(|(id, _, _)| id.role == WeightRole::Moe(MoeRole::Router))
    }

    /// SC-007: a `granitemoe` HF fixture resolves to the granite family with experts; the dense
    /// `granite` fixture of the same family has none.
    #[test]
    fn a_granitemoe_fixture_resolves_to_granite_with_experts() {
        let registry = Registry::builtin().unwrap();
        let fx = moe_fixture();
        assert_eq!(registry.resolve(&fx.raw()).unwrap().family, FAMILY);
        let model = registry.build(&fx.raw(), &fx.store).unwrap();
        assert!(has_router(&*model));
        assert_eq!(model.config().chat, ChatFormat::Granite);
        let dense = fixture();
        assert!(!has_router(
            &*registry.build(&dense.raw(), &dense.store).unwrap()
        ));
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = moe_fixture();
        let raw = fx.raw();
        assert_chunked_prefill_equals_decode(&*built(&ENTRY, &raw, &fx.store), &fx.store);
    }

    /// The fused HF experts are one stored `[experts, 2 * inter, width]` stack and cannot be one
    /// `[out, K]` packed payload, so only the attention projections pack here; the experts stay
    /// BF16 stacks. (The packed experts are the GGUF row below.)
    #[test]
    fn bf16_experts_trace_through_the_one_body_and_match_the_decoded_reference() {
        assert_bf16_and_q8_0_experts_trace_and_match_dense_experts(&moe_fixture());
    }

    fn assert_bf16_and_q8_0_experts_trace_and_match_dense_experts(fixture: &Fixture) {
        // The helper packs what `is_packed` names: the attention projections, never an expert.
        assert_bf16_and_q8_0_experts_trace_and_match(&ENTRY, fixture, is_attention);
    }

    /// SC-005: a granitemoe store missing the router or a fused expert tensor fails
    /// `Registry::build` with the typed missing-weight error naming it.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = moe_fixture();
        for removed in [
            "model.layers.1.block_sparse_moe.router.layer.weight",
            "model.layers.0.block_sparse_moe.input_linear.weight",
            "model.layers.1.block_sparse_moe.output_linear.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
        }
    }

    /// `top_k` above the expert count and a zero `top_k` are typed config errors naming the field.
    #[test]
    fn an_expert_config_granite_cannot_trace_is_a_typed_error() {
        let fx = moe_fixture();
        let cases = [
            (
                "num_experts_per_tok",
                json!(5),
                "top_k",
                ConfigReason::Exceeds { max: EXPERTS },
            ),
            (
                "num_experts_per_tok",
                json!(0),
                "num_experts_per_tok",
                ConfigReason::Zero,
            ),
        ];
        for (key, value, field, reason) in cases {
            let mut config = moe_config();
            config[key] = value;
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
            };
            match build(&raw, &fx.store) {
                Err(ModelError::Config {
                    field: got,
                    reason: why,
                    ..
                }) => assert_eq!((got, why), (field, reason)),
                other => panic!("{key}: expected a config error, got {other:?}"),
            }
        }
    }

    /// A granitemoe GGUF and the HF store holding the same values: the experts are one random Q8_0
    /// payload per expert and projection, written as llama.cpp's three stacked tensors; the HF
    /// reference fuses their decoded gate and up rows into `input_linear` and keeps `down` as
    /// `output_linear`.
    fn gguf_case() -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = moe_fixture();
        let base = crate::components::testing::q8_0_store(&fx.store, is_attention);
        let mut gguf_source = WeightStore::builder();
        let mut reference = WeightStore::builder();
        for (key, entry) in base.iter() {
            if key.as_str().contains("block_sparse_moe") {
                let entry = if key.as_str().ends_with("router.layer.weight") {
                    entry.clone()
                } else {
                    continue;
                };
                gguf_source.insert(key.clone(), entry.clone()).unwrap();
                reference.insert(key.clone(), entry).unwrap();
            } else {
                gguf_source.insert(key.clone(), entry.clone()).unwrap();
                reference
                    .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                    .unwrap();
            }
        }
        let payload = |seed: u64, shape: [usize; 2]| {
            Arc::new(poot_quant::test_support::packed::random_payload(
                WeightFormat::Q8_0,
                shape,
                seed,
            ))
        };
        for layer in 0..2usize {
            let (mut fused, mut down) = (Vec::new(), Vec::new());
            let mut gates = Vec::new();
            let mut ups = Vec::new();
            for e in 0..EXPERTS {
                let seed = (layer * EXPERTS + e) as u64 * 3;
                let gate = payload(seed, [INTER, WIDTH]);
                let up = payload(seed + 1, [INTER, WIDTH]);
                let down_p = payload(seed + 2, [WIDTH, INTER]);
                for (proj, p) in [("gate", &gate), ("up", &up), ("down", &down_p)] {
                    gguf_source
                        .insert(
                            format!("L{layer}.E{e}.{proj}"),
                            WeightEntry::Packed(Arc::clone(p)),
                        )
                        .unwrap();
                }
                gates.push(values(&WeightEntry::Packed(gate)));
                ups.push(values(&WeightEntry::Packed(up)));
                down.extend(values(&WeightEntry::Packed(down_p)));
            }
            for e in 0..EXPERTS {
                fused.extend_from_slice(&gates[e]);
                fused.extend_from_slice(&ups[e]);
            }
            let k = |s: &str| format!("model.layers.{layer}.block_sparse_moe.{s}");
            reference
                .insert(
                    k("input_linear.weight"),
                    f32_entry(vec![EXPERTS, 2 * INTER, WIDTH], &fused),
                )
                .unwrap();
            reference
                .insert(
                    k("output_linear.weight"),
                    f32_entry(vec![EXPERTS, WIDTH, INTER], &down),
                )
                .unwrap();
        }
        let mut kvs = gguf_dense_kvs("granitemoe", &moe_config());
        kvs.extend([
            (
                "granitemoe.embedding_scale".to_string(),
                GgufValue::F32(2.0),
            ),
            (
                "granitemoe.attention.scale".to_string(),
                GgufValue::F32(0.2),
            ),
            ("granitemoe.residual_scale".to_string(), GgufValue::F32(0.5)),
            ("granitemoe.logit_scale".to_string(), GgufValue::F32(2.0)),
            (
                "granitemoe.expert_count".to_string(),
                GgufValue::U32(EXPERTS as u32),
            ),
            (
                "granitemoe.expert_used_count".to_string(),
                GgufValue::U32(2),
            ),
        ]);
        let (index, store) = gguf_checkpoint(&kvs, &gguf_source.build(), |key| {
            if let Some(rest) = key.strip_prefix('L') {
                let (layer, rest) = rest.split_once(".E").unwrap();
                let (index, proj) = rest.split_once('.').unwrap();
                return GgufName::Expert {
                    stack: format!("blk.{layer}.ffn_{proj}_exps.weight"),
                    index: index.parse().unwrap(),
                };
            }
            match key.strip_prefix("model.layers.") {
                Some(rest) if rest.ends_with("router.layer.weight") => {
                    let layer = rest.split_once('.').unwrap().0;
                    GgufName::Plain(format!("blk.{layer}.ffn_gate_inp.weight"))
                }
                _ => GgufName::Plain(gguf_name_with_head(key)),
            }
        });
        (index, store, reference.build())
    }

    /// SC-001/SC-004 (GGUF half): a granitemoe GGUF resolves to the granite family with packed
    /// per-expert owners and computes what the HF body computes over the fused F32 values.
    /// Mutation: drop `granitemoe` from the GGUF keys of `ENTRY`; `Unregistered`.
    #[test]
    fn a_q8_0_granitemoe_gguf_equals_the_fused_hf_body() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let raw = RawConfig::Gguf(&index);
        assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
        let model = registry.build(&raw, &store).unwrap();
        let stacks: Vec<_> = model
            .weights()
            .iter()
            .filter(|(id, _, _)| matches!(id.role, WeightRole::Moe(MoeRole::Stack(..))))
            .collect();
        assert_eq!(
            stacks.len(),
            2 * 3,
            "one stack per layer and expert projection"
        );
        for (id, view, handle) in stacks {
            assert!(
                matches!(handle.format, poot_quant::weights::HandleFormat::Packed(_)),
                "{id}: a packed stack"
            );
            assert_eq!(
                handle.shape[0], EXPERTS,
                "{id}: one packed owner per expert"
            );
            assert!(
                matches!(view, poot_quant::weights::WeightView::ExpertStack(keys) if keys.len() == EXPERTS),
                "{id}: the experts viewed as one stack"
            );
        }
        let reference = built(
            &ENTRY,
            &RawConfig::HfJson {
                config: &moe_config(),
                generation: None,
            },
            &reference_store,
        );
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

    /// A granitemoe GGUF missing one expert's stacked tensor names it.
    #[test]
    fn a_granitemoe_gguf_without_an_expert_stack_is_named() {
        let (index, store, _) = gguf_case();
        let mut without = WeightStore::builder();
        for (key, entry) in store.iter() {
            if !key.as_str().starts_with("blk.1.ffn_up_exps.weight") {
                without.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        let err = Registry::builtin()
            .unwrap()
            .build(&RawConfig::Gguf(&index), &without.build())
            .map(|_| ())
            .unwrap_err();
        let LoadModelError::Model(ModelError::Weight {
            source: WeightMapError::Missing { key, .. },
            ..
        }) = &err
        else {
            panic!("{err}");
        };
        assert_eq!(key, &WeightKey::from("blk.1.ffn_up_exps.weight"));
    }
}
