//! Gemma 3 text behind [`Model`] (HF `gemma3_text`, GGUF `gemma3`): the sandwich-norm block
//! (`x += post(block(pre(x)))` around attention and a GeGLU feed-forward), per-head Q/K RMSNorm
//! before RoPE, embeddings scaled by `sqrt(width)`, and per-layer attention: every
//! `sliding_window_pattern`-th layer is global (full causal, the config's RoPE base and scaling), the
//! rest are local (a sliding window and `rope_local_base_freq`).
//!
//! An HF checkpoint stores each norm scale as an offset from one (`1 + w`); llama.cpp's GGUF folds the
//! one in, so the GGUF source uses the plain RMS kind.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::rope_table::RopeFlavor;
use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{WeightMap, WeightStore};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, geglu_ffn};
use crate::components::linear::WeightError;
use crate::components::norm::{NormKind, NormParams};
use crate::components::rope::RopeParams;
use crate::components::standard::{
    LayerNorms, LayerParams, NormPlacement, StackParams, StackWeights, Step, check_step,
    standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::{GGUF_BASE, GGUF_QK_NORM, GGUF_SANDWICH_NORMS};
use crate::names::{HF_BASE, HF_QK_NORM, HF_SANDWICH_NORMS, family_weights, hf_base_shapes};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

/// The Gemma vocabulary's `<bos>` id, for a config that names none.
const GEMMA_BOS: u32 = 2;

pub const FAMILY: FamilyKey = FamilyKey::new("gemma3");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "gemma3_text"),
        (ConfigSource::GgufArchitecture, "gemma3"),
    ],
    build,
    fixture,
};

/// Gemma 3's config facts: `Gemma3TextConfig`'s defaults (eos `1`, global base `1e6`) and
/// llama.cpp's GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("gemma3"),
    hf_theta: 1_000_000.0,
    gguf_theta: 1_000_000.0,
    eps: 1e-6,
    eos: Some(1),
};

const SLIDING_WINDOW: Field = Field::new("sliding_window", "gemma3.attention.sliding_window");
const LOCAL_THETA: Field = Field::new("rope_local_base_freq", "gemma3.rope.freq_base_swa");
const PATTERN: Field = Field::hf("sliding_window_pattern");
const QUERY_SCALAR: Field = Field::hf("query_pre_attn_scalar");
const TIED: Field = Field::hf("tie_word_embeddings");
/// The local RoPE base a config omits (the GGUF omits it too).
const DEFAULT_LOCAL_THETA: f32 = 10_000.0;
/// Every sixth layer is global when a config states no pattern.
const DEFAULT_PATTERN: usize = 6;

/// Gemma 3's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma3Params {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    /// One per layer: the global or the local attention.
    attention: Vec<AttentionParams>,
    layer: LayerParams,
    stack: StackParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

/// Whether each layer is global (full attention), from HF's `layer_types` or the pattern.
fn global_layers(fields: &Fields<'_>, layers: usize) -> Result<Vec<bool>, ModelError> {
    if let RawConfig::HfJson { config, .. } = fields.raw()
        && let Some(types) = config.get("layer_types").filter(|v| !v.is_null())
    {
        let wrong = || fields.error("layer_types", ConfigReason::WrongType);
        let types = types.as_array().ok_or_else(wrong)?;
        if types.len() != layers {
            return Err(fields.error("layer_types", ConfigReason::WrongType));
        }
        return types
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(true),
                Some("sliding_attention") => Ok(false),
                _ => Err(wrong()),
            })
            .collect();
    }
    let pattern = fields.count(PATTERN, Some(DEFAULT_PATTERN))?;
    Ok((0..layers).map(|l| (l + 1) % pattern == 0).collect())
}

impl Gemma3Params {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let mut dims = DenseDims::read(&fields, &SPEC)?;
        let hf = matches!(raw, RawConfig::HfJson { .. });
        // HF Gemma ties the head unless the config says otherwise.
        dims.head_required = hf && !fields.flag(TIED, true)?;
        for unsupported in ["attn_logit_softcapping", "final_logit_softcapping"] {
            if fields.opt_f32(Field::hf(unsupported))?.is_some() {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let kind = if hf {
            NormKind::RmsPlusOne
        } else {
            NormKind::Rms
        };
        let norm = NormParams::of(kind, dims.eps)
            .map_err(|_| fields.error("rms_norm_eps", ConfigReason::NotFinitePositive))?;
        let window =
            match fields.opt_u64(SLIDING_WINDOW)? {
                None => None,
                Some(w) => Some(usize::try_from(w).map_err(|_| {
                    fields.error(fields.key(SLIDING_WINDOW), ConfigReason::WrongType)
                })?),
            };
        let scalar = fields
            .opt_f32(QUERY_SCALAR)?
            .unwrap_or(dims.head_dim as f32);
        let global_rope = dims.rope(&fields, &SPEC, dims.head_dim)?;
        let local_theta = fields.float(LOCAL_THETA, DEFAULT_LOCAL_THETA)?;
        let local_rope = RopeParams::new(dims.head_dim, local_theta, &RopeFlavor::Plain)
            .map_err(|e| fields.error(fields.key(LOCAL_THETA), e.reason))?;
        let param_error = |e: crate::components::standard::ParamError| e.for_family(FAMILY);
        let attention = global_layers(&fields, dims.layers)?
            .into_iter()
            .map(|global| {
                let rope = if global { global_rope } else { local_rope };
                let mut a =
                    AttentionParams::new(dims.heads, dims.kv_heads, dims.head_dim, false, rope)
                        .map_err(param_error)?
                        .with_qk_norm(QkNorm::PerHead(norm))
                        .with_scale(1.0 / scalar.sqrt())
                        .map_err(param_error)?;
                if let (false, Some(w)) = (global, window) {
                    a = a.with_window(w).map_err(param_error)?;
                }
                Ok(a)
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        Ok(Self {
            attention,
            layer: LayerParams::new(norm).with_placement(NormPlacement::Sandwich),
            stack: StackParams::new(norm)
                .with_embed_scale((dims.width as f32).sqrt())
                .map_err(param_error)?,
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
        })
    }
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    ffn: GatedFfnWeights,
}

#[derive(Debug)]
pub struct Gemma3 {
    params: Gemma3Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Gemma3Params::from_raw(raw)?;
    let tables: [&[_]; 3] = match raw.source() {
        ConfigSource::HfModelType => [HF_BASE, HF_QK_NORM, HF_SANDWICH_NORMS],
        ConfigSource::GgufArchitecture => [GGUF_BASE, GGUF_QK_NORM, GGUF_SANDWICH_NORMS],
    };
    let weights = family_weights(FAMILY, store, &tables, p.layers, p.head_required)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::sandwich(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention[l])?,
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
        // Gemma is trained with a mandatory BOS (`<bos>` is id 2 in every Gemma vocabulary).
        prompt_bos: Some(p.bos.unwrap_or(GEMMA_BOS)),
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::Gemma,
    };
    Ok(Box::new(Gemma3 {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Gemma3 {
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
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention[l]),
                |b, h| geglu_ffn(b, h, &layer.ffn),
            )
        });
        Ok(step.finish(b, logits))
    }
}

/// The fixture's config: a tiny untied Gemma 3 with three layers (local, global, local under
/// pattern 2), a 3-key window, distinct global and local bases and a query scalar that is not the
/// head width; projection widths are multiples of 32 (so a Q8_0 copy is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma3_text",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rope_local_base_freq": 100.0,
        "sliding_window": 3,
        "sliding_window_pattern": 2,
        "query_pre_attn_scalar": 20.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

/// The `[out, in]` (or `[n]`) shape of every tensor of a Gemma 3 checkpoint with `config`'s dims
/// by its HF name.
fn fixture_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (h, d) = (n("hidden_size"), n("head_dim"));
    let mut shapes = hf_base_shapes(config, false);
    for l in 0..n("num_hidden_layers") {
        for tensor in ["pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            shapes.push((format!("model.layers.{l}.{tensor}.weight"), vec![h]));
        }
        for tensor in ["q_norm", "k_norm"] {
            shapes.push((
                format!("model.layers.{l}.self_attn.{tensor}.weight"),
                vec![d],
            ));
        }
    }
    shapes
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = fixture_shapes(&config);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, Recorded, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_recorded, assert_missing_weight_is_named, assert_same_graph, build as built,
        close, f32_entry, gguf_dense_kvs, gguf_name_with_head, gguf_tensor, q8_0_store,
        read_back_owned, step_shape, values, with_norm_offsets_folded,
    };
    use crate::model::LogitRows;
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

    /// The new body against the logits recorded from the legacy Gemma 3 tracers at the base commit
    /// `bbd5c3232` (offsets folded into the legacy weights), over the fixture.
    fn assert_matches_recorded_at(window: Option<usize>, rec: &Recorded) {
        let fx = fixture();
        let mut config = fixture_config();
        match window {
            Some(w) => config["sliding_window"] = json!(w),
            None => {
                config.as_object_mut().unwrap().remove("sliding_window");
            }
        }
        let model = model_of(&config, &fx.store);
        assert_matches_recorded(&*model, &fx.store, rec);
    }

    /// SC-002: Gemma 3's body (sandwich norms, `1 + w` scales, per-head Q/K norm, scaled
    /// embedding, GeGLU, local and global layers with their own RoPE bases) gives the top-five
    /// logits recorded from the legacy tracers on the fixture (ADR-0101 tiers 2 and 3). Mutation:
    /// flip `NormPlacement::Sandwich` to `Pre` in the params; red.
    #[test]
    fn gemma3_matches_the_recorded_legacy_logits() {
        assert_matches_recorded_at(
            Some(3),
            &Recorded {
                prefill: [
                    (43, 4.5382624),
                    (39, 3.8156314),
                    (32, 3.5160546),
                    (8, 3.3004577),
                    (34, 2.4858801),
                ],
                decode: [
                    [
                        (33, 4.5783234),
                        (20, 4.1840363),
                        (4, 4.076243),
                        (43, 3.8980255),
                        (28, 3.424512),
                    ],
                    [
                        (20, 4.3983316),
                        (13, 3.2285497),
                        (0, 2.8164723),
                        (11, 2.6864161),
                        (10, 2.3676808),
                    ],
                    [
                        (38, 5.280164),
                        (41, 4.510693),
                        (28, 4.108614),
                        (22, 3.831098),
                        (2, 3.7075896),
                    ],
                ],
            },
        );
    }

    /// With no window the local layers differ from global only by their RoPE base; still equal to
    /// the recorded legacy logits, and the window changes the logits.
    #[test]
    fn gemma3_without_a_window_matches_the_recorded_legacy_logits_and_the_window_matters() {
        assert_matches_recorded_at(
            None,
            &Recorded {
                prefill: [
                    (8, 4.364935),
                    (36, 4.0400724),
                    (47, 3.8377988),
                    (37, 3.535937),
                    (33, 3.2192273),
                ],
                decode: [
                    [
                        (47, 4.0158963),
                        (29, 3.9183118),
                        (31, 3.6892078),
                        (36, 3.044827),
                        (43, 2.816492),
                    ],
                    [
                        (21, 3.649165),
                        (43, 3.1257656),
                        (47, 2.9423616),
                        (36, 2.4730527),
                        (5, 2.3442385),
                    ],
                    [
                        (5, 3.3665996),
                        (30, 3.3190782),
                        (24, 2.9236283),
                        (21, 2.64728),
                        (10, 2.5576835),
                    ],
                ],
            },
        );
        let fx = fixture();
        let run = |config: &serde_json::Value| {
            let model = model_of(config, &fx.store);
            run_step(
                &*model,
                &fx.store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40, 8, 25],
                0,
                &[],
            )
            .0
        };
        let mut open = fixture_config();
        open.as_object_mut().unwrap().remove("sliding_window");
        assert_ne!(run(&fixture_config()), run(&open));
    }

    /// `layer_types` overrides the pattern: explicit all-global layers equal a model with no window.
    #[test]
    fn layer_types_override_the_pattern() {
        let fx = fixture();
        let mut explicit = fixture_config();
        explicit["layer_types"] = json!(["full_attention", "full_attention", "full_attention"]);
        let mut open = fixture_config();
        open.as_object_mut().unwrap().remove("sliding_window");
        let trace = |config: &serde_json::Value| {
            model_of(config, &fx.store)
                .trace(Phase::Prefill, step_shape(3, CAP, LogitRows::All))
                .unwrap()
        };
        // Local layers would still rotate with the local base, so compare a window-free model whose
        // layers are all global against the explicit one: both rotate globally everywhere.
        open["sliding_window_pattern"] = json!(1);
        assert_same_graph(&trace(&explicit), &trace(&open));
        let mut bad = fixture_config();
        bad["layer_types"] = json!(["full_attention"]);
        assert!(matches!(
            build(
                &RawConfig::HfJson {
                    config: &bad,
                    generation: None
                },
                &fx.store
            ),
            Err(ModelError::Config {
                field: "layer_types",
                ..
            })
        ));
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

    /// SC-003: a store missing one required tensor (a sandwich norm, a Q/K norm) fails
    /// `Registry::build` with the typed missing-weight error naming it, before any trace.
    /// Mutation: skip the row check; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.2.pre_feedforward_layernorm.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
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
    fn a_config_gemma3_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attn_logit_softcapping",
                json!(50.0),
                "attn_logit_softcapping",
                ConfigReason::Unsupported,
            ),
            (
                "final_logit_softcapping",
                json!(30.0),
                "final_logit_softcapping",
                ConfigReason::Unsupported,
            ),
            ("sliding_window", json!(0), "window", ConfigReason::Zero),
            (
                "sliding_window_pattern",
                json!(0),
                "sliding_window_pattern",
                ConfigReason::Zero,
            ),
            (
                "query_pre_attn_scalar",
                json!(-1.0),
                "scale",
                ConfigReason::NotFinitePositive,
            ),
            (
                "rope_local_base_freq",
                json!(0.0),
                "rope_local_base_freq",
                ConfigReason::NotFinitePositive,
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

    /// The eos a config omits is Gemma 3's reference default (`1`); the head ties by default.
    #[test]
    fn a_missing_eos_is_the_reference_default_and_the_head_ties_by_default() {
        let fx = fixture();
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        let model = model_of(&config, &fx.store);
        assert_eq!(model.config().eos, BTreeSet::from([1]));
        assert_eq!(model.config().chat, ChatFormat::Gemma);

        // A checkpoint with no `lm_head` and no `tie_word_embeddings` key is tied.
        config
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        let mut store = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            if key.as_str() != "lm_head.weight" {
                store.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        model_of(&config, &store.build());
    }

    /// llama.cpp's GGUF name of a Gemma 3 HF tensor.
    fn gemma_gguf_name(hf: &str) -> String {
        if let Some(rest) = hf.strip_prefix("model.layers.") {
            let (layer, tensor) = rest.split_once('.').unwrap();
            let renamed = match tensor {
                "post_attention_layernorm.weight" => Some("post_attention_norm.weight"),
                "pre_feedforward_layernorm.weight" => Some("ffn_norm.weight"),
                "post_feedforward_layernorm.weight" => Some("post_ffw_norm.weight"),
                _ => None,
            };
            if let Some(renamed) = renamed {
                return format!("blk.{layer}.{renamed}");
            }
        }
        gguf_name_with_head(hf)
    }

    /// A Q8_0 Gemma 3 GGUF (projections Q8_0, norms folded to `1 + w` as llama.cpp writes them, the
    /// rest F32, untied head) and the HF F32 store with the same logical values (offsets unfolded).
    fn gguf_case() -> (GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let folded = with_norm_offsets_folded(&q8);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in folded.iter() {
            tensors.push(gguf_tensor(&gemma_gguf_name(key.as_str()), entry));
            let original = q8.get(key.as_str()).unwrap();
            reference
                .insert(key.clone(), f32_entry(original.shape(), &values(original)))
                .unwrap();
        }
        let mut kvs = gguf_dense_kvs("gemma3", &fx.config);
        kvs.push(("gemma3.attention.sliding_window".into(), GgufValue::U32(3)));
        kvs.push(("gemma3.rope.freq_base_swa".into(), GgufValue::F32(100.0)));
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004: a Q8_0 Gemma 3 GGUF resolves by its architecture key, builds packed projections
    /// traced as `PackedDequant`, and computes what the HF body computes over the same values (the
    /// GGUF's folded scales equal the HF offsets). Mutation: drop the GGUF key from `ENTRY.keys`;
    /// `Unregistered`.
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 21);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        // A GGUF carries no query scalar (the head width, 16 here) and no pattern (every sixth
        // layer is global: none of these three).
        let mut hf = fixture_config();
        hf["query_pre_attn_scalar"] = json!(16.0);
        hf["sliding_window_pattern"] = json!(6);
        let reference = model_of(&hf, &reference_store);
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

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = GgufIndex::from_bytes(&write_gguf(
            &[
                ("general.architecture", GgufValue::Str("gemma3".into())),
                ("gemma3.embedding_length", GgufValue::U32(64)),
                ("gemma3.feed_forward_length", GgufValue::U32(96)),
                ("gemma3.block_count", GgufValue::U32(2)),
                ("gemma3.attention.head_count", GgufValue::U32(0)),
                ("gemma3.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        ))
        .unwrap();
        let err = Gemma3Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "gemma3.attention.head_count",
                    reason: ConfigReason::Zero,
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
