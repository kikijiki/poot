//! Gemma 2 behind [`Model`] (HF `gemma2`, GGUF `gemma2`). Differs from Qwen2 and Gemma 3 in three
//! ways:
//! - Alternating attention: even layers slide a window, odd layers are global (full causal).
//! - Attention logit softcapping before the softmax, and a final logit softcap.
//! - No per-head QK norm (unlike Gemma 3); one RoPE table (not local and global).
//!
//! The block topology is the Gemma sandwich: four norms per layer (input, post-attention,
//! pre-feedforward and post-feedforward), a GeGLU feed-forward, and the embedding scaled by
//! `sqrt(hidden_size)`.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{WeightMap, WeightStore};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, geglu_ffn};
use crate::components::linear::WeightError;
use crate::components::norm::{NormKind, NormParams};
use crate::components::standard::{
    LayerNorms, LayerParams, NormPlacement, ParamError, StackParams, StackWeights, Step,
    check_step, standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::{GGUF_BASE, GGUF_SANDWICH_NORMS};
use crate::names::{HF_BASE, HF_SANDWICH_NORMS, family_weights, hf_base_shapes};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

/// The Gemma vocabulary's `<bos>` id, for a config that names none.
const GEMMA_BOS: u32 = 2;

pub const FAMILY: FamilyKey = FamilyKey::new("gemma2");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "gemma2"),
        (ConfigSource::GgufArchitecture, "gemma2"),
    ],
    build,
    fixture,
};

/// Gemma 2's config facts: `Gemma2Config`'s defaults (eos `1`, base `10000`) and llama.cpp's GGUF
/// defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("gemma2"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000.0,
    eps: 1e-6,
    eos: Some(1),
};

const WINDOW: Field = Field::new("sliding_window", "gemma2.attention.sliding_window");
const ATTN_CAP: Field = Field::new("attn_logit_softcapping", "gemma2.attn_logit_softcapping");
const FINAL_CAP: Field = Field::new("final_logit_softcapping", "gemma2.final_logit_softcapping");
const QUERY_SCALAR: Field = Field::hf("query_pre_attn_scalar");
const TIED: Field = Field::hf("tie_word_embeddings");
/// `Gemma2Config`'s defaults for what a config omits (an explicit `null` is no softcap).
const HF_DEFAULT_ATTN_CAP: f32 = 50.0;
const HF_DEFAULT_FINAL_CAP: f32 = 30.0;
const HF_DEFAULT_WINDOW: usize = 4096;
const HF_DEFAULT_QUERY_SCALAR: f32 = 256.0;
/// The 27B model (46 layers) scales queries by the hidden width over the heads, which llama.cpp's
/// GGUF does not record.
const GGUF_27B_LAYERS: usize = 46;

/// Gemma 2's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma2Params {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    /// One per layer: global or sliding-window attention.
    attention: Vec<AttentionParams>,
    layer: LayerParams,
    stack: StackParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

/// A softcap: a config that omits the key takes `default` (HF only), an explicit `null` or a
/// GGUF without the key has none.
fn softcap_of(fields: &Fields<'_>, field: Field, default: f32) -> Result<Option<f32>, ModelError> {
    if let RawConfig::HfJson { config, .. } = fields.raw() {
        match config.get(field.hf) {
            None => return Ok(Some(default)),
            Some(v) if v.is_null() => return Ok(None),
            Some(_) => {}
        }
    }
    fields.opt_f32(field)
}

/// Whether each layer slides, from HF's `layer_types` or Gemma 2's alternation. The even layers
/// slide (HF `Gemma2Config.layer_types` and llama.cpp's `swa_pattern 2`); the odd layers are global.
fn sliding_layers(fields: &Fields<'_>, layers: usize) -> Result<Vec<bool>, ModelError> {
    if let RawConfig::HfJson { config, .. } = fields.raw()
        && let Some(types) = config.get("layer_types").filter(|v| !v.is_null())
    {
        let wrong = || fields.error("layer_types", ConfigReason::WrongType);
        let types = types.as_array().ok_or_else(wrong)?;
        if types.len() != layers {
            return Err(wrong());
        }
        return types
            .iter()
            .map(|t| match t.as_str() {
                Some("sliding_attention") => Ok(true),
                Some("full_attention") => Ok(false),
                _ => Err(wrong()),
            })
            .collect();
    }
    Ok((0..layers).map(|l| l % 2 == 0).collect())
}

impl Gemma2Params {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let mut dims = DenseDims::read(&fields, &SPEC)?;
        let hf = matches!(raw, RawConfig::HfJson { .. });
        // HF Gemma ties the head unless the config says otherwise.
        dims.head_required = hf && !fields.flag(TIED, true)?;
        let kind = if hf {
            NormKind::RmsPlusOne
        } else {
            NormKind::Rms
        };
        let norm = NormParams::of(kind, dims.eps)
            .map_err(|_| fields.error("rms_norm_eps", ConfigReason::NotFinitePositive))?;
        let window = match fields.opt_u64(WINDOW)? {
            Some(w) => usize::try_from(w)
                .map_err(|_| fields.error(fields.key(WINDOW), ConfigReason::WrongType))?,
            None if hf => HF_DEFAULT_WINDOW,
            None => dims.max_positions,
        };
        let attn_cap = softcap_of(&fields, ATTN_CAP, HF_DEFAULT_ATTN_CAP)?;
        let final_cap = softcap_of(&fields, FINAL_CAP, HF_DEFAULT_FINAL_CAP)?;
        let scalar = match fields.opt_f32(QUERY_SCALAR)? {
            Some(s) => s,
            None if hf => HF_DEFAULT_QUERY_SCALAR,
            None if dims.layers == GGUF_27B_LAYERS => (dims.width / dims.heads) as f32,
            None => dims.head_dim as f32,
        };
        let rope = dims.rope(&fields, &SPEC, dims.head_dim)?;
        let param_error = |e: ParamError| e.for_family(FAMILY);
        let attention = sliding_layers(&fields, dims.layers)?
            .into_iter()
            .map(|sliding| {
                let mut a =
                    AttentionParams::new(dims.heads, dims.kv_heads, dims.head_dim, false, rope)
                        .map_err(param_error)?
                        .with_scale(1.0 / scalar.sqrt())
                        .map_err(param_error)?;
                if sliding {
                    a = a.with_window(window).map_err(param_error)?;
                }
                if let Some(cap) = attn_cap {
                    a = a.with_softcap(cap).map_err(param_error)?;
                }
                Ok(a)
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        let mut stack = StackParams::new(norm)
            .with_embed_scale((dims.width as f32).sqrt())
            .map_err(param_error)?;
        if let Some(cap) = final_cap {
            stack = stack.with_softcap(cap).map_err(param_error)?;
        }
        Ok(Self {
            attention,
            layer: LayerParams::new(norm).with_placement(NormPlacement::Sandwich),
            stack,
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
pub struct Gemma2 {
    params: Gemma2Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Gemma2Params::from_raw(raw)?;
    let tables: [&[_]; 2] = match raw.source() {
        ConfigSource::HfModelType => [HF_BASE, HF_SANDWICH_NORMS],
        ConfigSource::GgufArchitecture => [GGUF_BASE, GGUF_SANDWICH_NORMS],
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
    Ok(Box::new(Gemma2 {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Gemma2 {
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

/// The fixture's config: a tiny untied Gemma 2 with four layers (sliding, global, sliding,
/// global), a 3-key window, softcaps small enough to bite and a query scalar that is not the head
/// width; projection widths are multiples of 32 (so a Q8_0 copy is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma2",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "sliding_window": 3,
        "query_pre_attn_scalar": 20.0,
        "attn_logit_softcapping": 5.0,
        "final_logit_softcapping": 3.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let h = config["hidden_size"].as_u64().unwrap() as usize;
    let mut shapes = hf_base_shapes(&config, false);
    for l in 0..config["num_hidden_layers"].as_u64().unwrap() {
        for tensor in ["pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            shapes.push((format!("model.layers.{l}.{tensor}.weight"), vec![h]));
        }
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod family_tests {
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

    /// SC-002: Gemma 2's body (sandwich norms, `1 + w` scales, scaled embedding, GeGLU, attention
    /// and final softcaps, the query scalar) gives the top-five logits recorded from the legacy
    /// tracers at the base commit `bbd5c3232` (token ids exact, logits tier 2), at the last prefill
    /// position and three decode steps, with the legacy weights holding the folded scales and a
    /// window no position reaches. Mutation: flip `NormPlacement::Sandwich` to `Pre`; red.
    #[test]
    fn gemma2_matches_the_recorded_legacy_logits() {
        let fx = fixture();
        let mut config = fixture_config();
        config["sliding_window"] = json!(64);
        let model = model_of(&config, &fx.store);
        assert_matches_recorded(
            &*model,
            &fx.store,
            &Recorded {
                prefill: [
                    (13, 2.6083376),
                    (37, 2.133357),
                    (0, 2.1139193),
                    (46, 1.8942794),
                    (25, 1.817059),
                ],
                decode: [
                    [
                        (45, 2.677609),
                        (31, 2.6584907),
                        (43, 2.5359528),
                        (29, 2.4708424),
                        (8, 2.4562454),
                    ],
                    [
                        (0, 2.8551054),
                        (20, 2.4178848),
                        (43, 2.4127593),
                        (45, 2.2826717),
                        (36, 2.281466),
                    ],
                    [
                        (45, 2.5572348),
                        (5, 2.5558841),
                        (20, 2.3489432),
                        (29, 2.1794949),
                        (31, 2.0976849),
                    ],
                ],
            },
        );
    }

    /// The sliding layers are the even ones (HF `layer_types` and llama.cpp's pattern), `layer_types`
    /// overrides that, and the window matters: the default equals the explicit even-sliding list
    /// (same graph), differs from the odd-sliding list and from no window (a window no position
    /// reaches), which in turn equals all-full layers.
    #[test]
    fn the_even_layers_slide_and_layer_types_override() {
        let fx = fixture();
        let with_types = |types: [&str; 4]| {
            let mut config = fixture_config();
            config["layer_types"] = json!(types);
            config
        };
        let trace = |config: &serde_json::Value| {
            model_of(config, &fx.store)
                .trace(Phase::Prefill, step_shape(3, CAP, LogitRows::All))
                .unwrap()
        };
        let [s, f] = ["sliding_attention", "full_attention"];
        assert_same_graph(&trace(&fixture_config()), &trace(&with_types([s, f, s, f])));

        let run = |config: &serde_json::Value| {
            run_step(
                &*model_of(config, &fx.store),
                &fx.store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40, 8, 25],
                0,
                &[],
            )
            .0
        };
        let default = run(&fixture_config());
        assert_ne!(
            default,
            run(&with_types([f, s, f, s])),
            "the parity matters"
        );
        let all_full = run(&with_types([f, f, f, f]));
        assert_ne!(default, all_full, "the window matters");
        let mut wide = fixture_config();
        wide["sliding_window"] = json!(64);
        close(&run(&wide), &all_full);
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

    /// SC-003: a store missing one required tensor (a sandwich norm) fails `Registry::build` with
    /// the typed missing-weight error naming it, before any trace. Mutation: skip the row check;
    /// red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.3.pre_feedforward_layernorm.weight",
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
    fn a_config_gemma2_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            ("sliding_window", json!(0), "window", ConfigReason::Zero),
            (
                "attn_logit_softcapping",
                json!(-1.0),
                "softcap",
                ConfigReason::NotFinitePositive,
            ),
            (
                "final_logit_softcapping",
                json!(0.0),
                "final_softcap",
                ConfigReason::NotFinitePositive,
            ),
            (
                "query_pre_attn_scalar",
                json!(-1.0),
                "scale",
                ConfigReason::NotFinitePositive,
            ),
            (
                "layer_types",
                json!(["full_attention"]),
                "layer_types",
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

    /// A config that omits the eos, the softcaps and the tie flag takes Gemma 2's defaults: eos
    /// `1`, softcaps 50 and 30 (an explicit null removes one), a head tied to the embedding.
    #[test]
    fn omitted_keys_take_the_reference_defaults() {
        let fx = fixture();
        let mut config = fixture_config();
        for key in [
            "eos_token_id",
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "tie_word_embeddings",
        ] {
            config.as_object_mut().unwrap().remove(key);
        }
        let mut store = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            if key.as_str() != "lm_head.weight" {
                store.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        let store = store.build();
        let model = model_of(&config, &store);
        assert_eq!(model.config().eos, BTreeSet::from([1]));
        assert_eq!(model.config().chat, ChatFormat::Gemma);

        let run = |config: &serde_json::Value| {
            run_step(
                &*model_of(config, &store),
                &store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40],
                0,
                &[],
            )
            .0
        };
        let mut explicit = config.clone();
        explicit["attn_logit_softcapping"] = json!(50.0);
        explicit["final_logit_softcapping"] = json!(30.0);
        close(&run(&config), &run(&explicit));
        let mut none = config.clone();
        none["final_logit_softcapping"] = json!(null);
        assert_ne!(run(&config), run(&none), "null drops the final softcap");
    }

    /// llama.cpp's GGUF name of a Gemma 2 HF tensor.
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

    /// A Q8_0 Gemma 2 GGUF (projections Q8_0, norms folded to `1 + w` as llama.cpp writes them,
    /// the rest F32, untied head) and the HF F32 store with the same logical values.
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
        let mut kvs = gguf_dense_kvs("gemma2", &fx.config);
        kvs.push(("gemma2.attention.sliding_window".into(), GgufValue::U32(3)));
        kvs.push(("gemma2.attn_logit_softcapping".into(), GgufValue::F32(5.0)));
        kvs.push(("gemma2.final_logit_softcapping".into(), GgufValue::F32(3.0)));
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004: a Q8_0 Gemma 2 GGUF resolves by its architecture key, builds packed projections
    /// traced as `PackedDequant`, and computes what the HF body computes over the same values (the
    /// GGUF's folded scales equal the HF offsets). Mutation: drop the GGUF key from `ENTRY.keys`;
    /// `Unregistered`.
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 28);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        // A GGUF carries no query scalar: it is the head width, 16 here.
        let mut hf = fixture_config();
        hf["query_pre_attn_scalar"] = json!(16.0);
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
                ("general.architecture", GgufValue::Str("gemma2".into())),
                ("gemma2.embedding_length", GgufValue::U32(64)),
                ("gemma2.feed_forward_length", GgufValue::U32(96)),
                ("gemma2.block_count", GgufValue::U32(2)),
                ("gemma2.attention.head_count", GgufValue::U32(0)),
                ("gemma2.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        ))
        .unwrap();
        let err = Gemma2Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "gemma2.attention.head_count",
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

    /// Gemma 2-2B's real dimensions at four layers (two global, two sliding) and a trimmed
    /// vocabulary and feed-forward width, as a zero-weight checkpoint: only the graph's structure
    /// matters here, and a host-only trace of it is fast. `softcap` is the attention-logit cap.
    fn real_shape_prefill(
        softcap: Option<f32>,
        n: usize,
    ) -> poot_graph_ir::Graph<ValidationOutputs> {
        let mut config = json!({
            "model_type": "gemma2",
            "vocab_size": 4096,
            "hidden_size": 2304,
            "intermediate_size": 2560,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "query_pre_attn_scalar": 256.0,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": false,
            "eos_token_id": 1,
            "bos_token_id": 2,
        });
        config["attn_logit_softcapping"] = json!(softcap);
        let n_of = |f: &str| config[f].as_u64().unwrap() as usize;
        let h = n_of("hidden_size");
        let mut shapes = hf_base_shapes(&config, false);
        for l in 0..n_of("num_hidden_layers") {
            for tensor in ["pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                shapes.push((format!("model.layers.{l}.{tensor}.weight"), vec![h]));
            }
        }
        let mut store = WeightStore::builder();
        for (key, shape) in shapes {
            let bytes = vec![0u8; 2 * shape.iter().product::<usize>()];
            let dense = poot_quant::weights::DenseWeight::try_new(
                poot_tensor::DType::BF16,
                shape,
                bytes.into(),
            )
            .unwrap();
            store
                .insert(key, poot_quant::weights::WeightEntry::Dense(dense))
                .unwrap();
        }
        let model = model_of(&config, &store.build());
        model
            .trace(Phase::Prefill, step_shape(n, n, LogitRows::Last))
            .unwrap()
    }

    /// Every layer of the real-dimension prefill flash-fuses with the real softcap and no
    /// materialized `[., ., L, L]` score matrix remains. `FlashAttentionDecode` has no softcap
    /// parameter, so a softcapped decode does not fuse; a prefill carries the cap in
    /// `FlashAttentionPrefill { softcap }`. `n = 2048` collides with no dimension of this config
    /// (hidden 2304, feed-forward 2560, head 256, vocabulary 4096), so the shape-aware score
    /// check below cannot mistake a projection for the scores. Mutation: drop the attention
    /// softcap in the body; the fused ops carry no cap and the first assertion fails.
    #[test]
    fn prefill_with_the_real_softcap_flash_fuses_every_layer() {
        let n = 2048;
        let g = real_shape_prefill(Some(50.0), n);
        let opt = crate::test_support::optimize(&g);
        let softcaps: Vec<Option<f32>> = opt
            .eqns
            .iter()
            .filter_map(|e| match e.op {
                poot_graph_ir::OpKind::FlashAttentionPrefill { softcap, .. } => Some(softcap),
                _ => None,
            })
            .collect();
        assert_eq!(
            softcaps,
            vec![Some(50.0); 4],
            "one fused flash prefill per layer, each carrying gemma2's real softcap (window only \
             changes mask values, not graph structure)"
        );
        assert_no_materialized_scores(&opt, n);
    }

    /// The uncapped variant isolates the body's non-softcap structure: every layer still fuses.
    #[test]
    fn prefill_without_a_softcap_flash_fuses_every_layer() {
        let n = 2048;
        let opt = crate::test_support::optimize(&real_shape_prefill(None, n));
        let fused = opt
            .eqns
            .iter()
            .filter(|e| matches!(e.op, poot_graph_ir::OpKind::FlashAttentionPrefill { .. }))
            .count();
        assert_eq!(fused, 4, "one fused flash prefill per layer");
        assert_no_materialized_scores(&opt, n);
    }

    /// A rank-4 `[., Hq, L, L']` MatMul output with both trailing dims at least `n` is the
    /// materialized softmax(QK^T) matrix, not a projection (rank 3) or the last-position-only
    /// lm_head (second-to-last dim 1).
    fn assert_no_materialized_scores<V: poot_graph_ir::ValidationChannel>(
        g: &poot_graph_ir::Graph<V>,
        n: usize,
    ) {
        for e in &g.eqns {
            if matches!(e.op, poot_graph_ir::OpKind::MatMul) {
                let shape = &g.aval(e.out).shape;
                let r = shape.len();
                assert!(
                    !(r == 4 && shape[r - 1] >= n && shape[r - 2] >= n),
                    "a materialized N^2-scale MatMul the flash fusion should have replaced: {shape:?}"
                );
            }
        }
    }
}
