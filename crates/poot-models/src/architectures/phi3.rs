//! phi3 behind [`Model`]: the Phi-3 / Phi-4-mini dense decoder. It is the plain GQA + RoPE +
//! SwiGLU decoder with two differences: the checkpoint fuses `q|k|v` into one `qkv_proj` and
//! `gate|up` into one `gate_up_proj` (read here as row ranges of the stored tensors, so a packed
//! checkpoint stays packed), and RoPE rotates only the leading `partial_rotary_factor` of each
//! head. LongRoPE (`rope_scaling.type == longrope`, and a GGUF's `rope_factors_*` tensors) is a
//! typed refusal: its per-frequency factors need a computed constant that names a weight.
//!
//! A phi3 GGUF stores Q/K in HF row order (no llama.cpp permutation), so both formats use the
//! half-split rotation.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{WeightMap, WeightStore};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::linear::WeightError;
use crate::components::standard::{
    LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step, standard_layer,
    standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::{GGUF_BASE, GGUF_FUSED_QKV_GATE_UP};
use crate::names::{FusedDims, HF_BASE, HF_FUSED_QKV_GATE_UP, family_weights_fused};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("phi3");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "phi3"),
        (ConfigSource::GgufArchitecture, "phi3"),
    ],
    build,
    fixture,
};

/// phi3's config facts: `Phi3Config`'s defaults (`rms_norm_eps` `1e-5`, `eos_token_id` `32000`)
/// and llama.cpp's GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("phi3"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000.0,
    eps: 1e-5,
    eos: Some(32_000),
};

const ROTARY_FACTOR: Field = Field::hf("partial_rotary_factor");
const GGUF_ROTARY_DIM: &str = "phi3.rope.dimension_count";

/// Tensors a LongRoPE GGUF carries: per-frequency factors a computed table cannot hold.
const GGUF_LONGROPE_FACTORS: [&str; 2] = ["rope_factors_long.weight", "rope_factors_short.weight"];

/// phi3's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Phi3Params {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    q_rows: usize,
    kv_rows: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

/// The leading head dims that rotate: HF rounds `head_dim * partial_rotary_factor` down to even;
/// a GGUF states the width in `rope.dimension_count`.
fn rotary_dim(fields: &Fields<'_>, dims: &DenseDims) -> Result<usize, ModelError> {
    match fields.raw() {
        RawConfig::HfJson { .. } => match fields.opt_f32(ROTARY_FACTOR)? {
            None => Ok(dims.head_dim),
            Some(f) if !(f.is_finite() && f > 0.0) => {
                Err(fields.error(ROTARY_FACTOR.hf, ConfigReason::NotFinitePositive))
            }
            Some(f) if f > 1.0 => {
                Err(fields.error(ROTARY_FACTOR.hf, ConfigReason::Exceeds { max: 1 }))
            }
            Some(1.0) => Ok(dims.head_dim),
            Some(f) => {
                let r = (dims.head_dim as f32 * f) as usize;
                Ok(r - r % 2)
            }
        },
        RawConfig::Gguf(_) => fields.count(Field::new("", GGUF_ROTARY_DIM), Some(dims.head_dim)),
    }
}

impl Phi3Params {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        for unsupported in ["attention_bias", "mlp_bias"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let rope = dims.rope(&fields, &SPEC, rotary_dim(&fields, &dims)?)?;
        let mut attention =
            AttentionParams::new(dims.heads, dims.kv_heads, dims.head_dim, false, rope)
                .map_err(|e| e.for_family(FAMILY))?;
        if let Some(window) = fields.opt_u64(Field::hf("sliding_window"))? {
            let window = usize::try_from(window)
                .map_err(|_| fields.error("sliding_window", ConfigReason::WrongType))?;
            attention = attention
                .with_window(window)
                .map_err(|e| e.for_family(FAMILY))?;
        }
        let norm = dims.norm(&fields, &SPEC)?;
        Ok(Self {
            attention,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            q_rows: dims.heads * dims.head_dim,
            kv_rows: dims.kv_heads * dims.head_dim,
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
pub struct Phi3 {
    params: Phi3Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Phi3Params::from_raw(raw)?;
    let (base, fused): (_, &[_]) = match raw.source() {
        ConfigSource::HfModelType => (HF_BASE, HF_FUSED_QKV_GATE_UP),
        ConfigSource::GgufArchitecture => {
            if let Some(factors) = GGUF_LONGROPE_FACTORS.iter().find(|n| store.contains(n)) {
                return Err(ModelError::Config {
                    family: FAMILY,
                    field: factors,
                    reason: ConfigReason::Unsupported,
                });
            }
            (GGUF_BASE, GGUF_FUSED_QKV_GATE_UP)
        }
    };
    let split = FusedDims {
        q: p.q_rows,
        kv: p.kv_rows,
        inter: p.inter,
    };
    let weights = family_weights_fused(
        FAMILY,
        store,
        &[base, fused],
        p.layers,
        p.head_required,
        split,
    )?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention)?,
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
        chat: ChatFormat::Phi3,
    };
    Ok(Box::new(Phi3 {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Phi3 {
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
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention),
                |b, h| gated_ffn(b, h, &layer.ffn),
            )
        });
        Ok(step.finish(b, logits))
    }
}

/// The fixture's config: a tiny untied phi3 rotating half of each head, whose projection widths are
/// multiples of 32 (so a Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "phi3",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

/// The tensors of a phi3 checkpoint with `config`'s HF dims, by HF name: `hf_base_shapes` with
/// each layer's separate projections replaced by the fused `qkv_proj` and `gate_up_proj`.
pub(crate) fn fixture_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (h, inter, heads) = (
        n("hidden_size"),
        n("intermediate_size"),
        n("num_attention_heads"),
    );
    let d = h / heads;
    let (q, kv) = (heads * d, n("num_key_value_heads") * d);
    let mut shapes = crate::names::hf_base_shapes(config, false);
    shapes.retain(|(key, _)| {
        !["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
            .iter()
            .any(|part| key.contains(part))
    });
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        shapes.push((k("self_attn.qkv_proj.weight"), vec![q + 2 * kv, h]));
        shapes.push((k("mlp.gate_up_proj.weight"), vec![2 * inter, h]));
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
        assert_matches_recorded, assert_missing_weight_is_named, build as built, close, close_all,
        f32_entry, gguf_dense_kvs, gguf_tensor, q8_0_store, read_back_owned, step_shape, values,
    };
    use crate::model::LogitRows;
    use crate::registry::Registry;

    const ROTARY: usize = 8;

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

    /// SC-002: phi3's body (fused split by row range, half-width rotation) gives the top-five
    /// logits recorded from the legacy prefill and decode tracers over the split store at the base
    /// commit `bbd5c3232` (token ids exact, logits tier 2). Mutation: change `rotary_dim` in the
    /// body; red.
    fn assert_matches_recorded_rotary(config: &serde_json::Value) {
        let fx = fixture();
        let model = model_of(config, &fx.store);
        assert_matches_recorded(
            &*model,
            &fx.store,
            &Recorded {
                prefill: [
                    (25, 2.604428),
                    (38, 1.8423237),
                    (2, 1.7933061),
                    (28, 1.737822),
                    (33, 1.5011227),
                ],
                decode: [
                    [
                        (22, 2.158746),
                        (7, 2.1191413),
                        (32, 2.1023972),
                        (37, 1.9883846),
                        (3, 1.8324578),
                    ],
                    [
                        (10, 2.3266144),
                        (37, 2.3251483),
                        (3, 1.6811864),
                        (7, 1.496801),
                        (35, 1.4930514),
                    ],
                    [
                        (2, 1.6799711),
                        (8, 1.2498955),
                        (41, 1.1959983),
                        (10, 1.0664647),
                        (36, 1.0658802),
                    ],
                ],
            },
        );
    }

    #[test]
    fn phi3_matches_the_recorded_legacy_logits_with_partial_rotary() {
        assert_matches_recorded_rotary(&fixture_config());
    }

    /// The rotary width is a real input: a full-width rotation does not give the recorded logits.
    #[test]
    #[should_panic(expected = "prefill: the five largest token ids")]
    fn a_different_rotary_width_is_a_different_model() {
        let mut config = fixture_config();
        config["partial_rotary_factor"] = json!(1.0);
        assert_matches_recorded_rotary(&config);
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        assert_chunked_prefill_equals_decode(&*model, &fx.store);
    }

    /// SC-004 (HF half): the fused BF16 store and its Q8_0 copy trace through the
    /// one body; each fused part is one `PackedDequant` over a row range.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body_and_the_packed_transform() {
        assert_bf16_and_q8_0_trace_through_the_packed_transform(&ENTRY, &fixture(), is_projection);
    }

    /// SC-003: a store missing one required tensor, a fused projection included, fails
    /// `Registry::build` with the typed missing-weight error naming it, before any trace.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.self_attn.qkv_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
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
    fn a_config_phi3_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "partial_rotary_factor",
                json!(1.5),
                "partial_rotary_factor",
                ConfigReason::Exceeds { max: 1 },
            ),
            (
                "partial_rotary_factor",
                json!(-0.5),
                "partial_rotary_factor",
                ConfigReason::NotFinitePositive,
            ),
            (
                "partial_rotary_factor",
                json!("x"),
                "partial_rotary_factor",
                ConfigReason::WrongType,
            ),
            ("sliding_window", json!(0), "window", ConfigReason::Zero),
            (
                "rope_scaling",
                json!({"type": "longrope", "short_factor": [1.0], "long_factor": [1.0]}),
                "rope_scaling.rope_type",
                ConfigReason::Unsupported,
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

    #[test]
    fn defaults_are_phi3s_and_the_chat_format_is_phi3() {
        let fx = fixture();
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        let model = model_of(&config, &fx.store);
        assert_eq!(model.config().eos, BTreeSet::from([32_000]));
        assert_eq!(model.config().chat, ChatFormat::Phi3);
    }

    /// A Q8_0 phi3 GGUF (fused `attn_qkv` and `ffn_up` = gate||up, HF row order) and the HF F32
    /// store with the same values.
    fn gguf_case() -> (GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            let key = key.as_str();
            let name = match key {
                "lm_head.weight" => "output.weight".to_string(),
                "model.embed_tokens.weight" => "token_embd.weight".to_string(),
                "model.norm.weight" => "output_norm.weight".to_string(),
                _ => {
                    let (layer, tensor) = key
                        .strip_prefix("model.layers.")
                        .unwrap()
                        .split_once('.')
                        .unwrap();
                    let tensor = match tensor {
                        "input_layernorm.weight" => "attn_norm.weight",
                        "post_attention_layernorm.weight" => "ffn_norm.weight",
                        "self_attn.qkv_proj.weight" => "attn_qkv.weight",
                        "self_attn.o_proj.weight" => "attn_output.weight",
                        "mlp.gate_up_proj.weight" => "ffn_up.weight",
                        "mlp.down_proj.weight" => "ffn_down.weight",
                        other => panic!("unmapped {other}"),
                    };
                    format!("blk.{layer}.{tensor}")
                }
            };
            tensors.push(gguf_tensor(&name, entry));
            reference
                .insert(key, f32_entry(entry.shape(), &values(entry)))
                .unwrap();
        }
        let mut kvs = gguf_dense_kvs("phi3", &fx.config);
        kvs.push((GGUF_ROTARY_DIM.to_string(), GgufValue::U32(ROTARY as u32)));
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004 (R-562-3): a Q8_0 phi3 GGUF resolves by its architecture key, builds packed
    /// projections traced as `PackedDequant` (the fused row ranges included), reads the rotary
    /// width from `rope.dimension_count`, and equals the HF body over the same values. Mutation:
    /// drop the GGUF key (`Unregistered`).
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

    /// A GGUF carrying LongRoPE factor tensors is refused naming the tensor.
    #[test]
    fn a_gguf_with_longrope_factors_is_refused_naming_the_tensor() {
        let (index, store, _) = gguf_case();
        let mut with = WeightStore::builder();
        for (key, entry) in store.iter() {
            with.insert(key.clone(), entry.clone()).unwrap();
        }
        with.insert(GGUF_LONGROPE_FACTORS[0], f32_entry(vec![8], &[1.0; 8]))
            .unwrap();
        match build(&RawConfig::Gguf(&index), &with.build()) {
            Err(ModelError::Config { field, reason, .. }) => {
                assert_eq!(
                    (field, reason),
                    (GGUF_LONGROPE_FACTORS[0], ConfigReason::Unsupported)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = GgufIndex::from_bytes(&write_gguf(
            &[
                ("general.architecture", GgufValue::Str("phi3".into())),
                ("phi3.embedding_length", GgufValue::U32(64)),
                ("phi3.feed_forward_length", GgufValue::U32(96)),
                ("phi3.block_count", GgufValue::U32(2)),
                ("phi3.attention.head_count", GgufValue::U32(0)),
                ("phi3.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        ))
        .unwrap();
        let err = Phi3Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "phi3.attention.head_count",
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
