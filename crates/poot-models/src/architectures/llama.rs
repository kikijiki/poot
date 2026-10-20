//! llama behind [`Model`]: the plain GQA + RoPE + SwiGLU decoder. HF `llama` and `mistral` (which
//! adds an optional sliding `window`) and GGUF `llama` (which llama.cpp also uses for Mistral) are
//! keys of this one family; there is no separate Mistral tracer.
//!
//! The body is `standard_stack` over `standard_layer` with the shared attention and gated FFN. A
//! llama-architecture GGUF stores Q and K rows permuted for llama.cpp's interleaved RoPE; its name
//! table ([`GGUF_PAIRED_QK`]) reads them back in the half-split order, so the GGUF and the HF checkpoint
//! trace one and the same rotation, which the planner fuses.

use poot_quant::weights::WeightStore;
use poot_weights::{ExpertProj, WeightMap};

use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::moe::MoeParams;
use crate::components::rope::{GGUF_ROPE_FREQS, rope_freq_factors};
use crate::components::router::RouterKind;
use crate::model::{ConfigReason, FamilyKey, ModelError};
use crate::names::gguf::{GGUF_BASE, GGUF_PAIRED_QK};
#[cfg(feature = "test-support")]
use crate::names::hf_base_shapes;
use crate::names::{ExpertNames, HF_BASE, MoeNames, NameRow};
#[cfg(feature = "test-support")]
use crate::registry::Fixture;
use crate::registry::{ConfigSource, FamilyEntry, RawConfig};
use std::sync::Arc;

use poot_text::{ChatFormat, LlamaToolCodec, MistralToolCodec, ToolCallCodec};

use crate::components::standard_family::{
    Mlp, StandardFamily, StandardParams, entry, require_whole_head_rotation, weights_with_experts,
};

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("llama");

pub(crate) const ENTRY: FamilyEntry = entry::<Llama>();

/// llama's config facts: `LlamaConfig`'s and `MistralConfig`'s defaults (they agree), and
/// llama.cpp's GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("llama"),
    hf_theta: 10_000.0,
    gguf_theta: Some(10_000.0),
    eps: 1e-6,
    eos: Some(2),
};

/// Mixtral's experts: `intermediate_size` is each expert's width and `w1`, `w3`, `w2` are its gate,
/// up and down projections.
const HF_MOE: MoeNames = MoeNames {
    prefix: "model.layers.",
    router: ".block_sparse_moe.gate",
    router_bias: None,
    shared: None,
    experts: &[
        ExpertNames {
            proj: ExpertProj::Gate,
            stack: None,
            each: Some((".block_sparse_moe.experts.", ".w1.weight")),
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Up,
            stack: None,
            each: Some((".block_sparse_moe.experts.", ".w3.weight")),
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Down,
            stack: None,
            each: Some((".block_sparse_moe.experts.", ".w2.weight")),
            bias: None,
        },
    ],
};

/// A llama-architecture GGUF with experts: one stacked tensor per projection, or the older
/// converter's one tensor per expert (`blk.L.ffn_gate.E.weight`).
const GGUF_MOE: MoeNames = MoeNames {
    prefix: "blk.",
    router: ".ffn_gate_inp",
    router_bias: None,
    shared: None,
    experts: &[
        ExpertNames {
            proj: ExpertProj::Gate,
            stack: Some(".ffn_gate_exps.weight"),
            each: Some((".ffn_gate.", ".weight")),
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Up,
            stack: Some(".ffn_up_exps.weight"),
            each: Some((".ffn_up.", ".weight")),
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Down,
            stack: Some(".ffn_down_exps.weight"),
            each: Some((".ffn_down.", ".weight")),
            bias: None,
        },
    ],
};

const EXPERTS: Field = Field::new("num_local_experts", "llama.expert_count");
const TOP_K: Field = Field::new("num_experts_per_tok", "llama.expert_used_count");

/// Which instruct dialect a llama-architecture checkpoint speaks: the chat format it falls back to and
/// the tool-call markup it emits go together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dialect {
    Llama3,
    Mistral,
}

impl Dialect {
    fn chat(self) -> ChatFormat {
        match self {
            Dialect::Llama3 => ChatFormat::Llama3,
            Dialect::Mistral => ChatFormat::Mistral,
        }
    }

    fn tool_codec(self) -> Arc<dyn ToolCallCodec> {
        match self {
            Dialect::Llama3 => Arc::new(LlamaToolCodec),
            Dialect::Mistral => Arc::new(MistralToolCodec),
        }
    }
}

/// llama: the plain GQA + RoPE + SwiGLU decoder, or Mixtral's top-k experts when the config states
/// an expert count.
pub(crate) struct Llama;

impl StandardFamily for Llama {
    const FAMILY: FamilyKey = FAMILY;
    const KEYS: &'static [(ConfigSource, &'static str)] = &[
        (ConfigSource::HfModelType, "llama"),
        (ConfigSource::HfModelType, "mistral"),
        (ConfigSource::HfModelType, "mixtral"),
        (ConfigSource::GgufArchitecture, "llama"),
    ];

    fn configure(raw: &RawConfig<'_>) -> Result<StandardParams, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        // `attention_bias` biases the output projection too, which no weight role carries yet.
        for unsupported in ["attention_bias", "mlp_bias"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let mut attention = dims.attention(&fields, &SPEC, false)?;
        if let Some(window) = fields.opt_u64(Field::hf("sliding_window"))? {
            let window = usize::try_from(window)
                .map_err(|_| fields.error("sliding_window", ConfigReason::WrongType))?;
            attention = attention
                .with_window(window)
                .map_err(|e| e.for_family(FAMILY))?;
        }
        // The `mistral` and `mixtral` keys are Mistral checkpoints: their own `[INST]` template and
        // tool-call markup, not Llama 3's.
        let dialect = match raw {
            RawConfig::HfJson { config, .. }
                if matches!(
                    config.get("model_type").and_then(|v| v.as_str()),
                    Some("mistral" | "mixtral")
                ) =>
            {
                Dialect::Mistral
            }
            RawConfig::HfJson { .. } | RawConfig::Gguf(_) => Dialect::Llama3,
        };
        let norm = dims.norm(&fields, &SPEC)?;
        let mut params = StandardParams {
            embeddable: true,
            ..StandardParams::dense(
                &dims,
                norm,
                attention,
                dialect.chat(),
                Some(dialect.tool_codec()),
            )
        };
        // The feed-forward routes exactly when the config states an expert count (a GGUF writes
        // `llama.expert_count = 0` for a dense model); `inter` is then each expert's width.
        if !matches!(fields.opt_u64(EXPERTS)?, None | Some(0)) {
            let moe = MoeParams::new(
                FAMILY,
                dims.width,
                dims.inter,
                fields.count(EXPERTS, None)?,
                fields.count(TOP_K, None)?,
                RouterKind::TopKSoftmax,
            )?;
            params.mlp = vec![Mlp::Routed(moe); dims.layers];
        }
        Ok(params)
    }

    fn weights(
        raw: &RawConfig<'_>,
        store: &WeightStore,
        params: &mut StandardParams,
    ) -> Result<WeightMap, ModelError> {
        let (tables, moe_names): (&[&[NameRow]], _) = match raw.source() {
            ConfigSource::HfModelType => (&[HF_BASE], &HF_MOE),
            ConfigSource::GgufArchitecture => {
                require_whole_head_rotation(FAMILY, params)?;
                if let Some(entry) = store.get(GGUF_ROPE_FREQS) {
                    let factors = rope_freq_factors(FAMILY, entry)?;
                    for attention in &mut params.attention {
                        *attention = attention
                            .with_rope_freq_factors(&factors)
                            .map_err(|e| e.for_family(FAMILY))?;
                    }
                }
                (&[GGUF_BASE, GGUF_PAIRED_QK], &GGUF_MOE)
            }
        };
        weights_with_experts(FAMILY, store, tables, moe_names, params)
    }
}

/// The fixture's config: a tiny untied llama whose projection widths are multiples of 32 (so a
/// Q8_0 copy of its store is a valid checkpoint too).
#[cfg(feature = "test-support")]
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "llama",
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
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

#[cfg(feature = "test-support")]
pub(crate) fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_base_shapes(&config, false);
    Fixture::bf16(config, shapes)
}

/// How many experts the Mixtral fixture holds.
#[cfg(feature = "test-support")]
const FIXTURE_EXPERTS: usize = 4;

#[cfg(feature = "test-support")]
fn mixtral_config() -> serde_json::Value {
    let mut config = fixture_config();
    config["model_type"] = serde_json::json!("mixtral");
    config["intermediate_size"] = serde_json::json!(64);
    config["num_local_experts"] = serde_json::json!(FIXTURE_EXPERTS);
    config["num_experts_per_tok"] = serde_json::json!(2);
    config
}

/// A tiny untied Mixtral: every layer routes, `intermediate_size` is each expert's width.
#[cfg(feature = "test-support")]
pub(crate) fn mixtral_fixture() -> Fixture {
    let config = mixtral_config();
    let h = 64;
    let inter = 64;
    let mut shapes: Vec<_> = hf_base_shapes(&config, false)
        .into_iter()
        .filter(|(name, _)| !name.contains(".mlp."))
        .collect();
    for l in 0..2 {
        let k = |s: &str| format!("model.layers.{l}.block_sparse_moe.{s}");
        shapes.push((k("gate.weight"), vec![FIXTURE_EXPERTS, h]));
        for e in 0..FIXTURE_EXPERTS {
            shapes.push((k(&format!("experts.{e}.w1.weight")), vec![inter, h]));
            shapes.push((k(&format!("experts.{e}.w3.weight")), vec![inter, h]));
            shapes.push((k(&format!("experts.{e}.w2.weight")), vec![h, inter]));
        }
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_load::gguf::{GgufValue, write_gguf};
    use poot_quant::StoredBytes;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::standard_family::StandardFamily;
    use crate::components::testing::{
        CAP, Recorded, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_recorded, assert_missing_weight_is_named, assert_same_graph, build as built,
        close, close_all, f32_entry, gguf_dense_kvs, gguf_name_with_head, gguf_tensor, q8_0_store,
        read_back_owned, step_shape, values,
    };
    use crate::model::Model;
    use crate::model::StepSpec;
    use crate::model::TraceError;
    use crate::model::{LogitRows, StepShape};
    use crate::registry::Registry;
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn config_with(model_type: &str, window: Option<usize>) -> serde_json::Value {
        let mut config = fixture_config();
        config["model_type"] = json!(model_type);
        if let Some(window) = window {
            config["sliding_window"] = json!(window);
        }
        config
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
            preprocessor: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// The new body against the logits recorded from the legacy tracers at the base commit
    /// `bbd5c3232`, over the fixture, for a window of `window` keys or full attention.
    fn assert_matches_recorded_at(model_type: &str, window: Option<usize>, rec: &Recorded) {
        let fx = fixture();
        let model = model_of(&config_with(model_type, window), &fx.store);
        assert_matches_recorded(&*model, &fx.store, rec);
    }

    /// SC-002: llama's body gives the top-five logits recorded from the legacy prefill and decode
    /// tracers on the fixture (ADR-0101 tiers 2 and 3). Mutation: pass `Post` placement or change
    /// the RoPE theta; red.
    #[test]
    fn llama_matches_the_recorded_legacy_logits() {
        assert_matches_recorded_at(
            "llama",
            None,
            &Recorded {
                prefill: [
                    (47, 2.5254447),
                    (46, 2.19044),
                    (24, 1.8708938),
                    (13, 1.7305588),
                    (20, 1.2541254),
                ],
                decode: [
                    [
                        (47, 2.554118),
                        (1, 1.6575682),
                        (14, 1.5247586),
                        (39, 1.5132793),
                        (29, 1.3596327),
                    ],
                    [
                        (24, 2.7636833),
                        (46, 2.3170958),
                        (45, 1.5511551),
                        (0, 1.2757467),
                        (41, 1.1615623),
                    ],
                    [
                        (35, 2.2088459),
                        (46, 2.1401362),
                        (45, 2.0528457),
                        (34, 1.5905961),
                        (10, 1.3851273),
                    ],
                ],
            },
        );
    }

    /// SC-002: the `mistral` key with a sliding window of 3 keys, which the prompt and the decode
    /// steps exceed, gives the top-five logits recorded from the legacy windowed tracers, and the
    /// window changes the output.
    #[test]
    fn mistral_with_a_window_matches_the_recorded_legacy_logits() {
        assert_matches_recorded_at(
            "mistral",
            Some(3),
            &Recorded {
                prefill: [
                    (39, 2.187981),
                    (47, 2.0849376),
                    (6, 1.6817052),
                    (38, 1.4759992),
                    (32, 1.2051036),
                ],
                decode: [
                    [
                        (47, 2.9925487),
                        (15, 2.0763311),
                        (41, 1.4485627),
                        (36, 1.4396846),
                        (29, 1.3139923),
                    ],
                    [
                        (15, 2.247693),
                        (1, 1.9912384),
                        (25, 1.6233634),
                        (7, 1.4112468),
                        (13, 1.3536422),
                    ],
                    [
                        (20, 2.7654626),
                        (19, 1.9462117),
                        (33, 1.6563671),
                        (25, 1.4697057),
                        (27, 1.467036),
                    ],
                ],
            },
        );
        let fx = fixture();
        let run = |window| {
            let model = model_of(&config_with("mistral", window), &fx.store);
            run_step(&*model, &fx.store, CAP, &[3, 17, 40, 8, 25], 0, &[]).0
        };
        assert_ne!(run(Some(3)), run(None), "the window must change the logits");
    }

    /// SC-001 (spec 999 SC-002): `llama` and `mistral` with no window trace structurally identical
    /// graphs, compared equation by equation with equal names, through one registry key each.
    /// Mutation: add one op to the mistral path outside the standard functions (a scale on the
    /// embedding); red.
    #[test]
    fn llama_and_mistral_without_a_window_trace_identical_graphs() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        let mut graphs = Vec::new();
        for model_type in ["llama", "mistral"] {
            let config = config_with(model_type, None);
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
                preprocessor: None,
            };
            assert_eq!(
                registry.resolve(&raw).unwrap().family,
                FAMILY,
                "{model_type}"
            );
            let model = registry.build(&raw, &fx.store).unwrap();
            graphs.push(
                model
                    .model
                    .trace(
                        &StepSpec::step(step_shape(3, CAP, LogitRows::All)),
                        poot_graph_ir::TraceLimits::for_tests(),
                    )
                    .unwrap(),
            );
        }
        assert_same_graph(&graphs[0], &graphs[1]);
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        assert_chunked_prefill_equals_decode(&*model, &fx.store);
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
            preprocessor: None,
        };
        match (ENTRY.build)(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn a_config_llama_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "mlp_bias",
                json!(true),
                "mlp_bias",
                ConfigReason::Unsupported,
            ),
            ("sliding_window", json!(0), "window", ConfigReason::Zero),
            (
                "sliding_window",
                json!("4"),
                "sliding_window",
                ConfigReason::WrongType,
            ),
            (
                "num_attention_heads",
                json!(0),
                "num_attention_heads",
                ConfigReason::Zero,
            ),
            (
                "rope_scaling",
                json!({"type": "longrope", "short_factor": [1.0]}),
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

    /// The eos set a config omits is llama's reference default (`2`); the union with the
    /// generation config otherwise.
    #[test]
    fn a_missing_eos_is_the_reference_default() {
        let fx = fixture();
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        let model = model_of(&config, &fx.store);
        assert_eq!(model.config().eos, BTreeSet::from([2]));
        assert_eq!(model.config().chat, ChatFormat::Llama3);
        // The family names its tool-call markup: Llama's python tag, not Mistral's.
        let parsed = |model: &dyn Model, text: &str| {
            model
                .config()
                .tool_codec
                .as_ref()
                .map(|codec| codec.parse(text).len())
        };
        let llama_call = "<|python_tag|>{\"name\": \"f\", \"parameters\": {}}";
        let mistral_call = "[TOOL_CALLS][{\"name\": \"f\", \"arguments\": {}}]";
        assert_eq!(parsed(&*model, llama_call), Some(1));
        assert_eq!(parsed(&*model, mistral_call), Some(0));
        let config = config_with("mistral", None);
        let mistral = model_of(&config, &fx.store);
        assert_eq!(mistral.config().chat, ChatFormat::Mistral);
        assert_eq!(parsed(&*mistral, mistral_call), Some(1));
        assert_eq!(parsed(&*mistral, llama_call), Some(0));
    }

    /// Un-permute the rows of llama.cpp's Q/K layout for interleaved RoPE: llama.cpp stores,
    /// within each head block of `d` rows, HF row `j` at `2j` and HF row `j + d / 2` at `2j + 1`.
    fn unpermute_rows(rows: &[Vec<f32>], d: usize) -> Vec<Vec<f32>> {
        let mut out = rows.to_vec();
        for (head, block) in rows.chunks(d).enumerate() {
            for j in 0..d / 2 {
                out[head * d + j] = block[2 * j].clone();
                out[head * d + d / 2 + j] = block[2 * j + 1].clone();
            }
        }
        out
    }

    /// A Q8_0 llama GGUF (Q/K payloads in llama.cpp's permuted order, projections Q8_0, the rest
    /// F32) and the HF F32 store holding the same values un-permuted.
    fn gguf_case() -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            tensors.push(gguf_tensor(&gguf_name_with_head(key.as_str()), entry));
            let shape = entry.shape();
            let decoded = values(entry);
            let is_qk =
                key.as_str().ends_with("q_proj.weight") || key.as_str().ends_with("k_proj.weight");
            let decoded = if is_qk {
                let rows: Vec<Vec<f32>> = decoded.chunks(shape[1]).map(<[f32]>::to_vec).collect();
                unpermute_rows(&rows, 16).concat()
            } else {
                decoded
            };
            reference
                .insert(key.clone(), f32_entry(shape, &decoded))
                .unwrap();
        }
        let kvs = gguf_dense_kvs("llama", &fx.config);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004 (R-562-3): a Q8_0 llama GGUF resolves by its architecture key, builds packed
    /// projections traced as `PackedDequant`, and computes what the HF body computes over the same
    /// values un-permuted: the interleaved rotation over the stored rows is the half-split one over
    /// the HF rows. Mutation: drop the GGUF key (`Unregistered`), or trace the GGUF rows half-split
    /// (the logits diverge).
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body_over_unpermuted_rows() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 14);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        let reference = model_of(&fixture_config(), &reference_store);

        let prompt = [3, 17, 40, 8];
        let (got, state) = run_step(&*model, &store, CAP, &prompt, 0, &[]);
        let (want, want_state) = run_step(&*reference, &reference_store, CAP, &prompt, 0, &[]);
        close(&got, &want);
        // The GGUF rows are read back in the half-split order at load, so the K cache holds the same
        // rotated rows as the HF body's, not permuted ones: compare the whole state and the next logits.
        let (got, _) = run_step(&*model, &store, CAP, &[5], 4, &state);
        let (want, _) = run_step(&*reference, &reference_store, CAP, &[5], 4, &want_state);
        close(&got, &want);
        close_all(&state, &want_state);
    }

    /// SC-002 (POOT-1238): the plan of a Q8_0 llama GGUF decode fuses the rotation of every layer's Q and
    /// K to a `Rope` op, exactly as the HF checkpoint's does, and records no rope decline: the GGUF's
    /// permuted Q/K rows are read back in the half-split order by the name table, so no tracer chooses a
    /// rotation by container. This row guards the fusion (the container branch), not the row order: that
    /// the stored rows are un-permuted correctly is guarded by the Q8_0 equivalence test above, which
    /// compares the logits and the whole state, and by `the_half_split_order_inverts_llama_cpps_permute`.
    /// Mutation: restore the container branch in `configure` (`if matches!(raw, RawConfig::Gguf(_)) {
    /// attention = attention.with_interleaved_rope() }`) and read the stored rows with `GGUF_BASE` alone;
    /// the logits still agree but the GGUF plan has no `Rope` op: `left: 0`, `right: 4`.
    #[test]
    fn a_gguf_llama_decode_plan_fuses_rope_like_the_hf_checkpoint() {
        use poot_graph_ir::OpKind;
        use poot_graph_plan::{CompileLimits, CompileOptions, FusionDeclineReason, Target};
        use poot_target::Backend;

        use crate::test_support::weight_map::trace;

        let target = Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
        };
        let ropes = |model: &dyn Model| {
            let graph = trace(model, model.weights(), 1, CAP);
            let program = poot_graph_plan::compile(
                &graph,
                &target,
                &CompileOptions::new(CompileLimits::STANDARD),
            )
            .unwrap();
            let rope_declines = program
                .declines()
                .iter()
                .filter(|decline| matches!(decline.reason, FusionDeclineReason::Rope(_)))
                .count();
            let ropes = program
                .graph()
                .eqns()
                .iter()
                .filter(|eqn| matches!(eqn.op, OpKind::Rope { .. }))
                .count();
            (ropes, rope_declines)
        };
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let gguf = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 14);
        let hf = model_of(&fixture_config(), &reference_store);
        let layers = 2;
        assert_eq!(
            ropes(&*hf),
            (2 * layers, 0),
            "the HF body fuses q and k of every layer"
        );
        assert_eq!(ropes(&*gguf), ropes(&*hf));
    }

    /// llama.cpp's converter rule for `rope_freqs.weight` (llama3 scaling), in f64 over the fixture's
    /// head dim: the per-frequency divisor of each inverse frequency.
    fn llama3_rope_freqs(factor: f64, low: f64, high: f64, original: f64) -> Vec<f32> {
        let (head_dim, theta) = (16, 10_000.0f64);
        (0..head_dim / 2)
            .map(|j| {
                let wavelen =
                    2.0 * std::f64::consts::PI * theta.powf((2 * j) as f64 / head_dim as f64);
                let divisor = if wavelen < original / high {
                    1.0
                } else if wavelen > original / low {
                    factor
                } else {
                    let smooth = (original / wavelen - low) / (high - low);
                    1.0 / ((1.0 - smooth) / factor + smooth)
                };
                divisor as f32
            })
            .collect()
    }

    fn with_rope_freqs(store: &WeightStore, freqs: &[f32]) -> WeightStore {
        let mut out = WeightStore::builder();
        out.extend_from(store).unwrap();
        out.insert(GGUF_ROPE_FREQS, f32_entry(vec![freqs.len()], freqs))
            .unwrap();
        out.build()
    }

    /// A GGUF carrying llama3's `rope_freqs.weight` rotates by the rescaled frequencies: it equals
    /// the HF body whose config states the same `rope_scaling`, over the same un-permuted values, and
    /// differs from plain RoPE (so the tensor is not ignored). Mutation: skip the tensor in `build`
    /// (the logits equal plain RoPE's, no longer the HF llama3 body's).
    #[test]
    fn a_gguf_with_llama3_rope_freqs_equals_the_hf_llama3_body() {
        let (index, store, reference_store) = gguf_case();
        let freqs = llama3_rope_freqs(8.0, 1.0, 4.0, 64.0);
        assert!(freqs.iter().any(|f| *f > 1.0 && *f < 8.0), "{freqs:?}");
        let store = with_rope_freqs(&store, &freqs);
        let model = (ENTRY.build)(&RawConfig::Gguf(&index), &store).unwrap();
        let mut config = fixture_config();
        config["rope_scaling"] = json!({
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 64,
        });
        let reference = model_of(&config, &reference_store);
        let plain = model_of(&fixture_config(), &reference_store);

        let prompt = [3, 17, 40, 8, 22, 9, 31, 12];
        let run = |m: &dyn Model, s: &WeightStore| run_step(m, s, CAP, &prompt, 0, &[]).0;
        let got = run(&*model.model, &store);
        close(&got, &run(&*reference, &reference_store));
        assert!(
            got.iter()
                .zip(run(&*plain, &reference_store))
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "llama3 rescaling changed nothing at these positions"
        );
    }

    /// A `rope_freqs.weight` that is not a llama3 rescale of the head's frequencies, or not a
    /// per-frequency F32 vector, is refused naming the tensor, never approximated.
    #[test]
    fn a_gguf_rope_freqs_that_is_not_a_llama3_rescale_is_refused_naming_the_tensor() {
        let (index, store, _) = gguf_case();
        let refused = |store: &WeightStore| match (ENTRY.build)(&RawConfig::Gguf(&index), store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{other:?}"),
        };
        // One blended frequency pins no slope; a descending ramp is no llama3 shape.
        let one_blend = [1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 8.0, 8.0];
        let descending = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5];
        for freqs in [&one_blend[..], &descending[..]] {
            assert_eq!(
                refused(&with_rope_freqs(&store, freqs)),
                (GGUF_ROPE_FREQS, ConfigReason::Unsupported),
                "{freqs:?}"
            );
        }
        // Eight frequencies are expected for a 16-wide head.
        assert_eq!(
            refused(&with_rope_freqs(&store, &[1.0; 4])),
            (GGUF_ROPE_FREQS, ConfigReason::WrongType)
        );
    }

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = poot_load::gguf::GgufIndex::parse(&StoredBytes::from(write_gguf(
            &[
                ("general.architecture", GgufValue::Str("llama".into())),
                ("llama.embedding_length", GgufValue::U32(64)),
                ("llama.feed_forward_length", GgufValue::U32(96)),
                ("llama.block_count", GgufValue::U32(2)),
                ("llama.attention.head_count", GgufValue::U32(0)),
                ("llama.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        )))
        .unwrap();
        let err = Llama::configure(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "llama.attention.head_count",
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
        let shape = |rows, tokens, capacity| StepShape::new(n(rows), n(tokens), n(capacity));
        assert!(matches!(
            model.trace(
                &StepSpec::step(shape(1, 9, 8)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported { .. })
        ));
        assert!(matches!(
            model.trace(
                &StepSpec::step(shape(1, 1, 65)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported { .. })
        ));
    }
}

/// Mixtral: the llama body whose feed-forward routes every layer through experts.
#[cfg(test)]
mod moe_tests {
    use poot_load::gguf::GgufValue;
    use poot_quant::weights::{WeightEntry, WeightKey};
    use poot_weights::{MoeRole, WeightMapError, WeightRole};
    use serde_json::json;

    use super::*;
    use crate::architectures::olmoe::moe_support::{
        GgufName, assert_bf16_and_q8_0_experts_trace_and_match, gguf_checkpoint,
    };
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build as built,
        close, f32_entry, gguf_dense_kvs, gguf_name_with_head, gguf_tensor, q8_0_store,
        read_back_owned, values,
    };
    use crate::model::Model;
    use crate::registry::{LoadModelError, Registry};

    use super::FIXTURE_EXPERTS as EXPERTS;

    fn is_packed(key: &str) -> bool {
        key.ends_with("_proj.weight")
            || ["w1", "w2", "w3"]
                .iter()
                .any(|w| key.ends_with(&format!(".{w}.weight")))
    }

    fn has_router(model: &dyn Model) -> bool {
        model
            .weights()
            .iter()
            .any(|(id, _, _)| id.role == WeightRole::Moe(MoeRole::Router))
    }

    /// SC-007: the `mixtral` key resolves to the llama family, and its model has a router and
    /// experts on every layer.
    #[test]
    fn the_mixtral_key_resolves_to_llama_with_experts() {
        let registry = Registry::builtin().unwrap();
        let fx = mixtral_fixture();
        assert_eq!(registry.resolve(&fx.raw()).unwrap().family, FAMILY);
        let model = registry.build(&fx.raw(), &fx.store).unwrap();
        assert!(has_router(&*model.model));
        assert_eq!(model.model.config().chat, ChatFormat::Mistral);
        let routers = model
            .model
            .weights()
            .iter()
            .filter(|(id, _, _)| id.role == WeightRole::Moe(MoeRole::Router))
            .count();
        assert_eq!(routers, 2);
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = mixtral_fixture();
        let raw = fx.raw();
        assert_chunked_prefill_equals_decode(&*built(&ENTRY, &raw, &fx.store), &fx.store);
    }

    #[test]
    fn bf16_and_q8_0_experts_trace_through_the_one_body_and_match_the_decoded_reference() {
        assert_bf16_and_q8_0_experts_trace_and_match(&ENTRY, &mixtral_fixture(), is_packed);
    }

    /// SC-005: a Mixtral store missing one required tensor, the router or one expert's projection,
    /// fails `Registry::build` with the typed missing-weight error naming it. Mutation: report the
    /// first expert instead of the absent one (`ExpertNames::find`); the expert-3 row goes red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = mixtral_fixture();
        for removed in [
            "model.layers.1.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.1.block_sparse_moe.experts.3.w2.weight",
            "model.layers.1.input_layernorm.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
        }
    }

    /// The experts' width is `intermediate_size`; a `top_k` above the expert count is a typed config
    /// error, and so is a zero expert count (no routed layer without experts is silently dense).
    #[test]
    fn an_expert_config_llama_cannot_trace_is_a_typed_error() {
        let fx = mixtral_fixture();
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
            let mut config = mixtral_config();
            config[key] = value;
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
                preprocessor: None,
            };
            match (ENTRY.build)(&raw, &fx.store) {
                Err(ModelError::Config {
                    field: got,
                    reason: why,
                    ..
                }) => assert_eq!((got, why), (field, reason)),
                other => panic!("{key}: expected a config error, got {other:?}"),
            }
        }
    }

    /// llama.cpp's Q/K row permutation, undone (see the dense GGUF row).
    fn unpermuted(key: &str, entry: &WeightEntry) -> WeightEntry {
        let shape = entry.shape();
        let decoded = values(entry);
        if key.ends_with("q_proj.weight") || key.ends_with("k_proj.weight") {
            let mut rows: Vec<Vec<f32>> = decoded.chunks(shape[1]).map(<[f32]>::to_vec).collect();
            let d = 16;
            let stored = rows.clone();
            for (head, block) in stored.chunks(d).enumerate() {
                for j in 0..d / 2 {
                    rows[head * d + j] = block[2 * j].clone();
                    rows[head * d + d / 2 + j] = block[2 * j + 1].clone();
                }
            }
            f32_entry(shape, &rows.concat())
        } else {
            f32_entry(shape, &decoded)
        }
    }

    fn gguf_kvs(experts: Option<usize>) -> Vec<(String, GgufValue)> {
        let mut kvs = gguf_dense_kvs("llama", &mixtral_config());
        if let Some(experts) = experts {
            kvs.push(("llama.expert_count".into(), GgufValue::U32(experts as u32)));
            kvs.push(("llama.expert_used_count".into(), GgufValue::U32(2)));
        }
        kvs
    }

    /// llama.cpp's name of a Mixtral tensor: the stacked `ffn_*_exps` or, for the older converter,
    /// one tensor per expert (`blk.L.ffn_gate.E.weight`).
    fn gguf_name_of(hf: &str, stacked: bool) -> GgufName {
        let Some(rest) = hf.strip_prefix("model.layers.") else {
            return GgufName::Plain(gguf_name_with_head(hf));
        };
        let (layer, tensor) = rest.split_once('.').unwrap();
        let Some(moe) = tensor.strip_prefix("block_sparse_moe.") else {
            return GgufName::Plain(gguf_name_with_head(hf));
        };
        if moe == "gate.weight" {
            return GgufName::Plain(format!("blk.{layer}.ffn_gate_inp.weight"));
        }
        let (index, w) = moe
            .strip_prefix("experts.")
            .and_then(|e| e.split_once('.'))
            .unwrap();
        let index: usize = index.parse().unwrap();
        let proj = match w {
            "w1.weight" => "gate",
            "w3.weight" => "up",
            "w2.weight" => "down",
            other => panic!("{other}"),
        };
        if stacked {
            GgufName::Expert {
                stack: format!("blk.{layer}.ffn_{proj}_exps.weight"),
                index,
            }
        } else {
            GgufName::Plain(format!("blk.{layer}.ffn_{proj}.{index}.weight"))
        }
    }

    /// A Q8_0 Mixtral GGUF (general.architecture = llama, `llama.expert_count = 4`; the experts
    /// stacked or one tensor each) and the HF F32 store holding the same values un-permuted.
    fn gguf_case(stacked: bool) -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = mixtral_fixture();
        let q8 = q8_0_store(&fx.store, is_packed);
        let (index, store) = if stacked {
            gguf_checkpoint(&gguf_kvs(Some(EXPERTS)), &q8, |k| gguf_name_of(k, true))
        } else {
            let tensors: Vec<_> = q8
                .iter()
                .map(|(key, entry)| match gguf_name_of(key.as_str(), false) {
                    GgufName::Plain(name) => gguf_tensor(&name, entry),
                    GgufName::Expert { .. } => unreachable!(),
                })
                .collect();
            read_back_owned(&gguf_kvs(Some(EXPERTS)), &tensors)
        };
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            reference
                .insert(key.clone(), unpermuted(key.as_str(), entry))
                .unwrap();
        }
        (index, store, reference.build())
    }

    fn assert_equals_the_hf_body(
        index: &poot_load::gguf::GgufIndex,
        store: &WeightStore,
        reference_store: &WeightStore,
    ) {
        let registry = Registry::builtin().unwrap();
        let raw = RawConfig::Gguf(index);
        assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
        let model = registry.build(&raw, store).unwrap();
        assert!(
            has_router(&*model.model),
            "expert_count = 4 resolves with experts"
        );
        let reference = built(
            &ENTRY,
            &RawConfig::HfJson {
                config: &mixtral_config(),
                generation: None,
                preprocessor: None,
            },
            reference_store,
        );
        let prompt = [3, 17, 40, 8];
        let (got, state) = run_step(&*model.model, store, CAP, &prompt, 0, &[]);
        let (want, want_state) = run_step(&*reference, reference_store, CAP, &prompt, 0, &[]);
        close(&got, &want);
        let (got, _) = run_step(&*model.model, store, CAP, &[5], 4, &state);
        let (want, _) = run_step(&*reference, reference_store, CAP, &[5], 4, &want_state);
        close(&got, &want);
    }

    /// SC-007 and the held Mixtral rows: a Mixtral GGUF (architecture `llama`, `llama.expert_count`
    /// 4) resolves to the llama family with experts and computes what the HF body computes over the
    /// same values, with the experts stacked or in the older one-tensor-per-expert layout.
    /// Mutation: key the MoE choice on the HF `model_type` string (a GGUF has none); the GGUF
    /// resolves dense and `build` refuses the stacked expert tensors it does not read.
    #[test]
    fn a_mixtral_gguf_resolves_to_llama_with_experts_in_both_layouts() {
        for stacked in [true, false] {
            let (index, store, reference_store) = gguf_case(stacked);
            assert_equals_the_hf_body(&index, &store, &reference_store);
        }
    }

    /// Without `llama.expert_count` (or with a zero one) a llama GGUF is the dense llama.
    #[test]
    fn a_llama_gguf_without_an_expert_count_is_dense() {
        let fx = fixture();
        let mut tensors = Vec::new();
        for (key, entry) in fx.store.iter() {
            tensors.push(gguf_tensor(&gguf_name_with_head(key.as_str()), entry));
        }
        for experts in [None, Some(0)] {
            let kvs = {
                let mut kvs = gguf_dense_kvs("llama", &fx.config);
                if let Some(n) = experts {
                    kvs.push(("llama.expert_count".into(), GgufValue::U32(n)));
                }
                kvs
            };
            let (index, store) = read_back_owned(&kvs, &tensors);
            let registry = Registry::builtin().unwrap();
            let model = registry.build(&RawConfig::Gguf(&index), &store).unwrap();
            assert!(!has_router(&*model.model), "expert_count {experts:?}");
            assert_eq!(model.model.config().family, FAMILY);
        }
    }

    /// A Mixtral GGUF missing its router names `blk.L.ffn_gate_inp.weight`.
    #[test]
    fn a_mixtral_gguf_without_a_router_is_named() {
        let (index, store, _) = gguf_case(true);
        let mut without = WeightStore::builder();
        for (key, entry) in store.iter() {
            if key.as_str() != "blk.0.ffn_gate_inp.weight" {
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
        assert_eq!(key, &WeightKey::from("blk.0.ffn_gate_inp.weight"));
    }
}
