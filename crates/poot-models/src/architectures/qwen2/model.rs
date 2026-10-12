//! qwen2 behind [`Model`]: its typed config, its HF and GGUF name rows, and ONE body for prefill and
//! decode.
//!
//! A step of `tokens` new tokens continues from the absolute positions in `Slot::Pos`; decode is the
//! same body with `tokens = 1`. The body is `standard_stack` over `standard_layer` with the shared
//! attention (QKV bias, RoPE) and gated FFN components. Weights stay in checkpoint `[out, in]`
//! orientation at their stored format: a dense weight is declared at its stored dtype, a packed one
//! is rewritten by the one packed-weight transform from the `WeightMap`.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{WeightMap, WeightStore};

use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::linear::WeightError;
use crate::components::standard::{
    BlockInput, EmbeddablePhase, LayerNorms, LayerParams, StackParams, StackWeights, Step,
    check_embeddable_step, standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::GGUF_BASE;
use crate::names::{HF_BASE, family_weights, hf_base_shapes};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_text::ChatFormat;

/// qwen2's [`FamilyKey`].
pub const FAMILY: FamilyKey = FamilyKey::new("qwen2");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "qwen2"),
        (ConfigSource::GgufArchitecture, "qwen2"),
    ],
    build,
    fixture,
    variants: &[],
};

/// The Qwen2.5-VL text tower: the qwen2 body with sectioned RoPE over `Slot::MropePosition`.
pub const VL_FAMILY: FamilyKey = FamilyKey::new("qwen2_5_vl");

/// The text tower's entry: registered beside the vision tower that feeds it
/// ([`qwen_vl_vision`](crate::qwen_vl_vision)).
pub(crate) const VL_ENTRY: FamilyEntry = FamilyEntry {
    family: VL_FAMILY,
    keys: &[(ConfigSource::HfModelType, "qwen2_5_vl")],
    build: build_vl,
    fixture: vl_fixture,
    variants: &[],
};

/// qwen2's config facts: HF's `Qwen2Config` defaults, and llama.cpp's GGUF defaults (a `1e6` base
/// when the GGUF omits it). The class declares no `eos_token_id`: a checkpoint names it.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("qwen2"),
    hf_theta: 10_000.0,
    gguf_theta: 1_000_000.0,
    eps: 1e-6,
    eos: None,
};

/// qwen2's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2Params {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl Qwen2Params {
    /// Reads the typed config from an HF `config.json` or a GGUF index; sliding-window attention and
    /// other fields the family cannot serve are a typed [`ModelError::Config`].
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        Self::read(raw, FAMILY).map(|(params, _)| params)
    }

    /// The dimensions every qwen2-shaped config carries, errors reported for `family`.
    fn read(raw: &RawConfig<'_>, family: FamilyKey) -> Result<(Self, DenseDims), ModelError> {
        let spec = DenseSpec { family, ..SPEC };
        let fields = Fields::new(raw, family);
        let dims = DenseDims::read(&fields, &spec)?;
        if fields.flag(Field::hf("use_sliding_window"), false)? {
            return Err(fields.error("use_sliding_window", ConfigReason::Unsupported));
        }
        let norm = dims.norm(&fields, &spec)?;
        let params = Self {
            attention: dims.attention(&fields, &spec, true)?,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            head_required: dims.head_required,
            eos: dims.eos.clone(),
            bos: dims.bos,
        };
        Ok((params, dims))
    }

    /// The Qwen2.5-VL text tower's config: the qwen2 dimensions of the checkpoint's top level, under
    /// the conventions the sectioned body is written for. Anything else is refused, naming the
    /// field: another model type (Qwen2-VL included), a checkpoint without QKV bias, a head width
    /// other than `hidden_size / num_attention_heads`, a partial rotary, and a `mrope_section` that
    /// is not three widths covering the rotary pairs exactly. The tower is HF-only (a GGUF config is
    /// refused at `model_type`), and the old convention's `qk_norm=false` half has no analogue to
    /// check: the qwen2 body holds no q/k norm tensors.
    pub fn from_vl_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, VL_FAMILY);
        let RawConfig::HfJson { config, generation } = raw else {
            return Err(fields.error("model_type", ConfigReason::Unsupported));
        };
        if config.get("model_type").and_then(|v| v.as_str()) != Some("qwen2_5_vl") {
            return Err(fields.error("model_type", ConfigReason::Unsupported));
        }
        // The released config names its RoPE `mrope`: plain frequencies, whose sections are read below.
        let plain = plain_rope_type(config);
        let raw = &RawConfig::HfJson {
            config: &plain,
            generation: *generation,
        };
        if !fields.flag(Field::hf("attention_bias"), true)? {
            return Err(fields.error("attention_bias", ConfigReason::Unsupported));
        }
        let (mut params, dims) = Self::read(raw, VL_FAMILY)?;
        let natural = dims.width / dims.heads;
        if !dims.width.is_multiple_of(dims.heads) || dims.head_dim != natural {
            return Err(fields.error(
                "head_dim",
                ConfigReason::Mismatch {
                    expected: natural,
                    found: dims.head_dim,
                },
            ));
        }
        if let Some(factor) = fields.opt_f32(Field::hf("partial_rotary_factor"))?
            && factor != 1.0
        {
            return Err(fields.error(
                "partial_rotary_factor",
                ConfigReason::Mismatch {
                    expected: dims.head_dim,
                    found: (dims.head_dim as f32 * factor) as usize,
                },
            ));
        }
        let sections = mrope_section(&fields, raw)?;
        params.attention = params
            .attention
            .with_sectioned_rope(sections)
            .map_err(|e| e.for_family(VL_FAMILY))?;
        Ok(params)
    }
}

/// `config` with the RoPE type `mrope` of its `rope_scaling` or `rope_parameters` block spelled
/// `default`: the rotation's frequencies are plain, only the positions are sectioned.
fn plain_rope_type(config: &serde_json::Value) -> serde_json::Value {
    let mut config = config.clone();
    for block in ["rope_scaling", "rope_parameters"] {
        let Some(block) = config.get_mut(block).and_then(|b| b.as_object_mut()) else {
            continue;
        };
        for key in ["type", "rope_type"] {
            if block.get(key).and_then(|v| v.as_str()) == Some("mrope") {
                block.insert(key.to_string(), "default".into());
            }
        }
    }
    config
}

/// The three section widths of `rope_scaling.mrope_section` (or `rope_parameters.mrope_section`):
/// temporal, height and width, in rotary pairs.
fn mrope_section(fields: &Fields<'_>, raw: &RawConfig<'_>) -> Result<[usize; 3], ModelError> {
    const KEY: &str = "rope_scaling.mrope_section";
    let RawConfig::HfJson { config, .. } = raw else {
        return Err(fields.error(KEY, ConfigReason::Missing));
    };
    let section = ["rope_scaling", "rope_parameters"]
        .iter()
        .filter_map(|block| config.get(block)?.get("mrope_section"))
        .find(|v| !v.is_null())
        .ok_or(fields.error(KEY, ConfigReason::Missing))?;
    let widths = section
        .as_array()
        .ok_or(fields.error(KEY, ConfigReason::WrongType))?;
    if widths.len() != 3 {
        return Err(fields.error(KEY, ConfigReason::Length { expected: 3 }));
    }
    let mut sections = [0usize; 3];
    for (slot, width) in sections.iter_mut().zip(widths) {
        *slot = width
            .as_u64()
            .and_then(|w| usize::try_from(w).ok())
            .ok_or(fields.error(KEY, ConfigReason::WrongType))?;
    }
    Ok(sections)
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    ffn: GatedFfnWeights,
}

/// A built qwen2-shaped model: config, name-resolved weights, the stack-level tensors and the
/// per-layer weights. `family` is the key its errors carry, so a family that reuses this body
/// reports under its own name. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct Qwen2 {
    family: FamilyKey,
    params: Qwen2Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    build_from(FAMILY, Qwen2Params::from_raw(raw)?, raw, store)
}

fn build_vl(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    build_from(VL_FAMILY, Qwen2Params::from_vl_raw(raw)?, raw, store)
}

fn build_from(
    family: FamilyKey,
    p: Qwen2Params,
    raw: &RawConfig<'_>,
    store: &WeightStore,
) -> Result<Box<dyn Model>, ModelError> {
    let rows = match raw.source() {
        ConfigSource::HfModelType => HF_BASE,
        ConfigSource::GgufArchitecture => GGUF_BASE,
    };
    let weights = family_weights(family, store, &[rows], p.layers, p.head_required)?;
    let weight_error = |e: WeightError| e.for_family(family);
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
        family,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(Qwen2 {
        family,
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Qwen2 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, phase: Phase, shape: StepShape, b: Builder) -> Result<Graph, TraceError> {
        let p = &self.params;
        let phase = check_embeddable_step(self.family, phase, shape, p.max_positions)?;
        let step = match phase {
            EmbeddablePhase::Prefill | EmbeddablePhase::Decode => Step::new(&b, shape)?,
            EmbeddablePhase::PrefillEmbeds => Step::with_embeds(&b, shape, p.width)?,
            EmbeddablePhase::Encode => Step::new(&b, shape)?.returning_hidden(),
        };
        let step = if p.attention.is_sectioned() {
            step.with_mrope(&b)?
        } else {
            step
        };
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention),
                |b, BlockInput { block: h, .. }| gated_ffn(b, h, &layer.ffn),
            )
        })?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny untied qwen2 whose projection widths are multiples of 32 (so a
/// Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen2",
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

/// The Qwen2.5-VL text tower's fixture config: the qwen2 fixture with the released config's keys
/// (`model_type`, `rope_scaling` naming `mrope` and its sections over the 8 rotary pairs).
fn vl_fixture_config() -> serde_json::Value {
    let mut config = fixture_config();
    config["model_type"] = "qwen2_5_vl".into();
    config["rope_scaling"] = serde_json::json!({"type": "mrope", "mrope_section": [2, 3, 3]});
    config
}

fn vl_fixture() -> Fixture {
    let config = vl_fixture_config();
    let shapes = hf_base_shapes(&config, true);
    Fixture::bf16(config, shapes)
}

/// A tiny deterministic BF16 qwen2 checkpoint (`FamilyEntry::fixture`).
fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_base_shapes(&config, true);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{OpKind, Slot};
    use poot_graph_plan::{WeightFormats, bind_packed_weights};
    use poot_load::gguf::{GgufIndex, GgufValue, IdentityNames, read_gguf, write_gguf};
    use poot_load::plan::default_workers;
    use poot_quant::SourceRole;
    use poot_quant::StoredBytes;
    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{WeightEntry, WeightId, WeightRole};
    use serde_json::json;
    use std::sync::Arc;

    use super::*;
    use crate::components::standard::oracle::{self, run_step};
    /// The first four dimensions of the mean over `PROMPT`'s positions of the `Encode` hidden state on
    /// the fixture: the sentence vector before any normalization (tier 2).
    const RECORDED_MEAN: [f32; 4] = [-0.6346052, 0.3129701, 0.27594122, -0.24472162];

    use crate::components::testing::{
        CAP, Recorded, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_matches_recorded, build as build_ok, close,
        close_all, f32_entry, projection_consts, q8_0_store as q8_0_of, values,
    };
    use crate::model::{KvLayout, LogitRows, ShapeReason};
    use crate::registry::Registry;

    const VOCAB: usize = 48;

    fn model(raw: &RawConfig<'_>, store: &WeightStore) -> Box<dyn Model> {
        build_ok(&ENTRY, raw, store)
    }

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    /// The fixture with each projection replaced by a Q8_0 payload of its shape.
    fn q8_0_store(store: &WeightStore) -> WeightStore {
        q8_0_of(store, is_projection)
    }

    /// POOT-733 SC-007: the qwen2 fixture traced through `standard_stack` under finite limits that
    /// exactly fit it produces the graph and the logits the unrestricted trace does (every element of
    /// the output and of the carried state against the oracle's), and under `max_eqns = 2` the trace is
    /// refused at the third attempted equation with a typed budget cause: no graph, and no equation
    /// past the limit was appended. Mutation: create an unbudgeted nested `Builder` in `standard_stack`
    /// (`Builder::new(TraceLimits::for_tests())` for the embedding); the restrictive row below no
    /// longer refuses at the third equation.
    #[test]
    fn exact_finite_limits_preserve_the_trace_and_max_eqns_2_refuses_the_third_equation() {
        use poot_graph_ir::{Attempted, BudgetResource, TraceLimits};
        use std::num::NonZeroU64;

        use crate::components::standard::oracle::run_shape_under;
        use crate::components::testing::step_shape;

        let fx = fixture();
        let model = model(&fx.raw(), &fx.store);
        let shape = step_shape(3, CAP, LogitRows::All);
        let roomy = TraceLimits::for_tests();
        let graph = model.trace(Phase::Prefill, shape, roomy).unwrap();

        let exact = TraceLimits {
            max_tokens: NonZeroUsize::new(3).unwrap(),
            max_values: NonZeroUsize::new(graph.values().len()).unwrap(),
            max_eqns: NonZeroUsize::new(graph.eqns().len()).unwrap(),
            max_literal_bytes: NonZeroU64::new(1 << 20).unwrap(),
            max_owned_bytes: NonZeroU64::new(1 << 30).unwrap(),
        };
        let under_exact = model.trace(Phase::Prefill, shape, exact).unwrap();
        crate::components::testing::assert_same_graph(&under_exact, &graph);
        let tokens = [5, 9, 2];
        let want = run_shape_under(
            roomy,
            &*model,
            &fx.store,
            Phase::Prefill,
            shape,
            &tokens,
            0,
            &[],
        );
        let got = run_shape_under(
            exact,
            &*model,
            &fx.store,
            Phase::Prefill,
            shape,
            &tokens,
            0,
            &[],
        );
        assert_eq!(got.0.len(), want.0.len());
        for (i, (g, w)) in got.0.iter().zip(&want.0).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "logit {i}");
        }
        assert_eq!(got.1, want.1, "every carried-state element");

        let tight = TraceLimits {
            max_eqns: NonZeroUsize::new(2).unwrap(),
            ..roomy
        };
        let refused = match model.trace(Phase::Prefill, shape, tight) {
            Err(TraceError::Budget(budget)) => budget,
            other => panic!("expected the typed budget refusal, got {other:?}"),
        };
        assert_eq!(refused.resource, BudgetResource::Equations);
        assert_eq!((refused.limit, refused.used, refused.requested), (2, 2, 1));
        assert_eq!(
            refused.attempted,
            Attempted::Total(3),
            "the third attempted equation"
        );
    }

    /// SC-002: on the fixture, the new body gives the top-five logits recorded from the legacy
    /// `trace_prefill_kv` and `trace_decode_kv_masked` at the base commit `bbd5c3232` (last prefill
    /// position and three decode steps; token ids exact, logits tier 2). Mutation: change the
    /// RMSNorm epsilon in the new body; red.
    #[test]
    fn the_one_body_matches_the_recorded_legacy_logits() {
        let fx = fixture();
        let model = model(&fx.raw(), &fx.store);
        assert_matches_recorded(
            &*model,
            &fx.store,
            &Recorded {
                prefill: [
                    (41, 1.883508),
                    (10, 1.8618252),
                    (40, 1.0994023),
                    (33, 1.0523599),
                    (46, 0.99382526),
                ],
                decode: [
                    [
                        (30, 1.9173205),
                        (25, 1.7693973),
                        (47, 1.5112326),
                        (14, 1.2442797),
                        (44, 1.177754),
                    ],
                    [
                        (41, 2.661889),
                        (23, 1.4328815),
                        (17, 1.4130934),
                        (12, 1.3635551),
                        (36, 1.3032329),
                    ],
                    [
                        (33, 2.1099987),
                        (41, 1.439606),
                        (32, 1.3048033),
                        (10, 1.3015712),
                        (46, 0.98898333),
                    ],
                ],
            },
        );
    }

    /// SC-003: chunked prefill continuing from the absolute positions in `Pos` equals
    /// token-by-token decode. Mutation: start every chunk's positions at 0; red.
    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model(&fx.raw(), &fx.store), &fx.store);
    }

    /// The prompt every encode and embeds test traces.
    const PROMPT: [i32; 5] = [3, 17, 40, 8, 25];

    fn zero_state(g: &Graph) -> Vec<Vec<f32>> {
        g.state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect()
    }

    /// One `PrefillEmbeds` step of `PROMPT` on the oracle: the logits at every position, with
    /// `embeds` (`[tokens, width]`) over the positions `at` marks.
    fn prefill_embeds(
        model: &dyn Model,
        store: &WeightStore,
        tokens: &[i32],
        embeds: &[f32],
        at: &[i32],
    ) -> Vec<f32> {
        let shape = crate::components::testing::step_shape(tokens.len(), CAP, LogitRows::All);
        let g = model
            .trace(
                Phase::PrefillEmbeds,
                shape,
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let state = zero_state(&g);
        let state: Vec<&[f32]> = state.iter().map(Vec::as_slice).collect();
        let pos: Vec<i32> = (0..tokens.len() as i32).collect();
        oracle::eval_slots(
            &g,
            store,
            model.weights(),
            &[],
            &[(Slot::Token, tokens), (Slot::Pos, &pos)],
            (
                &[("activation.embeds_at", at)],
                &[("activation.embeds", embeds)],
            ),
            &state,
        )
        .0
    }

    /// SC-001 (model half): `PrefillEmbeds` scatters the caller's rows over the embedded tokens.
    /// Rows equal to the table rows of the tokens they replace change nothing; a row that is another
    /// token's embedding gives that token's logits; with nothing marked the rows are never read.
    /// Mutation: ignore the mask (take the rows at every position); the unmarked case goes red.
    #[test]
    fn prefill_embeds_replaces_only_the_marked_positions() {
        let fx = fixture();
        let model = model(&fx.raw(), &fx.store);
        let w = 64;
        let table = values(fx.store.get("model.embed_tokens.weight").unwrap());
        let row = |t: i32| &table[t as usize * w..(t as usize + 1) * w];
        let (plain, _) = run_step(&*model, &fx.store, Phase::Prefill, CAP, &PROMPT, 0, &[]);

        // Table rows over positions 1 and 3: the same logits as the tokens.
        let mut embeds = vec![0.0; PROMPT.len() * w];
        for i in [1, 3] {
            embeds[i * w..(i + 1) * w].copy_from_slice(row(PROMPT[i]));
        }
        let at = [0, 1, 0, 1, 0];
        close(
            &prefill_embeds(&*model, &fx.store, &PROMPT, &embeds, &at),
            &plain,
        );

        // Position 2 takes token 7's row: the logits of the prompt with token 7 there.
        let mut embeds = vec![0.0; PROMPT.len() * w];
        embeds[2 * w..3 * w].copy_from_slice(row(7));
        let swapped = [3, 17, 7, 8, 25];
        let (want, _) = run_step(&*model, &fx.store, Phase::Prefill, CAP, &swapped, 0, &[]);
        let got = prefill_embeds(&*model, &fx.store, &PROMPT, &embeds, &[0, 0, 1, 0, 0]);
        close(&got, &want);
        assert_ne!(got, plain, "the replaced row did not reach the logits");

        // Nothing marked: whatever the rows hold is never read.
        let junk = vec![7.5; PROMPT.len() * w];
        close(
            &prefill_embeds(&*model, &fx.store, &PROMPT, &junk, &[0; 5]),
            &plain,
        );
    }

    /// SC-004: the `Encode` phase returns the head's input, the final-norm hidden state of every
    /// position: its product with the head weight is the `Prefill` logits at every position (tier
    /// 2), and mean pooling it on the host gives the recorded sentence vector. Mutation: take the
    /// hidden state one layer early (skip the last layer); the head product no longer equals the
    /// logits and this goes red.
    #[test]
    fn encode_returns_the_head_input_and_its_host_mean_is_the_sentence_vector() {
        let fx = fixture();
        let model = model(&fx.raw(), &fx.store);
        let w = 64;
        let (hidden, state) = run_step(&*model, &fx.store, Phase::Encode, CAP, &PROMPT, 0, &[]);
        assert_eq!(hidden.len(), PROMPT.len() * w);
        let _ = state;
        let (logits, _) = run_step(&*model, &fx.store, Phase::Prefill, CAP, &PROMPT, 0, &[]);
        let head: Vec<f64> = values(fx.store.get("lm_head.weight").unwrap())
            .into_iter()
            .map(f64::from)
            .collect();
        let hidden64: Vec<f64> = hidden.iter().map(|&v| f64::from(v)).collect();
        let want = crate::architectures::bert::reference::dense(
            &hidden64,
            PROMPT.len(),
            &head,
            &vec![0.0; VOCAB],
        );
        oracle::assert_matches_f64(&logits, &want, 1e-4);

        let mean: Vec<f32> = (0..w)
            .map(|j| {
                (0..PROMPT.len()).map(|i| hidden[i * w + j]).sum::<f32>() / PROMPT.len() as f32
            })
            .collect();
        close(&mean[..4], &RECORDED_MEAN);
    }

    /// SC-004: BF16 and Q8_0 stores trace through the one body and the packed
    /// transform. Mutation: make the transform read every handle as dense; the Q8_0 row goes red.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body_and_the_packed_transform() {
        assert_bf16_and_q8_0_trace_through_the_packed_transform(&ENTRY, &fixture(), is_projection);
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

    /// SC-005: a malformed qwen2 config is a typed config error naming the field, never a panic
    /// (the legacy tracer divides by the zero head count and panics on it).
    #[test]
    fn a_malformed_config_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "num_attention_heads",
                json!(0),
                "num_attention_heads",
                ConfigReason::Zero,
            ),
            (
                "num_attention_heads",
                json!(3),
                "hidden_size",
                ConfigReason::NotDivisible { by: 3 },
            ),
            (
                "num_key_value_heads",
                json!(3),
                "num_attention_heads",
                ConfigReason::NotDivisible { by: 3 },
            ),
            (
                "num_attention_heads",
                json!(64),
                "rotary_dim",
                ConfigReason::NotDivisible { by: 2 },
            ),
            (
                "rms_norm_eps",
                json!(0.0),
                "rms_norm_eps",
                ConfigReason::NotFinitePositive,
            ),
            (
                "rope_theta",
                json!(-1.0),
                "rope_theta",
                ConfigReason::NotFinitePositive,
            ),
            (
                "hidden_size",
                json!("64"),
                "hidden_size",
                ConfigReason::WrongType,
            ),
            (
                "rope_scaling",
                json!({"type": "longrope", "short_factor": [1.0]}),
                "rope_scaling.rope_type",
                ConfigReason::Unsupported,
            ),
            (
                "use_sliding_window",
                json!(true),
                "use_sliding_window",
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
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("vocab_size");
        assert_eq!(
            config_error_of(&config),
            ("vocab_size", ConfigReason::Missing)
        );
    }

    /// A HF `rope_scaling` the moved table flavors cover is accepted, and an untied checkpoint
    /// without `lm_head.weight` is refused naming it.
    #[test]
    fn yarn_scaling_is_accepted_and_a_missing_untied_head_is_named() {
        let fx = fixture();
        let mut config = fixture_config();
        config["rope_scaling"] =
            json!({"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16});
        let raw = RawConfig::HfJson {
            config: &config,
            generation: None,
        };
        build(&raw, &fx.store).unwrap();

        let mut store = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            if key.as_str() != "lm_head.weight" {
                store.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        let err = build(&fx.raw(), &store.build()).unwrap_err();
        assert!(err.to_string().contains("lm_head.weight"), "{err}");
    }

    #[test]
    fn a_shape_the_body_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = model(&fx.raw(), &fx.store);
        let n = |v| NonZeroUsize::new(v).unwrap();
        let shape = |rows, tokens, capacity| StepShape {
            rows: n(rows),
            tokens: n(tokens),
            capacity: n(capacity),
            kv: KvLayout::Contiguous,
            logits: LogitRows::Last,
            hidden: None,
        };
        let cases = [
            (Phase::Decode, shape(1, 2, 8), ShapeReason::DecodeTokens),
            (
                Phase::Prefill,
                shape(1, 1, 65),
                ShapeReason::CapacityAboveMax { max: 64 },
            ),
            (
                Phase::Prefill,
                shape(1, 9, 8),
                ShapeReason::TokensAboveCapacity,
            ),
        ];
        for (phase, shape, want) in cases {
            match model.trace(phase, shape, poot_graph_ir::TraceLimits::for_tests()) {
                Err(TraceError::ShapeUnsupported { reason, .. }) => assert_eq!(reason, want),
                other => panic!("{phase:?} {shape:?}: {other:?}"),
            }
        }
    }

    use crate::components::testing::{GGML_F32, GGML_Q8_0};

    /// The GGUF name of a fixture tensor, by llama.cpp's qwen2 table; `None` for the head (a GGUF
    /// without `output.weight` ties it to the embedding).
    fn gguf_name(hf: &str) -> Option<String> {
        let renamed = match hf {
            "model.embed_tokens.weight" => "token_embd.weight".to_string(),
            "model.norm.weight" => "output_norm.weight".to_string(),
            "lm_head.weight" => return None,
            _ => {
                let rest = hf.strip_prefix("model.layers.").unwrap();
                let (layer, tensor) = rest.split_once('.').unwrap();
                let tensor = match tensor {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "self_attn.q_proj.bias" => "attn_q.bias",
                    "self_attn.k_proj.bias" => "attn_k.bias",
                    "self_attn.v_proj.bias" => "attn_v.bias",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    other => panic!("unmapped fixture tensor {other}"),
                };
                format!("blk.{layer}.{tensor}")
            }
        };
        Some(renamed)
    }

    /// A Q8_0 qwen2 GGUF (projections Q8_0, the embedding too when `q8_0_embedding`, everything
    /// else F32, head tied) written with the GGUF fixture writer, plus the same values as an HF F32
    /// store.
    fn gguf_fixture(q8_0_embedding: bool) -> (Vec<u8>, WeightStore) {
        let q8 = q8_0_store(&fixture().store);
        let embed = "model.embed_tokens.weight";
        let q8_embed = WeightEntry::Packed(Arc::new(
            poot_quant::test_support::packed::random_payload(WeightFormat::Q8_0, [VOCAB, 64], 99),
        ));
        let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            let Some(name) = gguf_name(key.as_str()) else {
                continue;
            };
            let entry = if q8_0_embedding && key.as_str() == embed {
                &q8_embed
            } else {
                entry
            };
            let shape = entry.shape();
            let dims = shape.iter().rev().map(|&d| d as u64).collect();
            let (ty, bytes) = match entry {
                WeightEntry::Packed(payload) => {
                    (GGML_Q8_0, payload.bytes(SourceRole::Blocks).to_vec())
                }
                WeightEntry::Dense(_) => (
                    GGML_F32,
                    values(entry).iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            };
            tensors.push((name, dims, ty, bytes));
            reference
                .insert(key.clone(), f32_entry(shape, &values(entry)))
                .unwrap();
        }
        let u = GgufValue::U32;
        let kvs = [
            ("general.architecture", GgufValue::Str("qwen2".into())),
            ("qwen2.embedding_length", u(64)),
            ("qwen2.feed_forward_length", u(96)),
            ("qwen2.block_count", u(2)),
            ("qwen2.attention.head_count", u(4)),
            ("qwen2.attention.head_count_kv", u(2)),
            (
                "qwen2.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("qwen2.context_length", u(64)),
            ("qwen2.rope.freq_base", GgufValue::F32(10_000.0)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(
                    (0..VOCAB)
                        .map(|t| GgufValue::Str(format!("t{t}")))
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.eos_token_id", u(47)),
        ];
        let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors
            .iter()
            .map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone()))
            .collect();
        (write_gguf(&kvs, &tensors), reference.build())
    }

    /// SC-006 (R-562-3): a Q8_0 qwen2 GGUF read back by `read_gguf` resolves through
    /// `ConfigSource::GgufArchitecture`, builds a model whose projection handles are packed and
    /// whose head is tied, and computes what the HF body computes over the same values. Mutation:
    /// map one GGUF row to the wrong tensor name; `build` fails naming it.
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_builds_packed_projections() {
        let (bytes, reference_store) = gguf_fixture(false);
        let file = StoredBytes::from(bytes);
        let index = GgufIndex::parse(&file).unwrap();
        let store = read_gguf(&index, &file, &IdentityNames, default_workers()).unwrap();
        let raw = RawConfig::Gguf(&index);
        let registry = Registry::builtin().unwrap();
        assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
        let model = registry
            .build(&raw, &store)
            .unwrap_or_else(|e| panic!("{e}"));
        let projections = projection_consts(&*model);
        assert_eq!(projections.len(), 14);
        for (id, _, handle) in model.weights().iter() {
            if projections.contains(&id.const_name()) {
                assert!(
                    matches!(handle.format, poot_quant::weights::HandleFormat::Packed(_)),
                    "{id} is {:?}",
                    handle.format
                );
            }
        }
        assert_tied(&*model);
        assert_eq!(model.config().eos, BTreeSet::from([47]));

        let reference = tied_reference(&reference_store);
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

    /// The head reads its own map entry, viewing `token_embd.weight`.
    fn assert_tied(model: &dyn Model) {
        let view = |role| model.weights().view(WeightId::model(role));
        assert_eq!(
            view(WeightRole::Head),
            Some(&poot_quant::weights::WeightView::Stored(
                "token_embd.weight".into()
            ))
        );
        assert_eq!(view(WeightRole::Head), view(WeightRole::Embed));
    }

    /// The HF body over `store` (decoded F32, no `lm_head`) with a tied head.
    fn tied_reference(store: &WeightStore) -> Box<dyn Model> {
        let mut config = fixture_config();
        config["tie_word_embeddings"] = json!(true);
        model(
            &RawConfig::HfJson {
                config: &config,
                generation: None,
            },
            store,
        )
    }

    /// The common qwen2.5 GGUF case: a Q8_0 `token_embd` and no `output.weight`. The tied head is
    /// its own `w.head` entry over the embedding's payload, so after the packed-weight transform and
    /// `compile`'s two claims the embedding is one `PackedRowGather` and the head one
    /// `PackedContraction` (beside the 14 projections), with no `PackedDequant` left to escape; the
    /// claimed graph equals the HF body over the decoded values. Mutation: make `from_weight_map`
    /// register only one of the two (skip the head); red.
    #[test]
    fn a_q8_0_embedding_with_a_tied_head_claims_a_row_gather_and_a_contraction() {
        use poot_graph_plan::{recognize_packed_contractions, recognize_packed_row_gathers};

        let (bytes, reference_store) = gguf_fixture(true);
        let file = StoredBytes::from(bytes);
        let index = GgufIndex::parse(&file).unwrap();
        let store = read_gguf(&index, &file, &IdentityNames, default_workers()).unwrap();
        let model = model(&RawConfig::Gguf(&index), &store);
        assert_tied(&*model);
        let embed = model
            .weights()
            .handle(WeightId::model(WeightRole::Embed))
            .unwrap();
        assert!(matches!(
            embed.format,
            poot_quant::weights::HandleFormat::Packed(_)
        ));

        let prompt = [3, 17, 40, 8];
        let shape = StepShape {
            rows: NonZeroUsize::MIN,
            tokens: NonZeroUsize::new(prompt.len()).unwrap(),
            capacity: NonZeroUsize::new(CAP).unwrap(),
            kv: KvLayout::Contiguous,
            logits: LogitRows::All,
            hidden: None,
        };
        let g = model
            .trace(
                Phase::Prefill,
                shape,
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let bound =
            bind_packed_weights(&g, &WeightFormats::from_weight_map(model.weights())).unwrap();
        let claimed =
            recognize_packed_row_gathers(&recognize_packed_contractions(&bound).unwrap()).unwrap();
        let count = |f: fn(&OpKind) -> bool| claimed.eqns().iter().filter(|e| f(&e.op)).count();
        assert_eq!(
            count(|op| matches!(op, OpKind::PackedRowGather { .. })),
            1,
            "the embedding"
        );
        assert_eq!(
            count(|op| matches!(op, OpKind::PackedContraction { .. })),
            15,
            "14 projections and the head"
        );
        assert_eq!(count(|op| matches!(op, OpKind::PackedDequant { .. })), 0);

        let zeros: Vec<Vec<f32>> = claimed
            .state()
            .map(|(si, _)| vec![0.0; claimed.aval(si).numel()])
            .collect();
        let zeros: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).collect();
        let (got, state) = oracle::eval(
            &claimed,
            &store,
            model.weights(),
            &[],
            &[(Slot::Token, &prompt), (Slot::Pos, &[0, 1, 2, 3])],
            &zeros,
        );
        let reference = tied_reference(&reference_store);
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

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = GgufIndex::parse(&StoredBytes::from(write_gguf(
            &[
                ("general.architecture", GgufValue::Str("qwen2".into())),
                ("qwen2.embedding_length", GgufValue::U32(64)),
                ("qwen2.feed_forward_length", GgufValue::U32(96)),
                ("qwen2.block_count", GgufValue::U32(2)),
                ("qwen2.attention.head_count", GgufValue::U32(0)),
                ("qwen2.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        )))
        .unwrap();
        let err = Qwen2Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "qwen2.attention.head_count",
                    reason: ConfigReason::Zero,
                    ..
                }
            ),
            "{err}"
        );
    }
}
