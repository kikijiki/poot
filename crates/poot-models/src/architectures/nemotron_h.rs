//! Nemotron-H (NVIDIA's hybrid Mamba2 / attention family) behind [`Model`].
//!
//! Each layer is exactly one of three mixers, not a mixer and a feed-forward pair:
//! `x += mixer(rmsnorm(x))`. `hybrid_override_pattern` gives one character per layer: `M` a Mamba2
//! mixer ([`mamba2`]), `*` grouped-query attention with no positional embedding, `-` a plain MLP
//! with the squared-ReLU activation (`relu2`, no gate). The pattern is checkpoint input: a
//! character it does not know, or a length that is not the layer count, is a typed config error.
//!
//! A step carries the attention layers' KV caches and, for each Mamba2 layer, its conv window and
//! SSM state, so a prompt prefilled in chunks continues exactly where the previous chunk stopped
//! and one token is a decode step.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{BinOp, Builder, Graph};
use poot_quant::weights::{
    AttnRole, FfnRole, Mamba2Role, NormRole, WeightId, WeightKey, WeightMap, WeightRole,
    WeightStore, WeightView,
};

use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{Field, Fields};
use crate::components::ffn::{PlainMlpWeights, relu2_mlp};
use crate::components::linear::{Weight, WeightError};
use crate::components::mamba2::{Mamba2Params, Mamba2Weights, mamba2};
use crate::components::norm::{NormParams, norm};
use crate::components::standard::{
    StackParams, StackWeights, Step, decoder_program, standard_stack,
};
use crate::io::{ModelSchema, StateSpec};
use crate::model::{ConfigReason, FamilyKey, Model, ModelConfig, ModelError, StepSpec, TraceError};
use crate::names::TensorRows;
#[cfg(feature = "test-support")]
use crate::registry::Fixture;
use crate::registry::{ConfigSource, FamilyEntry, ModelParts, RawConfig};
use poot_text::{ChatFormat, HermesToolCodec};

/// Nemotron-H's [`FamilyKey`].
pub const FAMILY: FamilyKey = FamilyKey::new("nemotron_h");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "nemotron_h")],
    build,
};

/// One layer's mixer, decoded from a `hybrid_override_pattern` character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// `'M'`: a Mamba2 SSM.
    Mamba,
    /// `'*'`: grouped-query softmax attention with no positional embedding.
    Attention,
    /// `'-'`: a plain squared-ReLU MLP.
    Mlp,
}

/// The layer kinds `pattern` names, one per layer. A character other than `M`, `*` or `-`, or a
/// pattern of another length than `layers`, is refused.
fn parse_pattern(
    fields: &Fields<'_>,
    pattern: &str,
    layers: usize,
) -> Result<Vec<LayerKind>, ModelError> {
    const KEY: &str = "hybrid_override_pattern";
    let kinds = pattern
        .chars()
        .map(|c| match c {
            'M' => Ok(LayerKind::Mamba),
            '*' => Ok(LayerKind::Attention),
            '-' => Ok(LayerKind::Mlp),
            _ => Err(fields.error(KEY, ConfigReason::Unsupported)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if kinds.len() != layers {
        return Err(fields.error(KEY, ConfigReason::Length { expected: layers }));
    }
    Ok(kinds)
}

/// Nemotron-H's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct NemotronHConfig {
    vocab: usize,
    width: usize,
    mlp_inter: usize,
    max_positions: usize,
    kinds: Vec<LayerKind>,
    norm: NormParams,
    mamba: Mamba2Params,
    attention: AttentionParams,
    stack: StackParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl NemotronHConfig {
    /// Reads the typed config from an HF `config.json`, expanding `hybrid_override_pattern` into one
    /// layer kind per layer; an unknown pattern character or a length that is not the layer count is a
    /// typed [`ModelError::Config`].
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let count = |key: &'static str| fields.count(Field::hf(key), None);
        let width = count("hidden_size")?;
        let layers = count("num_hidden_layers")?;
        let RawConfig::HfJson { config, .. } = raw else {
            // llama.cpp has no Nemotron-H architecture key: only an HF checkpoint names the family.
            return Err(fields.error("general.architecture", ConfigReason::Unsupported));
        };
        let pattern = config
            .get("hybrid_override_pattern")
            .ok_or(fields.error("hybrid_override_pattern", ConfigReason::Missing))?
            .as_str()
            .ok_or(fields.error("hybrid_override_pattern", ConfigReason::WrongType))?;
        // `rms_norm_eps` and `layer_norm_epsilon` are one value (both 1e-5 on every released config).
        let eps = match fields.opt_f32(Field::hf("rms_norm_eps"))? {
            Some(eps) => eps,
            None => fields.float(Field::hf("layer_norm_epsilon"), 1e-5)?,
        };
        let norm = NormParams::new(eps).map_err(|e| e.for_family(FAMILY))?;
        let mamba = Mamba2Params::new(
            count("mamba_num_heads")?,
            count("mamba_head_dim")?,
            count("ssm_state_size")?,
            count("n_groups")?,
            count("conv_kernel")?,
            norm,
        )
        .map_err(|e| e.for_family(FAMILY))?;
        let attention = AttentionParams::nope(
            count("num_attention_heads")?,
            fields.count(Field::hf("num_key_value_heads"), None)?,
            count("attention_head_dim")?,
            false,
        )
        .map_err(|e| e.for_family(FAMILY))?;
        for unsupported in ["use_bias", "mamba_proj_bias", "attention_bias", "mlp_bias"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let head_required = !config
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(Self {
            vocab: fields.vocab()?,
            width,
            mlp_inter: count("intermediate_size")?,
            max_positions: count("max_position_embeddings")?,
            kinds: parse_pattern(&fields, pattern, layers)?,
            norm,
            mamba,
            attention,
            stack: StackParams::new(norm),
            head_required,
            eos: fields.eos(Some(2))?,
            bos: fields.bos()?.or(Some(1)),
        })
    }
}

/// The weight map of a checkpoint: the model-level weights and, per layer, the pre-mixer norm and
/// the mixer's weights its kind names. `in_proj` is one stored tensor whose three row ranges are
/// the Mamba2 gate, conv input and timestep projections.
fn weights_of(p: &NemotronHConfig, store: &WeightStore) -> Result<WeightMap, ModelError> {
    const EMBED: &str = "backbone.embeddings.weight";
    let mut rows = TensorRows::new(FAMILY, store);
    rows.tensor(WeightRole::Embed, None, EMBED.into())?;
    rows.tensor(WeightRole::FinalNorm, None, "backbone.norm_f.weight".into())?;
    rows.head(EMBED, "lm_head", p.head_required)?;

    let inner = p.mamba.inner();
    for (l, kind) in p.kinds.iter().enumerate() {
        let at = Some(l);
        let layer = format!("backbone.layers.{l}");
        rows.tensor(
            WeightRole::Norm(NormRole::Attn),
            at,
            format!("{layer}.norm.weight"),
        )?;
        let mixer = |name: &str| format!("{layer}.mixer.{name}");
        match kind {
            LayerKind::Mamba => {
                let z = WeightId::layer(l, WeightRole::Mamba2(Mamba2Role::Z));
                let in_proj = rows.linear_key(&mixer("in_proj"));
                if !store.contains(&in_proj) {
                    return Err(rows.missing(z, in_proj));
                }
                let conv = p.mamba.conv_channels();
                for (role, range) in [
                    (Mamba2Role::Z, 0..inner),
                    (Mamba2Role::Xbc, inner..inner + conv),
                    (Mamba2Role::Dt, inner + conv..inner + conv + p.mamba.heads()),
                ] {
                    rows.add(
                        WeightId::layer(l, WeightRole::Mamba2(role)),
                        WeightView::RowRange {
                            key: WeightKey::from(in_proj.as_str()),
                            rows: range,
                        },
                    )?;
                }
                for (role, name) in [
                    (Mamba2Role::DtBias, "dt_bias"),
                    (Mamba2Role::Conv, "conv1d.weight"),
                    (Mamba2Role::ConvBias, "conv1d.bias"),
                    (Mamba2Role::ALog, "A_log"),
                    (Mamba2Role::D, "D"),
                    (Mamba2Role::GateNorm, "norm.weight"),
                ] {
                    rows.tensor(WeightRole::Mamba2(role), at, mixer(name))?;
                }
                rows.linear(WeightRole::Mamba2(Mamba2Role::Out), at, &mixer("out_proj"))?;
            }
            LayerKind::Attention => {
                for (role, name) in [
                    (AttnRole::Q, "q_proj"),
                    (AttnRole::K, "k_proj"),
                    (AttnRole::V, "v_proj"),
                    (AttnRole::O, "o_proj"),
                ] {
                    rows.linear(WeightRole::Attn(role), at, &mixer(name))?;
                }
            }
            LayerKind::Mlp => {
                for (role, name) in [(FfnRole::Up, "up_proj"), (FfnRole::Down, "down_proj")] {
                    rows.linear(WeightRole::Ffn(role), at, &mixer(name))?;
                }
            }
        }
    }
    Ok(rows.finish())
}

#[derive(Debug)]
enum Mixer {
    Mamba(Box<Mamba2Weights>),
    Attention(Box<AttentionWeights>),
    Mlp(Box<PlainMlpWeights>),
}

#[derive(Debug)]
struct Layer {
    norm: Weight,
    mixer: Mixer,
}

/// The built Nemotron-H model: config, name-resolved weights, the stack-level tensors and one
/// Mamba2 / attention / MLP layer each. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct NemotronH {
    params: NemotronHConfig,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let p = NemotronHConfig::from_raw(raw)?;
    let weights = weights_of(&p, store)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = p
        .kinds
        .iter()
        .enumerate()
        .map(|(l, kind)| {
            let mixer = match kind {
                LayerKind::Mamba => Mixer::Mamba(Box::new(Mamba2Weights::new(
                    &weights, l, p.width, &p.mamba,
                )?)),
                LayerKind::Attention => Mixer::Attention(Box::new(AttentionWeights::new(
                    &weights,
                    l,
                    p.width,
                    &p.attention,
                )?)),
                LayerKind::Mlp => Mixer::Mlp(Box::new(PlainMlpWeights::new(
                    &weights,
                    l,
                    p.width,
                    p.mlp_inter,
                    false,
                )?)),
            };
            Ok(Layer {
                norm: Weight::new(
                    &weights,
                    WeightId::layer(l, WeightRole::Norm(NormRole::Attn)),
                    &[p.width],
                )?,
                mixer,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let mut step = decoder_program(p.vocab);
    if p.kinds.contains(&LayerKind::Mamba) {
        step = step.carrying(StateSpec::recurrent("mamba2"));
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
    Ok(ModelParts::new(Box::new(NemotronH {
        params: p,
        weights,
        stack,
        layers,
        config,
        schema,
    })))
}

impl Model for NemotronH {
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
                let h = norm(b, x, &layer.norm, p.norm)?;
                let y = match &layer.mixer {
                    Mixer::Mamba(w) => mamba2(b, &step, l, h, w, &p.mamba)?,
                    Mixer::Attention(w) => attention(b, &step, l, h, w, &p.attention)?,
                    Mixer::Mlp(w) => relu2_mlp(b, h, w)?,
                };
                Ok::<_, TraceError>(b.binary(BinOp::Add, x, y)?)
            },
        )?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a four-layer hybrid (`M*M-`) whose Mamba2 mixer has 4 heads of 8 over 2
/// groups, so widths are multiples of 32 where a Q8_0 copy needs them.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "M*M-",
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "ssm_state_size": 4,
        "n_groups": 2,
        "conv_kernel": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "attention_head_dim": 8,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
    })
}

/// The tensors of a Nemotron-H checkpoint of `config`, by HF name.
#[cfg(feature = "test-support")]
pub(crate) fn hf_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| crate::registry::Fixture::count(config, f);
    let (vocab, w, inter) = (n("vocab_size"), n("hidden_size"), n("intermediate_size"));
    let (heads, hd, state, groups, kernel) = (
        n("mamba_num_heads"),
        n("mamba_head_dim"),
        n("ssm_state_size"),
        n("n_groups"),
        n("conv_kernel"),
    );
    let inner = heads * hd;
    let conv = inner + 2 * groups * state;
    let (aq, akv, ad) = (
        n("num_attention_heads"),
        n("num_key_value_heads"),
        n("attention_head_dim"),
    );
    let mut shapes = vec![
        ("backbone.embeddings.weight".to_string(), vec![vocab, w]),
        ("backbone.norm_f.weight".to_string(), vec![w]),
        ("lm_head.weight".to_string(), vec![vocab, w]),
    ];
    for (l, c) in config["hybrid_override_pattern"]
        .as_str()
        .unwrap()
        .chars()
        .enumerate()
    {
        let k = |s: &str| format!("backbone.layers.{l}.{s}");
        shapes.push((k("norm.weight"), vec![w]));
        match c {
            'M' => shapes.extend([
                (k("mixer.in_proj.weight"), vec![inner + conv + heads, w]),
                (k("mixer.conv1d.weight"), vec![conv, 1, kernel]),
                (k("mixer.conv1d.bias"), vec![conv]),
                (k("mixer.dt_bias"), vec![heads]),
                (k("mixer.A_log"), vec![heads]),
                (k("mixer.D"), vec![heads]),
                (k("mixer.norm.weight"), vec![inner]),
                (k("mixer.out_proj.weight"), vec![w, inner]),
            ]),
            '*' => shapes.extend([
                (k("mixer.q_proj.weight"), vec![aq * ad, w]),
                (k("mixer.k_proj.weight"), vec![akv * ad, w]),
                (k("mixer.v_proj.weight"), vec![akv * ad, w]),
                (k("mixer.o_proj.weight"), vec![w, aq * ad]),
            ]),
            _ => shapes.extend([
                (k("mixer.up_proj.weight"), vec![inter, w]),
                (k("mixer.down_proj.weight"), vec![w, inter]),
            ]),
        }
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
    use poot_graph_ir::Slot;
    use poot_graph_plan::{WeightFormats, bind_packed_weights};
    use poot_quant::weights::HandleFormat;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::{eval, run_step};
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build, close,
        close_all, decoded, greedy_tokens, q8_0_store, step_shape,
    };
    use crate::model::{LogitRows, StepShape};
    use crate::registry::Registry;

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
            preprocessor: None,
        };
        build(&ENTRY, &raw, store)
    }

    /// SC-003 (F2): prefill of six tokens equals two prefills of three equals prefill of three then
    /// three decode steps equals six decode steps, at every logit and every carried state element
    /// (the attention caches, every Mamba2 layer's conv window and SSM state). Mutation: ignore the
    /// carried conv window in a continued chunk (today's zero pad); the second chunk goes red at
    /// the first Mamba2 layer.
    #[test]
    fn chunked_prefill_continuing_from_the_carried_state_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// The step carries the two caches of the attention layer and the conv window and SSM state of
    /// each Mamba2 layer, none for the MLP layer: the state of `M*M-` is six values.
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
                "mamba2.l0.conv",
                "mamba2.l0.ssm",
                "kv.l1.k",
                "kv.l1.v",
                "mamba2.l2.conv",
                "mamba2.l2.ssm"
            ]
        );
    }

    /// The `rows` of a step fold into the SSD op's head axis without mixing their states: two rows
    /// with different prompts give each row's logits and state of the same prompt traced alone.
    #[test]
    fn two_rows_do_not_share_state() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let prompts: [[i32; 3]; 2] = [[3, 17, 40], [8, 25, 11]];
        let single: Vec<_> = prompts
            .iter()
            .map(|p| run_step(&*model, &fx.store, CAP, p, 0, &[]))
            .collect();

        let shape = StepShape::new(
            std::num::NonZeroUsize::new(2).unwrap(),
            std::num::NonZeroUsize::new(3).unwrap(),
            std::num::NonZeroUsize::new(CAP).unwrap(),
        )
        .with_logits(LogitRows::All);
        let g = model
            .trace(
                &StepSpec::step(shape),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let g = bind_packed_weights(&g, &WeightFormats::from_weight_map(model.weights())).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).collect();
        let tokens: Vec<i32> = prompts.concat();
        let pos: Vec<i32> = vec![0, 1, 2, 0, 1, 2];
        let (logits, states) = eval(
            &g,
            &fx.store,
            model.weights(),
            &[],
            &[(Slot::Token, &tokens), (Slot::Pos, &pos)],
            &state,
        );
        let vocab = model.config().vocab;
        for (row, (want, want_state)) in single.iter().enumerate() {
            close(&logits[row * 3 * vocab..(row + 1) * 3 * vocab], want);
            for (i, state) in states.iter().enumerate() {
                let per_row = state.len() / 2;
                close(&state[row * per_row..(row + 1) * per_row], &want_state[i]);
            }
        }
    }

    /// SC-003 (tier 3): the fixture's 24 greedy tokens after the 7-token prompt equal the literals
    /// recorded from the pre-driver engine's CPU generate at the base commit (`64602bced`). Mutation: swap
    /// the carried conv window for zeros in a continued step (the decode steps); the tokens change.
    #[test]
    fn the_fixture_generates_the_tokens_the_runner_recorded() {
        const RECORDED: [u32; 24] = [
            28, 31, 1, 1, 1, 27, 2, 0, 36, 25, 22, 25, 28, 31, 1, 1, 1, 27, 32, 34, 31, 1, 25, 28,
        ];
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let got = greedy_tokens(&*model, &fx.store, &[3, 17, 40, 8, 25, 11, 2], 24, 32);
        assert_eq!(got, RECORDED);
    }

    /// SC-004 (F6): a store missing one required tensor, of any layer kind, fails `Registry::build`
    /// with the typed missing-weight error naming it, before any trace. Mutation: skip the row check
    /// in `weights_of`; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "backbone.layers.0.mixer.in_proj.weight",
            "backbone.layers.0.mixer.A_log",
            "backbone.layers.2.mixer.conv1d.bias",
            "backbone.layers.2.mixer.out_proj.weight",
            "backbone.layers.1.mixer.k_proj.weight",
            "backbone.layers.3.mixer.down_proj.weight",
            "backbone.layers.1.norm.weight",
            "backbone.norm_f.weight",
            "backbone.embeddings.weight",
            "lm_head.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
        }
    }

    /// The BF16 store and its Q8_0 copy trace through the one body and equal the same body over the
    /// store decoded to F32. The fused `in_proj` stays dense: a random Q8_0 payload has wild block
    /// scales, which overflow the timestep's softplus in both the packed and the decoded run.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body() {
        let fx = fixture();
        let packable = |key: &str| {
            key.ends_with("_proj.weight")
                && !key.contains("in_proj")
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

    fn config_error_of(config: &serde_json::Value) -> (&'static str, ConfigReason) {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
            preprocessor: None,
        };
        match build_model(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    fn build_model(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
        (ENTRY.build)(raw, store).map(|parts| parts.model)
    }

    /// A pattern character the family does not know and a pattern whose length is not the layer
    /// count are typed config errors, as is a config the family cannot trace.
    #[test]
    fn a_config_nemotron_h_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "hybrid_override_pattern",
                json!("M*X-"),
                "hybrid_override_pattern",
                ConfigReason::Unsupported,
            ),
            (
                "hybrid_override_pattern",
                json!("M*M"),
                "hybrid_override_pattern",
                ConfigReason::Length { expected: 4 },
            ),
            (
                "hybrid_override_pattern",
                json!(7),
                "hybrid_override_pattern",
                ConfigReason::WrongType,
            ),
            (
                "n_groups",
                json!(3),
                "mamba_num_heads",
                ConfigReason::NotDivisible { by: 3 },
            ),
            ("conv_kernel", json!(1), "conv_kernel", ConfigReason::Zero),
            (
                "use_bias",
                json!(true),
                "use_bias",
                ConfigReason::Unsupported,
            ),
            (
                "mamba_num_heads",
                json!(0),
                "mamba_num_heads",
                ConfigReason::Zero,
            ),
            (
                "num_key_value_heads",
                json!(3),
                "heads",
                ConfigReason::NotDivisible { by: 3 },
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
        config
            .as_object_mut()
            .unwrap()
            .remove("max_position_embeddings");
        assert_eq!(
            config_error_of(&config),
            ("max_position_embeddings", ConfigReason::Missing)
        );
    }

    /// A GGUF has no Nemotron-H architecture: its key is not registered, so the family refuses a
    /// GGUF config typed instead of guessing.
    #[test]
    fn the_family_is_keyed_by_the_hf_model_type_only() {
        assert_eq!(ENTRY.keys, &[(ConfigSource::HfModelType, "nemotron_h")]);
    }
}
