//! OLMoE behind [`Model`]: a standard pre-norm block with olmo2's RMSNorm over the whole Q and K
//! projections before RoPE (`QkNorm::Projection`), whose feed-forward routes every layer through a
//! mixture of experts. The router is a softmax over all experts, then the `top_k` probabilities,
//! divided by their sum only when `norm_topk_prob` is set (HF's default is not; a GGUF has no key for
//! it and llama.cpp never renormalizes OLMoE).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{ExpertProj, WeightMap, WeightStore};

use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
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
use poot_text::ChatFormat;

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("olmoe");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "olmoe"),
        (ConfigSource::GgufArchitecture, "olmoe"),
    ],
    build,
    fixture,
    variants: &[],
};

/// OLMoE's config facts: `OlmoeConfig`'s defaults (`rms_norm_eps` 1e-5, `rope_theta` 1e4; the class
/// declares no `eos_token_id`) and llama.cpp's GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("olmoe"),
    hf_theta: 10_000.0,
    gguf_theta: Some(10_000.0),
    eps: 1e-5,
    eos: None,
};

const EXPERTS: Field = Field::new("num_experts", "olmoe.expert_count");
const TOP_K: Field = Field::new("num_experts_per_tok", "olmoe.expert_used_count");

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

/// OLMoE's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OlmoeConfig {
    vocab: usize,
    width: usize,
    layers: usize,
    max_positions: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    moe: MoeParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl OlmoeConfig {
    pub(crate) fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        // `intermediate_size` is the per-expert width: every layer routes.
        let dims = DenseDims::read(&fields, &SPEC)?;
        if fields.flag(Field::hf("attention_bias"), false)? {
            return Err(fields.error("attention_bias", ConfigReason::Unsupported));
        }
        if fields.opt_f32(Field::hf("clip_qkv"))?.is_some() {
            return Err(fields.error("clip_qkv", ConfigReason::Unsupported));
        }
        let norm = dims.norm(&fields, &SPEC)?;
        let renormalize = fields.flag(Field::hf("norm_topk_prob"), false)?;
        let moe = MoeParams::new(
            FAMILY,
            dims.width,
            dims.inter,
            fields.count(EXPERTS, None)?,
            fields.count(TOP_K, None)?,
            RouterKind::SoftmaxTopK { renormalize },
        )?;
        Ok(Self {
            attention: dims
                .attention(&fields, &SPEC, false)?
                .with_qk_norm(QkNorm::Projection(norm)),
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            layers: dims.layers,
            max_positions: dims.max_positions,
            moe,
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
    moe: MoeWeights,
}

#[derive(Debug)]
pub(crate) struct Olmoe {
    params: OlmoeConfig,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = OlmoeConfig::from_raw(raw)?;
    let (tables, names): ([&[_]; 2], &MoeNames) = match raw.source() {
        ConfigSource::HfModelType => ([HF_BASE, HF_QK_NORM], &HF_MOE),
        ConfigSource::GgufArchitecture => ([GGUF_BASE, GGUF_QK_NORM], &GGUF_MOE),
    };
    let sparse = vec![true; p.layers];
    let weights = family_weights_routed(
        FAMILY,
        store,
        &tables,
        p.layers,
        p.head_required,
        MoeLayers {
            names,
            sparse: &sparse,
            experts: p.moe.experts(),
        },
    )?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention)?,
                moe: MoeWeights::new(&weights, l, &p.moe)?,
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
    Ok(Box::new(Olmoe {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Olmoe {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, phase: Phase, shape: StepShape, b: Builder) -> Result<Graph, TraceError> {
        let p = &self.params;
        check_step(FAMILY, phase, shape, p.max_positions)?;
        moe::check_step(FAMILY, shape, &p.moe)?;
        let step = Step::new(&b, shape)?;
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention),
                |b, BlockInput { block: h, .. }| moe::moe(b, &step, l, h, None, &layer.moe, &p.moe),
            )
        })?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny untied OLMoE (plain MHA, no renormalization) whose projection
/// widths are multiples of 32 (so a Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "olmoe",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": false,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let n = |f: &str| crate::registry::Fixture::count(&config, f);
    let (h, experts, inter) = (n("hidden_size"), n("num_experts"), n("intermediate_size"));
    let heads = n("num_attention_heads");
    let (q, kv) = (h, h / heads * n("num_key_value_heads"));
    let mut shapes: Vec<_> = hf_base_shapes(&config, false)
        .into_iter()
        .filter(|(name, _)| !name.contains(".mlp."))
        .collect();
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        shapes.push((k("self_attn.q_norm.weight"), vec![q]));
        shapes.push((k("self_attn.k_norm.weight"), vec![kv]));
        shapes.push((k("mlp.gate.weight"), vec![experts, h]));
        for e in 0..experts {
            for (proj, shape) in [
                ("gate_proj", [inter, h]),
                ("up_proj", [inter, h]),
                ("down_proj", [h, inter]),
            ] {
                shapes.push((k(&format!("mlp.experts.{e}.{proj}.weight")), shape.to_vec()));
            }
        }
    }
    Fixture::bf16(config, shapes)
}

/// Test support the routed families share: the packed-experts row and the GGUF writer for stacked
/// experts.
#[cfg(test)]
pub(crate) mod moe_support {
    use std::collections::BTreeMap;

    use poot_graph_ir::OpKind;
    use poot_graph_plan::{WeightFormats, bind_packed_weights};
    use poot_load::gguf::{GgufIndex, GgufValue};
    use poot_quant::weights::{HandleFormat, WeightEntry, WeightStore};

    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, build, close, close_all, declared_dtype, decoded, gguf_tensor, q8_0_store,
        read_back_owned, step_shape,
    };
    use crate::model::{LogitRows, Phase};
    use crate::registry::{FamilyEntry, Fixture};

    /// SC-004's HF half for a routed family: the BF16 `fixture` store and its Q8_0 copy (each tensor
    /// `is_packed` names packed, the per-expert projections included) trace through the one body.
    /// The BF16 graph declares every expert stack at its stored BF16; the Q8_0 graph, after the
    /// packed-weight transform fed by the model's map, holds one `PackedDequant` per packed weight
    /// and no packed const. Each equals (tier 2) the same body over its store decoded to F32, in
    /// prefill and a decode step.
    pub(crate) fn assert_bf16_and_q8_0_experts_trace_and_match(
        entry: &FamilyEntry,
        fixture: &Fixture,
        is_packed: impl Fn(&str) -> bool,
    ) {
        let raw = fixture.raw();
        let prompt = [3, 17, 40, 8];
        for (label, store) in [
            ("bf16", fixture.store.clone()),
            ("q8_0", q8_0_store(&fixture.store, &is_packed)),
        ] {
            let model = build(entry, &raw, &store);
            let g = model
                .trace(
                    Phase::Prefill,
                    step_shape(prompt.len(), CAP, LogitRows::All),
                    poot_graph_ir::TraceLimits::for_tests(),
                )
                .unwrap();
            let packed = model
                .weights()
                .iter()
                .filter(|(_, _, h)| matches!(h.format, HandleFormat::Packed(_)))
                .count();
            let bound =
                bind_packed_weights(&g, &WeightFormats::from_weight_map(model.weights())).unwrap();
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
            if label == "q8_0" {
                assert!(packed > 0, "{label}: nothing packed");
                let stacks = model
                    .weights()
                    .iter()
                    .filter(|(_, _, h)| {
                        matches!(h.format, HandleFormat::Packed(_)) && h.shape.len() == 3
                    })
                    .count();
                assert_eq!(
                    (dequants + stacks, indexed),
                    (packed, stacks),
                    "{label}: one decode per packed projection, one indexed contraction per packed stack"
                );
            } else {
                assert_eq!((packed, dequants), (0, 0), "{label}");
                let stacks: Vec<_> = model
                    .weights()
                    .iter()
                    .filter(|(id, _, _)| id.const_name().contains(".moe.stack."))
                    .collect();
                assert!(!stacks.is_empty(), "{label}: no expert stack");
                for (id, _, _) in stacks {
                    assert_eq!(
                        declared_dtype(&g, &id.const_name()),
                        Some(poot_tensor::DType::BF16),
                        "{label}: {id}"
                    );
                }
            }

            let reference_store = decoded(&store);
            let reference = build(entry, &raw, &reference_store);
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

    /// Where a checkpoint tensor goes in a GGUF.
    pub(crate) enum GgufName {
        /// One tensor of its own.
        Plain(String),
        /// Expert `index` of the stacked tensor `stack`.
        Expert { stack: String, index: usize },
    }

    /// A GGUF of `kvs` holding every tensor of `store` as `name_of` places it, read back through the
    /// real reader. The experts of a stack are written as the one rank-3 tensor llama.cpp writes
    /// (`[K, out, experts]`), in expert order.
    pub(crate) fn gguf_checkpoint(
        kvs: &[(String, GgufValue)],
        store: &WeightStore,
        name_of: impl Fn(&str) -> GgufName,
    ) -> (GgufIndex, WeightStore) {
        let mut tensors = Vec::new();
        let mut stacks: BTreeMap<String, Vec<(usize, &WeightEntry)>> = BTreeMap::new();
        for (key, entry) in store.iter() {
            match name_of(key.as_str()) {
                GgufName::Plain(name) => tensors.push(gguf_tensor(&name, entry)),
                GgufName::Expert { stack, index } => {
                    stacks.entry(stack).or_default().push((index, entry));
                }
            }
        }
        for (name, mut experts) in stacks {
            experts.sort_by_key(|&(index, _)| index);
            let count = experts.len() as u64;
            let (_, mut dims, ty, mut bytes) = gguf_tensor(&name, experts[0].1);
            for &(_, entry) in &experts[1..] {
                bytes.extend(gguf_tensor(&name, entry).3);
            }
            dims.push(count);
            tensors.push((name, dims, ty, bytes));
        }
        read_back_owned(kvs, &tensors)
    }
}

#[cfg(test)]
mod family_tests {
    use poot_graph_ir::OpKind;
    use poot_graph_plan::{
        CompileLimits, CompileOptions, Plan, Target, compile, production_plan::Submission,
    };
    use poot_load::gguf::GgufValue;
    use poot_quant::weights::{MoeRole, WeightRole};
    use poot_target::{AmdArch, Backend, DeviceCaps};
    use serde_json::json;

    use super::moe_support::{
        GgufName, assert_bf16_and_q8_0_experts_trace_and_match, gguf_checkpoint,
    };
    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build as built,
        close, decoded, q8_0_store, step_shape,
    };
    use crate::model::{KvLayout, LogitRows};
    use crate::registry::Registry;
    use poot_quant::weights::WeightKey;

    fn is_packed(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// Chunked prefill continuing from `Pos` equals token-by-token decode (the routed body is one
    /// body for both phases).
    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// SC-004 (HF half): BF16 and Q8_0 stores, experts included, trace through the one body.
    #[test]
    fn bf16_and_q8_0_experts_trace_through_the_one_body_and_match_the_decoded_reference() {
        assert_bf16_and_q8_0_experts_trace_and_match(&ENTRY, &fixture(), is_packed);
    }

    /// SC-005: a store missing one required tensor, a router and a single expert's projection
    /// included, fails `Registry::build` with the typed missing-weight error naming it, before any
    /// trace. Mutation: skip the row check in `map_moe_layer`; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.mlp.gate.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.experts.7.down_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "lm_head.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
        }
    }

    /// `norm_topk_prob` is a real parameter of the router: the same weights give different logits
    /// with and without it (OLMoE's released checkpoint does not renormalize).
    #[test]
    fn norm_topk_prob_changes_the_logits() {
        let fx = fixture();
        let run = |norm: bool| {
            let mut config = fx.config.clone();
            config["norm_topk_prob"] = json!(norm);
            let model = model_of(&config, &fx.store);
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
        assert_ne!(run(true), run(false));
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
    fn a_config_olmoe_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "clip_qkv",
                json!(8.0),
                "clip_qkv",
                ConfigReason::Unsupported,
            ),
            (
                "num_experts_per_tok",
                json!(9),
                "top_k",
                ConfigReason::Exceeds { max: 8 },
            ),
            (
                "num_experts_per_tok",
                json!(0),
                "num_experts_per_tok",
                ConfigReason::Zero,
            ),
            ("num_experts", json!(0), "num_experts", ConfigReason::Zero),
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

    /// SC-008: the decode plan of the OLMoE fixture, whose experts are stored `[E, N, K]` and read
    /// through `transpose(w, [0, 2, 1])`, launches no `transpose` kernel (the indexed contraction
    /// reads the stored orientation in place) on every backend. Mutation: drop the indexed arm of
    /// `is_strided_capable` (`predicates.rs`); a per-step `transpose` dispatch appears and this row
    /// goes red.
    #[test]
    fn expert_transposes_are_views_the_indexed_contraction_reads() {
        // The stored BF16 stack, and its F32 decode (whose mutated plan has a launched transpose
        // where the BF16 one is refused outright).
        let fx = fixture();
        for store in [decoded(&fx.store), fx.store.clone()] {
            assert_no_expert_transpose(&*model_of(&fx.config, &store));
        }
    }

    fn assert_no_expert_transpose(model: &dyn Model) {
        let shape = StepShape {
            logits: LogitRows::Last,
            kv: KvLayout::Contiguous,
            ..step_shape(1, CAP, LogitRows::Last)
        };
        let graph = model
            .trace(
                Phase::Decode,
                shape,
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let graph = poot_graph_plan::bind_packed_weights(
            &graph,
            &poot_graph_plan::WeightFormats::from_weight_map(model.weights()),
        )
        .unwrap();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let target = Target::Gpu {
                backend,
                caps: match backend {
                    Backend::SpirvVulkan => DeviceCaps::wgpu_rdna3_igpu(),
                    Backend::Nvptx => DeviceCaps::ptx_default(),
                    Backend::AmdGcn(_) => DeviceCaps::rocm_default(),
                },
            };
            let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
            let program = compile(&graph, &target, &options).unwrap();
            let launched: Vec<String> = program
                .planned()
                .filter(|(eqn, plan)| {
                    matches!(eqn.op, OpKind::Transpose { .. })
                        && matches!(plan, Plan::Dispatches(_))
                })
                .filter_map(|(eqn, _)| match eqn.inputs.first() {
                    Some(poot_graph_ir::Operand::Value(v)) => program.graph().meta(*v).name.clone(),
                    _ => None,
                })
                .collect();
            assert_eq!(
                launched
                    .iter()
                    .filter(|n| n.contains(".moe.stack."))
                    .count(),
                0,
                "{backend:?}: an expert transpose kernel runs: {launched:?}"
            );
        }
    }

    fn gguf_name_of(hf: &str) -> GgufName {
        let plain = |s: &str| GgufName::Plain(s.to_string());
        match hf {
            "model.embed_tokens.weight" => return plain("token_embd.weight"),
            "model.norm.weight" => return plain("output_norm.weight"),
            "lm_head.weight" => return plain("output.weight"),
            _ => {}
        }
        let rest = hf.strip_prefix("model.layers.").unwrap();
        let (layer, tensor) = rest.split_once('.').unwrap();
        if let Some(expert) = tensor.strip_prefix("mlp.experts.") {
            let (index, proj) = expert.split_once('.').unwrap();
            let stack = match proj {
                "gate_proj.weight" => "ffn_gate_exps.weight",
                "up_proj.weight" => "ffn_up_exps.weight",
                "down_proj.weight" => "ffn_down_exps.weight",
                other => panic!("{other}"),
            };
            return GgufName::Expert {
                stack: format!("blk.{layer}.{stack}"),
                index: index.parse().unwrap(),
            };
        }
        let name = match tensor {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "mlp.gate.weight" => "ffn_gate_inp.weight",
            other => panic!("unmapped fixture tensor {other}"),
        };
        plain(&format!("blk.{layer}.{name}"))
    }

    fn gguf_kvs() -> Vec<(String, GgufValue)> {
        let mut kvs = crate::components::testing::gguf_dense_kvs("olmoe", &fixture_config());
        kvs.push(("olmoe.expert_count".into(), GgufValue::U32(8)));
        kvs.push(("olmoe.expert_used_count".into(), GgufValue::U32(2)));
        kvs
    }

    /// A Q8_0 OLMoE GGUF (projections and experts Q8_0, router, norms and the rest F32): the
    /// experts are one rank-3 tensor per projection, which the reader splits into one packed owner
    /// per expert.
    fn gguf_case(q8: bool) -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let store = if q8 {
            q8_0_store(&fx.store, is_packed)
        } else {
            fx.store.clone()
        };
        let (index, gguf_store) = gguf_checkpoint(&gguf_kvs(), &store, gguf_name_of);
        (index, gguf_store, decoded(&store))
    }

    /// SC-001/SC-004 (GGUF half): an OLMoE GGUF resolves by its architecture key, builds packed
    /// (Q8_0) or stacked dense (F32) experts and computes what the HF body computes over the same
    /// values, prefill and decode. Mutation: drop the GGUF key from `ENTRY.keys`; `Unregistered`.
    #[test]
    fn a_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        for q8 in [true, false] {
            let (index, store, reference_store) = gguf_case(q8);
            let registry = Registry::builtin().unwrap();
            let raw = RawConfig::Gguf(&index);
            assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
            let model = registry.build(&raw, &store).unwrap();
            let stacks: Vec<_> = model
                .weights()
                .iter()
                .filter(|(id, _, handle)| {
                    matches!(id.role, WeightRole::Moe(MoeRole::Stack(..)))
                        && matches!(handle.format, poot_quant::weights::HandleFormat::Packed(_))
                })
                .collect();
            assert_eq!(stacks.is_empty(), !q8, "q8={q8}: packed expert stacks");
            if q8 {
                assert_eq!(stacks.len(), 2 * 3);
                assert!(stacks.iter().all(|(_, _, handle)| handle.shape[0] == 8));
            }
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
    }

    /// A store whose per-expert tensors are missing is a typed missing-weight naming the first
    /// absent expert, not a short stack.
    #[test]
    fn a_missing_gguf_router_is_named() {
        let (index, store, _) = gguf_case(false);
        let mut without = WeightStore::builder();
        for (key, entry) in store.iter() {
            if key.as_str() != "blk.1.ffn_gate_inp.weight" {
                without.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        let err = Registry::builtin()
            .unwrap()
            .build(&RawConfig::Gguf(&index), &without.build())
            .map(|_| ())
            .unwrap_err();
        let crate::registry::LoadModelError::Model(ModelError::Weight {
            source: poot_quant::weights::WeightMapError::Missing { key, .. },
            ..
        }) = &err
        else {
            panic!("{err}");
        };
        assert_eq!(key, &WeightKey::from("blk.1.ffn_gate_inp.weight"));
    }
}
