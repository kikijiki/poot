//! MPT decoder tracer for MosaicML's `mosaicml/mpt-7b` and relatives, the
//! second ALiBi architecture (spec 254) after BLOOM. Checked against `llmfoundry/models/mpt/configuration_mpt.py`,
//! `llmfoundry/models/layers/{attention,blocks,ffn}.py`, and HF `transformers/models/mpt/modeling_mpt.py`; several
//! details differ from BLOOM.
//!
//! **Fused QKV layout.** MPT's `Wqkv` output (`[3*hidden, hidden]`) is split by `MultiheadAttention.forward` via a
//! contiguous `qkv.split([d_model, kv_n_heads*head_dim, kv_n_heads*head_dim], dim=2)` (HF: `chunk(3, dim=2)`):
//! block-concatenated (`[all_q | all_k | all_v]`), not BLOOM's per-head-interleaved layout (`crate::bloom`). A
//! loader must not reuse BLOOM's `split_qkv_weight`/`split_qkv_bias`; a contiguous 3-way `row_slice` (the phi3
//! precedent, `crates/poot-llm/src/runner.rs`'s `is_phi3()`) is correct. This tracer declares three separate
//! `attn.{q,k,v}_proj.weight` constants, as BLOOM's does (splitting is the loader's job).
//!
//! **No learnable bias.** `MPTConfig.no_bias` defaults to `true`: every linear (`Wqkv`, `out_proj`, `up_proj`,
//! `down_proj`) has `bias=False` and the norm bias is disabled (HF `modeling_mpt.py`: `self.norm_1.bias = None`).
//! No `.bias` constant is declared: every `linear()` call passes `None` and every norm site uses
//! `poot_graph_ir::ops::layernorm_no_bias`.
//!
//! **No post-embedding norm.** Unlike BLOOM's `word_embeddings_layernorm`, `MPTModel.forward` feeds
//! `self.wte(input_ids)` straight into block 0.
//!
//! **MLP: exact GELU**, not the tanh approximation. `layers/ffn.py` uses `nn.GELU(approximate="none")`; BLOOM's
//! `bloom_gelu_forward` is the tanh form (`ops::activation::gelu`), so this tracer uses
//! `ops::activation::gelu_erf`. Otherwise the same non-gated `down(gelu_erf(up(x)))`, width
//! `expansion_ratio * hidden` (`4 * hidden`), names `ffn.up_proj`/`ffn.down_proj`.
//!
//! **QK clipping / scale not implemented.** `attn_config` has `clip_qkv` (default `None`) and `qk_ln` (default
//! `False`) (`attn_config_defaults`); every public `mosaicml/mpt-*` checkpoint uses the defaults, so neither is
//! implemented (a checkpoint enabling them needs a follow-on). `softmax_scale` defaults to `None` (the standard
//! `1/sqrt(head_dim)`).
//!
//! **ALiBi, no RoPE.** `attn_config.alibi` defaults to `true` and no public checkpoint uses another path, so this
//! tracer always applies ALiBi and declares no rotary table. It uses the same `alibi_decode_mask_row`/
//! `alibi_prefill_mask` primitives (`poot_llm::graphs`, spec 254) and widened mask shapes as BLOOM (a
//! driver-level binder concern).
//!
//! **Attention shape.** Default `attn_type` is `"multihead_attention"` (plain MHA, `kv_n_heads == n_heads`), so
//! `n_rep = 1`. `multiquery_attention`/grouped variants are not implemented (BLOOM has no GQA either).
//!
//! **Tied lm_head.** `MPTForCausalLM` ties `lm_head` to `wte` by default (`config.tie_word_embeddings`) on every
//! public checkpoint; reused transposed, as BLOOM/gemma4.
//!
//! Structural template: `crate::bloom`. Primitive composition only (AGENTS.md): every op is an existing
//! `poot_graph_ir::ops` composition (`layernorm_no_bias`/`linear`/`attention_masked`/`attention_prefill`).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::rope_table::RopeFlavor;
use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{AttnRole, FfnRole, NormRole, WeightMap, WeightRole, WeightStore};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{Field, Fields};
use crate::components::ffn::{Gelu, PlainMlpWeights, plain_mlp};
use crate::components::linear::WeightError;
use crate::components::norm::{NormKind, NormParams};
use crate::components::rope::RopeParams;
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step,
    standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::GGUF_MPT;
use crate::names::{FusedDims, FusedPart, NameRow, Presence, Source, family_weights_fused};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

// The `mpt` family behind [`Model`]: bias-free LayerNorm blocks, ALiBi attention over a fused
// contiguous `Wqkv`, and a plain exact-GELU MLP, in the standard stack and layer.

pub const FAMILY: FamilyKey = FamilyKey::new("mpt");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "mpt"),
        (ConfigSource::GgufArchitecture, "mpt"),
    ],
    build,
    fixture,
    variants: &[],
};

const WIDTH: Field = Field::new("d_model", "mpt.embedding_length");
const HEADS: Field = Field::new("n_heads", "mpt.attention.head_count");
const LAYERS: Field = Field::new("n_layers", "mpt.block_count");
const MAX_POSITIONS: Field = Field::new("max_seq_len", "mpt.context_length");
const EPS: Field = Field::new("layer_norm_epsilon", "mpt.attention.layer_norm_epsilon");
const FFN: Field = Field::new("", "mpt.feed_forward_length");
const EXPANSION_RATIO: Field = Field::hf("expansion_ratio");
const KV_HEADS: Field = Field::new("attn_config.kv_n_heads", "mpt.attention.head_count_kv");

/// MPT's rows in its HF names (`MPTForCausalLM`): a fused `Wqkv` (contiguous `q | k | v`), no
/// biases, and a head tied to the embedding.
const HF_ROWS: &[NameRow] = &[
    NameRow {
        role: WeightRole::Embed,
        source: Source::Tensor("transformer.wte.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNorm,
        source: Source::Tensor("transformer.norm_f.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("lm_head"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    row_qkv(AttnRole::Q, FusedPart::Q),
    row_qkv(AttnRole::K, FusedPart::K),
    row_qkv(AttnRole::V, FusedPart::V),
    row_linear(WeightRole::Attn(AttnRole::O), ".attn.out_proj"),
    row_norm(NormRole::Attn, ".norm_1.weight"),
    row_norm(NormRole::Ffn, ".norm_2.weight"),
    row_linear(WeightRole::Ffn(FfnRole::Up), ".ffn.up_proj"),
    row_linear(WeightRole::Ffn(FfnRole::Down), ".ffn.down_proj"),
];

const BLOCKS: &str = "transformer.blocks.";

const fn row_linear(role: WeightRole, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear {
            prefix: BLOCKS,
            stem,
        },
        presence: Presence::Required,
    }
}

/// Part `part` (`q`, `k` or `v`) of the fused `Wqkv`.
const fn row_qkv(role: AttnRole, part: FusedPart) -> NameRow {
    NameRow {
        role: WeightRole::Attn(role),
        source: Source::LayerFused {
            prefix: BLOCKS,
            stem: ".attn.Wqkv",
            part,
        },
        presence: Presence::Required,
    }
}

const fn row_norm(role: NormRole, suffix: &'static str) -> NameRow {
    NameRow {
        role: WeightRole::Norm(role),
        source: Source::LayerTensor {
            prefix: BLOCKS,
            suffix,
        },
        presence: Presence::Required,
    }
}

/// A GGUF tensor of a learned-position MPT, which this ALiBi family does not carry.
const GGUF_POSITIONS: &str = "position_embd.weight";

/// mpt's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct MptParams {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

/// An `attn_config` entry, when the HF config sets it to something other than `null`.
fn attn_config<'a>(raw: &'a RawConfig<'_>, key: &str) -> Option<&'a serde_json::Value> {
    match raw {
        RawConfig::HfJson { config, .. } => config
            .get("attn_config")
            .and_then(|a| a.get(key))
            .filter(|v| !v.is_null()),
        RawConfig::Gguf(_) => None,
    }
}

impl MptParams {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let unsupported = |field| fields.error(field, ConfigReason::Unsupported);
        // This family is the bias-free, ALiBi, plain-MHA MPT every public checkpoint is.
        if let RawConfig::HfJson { config, .. } = raw {
            if config.get("no_bias").and_then(|v| v.as_bool()) == Some(false) {
                return Err(unsupported("no_bias"));
            }
            if attn_config(raw, "alibi").and_then(|v| v.as_bool()) == Some(false) {
                return Err(unsupported("attn_config.alibi"));
            }
            for key in ["clip_qkv", "softmax_scale"] {
                if attn_config(raw, key).is_some() {
                    return Err(unsupported(match key {
                        "clip_qkv" => "attn_config.clip_qkv",
                        _ => "attn_config.softmax_scale",
                    }));
                }
            }
            if attn_config(raw, "qk_ln").and_then(|v| v.as_bool()) == Some(true) {
                return Err(unsupported("attn_config.qk_ln"));
            }
        }
        if let RawConfig::Gguf(gguf) = raw {
            let bias = gguf
                .get("mpt.attention.max_alibi_bias")
                .and_then(|v| v.as_f32());
            if bias == Some(0.0) {
                return Err(unsupported("mpt.attention.max_alibi_bias"));
            }
        }
        let vocab = fields.vocab()?;
        let width = fields.count(WIDTH, None)?;
        let heads = fields.count(HEADS, None)?;
        if let Some(kv) = match raw {
            RawConfig::HfJson { .. } => attn_config(raw, "kv_n_heads").and_then(|v| v.as_u64()),
            RawConfig::Gguf(_) => fields.opt_u64(KV_HEADS)?,
        } && kv != heads as u64
        {
            return Err(unsupported(fields.key(KV_HEADS)));
        }
        if !width.is_multiple_of(heads) {
            return Err(fields.error(fields.key(WIDTH), ConfigReason::NotDivisible { by: heads }));
        }
        let head_dim = width / heads;
        let layers = fields.count(LAYERS, None)?;
        let max_positions = fields.count(MAX_POSITIONS, Some(2048))?;
        if u32::try_from(max_positions).is_err() {
            return Err(fields.error(
                fields.key(MAX_POSITIONS),
                ConfigReason::Exceeds {
                    max: u32::MAX as usize,
                },
            ));
        }
        let inter = match raw {
            RawConfig::HfJson { .. } => width
                .checked_mul(fields.count(EXPANSION_RATIO, Some(4))?)
                .ok_or(fields.error(
                    fields.key(EXPANSION_RATIO),
                    ConfigReason::Exceeds { max: usize::MAX },
                ))?,
            RawConfig::Gguf(_) => fields.count(FFN, None)?,
        };
        let eps = fields.float(EPS, 1e-5)?;
        let norm = NormParams::of(NormKind::Layer, eps)
            .map_err(|_| fields.error(fields.key(EPS), ConfigReason::NotFinitePositive))?;
        let param = |e: crate::components::standard::ParamError| e.for_family(FAMILY);
        // ALiBi replaces RoPE; the checked rope only satisfies the constructor.
        let rope = RopeParams::new(head_dim, 10_000.0, &RopeFlavor::Plain).map_err(param)?;
        let attention = AttentionParams::new(heads, heads, head_dim, false, rope)
            .map_err(param)?
            .with_alibi();
        Ok(Self {
            vocab,
            width,
            inter,
            layers,
            max_positions,
            attention,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            eos: fields.eos(Some(0))?,
            bos: fields.bos()?,
        })
    }
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    ffn: PlainMlpWeights,
}

#[derive(Debug)]
pub struct Mpt {
    params: MptParams,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = MptParams::from_raw(raw)?;
    let rows = match raw.source() {
        ConfigSource::HfModelType => HF_ROWS,
        ConfigSource::GgufArchitecture => {
            if store.contains(GGUF_POSITIONS) {
                return Err(ModelError::Config {
                    family: FAMILY,
                    field: GGUF_POSITIONS,
                    reason: ConfigReason::Unsupported,
                });
            }
            GGUF_MPT
        }
    };
    let h = p.width;
    let fused = FusedDims {
        q: h,
        kv: h,
        inter: 0,
    };
    let weights = family_weights_fused(FAMILY, store, &[rows], p.layers, false, fused)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention)?,
                ffn: PlainMlpWeights::new(&weights, l, p.width, p.inter, false)?,
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
    Ok(Box::new(Mpt {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Mpt {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace(&self, phase: Phase, shape: StepShape) -> Result<Graph, TraceError> {
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
                |b, BlockInput { block: h, .. }| plain_mlp(b, h, &layer.ffn, Gelu::Erf),
            )
        });
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny MPT whose projection widths are multiples of 32 (so a Q8_0 copy of
/// its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "mpt",
        "vocab_size": 48,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "expansion_ratio": 2,
        "max_seq_len": 64,
        "layer_norm_epsilon": 1e-5,
        "no_bias": true,
        "attn_config": { "alibi": true },
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

/// The `[out, in]` (or `[n]`) shape of every tensor an MPT checkpoint with `config`'s dims carries.
pub(crate) fn fixture_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (vocab, h) = (n("vocab_size"), n("d_model"));
    let inter = h * n("expansion_ratio");
    let mut shapes = vec![
        ("transformer.wte.weight".to_string(), vec![vocab, h]),
        ("transformer.norm_f.weight".to_string(), vec![h]),
        // MPT ties its head; an explicit one keeps the fixture's greedy decode from echoing.
        ("lm_head.weight".to_string(), vec![vocab, h]),
    ];
    for l in 0..n("n_layers") {
        let k = |s: &str| format!("transformer.blocks.{l}.{s}");
        shapes.extend([
            (k("norm_1.weight"), vec![h]),
            (k("norm_2.weight"), vec![h]),
            (k("attn.Wqkv.weight"), vec![3 * h, h]),
            (k("attn.out_proj.weight"), vec![h, h]),
            (k("ffn.up_proj.weight"), vec![inter, h]),
            (k("ffn.down_proj.weight"), vec![h, inter]),
        ]);
    }
    shapes
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = fixture_shapes(&config);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod family_tests {
    use poot_load::gguf::GgufValue;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_missing_weight_is_named, build as built, close, close_all, f32_entry, gguf_tensor,
        q8_0_store, read_back_owned, values,
    };
    use crate::registry::Registry;

    const VOCAB: usize = 48;
    const H: usize = 64;

    fn is_projection(key: &str) -> bool {
        ["Wqkv", "out_proj", "up_proj", "down_proj"]
            .iter()
            .any(|p| key.ends_with(&format!("{p}.weight")))
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// The fallback chat format this family names: ChatML (the template a checkpoint ships overrides
    /// it). Mutation: name another format in the family's config; the row goes red.
    #[test]
    fn the_fallback_chat_format_is_chatml() {
        let fixture = fixture();
        let model = (ENTRY.build)(&fixture.raw(), &fixture.store).unwrap();
        assert_eq!(model.config().chat, ChatFormat::ChatML);
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// SC-004 (HF half): BF16 and Q8_0 stores (the fused `Wqkv` packed, read as
    /// three row-range views) trace through the one body and the packed transform.
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
            "transformer.blocks.1.attn.Wqkv.weight",
            "transformer.blocks.0.ffn.down_proj.weight",
            "transformer.blocks.1.norm_2.weight",
            "transformer.norm_f.weight",
            "transformer.wte.weight",
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
    fn a_config_mpt_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "no_bias",
                json!(false),
                "no_bias",
                ConfigReason::Unsupported,
            ),
            (
                "attn_config",
                json!({"alibi": false}),
                "attn_config.alibi",
                ConfigReason::Unsupported,
            ),
            (
                "attn_config",
                json!({"alibi": true, "clip_qkv": 8.0}),
                "attn_config.clip_qkv",
                ConfigReason::Unsupported,
            ),
            (
                "attn_config",
                json!({"alibi": true, "qk_ln": true}),
                "attn_config.qk_ln",
                ConfigReason::Unsupported,
            ),
            (
                "attn_config",
                json!({"alibi": true, "kv_n_heads": 2}),
                "attn_config.kv_n_heads",
                ConfigReason::Unsupported,
            ),
            ("n_heads", json!(0), "n_heads", ConfigReason::Zero),
            (
                "n_heads",
                json!(3),
                "d_model",
                ConfigReason::NotDivisible { by: 3 },
            ),
            ("d_model", json!("64"), "d_model", ConfigReason::WrongType),
            (
                "layer_norm_epsilon",
                json!(0.0),
                "layer_norm_epsilon",
                ConfigReason::NotFinitePositive,
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

    /// An omitted eos is MPT's `<|endoftext|>` (0); the FFN width is `expansion_ratio * d_model`
    /// (default 4), so a store of the wrong width is a typed shape error naming the weight.
    #[test]
    fn omitted_fields_take_the_reference_defaults() {
        let fx = fixture();
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        assert_eq!(
            model_of(&config, &fx.store).config().eos,
            BTreeSet::from([0])
        );
        config.as_object_mut().unwrap().remove("expansion_ratio");
        match build(
            &RawConfig::HfJson {
                config: &config,
                generation: None,
            },
            &fx.store,
        ) {
            Err(ModelError::WeightShape { expected, .. }) => assert_eq!(expected[0], 4 * H),
            other => panic!("{other:?}"),
        }
    }

    /// A Q8_0 MPT GGUF (projections Q8_0, the rest F32, head tied) and the HF F32 store holding
    /// the same values, fused `Wqkv` included.
    fn gguf_case() -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let name = |hf: &str| -> String {
            match hf {
                "transformer.wte.weight" => "token_embd.weight".to_string(),
                "transformer.norm_f.weight" => "output_norm.weight".to_string(),
                _ => {
                    let rest = hf.strip_prefix("transformer.blocks.").unwrap();
                    let (layer, tensor) = rest.split_once('.').unwrap();
                    let tensor = match tensor {
                        "norm_1.weight" => "attn_norm.weight",
                        "norm_2.weight" => "ffn_norm.weight",
                        "attn.Wqkv.weight" => "attn_qkv.weight",
                        "attn.out_proj.weight" => "attn_output.weight",
                        "ffn.up_proj.weight" => "ffn_up.weight",
                        "ffn.down_proj.weight" => "ffn_down.weight",
                        other => panic!("unmapped fixture tensor {other}"),
                    };
                    format!("blk.{layer}.{tensor}")
                }
            }
        };
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter().filter(|(k, _)| k.as_str() != "lm_head.weight") {
            tensors.push(gguf_tensor(&name(key.as_str()), entry));
            reference
                .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                .unwrap();
        }
        let u = GgufValue::U32;
        let kvs = vec![
            (
                "general.architecture".to_string(),
                GgufValue::Str("mpt".into()),
            ),
            ("mpt.embedding_length".to_string(), u(H as u32)),
            ("mpt.feed_forward_length".to_string(), u(2 * H as u32)),
            ("mpt.block_count".to_string(), u(2)),
            ("mpt.attention.head_count".to_string(), u(4)),
            ("mpt.attention.head_count_kv".to_string(), u(4)),
            (
                "mpt.attention.layer_norm_epsilon".to_string(),
                GgufValue::F32(1e-5),
            ),
            ("mpt.context_length".to_string(), u(64)),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(
                    (0..VOCAB)
                        .map(|t| GgufValue::Str(format!("t{t}")))
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.eos_token_id".to_string(), u(47)),
        ];
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004 (R-562-3): a Q8_0 MPT GGUF resolves by its architecture key, builds packed
    /// projections (the fused `attn_qkv` read as three packed row ranges) traced as
    /// `PackedDequant`, and computes what the HF body computes over the same values. Mutation:
    /// drop the GGUF key from `ENTRY.keys` (`Unregistered`).
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 12);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        // The head is tied to the embedding in both.
        let mut config = fixture_config();
        config["tie_word_embeddings"] = json!(true);
        let reference = model_of(&config, &reference_store);
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

    /// A GGUF the ALiBi family cannot represent is refused naming the key or tensor.
    #[test]
    fn a_learned_position_gguf_is_refused() {
        let (index, store, _) = gguf_case();
        let mut with_positions = WeightStore::builder();
        for (key, entry) in store.iter() {
            with_positions.insert(key.clone(), entry.clone()).unwrap();
        }
        with_positions
            .insert(GGUF_POSITIONS, f32_entry(vec![4, 4], &[0.0; 16]))
            .unwrap();
        match build(&RawConfig::Gguf(&index), &with_positions.build()) {
            Err(ModelError::Config { field, reason, .. }) => {
                assert_eq!((field, reason), (GGUF_POSITIONS, ConfigReason::Unsupported));
            }
            other => panic!("{other:?}"),
        }
    }
}
