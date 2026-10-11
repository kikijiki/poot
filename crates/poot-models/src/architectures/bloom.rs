//! BLOOM decoder tracer (card 255 item 2) for `bigscience/bloom-560m` and siblings, an ALiBi architecture
//! (Press et al., spec 254).
//!
//! BLOOM does not fit the qwen2 family as a flag:
//! it uses plain LayerNorm for both block norms, an extra LayerNorm after the token embedding
//! (`word_embeddings_layernorm`) applied once before layer 0, a non-gated GELU MLP
//! (`down(gelu(up(x)))`), a fused QKV projection in the checkpoint, and no RoPE (ALiBi is always the
//! only positional signal). So it gets its own config and prefill/decode tracer pair, like
//! `crate::gemma4`.
//!
//! Blocks compose existing primitives ([`layernorm`], [`linear`], [`attention_masked`],
//! [`attention_prefill`] from `poot_graph_ir::ops`); no new graph-IR op or hand-fused kernel.
//!
//! **Fused QKV.** The checkpoint's `query_key_value` (`[3*hidden, hidden]`, out-major) is declared
//! here as three separate `self_attention.{q,k,v}_proj.weight`/`.bias` constants, each with every
//! head's rows concatenated in block order (`h0, h1, ...`). The real fused tensor is not
//! `[all_q | all_k | all_v]`: `modeling_bloom.py`'s `BloomAttention._reshape` uses
//! `(num_heads, 3, head_dim)`, so rows are per-head interleaved triples
//! (`q_h0, k_h0, v_h0, q_h1, ...`). `crates/poot-llm/src/bloom_load.rs`
//! (`split_qkv_weight`/`split_qkv_bias`) de-interleaves before binding; the phi3 contiguous
//! `row_slice` is not sufficient.
//!
//! **ALiBi mask filling.** `poot_llm::graphs::alibi_decode_mask_row`/`alibi_prefill_mask` (spec 254)
//! are driver-level binder concerns. This module only declares the `[n_heads, cap]` (decode) /
//! `[1, n_heads, l, l]` (prefill) mask shape.
//!
//! **Attention shape.** Plain multi-head attention (`n_kv_heads == n_heads`), so `n_rep = 1`
//! throughout.

use poot_graph_ir::{Builder, Graph};

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::rope_table::RopeFlavor;
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
use crate::names::gguf::GGUF_BLOOM;
use crate::names::{FusedDims, NameRow, Presence, Source, family_weights_fused};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

// The `bloom` family behind [`Model`]: LayerNorm blocks with biases, an embedding LayerNorm,
// ALiBi attention over a fused QKV projection and a plain tanh-GELU MLP with biases, in the
// standard stack and layer. An HF checkpoint's fused QKV interleaves per head, so it is one weight
// (`AttnRole::Qkv`) the attention component splits in the graph; llama.cpp's converter
// re-packs it block-concatenated, so a GGUF's is read as three row ranges.

pub const FAMILY: FamilyKey = FamilyKey::new("bloom");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "bloom"),
        (ConfigSource::GgufArchitecture, "bloom"),
    ],
    build,
    fixture,
    variants: &[],
};

/// The HF rows of a BLOOM checkpoint whose tensor names start with `$p` (`"transformer."`, or
/// `""` for a base-model checkpoint, which has no prefix): two LayerNorms with biases per layer,
/// the embedding LayerNorm, the fused `query_key_value` and a head tied to the embedding.
macro_rules! hf_rows {
    ($p:literal) => {
        &[
            NameRow {
                role: WeightRole::Embed,
                source: Source::Tensor(concat!($p, "word_embeddings.weight")),
                presence: Presence::Required,
            },
            NameRow {
                role: WeightRole::EmbedNorm,
                source: Source::Tensor(concat!($p, "word_embeddings_layernorm.weight")),
                presence: Presence::Required,
            },
            NameRow {
                role: WeightRole::EmbedNormBias,
                source: Source::Tensor(concat!($p, "word_embeddings_layernorm.bias")),
                presence: Presence::Required,
            },
            NameRow {
                role: WeightRole::FinalNorm,
                source: Source::Tensor(concat!($p, "ln_f.weight")),
                presence: Presence::Required,
            },
            NameRow {
                role: WeightRole::FinalNormBias,
                source: Source::Tensor(concat!($p, "ln_f.bias")),
                presence: Presence::Required,
            },
            NameRow {
                role: WeightRole::Head,
                source: Source::Linear("lm_head"),
                presence: Presence::TiedTo(WeightRole::Embed),
            },
            linear_row(
                WeightRole::Attn(AttnRole::Qkv),
                concat!($p, "h."),
                ".self_attention.query_key_value",
            ),
            tensor_row(
                WeightRole::Attn(AttnRole::QkvBias),
                concat!($p, "h."),
                ".self_attention.query_key_value.bias",
            ),
            linear_row(
                WeightRole::Attn(AttnRole::O),
                concat!($p, "h."),
                ".self_attention.dense",
            ),
            tensor_row(
                WeightRole::Attn(AttnRole::OBias),
                concat!($p, "h."),
                ".self_attention.dense.bias",
            ),
            tensor_row(
                WeightRole::Norm(NormRole::Attn),
                concat!($p, "h."),
                ".input_layernorm.weight",
            ),
            tensor_row(
                WeightRole::Norm(NormRole::AttnBias),
                concat!($p, "h."),
                ".input_layernorm.bias",
            ),
            tensor_row(
                WeightRole::Norm(NormRole::Ffn),
                concat!($p, "h."),
                ".post_attention_layernorm.weight",
            ),
            tensor_row(
                WeightRole::Norm(NormRole::FfnBias),
                concat!($p, "h."),
                ".post_attention_layernorm.bias",
            ),
            linear_row(
                WeightRole::Ffn(FfnRole::Up),
                concat!($p, "h."),
                ".mlp.dense_h_to_4h",
            ),
            tensor_row(
                WeightRole::Ffn(FfnRole::UpBias),
                concat!($p, "h."),
                ".mlp.dense_h_to_4h.bias",
            ),
            linear_row(
                WeightRole::Ffn(FfnRole::Down),
                concat!($p, "h."),
                ".mlp.dense_4h_to_h",
            ),
            tensor_row(
                WeightRole::Ffn(FfnRole::DownBias),
                concat!($p, "h."),
                ".mlp.dense_4h_to_h.bias",
            ),
        ]
    };
}

const HF_ROWS_PREFIXED: &[NameRow] = hf_rows!("transformer.");
const HF_ROWS_BARE: &[NameRow] = hf_rows!("");

const fn linear_row(role: WeightRole, prefix: &'static str, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear { prefix, stem },
        presence: Presence::Required,
    }
}

const fn tensor_row(role: WeightRole, prefix: &'static str, suffix: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor { prefix, suffix },
        presence: Presence::Required,
    }
}

/// The HF `bloom` fields, with the aliases `BloomConfig` accepts.
const WIDTH_HF: [&str; 2] = ["hidden_size", "n_embed"];
const HEADS_HF: [&str; 2] = ["n_head", "num_attention_heads"];
const LAYERS_HF: [&str; 2] = ["n_layer", "num_hidden_layers"];

/// The field of `aliases` an HF config sets (the first alias when none is set).
fn hf_alias(fields: &Fields<'_>, aliases: [&'static str; 2], gguf: &'static str) -> Field {
    let set = |key: &str| match fields.raw() {
        RawConfig::HfJson { config, .. } => config.get(key).is_some_and(|v| !v.is_null()),
        RawConfig::Gguf(_) => false,
    };
    let hf = if set(aliases[0]) || !set(aliases[1]) {
        aliases[0]
    } else {
        aliases[1]
    };
    Field::new(hf, gguf)
}

const EPS: Field = Field::new("layer_norm_epsilon", "bloom.attention.layer_norm_epsilon");
const MAX_POSITIONS: Field = Field::new("seq_length", "bloom.context_length");
const FFN: Field = Field::new("", "bloom.feed_forward_length");

/// bloom's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct BloomParams {
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

impl BloomParams {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let vocab = fields.vocab()?;
        let width = fields.count(hf_alias(&fields, WIDTH_HF, "bloom.embedding_length"), None)?;
        let heads_field = hf_alias(&fields, HEADS_HF, "bloom.attention.head_count");
        let heads = fields.count(heads_field, None)?;
        if !width.is_multiple_of(heads) {
            return Err(fields.error(
                fields.key(heads_field),
                ConfigReason::NotDivisible { by: heads },
            ));
        }
        let head_dim = width / heads;
        let layers = fields.count(hf_alias(&fields, LAYERS_HF, "bloom.block_count"), None)?;
        // ALiBi has no position limit; the HF config's `seq_length` (default 2048) bounds the cache.
        let max_positions = fields.count(MAX_POSITIONS, Some(2048))?;
        if u32::try_from(max_positions).is_err() {
            return Err(fields.error(
                fields.key(MAX_POSITIONS),
                ConfigReason::Exceeds {
                    max: u32::MAX as usize,
                },
            ));
        }
        // `modeling_bloom.py` hardcodes the FFN width as `4 * hidden_size`.
        let inter = match raw {
            RawConfig::HfJson { .. } => 4 * width,
            RawConfig::Gguf(_) => fields.count(FFN, None)?,
        };
        let eps = fields.float(EPS, 1e-5)?;
        let norm = NormParams::of(NormKind::Layer, eps)
            .map_err(|_| fields.error(fields.key(EPS), ConfigReason::NotFinitePositive))?;
        let param = |e: crate::components::standard::ParamError| e.for_family(FAMILY);
        // ALiBi replaces RoPE; the checked rope only satisfies the constructor.
        let rope = RopeParams::new(head_dim, 10_000.0, &RopeFlavor::Plain).map_err(param)?;
        let mut attention = AttentionParams::new(heads, heads, head_dim, true, rope)
            .map_err(param)?
            .with_alibi()
            .with_out_bias();
        if matches!(raw, RawConfig::HfJson { .. }) {
            attention = attention.with_fused_qkv().map_err(param)?;
        }
        Ok(Self {
            vocab,
            width,
            inter,
            layers,
            max_positions,
            attention,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            eos: fields.eos(Some(2))?,
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
pub struct Bloom {
    params: BloomParams,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = BloomParams::from_raw(raw)?;
    let rows = match raw.source() {
        ConfigSource::HfModelType if store.contains("word_embeddings.weight") => HF_ROWS_BARE,
        ConfigSource::HfModelType => HF_ROWS_PREFIXED,
        ConfigSource::GgufArchitecture => GGUF_BLOOM,
    };
    let h = p.width;
    let fused = FusedDims {
        q: h,
        kv: h,
        inter: 0,
    };
    let weights = family_weights_fused(FAMILY, store, &[rows], p.layers, false, fused)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width)
        .and_then(|s| s.with_final_norm_bias(&weights, p.width))
        .and_then(|s| s.with_embed_norm(&weights, p.width))
        .map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::biased(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention)?,
                ffn: PlainMlpWeights::new(&weights, l, p.width, p.inter, true)?,
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
    Ok(Box::new(Bloom {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Bloom {
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
                |b, BlockInput { block: h, .. }| plain_mlp(b, h, &layer.ffn, Gelu::Tanh),
            )
        });
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny BLOOM whose projection widths are multiples of 32 (so a Q8_0 copy
/// of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "bloom",
        "vocab_size": 48,
        "hidden_size": 64,
        "n_head": 4,
        "n_layer": 2,
        "layer_norm_epsilon": 1e-5,
        "seq_length": 64,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

/// The shape of every tensor a BLOOM checkpoint with `config`'s dims carries, by HF name.
pub(crate) fn fixture_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (vocab, h) = (n("vocab_size"), n("hidden_size"));
    let inter = 4 * h;
    let mut shapes = vec![
        (
            "transformer.word_embeddings.weight".to_string(),
            vec![vocab, h],
        ),
        (
            "transformer.word_embeddings_layernorm.weight".to_string(),
            vec![h],
        ),
        (
            "transformer.word_embeddings_layernorm.bias".to_string(),
            vec![h],
        ),
        ("transformer.ln_f.weight".to_string(), vec![h]),
        ("transformer.ln_f.bias".to_string(), vec![h]),
        // BLOOM ties its head; an explicit one keeps the fixture's greedy decode from echoing.
        ("lm_head.weight".to_string(), vec![vocab, h]),
    ];
    for l in 0..n("n_layer") {
        let k = |s: &str| format!("transformer.h.{l}.{s}");
        shapes.extend([
            (k("input_layernorm.weight"), vec![h]),
            (k("input_layernorm.bias"), vec![h]),
            (k("post_attention_layernorm.weight"), vec![h]),
            (k("post_attention_layernorm.bias"), vec![h]),
            (k("self_attention.query_key_value.weight"), vec![3 * h, h]),
            (k("self_attention.query_key_value.bias"), vec![3 * h]),
            (k("self_attention.dense.weight"), vec![h, h]),
            (k("self_attention.dense.bias"), vec![h]),
            (k("mlp.dense_h_to_4h.weight"), vec![inter, h]),
            (k("mlp.dense_h_to_4h.bias"), vec![inter]),
            (k("mlp.dense_4h_to_h.weight"), vec![h, inter]),
            (k("mlp.dense_4h_to_h.bias"), vec![h]),
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
    const HEADS: usize = 4;
    const D: usize = H / HEADS;

    fn is_projection(key: &str) -> bool {
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
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

    /// Row `part * H + h * D + e` of a block-concatenated `[q | k | v]` tensor is row
    /// `(h * 3 + part) * D + e` of BLOOM's per-head-interleaved fused tensor.
    fn interleaved_row(i: usize) -> usize {
        let (part, rest) = (i / H, i % H);
        ((rest / D) * 3 + part) * D + rest % D
    }

    /// Row `r` of `v` (`rows` rows of `cols`) moved to row `to(r)`.
    fn permute_rows(v: &[f32], cols: usize, to: impl Fn(usize) -> usize) -> Vec<f32> {
        let mut out = vec![0.0; v.len()];
        for (r, row) in v.chunks(cols).enumerate() {
            out[to(r) * cols..][..cols].copy_from_slice(row);
        }
        out
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

    /// SC-004 (HF half): BF16 and Q8_0 stores (the fused `query_key_value` packed)
    /// trace through the one body and the packed transform.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body_and_the_packed_transform() {
        assert_bf16_and_q8_0_trace_through_the_packed_transform(&ENTRY, &fixture(), is_projection);
    }

    /// A base-model checkpoint (no `transformer.` prefix, as the real safetensors files are)
    /// builds the same model.
    #[test]
    fn an_unprefixed_checkpoint_builds_the_same_model() {
        let fx = fixture();
        let mut bare = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            let name = key
                .as_str()
                .strip_prefix("transformer.")
                .unwrap_or(key.as_str());
            bare.insert(name, entry.clone()).unwrap();
        }
        let bare = bare.build();
        let prompt = [3, 17, 40];
        let (got, _) = run_step(
            &*model_of(&fx.config, &bare),
            &bare,
            Phase::Prefill,
            CAP,
            &prompt,
            0,
            &[],
        );
        let (want, _) = run_step(
            &*model_of(&fx.config, &fx.store),
            &fx.store,
            Phase::Prefill,
            CAP,
            &prompt,
            0,
            &[],
        );
        close(&got, &want);
    }

    /// SC-003: a store missing one required tensor fails `Registry::build` with the typed
    /// missing-weight error naming it, before any trace. Mutation: skip the row check; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "transformer.h.1.self_attention.query_key_value.weight",
            "transformer.h.0.self_attention.query_key_value.bias",
            "transformer.h.1.self_attention.dense.bias",
            "transformer.h.0.mlp.dense_4h_to_h.weight",
            "transformer.h.1.post_attention_layernorm.bias",
            "transformer.ln_f.bias",
            "transformer.word_embeddings_layernorm.weight",
            "transformer.word_embeddings.weight",
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
    fn a_malformed_config_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            ("n_head", json!(0), "n_head", ConfigReason::Zero),
            (
                "n_head",
                json!(3),
                "n_head",
                ConfigReason::NotDivisible { by: 3 },
            ),
            (
                "hidden_size",
                json!("64"),
                "hidden_size",
                ConfigReason::WrongType,
            ),
            (
                "layer_norm_epsilon",
                json!(0.0),
                "layer_norm_epsilon",
                ConfigReason::NotFinitePositive,
            ),
            ("n_layer", json!(0), "n_layer", ConfigReason::Zero),
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

    /// The HF aliases `n_embed` and `num_attention_heads` configure the same model, and an
    /// omitted eos is BLOOM's `</s>` (2).
    #[test]
    fn aliased_fields_and_the_default_eos_are_read() {
        let fx = fixture();
        let mut config = fixture_config();
        let object = config.as_object_mut().unwrap();
        let width = object.remove("hidden_size").unwrap();
        object.insert("n_embed".into(), width);
        let heads = object.remove("n_head").unwrap();
        object.insert("num_attention_heads".into(), heads);
        object.remove("eos_token_id");
        let model = model_of(&config, &fx.store);
        assert_eq!(model.config().eos, BTreeSet::from([2]));
    }

    /// A Q8_0 BLOOM GGUF (llama.cpp's block-concatenated `attn_qkv` packed, the rest F32, head
    /// tied) and the HF F32 store holding the same values with the fused rows interleaved per
    /// head.
    fn gguf_case() -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let name = |hf: &str| -> String {
            let table = [
                ("word_embeddings_layernorm.weight", "token_embd_norm.weight"),
                ("word_embeddings_layernorm.bias", "token_embd_norm.bias"),
                ("word_embeddings.weight", "token_embd.weight"),
                ("ln_f.weight", "output_norm.weight"),
                ("ln_f.bias", "output_norm.bias"),
            ];
            let bare = hf.strip_prefix("transformer.").unwrap();
            if let Some((_, gguf)) = table.iter().find(|(h, _)| *h == bare) {
                return gguf.to_string();
            }
            let rest = bare.strip_prefix("h.").unwrap();
            let (layer, tensor) = rest.split_once('.').unwrap();
            let tensor = match tensor {
                "input_layernorm.weight" => "attn_norm.weight",
                "input_layernorm.bias" => "attn_norm.bias",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "post_attention_layernorm.bias" => "ffn_norm.bias",
                "self_attention.query_key_value.weight" => "attn_qkv.weight",
                "self_attention.query_key_value.bias" => "attn_qkv.bias",
                "self_attention.dense.weight" => "attn_output.weight",
                "self_attention.dense.bias" => "attn_output.bias",
                "mlp.dense_h_to_4h.weight" => "ffn_up.weight",
                "mlp.dense_h_to_4h.bias" => "ffn_up.bias",
                "mlp.dense_4h_to_h.weight" => "ffn_down.weight",
                "mlp.dense_4h_to_h.bias" => "ffn_down.bias",
                other => panic!("unmapped fixture tensor {other}"),
            };
            format!("blk.{layer}.{tensor}")
        };
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter().filter(|(k, _)| k.as_str() != "lm_head.weight") {
            let key_str = key.as_str();
            if key_str.ends_with("query_key_value.weight") {
                // The packed tensor is in GGUF order; the HF reference interleaves its values.
                tensors.push(gguf_tensor(&name(key_str), entry));
                let hf = permute_rows(&values(entry), H, interleaved_row);
                reference
                    .insert(key.clone(), f32_entry(entry.shape(), &hf))
                    .unwrap();
            } else if key_str.ends_with("query_key_value.bias") {
                // The fixture's bias is HF-ordered; GGUF stores it block-concatenated.
                let hf = values(entry);
                let blocked = {
                    let mut out = vec![0.0; hf.len()];
                    for i in 0..hf.len() {
                        out[i] = hf[interleaved_row(i)];
                    }
                    out
                };
                tensors.push(gguf_tensor(
                    &name(key_str),
                    &f32_entry(entry.shape(), &blocked),
                ));
                reference.insert(key.clone(), entry.clone()).unwrap();
            } else {
                tensors.push(gguf_tensor(&name(key_str), entry));
                reference
                    .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                    .unwrap();
            }
        }
        let u = GgufValue::U32;
        let kvs = vec![
            (
                "general.architecture".to_string(),
                GgufValue::Str("bloom".into()),
            ),
            ("bloom.embedding_length".to_string(), u(H as u32)),
            ("bloom.feed_forward_length".to_string(), u(4 * H as u32)),
            ("bloom.block_count".to_string(), u(2)),
            ("bloom.attention.head_count".to_string(), u(HEADS as u32)),
            (
                "bloom.attention.layer_norm_epsilon".to_string(),
                GgufValue::F32(1e-5),
            ),
            ("bloom.context_length".to_string(), u(64)),
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

    /// SC-004 (R-562-3): a Q8_0 BLOOM GGUF resolves by its architecture key, builds packed
    /// projections (the block-concatenated `attn_qkv` read as three packed row ranges) traced as
    /// `PackedDequant`, and computes what the HF body (fused interleaved QKV) computes over the
    /// same values. Mutation: drop the GGUF key from `ENTRY.keys` (`Unregistered`).
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 12);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        // The HF reference has no `lm_head.weight`: its head is tied to the embedding too.
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
}
