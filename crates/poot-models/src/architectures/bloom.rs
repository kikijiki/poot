//! BLOOM decoder tracer (card 255 item 2) for `bigscience/bloom-560m` and siblings, an ALiBi architecture
//! (Press et al., spec 254).
//!
//! BLOOM does not fit `crate::qwen2::Qwen2Config` as a flag:
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
//! are `Runner`-level binder concerns. This module only declares the `[n_heads, cap]` (decode) /
//! `[1, n_heads, l, l]` (prefill) mask shape, mirroring `Qwen2Config::alibi=true`.
//!
//! **Attention shape.** Plain multi-head attention (`n_kv_heads == n_heads`), so `n_rep = 1`
//! throughout.

use poot_graph_ir::ops::{
    alibi_mask_from_pos, attention_masked, attention_prefill, gelu, layernorm, linear,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::ValidationOutputs;
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
    LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step, standard_layer,
    standard_stack,
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
// re-packs it block-concatenated, so a GGUF's is read as three row ranges. The tracers below this
// section are the legacy ones the Runner still calls until Card 737; the tests use them as the
// independent reference.

pub const FAMILY: FamilyKey = FamilyKey::new("bloom");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "bloom"),
        (ConfigSource::GgufArchitecture, "bloom"),
    ],
    build,
    fixture,
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
                |b, h| plain_mlp(b, h, &layer.ffn, Gelu::Tanh),
            )
        });
        Ok(step.finish(b, logits))
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
        legacy_eval, q8_0_store, read_back_owned, values,
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

    fn legacy_cfg() -> BloomConfig {
        BloomConfig {
            vocab: VOCAB,
            hidden: H,
            n_heads: HEADS,
            layers: 2,
            ffn_inter: 4 * H,
            eps: 1e-5,
        }
    }

    /// The legacy tracers' store: the fixture decoded to F32 with each fused
    /// `query_key_value` de-interleaved into the separate `q_proj`/`k_proj`/`v_proj` weights and
    /// biases they declare.
    fn legacy_store(store: &WeightStore) -> WeightStore {
        let mut out = WeightStore::builder();
        for (key, entry) in store.iter() {
            let v = values(entry);
            let name = key.as_str();
            let split = [
                ("query_key_value.weight", "weight", H),
                ("query_key_value.bias", "bias", 1),
            ]
            .into_iter()
            .find_map(|(suffix, kind, cols)| {
                name.strip_suffix(suffix).map(|prefix| (prefix, kind, cols))
            });
            match split {
                Some((prefix, kind, cols)) => {
                    // Part `i`'s row `r` is block-concatenated row `i * H + r`.
                    for (i, proj) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
                        let part: Vec<f32> = (0..H)
                            .flat_map(|r| {
                                let at = interleaved_row(i * H + r);
                                v[at * cols..][..cols].to_vec()
                            })
                            .collect();
                        let shape = if cols == 1 { vec![H] } else { vec![H, H] };
                        out.insert(format!("{prefix}{proj}.{kind}"), f32_entry(shape, &part))
                            .unwrap();
                    }
                }
                None => out
                    .insert(key.clone(), f32_entry(entry.shape(), &v))
                    .unwrap(),
            }
        }
        out.build()
    }

    /// SC-002 (ADR-0101 tiers 1 and 2): the standard-stack body equals the legacy BLOOM prefill
    /// and decode tracers on the fixture, with ALiBi slopes `2^(-8 (h + 1) / 4)`. Mutation:
    /// split the fused QKV 3-axis as `[3, heads, head_dim]` instead of `[heads, 3, head_dim]`;
    /// red.
    #[test]
    fn bloom_matches_the_legacy_tracers() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let reference = legacy_store(&fx.store);
        let cfg = legacy_cfg();
        let slopes: Vec<f32> = (1..=HEADS)
            .map(|h| 2f32.powf(-8.0 * h as f32 / HEADS as f32))
            .collect();
        let host = [("alibi.slopes", slopes)];
        let eval = |g: &Graph, tokens: &[i32], pos: &[i32], state: &[Vec<f32>]| {
            legacy_eval(
                g,
                &reference,
                &host,
                &["transformer.word_embeddings.weight"],
                tokens,
                pos,
                state,
            )
        };
        // The legacy prefill returns no cache: its last-position logits equal the new prefill's
        // at every prefix; the legacy decode, carrying its own cache token by token, equals the
        // new decode step for step (logits and every cache element).
        let prompt = [3, 17, 40, 8, 25, 11, 30, 2];
        let (logits, _) = run_step(
            &*model,
            &fx.store,
            Phase::Prefill,
            CAP,
            &prompt[..5],
            0,
            &[],
        );
        for n in 1..=5 {
            let pos: Vec<i32> = (0..n as i32).collect();
            let (old, _) = eval(&trace_bloom_prefill(&cfg, n), &prompt[..n], &pos, &[]);
            close(&logits[(n - 1) * VOCAB..n * VOCAB], &old);
        }
        let legacy_decode = trace_bloom_decode_kv_masked(&cfg, CAP);
        let (mut state, mut legacy_state) = (Vec::new(), Vec::new());
        for (i, &token) in prompt.iter().enumerate() {
            let pos = i as i32;
            let (new, new_state) = run_step(
                &*model,
                &fx.store,
                Phase::Decode,
                CAP,
                &[token],
                pos,
                &state,
            );
            let (old, old_state) = eval(&legacy_decode, &[token], &[pos], &legacy_state);
            close(&new, &old);
            close_all(&new_state, &old_state);
            (state, legacy_state) = (new_state, old_state);
        }
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

// ---- Legacy tracers (kept for the Runner until Card 737; the tests' independent reference) ----

/// A BLOOM decoder config, matching `bigscience/bloom-560m`'s `config.json` (`hidden_size`,
/// `n_head`, `n_layer`, `layer_norm_epsilon`, `vocab_size`). The FFN width has no config key
/// (`modeling_bloom.py` hardcodes `4 * hidden_size`) and is an explicit field here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomConfig {
    pub vocab: usize,
    pub hidden: usize,
    /// Query/key/value head count (`n_head`); plain MHA, so no separate KV count.
    pub n_heads: usize,
    /// Transformer block count (`n_layer` in the real config).
    pub layers: usize,
    /// MLP intermediate width (`dense_h_to_4h` output); `4 * hidden` on real checkpoints.
    pub ffn_inter: usize,
    /// LayerNorm epsilon (`layer_norm_epsilon`; `1e-5` on real checkpoints).
    pub eps: f32,
}

impl BloomConfig {
    /// `hidden / n_heads`; BLOOM has no `head_dim` config key.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
}

/// Declare this layer's split QKV projection (see "Fused QKV" in the module docs) and return
/// `(q, k, v)`, each `[1, .., n_heads, head_dim]` (`..` is `l` for prefill, `1` for decode).
fn bloom_qkv(
    b: &Builder,
    x: Traced,
    p: impl Fn(&str) -> String,
    h: usize,
) -> (Traced, Traced, Traced) {
    let wq = b.constant(
        &p("self_attention.q_proj.weight"),
        TensorType::f32(vec![h, h]),
    );
    let bq = b.constant(&p("self_attention.q_proj.bias"), TensorType::f32(vec![h]));
    let wk = b.constant(
        &p("self_attention.k_proj.weight"),
        TensorType::f32(vec![h, h]),
    );
    let bk = b.constant(&p("self_attention.k_proj.bias"), TensorType::f32(vec![h]));
    let wv = b.constant(
        &p("self_attention.v_proj.weight"),
        TensorType::f32(vec![h, h]),
    );
    let bv = b.constant(&p("self_attention.v_proj.bias"), TensorType::f32(vec![h]));
    let q = linear(b, x, wq, Some(bq));
    let k = linear(b, x, wk, Some(bk));
    let v = linear(b, x, wv, Some(bv));
    (q, k, v)
}

/// Non-gated GELU MLP: `dense_h_to_4h` -> GELU -> `dense_4h_to_h`, both with bias. `ops::activation::gelu`
/// matches `modeling_bloom.py`'s tanh-approximate `bloom_gelu_forward`.
fn bloom_mlp(b: &Builder, x: Traced, p: impl Fn(&str) -> String, h: usize, inter: usize) -> Traced {
    let wi = b.constant(
        &p("mlp.dense_h_to_4h.weight"),
        TensorType::f32(vec![h, inter]),
    );
    let bi = b.constant(&p("mlp.dense_h_to_4h.bias"), TensorType::f32(vec![inter]));
    let up = linear(b, x, wi, Some(bi));
    let act = gelu(b, up);
    let wo = b.constant(
        &p("mlp.dense_4h_to_h.weight"),
        TensorType::f32(vec![inter, h]),
    );
    let bo = b.constant(&p("mlp.dense_4h_to_h.bias"), TensorType::f32(vec![h]));
    linear(b, act, wo, Some(bo))
}

/// Trace a full-sequence BLOOM prefill forward (the CPU-oracle path, mirroring
/// `crate::qwen2::trace_prefill`): embeddings, `word_embeddings_layernorm`, `cfg.layers` pre-norm
/// blocks (plain LayerNorm), `ln_f`, tied lm_head.
///
/// The lm_head reads a separate `lm_head.weight` constant (`[hidden,vocab]`, the loader's host-side
/// `transpose2d` of the embedding table in `crates/poot-llm/src/bloom_load.rs`), not an in-graph
/// `Transpose` of `embed`. The in-graph transpose re-materialized the full ~1GB table every decode
/// step (`generate_gpu_reprefill` retraces per step) and triggered a wgpu/RADV display-watchdog hang
/// (card 258); a pre-transposed weight
/// never hung. This matches `Runner::bind` and the other tied-lm_head tracers.
///
/// Expects these constants bound at eval time: `transformer.word_embeddings.weight`
/// `[vocab,hidden]` (gather source, untransposed), `lm_head.weight` `[hidden,vocab]` (the same table
/// transposed), `transformer.word_embeddings_layernorm.weight`/`.bias` `[hidden]`, per-layer
/// `transformer.h.{li}.{input_layernorm,post_attention_layernorm}.{weight,bias}`,
/// `self_attention.{q,k,v}_proj.{weight,bias}`, `self_attention.dense.{weight,bias}` (o_proj),
/// `mlp.{dense_h_to_4h,dense_4h_to_h}.{weight,bias}`, and `transformer.ln_f.weight`/`.bias`, plus the
/// `mask.prefill` step input (card 550a): the widened ALiBi mask `[1,n_heads,l,l]` from spec 254's
/// `alibi_prefill_mask`, not the plain `[1,1,l,l]` causal mask.
pub fn trace_bloom_prefill(cfg: &BloomConfig, seq_len: usize) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let d = cfg.head_dim();
    let l = seq_len;
    let scale = 1.0 / (d as f32).sqrt();

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    // this one-shot prefill always starts at position 0 (card 550).
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    // Per-head ALiBi mask (visibility + `-slope[h]*(i-j)`), built in-graph from `pos` against `iota(l)`
    // (card 550); the only positional signal, no RoPE tables.
    let slopes = b.constant("alibi.slopes", TensorType::f32(vec![hq]));
    let mask = alibi_mask_from_pos(&b, pos, l, None, slopes, hq); // [1,hq,l,l]

    let embed = b.constant(
        "transformer.word_embeddings.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, H]
    let emb = b.reshape(emb, vec![1, l, h]);

    // Extra post-embedding LayerNorm, applied once before layer 0.
    let emb_ln_w = b.constant(
        "transformer.word_embeddings_layernorm.weight",
        TensorType::f32(vec![h]),
    );
    let emb_ln_b = b.constant(
        "transformer.word_embeddings_layernorm.bias",
        TensorType::f32(vec![h]),
    );
    let mut x = layernorm(&b, emb, emb_ln_w, emb_ln_b, cfg.eps);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("transformer.h.{li}.{s}");

        let ln1_w = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let ln1_b = b.constant(&p("input_layernorm.bias"), TensorType::f32(vec![h]));
        let normed = layernorm(&b, x, ln1_w, ln1_b, cfg.eps);

        let (q, k, v) = bloom_qkv(&b, normed, p, h);
        let q = b.transpose(b.reshape(q, vec![1, l, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, l, hq, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, l, hq, d]), vec![0, 2, 1, 3]);

        // No RoPE (ALiBi is baked into `mask`); plain MHA, so n_rep = 1.
        let attn = attention_prefill(&b, q, k, v, 1, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, h]);

        let wo = b.constant(
            &p("self_attention.dense.weight"),
            TensorType::f32(vec![h, h]),
        );
        let bo = b.constant(&p("self_attention.dense.bias"), TensorType::f32(vec![h]));
        let attn = linear(&b, attn, wo, Some(bo));
        x = b.binary(BinOp::Add, x, attn);

        let ln2_w = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let ln2_b = b.constant(
            &p("post_attention_layernorm.bias"),
            TensorType::f32(vec![h]),
        );
        let normed = layernorm(&b, x, ln2_w, ln2_b, cfg.eps);
        let mlp = bloom_mlp(&b, normed, p, h, cfg.ffn_inter);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let ln_f_w = b.constant("transformer.ln_f.weight", TensorType::f32(vec![h]));
    let ln_f_b = b.constant("transformer.ln_f.bias", TensorType::f32(vec![h]));
    let x = layernorm(&b, x, ln_f_w, ln_f_b, cfg.eps);
    let last = b.slice(x, 1, l - 1, l); // only the last position feeds the LM head
    // Tied lm_head as a separate pre-transposed constant (see the doc comment, card 258).
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// BLOOM single-token fixed-KV masked decode (mirrors `crate::qwen2::trace_decode_kv_masked`): one
/// token in (`Slot::Token` at `Slot::Pos`), new k/v scattered into each layer's
/// `[1, n_heads, cap, head_dim]` cache, attention over the full cache with a host-filled ALiBi mask
/// (`Slot::Mask`, `[n_heads, cap]`, filled via `poot_llm::graphs::alibi_decode_mask_row`), no RoPE.
///
/// Expects the same constants as [`trace_bloom_prefill`], whose mask is likewise a `Slot::Mask` step
/// input (the prefill's tagged `mask.prefill`, here the untagged `[n_heads, cap]` row),
/// plus a `transformer.h.{li}.kv.{k,v}_cache` state pair per layer
/// (`[1, n_heads, cap, head_dim]`, zero-initialized by the caller).
pub fn trace_bloom_decode_kv_masked(cfg: &BloomConfig, cap: usize) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let d = cfg.head_dim();
    let scale = 1.0 / (d as f32).sqrt();

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, 1], DType::I32));
    let pos_slot = b.reshape(pos, vec![]);

    // Per-head ALiBi mask (spec 254 FR-003), always active for BLOOM, built in-graph from `pos` against
    // `iota(cap)` (card 550).
    let slopes = b.constant("alibi.slopes", TensorType::f32(vec![hq]));
    let mask = alibi_mask_from_pos(&b, pos, cap, None, slopes, hq); // [1,hq,1,cap]

    let embed = b.constant(
        "transformer.word_embeddings.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather_scalar(embed, 0, token);
    let emb = b.reshape(emb, vec![1, 1, h]);
    let emb_ln_w = b.constant(
        "transformer.word_embeddings_layernorm.weight",
        TensorType::f32(vec![h]),
    );
    let emb_ln_b = b.constant(
        "transformer.word_embeddings_layernorm.bias",
        TensorType::f32(vec![h]),
    );
    let mut x = layernorm(&b, emb, emb_ln_w, emb_ln_b, cfg.eps);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("transformer.h.{li}.{s}");

        let ln1_w = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let ln1_b = b.constant(&p("input_layernorm.bias"), TensorType::f32(vec![h]));
        let normed = layernorm(&b, x, ln1_w, ln1_b, cfg.eps);

        let (q, k, v) = bloom_qkv(&b, normed, p, h);
        let q = b.transpose(b.reshape(q, vec![1, 1, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, 1, hq, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, 1, hq, d]), vec![0, 2, 1, 3]);

        let kcache = b.state_input(
            &p("kv.k_cache"),
            TensorType::f32(vec![1, hq, cap, d]),
            StateRole::Recurrent,
        );
        let vcache = b.state_input(
            &p("kv.v_cache"),
            TensorType::f32(vec![1, hq, cap, d]),
            StateRole::Recurrent,
        );
        let kcache_out = b.dynamic_update_slice_dyn(kcache, k, pos_slot, 2);
        let vcache_out = b.dynamic_update_slice_dyn(vcache, v, pos_slot, 2);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

        let attn = attention_masked(&b, q, kcache_out, vcache_out, 1, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, h]);

        let wo = b.constant(
            &p("self_attention.dense.weight"),
            TensorType::f32(vec![h, h]),
        );
        let bo = b.constant(&p("self_attention.dense.bias"), TensorType::f32(vec![h]));
        let attn = linear(&b, attn, wo, Some(bo));
        x = b.binary(BinOp::Add, x, attn);

        let ln2_w = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let ln2_b = b.constant(
            &p("post_attention_layernorm.bias"),
            TensorType::f32(vec![h]),
        );
        let normed = layernorm(&b, x, ln2_w, ln2_b, cfg.eps);
        let mlp = bloom_mlp(&b, normed, p, h, cfg.ffn_inter);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let ln_f_w = b.constant("transformer.ln_f.weight", TensorType::f32(vec![h]));
    let ln_f_b = b.constant("transformer.ln_f.bias", TensorType::f32(vec![h]));
    let x = layernorm(&b, x, ln_f_w, ln_f_b, cfg.eps);
    // Tied lm_head as a separate pre-transposed constant (see `trace_bloom_prefill`, card 258).
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Storage;

    /// The fallback chat format this family names: ChatML (the template a checkpoint ships overrides
    /// it). Mutation: name another format in the family's config; the row goes red.
    #[test]
    fn the_fallback_chat_format_is_chatml() {
        let fixture = fixture();
        let model = (ENTRY.build)(&fixture.raw(), &fixture.store).unwrap();
        assert_eq!(model.config().chat, ChatFormat::ChatML);
    }

    fn tiny_cfg() -> BloomConfig {
        // Small dims with bloom-560m ratios: plain MHA, hidden divisible by n_heads.
        BloomConfig {
            vocab: 12,
            hidden: 8,
            n_heads: 2,
            layers: 2,
            ffn_inter: 16,
            eps: 1e-5,
        }
    }

    #[test]
    fn bloom_prefill_validates_and_has_no_rope_tables() {
        let cfg = tiny_cfg();
        let g = trace_bloom_prefill(&cfg, 5);
        g.validate().expect("bloom prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

        // No rope.cos/rope.sin const: ALiBi (in the `mask.prefill` step input) is the only positional signal.
        for id in &g.inputs {
            let m = &g.values[*id];
            if let Storage::Const = m.storage {
                let name = m.name.as_deref().unwrap_or("");
                assert!(
                    name != "rope.cos" && name != "rope.sin",
                    "BLOOM never uses RoPE, found {name}"
                );
            }
        }
        // Card 550: no `Slot::Mask` step input any more - the ALiBi mask is a graph computation over
        // `Slot::Pos` and `iota`.
        assert!(
            !g.inputs
                .iter()
                .any(|id| matches!(&g.values[*id].storage, Storage::Slot(Slot::Mask))),
            "bloom prefill must have no Slot::Mask step input"
        );
        let pos_id = g
            .inputs
            .iter()
            .find(|id| matches!(&g.values[**id].storage, Storage::Slot(Slot::Pos)))
            .expect("Slot::Pos step input present");
        assert_eq!(g.values[*pos_id].aval.shape, vec![1, 5]);
    }

    #[test]
    fn bloom_decode_kv_masked_validates_state_and_mask_shapes() {
        let cfg = tiny_cfg();
        let cap = 16;
        let g = trace_bloom_decode_kv_masked(&cfg, cap);
        g.validate().expect("bloom decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
        assert_eq!(g.state.len(), 2 * cfg.layers);
        for (si, _so) in &g.state {
            assert_eq!(g.aval(*si).shape, vec![1, cfg.n_heads, cap, cfg.head_dim()]);
            assert_eq!(g.values[*si].storage, Storage::State);
        }

        // Card 550: no `Slot::Mask` step input any more.
        assert!(
            !g.inputs
                .iter()
                .any(|id| matches!(&g.values[*id].storage, Storage::Slot(Slot::Mask))),
            "bloom decode must have no Slot::Mask step input"
        );
        let pos_id = g
            .inputs
            .iter()
            .find(|id| matches!(&g.values[**id].storage, Storage::Slot(Slot::Pos)))
            .expect("Slot::Pos present");
        assert_eq!(g.values[*pos_id].aval.shape, vec![1, 1]);
    }

    // ---- CPU-oracle numerics (poot_eval): decomposition checked against an independent Rust
    // reference forward pass. ----

    mod cpu_oracle {
        use super::*;
        use std::collections::HashMap;

        use poot_test_util::fill;

        use poot_test_util::seed_of;

        /// Small-magnitude xorshift noise; LayerNorm gamma is biased near 1.0, the rest near 0.
        fn weight(name: &str, n: usize, ln_gamma: bool) -> Vec<f32> {
            let raw = fill(n, seed_of(name));
            if ln_gamma {
                raw.iter().map(|v| 1.0 + v * 0.05).collect()
            } else {
                raw.iter().map(|v| v * 0.1).collect()
            }
        }

        /// All named constants [`trace_bloom_prefill`]/[`trace_bloom_decode_kv_masked`] declare, keyed
        /// by exact graph const name, generated deterministically from `cfg`.
        fn all_weights(cfg: &BloomConfig) -> HashMap<String, Vec<f32>> {
            let h = cfg.hidden;
            let mut w = HashMap::new();
            let embed = weight("embed", cfg.vocab * h, false);
            // `lm_head.weight` is the pre-transposed `embed` (card 258), keeping the tied relationship.
            let mut lm_head = vec![0.0f32; embed.len()];
            for v in 0..cfg.vocab {
                for hh in 0..h {
                    lm_head[hh * cfg.vocab + v] = embed[v * h + hh];
                }
            }
            w.insert("transformer.word_embeddings.weight".to_string(), embed);
            w.insert("lm_head.weight".to_string(), lm_head);
            w.insert(
                "transformer.word_embeddings_layernorm.weight".to_string(),
                weight("emb_ln_w", h, true),
            );
            w.insert(
                "transformer.word_embeddings_layernorm.bias".to_string(),
                weight("emb_ln_b", h, false),
            );
            for li in 0..cfg.layers {
                let p = |s: &str| format!("transformer.h.{li}.{s}");
                w.insert(p("input_layernorm.weight"), weight(&p("ln1w"), h, true));
                w.insert(p("input_layernorm.bias"), weight(&p("ln1b"), h, false));
                for proj in ["q_proj", "k_proj", "v_proj"] {
                    w.insert(
                        p(&format!("self_attention.{proj}.weight")),
                        weight(&p(&format!("{proj}w")), h * h, false),
                    );
                    w.insert(
                        p(&format!("self_attention.{proj}.bias")),
                        weight(&p(&format!("{proj}b")), h, false),
                    );
                }
                w.insert(
                    p("self_attention.dense.weight"),
                    weight(&p("ow"), h * h, false),
                );
                w.insert(p("self_attention.dense.bias"), weight(&p("ob"), h, false));
                w.insert(
                    p("post_attention_layernorm.weight"),
                    weight(&p("ln2w"), h, true),
                );
                w.insert(
                    p("post_attention_layernorm.bias"),
                    weight(&p("ln2b"), h, false),
                );
                w.insert(
                    p("mlp.dense_h_to_4h.weight"),
                    weight(&p("fc1w"), h * cfg.ffn_inter, false),
                );
                w.insert(
                    p("mlp.dense_h_to_4h.bias"),
                    weight(&p("fc1b"), cfg.ffn_inter, false),
                );
                w.insert(
                    p("mlp.dense_4h_to_h.weight"),
                    weight(&p("fc2w"), cfg.ffn_inter * h, false),
                );
                w.insert(p("mlp.dense_4h_to_h.bias"), weight(&p("fc2b"), h, false));
            }
            w.insert(
                "transformer.ln_f.weight".to_string(),
                weight("ln_f_w", h, true),
            );
            w.insert(
                "transformer.ln_f.bias".to_string(),
                weight("ln_f_b", h, false),
            );
            w
        }

        // ALiBi slopes/mask and LayerNorm are shared by the family tests via `crate::reference_ops`
        // (R474-014).
        use crate::reference_ops::{alibi_prefill_mask_ref, layernorm_ref};

        fn gelu_ref(v: f32) -> f32 {
            let z = (2.0_f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v);
            v / (1.0 + (-2.0 * z).exp())
        }

        /// `y[out] = x @ w[in,out] + b[out]`, `x`/`w`/`b`/`y` flat row-major.
        fn linear_ref(
            x: &[f32],
            w: &[f32],
            b: Option<&[f32]>,
            in_dim: usize,
            out_dim: usize,
        ) -> Vec<f32> {
            let mut y = vec![0.0f32; out_dim];
            for o in 0..out_dim {
                let mut acc = b.map(|bb| bb[o]).unwrap_or(0.0);
                for i in 0..in_dim {
                    acc += x[i] * w[i * out_dim + o];
                }
                y[o] = acc;
            }
            y
        }

        /// Direct-loop reference forward pass for [`trace_bloom_prefill`] (last token only), not a
        /// copy of the graph's op decomposition.
        fn bloom_prefill_ref(
            cfg: &BloomConfig,
            tokens: &[usize],
            w: &HashMap<String, Vec<f32>>,
        ) -> Vec<f32> {
            let (h, hq, d, l) = (cfg.hidden, cfg.n_heads, cfg.head_dim(), tokens.len());
            let scale = 1.0 / (d as f32).sqrt();
            let embed = &w["transformer.word_embeddings.weight"];

            let mut x: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| embed[t * h..(t + 1) * h].to_vec())
                .collect();
            let emb_ln_w = &w["transformer.word_embeddings_layernorm.weight"];
            let emb_ln_b = &w["transformer.word_embeddings_layernorm.bias"];
            for row in x.iter_mut() {
                *row = layernorm_ref(row, emb_ln_w, emb_ln_b, h, cfg.eps);
            }

            let mask = alibi_prefill_mask_ref(hq, l);

            for li in 0..cfg.layers {
                let p = |s: &str| format!("transformer.h.{li}.{s}");
                let ln1w = &w[&p("input_layernorm.weight")];
                let ln1b = &w[&p("input_layernorm.bias")];
                let normed: Vec<Vec<f32>> = x
                    .iter()
                    .map(|row| layernorm_ref(row, ln1w, ln1b, h, cfg.eps))
                    .collect();

                let qw = &w[&p("self_attention.q_proj.weight")];
                let qb = &w[&p("self_attention.q_proj.bias")];
                let kw = &w[&p("self_attention.k_proj.weight")];
                let kb = &w[&p("self_attention.k_proj.bias")];
                let vw = &w[&p("self_attention.v_proj.weight")];
                let vb = &w[&p("self_attention.v_proj.bias")];
                let q: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, qw, Some(qb), h, h))
                    .collect();
                let k: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, kw, Some(kb), h, h))
                    .collect();
                let v: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, vw, Some(vb), h, h))
                    .collect();

                // per-head causal + ALiBi attention, direct loop (independent of attention_prefill's
                // matmul/transpose/reduce decomposition).
                let mut attn_out = vec![vec![0.0f32; h]; l];
                for hh in 0..hq {
                    for i in 0..l {
                        let mut scores = vec![0.0f32; l];
                        for j in 0..l {
                            let mut s = 0.0f32;
                            for dd in 0..d {
                                s += q[i][hh * d + dd] * k[j][hh * d + dd];
                            }
                            scores[j] = s * scale + mask[(hh * l + i) * l + j];
                        }
                        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut denom = 0.0f32;
                        let mut e = vec![0.0f32; l];
                        for (j, sc) in scores.iter().enumerate() {
                            e[j] = (sc - m).exp();
                            denom += e[j];
                        }
                        for dd in 0..d {
                            let mut acc = 0.0f32;
                            for j in 0..l {
                                acc += (e[j] / denom) * v[j][hh * d + dd];
                            }
                            attn_out[i][hh * d + dd] = acc;
                        }
                    }
                }

                let ow = &w[&p("self_attention.dense.weight")];
                let ob = &w[&p("self_attention.dense.bias")];
                for i in 0..l {
                    let proj = linear_ref(&attn_out[i], ow, Some(ob), h, h);
                    for c in 0..h {
                        x[i][c] += proj[c];
                    }
                }

                let ln2w = &w[&p("post_attention_layernorm.weight")];
                let ln2b = &w[&p("post_attention_layernorm.bias")];
                let fc1w = &w[&p("mlp.dense_h_to_4h.weight")];
                let fc1b = &w[&p("mlp.dense_h_to_4h.bias")];
                let fc2w = &w[&p("mlp.dense_4h_to_h.weight")];
                let fc2b = &w[&p("mlp.dense_4h_to_h.bias")];
                for row in x.iter_mut().take(l) {
                    let normed = layernorm_ref(row, ln2w, ln2b, h, cfg.eps);
                    let up = linear_ref(&normed, fc1w, Some(fc1b), h, cfg.ffn_inter);
                    let act: Vec<f32> = up.iter().map(|&v| gelu_ref(v)).collect();
                    let down = linear_ref(&act, fc2w, Some(fc2b), cfg.ffn_inter, h);
                    for (c, d) in down.iter().enumerate() {
                        row[c] += d;
                    }
                }
            }

            let ln_f_w = &w["transformer.ln_f.weight"];
            let ln_f_b = &w["transformer.ln_f.bias"];
            let last = layernorm_ref(&x[l - 1], ln_f_w, ln_f_b, h, cfg.eps);
            // tied lm_head: logits[v] = last . embed[v]
            let mut logits = vec![0.0f32; cfg.vocab];
            for vv in 0..cfg.vocab {
                let mut acc = 0.0f32;
                for c in 0..h {
                    acc += last[c] * embed[vv * h + c];
                }
                logits[vv] = acc;
            }
            logits
        }

        fn eval_bloom_prefill(
            g: &Graph,
            tokens: &[usize],
            weights: &HashMap<String, Vec<f32>>,
        ) -> poot_tensor::HostTensor {
            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let t = match &meta.storage {
                    Storage::Slot(Slot::Token) => poot_tensor::HostTensor::i32(
                        vec![tokens.len()],
                        tokens.iter().map(|&t| t as i32).collect(),
                    ),
                    Storage::Slot(Slot::Pos) => {
                        // this one-shot prefill always starts at position 0 (card 550).
                        let l = meta.aval.shape[1];
                        poot_tensor::HostTensor::i32(
                            meta.aval.shape.clone(),
                            (0..l as i32).collect(),
                        )
                    }
                    Storage::Const => {
                        let name = meta.name.as_deref().expect("const without a name");
                        if name == "alibi.slopes" {
                            let n_heads = meta.aval.shape[0];
                            poot_tensor::HostTensor::f32(
                                meta.aval.shape.clone(),
                                crate::reference_ops::alibi_slopes_pow2(n_heads),
                            )
                        } else {
                            let data = weights
                                .get(name)
                                .unwrap_or_else(|| panic!("no weight bound for {name}"));
                            poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data.clone())
                        }
                    }
                    other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
                };
                inputs.insert(id, t.into());
            }
            poot_eval::eval(
                g,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("cpu eval")
            .output
            .into_host()
            .expect("dense output")
        }

        /// `trace_bloom_prefill` matches the independent reference within f32 tolerance. Uses `l=4`
        /// so causal+ALiBi masking exercises cross-position visibility.
        #[test]
        fn bloom_prefill_matches_hand_rolled_reference() {
            let cfg = tiny_cfg();
            let tokens = [3usize, 7, 1, 9];
            let weights = all_weights(&cfg);

            let g = trace_bloom_prefill(&cfg, tokens.len());
            let got = eval_bloom_prefill(&g, &tokens, &weights);

            let want = bloom_prefill_ref(&cfg, &tokens, &weights);
            assert_eq!(got.shape(), vec![1, 1, cfg.vocab]);
            poot_test_util::assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
            assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
            // Non-degenerate output.
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// Perturbing one entry of a mid-layer MLP down-projection bias changes the output, so the
        /// constant is read. One entry, not the whole vector: a uniform shift is cancelled by
        /// the final `ln_f` mean subtraction.
        #[test]
        fn bloom_prefill_output_is_sensitive_to_a_perturbed_weight() {
            let cfg = tiny_cfg();
            let tokens = [2usize, 5, 8];
            let mut weights = all_weights(&cfg);
            let g = trace_bloom_prefill(&cfg, tokens.len());
            let base = eval_bloom_prefill(&g, &tokens, &weights);

            let key = "transformer.h.1.mlp.dense_4h_to_h.bias".to_string();
            weights.get_mut(&key).unwrap()[0] += 5.0;
            let perturbed = eval_bloom_prefill(&g, &tokens, &weights);

            let max_diff =
                poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
            assert!(
                max_diff > 1e-3,
                "perturbing {key} should change the output; max_diff={max_diff:.2e}"
            );
        }
    }
}
