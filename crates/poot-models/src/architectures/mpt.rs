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
//! `Runner`-level binder concern).
//!
//! **Attention shape.** Default `attn_type` is `"multihead_attention"` (plain MHA, `kv_n_heads == n_heads`), so
//! `n_rep = 1`. `multiquery_attention`/grouped variants are not implemented (BLOOM has no GQA either).
//!
//! **Tied lm_head.** `MPTForCausalLM` ties `lm_head` to `wte` by default (`config.tie_word_embeddings`) on every
//! public checkpoint; reused transposed, as BLOOM/gemma4.
//!
//! Structural template: `crate::bloom`. Primitive composition only (AGENTS.md): every op is an existing
//! `poot_graph_ir::ops` composition (`layernorm_no_bias`/`linear`/`attention_masked`/`attention_prefill`).

use poot_graph_ir::ops::{
    alibi_mask_from_pos, attention_masked, attention_prefill, gelu_erf, layernorm_no_bias, linear,
};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::rope_table::RopeFlavor;
use poot_graph_ir::{
    BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced, ValidationOutputs,
};
use poot_quant::weights::{AttnRole, FfnRole, NormRole, WeightMap, WeightRole, WeightStore};
use poot_tensor::DType;

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
use crate::names::gguf::GGUF_MPT;
use crate::names::{FusedDims, FusedPart, NameRow, Presence, Source, family_weights_fused};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

// The `mpt` family behind [`Model`]: bias-free LayerNorm blocks, ALiBi attention over a fused
// contiguous `Wqkv`, and a plain exact-GELU MLP, in the standard stack and layer. The tracers
// below this section are the legacy ones the Runner still calls until Card 737; the tests use
// them as the independent reference.

pub const FAMILY: FamilyKey = FamilyKey::new("mpt");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "mpt"),
        (ConfigSource::GgufArchitecture, "mpt"),
    ],
    build,
    fixture,
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
                |b, h| plain_mlp(b, h, &layer.ffn, Gelu::Erf),
            )
        });
        Ok(step.finish(b, logits))
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

// ---- Legacy tracers (kept for the Runner until Card 737; the tests' independent reference) ----

/// An MPT decoder config, matching `mosaicml/mpt-7b`'s `config.json` (`d_model` -> `hidden`, `n_heads`, `n_layers` ->
/// `layers`, `vocab_size` -> `vocab`, `layer_norm_epsilon` -> `eps`). The FFN width has no config key
/// (`resolve_ffn_hidden_size()` gives `expansion_ratio * d_model`) and is an explicit field, as
/// `BloomConfig::ffn_inter`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MptConfig {
    pub vocab: usize,
    pub hidden: usize,
    /// Query/key/value head count (`n_heads`). MPT's default `attn_type` is plain MHA, so `n_kv_heads == n_heads`
    /// (see the module doc, "Attention shape").
    pub n_heads: usize,
    /// Transformer block count (`n_layers` in the real config).
    pub layers: usize,
    /// MLP intermediate width (`ffn.up_proj` output / `ffn.down_proj` input). `4 * hidden` (`expansion_ratio = 4`).
    pub ffn_inter: usize,
    /// LayerNorm epsilon (`layer_norm_epsilon`; `1e-5` on every real checkpoint).
    pub eps: f32,
}

impl MptConfig {
    /// `hidden / n_heads` (MPT has no `head_dim` config key).
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
}

/// Declare this layer's split QKV projection (see the module doc, "Fused QKV layout") and return `(q, k, v)`, each
/// `[1, .., n_heads, head_dim]` (`..` is `l` for prefill, `1` for decode; the caller reshapes/transposes). No bias.
fn mpt_qkv(
    b: &Builder,
    x: Traced,
    p: impl Fn(&str) -> String,
    h: usize,
) -> (Traced, Traced, Traced) {
    let wq = b.constant(&p("attn.q_proj.weight"), TensorType::f32(vec![h, h]));
    let wk = b.constant(&p("attn.k_proj.weight"), TensorType::f32(vec![h, h]));
    let wv = b.constant(&p("attn.v_proj.weight"), TensorType::f32(vec![h, h]));
    let q = linear(b, x, wq, None);
    let k = linear(b, x, wk, None);
    let v = linear(b, x, wv, None);
    (q, k, v)
}

/// This layer's non-gated GELU MLP: `down(gelu_erf(up(x)))`, both bias-free. `ops::activation::gelu_erf`
/// (exact, erf-based) matches `layers/ffn.py`'s `nn.GELU(approximate="none")`, not the tanh approximation
/// `crate::bloom` uses.
fn mpt_mlp(b: &Builder, x: Traced, p: impl Fn(&str) -> String, h: usize, inter: usize) -> Traced {
    let wi = b.constant(&p("ffn.up_proj.weight"), TensorType::f32(vec![h, inter]));
    let up = linear(b, x, wi, None);
    let act = gelu_erf(b, up);
    let wo = b.constant(&p("ffn.down_proj.weight"), TensorType::f32(vec![inter, h]));
    linear(b, act, wo, None)
}

/// Trace a full-sequence MPT prefill forward (the CPU-oracle path; shape of `crate::bloom::trace_bloom_prefill`):
/// embeddings, `cfg.layers` blocks (pre-norm attention + pre-norm MLP, bias-free LayerNorm), final `norm_f`, tied
/// lm_head. Unlike BLOOM there is no post-embedding norm.
///
/// Card 258: the lm_head reads a separate `lm_head.weight` constant (the loader's host-side `transpose2d` of
/// `wte.weight`), not an in-graph `Transpose`; see `crate::bloom::trace_bloom_prefill` for the display-watchdog
/// hang rationale.
///
/// Named constants bound at eval time: `transformer.wte.weight` `[vocab,hidden]` (the embedding gather source,
/// untransposed), `lm_head.weight` `[hidden,vocab]` (the same table transposed),
/// per-layer `transformer.blocks.{li}.{norm_1,norm_2}.weight` (no bias),
/// `attn.{q,k,v}_proj.weight`, `attn.out_proj.weight`, `ffn.{up_proj,down_proj}.weight` (no bias), and
/// `transformer.norm_f.weight`, plus the `mask.prefill` step input (card 550a): the widened ALiBi
/// visibility+bias mask `[1,n_heads,l,l]`, spec 254's `alibi_prefill_mask`, not the plain
/// `[1,1,l,l]` causal mask.
pub fn trace_mpt_prefill(cfg: &MptConfig, seq_len: usize) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let d = cfg.head_dim();
    let l = seq_len;
    let scale = 1.0 / (d as f32).sqrt();

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    // this one-shot prefill always starts at position 0 (card 550).
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    // The widened per-head ALiBi mask (visibility + `-slope[h]*(i-j)`), built in-graph from `pos` against
    // `iota(l)` (card 550): MPT's only positional signal; no RoPE tables.
    let slopes = b.constant("alibi.slopes", TensorType::f32(vec![hq]));
    let mask = alibi_mask_from_pos(&b, pos, l, None, slopes, hq); // [1,hq,l,l]

    let embed = b.constant(
        "transformer.wte.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, H]
    // No post-embedding norm (unlike BLOOM): the embedding feeds block 0 directly.
    let mut x = b.reshape(emb, vec![1, l, h]);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("transformer.blocks.{li}.{s}");

        let ln1_w = b.constant(&p("norm_1.weight"), TensorType::f32(vec![h]));
        let normed = layernorm_no_bias(&b, x, ln1_w, cfg.eps);

        let (q, k, v) = mpt_qkv(&b, normed, p, h);
        let q = b.transpose(b.reshape(q, vec![1, l, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, l, hq, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, l, hq, d]), vec![0, 2, 1, 3]);

        // No RoPE: ALiBi's bias (in `mask`) is the only positional signal. Plain MHA, so n_rep = 1.
        let attn = attention_prefill(&b, q, k, v, 1, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, h]);

        let wo = b.constant(&p("attn.out_proj.weight"), TensorType::f32(vec![h, h]));
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2_w = b.constant(&p("norm_2.weight"), TensorType::f32(vec![h]));
        let normed = layernorm_no_bias(&b, x, ln2_w, cfg.eps);
        let mlp = mpt_mlp(&b, normed, p, h, cfg.ffn_inter);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let ln_f_w = b.constant("transformer.norm_f.weight", TensorType::f32(vec![h]));
    let x = layernorm_no_bias(&b, x, ln_f_w, cfg.eps);
    let last = b.slice(x, 1, l - 1, l); // only the last position feeds the LM head
    // Tied lm_head: a separate pre-transposed constant, not an in-graph Transpose of `embed` (card 258).
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// MPT single-token fixed-KV masked decode (capture/replay shape of `crate::bloom::trace_bloom_decode_kv_masked`):
/// one token in (`Slot::Token` at `Slot::Pos`), the new k/v scattered into each layer's `[1, n_heads, cap,
/// head_dim]` cache, attention over the full cache with a host-filled widened ALiBi mask (`Slot::Mask`,
/// `[n_heads, cap]`, spec 254 FR-003; filled at bind time via `poot_llm::graphs::alibi_decode_mask_row`), no RoPE.
///
/// Same named constants as [`trace_mpt_prefill`], whose mask is likewise a `Slot::Mask` step input (the
/// prefill's tagged `mask.prefill`, here the untagged `[n_heads, cap]` row), plus a
/// `transformer.blocks.{li}.kv.{k,v}_cache` state pair per layer (`[1, n_heads, cap, head_dim]`, zero-initialized
/// by the caller).
pub fn trace_mpt_decode_kv_masked(cfg: &MptConfig, cap: usize) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let d = cfg.head_dim();
    let scale = 1.0 / (d as f32).sqrt();

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, 1], DType::I32));
    let pos_slot = b.reshape(pos, vec![]);

    // Widened per-head ALiBi mask, always active for MPT (no plain-RoPE fallback), built in-graph from
    // `pos` against `iota(cap)` (card 550).
    let slopes = b.constant("alibi.slopes", TensorType::f32(vec![hq]));
    let mask = alibi_mask_from_pos(&b, pos, cap, None, slopes, hq); // [1,hq,1,cap]

    let embed = b.constant(
        "transformer.wte.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather_scalar(embed, 0, token);
    // No post-embedding norm (unlike BLOOM).
    let mut x = b.reshape(emb, vec![1, 1, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("transformer.blocks.{li}.{s}");

        let ln1_w = b.constant(&p("norm_1.weight"), TensorType::f32(vec![h]));
        let normed = layernorm_no_bias(&b, x, ln1_w, cfg.eps);

        let (q, k, v) = mpt_qkv(&b, normed, p, h);
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

        let wo = b.constant(&p("attn.out_proj.weight"), TensorType::f32(vec![h, h]));
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2_w = b.constant(&p("norm_2.weight"), TensorType::f32(vec![h]));
        let normed = layernorm_no_bias(&b, x, ln2_w, cfg.eps);
        let mlp = mpt_mlp(&b, normed, p, h, cfg.ffn_inter);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let ln_f_w = b.constant("transformer.norm_f.weight", TensorType::f32(vec![h]));
    let x = layernorm_no_bias(&b, x, ln_f_w, cfg.eps);
    // Tied lm_head: a separate pre-transposed constant (card 258); see `trace_mpt_prefill`.
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

    fn tiny_cfg() -> MptConfig {
        // Small but non-degenerate dims (real mpt-7b ratios, scaled down): plain MHA, hidden divisible by n_heads.
        MptConfig {
            vocab: 12,
            hidden: 8,
            n_heads: 2,
            layers: 2,
            ffn_inter: 16,
            eps: 1e-5,
        }
    }

    #[test]
    fn mpt_prefill_validates_and_has_no_rope_tables() {
        let cfg = tiny_cfg();
        let g = trace_mpt_prefill(&cfg, 5);
        g.validate().expect("mpt prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

        // No rope.cos/rope.sin const: ALiBi (in the `mask.prefill` step input) is the only positional signal.
        for id in &g.inputs {
            let m = &g.values[*id];
            if let Storage::Const = m.storage {
                let name = m.name.as_deref().unwrap_or("");
                assert!(
                    name != "rope.cos" && name != "rope.sin",
                    "MPT never uses RoPE, found {name}"
                );
            }
        }
        // Card 550: no `Slot::Mask` step input any more - the widened ALiBi mask is a graph computation
        // over `Slot::Pos` and `iota`.
        assert!(
            !g.inputs
                .iter()
                .any(|id| matches!(&g.values[*id].storage, Storage::Slot(Slot::Mask))),
            "mpt prefill must have no Slot::Mask step input"
        );
        let pos_id = g
            .inputs
            .iter()
            .find(|id| matches!(&g.values[**id].storage, Storage::Slot(Slot::Pos)))
            .expect("Slot::Pos step input present");
        assert_eq!(g.values[*pos_id].aval.shape, vec![1, 5]);
    }

    #[test]
    fn mpt_prefill_declares_no_bias_constants() {
        // MPT's `no_bias=true` default: no linear or norm site declares a `.bias` constant.
        let cfg = tiny_cfg();
        let g = trace_mpt_prefill(&cfg, 5);
        for id in &g.inputs {
            let m = &g.values[*id];
            if let Storage::Const = m.storage {
                let name = m.name.as_deref().unwrap_or("");
                assert!(
                    !name.ends_with(".bias"),
                    "MPT (no_bias=true) should declare no bias constant, found {name}"
                );
            }
        }
    }

    #[test]
    fn mpt_decode_kv_masked_validates_state_and_mask_shapes() {
        let cfg = tiny_cfg();
        let cap = 16;
        let g = trace_mpt_decode_kv_masked(&cfg, cap);
        g.validate().expect("mpt decode graph should validate");
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
            "mpt decode must have no Slot::Mask step input"
        );
        let pos_id = g
            .inputs
            .iter()
            .find(|id| matches!(&g.values[**id].storage, Storage::Slot(Slot::Pos)))
            .expect("Slot::Pos present");
        assert_eq!(g.values[*pos_id].aval.shape, vec![1, 1]);
    }

    // ---- CPU-oracle numerics: checked against an independent from-scratch Rust reference forward pass. ----

    mod cpu_oracle {
        use super::*;
        use std::collections::HashMap;

        use poot_test_util::fill;

        use poot_test_util::seed_of;

        /// Weight init: small xorshift noise; LayerNorm gammas near 1.0 (as real checkpoints, update 0603), the rest
        /// near 0.
        fn weight(name: &str, n: usize, ln_gamma: bool) -> Vec<f32> {
            let raw = fill(n, seed_of(name));
            if ln_gamma {
                raw.iter().map(|v| 1.0 + v * 0.05).collect()
            } else {
                raw.iter().map(|v| v * 0.1).collect()
            }
        }

        /// All named constants the two tracers declare, keyed by graph const name, generated from `cfg`. No `.bias`.
        fn all_weights(cfg: &MptConfig) -> HashMap<String, Vec<f32>> {
            let h = cfg.hidden;
            let mut w = HashMap::new();
            let embed = weight("embed", cfg.vocab * h, false);
            // Card 258: `lm_head.weight` is a separate pre-transposed constant (`[hidden,vocab]`); transpose the same
            // `embed` data so the tied-weight relationship holds.
            let mut lm_head = vec![0.0f32; embed.len()];
            for v in 0..cfg.vocab {
                for hh in 0..h {
                    lm_head[hh * cfg.vocab + v] = embed[v * h + hh];
                }
            }
            w.insert("transformer.wte.weight".to_string(), embed);
            w.insert("lm_head.weight".to_string(), lm_head);
            for li in 0..cfg.layers {
                let p = |s: &str| format!("transformer.blocks.{li}.{s}");
                w.insert(p("norm_1.weight"), weight(&p("ln1w"), h, true));
                for proj in ["q_proj", "k_proj", "v_proj"] {
                    w.insert(
                        p(&format!("attn.{proj}.weight")),
                        weight(&p(&format!("{proj}w")), h * h, false),
                    );
                }
                w.insert(p("attn.out_proj.weight"), weight(&p("ow"), h * h, false));
                w.insert(p("norm_2.weight"), weight(&p("ln2w"), h, true));
                w.insert(
                    p("ffn.up_proj.weight"),
                    weight(&p("fc1w"), h * cfg.ffn_inter, false),
                );
                w.insert(
                    p("ffn.down_proj.weight"),
                    weight(&p("fc2w"), cfg.ffn_inter * h, false),
                );
            }
            w.insert(
                "transformer.norm_f.weight".to_string(),
                weight("ln_f_w", h, true),
            );
            w
        }

        // ALiBi slopes/mask and the bias-free LayerNorm reference are shared by the family tests via
        // `crate::reference_ops` (R474-014).
        use crate::reference_ops::{alibi_prefill_mask_ref, layernorm_no_bias_ref};

        /// Standalone erf approximation (Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7), independent of
        /// `poot_eval`'s `ops::activation::gelu_erf` (`libm::erf` in f64).
        fn erf_approx(x: f64) -> f64 {
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let p = 0.3275911;
            let t = 1.0 / (1.0 + p * x);
            let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
            sign * y
        }

        /// Exact (erf-based) GELU reference: `0.5*x*(1 + erf(x/sqrt(2)))`, not `crate::bloom`'s tanh approximation.
        fn gelu_erf_ref(v: f32) -> f32 {
            let x = v as f64;
            (0.5 * x * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))) as f32
        }

        use poot_test_util::linear_ref;

        /// Independent reference forward pass for [`trace_mpt_prefill`] (last token only), written as direct loops
        /// rather than a copy of the graph's decomposition, like `crate::bloom`'s `bloom_prefill_ref`.
        fn mpt_prefill_ref(
            cfg: &MptConfig,
            tokens: &[usize],
            w: &HashMap<String, Vec<f32>>,
        ) -> Vec<f32> {
            let (h, hq, d, l) = (cfg.hidden, cfg.n_heads, cfg.head_dim(), tokens.len());
            let scale = 1.0 / (d as f32).sqrt();
            let embed = &w["transformer.wte.weight"];

            // No post-embedding norm (unlike BLOOM).
            let mut x: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| embed[t * h..(t + 1) * h].to_vec())
                .collect();

            let mask = alibi_prefill_mask_ref(hq, l);

            for li in 0..cfg.layers {
                let p = |s: &str| format!("transformer.blocks.{li}.{s}");
                let ln1w = &w[&p("norm_1.weight")];
                let normed: Vec<Vec<f32>> = x
                    .iter()
                    .map(|row| layernorm_no_bias_ref(row, ln1w, h, cfg.eps))
                    .collect();

                let qw = &w[&p("attn.q_proj.weight")];
                let kw = &w[&p("attn.k_proj.weight")];
                let vw = &w[&p("attn.v_proj.weight")];
                let q: Vec<Vec<f32>> = normed.iter().map(|row| linear_ref(row, qw, h, h)).collect();
                let k: Vec<Vec<f32>> = normed.iter().map(|row| linear_ref(row, kw, h, h)).collect();
                let v: Vec<Vec<f32>> = normed.iter().map(|row| linear_ref(row, vw, h, h)).collect();

                // Per-head causal + ALiBi attention, direct loop (independent of attention_prefill's decomposition).
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

                let ow = &w[&p("attn.out_proj.weight")];
                for i in 0..l {
                    let proj = linear_ref(&attn_out[i], ow, h, h);
                    for c in 0..h {
                        x[i][c] += proj[c];
                    }
                }

                let ln2w = &w[&p("norm_2.weight")];
                let fc1w = &w[&p("ffn.up_proj.weight")];
                let fc2w = &w[&p("ffn.down_proj.weight")];
                for row in x.iter_mut().take(l) {
                    let normed = layernorm_no_bias_ref(row, ln2w, h, cfg.eps);
                    let up = linear_ref(&normed, fc1w, h, cfg.ffn_inter);
                    let act: Vec<f32> = up.iter().map(|&v| gelu_erf_ref(v)).collect();
                    let down = linear_ref(&act, fc2w, cfg.ffn_inter, h);
                    for (c, dd) in down.iter().enumerate() {
                        row[c] += dd;
                    }
                }
            }

            let ln_f_w = &w["transformer.norm_f.weight"];
            let last = layernorm_no_bias_ref(&x[l - 1], ln_f_w, h, cfg.eps);
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

        fn eval_mpt_prefill(
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

        /// `trace_mpt_prefill` matches an independent hand-loop reference within f32 tolerance. Multi-token (`l=4`)
        /// so causal+ALiBi masking exercises cross-position visibility.
        #[test]
        fn mpt_prefill_matches_hand_rolled_reference() {
            let cfg = tiny_cfg();
            let tokens = [3usize, 7, 1, 9];
            let weights = all_weights(&cfg);

            let g = trace_mpt_prefill(&cfg, tokens.len());
            let got = eval_mpt_prefill(&g, &tokens, &weights);

            let want = mpt_prefill_ref(&cfg, &tokens, &weights);
            assert_eq!(got.shape(), vec![1, 1, cfg.vocab]);
            poot_test_util::assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
            assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
            // Non-degenerate output (not all-zero/all-equal).
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// Perturbing one entry of a mid-layer MLP down-projection weight changes the output: the tracer is not
        /// ignoring that constant (MPT's MLP has no bias vector to perturb).
        #[test]
        fn mpt_prefill_output_is_sensitive_to_a_perturbed_weight() {
            let cfg = tiny_cfg();
            let tokens = [2usize, 5, 8];
            let mut weights = all_weights(&cfg);
            let g = trace_mpt_prefill(&cfg, tokens.len());
            let base = eval_mpt_prefill(&g, &tokens, &weights);

            let key = "transformer.blocks.1.ffn.down_proj.weight".to_string();
            weights.get_mut(&key).unwrap()[0] += 5.0;
            let perturbed = eval_mpt_prefill(&g, &tokens, &weights);

            let max_diff =
                poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
            assert!(
                max_diff > 1e-3,
                "perturbing {key} should change the output; max_diff={max_diff:.2e}"
            );
        }
    }
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

    fn legacy_cfg() -> MptConfig {
        MptConfig {
            vocab: VOCAB,
            hidden: H,
            n_heads: 4,
            layers: 2,
            ffn_inter: 2 * H,
            eps: 1e-5,
        }
    }

    /// The legacy tracers' store: the fixture decoded to F32 with each fused `Wqkv` cut into the
    /// separate `q_proj`/`k_proj`/`v_proj` weights they declare.
    fn legacy_store(store: &WeightStore) -> WeightStore {
        let mut out = WeightStore::builder();
        for (key, entry) in store.iter() {
            let v = values(entry);
            match key.as_str().strip_suffix("attn.Wqkv.weight") {
                Some(prefix) => {
                    for (i, name) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
                        let part = v[i * H * H..(i + 1) * H * H].to_vec();
                        out.insert(
                            format!("{prefix}attn.{name}.weight"),
                            f32_entry(vec![H, H], &part),
                        )
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

    /// SC-002 (ADR-0101 tiers 1 and 2): the standard-stack body equals the legacy MPT prefill and
    /// decode tracers on the fixture, with ALiBi slopes `2^(-8 (h + 1) / 4)`. Mutation: trace the
    /// family with `NormKind::Rms` instead of `NormKind::Layer`; red.
    #[test]
    fn mpt_matches_the_legacy_tracers() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let reference = legacy_store(&fx.store);
        let cfg = legacy_cfg();
        let slopes: Vec<f32> = (1..=4).map(|h| 2f32.powf(-8.0 * h as f32 / 4.0)).collect();
        let host = [("alibi.slopes", slopes)];
        let eval = |g: &Graph, tokens: &[i32], pos: &[i32], state: &[Vec<f32>]| {
            legacy_eval(
                g,
                &reference,
                &host,
                &["transformer.wte.weight"],
                tokens,
                pos,
                state,
            )
        };
        // The legacy prefill returns no cache: its last-position logits equal the new prefill's at
        // every prefix; the legacy decode, carrying its own cache token by token, equals the new
        // decode step for step (logits and every cache element).
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
            let (old, _) = eval(&trace_mpt_prefill(&cfg, n), &prompt[..n], &pos, &[]);
            close(&logits[(n - 1) * VOCAB..n * VOCAB], &old);
        }
        let legacy_decode = trace_mpt_decode_kv_masked(&cfg, CAP);
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
