//! SmolLM3 decoder tracer (card 135b, `specs/260-smollm3-tracer/spec.md`) for `HuggingFaceTB/SmolLM3-3B`,
//! checked against that checkpoint's `config.json` and `transformers`' `modeling_smollm3.py`/
//! `configuration_smollm3.py`.
//!
//! **NoPE (no positional embedding).** SmolLM3 is otherwise a plain Llama/Qwen2-shaped dense decoder (RMSNorm,
//! GQA, gated SwiGLU MLP, RoPE), but a periodic subset of layers skip RoPE and run attention on the unrotated
//! Q/K. The config carries this as `no_rope_layers`, a `[num_hidden_layers]` array of `0`/`1` ints (`1` = apply
//! RoPE, `0` = NoPE); `SmolLM3Attention` calls `apply_rotary_pos_emb` only `if self.use_rope`, with no other
//! change to the attention math. When a config omits `no_rope_layers`, `configuration_smollm3.py` generates it
//! from `no_rope_layer_interval` (default `4`): `[int((layer_idx + 1) % no_rope_layer_interval != 0) for
//! layer_idx in range(num_hidden_layers)]`, so every 4th layer (0-indexed 3, 7, 11, ...) is NoPE. The skip is a
//! per-layer tracing-time branch (`Smollm3Config::use_rope: Vec<bool>`): `rope`/`rope_prefill` are not called
//! for a NoPE layer, and there is no new graph-IR op or second attention definition.
//!
//! **Everything else is standard.** No bias anywhere (`attention_bias: false`/`mlp_bias: false`); pre-norm RMSNorm;
//! gated SwiGLU MLP (`down(silu(gate(x)) * up(x))`); GQA (16 query / 4 KV heads on the 3B checkpoint);
//! `tie_word_embeddings: true`; no embedding scaling, QK-norm or attention-logit softcap.
//! `layer_types`/`use_sliding_window`/`sliding_window` exist in the config schema but the 3B checkpoint disables
//! them (`"full_attention"`/`false`/`null`); not implemented (spec 260 "Out of scope").
//!
//! **No from-scratch loader.** Unlike BLOOM/MPT (`crate::bloom`/`crate::mpt`), SmolLM3's weight names and shapes
//! match what `crates/poot-llm/src/runner.rs`'s `Qwen2HfConfig`/`build_weights` already produces for
//! llama/qwen2/mistral (`model.embed_tokens.weight`, `model.layers.{li}.self_attn.{q,k,v,o}_proj.weight`,
//! `model.layers.{li}.{input_layernorm,post_attention_layernorm}.weight`, `model.layers.{li}.mlp.{gate,up,down}_proj.weight`,
//! `model.norm.weight`, tied `lm_head.weight`, one shared `rope.cos`/`rope.sin`). This tracer uses the same names,
//! so `crates/poot-llm/src/smollm3_load.rs` only parses `no_rope_layers`/`no_rope_layer_interval`.
//!
//! Structural template: `crate::qwen2::trace_prefill_impl`/`trace_decode_kv_masked_impl`, except for the NoPE
//! branch. Every op is an existing `poot_graph_ir::ops` composition.

use poot_graph_ir::ops::{
    attention_masked, attention_prefill, causal_mask_from_pos, linear, rmsnorm, rope, rope_prefill,
    swiglu,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

// ---------------------------------------------------------------------------------------------
// The `Model` family. The legacy tracers further down stay until the Runner stops calling them.
// ---------------------------------------------------------------------------------------------

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::ValidationOutputs;
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
use crate::names::gguf::GGUF_BASE;
use crate::names::{HF_BASE, family_weights, hf_base_shapes};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("smollm3");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "smollm3"),
        (ConfigSource::GgufArchitecture, "smollm3"),
    ],
    build,
    fixture,
};

/// smollm3's config facts: `SmolLM3Config`'s defaults (a `2e6` base, `eos_token_id` `128001`) and
/// llama.cpp's GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("smollm3"),
    hf_theta: 2_000_000.0,
    gguf_theta: 2_000_000.0,
    eps: 1e-6,
    eos: Some(128_001),
};

/// `SmolLM3Config`'s default `no_rope_layer_interval`, and llama.cpp's fixed GGUF step.
const DEFAULT_NO_ROPE_INTERVAL: usize = 4;

/// smollm3's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Smollm3Params {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    /// The attention of each layer: RoPE-rotated, or NoPE where the config says so.
    attention: Vec<AttentionParams>,
    layer: LayerParams,
    stack: StackParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
    chat: ChatFormat,
}

/// Which layers rotate: HF's `no_rope_layers` (one `0`/`1` per layer, `1` = RoPE), or the pattern
/// `no_rope_layer_interval` generates when it is absent.
fn rotates(fields: &Fields<'_>, layers: usize) -> Result<Vec<bool>, ModelError> {
    const LAYERS: &str = "no_rope_layers";
    const INTERVAL: &str = "no_rope_layer_interval";
    let RawConfig::HfJson { config, .. } = fields.raw() else {
        return Ok(Smollm3Config::no_rope_pattern(
            layers,
            DEFAULT_NO_ROPE_INTERVAL,
        ));
    };
    match config.get(LAYERS).filter(|v| !v.is_null()) {
        Some(list) => {
            let flags: Option<Vec<bool>> = list.as_array().and_then(|list| {
                list.iter()
                    .map(|v| match v.as_u64() {
                        Some(0) => Some(false),
                        Some(1) => Some(true),
                        _ => None,
                    })
                    .collect()
            });
            match flags {
                Some(flags) if flags.len() == layers => Ok(flags),
                _ => Err(fields.error(LAYERS, ConfigReason::WrongType)),
            }
        }
        None => {
            let interval = fields.count(Field::hf(INTERVAL), Some(DEFAULT_NO_ROPE_INTERVAL))?;
            Ok(Smollm3Config::no_rope_pattern(layers, interval))
        }
    }
}

impl Smollm3Params {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        for unsupported in ["attention_bias", "mlp_bias", "use_sliding_window"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let mut base = dims.attention(&fields, &SPEC, false)?;
        // A GGUF stores Q/K permuted for interleaved RoPE, as llama's does.
        if matches!(raw, RawConfig::Gguf(_)) {
            base = base.with_interleaved_rope();
        }
        let attention = rotates(&fields, dims.layers)?
            .into_iter()
            .map(|rope| if rope { base } else { base.without_rope() })
            .collect();
        let norm = dims.norm(&fields, &SPEC)?;
        let chat = ChatFormat::ChatML;
        Ok(Self {
            attention,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
            chat,
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
pub struct Smollm3 {
    params: Smollm3Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Smollm3Params::from_raw(raw)?;
    let rows = match raw.source() {
        ConfigSource::HfModelType => HF_BASE,
        ConfigSource::GgufArchitecture => GGUF_BASE,
    };
    let weights = family_weights(FAMILY, store, &[rows], p.layers, p.head_required)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
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
        prompt_bos: None,
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: NonZeroUsize::MIN,
        chat: p.chat,
    };
    Ok(Box::new(Smollm3 {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Smollm3 {
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
                |b, h| gated_ffn(b, h, &layer.ffn),
            )
        });
        Ok(step.finish(b, logits))
    }
}

/// The fixture's config: a tiny untied smollm3 of four layers, every second one NoPE, whose
/// projection widths are multiples of 32 (so a Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "smollm3",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "no_rope_layer_interval": 2,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_base_shapes(&config, false);
    Fixture::bf16(config, shapes)
}

/// A SmolLM3 decoder config; field names/values match `HuggingFaceTB/SmolLM3-3B`'s `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Smollm3Config {
    pub vocab: usize,
    pub hidden: usize,
    /// MLP intermediate width (`intermediate_size`; `11008` on the 3B checkpoint, not `4*hidden`).
    pub inter: usize,
    pub layers: usize,
    /// Query head count (`num_attention_heads`; 16 on the 3B checkpoint).
    pub n_heads: usize,
    /// KV head count (`num_key_value_heads`; 4 on the 3B checkpoint; `n_rep = n_heads/n_kv_heads`).
    pub n_kv_heads: usize,
    /// RMSNorm epsilon (`rms_norm_eps`; `1e-6` on the 3B checkpoint).
    pub eps: f32,
    /// RoPE table length (`max_position_embeddings`; `65536` on the 3B checkpoint). The rope base (`rope_theta`,
    /// `5000000.0` on the 3B checkpoint) is a loader concern: it only affects the values of `rope.cos`/`rope.sin`.
    pub max_pos: usize,
    /// Per-layer RoPE gate (`no_rope_layers`: `1` = apply RoPE -> `true`, `0` = NoPE -> `false`). Length must
    /// equal `layers`.
    pub use_rope: Vec<bool>,
}

impl Smollm3Config {
    /// `hidden / n_heads`; no released checkpoint has an explicit `head_dim` key.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }

    /// `configuration_smollm3.py`'s default `no_rope_layers` formula (for configs that omit the array): layer `li`
    /// (0-indexed) is NoPE iff `(li + 1) % interval == 0`. For the 3B checkpoint's `interval = 4`, layers 3, 7,
    /// 11, ... are NoPE (checked against the published `no_rope_layers` array).
    pub fn no_rope_pattern(layers: usize, interval: usize) -> Vec<bool> {
        (0..layers).map(|li| (li + 1) % interval != 0).collect()
    }
}

/// Trace a full-sequence SmolLM3 prefill forward (CPU-oracle coherence path). Mirrors
/// `crate::qwen2::trace_prefill_impl`'s shape and weight naming so the existing `Qwen2HfConfig`/`build_weights`
/// loader and `Runner::bind` work unchanged: embeddings -> `cfg.layers` pre-norm blocks (RMSNorm -> split Q/K/V ->
/// per-layer conditional RoPE (`Smollm3Config::use_rope`) -> causal GQA attention -> `o_proj` + residual ->
/// RMSNorm -> SwiGLU MLP + residual) -> `model.norm` -> tied `lm_head.weight`.
///
/// Expects these named constants bound at eval time: `model.embed_tokens.weight` `[vocab,hidden]`,
/// `rope.cos`/`rope.sin` `[max_pos,head_dim]` (one shared table; `rope_theta` is a loader concern),
/// per-layer `model.layers.{li}.{input_layernorm,post_attention_layernorm}.weight`, `self_attn.{q,k,v,o}_proj.weight`,
/// `mlp.{gate,up,down}_proj.weight` (no bias), `model.norm.weight`, and `lm_head.weight` (materialized dense even
/// for a tied checkpoint; `build_weights` handles the tie), plus the `mask.prefill` step input (card 550a)
/// `[1,1,l,l]`.
pub fn trace_smollm3_prefill(cfg: &Smollm3Config, seq_len: usize) -> Graph {
    assert_eq!(
        cfg.use_rope.len(),
        cfg.layers,
        "Smollm3Config::use_rope must have one entry per layer"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim(), cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();
    let l = seq_len;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    // this one-shot prefill always starts at position 0 (card 550).
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    let mask = causal_mask_from_pos(&b, pos, l, None);
    // One shared RoPE table for every layer with use_rope[li] == true (FR-002), not a per-layer table.
    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, d]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, d]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, H]
    let mut x = b.reshape(emb, vec![1, l, h]);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::f32(vec![h, q_dim]),
        );
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![q_dim, h]),
        );
        let q = linear(&b, normed, wq, None);
        let k = linear(&b, normed, wk, None);
        let v = linear(&b, normed, wv, None);

        let q = b.transpose(b.reshape(q, vec![1, l, hq, d]), vec![0, 2, 1, 3]); // [1,Hq,L,D]
        let k = b.transpose(b.reshape(k, vec![1, l, hkv, d]), vec![0, 2, 1, 3]); // [1,Hkv,L,D]
        let v = b.transpose(b.reshape(v, vec![1, l, hkv, d]), vec![0, 2, 1, 3]);

        // NoPE (see the module docs): apply RoPE to this layer's Q/K iff cfg.use_rope[li], otherwise use the raw
        // projected Q/K. The attention math is otherwise unchanged (FR-001).
        let (q, k) = if cfg.use_rope[li] {
            (
                rope_prefill(&b, q, cos, sin, l),
                rope_prefill(&b, k, cos, sin, l),
            )
        } else {
            (q, k)
        };

        let attn = attention_prefill(&b, q, k, v, n_rep, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,L,Hq,D]
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let wg = b.constant(
            &p("mlp.gate_proj.weight"),
            TensorType::f32(vec![h, cfg.inter]),
        );
        let wu = b.constant(
            &p("mlp.up_proj.weight"),
            TensorType::f32(vec![h, cfg.inter]),
        );
        let wd = b.constant(
            &p("mlp.down_proj.weight"),
            TensorType::f32(vec![cfg.inter, h]),
        );
        let gate = linear(&b, normed, wg, None);
        let up = linear(&b, normed, wu, None);
        let act = swiglu(&b, gate, up);
        let mlp = linear(&b, act, wd, None);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l); // only the last position feeds the LM head
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// Single-token fixed-KV masked decode (capture/replay shape of `crate::qwen2::trace_decode_kv_masked_impl`, weight
/// naming of `trace_smollm3_prefill`): one token in (`Slot::Token` at `Slot::Pos`), the new k/v scattered into each
/// layer's `[1, n_kv_heads, cap, head_dim]` cache, attention over the full cache with a host-filled causal
/// `Slot::Mask` (`[cap]`, one row broadcast over every head), and the same per-layer conditional RoPE as prefill.
///
/// Expects the same named constants as [`trace_smollm3_prefill`], whose mask is likewise a `Slot::Mask`
/// step input (the prefill's tagged `mask.prefill`, here the untagged `[cap]` row),
/// plus a `model.layers.{li}.kv.{k,v}_cache` state pair per layer (`[1, n_kv_heads, cap, head_dim]`,
/// zero-initialized by the caller).
pub fn trace_smollm3_decode_kv_masked(cfg: &Smollm3Config, cap: usize) -> Graph {
    assert_eq!(
        cfg.use_rope.len(),
        cfg.layers,
        "Smollm3Config::use_rope must have one entry per layer"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim(), cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, 1], DType::I32));
    let pos_slot = b.reshape(pos, vec![]);
    let mask = causal_mask_from_pos(&b, pos, cap, None); // [1,1,1,cap], broadcast over every head

    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, d]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, d]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather_scalar(embed, 0, token);
    let mut x = b.reshape(emb, vec![1, 1, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::f32(vec![h, q_dim]),
        );
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![q_dim, h]),
        );
        let q = linear(&b, normed, wq, None);
        let k = linear(&b, normed, wk, None);
        let v = linear(&b, normed, wv, None);

        let q = b.transpose(b.reshape(q, vec![1, 1, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, 1, hkv, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, 1, hkv, d]), vec![0, 2, 1, 3]);

        let (q, k) = if cfg.use_rope[li] {
            (
                rope(&b, q, cos, sin, pos_slot),
                rope(&b, k, cos, sin, pos_slot),
            )
        } else {
            (q, k)
        };

        let kcache = b.state_input(
            &p("kv.k_cache"),
            TensorType::f32(vec![1, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let vcache = b.state_input(
            &p("kv.v_cache"),
            TensorType::f32(vec![1, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let kcache_out = b.dynamic_update_slice_dyn(kcache, k, pos_slot, 2);
        let vcache_out = b.dynamic_update_slice_dyn(vcache, v, pos_slot, 2);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

        let attn = attention_masked(&b, q, kcache_out, vcache_out, n_rep, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, q_dim]);
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let wg = b.constant(
            &p("mlp.gate_proj.weight"),
            TensorType::f32(vec![h, cfg.inter]),
        );
        let wu = b.constant(
            &p("mlp.up_proj.weight"),
            TensorType::f32(vec![h, cfg.inter]),
        );
        let wd = b.constant(
            &p("mlp.down_proj.weight"),
            TensorType::f32(vec![cfg.inter, h]),
        );
        let gate = linear(&b, normed, wg, None);
        let up = linear(&b, normed, wu, None);
        let act = swiglu(&b, gate, up);
        let mlp = linear(&b, act, wd, None);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
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

    /// Small but non-degenerate dims with GQA (n_kv_heads < n_heads) and a `use_rope` pattern with both `true` and
    /// `false` layers (as in the interval=4 pattern, layer index interval-1 is the first NoPE layer).
    fn tiny_cfg() -> Smollm3Config {
        Smollm3Config {
            vocab: 12,
            hidden: 8,
            inter: 16,
            layers: 4,
            n_heads: 4,
            n_kv_heads: 2,
            eps: 1e-6,
            max_pos: 64,
            use_rope: Smollm3Config::no_rope_pattern(4, 2), // [true, false, true, false]
        }
    }

    #[test]
    fn smollm3_no_rope_pattern_matches_real_config_formula() {
        // FR-008: checked against the published `no_rope_layers` array (interval 4): index 3, 7, 11, ... (0-indexed)
        // is the first NoPE layer in each group of 4, not index 0; an off-by-one would silently rotate the wrong layers.
        let pattern = Smollm3Config::no_rope_pattern(8, 4);
        assert_eq!(
            pattern,
            vec![true, true, true, false, true, true, true, false]
        );
    }

    #[test]
    fn smollm3_prefill_validates_and_declares_one_shared_rope_table() {
        let cfg = tiny_cfg();
        let g = trace_smollm3_prefill(&cfg, 5);
        g.validate().expect("smollm3 prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

        // FR-002: exactly one rope.cos/rope.sin pair, not per-layer.
        let rope_consts: Vec<&str> = g
            .inputs
            .iter()
            .filter_map(|id| {
                let m = &g.values[*id];
                if let Storage::Const = m.storage {
                    let name = m.name.as_deref().unwrap_or("");
                    (name == "rope.cos" || name == "rope.sin").then_some(name)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            rope_consts.len(),
            2,
            "exactly one shared rope.cos/rope.sin pair"
        );

        // FR-004: no bias constant anywhere.
        for id in &g.inputs {
            let m = &g.values[*id];
            if let Storage::Const = m.storage {
                let name = m.name.as_deref().unwrap_or("");
                assert!(
                    !name.ends_with(".bias"),
                    "SmolLM3 has no bias, found {name}"
                );
            }
        }
        // Card 550: no `Slot::Mask` step input any more - the plain causal mask is a graph computation
        // over `Slot::Pos` and `iota`.
        assert!(
            !g.inputs
                .iter()
                .any(|id| matches!(&g.values[*id].storage, Storage::Slot(Slot::Mask))),
            "smollm3 prefill must have no Slot::Mask step input"
        );
        let pos_id = g
            .inputs
            .iter()
            .find(|id| matches!(&g.values[**id].storage, Storage::Slot(Slot::Pos)))
            .expect("Slot::Pos step input present");
        assert_eq!(g.values[*pos_id].aval.shape, vec![1, 5]);
    }

    #[test]
    fn smollm3_decode_kv_masked_validates_state_and_mask_shapes() {
        let cfg = tiny_cfg();
        let cap = 16;
        let g = trace_smollm3_decode_kv_masked(&cfg, cap);
        g.validate().expect("smollm3 decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
        assert_eq!(g.state.len(), 2 * cfg.layers);
        for (si, _so) in &g.state {
            // GQA: cache is sized by n_kv_heads, not n_heads.
            assert_eq!(
                g.aval(*si).shape,
                vec![1, cfg.n_kv_heads, cap, cfg.head_dim()]
            );
            assert_eq!(g.values[*si].storage, Storage::State);
        }

        // Card 550: no `Slot::Mask` step input any more.
        assert!(
            !g.inputs
                .iter()
                .any(|id| matches!(&g.values[*id].storage, Storage::Slot(Slot::Mask))),
            "smollm3 decode must have no Slot::Mask step input"
        );
        let pos_id = g
            .inputs
            .iter()
            .find(|id| matches!(&g.values[**id].storage, Storage::Slot(Slot::Pos)))
            .expect("Slot::Pos present");
        assert_eq!(g.values[*pos_id].aval.shape, vec![1, 1]);
    }

    // ---- CPU-oracle numerics (poot_eval): decomposition correctness against an independent Rust reference
    // forward pass ----

    mod cpu_oracle {
        use super::*;
        use std::collections::HashMap;

        use poot_test_util::fill;

        use poot_test_util::seed_of;

        fn weight(name: &str, n: usize, ln_gamma: bool) -> Vec<f32> {
            let raw = fill(n, seed_of(name));
            if ln_gamma {
                raw.iter().map(|v| 1.0 + v * 0.05).collect()
            } else {
                raw.iter().map(|v| v * 0.1).collect()
            }
        }

        /// All named constants [`trace_smollm3_prefill`] declares, keyed by exact graph const name.
        fn all_weights(cfg: &Smollm3Config) -> HashMap<String, Vec<f32>> {
            let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim(), cfg.n_heads, cfg.n_kv_heads);
            let (q_dim, kv_dim) = (hq * d, hkv * d);
            let mut w = HashMap::new();
            w.insert(
                "model.embed_tokens.weight".to_string(),
                weight("embed", cfg.vocab * h, false),
            );
            // rope tables: standard theta=10000 cos/sin, independent of poot_llm's rope_tables (this crate does not depend
            // on poot_llm); used only by layers with use_rope[li]=true.
            let theta = 10_000.0f32;
            let mut cos = vec![0.0f32; cfg.max_pos * d];
            let mut sin = vec![0.0f32; cfg.max_pos * d];
            for pos in 0..cfg.max_pos {
                for i in 0..d / 2 {
                    let freq = 1.0 / theta.powf(2.0 * i as f32 / d as f32);
                    let ang = pos as f32 * freq;
                    let (s, c) = ang.sin_cos();
                    cos[pos * d + i] = c;
                    cos[pos * d + i + d / 2] = c;
                    sin[pos * d + i] = s;
                    sin[pos * d + i + d / 2] = s;
                }
            }
            w.insert("rope.cos".to_string(), cos);
            w.insert("rope.sin".to_string(), sin);
            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                w.insert(p("input_layernorm.weight"), weight(&p("ln1w"), h, true));
                w.insert(
                    p("self_attn.q_proj.weight"),
                    weight(&p("qw"), h * q_dim, false),
                );
                w.insert(
                    p("self_attn.k_proj.weight"),
                    weight(&p("kw"), h * kv_dim, false),
                );
                w.insert(
                    p("self_attn.v_proj.weight"),
                    weight(&p("vw"), h * kv_dim, false),
                );
                w.insert(
                    p("self_attn.o_proj.weight"),
                    weight(&p("ow"), q_dim * h, false),
                );
                w.insert(
                    p("post_attention_layernorm.weight"),
                    weight(&p("ln2w"), h, true),
                );
                w.insert(
                    p("mlp.gate_proj.weight"),
                    weight(&p("gw"), h * cfg.inter, false),
                );
                w.insert(
                    p("mlp.up_proj.weight"),
                    weight(&p("uw"), h * cfg.inter, false),
                );
                w.insert(
                    p("mlp.down_proj.weight"),
                    weight(&p("dw"), cfg.inter * h, false),
                );
            }
            w.insert("model.norm.weight".to_string(), weight("ln_f_w", h, true));
            w.insert(
                "lm_head.weight".to_string(),
                weight("lm_head", h * cfg.vocab, false),
            );
            w
        }

        use poot_test_util::rmsnorm_ref;

        use poot_test_util::silu_ref;

        use poot_test_util::linear_ref;

        use poot_test_util::rope_ref;

        /// Independent reference forward pass for [`trace_smollm3_prefill`]: embed -> `cfg.layers` blocks (RMSNorm ->
        /// GQA split QKV -> per-layer conditional RoPE -> causal attention -> `o_proj` + residual -> RMSNorm -> SwiGLU MLP
        /// \+ residual) -> `model.norm` -> tied lm_head (last token only). A direct loop, as in `crate::bloom`'s
        /// `bloom_prefill_ref`/`crate::mpt`'s `mpt_prefill_ref`.
        fn smollm3_prefill_ref(
            cfg: &Smollm3Config,
            tokens: &[usize],
            w: &HashMap<String, Vec<f32>>,
        ) -> Vec<f32> {
            let (h, d, hq, hkv, l) = (
                cfg.hidden,
                cfg.head_dim(),
                cfg.n_heads,
                cfg.n_kv_heads,
                tokens.len(),
            );
            let n_rep = hq / hkv;
            let scale = 1.0 / (d as f32).sqrt();
            let embed = &w["model.embed_tokens.weight"];
            let cos = &w["rope.cos"];
            let sin = &w["rope.sin"];

            let mut x: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| embed[t * h..(t + 1) * h].to_vec())
                .collect();

            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                let ln1w = &w[&p("input_layernorm.weight")];
                let normed: Vec<Vec<f32>> = x
                    .iter()
                    .map(|row| rmsnorm_ref(row, ln1w, h, cfg.eps))
                    .collect();

                let qw = &w[&p("self_attn.q_proj.weight")];
                let kw = &w[&p("self_attn.k_proj.weight")];
                let vw = &w[&p("self_attn.v_proj.weight")];
                let mut q: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, qw, h, hq * d))
                    .collect();
                let mut k: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, kw, h, hkv * d))
                    .collect();
                let v: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, vw, h, hkv * d))
                    .collect();

                if cfg.use_rope[li] {
                    for (pos, qrow) in q.iter_mut().enumerate() {
                        for hh in 0..hq {
                            let rotated = rope_ref(&qrow[hh * d..(hh + 1) * d], cos, sin, pos, d);
                            qrow[hh * d..(hh + 1) * d].copy_from_slice(&rotated);
                        }
                    }
                    for (pos, krow) in k.iter_mut().enumerate() {
                        for hh in 0..hkv {
                            let rotated = rope_ref(&krow[hh * d..(hh + 1) * d], cos, sin, pos, d);
                            krow[hh * d..(hh + 1) * d].copy_from_slice(&rotated);
                        }
                    }
                }
                // else: NoPE layer - q/k stay exactly as projected.

                // per-head causal attention with GQA head repeat (kv head hh serves query heads [hh*n_rep, (hh+1)*n_rep)),
                // a direct loop independent of attention_prefill's matmul/transpose/reduce decomposition.
                let mut attn_out = vec![vec![0.0f32; h]; l];
                for qh in 0..hq {
                    let kh = qh / n_rep;
                    for i in 0..l {
                        let mut scores = vec![0.0f32; i + 1];
                        for (j, sc) in scores.iter_mut().enumerate() {
                            let mut s = 0.0f32;
                            for dd in 0..d {
                                s += q[i][qh * d + dd] * k[j][kh * d + dd];
                            }
                            *sc = s * scale;
                        }
                        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut denom = 0.0f32;
                        let mut e = vec![0.0f32; scores.len()];
                        for (j, sc) in scores.iter().enumerate() {
                            e[j] = (sc - m).exp();
                            denom += e[j];
                        }
                        for dd in 0..d {
                            let mut acc = 0.0f32;
                            for (j, ej) in e.iter().enumerate() {
                                acc += (ej / denom) * v[j][kh * d + dd];
                            }
                            attn_out[i][qh * d + dd] = acc;
                        }
                    }
                }

                let ow = &w[&p("self_attn.o_proj.weight")];
                for i in 0..l {
                    let proj = linear_ref(&attn_out[i], ow, hq * d, h);
                    for c in 0..h {
                        x[i][c] += proj[c];
                    }
                }

                let ln2w = &w[&p("post_attention_layernorm.weight")];
                let gw = &w[&p("mlp.gate_proj.weight")];
                let uw = &w[&p("mlp.up_proj.weight")];
                let dw = &w[&p("mlp.down_proj.weight")];
                for row in x.iter_mut().take(l) {
                    let normed = rmsnorm_ref(row, ln2w, h, cfg.eps);
                    let gate = linear_ref(&normed, gw, h, cfg.inter);
                    let up = linear_ref(&normed, uw, h, cfg.inter);
                    let act: Vec<f32> = gate
                        .iter()
                        .zip(up.iter())
                        .map(|(&g, &u)| silu_ref(g) * u)
                        .collect();
                    let down = linear_ref(&act, dw, cfg.inter, h);
                    for (c, dv) in down.iter().enumerate() {
                        row[c] += dv;
                    }
                }
            }

            let ln_f_w = &w["model.norm.weight"];
            let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, cfg.eps);
            let lm_head = &w["lm_head.weight"];
            linear_ref(&last, lm_head, h, cfg.vocab)
        }

        fn eval_smollm3_prefill(
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
                        let data = weights
                            .get(name)
                            .unwrap_or_else(|| panic!("no weight bound for {name}"));
                        poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data.clone())
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

        /// `trace_smollm3_prefill` matches an independent reference forward pass (hand loops) within f32 tolerance.
        /// `tiny_cfg`'s `use_rope = [true, false, true, false]` exercises both the RoPE and NoPE branches.
        #[test]
        fn smollm3_prefill_matches_hand_rolled_reference() {
            let cfg = tiny_cfg();
            let tokens = [3usize, 7, 1, 9];
            let weights = all_weights(&cfg);

            let g = trace_smollm3_prefill(&cfg, tokens.len());
            let got = eval_smollm3_prefill(&g, &tokens, &weights);

            let want = smollm3_prefill_ref(&cfg, &tokens, &weights);
            assert_eq!(got.shape(), vec![1, 1, cfg.vocab]);
            poot_test_util::assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
            assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// Sanity: perturbing one entry of a mid-layer MLP down-projection weight changes the output, so the tracer is
        /// not ignoring that constant.
        #[test]
        fn smollm3_prefill_output_is_sensitive_to_a_perturbed_weight() {
            let cfg = tiny_cfg();
            let tokens = [2usize, 5, 8];
            let mut weights = all_weights(&cfg);
            let g = trace_smollm3_prefill(&cfg, tokens.len());
            let base = eval_smollm3_prefill(&g, &tokens, &weights);

            let key = "model.layers.1.mlp.down_proj.weight".to_string();
            weights.get_mut(&key).unwrap()[0] += 5.0;
            let perturbed = eval_smollm3_prefill(&g, &tokens, &weights);

            let max_diff =
                poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
            assert!(
                max_diff > 1e-3,
                "perturbing {key} should change the output; max_diff={max_diff:.2e}"
            );
        }

        /// Sanity: perturbing the rope table changes the output of a config with at least one RoPE layer (tiny_cfg has
        /// layers 0/2 as RoPE), so rope.cos/rope.sin are read.
        #[test]
        fn smollm3_prefill_output_is_sensitive_to_a_perturbed_rope_table() {
            let cfg = tiny_cfg();
            let tokens = [2usize, 5, 8];
            let mut weights = all_weights(&cfg);
            let g = trace_smollm3_prefill(&cfg, tokens.len());
            let base = eval_smollm3_prefill(&g, &tokens, &weights);

            weights.get_mut("rope.cos").unwrap()[0] += 0.5;
            let perturbed = eval_smollm3_prefill(&g, &tokens, &weights);

            let max_diff =
                poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
            assert!(
                max_diff > 1e-6,
                "perturbing rope.cos should change the output for a config with RoPE layers; \
                 max_diff={max_diff:.2e}"
            );
        }
    }
}

#[cfg(test)]
mod family_tests {
    use poot_graph_ir::rope_table::{RopeFlavor, rope_tables};
    use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, Legacy, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_legacy, assert_missing_weight_is_named, build as built, close, close_all,
        f32_entry, gguf_dense_kvs, gguf_name_with_head, gguf_tensor, legacy_eval, q8_0_store,
        read_back_owned, step_shape, values,
    };
    use crate::model::LogitRows;
    use crate::registry::Registry;

    const VOCAB: usize = 48;

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn legacy_cfg(interval: usize) -> Smollm3Config {
        Smollm3Config {
            vocab: VOCAB,
            hidden: 64,
            inter: 96,
            layers: 4,
            n_heads: 4,
            n_kv_heads: 2,
            eps: 1e-6,
            max_pos: 64,
            use_rope: Smollm3Config::no_rope_pattern(4, interval),
        }
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// SC-002: smollm3's body (NoPE on every `interval`-th layer, per the config) equals the
    /// legacy prefill and decode tracers whose NoPE pattern is `legacy_interval` (ADR-0101 tiers
    /// 1 and 2). Mutation: change the body's NoPE interval; red.
    fn assert_matches_legacy_interval(config: &serde_json::Value, legacy_interval: usize) {
        let fx = fixture();
        let model = model_of(config, &fx.store);
        let cfg = legacy_cfg(legacy_interval);
        let tables = rope_tables(16, 64, 10_000.0, &RopeFlavor::Plain, None);
        let host = [("rope.cos", tables.cos), ("rope.sin", tables.sin)];
        let eval = |g: &Graph, tokens: &[i32], pos: &[i32], state: &[Vec<f32>]| {
            legacy_eval(
                g,
                &fx.store,
                &host,
                &["model.embed_tokens.weight"],
                tokens,
                pos,
                state,
            )
        };
        assert_matches_legacy(
            &*model,
            &fx.store,
            &Legacy {
                prefill: &|n| trace_smollm3_prefill(&cfg, n),
                decode: &trace_smollm3_decode_kv_masked(&cfg, CAP),
                eval: &eval,
            },
        );
    }

    #[test]
    fn smollm3_matches_the_legacy_tracers_with_nope_layers() {
        assert_matches_legacy_interval(&fixture_config(), 2);
    }

    /// The NoPE pattern is a real input: an explicit `no_rope_layers` array equals the interval it
    /// spells, and a different interval is a different model.
    #[test]
    fn the_nope_pattern_comes_from_the_config() {
        let mut explicit = fixture_config();
        explicit["no_rope_layers"] = json!([1, 0, 1, 0]);
        explicit
            .as_object_mut()
            .unwrap()
            .remove("no_rope_layer_interval");
        assert_matches_legacy_interval(&explicit, 2);
        let mut other = fixture_config();
        other["no_rope_layer_interval"] = json!(3);
        let fx = fixture();
        let run = |config: &serde_json::Value| {
            let model = model_of(config, &fx.store);
            run_step(
                &*model,
                &fx.store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40, 8, 25],
                0,
                &[],
            )
            .0
        };
        assert_ne!(run(&other), run(&fixture_config()));
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
    /// missing-weight error naming it, before any trace.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.3.self_attn.k_proj.weight",
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
    fn a_config_smollm3_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "use_sliding_window",
                json!(true),
                "use_sliding_window",
                ConfigReason::Unsupported,
            ),
            (
                "no_rope_layer_interval",
                json!(0),
                "no_rope_layer_interval",
                ConfigReason::Zero,
            ),
            (
                "no_rope_layers",
                json!([1, 0]),
                "no_rope_layers",
                ConfigReason::WrongType,
            ),
            (
                "no_rope_layers",
                json!([1, 0, 2, 1]),
                "no_rope_layers",
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

    #[test]
    fn a_missing_eos_is_the_reference_default() {
        let fx = fixture();
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        assert_eq!(
            model_of(&config, &fx.store).config().eos,
            BTreeSet::from([128_001])
        );
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

    /// A Q8_0 smollm3 GGUF (Q/K payloads in llama.cpp's permuted order) and the HF F32 store
    /// holding the same values un-permuted.
    fn gguf_case() -> (GgufIndex, WeightStore, WeightStore) {
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
        let kvs = gguf_dense_kvs("smollm3", &fx.config);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004 (R-562-3): a Q8_0 smollm3 GGUF resolves by its architecture key, builds packed
    /// projections traced as `PackedDequant`, applies llama.cpp's fixed NoPE step (every fourth
    /// layer) and computes what the HF body computes over the same values un-permuted. Mutation:
    /// drop the GGUF key (`Unregistered`), or trace the GGUF rows half-split (logits diverge).
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body_over_unpermuted_rows() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 28);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        // llama.cpp's step is 4: layer 3 is NoPE, layer 1 rotates.
        let mut config = fixture_config();
        config["no_rope_layer_interval"] = json!(4);
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
        // Every V cache (odd entries) is rotation-free and un-permuted: equal as stored.
        let v = |s: &[Vec<f32>]| s.iter().skip(1).step_by(2).cloned().collect::<Vec<_>>();
        close_all(&v(&state), &v(&want_state));
    }

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = GgufIndex::from_bytes(&write_gguf(
            &[
                ("general.architecture", GgufValue::Str("smollm3".into())),
                ("smollm3.embedding_length", GgufValue::U32(64)),
                ("smollm3.feed_forward_length", GgufValue::U32(96)),
                ("smollm3.block_count", GgufValue::U32(2)),
                ("smollm3.attention.head_count", GgufValue::U32(0)),
                ("smollm3.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        ))
        .unwrap();
        let err = Smollm3Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "smollm3.attention.head_count",
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
