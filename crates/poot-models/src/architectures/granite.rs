//! Granite tracer (spec 018). A pre-norm transformer block (GQA, RMSNorm, RoPE) plus Granite's four scalar
//! multipliers (embedding, attention, residual, logits). The MLP is a dense swiglu (`model_type: granite`) or a
//! top-k expert mixture (`model_type: granitemoe`, [`poot_graph_ir::ops::moe`]). Reuses [`Qwen2Config`] for the
//! shared dims.

use poot_graph_ir::ops::{
    attention_masked, attention_prefill, causal_mask_from_pos, linear, moe, rmsnorm, rope,
    rope_prefill, swiglu,
};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, ValidationOutputs};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_tensor::DType;

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
use crate::qwen2::Qwen2Config;
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

// The dense `granite` family behind [`Model`]: the standard stack and layer with Granite's four scalar
// multipliers as options of the shared components. The tracers below this section are the legacy
// ones the Runner still calls until Card 737; the tests use them as the independent reference.

pub const FAMILY: FamilyKey = FamilyKey::new("granite");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "granite"),
        (ConfigSource::GgufArchitecture, "granite"),
    ],
    build,
    fixture,
};

/// granite's config facts: `GraniteConfig`'s defaults (`rms_norm_eps` 1e-6, `rope_theta` 1e4, eos 2)
/// and llama.cpp's GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("granite"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000.0,
    eps: 1e-6,
    eos: Some(2),
};

const EMBEDDING_MULTIPLIER: Field = Field::new("embedding_multiplier", "granite.embedding_scale");
const ATTENTION_MULTIPLIER: Field = Field::new("attention_multiplier", "granite.attention.scale");
const RESIDUAL_MULTIPLIER: Field = Field::new("residual_multiplier", "granite.residual_scale");
const LOGITS_SCALING: Field = Field::new("logits_scaling", "granite.logit_scale");

/// A Granite scalar: HF's config class defaults it to 1; a GGUF must carry it.
fn scalar(fields: &Fields<'_>, field: Field) -> Result<f32, ModelError> {
    match (fields.opt_f32(field)?, fields.raw()) {
        (Some(v), _) => Ok(v),
        (None, RawConfig::HfJson { .. }) => Ok(1.0),
        (None, RawConfig::Gguf(_)) => Err(fields.error(fields.key(field), ConfigReason::Missing)),
    }
}

/// granite's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct GraniteConfig {
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

impl GraniteConfig {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        for unsupported in ["attention_bias", "mlp_bias"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        // GraniteConfig ties the head to the embedding unless told otherwise.
        let head_required = match raw {
            RawConfig::HfJson { .. } => !fields.flag(Field::hf("tie_word_embeddings"), true)?,
            RawConfig::Gguf(_) => false,
        };
        let param = |e: crate::components::standard::ParamError| e.for_family(FAMILY);
        let norm = dims.norm(&fields, &SPEC)?;
        let attention = dims
            .attention(&fields, &SPEC, false)?
            .with_scale(scalar(&fields, ATTENTION_MULTIPLIER)?)
            .map_err(param)?;
        let layer = LayerParams::new(norm)
            .with_residual_scale(scalar(&fields, RESIDUAL_MULTIPLIER)?)
            .map_err(param)?;
        let stack = StackParams::new(norm)
            .with_embed_scale(scalar(&fields, EMBEDDING_MULTIPLIER)?)
            .map_err(param)?
            .with_logit_divisor(scalar(&fields, LOGITS_SCALING)?)
            .map_err(param)?;
        Ok(Self {
            attention,
            layer,
            stack,
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            head_required,
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
pub struct Granite {
    params: GraniteConfig,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = GraniteConfig::from_raw(raw)?;
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
        chat: ChatFormat::Granite,
    };
    Ok(Box::new(Granite {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Granite {
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

/// The fixture's config: a tiny untied granite with non-trivial multipliers, whose projection widths
/// are multiples of 32 (so a Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "granite",
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
        "embedding_multiplier": 2.0,
        "attention_multiplier": 0.2,
        "residual_multiplier": 0.5,
        "logits_scaling": 2.0,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_base_shapes(&config, false);
    Fixture::bf16(config, shapes)
}

// ---- Legacy tracers (kept for the Runner until Card 737; the tests' independent reference) ----

/// The MoE shape for a `granitemoe` model. `None` on [`GraniteParams`] means a dense (`granite`) MLP.
#[derive(Clone, Copy, Debug)]
pub struct MoeShape {
    /// Total experts per layer (`num_local_experts`).
    pub n_experts: usize,
    /// Experts selected per token (`num_experts_per_tok`); softmax is over exactly these.
    pub top_k: usize,
    /// Per-expert FFN intermediate size (`intermediate_size`).
    pub inter: usize,
}

/// Granite parameters alongside a [`Qwen2Config`]: the four scalar multipliers plus the MLP kind.
#[derive(Clone, Copy, Debug)]
pub struct GraniteParams {
    /// `Some` for `granitemoe`; `None` for dense `granite`.
    pub moe: Option<MoeShape>,
    /// Token embeddings are multiplied by this (`embedding_multiplier`) in-graph: the embed table is tied to the
    /// output projection, so a load-time scale would also scale the logits.
    pub embed_mult: f32,
    /// Attention softmax scale (`attention_multiplier`), replacing `1/sqrt(head_dim)`.
    pub attn_mult: f32,
    /// Each sublayer output is multiplied by this (`residual_multiplier`) before the residual add.
    pub residual_mult: f32,
    /// Final logits are divided by this (`logits_scaling`).
    pub logits_scale: f32,
}

/// Trace a full-sequence Granite prefill forward (the CPU coherence path).
pub fn trace_granite_prefill(cfg: Qwen2Config, gp: GraniteParams, seq_len: usize) -> Graph {
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let l = seq_len;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    // this one-shot prefill always starts at position 0 (card 550).
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    let mask = causal_mask_from_pos(&b, pos, l, cfg.sliding_window);

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, hidden]
    let x0 = b.reshape(emb, vec![1, l, h]);
    let mut x = b.binary_scalar(BinOp::Mul, x0, Builder::f32(gp.embed_mult));

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

        let q = b.reshape(q, vec![1, l, hq, d]);
        let k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope_prefill(&b, q, cos, sin, l);
        let k = rope_prefill(&b, k, cos, sin, l);

        let attn = attention_prefill(&b, q, k, v, n_rep, gp.attn_mult, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, None);
        let attn = b.binary_scalar(BinOp::Mul, attn, Builder::f32(gp.residual_mult));
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = match gp.moe {
            Some(ms) => {
                // MoE expert weights (pre-transposed to [in, out]): router [H, E], input_linear (gate||up) [E, H, 2I],
                // output_linear [E, I, H].
                let router = b.constant(
                    &p("block_sparse_moe.router.layer.weight"),
                    TensorType::f32(vec![h, ms.n_experts]),
                );
                let w_in = b.constant(
                    &p("block_sparse_moe.input_linear.weight"),
                    TensorType::f32(vec![ms.n_experts, h, 2 * ms.inter]),
                );
                let w_out = b.constant(
                    &p("block_sparse_moe.output_linear.weight"),
                    TensorType::f32(vec![ms.n_experts, ms.inter, h]),
                );
                moe(
                    &b,
                    normed,
                    router,
                    w_in,
                    w_out,
                    ms.n_experts,
                    ms.top_k,
                    ms.inter,
                )
            }
            None => {
                // Dense swiglu MLP: gate/up/down projections.
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
                linear(&b, act, wd, None)
            }
        };
        let m = b.binary_scalar(BinOp::Mul, m, Builder::f32(gp.residual_mult));
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    let scaled = b.binary_scalar(BinOp::Mul, logits, Builder::f32(1.0 / gp.logits_scale));
    b.finish(scaled)
}

/// Granite batched prefill with KV-cache writes: [`trace_granite_prefill`] on the cache-writing skeleton of
/// [`crate::qwen2::trace_prefill_kv`]. One Q=N forward fills each layer's `[1,Hkv,cap,D]` K/V cache for
/// positions [0,N) (a `dynamic_update_slice` of the `[1,Hkv,N,D]` block at slot 0, axis 2), with the state
/// names/order of [`trace_granite_decode_kv_masked`] so the caches feed the decode loop from pos=N.
pub fn trace_granite_prefill_kv(
    cfg: Qwen2Config,
    gp: GraniteParams,
    n: usize,
    cap: usize,
) -> Graph {
    assert!(
        cap >= n,
        "cache capacity {cap} must be at least the prompt length {n}"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let l = n;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    // this one-shot prefill always starts at position 0 (card 550).
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    let mask = causal_mask_from_pos(&b, pos, l, cfg.sliding_window);

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens);
    let x0 = b.reshape(emb, vec![1, l, h]);
    let mut x = b.binary_scalar(BinOp::Mul, x0, Builder::f32(gp.embed_mult));

    let mut state: Vec<(poot_graph_ir::Traced, poot_graph_ir::Traced)> =
        Vec::with_capacity(2 * cfg.layers);

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

        let q = b.reshape(q, vec![1, l, hq, d]);
        let k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope_prefill(&b, q, cos, sin, l);
        let k = rope_prefill(&b, k, cos, sin, l);

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
        let kcache_out = b.dynamic_update_slice(kcache, k, 0, 2);
        let vcache_out = b.dynamic_update_slice(vcache, v, 0, 2);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

        let attn = attention_prefill(&b, q, k, v, n_rep, gp.attn_mult, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, None);
        let attn = b.binary_scalar(BinOp::Mul, attn, Builder::f32(gp.residual_mult));
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = match gp.moe {
            Some(ms) => {
                let router = b.constant(
                    &p("block_sparse_moe.router.layer.weight"),
                    TensorType::f32(vec![h, ms.n_experts]),
                );
                let w_in = b.constant(
                    &p("block_sparse_moe.input_linear.weight"),
                    TensorType::f32(vec![ms.n_experts, h, 2 * ms.inter]),
                );
                let w_out = b.constant(
                    &p("block_sparse_moe.output_linear.weight"),
                    TensorType::f32(vec![ms.n_experts, ms.inter, h]),
                );
                moe(
                    &b,
                    normed,
                    router,
                    w_in,
                    w_out,
                    ms.n_experts,
                    ms.top_k,
                    ms.inter,
                )
            }
            None => {
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
                linear(&b, act, wd, None)
            }
        };
        let m = b.binary_scalar(BinOp::Mul, m, Builder::f32(gp.residual_mult));
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    let scaled = b.binary_scalar(BinOp::Mul, logits, Builder::f32(1.0 / gp.logits_scale));
    b.finish_with_state(scaled, &state)
}

/// Granite shared-pool paged prefill (spec 249): the granite/granitemoe equivalent of
/// [`crate::qwen2::trace_prefill_kv_shared_pool_ext`]. Delegates its `Slot::Mask`/`Slot::SeqLen` step
/// inputs to [`crate::moe_prefill`], shared with the out-of-scope MoE/hybrid families (card 550's Out of
/// scope); it is not converted to the in-graph mask here and moves with those families in M5 (Card 567).
/// Delegates write/read mechanics and block scaffolding to
/// [`crate::moe_prefill::trace_moe_prefill_kv_shared_pool`], closing over Granite's scalar multipliers and the
/// same dense-or-MoE FFN as [`trace_granite_prefill_kv`] (identical weight names/shapes). `pool` is the shared
/// physical pool size (`pool >= n`).
pub fn trace_granite_prefill_kv_shared_pool(
    cfg: Qwen2Config,
    gp: GraniteParams,
    n: usize,
    pool: usize,
) -> Graph {
    crate::moe_prefill::trace_moe_prefill_kv_shared_pool(
        cfg,
        false, // granite: no qk-norm
        gp.embed_mult,
        gp.attn_mult,
        gp.residual_mult,
        1.0 / gp.logits_scale,
        n,
        pool,
        move |b, normed, li| {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let h = cfg.hidden;
            match gp.moe {
                Some(ms) => {
                    let router = b.constant(
                        &p("block_sparse_moe.router.layer.weight"),
                        TensorType::f32(vec![h, ms.n_experts]),
                    );
                    let w_in = b.constant(
                        &p("block_sparse_moe.input_linear.weight"),
                        TensorType::f32(vec![ms.n_experts, h, 2 * ms.inter]),
                    );
                    let w_out = b.constant(
                        &p("block_sparse_moe.output_linear.weight"),
                        TensorType::f32(vec![ms.n_experts, ms.inter, h]),
                    );
                    moe(
                        b,
                        normed,
                        router,
                        w_in,
                        w_out,
                        ms.n_experts,
                        ms.top_k,
                        ms.inter,
                    )
                }
                None => {
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
                    let gate = linear(b, normed, wg, None);
                    let up = linear(b, normed, wu, None);
                    let act = swiglu(b, gate, up);
                    linear(b, act, wd, None)
                }
            }
        },
    )
}

/// Granite batched shared-pool decode (spec 249): the granite/granitemoe equivalent of
/// [`crate::qwen2::trace_decode_kv_masked_batched_shared_pool_ext`], the decode half of
/// [`trace_granite_prefill_kv_shared_pool`]. Not converted to the in-graph mask (see that function's doc
/// comment: shared MoE/hybrid machinery, out of card 550's scope). Delegates to
/// [`crate::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool`], closing over Granite's scalar
/// multipliers and the same dense-or-MoE FFN as [`trace_granite_decode_kv_masked`] (identical weight
/// names/shapes). `cap` is each row's logical capacity; `batch` the number of concurrently decoding rows;
/// `pool_slots` the shared physical pool size.
pub fn trace_granite_decode_kv_masked_batched_shared_pool(
    cfg: Qwen2Config,
    gp: GraniteParams,
    cap: usize,
    batch: usize,
    pool_slots: usize,
) -> Graph {
    crate::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool(
        cfg,
        false, // granite: no qk-norm
        gp.embed_mult,
        gp.attn_mult,
        gp.residual_mult,
        1.0 / gp.logits_scale,
        cap,
        batch,
        pool_slots,
        move |b, normed, li| {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let h = cfg.hidden;
            match gp.moe {
                Some(ms) => {
                    let router = b.constant(
                        &p("block_sparse_moe.router.layer.weight"),
                        TensorType::f32(vec![h, ms.n_experts]),
                    );
                    let w_in = b.constant(
                        &p("block_sparse_moe.input_linear.weight"),
                        TensorType::f32(vec![ms.n_experts, h, 2 * ms.inter]),
                    );
                    let w_out = b.constant(
                        &p("block_sparse_moe.output_linear.weight"),
                        TensorType::f32(vec![ms.n_experts, ms.inter, h]),
                    );
                    moe(
                        b,
                        normed,
                        router,
                        w_in,
                        w_out,
                        ms.n_experts,
                        ms.top_k,
                        ms.inter,
                    )
                }
                None => {
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
                    let gate = linear(b, normed, wg, None);
                    let up = linear(b, normed, wu, None);
                    let act = swiglu(b, gate, up);
                    linear(b, act, wd, None)
                }
            }
        },
    )
}

/// Granite single-token fixed-KV masked decode, the decode analog of [`trace_granite_prefill`] on the
/// [`crate::qwen2::trace_decode_kv_masked`] skeleton: embedding multiplier, `attn_mult` as attention scale,
/// `residual_mult` on each sublayer output, dense-swiglu or MoE MLP per `gp.moe`, final logits / `logits_scale`.
/// No qk-norm, no qkv bias, two norms (input + post_attention), single rope base.
pub fn trace_granite_decode_kv_masked(cfg: Qwen2Config, gp: GraniteParams, cap: usize) -> Graph {
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, 1], DType::I32));
    let pos_slot = b.reshape(pos, vec![]);
    let mask = causal_mask_from_pos(&b, pos, cap, cfg.sliding_window); // [1,1,1,cap]

    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather_scalar(embed, 0, token);
    let x0 = b.reshape(x0, vec![1, 1, h]);
    let mut x = b.binary_scalar(BinOp::Mul, x0, Builder::f32(gp.embed_mult));

    let mut state: Vec<(poot_graph_ir::Traced, poot_graph_ir::Traced)> =
        Vec::with_capacity(2 * cfg.layers);

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

        let q = b.reshape(q, vec![1, 1, hq, d]);
        let k = b.reshape(k, vec![1, 1, hkv, d]);
        let v = b.reshape(v, vec![1, 1, hkv, d]);
        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope(&b, q, cos, sin, pos_slot);
        let k = rope(&b, k, cos, sin, pos_slot);

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

        let attn = attention_masked(&b, q, kcache_out, vcache_out, n_rep, gp.attn_mult, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, q_dim]);
        let attn = linear(&b, attn, wo, None);
        let attn = b.binary_scalar(BinOp::Mul, attn, Builder::f32(gp.residual_mult));
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = match gp.moe {
            Some(ms) => {
                let router = b.constant(
                    &p("block_sparse_moe.router.layer.weight"),
                    TensorType::f32(vec![h, ms.n_experts]),
                );
                let w_in = b.constant(
                    &p("block_sparse_moe.input_linear.weight"),
                    TensorType::f32(vec![ms.n_experts, h, 2 * ms.inter]),
                );
                let w_out = b.constant(
                    &p("block_sparse_moe.output_linear.weight"),
                    TensorType::f32(vec![ms.n_experts, ms.inter, h]),
                );
                moe(
                    &b,
                    normed,
                    router,
                    w_in,
                    w_out,
                    ms.n_experts,
                    ms.top_k,
                    ms.inter,
                )
            }
            None => {
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
                linear(&b, act, wd, None)
            }
        };
        let m = b.binary_scalar(BinOp::Mul, m, Builder::f32(gp.residual_mult));
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    let scaled = b.binary_scalar(BinOp::Mul, logits, Builder::f32(1.0 / gp.logits_scale));
    b.finish_with_state(scaled, &state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Storage;

    fn params(moe: Option<MoeShape>) -> GraniteParams {
        GraniteParams {
            moe,
            embed_mult: 12.0,
            attn_mult: 0.015625,
            residual_mult: 0.22,
            logits_scale: 6.0,
        }
    }

    #[test]
    fn granite_decode_kv_masked_validates_dense_and_moe() {
        let cfg = Qwen2Config::qwen2_0_5b();
        // Dense granite and a granite-MoE shape; both decode graphs must be well-formed.
        let cases = [
            params(None),
            params(Some(MoeShape {
                n_experts: 8,
                top_k: 2,
                inter: cfg.inter,
            })),
        ];
        for gp in cases {
            let g = trace_granite_decode_kv_masked(cfg, gp, 16);
            g.validate()
                .expect("granite kv decode graph should validate");
            assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
            assert_eq!(g.state.len(), 2 * cfg.layers);
            for (si, _so) in &g.state {
                assert_eq!(g.aval(*si).shape, vec![1, cfg.n_kv_heads, 16, cfg.head_dim]);
                assert_eq!(g.values[*si].storage, Storage::State);
            }
        }
    }

    #[test]
    fn granite_prefill_kv_validates_dense_and_moe() {
        let cfg = Qwen2Config::qwen2_0_5b();
        // The batched prefill must carry the same state pairs (names/order/shapes) as the decode graph. n=5, cap=16.
        let (n, cap) = (5, 16);
        let cases = [
            params(None),
            params(Some(MoeShape {
                n_experts: 8,
                top_k: 2,
                inter: cfg.inter,
            })),
        ];
        for gp in cases {
            let g = trace_granite_prefill_kv(cfg, gp, n, cap);
            g.validate()
                .expect("granite kv prefill graph should validate");
            assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
            assert_eq!(g.state.len(), 2 * cfg.layers);
            for (si, _so) in &g.state {
                assert_eq!(
                    g.aval(*si).shape,
                    vec![1, cfg.n_kv_heads, cap, cfg.head_dim]
                );
                assert_eq!(g.values[*si].storage, Storage::State);
            }
        }
    }

    /// `optimize()` flash-fuses dense Granite's prefill like qwen2/gemma3 (the raw-graph validation test above
    /// does not check that), including with the extra `gp.attn_mult` scale. MoE Granite is out of scope: routed
    /// expert prefill through the full fusion pipeline has hung on AMDGCN (card 186).
    #[test]
    fn optimize_fuses_flash_prefill_dense_no_materialized_scores() {
        let cfg = Qwen2Config::qwen2_0_5b();
        let gp = params(None); // dense granite (no MoE) - see the MoE caution above.
        let n = 2048usize;
        let g = trace_granite_prefill_kv(cfg, gp, n, n);
        let opt = crate::test_support::optimize(&g);

        let flash_prefill = opt
            .eqns
            .iter()
            .filter(|e| matches!(e.op, poot_graph_ir::OpKind::FlashAttentionPrefill { .. }))
            .count();
        assert_eq!(
            flash_prefill, cfg.layers,
            "expected one fused flash-prefill op per layer"
        );

        // No MatMul may produce a full [.,.,L,L] score matrix; flash attention never materializes it.
        for e in &opt.eqns {
            if matches!(e.op, poot_graph_ir::OpKind::MatMul) {
                // Shape-aware, not a size threshold (Card 557: the projections are plain `MatMul`s too,
                // and an `[1, L, inter]` MLP output can exceed L*L elements): a rank-4 `[., Hq, L, L']`
                // output with both trailing dims >= n is the materialized softmax(QK^T) score matrix.
                let shape = &opt.aval(e.out).shape;
                let r = shape.len();
                assert!(
                    !(r == 4 && shape[r - 2] >= n && shape[r - 1] >= n),
                    "found a materialized N^2-scale MatMul the flash fusion should have replaced: \
                     shape={shape:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod family_tests {
    use poot_graph_ir::rope_table::{RopeFlavor, rope_tables};
    use poot_load::gguf::GgufValue;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, Legacy, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_legacy, assert_missing_weight_is_named, build as built, close, close_all,
        f32_entry, gguf_dense_kvs, gguf_name_with_head, gguf_tensor, legacy_eval, q8_0_store,
        read_back_owned, values,
    };
    use crate::registry::Registry;

    const VOCAB: usize = 48;

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn legacy_cfg() -> Qwen2Config {
        Qwen2Config {
            vocab: VOCAB,
            hidden: 64,
            inter: 96,
            layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            rotary_dim: 16,
            eps: 1e-6,
            max_pos: 64,
            qkv_bias: false,
            ..Default::default()
        }
    }

    fn legacy_params() -> GraniteParams {
        GraniteParams {
            moe: None,
            embed_mult: 2.0,
            attn_mult: 0.2,
            residual_mult: 0.5,
            logits_scale: 2.0,
        }
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// SC-002 (ADR-0101 tiers 1 and 2): the standard-stack body equals the legacy granite prefill
    /// and decode tracers on the fixture. Mutation: change the residual multiplier the family
    /// passes (`with_residual_scale`); red.
    #[test]
    fn granite_matches_the_legacy_tracers() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let (cfg, gp) = (legacy_cfg(), legacy_params());
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
                prefill: &|n| trace_granite_prefill_kv(cfg, gp, n, CAP),
                decode: &trace_granite_decode_kv_masked(cfg, gp, CAP),
                eval: &eval,
            },
        );
    }

    /// Each of the four multipliers is a different model: changing it moves the logits.
    #[test]
    fn each_multiplier_changes_the_logits() {
        let fx = fixture();
        let run = |config: &serde_json::Value| {
            let model = model_of(config, &fx.store);
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
        let base = run(&fx.config);
        for (key, value) in [
            ("embedding_multiplier", 3.0),
            ("attention_multiplier", 0.5),
            ("residual_multiplier", 0.9),
            ("logits_scaling", 4.0),
        ] {
            let mut config = fx.config.clone();
            config[key] = json!(value);
            assert_ne!(run(&config), base, "{key}");
        }
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
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
        };
        match build(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn a_config_granite_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "logits_scaling",
                json!(0.0),
                "logit_divisor",
                ConfigReason::NotFinitePositive,
            ),
            (
                "residual_multiplier",
                json!(-1.0),
                "residual_scale",
                ConfigReason::NotFinitePositive,
            ),
            (
                "attention_multiplier",
                json!("x"),
                "attention_multiplier",
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

    /// GraniteConfig's defaults: a missing eos is 2, the scalars default to 1, and the head is
    /// tied unless the config says otherwise (no `lm_head.weight` needed).
    #[test]
    fn omitted_fields_take_the_reference_defaults_and_the_head_is_tied() {
        let fx = fixture();
        let mut config = fixture_config();
        for key in [
            "eos_token_id",
            "tie_word_embeddings",
            "embedding_multiplier",
            "attention_multiplier",
            "residual_multiplier",
            "logits_scaling",
        ] {
            config.as_object_mut().unwrap().remove(key);
        }
        let mut store = WeightStore::builder();
        for (key, entry) in fx.store.iter() {
            if key.as_str() != "lm_head.weight" {
                store.insert(key.clone(), entry.clone()).unwrap();
            }
        }
        let model = model_of(&config, &store.build());
        assert_eq!(model.config().eos, BTreeSet::from([2]));
        assert_eq!(model.config().chat, ChatFormat::Granite);
    }

    /// A Q8_0 granite GGUF (projections Q8_0, the rest F32, scalars in `granite.*` keys).
    fn gguf_case() -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            tensors.push(gguf_tensor(&gguf_name_with_head(key.as_str()), entry));
            reference
                .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                .unwrap();
        }
        let mut kvs = gguf_dense_kvs("granite", &fx.config);
        kvs.extend([
            ("granite.embedding_scale".to_string(), GgufValue::F32(2.0)),
            ("granite.attention.scale".to_string(), GgufValue::F32(0.2)),
            ("granite.residual_scale".to_string(), GgufValue::F32(0.5)),
            ("granite.logit_scale".to_string(), GgufValue::F32(2.0)),
        ]);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004 (R-562-3): a Q8_0 granite GGUF resolves by its architecture key, builds packed
    /// projections traced as `PackedDequant`, and computes what the HF body computes over the same
    /// values. Mutation: drop the GGUF key from `ENTRY.keys` (`Unregistered`).
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
    }

    /// A GGUF without one of the Granite scalars is a typed error naming its key.
    #[test]
    fn a_gguf_missing_a_scalar_names_its_key() {
        let (index, ..) = gguf_case();
        let mut kvs = gguf_dense_kvs("granite", &fixture_config());
        kvs.push(("granite.embedding_scale".to_string(), GgufValue::F32(2.0)));
        let (partial, _) = read_back_owned(&kvs, &[]);
        drop(index);
        let err = GraniteConfig::from_raw(&RawConfig::Gguf(&partial)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "granite.attention.scale",
                    reason: ConfigReason::Missing,
                    ..
                }
            ),
            "{err}"
        );
    }
}
