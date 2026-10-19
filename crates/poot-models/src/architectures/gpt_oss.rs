//! gpt-oss behind [`Model`]: attention with per-head sink logits and biased projections,
//! alternating sliding-window and full layers, and a routed block of biased clamped-GLU experts. The
//! body is `standard_stack` over `standard_layer` with the shared attention and `components::moe`.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{ExpertProj, WeightMap, WeightStore};

use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::Glu;
use crate::components::linear::WeightError;
use crate::components::moe::{self, GateUpForm, MoeParams, MoeWeights, Orientation};
use crate::components::router::{RouterBias, RouterKind};
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step,
    decoder_program, standard_layer, standard_stack,
};
use crate::io::ModelSchema;
use crate::model::{ConfigReason, FamilyKey, Model, ModelConfig, ModelError, StepSpec, TraceError};
use crate::names::gguf::{GGUF_BASE, GGUF_SINK_ATTENTION};
use crate::names::{
    ExpertNames, HF_BASE, HF_SINK_ATTENTION, MoeLayers, MoeNames, family_weights_routed,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig};
use poot_text::ChatFormat;

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("gpt_oss");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "gpt_oss"),
        (ConfigSource::GgufArchitecture, "gpt-oss"),
    ],
    build,
    fixture,
    variants: &[],
};

/// gpt-oss's config facts: `GptOssConfig`'s defaults (the class declares no `eos_token_id`: a
/// checkpoint names it) and the `rope_theta` both formats carry.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("gpt-oss"),
    hf_theta: 150_000.0,
    gguf_theta: Some(150_000.0),
    eps: 1e-5,
    eos: None,
};

const EXPERTS: Field = Field::new("num_local_experts", "gpt-oss.expert_count");
const TOP_K: Field = Field::new("num_experts_per_tok", "gpt-oss.expert_used_count");
const WINDOW: Field = Field::new("sliding_window", "gpt-oss.attention.sliding_window");

/// `GptOssExperts._apply_gate`'s hardcoded sigmoid-gate rescale; not a config field.
const GLU_ALPHA: f32 = 1.702;
/// `swiglu_limit`'s class default; llama.cpp hardcodes the same value, so a GGUF has no key.
const DEFAULT_GLU_LIMIT: f32 = 7.0;
/// `sliding_window`'s class default.
const DEFAULT_WINDOW: usize = 128;

const HF_MOE: MoeNames = MoeNames {
    prefix: "model.layers.",
    router: ".mlp.router",
    router_bias: Some(".mlp.router.bias"),
    shared: None,
    experts: &[
        ExpertNames {
            proj: ExpertProj::GateUp,
            stack: Some(".mlp.experts.gate_up_proj"),
            each: None,
            bias: Some(".mlp.experts.gate_up_proj_bias"),
        },
        ExpertNames {
            proj: ExpertProj::Down,
            stack: Some(".mlp.experts.down_proj"),
            each: None,
            bias: Some(".mlp.experts.down_proj_bias"),
        },
    ],
};

const GGUF_MOE: MoeNames = MoeNames {
    prefix: "blk.",
    router: ".ffn_gate_inp",
    router_bias: Some(".ffn_gate_inp.bias"),
    shared: None,
    experts: &[
        ExpertNames {
            proj: ExpertProj::Gate,
            stack: Some(".ffn_gate_exps.weight"),
            each: None,
            bias: Some(".ffn_gate_exps.bias"),
        },
        ExpertNames {
            proj: ExpertProj::Up,
            stack: Some(".ffn_up_exps.weight"),
            each: None,
            bias: Some(".ffn_up_exps.bias"),
        },
        ExpertNames {
            proj: ExpertProj::Down,
            stack: Some(".ffn_down_exps.weight"),
            each: None,
            bias: Some(".ffn_down_exps.bias"),
        },
    ],
};

/// gpt-oss's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GptOssConfig {
    vocab: usize,
    width: usize,
    layers: usize,
    max_positions: usize,
    /// Full attention and the sliding-window attention: the same parameters but the window.
    full: AttentionParams,
    sliding: AttentionParams,
    /// Per layer: sliding window (`true`) or full attention.
    layer_is_sliding: Vec<bool>,
    layer: LayerParams,
    stack: StackParams,
    moe: MoeParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl GptOssConfig {
    pub(crate) fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        // Every gpt-oss projection has a bias; a config that says otherwise describes another model.
        if !fields.flag(Field::hf("attention_bias"), true)? {
            return Err(fields.error("attention_bias", ConfigReason::Unsupported));
        }
        let param = |e: crate::components::standard::ParamError| e.for_family(FAMILY);
        let norm = dims.norm(&fields, &SPEC)?;
        let full = dims
            .attention(&fields, &SPEC, true)?
            .with_out_bias()
            .with_sinks()
            .map_err(param)?;
        let window = match fields.opt_u64(WINDOW)? {
            None => DEFAULT_WINDOW,
            Some(n) => usize::try_from(n)
                .map_err(|_| fields.error(fields.key(WINDOW), ConfigReason::WrongType))?,
        };
        let sliding = full.with_window(window).map_err(param)?;
        let limit = fields.float(Field::hf("swiglu_limit"), DEFAULT_GLU_LIMIT)?;
        let moe = MoeParams::new(
            FAMILY,
            dims.width,
            dims.inter,
            fields.count(EXPERTS, None)?,
            fields.count(TOP_K, None)?,
            RouterKind::TopKSoftmax,
        )?
        .with_glu(Glu::clamped(GLU_ALPHA, limit).map_err(param)?)
        .with_router_bias(RouterBias::AddedToLogits)
        .map_err(param)?
        .with_expert_bias();
        // The HF default alternates, sliding first; a GGUF has no key and llama.cpp's
        // `swa_period` of 2 makes the same choice.
        let layer_is_sliding = layer_schedule(&fields, dims.layers)?;
        let moe = match raw {
            RawConfig::HfJson { .. } => moe
                .with_orientation(Orientation::InOut)
                .with_gate_up(GateUpForm::FusedInterleaved),
            RawConfig::Gguf(_) => moe,
        };
        Ok(Self {
            full,
            sliding,
            layer_is_sliding,
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

/// Which layers slide: HF's `layer_types` (`sliding_attention` / `full_attention`), else every
/// even layer.
fn layer_schedule(fields: &Fields<'_>, layers: usize) -> Result<Vec<bool>, ModelError> {
    let RawConfig::HfJson { config, .. } = fields.raw() else {
        return Ok((0..layers).map(|l| l % 2 == 0).collect());
    };
    let key = "layer_types";
    let wrong = || fields.error(key, ConfigReason::WrongType);
    let types = match config.get(key) {
        None | Some(serde_json::Value::Null) => {
            return Ok((0..layers).map(|l| l % 2 == 0).collect());
        }
        Some(types) => types.as_array().ok_or_else(wrong)?,
    };
    if types.len() != layers {
        return Err(fields.error(key, ConfigReason::WrongType));
    }
    types
        .iter()
        .map(|t| match t.as_str() {
            Some("sliding_attention") => Ok(true),
            Some("full_attention") => Ok(false),
            Some(_) => Err(fields.error(key, ConfigReason::Unsupported)),
            None => Err(wrong()),
        })
        .collect()
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    moe: MoeWeights,
}

#[derive(Debug)]
pub(crate) struct GptOss {
    params: GptOssConfig,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let p = GptOssConfig::from_raw(raw)?;
    let (tables, names): ([&[_]; 2], &MoeNames) = match raw.source() {
        ConfigSource::HfModelType => ([HF_BASE, HF_SINK_ATTENTION], &HF_MOE),
        ConfigSource::GgufArchitecture => ([GGUF_BASE, GGUF_SINK_ATTENTION], &GGUF_MOE),
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
                attn: AttentionWeights::new(&weights, l, p.width, &p.full)?,
                moe: MoeWeights::new(&weights, l, &p.moe)?,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let schema = ModelSchema::builder()
        .step(decoder_program(p.vocab))
        .build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    };
    Ok(ModelParts::new(Box::new(GptOss {
        params: p,
        weights,
        stack,
        layers,
        config,
        schema,
    })))
}

impl Model for GptOss {
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
        check_step(FAMILY, shape, p.max_positions)?;
        moe::check_step(FAMILY, shape, &p.moe)?;
        let step = Step::new(&b, shape)?;
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            let attn = if p.layer_is_sliding[l] {
                &p.sliding
            } else {
                &p.full
            };
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, attn),
                |b, BlockInput { block: h, .. }| moe::moe(b, &step, l, h, None, &layer.moe, &p.moe),
            )
        })?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a tiny gpt-oss of two layers (sliding then full, a window of 3 keys that
/// a five-token prompt exceeds), 8 experts of which 2 route, biases everywhere, `head_dim` 32 and
/// tied embeddings off. Widths are multiples of 32 so a Q8_0 copy of the expert stacks is valid.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gpt_oss",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 3,
        "swiglu_limit": 7.0,
        "attention_bias": true,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "rope_theta": 150000.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let n = |f: &str| crate::registry::Fixture::count(&config, f);
    let (h, d, inter, experts) = (
        n("hidden_size"),
        n("head_dim"),
        n("intermediate_size"),
        n("num_local_experts"),
    );
    let (q, kv) = (n("num_attention_heads") * d, n("num_key_value_heads") * d);
    let mut shapes = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![n("vocab_size"), h],
        ),
        ("model.norm.weight".to_string(), vec![h]),
        ("lm_head.weight".to_string(), vec![n("vocab_size"), h]),
    ];
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        shapes.extend([
            (k("input_layernorm.weight"), vec![h]),
            (k("post_attention_layernorm.weight"), vec![h]),
            (k("self_attn.q_proj.weight"), vec![q, h]),
            (k("self_attn.q_proj.bias"), vec![q]),
            (k("self_attn.k_proj.weight"), vec![kv, h]),
            (k("self_attn.k_proj.bias"), vec![kv]),
            (k("self_attn.v_proj.weight"), vec![kv, h]),
            (k("self_attn.v_proj.bias"), vec![kv]),
            (k("self_attn.o_proj.weight"), vec![h, q]),
            (k("self_attn.o_proj.bias"), vec![h]),
            (k("self_attn.sinks"), vec![n("num_attention_heads")]),
            (k("mlp.router.weight"), vec![experts, h]),
            (k("mlp.router.bias"), vec![experts]),
            (k("mlp.experts.gate_up_proj"), vec![experts, h, 2 * inter]),
            (k("mlp.experts.gate_up_proj_bias"), vec![experts, 2 * inter]),
            (k("mlp.experts.down_proj"), vec![experts, inter, h]),
            (k("mlp.experts.down_proj_bias"), vec![experts, h]),
        ]);
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod family_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use poot_quant::SourceRole;
    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{HandleFormat, WeightStoreBuilder};
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::{assert_matches_f64, run_step};
    use crate::components::testing::{
        CAP, GGML_F32, GGML_Q8_0, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build as built,
        close, f32_entry, gguf_dense_kvs, read_back_owned, values,
    };
    use crate::model::StepShape;
    use crate::registry::Registry;
    use poot_load::gguf::{GgufIndex, GgufValue};

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
            preprocessor: None,
        };
        built(&ENTRY, &raw, store)
    }

    const PROMPT: [i32; 5] = [3, 17, 40, 8, 25];

    /// Logits at every position of the whole prompt prefilled in one step.
    fn prefill_logits(model: &dyn Model, store: &WeightStore) -> Vec<f32> {
        run_step(model, store, CAP, &PROMPT, 0, &[]).0
    }

    /// `x[k] . w[k, cols]` in f64, plus `bias`.
    fn matmul(x: &[f64], w: &[f64], k: usize, cols: usize, bias: Option<&[f64]>) -> Vec<f64> {
        (0..cols)
            .map(|o| {
                bias.map_or(0.0, |b| b[o]) + (0..k).map(|i| x[i] * w[i * cols + o]).sum::<f64>()
            })
            .collect()
    }

    fn rmsnorm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        x.iter()
            .zip(w)
            .map(|(v, w)| v / (ms + eps).sqrt() * w)
            .collect()
    }

    /// Plain half-split RoPE of one head at `pos`.
    fn rope(head: &[f64], pos: usize, theta: f64) -> Vec<f64> {
        let d = head.len();
        let mut out = vec![0.0; d];
        for i in 0..d / 2 {
            let angle = pos as f64 * theta.powf(-((2 * i) as f64) / d as f64);
            let (s, c) = angle.sin_cos();
            out[i] = head[i] * c - head[i + d / 2] * s;
            out[i + d / 2] = head[i + d / 2] * c + head[i] * s;
        }
        out
    }

    /// An independent f64 forward pass of the fixture family over `tokens`: direct loops over a
    /// transposed copy of every projection, and over the selected experts only. Returns the
    /// logits of every position.
    fn reference(config: &serde_json::Value, store: &WeightStore, tokens: &[usize]) -> Vec<f64> {
        let n = |f: &str| crate::registry::Fixture::count(config, f);
        let (h, d, hq, hkv) = (
            n("hidden_size"),
            n("head_dim"),
            n("num_attention_heads"),
            n("num_key_value_heads"),
        );
        let (experts, top_k, inter, vocab) = (
            n("num_local_experts"),
            n("num_experts_per_tok"),
            n("intermediate_size"),
            n("vocab_size"),
        );
        let eps = config["rms_norm_eps"].as_f64().unwrap();
        let theta = config["rope_theta"].as_f64().unwrap();
        let limit = config["swiglu_limit"].as_f64().unwrap();
        let window = n("sliding_window");
        let w: HashMap<String, Vec<f64>> = store
            .iter()
            .map(|(k, e)| {
                (
                    k.to_string(),
                    values(e).iter().map(|&v| f64::from(v)).collect(),
                )
            })
            .collect();
        // HF stores a linear `[out, in]`; the reference reads `[in, out]`.
        let t = |name: &str, out: usize, inp: usize| -> Vec<f64> {
            let m = &w[name];
            (0..inp * out)
                .map(|i| m[(i % out) * inp + i / out])
                .collect()
        };
        let layer_types = config["layer_types"].as_array().unwrap();
        let mut x: Vec<Vec<f64>> = tokens
            .iter()
            .map(|&tok| w["model.embed_tokens.weight"][tok * h..(tok + 1) * h].to_vec())
            .collect();
        for (l, kind) in layer_types.iter().enumerate() {
            let p = |s: &str| format!("model.layers.{l}.{s}");
            let sliding = kind == "sliding_attention";
            let normed: Vec<Vec<f64>> = x
                .iter()
                .map(|row| rmsnorm(row, &w[&p("input_layernorm.weight")], eps))
                .collect();
            let proj = |name: &str, out: usize, inp: usize| {
                let wt = t(&p(&format!("self_attn.{name}_proj.weight")), out, inp);
                let bias = w[&p(&format!("self_attn.{name}_proj.bias"))].clone();
                move |row: &[f64]| matmul(row, &wt, inp, out, Some(&bias))
            };
            let (qp, kp, vp) = (
                proj("q", hq * d, h),
                proj("k", hkv * d, h),
                proj("v", hkv * d, h),
            );
            let rotate = |row: Vec<f64>, heads: usize, pos: usize| -> Vec<f64> {
                (0..heads)
                    .flat_map(|hh| rope(&row[hh * d..(hh + 1) * d], pos, theta))
                    .collect()
            };
            let q: Vec<Vec<f64>> = normed
                .iter()
                .enumerate()
                .map(|(i, r)| rotate(qp(r), hq, i))
                .collect();
            let k: Vec<Vec<f64>> = normed
                .iter()
                .enumerate()
                .map(|(i, r)| rotate(kp(r), hkv, i))
                .collect();
            let v: Vec<Vec<f64>> = normed.iter().map(|r| vp(r)).collect();
            let sinks = &w[&p("self_attn.sinks")];
            let mut attn = vec![vec![0.0; hq * d]; tokens.len()];
            for qh in 0..hq {
                let kh = qh / (hq / hkv);
                for i in 0..tokens.len() {
                    let start = if sliding {
                        (i + 1).saturating_sub(window)
                    } else {
                        0
                    };
                    let scores: Vec<f64> = (start..=i)
                        .map(|j| {
                            (0..d)
                                .map(|e| q[i][qh * d + e] * k[j][kh * d + e])
                                .sum::<f64>()
                                / (d as f64).sqrt()
                        })
                        .collect();
                    let m = scores.iter().copied().fold(sinks[qh], f64::max);
                    let denom =
                        (sinks[qh] - m).exp() + scores.iter().map(|s| (s - m).exp()).sum::<f64>();
                    for e in 0..d {
                        attn[i][qh * d + e] = scores
                            .iter()
                            .enumerate()
                            .map(|(jj, s)| (s - m).exp() / denom * v[start + jj][kh * d + e])
                            .sum();
                    }
                }
            }
            let o = t(&p("self_attn.o_proj.weight"), h, hq * d);
            for i in 0..tokens.len() {
                let y = matmul(
                    &attn[i],
                    &o,
                    hq * d,
                    h,
                    Some(&w[&p("self_attn.o_proj.bias")]),
                );
                for c in 0..h {
                    x[i][c] += y[c];
                }
            }
            let router = t(&p("mlp.router.weight"), experts, h);
            let gate_up = &w[&p("mlp.experts.gate_up_proj")];
            let gu_bias = &w[&p("mlp.experts.gate_up_proj_bias")];
            let down = &w[&p("mlp.experts.down_proj")];
            let down_bias = &w[&p("mlp.experts.down_proj_bias")];
            for row in &mut x {
                let hn = rmsnorm(row, &w[&p("post_attention_layernorm.weight")], eps);
                let logits = matmul(&hn, &router, h, experts, Some(&w[&p("mlp.router.bias")]));
                let mut order: Vec<usize> = (0..experts).collect();
                order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
                let chosen = &order[..top_k];
                let top = logits[chosen[0]];
                let denom: f64 = chosen.iter().map(|&e| (logits[e] - top).exp()).sum();
                for &e in chosen {
                    let weight = (logits[e] - top).exp() / denom;
                    let gu = matmul(
                        &hn,
                        &gate_up[e * h * 2 * inter..(e + 1) * h * 2 * inter],
                        h,
                        2 * inter,
                        Some(&gu_bias[e * 2 * inter..(e + 1) * 2 * inter]),
                    );
                    let act: Vec<f64> = (0..inter)
                        .map(|i| {
                            let g = gu[2 * i].min(limit);
                            let u = gu[2 * i + 1].clamp(-limit, limit);
                            (u + 1.0) * g / (1.0 + (-1.702 * g).exp())
                        })
                        .collect();
                    let y = matmul(
                        &act,
                        &down[e * inter * h..(e + 1) * inter * h],
                        inter,
                        h,
                        Some(&down_bias[e * h..(e + 1) * h]),
                    );
                    for c in 0..h {
                        row[c] += weight * y[c];
                    }
                }
            }
        }
        let head = t("lm_head.weight", vocab, h);
        x.iter()
            .flat_map(|row| {
                matmul(
                    &rmsnorm(row, &w["model.norm.weight"], eps),
                    &head,
                    h,
                    vocab,
                    None,
                )
            })
            .collect()
    }

    /// `store` with the tensor `key` replaced by `change` of its values.
    fn changed(store: &WeightStore, key: &str, change: impl Fn(usize, f32) -> f32) -> WeightStore {
        let mut out = WeightStore::builder();
        for (k, entry) in store.iter() {
            let entry = if k.as_str() == key {
                let v: Vec<f32> = values(entry)
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| change(i, v))
                    .collect();
                f32_entry(entry.shape(), &v)
            } else {
                entry.clone()
            };
            out.insert(k.clone(), entry).unwrap();
        }
        out.build()
    }

    /// SC-001/SC-002: the family's prefill logits at every position equal an independent f64
    /// forward pass (biased attention with sinks, a window of 3 keys under a 5-token prompt on the
    /// sliding layer, a biased router, interleaved clamped-GLU experts with biases). Mutation: drop
    /// `with_sinks` from the attention parameters; red.
    #[test]
    fn prefill_matches_an_independent_f64_forward_pass() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let got = prefill_logits(&*model, &fx.store);
        let tokens: Vec<usize> = PROMPT.iter().map(|&t| t as usize).collect();
        let want = reference(&fx.config, &fx.store, &tokens);
        assert_matches_f64(&got, &want, 1e-4);
    }

    /// Every sink, bias and router weight matters: perturbing each changes the logits.
    #[test]
    fn each_sink_bias_and_router_weight_changes_the_logits() {
        let fx = fixture();
        let base = prefill_logits(&*model_of(&fx.config, &fx.store), &fx.store);
        for key in [
            "model.layers.0.self_attn.sinks",
            "model.layers.1.self_attn.sinks",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.1.self_attn.q_proj.bias",
            "model.layers.0.mlp.router.bias",
            "model.layers.1.mlp.router.weight",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.1.mlp.experts.down_proj_bias",
            "model.layers.0.mlp.experts.down_proj",
        ] {
            let store = changed(&fx.store, key, |i, v| v + 0.3 * ((i % 5) as f32 - 2.0));
            let got = prefill_logits(&*model_of(&fx.config, &store), &store);
            assert!(
                got.iter().zip(&base).any(|(a, b)| (a - b).abs() > 1e-5),
                "the logits do not depend on {key}"
            );
        }
    }

    /// The window and the layer schedule change the output: a window past the prompt, and the
    /// schedule swapped, each differ from the fixture's; an absent `layer_types` is the fixture's
    /// own alternation.
    #[test]
    fn the_window_and_the_layer_schedule_change_the_logits() {
        let fx = fixture();
        let run = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut config = fixture_config();
            edit(&mut config);
            prefill_logits(&*model_of(&config, &fx.store), &fx.store)
        };
        let base = run(&|_| {});
        assert_ne!(run(&|c| c["sliding_window"] = json!(64)), base);
        assert_ne!(
            run(&|c| c["layer_types"] = json!(["full_attention", "sliding_attention"])),
            base
        );
        assert_eq!(
            run(&|c| {
                c.as_object_mut().unwrap().remove("layer_types");
            }),
            base
        );
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// SC-004 (HF half): BF16 and Q8_0 attention projections trace through the one body.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body_and_the_packed_transform() {
        assert_bf16_and_q8_0_trace_through_the_packed_transform(&ENTRY, &fixture(), |key| {
            key.contains("self_attn.") && key.ends_with("_proj.weight")
        });
    }

    /// SC-005: a store missing one required tensor fails `Registry::build` with the typed
    /// missing-weight error naming it, before any trace. Mutation: skip the row check in
    /// `names.rs`; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.0.self_attn.sinks",
            "model.layers.1.self_attn.o_proj.bias",
            "model.layers.1.self_attn.k_proj.bias",
            "model.layers.0.mlp.router.weight",
            "model.layers.1.mlp.router.bias",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.1.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.1.mlp.experts.down_proj_bias",
            "model.layers.0.post_attention_layernorm.weight",
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
        match build(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn a_config_gpt_oss_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(false),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "layer_types",
                json!(["sliding_attention"]),
                "layer_types",
                ConfigReason::WrongType,
            ),
            (
                "layer_types",
                json!(["sliding_attention", "chunked"]),
                "layer_types",
                ConfigReason::Unsupported,
            ),
            ("sliding_window", json!(0), "window", ConfigReason::Zero),
            (
                "swiglu_limit",
                json!(0.0),
                "glu_limit",
                ConfigReason::NotFinitePositive,
            ),
            (
                "num_experts_per_tok",
                json!(9),
                "top_k",
                ConfigReason::Exceeds { max: 8 },
            ),
            (
                "num_local_experts",
                json!(0),
                "num_local_experts",
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

    fn bytes_of(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    type GgufTensor = (String, Vec<u64>, u32, Vec<u8>);

    /// A dense tensor of the fixture under its GGUF `name`, also kept (as F32) under its HF key.
    fn carry(
        fx: &Fixture,
        key: &str,
        name: String,
        tensors: &mut Vec<GgufTensor>,
        hf: &mut WeightStoreBuilder,
    ) {
        let entry = fx.store.get(key).unwrap();
        let v = values(entry);
        let dims = entry.shape().iter().rev().map(|&d| d as u64).collect();
        tensors.push((name, dims, GGML_F32, bytes_of(&v)));
        hf.insert(key, f32_entry(entry.shape(), &v)).unwrap();
    }

    /// The GGUF twin of the fixture: llama.cpp's names and layouts (split gate and up as
    /// `[experts, out, K]`, `post_attention_norm` before the feed-forward block, untied head), the
    /// experts `F32` or Q8_0 per-expert payloads. Returns the index, its store and the HF store
    /// holding the same logical values (the Q8_0 case derives them from the decoded payloads).
    fn gguf_case(q8_experts: bool) -> (GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let n = |f: &str| crate::registry::Fixture::count(&fx.config, f);
        let (h, inter, e) = (
            n("hidden_size"),
            n("intermediate_size"),
            n("num_local_experts"),
        );
        let mut tensors: Vec<GgufTensor> = Vec::new();
        let mut hf = WeightStore::builder();
        for (name, key) in [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ] {
            carry(&fx, key, name.to_string(), &mut tensors, &mut hf);
        }
        for l in 0..n("num_hidden_layers") {
            let k = |s: &str| format!("model.layers.{l}.{s}");
            for (gguf, hf_key) in [
                ("attn_norm.weight", "input_layernorm.weight"),
                (
                    "post_attention_norm.weight",
                    "post_attention_layernorm.weight",
                ),
                ("attn_q.weight", "self_attn.q_proj.weight"),
                ("attn_q.bias", "self_attn.q_proj.bias"),
                ("attn_k.weight", "self_attn.k_proj.weight"),
                ("attn_k.bias", "self_attn.k_proj.bias"),
                ("attn_v.weight", "self_attn.v_proj.weight"),
                ("attn_v.bias", "self_attn.v_proj.bias"),
                ("attn_output.weight", "self_attn.o_proj.weight"),
                ("attn_output.bias", "self_attn.o_proj.bias"),
                ("attn_sinks.weight", "self_attn.sinks"),
                ("ffn_gate_inp.weight", "mlp.router.weight"),
                ("ffn_gate_inp.bias", "mlp.router.bias"),
                ("ffn_down_exps.bias", "mlp.experts.down_proj_bias"),
            ] {
                carry(
                    &fx,
                    &k(hf_key),
                    format!("blk.{l}.{gguf}"),
                    &mut tensors,
                    &mut hf,
                );
            }
            // Split values: gate/up `[E, I, H]`, down `[E, H, I]`.
            let mut gate = vec![0.0f32; e * inter * h];
            let mut up = vec![0.0f32; e * inter * h];
            let mut down = vec![0.0f32; e * h * inter];
            let mut blocks: [Vec<u8>; 3] = Default::default();
            if q8_experts {
                for ex in 0..e {
                    for (slot, buf, rows, kk) in [
                        (0, &mut gate, inter, h),
                        (1, &mut up, inter, h),
                        (2, &mut down, h, inter),
                    ] {
                        let seed = (1000 * l + ex * 3 + slot) as u64;
                        let payload = Arc::new(poot_quant::test_support::packed::random_payload(
                            WeightFormat::Q8_0,
                            [rows, kk],
                            seed,
                        ));
                        blocks[slot].extend_from_slice(payload.bytes(SourceRole::Blocks));
                        for r in 0..rows {
                            let at = (ex * rows + r) * kk;
                            payload.decode_row(r, &mut buf[at..at + kk]).unwrap();
                        }
                    }
                }
            } else {
                let gu = values(fx.store.get(&k("mlp.experts.gate_up_proj")).unwrap());
                let dn = values(fx.store.get(&k("mlp.experts.down_proj")).unwrap());
                for ex in 0..e {
                    for i in 0..inter {
                        for c in 0..h {
                            gate[(ex * inter + i) * h + c] = gu[(ex * h + c) * 2 * inter + 2 * i];
                            up[(ex * inter + i) * h + c] = gu[(ex * h + c) * 2 * inter + 2 * i + 1];
                            down[(ex * h + c) * inter + i] = dn[(ex * inter + i) * h + c];
                        }
                    }
                }
            }
            // The HF fused tensors from the split values.
            let mut gu = vec![0.0f32; e * h * 2 * inter];
            let mut dn = vec![0.0f32; e * inter * h];
            for ex in 0..e {
                for i in 0..inter {
                    for c in 0..h {
                        gu[(ex * h + c) * 2 * inter + 2 * i] = gate[(ex * inter + i) * h + c];
                        gu[(ex * h + c) * 2 * inter + 2 * i + 1] = up[(ex * inter + i) * h + c];
                        dn[(ex * inter + i) * h + c] = down[(ex * h + c) * inter + i];
                    }
                }
            }
            hf.insert(
                k("mlp.experts.gate_up_proj"),
                f32_entry(vec![e, h, 2 * inter], &gu),
            )
            .unwrap();
            hf.insert(
                k("mlp.experts.down_proj"),
                f32_entry(vec![e, inter, h], &dn),
            )
            .unwrap();
            for (name, vals, shape, block) in [
                ("ffn_gate_exps.weight", &gate, [e, inter, h], &blocks[0]),
                ("ffn_up_exps.weight", &up, [e, inter, h], &blocks[1]),
                ("ffn_down_exps.weight", &down, [e, h, inter], &blocks[2]),
            ] {
                let (ty, bytes) = if q8_experts {
                    (GGML_Q8_0, block.clone())
                } else {
                    (GGML_F32, bytes_of(vals))
                };
                tensors.push((
                    format!("blk.{l}.{name}"),
                    shape.iter().rev().map(|&d| d as u64).collect(),
                    ty,
                    bytes,
                ));
            }
            // Gate and up biases `[E, I]` de-interleave from the HF `[E, 2I]` bias.
            let key = k("mlp.experts.gate_up_proj_bias");
            let gub = values(fx.store.get(&key).unwrap());
            let (mut gb, mut ub) = (vec![0.0f32; e * inter], vec![0.0f32; e * inter]);
            for ex in 0..e {
                for i in 0..inter {
                    gb[ex * inter + i] = gub[ex * 2 * inter + 2 * i];
                    ub[ex * inter + i] = gub[ex * 2 * inter + 2 * i + 1];
                }
            }
            let dims = vec![inter as u64, e as u64];
            tensors.push((
                format!("blk.{l}.ffn_gate_exps.bias"),
                dims.clone(),
                GGML_F32,
                bytes_of(&gb),
            ));
            tensors.push((
                format!("blk.{l}.ffn_up_exps.bias"),
                dims,
                GGML_F32,
                bytes_of(&ub),
            ));
            hf.insert(key, f32_entry(vec![e, 2 * inter], &gub)).unwrap();
        }
        let mut kvs = gguf_dense_kvs("gpt-oss", &fx.config);
        kvs.extend([
            ("gpt-oss.expert_count".to_string(), GgufValue::U32(e as u32)),
            (
                "gpt-oss.expert_used_count".to_string(),
                GgufValue::U32(n("num_experts_per_tok") as u32),
            ),
            (
                "gpt-oss.attention.sliding_window".to_string(),
                GgufValue::U32(n("sliding_window") as u32),
            ),
        ]);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, hf.build())
    }

    /// SC-001/SC-007 (GGUF half): a `gpt-oss` GGUF resolves by its architecture key and computes
    /// what the HF body computes over the same values, with dense `F32` experts and with Q8_0
    /// per-expert owners (the packed expert chain). Mutation: drop the GGUF key from
    /// `ENTRY.keys`; `Unregistered`.
    #[test]
    fn a_gguf_resolves_by_architecture_and_equals_the_hf_body_dense_and_q8_0() {
        let registry = Registry::builtin().unwrap();
        for q8 in [false, true] {
            let (index, store, hf_store) = gguf_case(q8);
            let raw = RawConfig::Gguf(&index);
            assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
            let model = registry.build(&raw, &store).unwrap();
            let reference = model_of(&fixture_config(), &hf_store);
            let (got, state) = run_step(&*model.model, &store, CAP, &PROMPT, 0, &[]);
            let (want, want_state) = run_step(&*reference, &hf_store, CAP, &PROMPT, 0, &[]);
            close(&got, &want);
            let (got, _) = run_step(&*model.model, &store, CAP, &[5], 5, &state);
            let (want, _) = run_step(&*reference, &hf_store, CAP, &[5], 5, &want_state);
            close(&got, &want);
            let packed = model
                .model
                .weights()
                .iter()
                .filter(|(_, _, handle)| matches!(handle.format, HandleFormat::Packed(_)))
                .count();
            assert_eq!(
                packed,
                if q8 { 2 * 3 } else { 0 },
                "q8 = {q8}: one packed stack per layer and projection"
            );
        }
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
