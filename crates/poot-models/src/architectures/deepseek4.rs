//! DeepSeek-V4 behind [`Model`] (arXiv 2606.19348): a decoder whose residual stream is `hc_mult`
//! parallel streams per token, mixed around every block by manifold-constrained hyper-connections
//! ([`crate::components::mhc`]), with three kinds of attention layer chosen by the checkpoint's
//! `compress_ratios` schedule:
//!
//! - **sliding**: the local window over a carried cache ([`sliding`]);
//! - **HCA** (heavily compressed attention): the window plus every complete window of a
//!   non-overlapping compression of the layer's keys ([`hca`]);
//! - **CSA** (compressed sparse attention): the window plus the best `index_topk` entries of an
//!   overlapping compression, selected per query token by a lightning indexer ([`csa`]).
//!
//! The attention core is not MLA: a low-rank query, one key-value head read as both key and value,
//! partial interleaved RoPE on the trailing `qk_rope_head_dim` dims of each head (undone on the
//! output), per-head attention sinks and a grouped low-rank output projection. Every layer routes
//! through a mixture of experts with the square-root-softplus score; the first `num_hash_layers`
//! layers take their experts from a token-to-experts table ([`moe`]).
//!
//! One body traces every phase. The window caches and the raw histories the compressors pool carry
//! the position axis like an attention cache, and the compressed entries are recomputed from them,
//! so a prefill chunk continues from the positions already written and one token is a decode step.
//! Every dimension, count and the schedule is the checkpoint's: a fine-tune with fewer layers or
//! experts or other compression rates is the same family, and the checkpoint's packed experts and
//! projections are read as stored.

use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightMap, WeightStore};
use poot_tensor::DType;

use crate::chat::ChatFormat;
use crate::components::standard::{Step, check_step};
use crate::model::{
    FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, ShapeReason, StepShape,
    TraceError,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

mod config;
mod csa;
mod hca;
mod moe;
mod names;
mod sliding;
mod stack;

#[cfg(test)]
mod tests;

pub use config::{DeepseekV4Config, LayerKind};

pub const FAMILY: FamilyKey = FamilyKey::new("deepseek4");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "deepseek_v4")],
    build,
    fixture,
    variants: &[("finetune", finetune_fixture)],
};

#[derive(Debug)]
pub struct DeepseekV4 {
    params: DeepseekV4Config,
    weights: WeightMap,
    tower: stack::Tower,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = DeepseekV4Config::from_raw(raw)?;
    let weights = names::weights_of(&p, store)?;
    let tower = stack::Tower::new(&weights, &p).map_err(|e| e.for_family(FAMILY))?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: p.granule,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(DeepseekV4 {
        params: p,
        weights,
        tower,
        config,
    }))
}

impl Model for DeepseekV4 {
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
        crate::components::moe::check_step(FAMILY, shape, &p.score_moe)?;
        let granule = p.granule.get();
        if phase == Phase::Prefill && !shape.tokens.get().is_multiple_of(granule) {
            return Err(TraceError::ShapeUnsupported {
                family: FAMILY,
                shape,
                reason: ShapeReason::Granule { granule },
            });
        }
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let logits = self.tower.trace(&b, &step, p);
        Ok(step.finish(b, logits))
    }
}

/// The fixture's config: four layers (sliding, CSA, HCA, CSA) with compression rates 2 and 4 (a
/// prefill granule of 4), two hyper-connection streams, a hash-routed first layer and score-routed
/// others over 8 experts of which 2 route, with every projection width a multiple of 32 (so a packed
/// copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v4",
        "vocab_size": 48,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "head_dim": 16,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "o_groups": 2,
        "o_lora_rank": 8,
        "sliding_window": 3,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 20000.0,
        "compress_rates": {"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        "compress_ratios": [0, 2, 4, 2],
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 2,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "num_hash_layers": 1,
        "moe_intermediate_size": 32,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        // Low enough that the fixture's activations reach the clamp, so the gate's one-sided clamp
        // and the up projection's two-sided one are both live.
        "swiglu_limit": 0.5,
        "max_position_embeddings": 64,
        "eos_token_id": 47,
    })
}

fn fixture() -> Fixture {
    fixture_of(fixture_config())
}

/// A fine-tune of the fixture's architecture with other counts than any release: three layers
/// (CSA, sliding, HCA) at other compression rates, two hash-routed layers, six experts, fewer
/// heads and a narrower window. It ships no manifest and is the same family.
fn finetune_fixture() -> Fixture {
    let mut config = fixture_config();
    config["num_hidden_layers"] = serde_json::json!(3);
    config["compress_rates"] = serde_json::json!({
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 2,
    });
    config["compress_ratios"] = serde_json::json!([4, 0, 2, 0, 0]);
    config["num_hash_layers"] = serde_json::json!(2);
    config["n_routed_experts"] = serde_json::json!(6);
    config["num_attention_heads"] = serde_json::json!(2);
    config["sliding_window"] = serde_json::json!(2);
    fixture_of(config)
}

/// Every tensor of a checkpoint of `config`'s dimensions: `(name, shape, how it is stored)`.
pub(crate) fn tensors(config: &serde_json::Value) -> Vec<(String, Vec<usize>, Stored)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (vocab, w, layers) = (n("vocab_size"), n("hidden_size"), n("num_hidden_layers"));
    let (heads, d) = (n("num_attention_heads"), n("head_dim"));
    let (ql, groups, ol) = (n("q_lora_rank"), n("o_groups"), n("o_lora_rank"));
    let (hc, experts, inter) = (
        n("hc_mult"),
        n("n_routed_experts"),
        n("moe_intermediate_size"),
    );
    let (k, hash) = (n("num_experts_per_tok"), n("num_hash_layers"));
    let rate = |name: &str| {
        config["compress_rates"][name].as_u64().map_or(
            if name.starts_with("compressed") {
                4
            } else {
                128
            },
            |r| r as usize,
        )
    };
    let csa = rate("compressed_sparse_attention");
    let mix = (2 + hc) * hc;
    use Stored::{Bf16, F32, I64Table};
    let mut shapes = vec![
        ("embed.weight".to_string(), vec![vocab, w], Bf16),
        ("norm.weight".to_string(), vec![w], Bf16),
        ("head.weight".to_string(), vec![vocab, w], Bf16),
        ("hc_head_fn".to_string(), vec![hc, hc * w], Bf16),
        ("hc_head_base".to_string(), vec![hc], F32),
        ("hc_head_scale".to_string(), vec![1], F32),
    ];
    for l in 0..layers {
        let at = |s: &str| format!("layers.{l}.{s}");
        let ratio = config["compress_ratios"][l].as_u64().unwrap() as usize;
        shapes.extend([
            (at("attn_norm.weight"), vec![w], Bf16),
            (at("ffn_norm.weight"), vec![w], Bf16),
            (at("attn.wq_a.weight"), vec![ql, w], Bf16),
            (at("attn.q_norm.weight"), vec![ql], Bf16),
            (at("attn.wq_b.weight"), vec![heads * d, ql], Bf16),
            (at("attn.wkv.weight"), vec![d, w], Bf16),
            (at("attn.kv_norm.weight"), vec![d], Bf16),
            (at("attn.attn_sink"), vec![heads], Bf16),
            (
                at("attn.wo_a.weight"),
                vec![groups * ol, heads * d / groups],
                Bf16,
            ),
            (at("attn.wo_b.weight"), vec![w, groups * ol], Bf16),
            (at("hc_attn_fn"), vec![mix, hc * w], Bf16),
            (at("hc_attn_base"), vec![mix], F32),
            (at("hc_attn_scale"), vec![3], F32),
            (at("hc_ffn_fn"), vec![mix, hc * w], Bf16),
            (at("hc_ffn_base"), vec![mix], F32),
            (at("hc_ffn_scale"), vec![3], F32),
            (at("ffn.gate.weight"), vec![experts, w], Bf16),
            (at("ffn.shared_experts.w1.weight"), vec![inter, w], Bf16),
            (at("ffn.shared_experts.w3.weight"), vec![inter, w], Bf16),
            (at("ffn.shared_experts.w2.weight"), vec![w, inter], Bf16),
        ]);
        if l < hash {
            shapes.push((at("ffn.gate.tid2eid"), vec![vocab, k], I64Table));
        } else {
            shapes.push((at("ffn.gate.bias"), vec![experts], F32));
        }
        for e in 0..experts {
            shapes.extend([
                (
                    at(&format!("ffn.experts.{e}.w1.weight")),
                    vec![inter, w],
                    Bf16,
                ),
                (
                    at(&format!("ffn.experts.{e}.w3.weight")),
                    vec![inter, w],
                    Bf16,
                ),
                (
                    at(&format!("ffn.experts.{e}.w2.weight")),
                    vec![w, inter],
                    Bf16,
                ),
            ]);
        }
        if ratio != 0 {
            let overlapping = ratio == csa;
            let (r, raw) = (ratio, if overlapping { 2 * d } else { d });
            shapes.extend([
                (at("attn.compressor.wkv.weight"), vec![raw, w], Bf16),
                (at("attn.compressor.wgate.weight"), vec![raw, w], Bf16),
                (at("attn.compressor.ape"), vec![r, raw], Bf16),
                (at("attn.compressor.norm.weight"), vec![d], Bf16),
            ]);
        }
        if ratio == csa {
            let (ih, id) = (n("index_n_heads"), n("index_head_dim"));
            shapes.extend([
                (at("attn.indexer.wq_b.weight"), vec![ih * id, ql], Bf16),
                (at("attn.indexer.weights_proj.weight"), vec![ih, w], Bf16),
                (
                    at("attn.indexer.compressor.wkv.weight"),
                    vec![2 * id, w],
                    Bf16,
                ),
                (
                    at("attn.indexer.compressor.wgate.weight"),
                    vec![2 * id, w],
                    Bf16,
                ),
                (at("attn.indexer.compressor.ape"), vec![csa, 2 * id], Bf16),
                (at("attn.indexer.compressor.norm.weight"), vec![id], Bf16),
            ]);
        }
    }
    shapes
}

/// How a fixture tensor is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stored {
    Bf16,
    F32,
    /// The hash router's `[vocab, top_k]` table of expert ids, stored as I64.
    I64Table,
}

/// A deterministic checkpoint of `config`: norm scales near one, hyper-connection scales near one,
/// every other weight small and centred, and a hash table whose row `t` names the experts `t,
/// t + 1, ...` (mod the expert count), so neighbouring tokens route differently.
fn fixture_of(config: serde_json::Value) -> Fixture {
    let experts = config["n_routed_experts"].as_u64().unwrap() as usize;
    let mut store = WeightStore::builder();
    let mut seed = 0x9e37_79b9u32;
    let mut uniform = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    for (key, shape, stored) in tensors(&config) {
        let n: usize = shape.iter().product();
        let value = |u: f32| {
            if key.contains("norm") || key.ends_with("_scale") {
                1.0 + 0.2 * u
            } else if key.ends_with("_base") || key.ends_with("gate.bias") {
                0.2 * u
            } else {
                0.5 * u
            }
        };
        let (dtype, bytes): (DType, Vec<u8>) = match stored {
            Stored::Bf16 => (
                DType::BF16,
                (0..n)
                    .flat_map(|_| ((value(uniform()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
            Stored::F32 => (
                DType::F32,
                (0..n)
                    .flat_map(|_| value(uniform()).to_le_bytes())
                    .collect(),
            ),
            Stored::I64Table => {
                let k = shape[1];
                (
                    DType::I64,
                    (0..n)
                        .map(|i| (((i / k) + (i % k)) % experts) as i64)
                        .flat_map(i64::to_le_bytes)
                        .collect(),
                )
            }
        };
        let dense = DenseWeight::try_new(dtype, shape, bytes.into())
            .expect("fixture bytes match their shape");
        store
            .insert(key, WeightEntry::Dense(dense))
            .expect("fixture keys are unique");
    }
    Fixture {
        config,
        generation: None,
        store: store.build(),
    }
}
