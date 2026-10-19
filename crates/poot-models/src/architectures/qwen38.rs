//! Qwen3.8-Flash-Next (HF `qwen4_exp`, GGUF `qwen4exp`) behind [`Model`]: a hybrid stack whose mixer is a Gated
//! DeltaNet linear-attention layer ([`gdn`](crate::components::gdn)) on most layers and gated
//! attention narrowed by a block indexer (QSA, [`qsa`]) on the rest, each followed by the routed
//! mixture of experts with a sigmoid-gated shared expert ([`moe`](crate::components::moe)). The
//! residual is `hc_count` parallel streams gated by Hyper-Connections ([`gated_residual`]), and the
//! layers `ple_layer_ids` names start with the per-layer n-gram embedding
//! ([`ple`](crate::components::ple)).
//!
//! The model is its config ([`config`]: typed, checked once, nothing pinned to a shipped checkpoint),
//! its weight-name table per format (also in [`config`]) and one body ([`trace`]). A step carries the QSA layers'
//! KV caches and indexer block keys, each GDN layer's conv window and state matrices, and each PLE layer's
//! n-gram history and conv window, so a prompt prefilled in chunks continues exactly where the
//! previous chunk stopped and one token is a decode step. A prefill is a whole number of
//! `indexer_compress_ratio` tokens (the model's prefill granule) and the cache capacity a multiple of
//! it; a prompt that is not one runs as whole-granule prefill plus decode steps.
//!
//! Storage stays as stored: the routed experts are dense stacks or packed per-expert weights (E4M3
//! in the official release, K-quant and IQ4 blocks in a GGUF), the BF16 owners are never widened per
//! step.

use std::num::NonZeroU32;
use std::sync::Arc;

use poot_graph_ir::{Builder, Graph, StateRole};
use poot_quant::format::{OperandRole, ScaleEncoding, WeightFormat};
use poot_quant::weights::{WeightEntry, WeightMap, WeightStore};
use poot_quant::{PackedPayload, PackedWeight, SourceRole, StoredBytes};

use crate::components::linear::WeightError;
use crate::components::moe;
use crate::components::standard::{Step, check_step, decoder_program};
use crate::io::{ModelSchema, StateSpec};
use crate::model::{
    ConfigReason, FamilyKey, KvLayout, Model, ModelConfig, ModelError, ShapeReason, StepSpec,
    TraceError,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_text::ChatFormat;

mod config;
mod gated_residual;
mod qsa;
mod trace;

pub(crate) use config::Qwen4ExpParams;

#[cfg(test)]
mod tests;

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("qwen4exp");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "qwen4_exp"),
        (ConfigSource::GgufArchitecture, "qwen4exp"),
    ],
    build,
    fixture,
    variants: &[("packed", packed_fixture), ("finetune", finetune_fixture)],
};

/// One layer's mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerKind {
    /// Gated DeltaNet linear attention.
    LinearAttention,
    /// Gated attention under the QSA block selection.
    FullAttention,
}

#[derive(Debug)]
pub(crate) struct Qwen4Exp {
    params: Qwen4ExpParams,
    weights: WeightMap,
    body: trace::Weights,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let mut p = Qwen4ExpParams::from_raw(raw)?;
    let (names, gate_up) = config::expert_layout(p.layout, store);
    p.moe = p.moe.with_gate_up(gate_up);
    let weights = config::weights_of(&p, store, names)?;
    let body = trace::Weights::new(&weights, &p).map_err(|e: WeightError| e.for_family(FAMILY))?;
    let mut step = decoder_program(p.vocab);
    if p.kinds.contains(&LayerKind::LinearAttention) {
        step = step.carrying(StateSpec::recurrent("gdn"));
    }
    if p.kinds.contains(&LayerKind::FullAttention) {
        let ratio = u32::try_from(p.ratio.get())
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(ModelError::Config {
                family: FAMILY,
                field: "indexer_compress_ratio",
                reason: ConfigReason::Exceeds {
                    max: u32::MAX as usize,
                },
            })?;
        step = step.carrying(StateSpec::new(
            "index_blocks",
            StateRole::Blocked { axis: 2, ratio },
        ));
    }
    let schema = ModelSchema::builder().step(step).build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        prefill_granule: p.ratio,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(Qwen4Exp {
        params: p,
        weights,
        body,
        config,
        schema,
    }))
}

impl Model for Qwen4Exp {
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
        // The indexer pools whole blocks of the cache, and a prefill reads and writes whole blocks.
        let granule = p.ratio.get();
        let off_granule = !shape.capacity.get().is_multiple_of(granule)
            || (!shape.is_decode() && !shape.tokens.get().is_multiple_of(granule));
        if off_granule {
            return Err(TraceError::ShapeUnsupported {
                family: FAMILY,
                shape,
                reason: ShapeReason::Granule { granule },
            });
        }
        // The block-key cache is a per-row buffer; its paging granule is POOT-1148's to decide.
        if matches!(shape.kv, KvLayout::Paged { .. }) {
            return Err(TraceError::ShapeUnsupported {
                family: FAMILY,
                shape,
                reason: ShapeReason::PagedIndexer,
            });
        }
        let step = Step::new(&b, shape)?;
        let logits = trace::trace_step(&b, &step, &self.body, p)?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: three layers (two Gated DeltaNet, the second with the PLE, then QSA
/// attention) over four routed experts and a gated shared expert, two Hyper-Connection streams and
/// blocks of two positions.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen4_exp",
        "tie_word_embeddings": false,
        "text_config": {
            "vocab_size": 48,
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "layer_types": ["linear_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "partial_rotary_factor": 0.5,
            "rope_parameters": { "rope_type": "default", "rope_theta": 10000.0 },
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 8,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "hc_count": 2,
            "hc_lowrank": 4,
            "ple_layer_ids": [2],
            "ple_embed_dim": 16,
            "ple_conv_kernel_size": 3,
            "ngram_size": 3,
            "ngram_vocab_size_base": 16,
            "heads_per_ngram": 2,
            "make_ngram_vocab_size_divisible_by": 8,
            "eos_token_id": 47,
        },
    })
}

/// How a fixture stores the routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Experts {
    /// One `gate_up_proj` and one `down_proj` stack per layer.
    Fused,
    /// Each expert's three projections as their own E4M3 block-scaled packed tensors.
    PackedE4m3,
}

/// The tensors of a Qwen4Exp checkpoint of `config`, by HF name, with the routed experts in the
/// stored form `experts` names.
pub(crate) fn hf_shapes(config: &serde_json::Value, experts: Experts) -> Vec<(String, Vec<usize>)> {
    let text = &config["text_config"];
    let n = |f: &str| crate::registry::Fixture::count(text, f);
    let params = Qwen4ExpParams::from_raw(&RawConfig::HfJson {
        config,
        generation: None,
    })
    .expect("the config is valid");
    let (vocab, w) = (n("vocab_size"), n("hidden_size"));
    let (heads, kv, hd) = (
        n("num_attention_heads"),
        n("num_key_value_heads"),
        n("head_dim"),
    );
    let (hk, hv, d, kernel) = (
        n("linear_num_key_heads"),
        n("linear_num_value_heads"),
        n("linear_key_head_dim"),
        n("linear_conv_kernel_dim"),
    );
    let (conv, value_dim) = (2 * hk * d + hv * d, hv * d);
    let (e, inter, shared) = (
        n("num_experts"),
        n("moe_intermediate_size"),
        n("shared_expert_intermediate_size"),
    );
    let (hc, rank) = (n("hc_count"), n("hc_lowrank"));
    let (hi, di) = (n("indexer_n_heads"), n("indexer_head_dim"));
    let stream = hc * w;
    let text_root = "model.language_model";
    let gate = |stem: String, inject: bool| {
        let mut rows = vec![
            (format!("{stem}.hc_norm.weight"), vec![stream]),
            (
                format!("{stem}.input_mix_weight_down.weight"),
                vec![rank, stream],
            ),
            (
                format!("{stem}.input_mix_weight_up.weight"),
                vec![stream, rank],
            ),
        ];
        if inject {
            rows.push((
                format!("{stem}.block_inject_weight.weight"),
                vec![hc, stream],
            ));
        }
        rows
    };
    let mut shapes = vec![
        (format!("{text_root}.embed_tokens.weight"), vec![vocab, w]),
        ("lm_head.weight".to_string(), vec![vocab, w]),
    ];
    shapes.extend(gate(format!("{text_root}.hyper_connection_mixer"), false));
    for (l, kind) in params.kinds.iter().enumerate() {
        let k = |s: &str| format!("{text_root}.layers.{l}.{s}");
        shapes.extend(gate(k("attn_hyper_connection"), true));
        shapes.extend(gate(k("mlp_hyper_connection"), true));
        if let Some(ple) = &params.ple[l] {
            let (embed, rows) = (ple.spec().embed_dim, ple.rows());
            shapes.extend([
                (
                    k("ple.ple_embedding.ngram_embedding.weight"),
                    vec![rows, ple.head_dim()],
                ),
                (k("ple.key_proj.weight"), vec![stream, embed]),
                (k("ple.value_proj.weight"), vec![w, embed]),
                (k("ple.norm_key.weight"), vec![stream]),
                (k("ple.norm_query.weight"), vec![stream]),
                (k("ple.norm_conv.weight"), vec![stream]),
                (
                    k("ple.conv1d.weight"),
                    vec![stream, 1, n("ple_conv_kernel_size")],
                ),
            ]);
        }
        match kind {
            LayerKind::FullAttention => shapes.extend([
                (k("self_attn.q_proj.weight"), vec![2 * heads * hd, w]),
                (k("self_attn.k_proj.weight"), vec![kv * hd, w]),
                (k("self_attn.v_proj.weight"), vec![kv * hd, w]),
                (k("self_attn.o_proj.weight"), vec![w, heads * hd]),
                (k("self_attn.q_norm.weight"), vec![hd]),
                (k("self_attn.k_norm.weight"), vec![hd]),
                (
                    k("self_attn.indexer.index_qk_proj.weight"),
                    vec![(hi + 1) * di, w],
                ),
                (k("self_attn.indexer.q_layernorm.weight"), vec![di]),
                (k("self_attn.indexer.k_layernorm.weight"), vec![di]),
            ]),
            LayerKind::LinearAttention => shapes.extend([
                (k("linear_attn.in_proj_qkv.weight"), vec![conv, w]),
                (k("linear_attn.in_proj_z.weight"), vec![value_dim, w]),
                (k("linear_attn.in_proj_b.weight"), vec![hv, w]),
                (k("linear_attn.in_proj_a.weight"), vec![hv, w]),
                (k("linear_attn.conv1d.weight"), vec![conv, 1, kernel]),
                (k("linear_attn.dt_bias"), vec![hv]),
                (k("linear_attn.A_log"), vec![hv]),
                (k("linear_attn.norm.weight"), vec![d]),
                (k("linear_attn.out_proj.weight"), vec![w, value_dim]),
            ]),
        }
        shapes.extend([
            (k("mlp.gate.weight"), vec![e, w]),
            (k("mlp.shared_expert.gate_proj.weight"), vec![shared, w]),
            (k("mlp.shared_expert.up_proj.weight"), vec![shared, w]),
            (k("mlp.shared_expert.down_proj.weight"), vec![w, shared]),
            (k("mlp.shared_expert_gate.weight"), vec![1, w]),
        ]);
        if experts == Experts::Fused {
            shapes.extend([
                (k("mlp.experts.gate_up_proj"), vec![e, 2 * inter, w]),
                (k("mlp.experts.down_proj"), vec![e, w, inter]),
            ]);
        }
    }
    shapes
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_shapes(&config, Experts::Fused);
    Fixture::bf16(config, shapes)
}

/// A checkpoint of the same architecture at sizes no shipped one has: two layers (the PLE on the
/// first), three experts routed one per token, three hyper-connection streams, blocks of four
/// positions and another head layout. No manifest, no pinned count: nothing about it is special.
fn finetune_fixture() -> Fixture {
    let mut config = fixture_config();
    let text = config["text_config"].as_object_mut().unwrap();
    for (key, value) in [
        ("hidden_size", serde_json::json!(24)),
        ("num_hidden_layers", serde_json::json!(2)),
        (
            "layer_types",
            serde_json::json!(["linear_attention", "full_attention"]),
        ),
        ("ple_layer_ids", serde_json::json!([1])),
        ("num_attention_heads", serde_json::json!(3)),
        ("num_experts", serde_json::json!(3)),
        ("num_experts_per_tok", serde_json::json!(1)),
        ("moe_intermediate_size", serde_json::json!(12)),
        ("shared_expert_intermediate_size", serde_json::json!(12)),
        ("hc_count", serde_json::json!(3)),
        ("hc_lowrank", serde_json::json!(5)),
        ("indexer_compress_ratio", serde_json::json!(4)),
        ("indexer_budget", serde_json::json!(4)),
        ("linear_num_key_heads", serde_json::json!(1)),
        ("linear_num_value_heads", serde_json::json!(3)),
        ("linear_key_head_dim", serde_json::json!(4)),
        ("linear_value_head_dim", serde_json::json!(4)),
        ("linear_conv_kernel_dim", serde_json::json!(2)),
    ] {
        text.insert(key.to_string(), value);
    }
    let shapes = hf_shapes(&config, Experts::Fused);
    Fixture::bf16(config, shapes)
}

/// The fixture with every routed expert packed as the official release stores it: one E4M3 tensor
/// per expert and projection with a block scale.
fn packed_fixture() -> Fixture {
    let config = fixture_config();
    let base = Fixture::bf16(config.clone(), hf_shapes(&config, Experts::PackedE4m3));
    let text = &config["text_config"];
    let n = |f: &str| crate::registry::Fixture::count(text, f);
    let (w, inter, experts) = (
        n("hidden_size"),
        n("moe_intermediate_size"),
        n("num_experts"),
    );
    let mut store = WeightStore::builder();
    store
        .extend_from(&base.store)
        .expect("the base store's keys are unique");
    for layer in 0..n("num_hidden_layers") {
        for expert in 0..experts {
            for (proj, shape) in [
                ("gate_proj", [inter, w]),
                ("up_proj", [inter, w]),
                ("down_proj", [w, inter]),
            ] {
                let seed = ((layer * experts + expert) * 3) as u64 + proj.len() as u64;
                store
                    .insert(
                        format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"),
                        WeightEntry::Packed(Arc::new(e4m3_payload(shape, seed))),
                    )
                    .expect("fixture keys are unique");
            }
        }
    }
    Fixture {
        store: store.build(),
        ..base
    }
}

/// A deterministic E4M3 block-scaled `[out, K]` weight: pseudo-random finite codes under scales
/// cycling through 0.5, 1, 1.5 and 2 (BF16), so every output depends on every scale.
fn e4m3_payload(shape: [usize; 2], seed: u64) -> PackedPayload {
    const SCALES: [u16; 4] = [0x3f00, 0x3f80, 0x3fc0, 0x4000];
    let format = WeightFormat::E4m3Block128 {
        scale: ScaleEncoding::Bf16,
    };
    let weight = PackedWeight::try_new(format, shape).expect("a block-scaled E4M3 weight");
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let sources: Vec<(SourceRole, StoredBytes)> = weight
        .sources()
        .into_iter()
        .map(|role| {
            let bytes: Vec<u8> = match role {
                SourceRole::Planar(OperandRole::Scale) => (0..weight.source_bytes(role) / 2)
                    .flat_map(|i| SCALES[i % SCALES.len()].to_le_bytes())
                    .collect(),
                _ => (0..weight.source_bytes(role))
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        // E4M3FN's NaN is `0x7f` and `0xff`: keep every code finite.
                        let code = (state >> 24) as u8;
                        if code & 0x7f == 0x7f { code & !1 } else { code }
                    })
                    .collect(),
            };
            (role, StoredBytes::from(bytes))
        })
        .collect();
    PackedPayload::try_new(weight, sources).expect("finite codes and scales are admitted")
}
