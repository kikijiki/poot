//! GLM-5.3-Flash behind [`Model`]: a decoder whose residual stream is `hc_mult` parallel streams per
//! token, mixed around every block by manifold-constrained hyper-connections
//! ([`crate::components::mhc`]), with two kinds of token mixer chosen per layer by the checkpoint's
//! `layer_types`:
//!
//! - **KDA** (Kimi Delta Attention, `linear_attention`): a gated delta rule over a carried matrix
//!   state with a decay per key channel ([`crate::components::kda`]);
//! - **DSA** (`deepseek_sparse_attention`): latent attention without rotary embedding whose queries
//!   read the positions a pooled indexer keeps ([`crate::components::dsa::pooled_dsa`]).
//!
//! Each layer's feed-forward is a dense gated block or, past the leading dense layers, a shared
//! expert plus routed experts chosen by the sigmoid router with a selection-only bias
//! ([`crate::components::moe`]); both gate with the clamped SwiGLU.
//!
//! One body traces every step shape. The mixers carry their conv windows, matrix states and caches, so a
//! prefill chunk continues from the positions already written and one token is a decode step. Every
//! dimension, count and the schedule is the checkpoint's: a fine-tune with fewer layers or experts
//! is the same family.
//!
//! The checkpoint's multi-token-prediction layer is traced by the `draft` program ([`mtp`]): a DSA block
//! over the hidden-state carry and a host token. A checkpoint that stores no such layer, or more than
//! one (the reference cycles them by draft step, which a step shape does not name), refuses the
//! program typed.

use poot_graph_ir::{Builder, Graph};
#[cfg(feature = "test-support")]
use poot_quant::weights::DenseWeight;
#[cfg(feature = "test-support")]
use poot_quant::weights::WeightEntry;
use poot_quant::weights::WeightStore;
#[cfg(feature = "test-support")]
use poot_tensor::DType;
use poot_weights::WeightMap;
use std::sync::Arc;

use crate::components::standard::{Step, decoder_program, draft_program};
use crate::io::{ModelSchema, ProgramId, StateSpec};
use crate::model::{FamilyKey, Model, ModelConfig, ModelError, StepSpec, TraceError};
#[cfg(feature = "test-support")]
use crate::registry::Fixture;
use crate::registry::{ConfigSource, FamilyEntry, ModelParts, RawConfig};
use poot_text::{ChatFormat, HermesToolCodec};

mod config;
mod mtp;
mod names;
mod stack;

#[cfg(test)]
mod tests;

pub use config::{Ffn, Glm53Config, LayerKind, Mixer};

/// GLM-5.3-Flash's [`FamilyKey`].
pub const FAMILY: FamilyKey = FamilyKey::new("glm53_flash");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "glm5_next")],
    build,
};

/// The built GLM-5.3-Flash model: config, name-resolved weights, the layer tower and the routed
/// experts. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct Glm53Flash {
    params: Glm53Config,
    weights: WeightMap,
    tower: stack::Tower,
    config: ModelConfig,
    schema: ModelSchema,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let p = Glm53Config::from_raw(raw)?;
    let weights = names::weights_of(&p, store)?;
    let tower = stack::Tower::new(&weights, &p).map_err(|e| e.for_family(FAMILY))?;
    let mut schema =
        ModelSchema::builder().step(decoder_program(p.vocab).carrying(StateSpec::recurrent("kda")));
    if tower.has_draft(&p) {
        schema = schema.draft(draft_program(p.vocab));
    }
    let schema = schema.build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        prefill_granule: std::num::NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
        tool_codec: Some(Arc::new(HermesToolCodec)),
    };
    Ok(ModelParts::new(Box::new(Glm53Flash {
        params: p,
        weights,
        tower,
        config,
        schema,
    })))
}

impl Model for Glm53Flash {
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
        let draft = spec.program == ProgramId::DRAFT;
        let step = Step::new(&b, shape)?;
        let logits = if draft {
            self.tower
                .trace_mtp(&b, &step, p)?
                .ok_or(TraceError::ProgramUnsupported {
                    family: FAMILY,
                    program: spec.program,
                })?
        } else {
            self.tower.trace(&b, &step, p)?
        };
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: four layers (a dense KDA layer, then DSA, KDA and DSA layers over routed
/// experts), two hyper-connection streams, 6 experts of which 2 route with one shared expert, and
/// every dimension small. The indexer pools 2 positions and keeps 2 pools, so a few tokens already
/// leave pools out of the attention. A checkpoint stores one multi-token-prediction layer after the
/// four: a DSA layer over routed experts with no hyper-connections.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_config() -> serde_json::Value {
    let mut text = serde_json::json!({
        "model_type": "glm5_next_text",
        "vocab_size": 48,
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "num_nextn_predict_layers": 1,
        "layer_types": [
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "hc_mult": 2,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 3,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "eos_token_id": [47],
    });
    let attention = serde_json::json!({
        "num_attention_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 6,
        "qk_rope_head_dim": 0,
        "v_head_dim": 6,
        "mla_use_nope": true,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "index_kpool": 2,
        "index_kpool_compress": true,
        "index_kpool_always_select_tail": true,
        "indexer_types": ["full", "full", "full", "full"],
    });
    let ffn = serde_json::json!({
        "intermediate_size": 24,
        "moe_intermediate_size": 12,
        "n_routed_experts": 6,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        // Low enough that the fixture's activations reach the clamp, so the gate's one-sided clamp
        // and the up projection's two-sided one are both live.
        "swiglu_limit": 0.5,
    });
    let object = text.as_object_mut().expect("a JSON object");
    for part in [attention, ffn] {
        object.extend(part.as_object().expect("a JSON object").clone());
    }
    serde_json::json!({
        "model_type": "glm5_next",
        "tie_word_embeddings": false,
        "text_config": text,
    })
}

#[cfg(feature = "test-support")]
pub(crate) fn fixture() -> Fixture {
    fixture_of(fixture_config())
}

/// A fine-tune of the fixture's architecture with other counts than any release: three layers in
/// another order (KDA over a dense block, DSA, then KDA), fewer routed experts and heads, another
/// pool size and no multi-token-prediction layer. It ships no manifest and is the same family.
#[cfg(feature = "test-support")]
pub(crate) fn finetune_fixture() -> Fixture {
    let mut config = fixture_config();
    let text = &mut config["text_config"];
    text["num_hidden_layers"] = serde_json::json!(3);
    text["num_nextn_predict_layers"] = serde_json::json!(0);
    text["layer_types"] = serde_json::json!([
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention"
    ]);
    text["mlp_layer_types"] = serde_json::json!(["dense", "dense", "sparse"]);
    text["indexer_types"] = serde_json::json!(["full", "full", "full"]);
    text["n_routed_experts"] = serde_json::json!(4);
    text["num_experts_per_tok"] = serde_json::json!(3);
    text["num_attention_heads"] = serde_json::json!(3);
    text["linear_attn_config"]["num_heads"] = serde_json::json!(3);
    text["index_kpool"] = serde_json::json!(3);
    text["index_topk"] = serde_json::json!(6);
    fixture_of(config)
}

/// How a fixture tensor is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(feature = "test-support")]
pub(crate) enum Stored {
    Bf16,
    F32,
}

/// Every tensor of a checkpoint of `config`'s dimensions: `(name, shape, how it is stored)`.
#[cfg(feature = "test-support")]
pub(crate) fn tensors(config: &serde_json::Value) -> Vec<(String, Vec<usize>, Stored)> {
    let text = &config["text_config"];
    let n = |f: &str| crate::registry::Fixture::count(text, f);
    let (vocab, w, layers) = (n("vocab_size"), n("hidden_size"), n("num_hidden_layers"));
    let hc = n("hc_mult");
    let mix = (2 + hc) * hc;
    let linear = &text["linear_attn_config"];
    let kn = |f: &str| crate::registry::Fixture::count(linear, f);
    let (kda_heads, kda_dim, kernel) = (
        kn("num_heads"),
        kn("head_dim"),
        kn("short_conv_kernel_size"),
    );
    let (heads, q_rank, kv_rank) = (
        n("num_attention_heads"),
        n("q_lora_rank"),
        n("kv_lora_rank"),
    );
    let (qk, v) = (n("qk_nope_head_dim"), n("v_head_dim"));
    let (index_heads, index_dim, pool) =
        (n("index_n_heads"), n("index_head_dim"), n("index_kpool"));
    let (inter, moe_inter, experts, shared) = (
        n("intermediate_size"),
        n("moe_intermediate_size"),
        n("n_routed_experts"),
        n("n_shared_experts"),
    );
    use Stored::{Bf16, F32};
    let p = "model.language_model.";
    let dsa_rows = |at: &dyn Fn(&str) -> String| -> Vec<(String, Vec<usize>, Stored)> {
        vec![
            (at("self_attn.q_a_proj.weight"), vec![q_rank, w], Bf16),
            (at("self_attn.q_a_layernorm.weight"), vec![q_rank], Bf16),
            (
                at("self_attn.q_b_proj.weight"),
                vec![heads * qk, q_rank],
                Bf16,
            ),
            (
                at("self_attn.kv_a_proj_with_mqa.weight"),
                vec![kv_rank, w],
                Bf16,
            ),
            (at("self_attn.kv_a_layernorm.weight"), vec![kv_rank], Bf16),
            (
                at("self_attn.kv_b_proj.weight"),
                vec![heads * (qk + v), kv_rank],
                Bf16,
            ),
            (at("self_attn.o_proj.weight"), vec![w, heads * v], Bf16),
            (
                at("self_attn.indexer.wq_b.weight"),
                vec![index_heads * index_dim, q_rank],
                Bf16,
            ),
            (at("self_attn.indexer.wk.weight"), vec![index_dim, w], Bf16),
            (
                at("self_attn.indexer.weights_proj.weight"),
                vec![index_heads, w],
                Bf16,
            ),
            (at("self_attn.indexer.k_norm.weight"), vec![index_dim], Bf16),
            (at("self_attn.indexer.k_norm.bias"), vec![index_dim], Bf16),
            (
                at("self_attn.indexer.index_kpool_compress_gate"),
                vec![index_dim, w],
                Bf16,
            ),
            (
                at("self_attn.indexer.index_kpool_compress_ape"),
                vec![pool, index_dim],
                Bf16,
            ),
        ]
    };
    let ffn_rows =
        |at: &dyn Fn(&str) -> String, dense: bool| -> Vec<(String, Vec<usize>, Stored)> {
            if dense {
                return vec![
                    (at("mlp.gate_proj.weight"), vec![inter, w], Bf16),
                    (at("mlp.up_proj.weight"), vec![inter, w], Bf16),
                    (at("mlp.down_proj.weight"), vec![w, inter], Bf16),
                ];
            }
            let mut rows = vec![
                (at("mlp.gate.weight"), vec![experts, w], Bf16),
                (at("mlp.gate.e_score_correction_bias"), vec![experts], F32),
                (
                    at("mlp.shared_experts.gate_proj.weight"),
                    vec![moe_inter * shared, w],
                    Bf16,
                ),
                (
                    at("mlp.shared_experts.up_proj.weight"),
                    vec![moe_inter * shared, w],
                    Bf16,
                ),
                (
                    at("mlp.shared_experts.down_proj.weight"),
                    vec![w, moe_inter * shared],
                    Bf16,
                ),
            ];
            for e in 0..experts {
                rows.extend([
                    (
                        at(&format!("mlp.experts.{e}.gate_proj.weight")),
                        vec![moe_inter, w],
                        Bf16,
                    ),
                    (
                        at(&format!("mlp.experts.{e}.up_proj.weight")),
                        vec![moe_inter, w],
                        Bf16,
                    ),
                    (
                        at(&format!("mlp.experts.{e}.down_proj.weight")),
                        vec![w, moe_inter],
                        Bf16,
                    ),
                ]);
            }
            rows
        };
    let mut shapes = vec![
        (format!("{p}embed_tokens.weight"), vec![vocab, w], Bf16),
        (format!("{p}norm.weight"), vec![w], Bf16),
        ("lm_head.weight".to_string(), vec![vocab, w], Bf16),
    ];
    for l in 0..layers {
        let at = |s: &str| format!("{p}layers.{l}.{s}");
        shapes.extend([
            (at("input_layernorm.weight"), vec![w], Bf16),
            (at("post_attention_layernorm.weight"), vec![w], Bf16),
            (at("hc_attn_fn"), vec![mix, hc * w], Bf16),
            (at("hc_attn_base"), vec![mix], F32),
            (at("hc_attn_scale"), vec![3], F32),
            (at("hc_ffn_fn"), vec![mix, hc * w], Bf16),
            (at("hc_ffn_base"), vec![mix], F32),
            (at("hc_ffn_scale"), vec![3], F32),
        ]);
        match text["layer_types"][l].as_str().unwrap() {
            "linear_attention" => {
                let qkv = kda_heads * kda_dim;
                shapes.extend([
                    (at("self_attn.q_proj.weight"), vec![qkv, w], Bf16),
                    (at("self_attn.k_proj.weight"), vec![qkv, w], Bf16),
                    (at("self_attn.v_proj.weight"), vec![qkv, w], Bf16),
                    (at("self_attn.q_conv1d.weight"), vec![qkv, 1, kernel], Bf16),
                    (at("self_attn.k_conv1d.weight"), vec![qkv, 1, kernel], Bf16),
                    (at("self_attn.v_conv1d.weight"), vec![qkv, 1, kernel], Bf16),
                    (at("self_attn.f_a_proj.weight"), vec![kda_dim, w], Bf16),
                    (at("self_attn.f_b_proj.weight"), vec![qkv, kda_dim], Bf16),
                    (at("self_attn.b_proj.weight"), vec![kda_heads, w], Bf16),
                    (at("self_attn.g_a_proj.weight"), vec![kda_dim, w], Bf16),
                    (at("self_attn.g_b_proj.weight"), vec![qkv, kda_dim], Bf16),
                    (at("self_attn.A_log"), vec![kda_heads], F32),
                    (at("self_attn.dt_bias"), vec![qkv], F32),
                    (at("self_attn.o_norm.weight"), vec![kda_dim], Bf16),
                    (at("self_attn.o_proj.weight"), vec![w, qkv], Bf16),
                ]);
            }
            _ => shapes.extend(dsa_rows(&at)),
        }
        shapes.extend(ffn_rows(
            &at,
            text["mlp_layer_types"][l].as_str().unwrap() == "dense",
        ));
    }
    // The multi-token-prediction layers past the text stack: an MLA and DSA block with the last
    // text layer's feed-forward kind and no hyper-connection weights, behind the fusing projection.
    let last_dense = text["mlp_layer_types"]
        .as_array()
        .and_then(|kinds| kinds.last())
        .is_some_and(|kind| kind == "dense");
    for l in layers..layers + n("num_nextn_predict_layers") {
        let at = |s: &str| format!("{p}layers.{l}.{s}");
        shapes.extend([
            (at("eh_proj.weight"), vec![w, 2 * w], Bf16),
            (at("enorm.weight"), vec![w], Bf16),
            (at("hnorm.weight"), vec![w], Bf16),
            (at("shared_head.norm.weight"), vec![w], Bf16),
            (at("input_layernorm.weight"), vec![w], Bf16),
            (at("post_attention_layernorm.weight"), vec![w], Bf16),
        ]);
        shapes.extend(dsa_rows(&at));
        shapes.extend(ffn_rows(&at, last_dense));
    }
    shapes
}

/// A deterministic checkpoint of `config`: norm scales near one, hyper-connection scales near one,
/// every other weight small and centred, so every expert differs and a routing error is observable.
#[cfg(feature = "test-support")]
pub(crate) fn fixture_of(config: serde_json::Value) -> Fixture {
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
            } else if key.ends_with("_base") || key.ends_with("bias") {
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
        preprocessor: None,
        store: store.build(),
    }
}
