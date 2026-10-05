//! Card 550a SC-002: over the traced corpus, the count of tracer-synthesized
//! `Storage::Const` inputs whose `aval` depends on the call reads 0.
//!
//! Every row traces one production family graph at two call sizes (prompt length and KV capacity)
//! and diffs the two graphs' `Storage::Const` inputs by name: a const present at only one size, or
//! with a different declared shape, is a call-dependent const the card exists to remove (the
//! baked-store growth of). Per-model tables (`rope.cos`/`rope.sin`, weights) are fixed
//! by the config and match across both calls; per-call tables must now be step inputs
//! (`Slot::Mask`/`Slot::Activation`), in-graph `iota` computations, or absent.
//!
//! The corpus can only trace what this crate exposes (no `#[cfg(test)]` tracers): DeepSeek-V4,
//! GLM-5.3, qwen38_27b's prefill arm and the qsa probe tracers are test-only and are counted by
//! their in-crate tests' own assertions instead.

use std::collections::BTreeMap;

use poot_executor_parity::dense::{Dense, Family, paged_step, plain, step};
use poot_graph_ir::{Graph, Storage};
use poot_models::bloom::{BloomConfig, trace_bloom_decode_kv_masked, trace_bloom_prefill};
use poot_models::deepseek2::{
    DeepseekV2Config, DeepseekV2MoeParams, trace_deepseek2_decode_kv_masked,
    trace_deepseek2_prefill,
};
use poot_models::deepseek3::{
    DeepseekV3MoeParams, trace_deepseek3_decode_kv_masked,
    trace_deepseek3_decode_kv_masked_batched_shared_pool, trace_deepseek3_prefill,
    trace_deepseek3_prefill_kv_shared_pool,
};
use poot_models::deepseek32::{
    DsaConfig, trace_deepseek32_dsa_decode, trace_deepseek32_dsa_prefill,
};
use poot_models::gpt_oss::{GptOssParams, trace_gptoss_decode_kv_masked, trace_gptoss_prefill};
use poot_models::granite::{
    GraniteParams, MoeShape, trace_granite_decode_kv_masked, trace_granite_prefill,
    trace_granite_prefill_kv, trace_granite_prefill_kv_shared_pool,
};
use poot_models::mixtral::{
    MixtralParams, trace_mixtral_decode_kv_masked, trace_mixtral_prefill,
    trace_mixtral_prefill_kv_shared_pool,
};
use poot_models::model::{LogitRows, Phase};
use poot_models::mpt::{MptConfig, trace_mpt_decode_kv_masked, trace_mpt_prefill};
use poot_models::nemotron_h::{
    NemotronHAttnConfig, NemotronHConfig, NemotronHLayerKind, NemotronHMambaConfig,
    trace_nemotron_h_decode, trace_nemotron_h_prefill,
};
use poot_models::olmoe::{OlmoeParams, trace_olmoe_decode_kv_masked, trace_olmoe_prefill};
use poot_models::qwen2::{
    trace_prefill_kv_embeds, trace_prefill_mrope, trace_qwen2_5_vl_prefill_kv_embeds,
};
use poot_models::qwen3moe::{
    Qwen3MoeParams, trace_qwen3_moe_decode_kv_masked, trace_qwen3_moe_prefill,
    trace_qwen3_moe_prefill_kv, trace_qwen3_moe_prefill_kv_shared_pool,
};
use poot_models::qwen3next::{Qwen3NextConfig, qwen3next_decode_trace};
use poot_models::qwen38::{
    QsaConfig, Qwen4ExpConfig, Qwen4ExpGdnConfig, Qwen4ExpModelConfig, Qwen4ExpPleConfig,
    trace_qwen38_decode, trace_qwen38_prefill,
};
use poot_models::qwen38_27b::{Qwen35TextConfig, trace_qwen38_text_decode};
use poot_models::smollm3::{Smollm3Config, trace_smollm3_decode_kv_masked, trace_smollm3_prefill};

const LAYERS: usize = 2;

/// Call A: prompt length and KV capacity.
const PROMPT_A: usize = 4;
const CAP_A: usize = 8;
/// Call B: a different prompt length and capacity (every row's multiples-of constraints hold).
const PROMPT_B: usize = 6;
const CAP_B: usize = 12;

struct Row {
    name: &'static str,
    build: fn(prompt: usize, cap: usize) -> Graph,
}

fn base_cfg() -> poot_models::qwen2::Qwen2Config {
    poot_models::qwen2::Qwen2Config {
        vocab: 24,
        hidden: 16,
        inter: 16,
        layers: LAYERS,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 6,
        rotary_dim: 6,
        eps: 1e-5,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..poot_models::qwen2::Qwen2Config::default()
    }
}

fn qwen2_cfg() -> poot_models::qwen2::Qwen2Config {
    poot_models::qwen2::Qwen2Config {
        qkv_bias: true,
        ..base_cfg()
    }
}

fn qwen2_mrope_cfg() -> poot_models::qwen2::Qwen2Config {
    poot_models::qwen2::Qwen2Config {
        // section widths must sum to rotary_dim / 2 (= 3 for `base_cfg`).
        mrope_section: Some([1, 1, 1]),
        ..qwen2_cfg()
    }
}

/// A dense family at the corpus dimensions, traced through `Model::trace` at `shape`.
fn dense_trace(family: Family, phase: Phase, shape: poot_models::model::StepShape) -> Graph {
    let mut dense = Dense::new(family)
        .vocab(24)
        .dims(16, 16, LAYERS)
        .heads(4, 2)
        .max_positions(32);
    // OLMo 2 normalizes the whole projection width, so its head width is `hidden / heads`.
    if family != Family::Olmo2 {
        dense = dense.head_dim(6);
    }
    plain(
        dense
            .f32_model()
            .model
            .trace(phase, shape)
            .unwrap_or_else(|e| panic!("{family:?}: {e}")),
    )
}

fn granite_params(moe: Option<MoeShape>) -> GraniteParams {
    GraniteParams {
        moe,
        embed_mult: 1.0,
        attn_mult: 1.0,
        residual_mult: 1.0,
        logits_scale: 1.0,
    }
}

fn granite_moe() -> MoeShape {
    MoeShape {
        n_experts: 6,
        top_k: 2,
        inter: 12,
    }
}

fn qwen3_moe_params() -> Qwen3MoeParams {
    Qwen3MoeParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        sparse_layer: vec![false, true],
    }
}

fn mixtral_params() -> MixtralParams {
    MixtralParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
    }
}

fn olmoe_params() -> OlmoeParams {
    OlmoeParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        norm_topk_prob: false,
    }
}

fn gpt_oss_params() -> GptOssParams {
    GptOssParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        swiglu_limit: 2.5,
        sliding_window: 2,
        layer_is_sliding: vec![true, false],
    }
}

fn bloom_cfg() -> BloomConfig {
    BloomConfig {
        vocab: 24,
        hidden: 16,
        n_heads: 4,
        layers: LAYERS,
        ffn_inter: 16,
        eps: 1e-5,
    }
}

fn mpt_cfg() -> MptConfig {
    MptConfig {
        vocab: 24,
        hidden: 16,
        n_heads: 4,
        layers: LAYERS,
        ffn_inter: 16,
        eps: 1e-5,
    }
}

fn smollm3_cfg() -> Smollm3Config {
    Smollm3Config {
        vocab: 24,
        hidden: 16,
        inter: 16,
        layers: LAYERS,
        n_heads: 4,
        n_kv_heads: 2,
        eps: 1e-6,
        max_pos: 32,
        use_rope: vec![true, false],
    }
}

fn deepseek_cfg(q_lora_rank: Option<usize>) -> DeepseekV2Config {
    DeepseekV2Config {
        vocab: 24,
        hidden: 16,
        layers: LAYERS,
        n_heads: 4,
        q_lora_rank,
        kv_lora_rank: 6,
        qk_nope_head_dim: 5,
        qk_rope_head_dim: 4,
        v_head_dim: 7,
        eps: 1e-5,
        max_pos: 32,
        rope_theta: 10_000.0,
        yarn: None,
    }
}

fn deepseek2_moe() -> DeepseekV2MoeParams {
    DeepseekV2MoeParams {
        n_routed_experts: 6,
        top_k: 2,
        moe_inter: 8,
        n_shared_experts: 2,
        dense_inter: 10,
        first_k_dense_replace: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    }
}

fn deepseek3_moe() -> DeepseekV3MoeParams {
    DeepseekV3MoeParams {
        n_routed_experts: 8,
        top_k: 3,
        moe_inter: 6,
        n_shared_experts: 1,
        dense_inter: 10,
        first_k_dense_replace: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    }
}

fn dsa_cfg() -> DsaConfig {
    DsaConfig {
        index_n_heads: 2,
        index_head_dim: 4,
        index_topk: 3,
    }
}

fn nemotron_h_cfg() -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 12,
        hidden: 6,
        mlp_inter: 10,
        eps: 1e-6,
        pattern: vec![
            NemotronHLayerKind::Mamba,
            NemotronHLayerKind::Attention,
            NemotronHLayerKind::Mlp,
        ],
        mamba: NemotronHMambaConfig {
            hidden: 6,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            ssm_state: 3,
            conv_kernel: 3,
        },
        attn: NemotronHAttnConfig {
            hidden: 6,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
        },
    }
}

fn qwen38_cfg_with_ple(ple: Option<Qwen4ExpPleConfig>) -> Qwen4ExpModelConfig {
    Qwen4ExpModelConfig {
        cfg: Qwen4ExpConfig {
            hidden: 8,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 2,
            eps: 1e-5,
        },
        qcfg: QsaConfig {
            index_n_heads: 2,
            index_kv_heads: 1,
            index_head_dim: 3,
            index_budget: 4,
            index_compress_ratio: 2,
        },
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4,
        hc_count: 2,
        hc_lowrank: 3,
        ple,
    }
}

/// The in-crate `tiny_ple` values (all fields are `pub`): real one-indexed layer id, tiny tables.
fn tiny_ple() -> Qwen4ExpPleConfig {
    Qwen4ExpPleConfig {
        ple_layer_ids: vec![2],
        ple_embed_dim: 8,
        ple_conv_kernel_size: 3,
        ngram_size: 3,
        ngram_vocab_size_base: 17,
        heads_per_ngram: 2,
        make_ngram_vocab_size_divisible_by: 8,
        seed: 1234,
        eos_token_id: 5,
        vocab_size: 11,
    }
}

fn qwen3next_cfg() -> Qwen3NextConfig {
    Qwen3NextConfig {
        vocab: 6,
        hidden: 12,
        n_layers: 4,
        full_attention_interval: 4,
        eps: 1e-5,
        max_pos: 32,
        rotary_dim: 4,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        gdn_num_k_heads: 2,
        gdn_num_v_heads: 4,
        gdn_head_dim: 4,
        conv_k: 3,
        n_experts: 4,
        top_k: 2,
        expert_inter: 8,
        shared_inter: 8,
    }
}

/// The named corpus: every production family tracer whose graph this crate can reach, at two call
/// sizes. Decode rows ignore `prompt`; prefill rows may use both.
fn corpus() -> Vec<Row> {
    vec![
        Row {
            name: "qwen2.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.decode_batched",
            build: |_, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Decode,
                    step(2, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.decode_paged",
            build: |_, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Decode,
                    paged_step(2, 1, cap, 2 * cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "qwen2.prefill_paged",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Prefill,
                    paged_step(1, prompt, cap, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.prefill_batched_paged",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Prefill,
                    paged_step(2, prompt, cap, 2 * cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.prefill_kv_embeds",
            build: |prompt, cap| trace_prefill_kv_embeds(qwen2_cfg(), prompt, cap),
        },
        Row {
            name: "qwen2.prefill_mrope",
            build: |prompt, _| trace_prefill_mrope(qwen2_mrope_cfg(), prompt),
        },
        Row {
            name: "qwen2.vl_prefill_kv_embeds",
            build: |prompt, cap| trace_qwen2_5_vl_prefill_kv_embeds(qwen2_mrope_cfg(), prompt, cap),
        },
        Row {
            name: "qwen3.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen3,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "olmo2.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Olmo2,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "olmo2.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Olmo2,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "gemma3.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Gemma3,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "gemma3.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Gemma3,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "gemma2.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Gemma2,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "gemma2.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Gemma2,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "granite.prefill",
            build: |prompt, _| trace_granite_prefill(base_cfg(), granite_params(None), prompt),
        },
        Row {
            name: "granite.prefill_kv",
            build: |prompt, cap| {
                trace_granite_prefill_kv(base_cfg(), granite_params(None), prompt, cap)
            },
        },
        Row {
            name: "granite.prefill_shared_pool",
            build: |prompt, cap| {
                trace_granite_prefill_kv_shared_pool(base_cfg(), granite_params(None), prompt, cap)
            },
        },
        Row {
            name: "granite.decode",
            build: |_, cap| trace_granite_decode_kv_masked(base_cfg(), granite_params(None), cap),
        },
        Row {
            name: "granitemoe.prefill",
            build: |prompt, _| {
                trace_granite_prefill(base_cfg(), granite_params(Some(granite_moe())), prompt)
            },
        },
        Row {
            name: "granitemoe.decode",
            build: |_, cap| {
                trace_granite_decode_kv_masked(base_cfg(), granite_params(Some(granite_moe())), cap)
            },
        },
        Row {
            name: "qwen3_moe.prefill",
            build: |prompt, _| trace_qwen3_moe_prefill(base_cfg(), qwen3_moe_params(), prompt),
        },
        Row {
            name: "qwen3_moe.prefill_kv",
            build: |prompt, cap| {
                trace_qwen3_moe_prefill_kv(base_cfg(), qwen3_moe_params(), prompt, cap)
            },
        },
        Row {
            name: "qwen3_moe.prefill_shared_pool",
            build: |prompt, cap| {
                trace_qwen3_moe_prefill_kv_shared_pool(base_cfg(), qwen3_moe_params(), prompt, cap)
            },
        },
        Row {
            name: "qwen3_moe.decode",
            build: |_, cap| trace_qwen3_moe_decode_kv_masked(base_cfg(), qwen3_moe_params(), cap),
        },
        Row {
            name: "mixtral.prefill",
            build: |prompt, _| trace_mixtral_prefill(base_cfg(), mixtral_params(), prompt),
        },
        Row {
            name: "mixtral.prefill_shared_pool",
            build: |prompt, cap| {
                trace_mixtral_prefill_kv_shared_pool(base_cfg(), mixtral_params(), prompt, cap)
            },
        },
        Row {
            name: "mixtral.decode",
            build: |_, cap| trace_mixtral_decode_kv_masked(base_cfg(), mixtral_params(), cap),
        },
        Row {
            name: "olmoe.prefill",
            build: |prompt, _| trace_olmoe_prefill(base_cfg(), olmoe_params(), prompt),
        },
        Row {
            name: "olmoe.decode",
            build: |_, cap| trace_olmoe_decode_kv_masked(base_cfg(), olmoe_params(), cap),
        },
        Row {
            name: "gpt_oss.prefill",
            build: |prompt, _| trace_gptoss_prefill(base_cfg(), gpt_oss_params(), prompt),
        },
        Row {
            name: "gpt_oss.decode",
            build: |_, cap| trace_gptoss_decode_kv_masked(base_cfg(), gpt_oss_params(), cap),
        },
        Row {
            name: "bloom.prefill",
            build: |prompt, _| trace_bloom_prefill(&bloom_cfg(), prompt),
        },
        Row {
            name: "bloom.decode",
            build: |_, cap| trace_bloom_decode_kv_masked(&bloom_cfg(), cap),
        },
        Row {
            name: "mpt.prefill",
            build: |prompt, _| trace_mpt_prefill(&mpt_cfg(), prompt),
        },
        Row {
            name: "mpt.decode",
            build: |_, cap| trace_mpt_decode_kv_masked(&mpt_cfg(), cap),
        },
        Row {
            name: "smollm3.prefill",
            build: |prompt, _| trace_smollm3_prefill(&smollm3_cfg(), prompt),
        },
        Row {
            name: "smollm3.decode",
            build: |_, cap| trace_smollm3_decode_kv_masked(&smollm3_cfg(), cap),
        },
        Row {
            name: "deepseek2.prefill",
            build: |prompt, _| {
                trace_deepseek2_prefill(deepseek_cfg(Some(3)), deepseek2_moe(), prompt)
            },
        },
        Row {
            name: "deepseek2.decode",
            build: |_, cap| {
                trace_deepseek2_decode_kv_masked(deepseek_cfg(Some(3)), deepseek2_moe(), cap)
            },
        },
        Row {
            name: "deepseek3.prefill",
            build: |prompt, _| {
                trace_deepseek3_prefill(deepseek_cfg(Some(3)), deepseek3_moe(), prompt)
            },
        },
        Row {
            name: "deepseek3.prefill_shared_pool",
            build: |prompt, cap| {
                trace_deepseek3_prefill_kv_shared_pool(
                    deepseek_cfg(Some(3)),
                    deepseek3_moe(),
                    prompt,
                    cap,
                )
            },
        },
        Row {
            name: "deepseek3.decode",
            build: |_, cap| {
                trace_deepseek3_decode_kv_masked(deepseek_cfg(Some(3)), deepseek3_moe(), cap)
            },
        },
        Row {
            name: "deepseek3.decode_batched_pool",
            build: |_, cap| {
                trace_deepseek3_decode_kv_masked_batched_shared_pool(
                    deepseek_cfg(Some(3)),
                    deepseek3_moe(),
                    cap,
                    2,
                    cap,
                )
            },
        },
        Row {
            name: "deepseek32.prefill",
            build: |prompt, _| {
                trace_deepseek32_dsa_prefill(
                    deepseek_cfg(Some(5)),
                    dsa_cfg(),
                    deepseek3_moe(),
                    prompt,
                )
            },
        },
        Row {
            name: "deepseek32.decode",
            build: |_, cap| {
                trace_deepseek32_dsa_decode(deepseek_cfg(Some(5)), dsa_cfg(), deepseek3_moe(), cap)
            },
        },
        Row {
            name: "nemotron_h.prefill",
            build: |prompt, _| trace_nemotron_h_prefill(&nemotron_h_cfg(), prompt),
        },
        Row {
            name: "nemotron_h.decode",
            build: |_, cap| trace_nemotron_h_decode(&nemotron_h_cfg(), cap),
        },
        Row {
            name: "qwen38.prefill",
            build: |prompt, _| trace_qwen38_prefill(&qwen38_cfg_with_ple(None), prompt),
        },
        Row {
            name: "qwen38.prefill_ple",
            build: |prompt, _| trace_qwen38_prefill(&qwen38_cfg_with_ple(Some(tiny_ple())), prompt),
        },
        Row {
            name: "qwen38.decode",
            build: |_, cap| trace_qwen38_decode(&qwen38_cfg_with_ple(None), cap),
        },
        Row {
            name: "qwen38.decode_ple",
            build: |_, cap| trace_qwen38_decode(&qwen38_cfg_with_ple(Some(tiny_ple())), cap),
        },
        Row {
            name: "qwen3next.decode",
            build: |_, cap| qwen3next_decode_trace(&qwen3next_cfg(), 1, cap),
        },
        Row {
            name: "qwen38_27b.decode",
            build: |_, cap| {
                trace_qwen38_text_decode(&Qwen35TextConfig::pinned(), 1, cap)
                    .expect("the pinned config traces")
            },
        },
    ]
}

/// Every `Storage::Const` input of `g`, keyed by name with its declared shape.
fn const_avals(g: &Graph) -> BTreeMap<String, Vec<usize>> {
    g.inputs
        .iter()
        .filter_map(|&id| {
            let m = g.meta(id);
            if m.storage != Storage::Const {
                return None;
            }
            let name = m.name.clone()?;
            Some((name, m.aval.shape.clone()))
        })
        .collect()
}

/// The call-dependent `Storage::Const` inputs of the (call A, call B) graph pair: a const missing at
/// one size or with a shape that changed between them.
fn call_dependent_consts(a: &Graph, b: &Graph) -> Vec<String> {
    let (ca, cb) = (const_avals(a), const_avals(b));
    let mut found = Vec::new();
    for (name, shape_a) in &ca {
        match cb.get(name) {
            None => found.push(format!(
                "{name}: present at call A only (shape {shape_a:?})"
            )),
            Some(shape_b) if shape_b != shape_a => found.push(format!(
                "{name}: shape {shape_a:?} at call A vs {shape_b:?} at call B"
            )),
            Some(_) => {}
        }
    }
    for name in cb.keys() {
        if !ca.contains_key(name) {
            found.push(format!("{name}: present at call B only"));
        }
    }
    found.sort();
    found
}

/// SC-002: the count over every family's traced graphs reads 0. Mutation: restore one
/// `b.constant("causal.mask", ..)` in any tracer above and this row reads its name once.
#[test]
fn traced_corpus_counts_zero_call_dependent_const_avals() {
    let rows = corpus();
    assert!(
        rows.len() >= 55,
        "the corpus must cover every family's tracers; got {} rows",
        rows.len()
    );
    let mut failures = Vec::new();
    for row in &rows {
        let a = (row.build)(PROMPT_A, CAP_A);
        let b = (row.build)(PROMPT_B, CAP_B);
        for found in call_dependent_consts(&a, &b) {
            failures.push(format!("{}: {found}", row.name));
        }
    }
    assert_eq!(
        failures.len(),
        0,
        "tracer-synthesized consts whose aval depends on the call (S46-3's count must read 0):\n{}",
        failures.join("\n")
    );
}

/// Card 550's dense families, by this corpus's row names: every row card 550 converted to the new
/// Pos/mask contract (`Slot::Pos` drives an in-graph `causal_mask_from_pos`/`alibi_mask_from_pos`
/// instead of a host-built `Slot::Mask` step input or tracer-synthesized mask `Storage::Const`).
/// Deliberately excluded: `granite.prefill_shared_pool` (delegates to the MoE shared-pool infra,
/// deferred to card 567).
const CARD_550_DENSE_ROWS: &[&str] = &[
    "qwen2.decode",
    "qwen2.decode_batched",
    "qwen2.decode_paged",
    "qwen2.prefill",
    "qwen2.prefill_paged",
    "qwen2.prefill_batched_paged",
    "qwen2.prefill_kv_embeds",
    "qwen2.prefill_mrope",
    "qwen2.vl_prefill_kv_embeds",
    "qwen3.prefill",
    "olmo2.prefill",
    "olmo2.decode",
    "gemma3.prefill",
    "gemma3.decode",
    "gemma2.prefill",
    "gemma2.decode",
    "granite.prefill",
    "granite.prefill_kv",
    "granite.decode",
    "bloom.prefill",
    "bloom.decode",
    "mpt.prefill",
    "mpt.decode",
    "smollm3.prefill",
    "smollm3.decode",
];

/// SC-005: over card 550's dense families' traced graphs, no mask is a step input any more: the
/// count of `Slot::Mask` step inputs and tracer-synthesized mask `Storage::Const` inputs (named
/// `mask.*`/`*.mask`/`causal.mask`, the pre-card convention) reads 0. Mutation: restore the `Slot::Mask`
/// step input in one qwen2 prefill tracer; the count reads 1 and the row goes red.
#[test]
fn card_550_dense_corpus_counts_zero_mask_step_inputs_and_consts() {
    let rows = corpus();
    let dense: Vec<&Row> = rows
        .iter()
        .filter(|r| CARD_550_DENSE_ROWS.contains(&r.name))
        .collect();
    assert_eq!(
        dense.len(),
        CARD_550_DENSE_ROWS.len(),
        "expected every name in CARD_550_DENSE_ROWS to match exactly one corpus row"
    );

    let mut mask_count = 0usize;
    let mut offenders = Vec::new();
    for row in &dense {
        let g = (row.build)(PROMPT_A, CAP_A);
        for &id in &g.inputs {
            let m = g.meta(id);
            let is_mask_slot = matches!(m.storage, Storage::Slot(poot_graph_ir::Slot::Mask));
            let is_mask_const = m.storage == Storage::Const
                && m.name
                    .as_deref()
                    .is_some_and(|n| n.contains("mask") || n == "causal.mask");
            if is_mask_slot || is_mask_const {
                mask_count += 1;
                offenders.push(format!("{}: {:?} (name={:?})", row.name, m.storage, m.name));
            }
        }
    }
    assert_eq!(
        mask_count,
        0,
        "card 550's dense families must have zero Mask step inputs/consts:\n{}",
        offenders.join("\n")
    );
}
