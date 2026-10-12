//! DeepSeek-V3 behind [`Model`]: V2's MLA attention and routed MoE (see [`super::deepseek2`], which
//! holds the body this family shares) with a sigmoid router: `sigmoid(logits)` scores, a learned
//! per-expert bias that shifts which experts are selected but not the weight they contribute,
//! group-limited selection by the sum of each group's two best biased scores, the selected
//! unbiased scores renormalized (`norm_topk_prob`) and scaled by `routed_scaling_factor`.

use super::deepseek2::{Kind, Scoring, build_family, hf_shapes};
use crate::model::{FamilyKey, Model, ModelError};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_quant::weights::WeightStore;

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("deepseek3");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "deepseek_v3")],
    build,
    fixture,
    variants: &[],
};

const KIND: Kind = Kind {
    family: FAMILY,
    scoring: Scoring::Sigmoid,
    dsa: false,
};

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    build_family(KIND, raw, store)
}

/// The fixture's config: a tiny V3 with a low-rank query, sigmoid routing with a selection bias,
/// `noaux_tc` group limiting, one shared expert and a renormalized, scaled gate.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v3",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "n_group": 4,
        "topk_group": 2,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 16,
            "mscale": 1.0,
            "mscale_all_dim": 1.0
        },
        "tie_word_embeddings": false,
        "eos_token_id": 47,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_shapes(&config, true, false);
    Fixture::bf16(config, shapes)
}

// ---- test references ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::super::deepseek2::family_tests::*;
    use super::*;
    use poot_graph_ir::ops::sigmoid;
    use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced};

    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        assert_missing_tensors_are_named(
            &fixture(),
            &[
                "model.layers.1.mlp.gate.weight",
                "model.layers.1.mlp.gate.e_score_correction_bias",
                "model.layers.1.mlp.experts.3.up_proj.weight",
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "model.layers.1.self_attn.q_b_proj.weight",
                "lm_head.weight",
            ],
        );
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        assert_chunks_equal_decode(&ENTRY, &fixture());
    }

    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body() {
        assert_bf16_and_q8_0_trace_and_match(&ENTRY, &fixture());
    }

    /// A `deepseek2`-architecture GGUF that declares the sigmoid gating function and carries
    /// `exp_probs_b.bias` is a V3 checkpoint: it resolves to the `deepseek2` family, builds the
    /// sigmoid router from its gating function and equals the HF V3 body over the same values.
    #[test]
    fn a_v3_gguf_equals_the_hf_body() {
        assert_gguf_equals_the_hf_body(&ENTRY, &fixture());
    }

    #[test]
    fn a_config_deepseek_cannot_trace_is_a_typed_error_naming_the_field() {
        assert_bad_configs_are_typed(&ENTRY, &fixture());
    }

    #[test]
    fn the_routing_scale_reaches_the_logits() {
        assert_the_routing_scale_reaches_the_logits(&ENTRY, &fixture());
    }

    /// SC-001 (tier 3 on the oracle): the pre-driver engine's recorded tokens.
    #[test]
    fn the_fixture_generates_the_tokens_the_runner_recorded() {
        assert_generates_the_recorded_tokens(
            &ENTRY,
            &fixture(),
            [
                11, 2, 11, 2, 11, 2, 11, 32, 35, 5, 15, 32, 23, 12, 2, 20, 18, 32, 27, 2, 38, 12,
                2, 20,
            ],
        );
    }

    /// DeepSeek-V3's routed-expert gate (`topk_method: "noaux_tc"`, `scoring_func: "sigmoid"`, with
    /// group-limited routing) and the rank-ordered expert ids: the independent reference the router
    /// component's sigmoid arm is compared against. The family itself routes through
    /// `components::router`.
    pub(crate) fn deepseek3_router_gate_and_ids(
        b: &Builder,
        logits: Traced,
        bias: Traced,
        k: usize,
        n_group: usize,
        topk_group: usize,
        routed_scaling_factor: f32,
    ) -> (Traced, Traced) {
        let (p, rank) = deepseek3_router_rank(b, logits, bias, n_group, topk_group);
        let ids = b.arg_top_k(rank, k).unwrap();
        (
            deepseek3_router_weights(b, p, rank, k, routed_scaling_factor),
            ids,
        )
    }

    /// The router's unbiased sigmoid scores and the stable rank of the group-masked biased scores.
    fn deepseek3_router_rank(
        b: &Builder,
        logits: Traced,
        bias: Traced,
        n_group: usize,
        topk_group: usize,
    ) -> (Traced, Traced) {
        let shape = b.aval(logits).unwrap().shape;
        let e = *shape.last().expect("router logits have an expert axis");
        let last = shape.len() - 1;
        assert!(
            n_group > 0 && e.is_multiple_of(n_group),
            "n_routed_experts ({e}) must be evenly divisible by n_group ({n_group})"
        );
        let group_size = e / n_group;

        let p = sigmoid(b, logits).unwrap(); // [.., E] - the UNBIASED score used for the final weight

        let mut bias_shape = vec![1usize; shape.len()];
        bias_shape[last] = e;
        let bias_r = b.reshape(bias, bias_shape).unwrap();
        let bias_b = b.broadcast(bias_r, shape.clone()).unwrap();
        let sc = b.binary(BinOp::Add, p, bias_b).unwrap(); // scores_for_choice: selection-only

        // group_scores = sc.view(.., n_group, group_size).topk(2, -1)[0].sum(-1) (MoEGate.forward)
        let mut group_shape = shape.clone();
        group_shape[last] = n_group;
        group_shape.push(group_size); // [.., n_group, group_size]
        let sc_grouped = b.reshape(sc, group_shape.clone()).unwrap();
        let in_group_rank = poot_graph_ir::ops::stable_descending_rank(b, sc_grouped).unwrap();
        let top2 = poot_graph_ir::ops::top_k_keep_mask(b, in_group_rank, 2).unwrap();
        let top2_vals = b.binary(BinOp::Mul, sc_grouped, top2).unwrap();
        let group_scores = b.reduce(RedOp::Sum, top2_vals, last + 1, false).unwrap(); // [.., n_group]

        // group_idx = group_scores.topk(topk_group, -1)[1]; mask scores_for_choice to the selected groups.
        let group_rank = poot_graph_ir::ops::stable_descending_rank(b, group_scores).unwrap();
        let group_keep = poot_graph_ir::ops::top_k_keep_mask(b, group_rank, topk_group).unwrap();

        let mut group_keep_shape = shape.clone();
        group_keep_shape[last] = n_group;
        group_keep_shape.push(1);
        let group_keep_r = b.reshape(group_keep, group_keep_shape).unwrap();
        let group_keep_b = b.broadcast(group_keep_r, group_shape).unwrap();
        let expert_keep = b.reshape(group_keep_b, shape.clone()).unwrap(); // [.., E], 1.0 iff expert's group is selected

        let not_keep = b
            .binary_scalar(BinOp::Mul, expert_keep, Scalar::F32(-1.0))
            .unwrap();
        let not_keep = b
            .binary_scalar(BinOp::Add, not_keep, Scalar::F32(1.0))
            .unwrap();
        let group_drop = b
            .binary_scalar(BinOp::Mul, not_keep, Scalar::F32(-1.0e30))
            .unwrap();
        let sc_group_masked = b.binary(BinOp::Add, sc, group_drop).unwrap(); // -1e30 outside the selected groups

        // topk(top_k) over the group-masked scores -> topk_idx
        let rank = poot_graph_ir::ops::stable_descending_rank(b, sc_group_masked).unwrap(); // [.., E]
        (p, rank)
    }

    /// Keep the top-`k` unbiased scores by rank, renormalize them, and apply the routed scaling factor.
    fn deepseek3_router_weights(
        b: &Builder,
        p: Traced,
        rank: Traced,
        k: usize,
        routed_scaling_factor: f32,
    ) -> Traced {
        let last = b.aval(p).unwrap().shape.len() - 1;
        let keep = poot_graph_ir::ops::top_k_keep_mask(b, rank, k).unwrap();

        let masked_p = b.binary(BinOp::Mul, p, keep).unwrap(); // UNBIASED score, zeroed outside top-k

        let denom = b.reduce(RedOp::Sum, masked_p, last, true).unwrap();
        let denom = b
            .binary_scalar(BinOp::Add, denom, Scalar::F32(1e-20))
            .unwrap();
        let normed = b.binary(BinOp::Div, masked_p, denom).unwrap(); // renormalized to sum to 1.0 over the kept experts

        b.binary_scalar(BinOp::Mul, normed, Scalar::F32(routed_scaling_factor))
            .unwrap()
    }
}
