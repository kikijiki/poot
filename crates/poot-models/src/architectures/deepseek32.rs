//! DeepSeek-V3.2 behind [`Model`]: V3's MLA attention and sigmoid-routed MoE (see
//! [`super::deepseek2`], which holds the body the DeepSeek families share) with DeepSeek sparse
//! attention ([`crate::components::dsa`]): a lightning indexer scores the cached positions and
//! each query attends to the `index_topk` best of them. The indexer reads the low-rank query, so a
//! config without `q_lora_rank` is refused.

use super::deepseek2::{Kind, Scoring, build_family, hf_shapes};
use crate::model::{FamilyKey, Model, ModelError};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_quant::weights::WeightStore;

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("deepseek32");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "deepseek_v32")],
    build,
    fixture,
    variants: &[],
};

const KIND: Kind = Kind {
    family: FAMILY,
    scoring: Scoring::Sigmoid,
    dsa: true,
};

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    build_family(KIND, raw, store)
}

/// The fixture's config: the V3 fixture plus an indexer of 2 heads of 16 dims keeping 3 positions
/// (fewer than the longest prompt the rows trace, so the selection drops positions).
pub(crate) fn fixture_config() -> serde_json::Value {
    let mut config = super::deepseek3::fixture_config();
    config["model_type"] = serde_json::json!("deepseek_v32");
    config["index_n_heads"] = serde_json::json!(2);
    config["index_head_dim"] = serde_json::json!(16);
    config["index_topk"] = serde_json::json!(3);
    config
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_shapes(&config, true, true);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::deepseek2::family_tests::*;
    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{CAP, build};
    use crate::model::{ConfigReason, Phase};

    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        assert_missing_tensors_are_named(
            &fixture(),
            &[
                "model.layers.0.self_attn.indexer.wq_b.weight",
                "model.layers.1.self_attn.indexer.wk.weight",
                "model.layers.1.self_attn.indexer.k_norm.weight",
                "model.layers.1.self_attn.indexer.k_norm.bias",
                "model.layers.0.self_attn.indexer.weights_proj.weight",
                "model.layers.1.mlp.gate.e_score_correction_bias",
                "model.layers.1.self_attn.kv_b_proj.weight",
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

    #[test]
    fn a_config_deepseek_cannot_trace_is_a_typed_error_naming_the_field() {
        assert_bad_configs_are_typed(&ENTRY, &fixture());
    }

    /// The selection reaches the logits: keeping one position per query gives other logits than
    /// keeping every position (`index_topk` above the longest prompt), on the same weights. A
    /// selection that did not reach the attention would leave them equal. Mutation: return the
    /// causal mask unchanged from `dsa_select`; the two runs agree and the row goes red.
    #[test]
    fn the_indexer_selection_reaches_the_logits() {
        let fx = fixture();
        let run = |topk: u64| {
            let mut config = fx.config.clone();
            config["index_topk"] = json!(topk);
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
            };
            let model = build(&ENTRY, &raw, &fx.store);
            run_step(
                &*model,
                &fx.store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40, 8, 25, 9],
                0,
                &[],
            )
            .0
        };
        assert_ne!(run(1), run(64));
        // Keeping every position is the full-attention model: a selection wider than the longest
        // prompt does not change with its width.
        assert_eq!(run(32), run(64));
    }

    /// SC-001 (tier 3 on the oracle): the pre-driver engine's recorded tokens.
    #[test]
    fn the_fixture_generates_the_tokens_the_runner_recorded() {
        assert_generates_the_recorded_tokens(
            &ENTRY,
            &fixture(),
            [
                14, 20, 2, 14, 9, 37, 19, 35, 11, 34, 33, 14, 17, 33, 14, 20, 46, 3, 17, 0, 1, 33,
                14, 8,
            ],
        );
    }

    /// DSA needs the low-rank query: a config without `q_lora_rank` is a typed refusal.
    #[test]
    fn a_dsa_config_without_the_low_rank_query_is_refused() {
        let fx = fixture();
        let mut config = fx.config.clone();
        config.as_object_mut().unwrap().remove("q_lora_rank");
        let raw = RawConfig::HfJson {
            config: &config,
            generation: None,
        };
        match (ENTRY.build)(&raw, &fx.store) {
            Err(ModelError::Config { field, reason, .. }) => {
                assert_eq!((field, reason), ("q_lora_rank", ConfigReason::Missing));
            }
            other => panic!("expected a config error, got {other:?}"),
        }
    }
}
