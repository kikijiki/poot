//! The reference models at their real dimensions, with all-zero weights: the planner decides on shapes,
//! dtypes and caps, not on weight values, so the graph a real checkpoint would trace is the graph these
//! models trace, and the supported-models tables compile it for the device limits of each target.
//!
//! A real model's dimensions come from its `config.json` (`Qwen/Qwen3.8-27B`'s is checked in beside the
//! poot-llm tests) or from the published dimensions of the dense checkpoints
//! (`Dense`): never from the registered fixtures, whose width is a few dozen.
#![allow(dead_code, reason = "shared by test binaries; each uses part of it")]
use std::path::Path;
use std::sync::Arc;

use poot_models::io::ProgramId;
use poot_models::model::Model;
use poot_models::registry::{Fixture, Registry};
use poot_models::test_support::dense::{Dense, Family};
use poot_models::test_support::real_dims::qwen3_8_27b;

use crate::corpus::{Fixtured, Shape};

/// Tokens of a real prefill step.
const PREFILL_TOKENS: usize = 512;

/// The steps a real model is compiled at: decode and a prefill chunk over the model's whole context,
/// the cache a served request at full length binds, and a prefill into a cache that holds only the
/// prompt.
fn shapes(context: usize) -> Vec<Shape> {
    let step = |name, tokens, capacity| Shape {
        name,
        program: ProgramId::STEP,
        tokens,
        capacity,
    };
    vec![
        step("decode", 1, context),
        step("prefill", PREFILL_TOKENS, PREFILL_TOKENS),
        step("prefill_kv", PREFILL_TOKENS, context),
    ]
}

/// `max_position_embeddings` of `Qwen/Qwen3.8-27B`.
const QWEN3_8_27B_CONTEXT: usize = 262_144;

/// The real vocabulary of the Qwen2.5 and Qwen3 dense checkpoints.
const QWEN_VOCAB: usize = 151_936;

/// The checked-in `config.json` of `Qwen/Qwen3.8-27B`, read from the workspace at run time.
fn qwen3_8_27b_config(workspace: &Path) -> serde_json::Value {
    let path = workspace.join("crates/poot-llm/tests/fixtures/qwen38_27b_config.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A dense checkpoint's model, built over zeros.
fn dense_fixture(label: &str, dense: Dense, context: usize) -> Fixtured {
    let config = dense.config();
    let model = dense.zeroed_model(poot_tensor::DType::BF16).model;
    Fixtured::at(label.to_string(), config, Arc::from(model), shapes(context))
}

/// A Mixtral checkpoint of `dense`'s llama body whose feed-forward routes every layer through `experts`
/// experts (`intermediate_size` is each expert's width), zero weights.
fn mixtral_fixture(dense: &Dense, experts: usize, per_token: usize) -> Fixture {
    let config = dense
        .clone()
        .with("model_type", "mixtral")
        .with("num_local_experts", experts)
        .with("num_experts_per_tok", per_token)
        .config();
    let count = |key| config[key].as_u64().expect("the config states its counts") as usize;
    let (hidden, inter, layers) = (
        count("hidden_size"),
        count("intermediate_size"),
        count("num_hidden_layers"),
    );
    let mut shapes: Vec<_> = dense
        .tensors()
        .into_iter()
        .filter(|(name, _)| !name.contains(".mlp."))
        .collect();
    for layer in 0..layers {
        let key = |s: &str| format!("model.layers.{layer}.block_sparse_moe.{s}");
        shapes.push((key("gate.weight"), vec![experts, hidden]));
        for expert in 0..experts {
            shapes.push((
                key(&format!("experts.{expert}.w1.weight")),
                vec![inter, hidden],
            ));
            shapes.push((
                key(&format!("experts.{expert}.w3.weight")),
                vec![inter, hidden],
            ));
            shapes.push((
                key(&format!("experts.{expert}.w2.weight")),
                vec![hidden, inter],
            ));
        }
    }
    Fixture::zeroed_bf16(config, shapes)
}

/// A real checkpoint's zero-weight fixture, built into its model.
fn built(fixture: &Fixture) -> Arc<dyn Model> {
    let parts = Registry::builtin()
        .expect("the builtin registry")
        .build(&fixture.raw(), &fixture.store)
        .expect("a real config builds over its zero checkpoint");
    Arc::from(parts.model)
}

/// Every real-dimension reference model, in matrix order. `workspace` is the checkout root.
pub(crate) fn real_models(workspace: &Path) -> Vec<Fixtured> {
    let qwen25_0_5b_context = 32_768;
    let qwen3_0_6b_context = 40_960;
    let qwen25_0_5b = Dense::new(Family::Qwen2)
        .vocab(QWEN_VOCAB)
        .dims(896, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(qwen25_0_5b_context);
    let qwen3_0_6b = Dense::new(Family::Qwen3)
        .vocab(QWEN_VOCAB)
        .dims(1024, 3072, 28)
        .heads(16, 8)
        .head_dim(128)
        .max_positions(qwen3_0_6b_context);
    let qwen3_8_27b = qwen3_8_27b(qwen3_8_27b_config(workspace));
    let mixtral = |label: &str, dense: Dense, context: usize| {
        let fixture = mixtral_fixture(&dense, 8, 2);
        Fixtured::at(
            label.to_string(),
            fixture.config.clone(),
            built(&fixture),
            shapes(context),
        )
    };
    let llama = |hidden, inter, layers, heads, vocab, context| {
        Dense::new(Family::Llama)
            .vocab(vocab)
            .dims(hidden, inter, layers)
            .heads(heads, 8)
            .max_positions(context)
    };
    vec![
        dense_fixture("Qwen2.5-0.5B", qwen25_0_5b, qwen25_0_5b_context),
        dense_fixture("Qwen3-0.6B", qwen3_0_6b, qwen3_0_6b_context),
        Fixtured::at(
            "Qwen3.8-27B".to_string(),
            qwen3_8_27b.config.clone(),
            built(&qwen3_8_27b),
            shapes(QWEN3_8_27B_CONTEXT),
        ),
        mixtral(
            "Mixtral-8x7B",
            llama(4096, 14_336, 32, 32, 32_000, 32_768),
            32_768,
        ),
        mixtral(
            "Mixtral-8x22B",
            llama(6144, 16_384, 56, 48, 32_768, 65_536),
            65_536,
        ),
    ]
}
