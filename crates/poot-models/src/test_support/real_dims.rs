//! Checkpoints at a real model's dimensions with all-zero weights, for tests that read only the traced
//! graph (the supported-models tables, the plan-invariant rows): the planner decides on shapes, dtypes and caps,
//! not on weight values.

use crate::architectures::qwen38_27b;
use crate::registry::Fixture;

/// The `Qwen/Qwen3.8-27B` checkpoint of `config` (its real multimodal-wrapper `config.json`): every
/// text-tower tensor at its real shape, BF16 zeros.
pub fn qwen3_8_27b(config: serde_json::Value) -> Fixture {
    let shapes = qwen38_27b::hf_shapes(&config["text_config"], "model.language_model");
    Fixture::zeroed_bf16(config, shapes)
}
