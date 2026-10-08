//! Card 567 SC-003: `Registry::builtin()` resolves every shipped config key to exactly one family, and
//! knows no key this table does not list. The table is written out here, not derived from the entries,
//! so dropping a key from a family's entry, or shipping a key nobody listed, fails a row. An
//! integration test is its own crate: it sees only the public registry surface.
//!
//! `Family::ALL` drift (a variant missing from the enum's list) is the in-crate row
//! `registry::tests::family_all_lists_every_variant_once_and_builtin_registers_exactly_them`, which
//! needs the crate-private enum.

use std::collections::BTreeSet;

use poot_models::registry::{ConfigSource, RawConfig, Registry};

use ConfigSource::{GgufArchitecture as Gguf, HfModelType as Hf};

/// Every shipped key, with the family that owns it.
const SHIPPED: &[(ConfigSource, &str, &str)] = &[
    (Hf, "qwen2", "qwen2"),
    (Gguf, "qwen2", "qwen2"),
    (Hf, "llama", "llama"),
    (Hf, "mistral", "llama"),
    (Hf, "mixtral", "llama"),
    (Gguf, "llama", "llama"),
    (Hf, "qwen3", "qwen3"),
    (Gguf, "qwen3", "qwen3"),
    (Hf, "olmo2", "olmo2"),
    (Gguf, "olmo2", "olmo2"),
    (Hf, "phi3", "phi3"),
    (Gguf, "phi3", "phi3"),
    (Hf, "smollm3", "smollm3"),
    (Gguf, "smollm3", "smollm3"),
    (Hf, "granite", "granite"),
    (Hf, "granitemoe", "granite"),
    (Gguf, "granite", "granite"),
    (Gguf, "granitemoe", "granite"),
    (Hf, "mpt", "mpt"),
    (Gguf, "mpt", "mpt"),
    (Hf, "bloom", "bloom"),
    (Gguf, "bloom", "bloom"),
    (Hf, "gemma3_text", "gemma3"),
    (Gguf, "gemma3", "gemma3"),
    (Hf, "gemma2", "gemma2"),
    (Gguf, "gemma2", "gemma2"),
    (Hf, "qwen3_moe", "qwen3moe"),
    (Gguf, "qwen3moe", "qwen3moe"),
    (Hf, "gpt_oss", "gpt_oss"),
    (Gguf, "gpt-oss", "gpt_oss"),
    (Hf, "olmoe", "olmoe"),
    (Gguf, "olmoe", "olmoe"),
    (Hf, "gemma4", "gemma4"),
    (Gguf, "gemma4", "gemma4"),
    (Hf, "minimax_m2", "minimax_m2"),
    (Hf, "deepseek_v2", "deepseek2"),
    (Gguf, "deepseek2", "deepseek2"),
    (Hf, "deepseek_v3", "deepseek3"),
    (Hf, "deepseek_v32", "deepseek32"),
    (Hf, "nemotron_h", "nemotron_h"),
    (Hf, "deepseek_v4", "deepseek4"),
    (Hf, "bert", "bert"),
    (Hf, "idefics3_vision", "idefics3_vision"),
    (Hf, "qwen3_next", "qwen3next"),
    (Gguf, "qwen35moe", "qwen3next"),
    (Hf, "qwen3_5", "qwen35"),
    (Hf, "qwen3_5_text", "qwen35"),
    (Hf, "qwen4_exp", "qwen4exp"),
    // `glm5_next_text` is the nested `text_config` type of a `glm5_next` checkpoint, never a
    // top-level `model_type`, so it is not a key.
    (Hf, "glm5_next", "glm53_flash"),
];

#[test]
fn builtin_resolves_every_shipped_key_to_one_family_and_knows_no_other() {
    let registry = Registry::builtin().expect("the shipped families register disjoint keys");

    for &(source, key, family) in SHIPPED {
        let owners: Vec<&str> = registry
            .entries()
            .iter()
            .filter(|e| e.keys.contains(&(source, key)))
            .map(|e| e.family.as_str())
            .collect();
        assert_eq!(
            owners,
            [family],
            "{source} {key:?} must belong to exactly the family {family:?}"
        );
        if source == ConfigSource::HfModelType {
            let config = serde_json::json!({ "model_type": key });
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
            };
            let entry = registry
                .resolve(&raw)
                .unwrap_or_else(|e| panic!("{key:?} must resolve: {e}"));
            assert_eq!(entry.family.as_str(), family, "{key:?}");
        }
    }

    let listed: BTreeSet<_> = SHIPPED.iter().map(|&(s, k, _)| (s, k)).collect();
    assert_eq!(listed.len(), SHIPPED.len(), "SHIPPED lists a key twice");
    let shipped: BTreeSet<_> = registry
        .entries()
        .iter()
        .flat_map(|e| e.keys.iter().copied())
        .collect();
    assert_eq!(
        shipped, listed,
        "the registry ships a key this table does not list (left), or lacks one it lists (right)"
    );
}
