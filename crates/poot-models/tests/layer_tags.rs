//! POOT-578 SC-001: every registered family's fixture graph carries layer tags that
//! [`poot_graph_ir::Graph::layer_runs`] accepts, with one run per configured layer. A family whose
//! tracer forgets `layer_scope` (or opens it late, or drops a layer) cannot be layer-cut by the stage
//! split; this row finds it without a per-family test.
//!
//! The corpus is `Registry::builtin().entries()` crossed with each entry's fixtures (the same walk the
//! plan-summary gate uses), so a family registered with a fixture joins this row with no edit here.
//! `Family::ALL` itself is crate-private; its drift row is
//! `registry::tests::family_all_lists_every_variant_once_and_builtin_registers_exactly_them`.
//!
//! Shapes are the corpus's own (decode, prefill at the 4-token prompt, prefill into a larger cache, or an
//! encoder's forward): 4 tokens, not 3, because compressed-attention families trace only multiples of
//! their prefill granule.
//!
//! The expected layer count is read from the fixture's own config JSON (its layer-count key), never
//! from the tracer under test.

#[path = "support/corpus.rs"]
mod corpus;

use corpus::{Fixtured, fixtured};
use poot_graph_ir::LayerTagError;
use poot_models::registry::Registry;
use serde_json::Value;

/// The config keys that carry a family's decoder (or encoder) layer count.
const LAYER_COUNT_KEYS: &[&str] = &["num_hidden_layers", "n_layer", "n_layers", "depth"];

/// The layer count a fixture's config declares: its layer-count key at the top level, else inside the
/// nested text or vision config.
fn configured_layers(config: &Value) -> usize {
    let find = |object: &Value| {
        LAYER_COUNT_KEYS
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_u64))
    };
    find(config)
        .or_else(|| {
            ["text_config", "vision_config"]
                .iter()
                .find_map(|nested| config.get(*nested).and_then(find))
        })
        .unwrap_or_else(|| panic!("fixture config names no layer count: {config}")) as usize
}

/// Every fixture with its configured layer count, in the corpus order.
fn fixtures_with_layers(registry: &Registry) -> Vec<(Fixtured, usize)> {
    let mut configs = Vec::new();
    for entry in registry.entries() {
        configs.push((entry.family.as_str().to_string(), (entry.fixture)().config));
        for (variant, make) in entry.variants {
            configs.push((
                format!("{}/{variant}", entry.family.as_str()),
                make().config,
            ));
        }
    }
    let fixtured = fixtured(registry);
    assert_eq!(fixtured.len(), configs.len());
    fixtured
        .into_iter()
        .zip(configs)
        .map(|(f, (label, config))| {
            assert_eq!(f.label, label, "corpus order drifted from the entry walk");
            (f, configured_layers(&config))
        })
        .collect()
}

#[test]
fn every_registered_family_tags_one_run_per_configured_layer() {
    let registry = Registry::builtin().expect("the shipped families register");
    let mut failures = Vec::new();
    for (fixture, expected) in fixtures_with_layers(&registry) {
        for &shape in fixture.shapes() {
            let graph = fixture.trace(shape);
            match graph.layer_runs() {
                Ok(runs) if runs.layers == expected => {}
                Ok(runs) => failures.push(format!(
                    "{} {}: layer_runs found {} layer(s), the config declares {expected}",
                    fixture.label, shape.name, runs.layers
                )),
                Err(
                    error @ (LayerTagError::Untagged { .. } | LayerTagError::NonContiguous { .. }),
                ) => failures.push(format!("{} {}: {error:?}", fixture.label, shape.name)),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "layer tags are wrong:\n{}",
        failures.join("\n")
    );
}
