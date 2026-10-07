use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use poot_eval::Value;
use poot_graph_ir::PackedSourceName;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};

use crate::error::{OptionExt, Result, ResultExt};

/// The baked [`WeightStore`] `Runner::load_on` (Card 546a's one Runner consumer) gives
/// `poot_executor::Executor::load_weights`: every `weights` value re-expressed as a
/// [`WeightEntry`], keyed exactly as the decode/prefill graphs' `Storage::Const` names already are
/// (`Value::Host` by its own name; a `Value::Packed` by its [`PackedSourceName`]'s
/// `linear_id`, Card 642's const naming -), so the engine's generic (store key, planned
/// storage) residency binds every const with no Runner-specific knowledge. Every role-component of
/// one packed linear shares its owner's identical `Arc<PackedPayload>` (the loader clones it
/// per role), so the first role seen for a `linear_id` inserts it and later roles are a no-op, not a
/// duplicate-key error.
pub(crate) fn weight_store(weights: &HashMap<String, Value>) -> Result<WeightStore> {
    let mut builder = WeightStore::builder();
    let mut packed_linear_ids: HashSet<String> = HashSet::new();
    for (name, value) in weights {
        match value {
            Value::Host(t) => {
                let dense = DenseWeight::try_new(
                    t.dtype(),
                    t.shape().to_vec(),
                    Arc::from(t.view().bytes()),
                )
                .with_context(|| format!("{name}: baking the executor's weight store"))?;
                builder
                    .insert(name.clone(), WeightEntry::Dense(dense))
                    .with_context(|| {
                        format!("{name}: duplicate dense weight in the baked store")
                    })?;
            }
            Value::Packed(component) => {
                let parsed = PackedSourceName::parse(name).with_context(|| {
                    format!("{name}: packed weight value whose name is not a packed source name")
                })?;
                if packed_linear_ids.insert(parsed.linear_id().to_string()) {
                    builder
                        .insert(
                            parsed.linear_id().to_string(),
                            WeightEntry::Packed(Arc::clone(component.owner())),
                        )
                        .with_context(|| {
                            format!(
                                "{}: duplicate packed weight in the baked store",
                                parsed.linear_id()
                            )
                        })?;
                }
            }
            Value::Owner(_) => {
                bail!(
                    "{name}: this weight value kind has no poot-executor WeightStore encoding yet"
                )
            }
        }
    }
    Ok(builder.build())
}
