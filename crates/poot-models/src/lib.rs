//! Model forward decompositions: the tracers that author a model's forward against poot-graph-ir's
//! high-level op surface ([`poot_graph_ir::ops`]) and produce a flat primitive [`poot_graph_ir::Graph`].
//!
//! These are consumers of the IR, not part of it. They live above `poot-graph-ir` (which stays
//! model-agnostic: IR, builder, transforms, op compositions) and below `poot-llm` (the runner), so the
//! lower executor crates can exercise them in tests without depending on the runner.
// The registry drift test counts `Family`'s variants with the compiler (`std::mem::variant_count`).
#![cfg_attr(test, feature(variant_count))]

mod architectures;
pub mod chat;
pub mod components;
pub mod model;
mod multimodal;
pub mod names;
pub mod registry;

pub use architectures::*;
pub use components::{moe_decode, moe_prefill};
pub use multimodal::*;

#[cfg(test)]
mod reference_ops;
#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::model_fixture_data;
