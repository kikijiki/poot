//! Graph-to-graph transforms: CSE, DCE, automatic fusion and dtype lowering (graph-architecture.md).
//! Kernel choice is not a transform: the planner records it (Card 557). [`crate::compile`] is their only caller (card 626, SC-001): this module is
//! private, so nothing outside this crate can name a pass or sequence its own pipeline, and
//! `poot_graph_ir::transform` (the pre-card-626 home) no longer exists at all. The read-only analyses
//! that used to live alongside these passes - [`poot_graph_ir::analysis`]'s `NumericsProperty`,
//! `PassDeclaration`/`PASS_DECLARATIONS`, `dispatch_count`, `peak_transient_bytes` - stay public in
//! `poot-graph-ir`, since they rewrite nothing and a caller may legitimately want to measure a graph
//! without reaching a pass.
//!
//! [`packed_bind`]'s weight-carrier types (`bind_packed_weights`, `PackedConst`, `PackedLayout`,
//! `WeightFormats`) are not a `compile` pass - they bind a traced graph's packed-weight constants once,
//! at checkpoint load, before `compile` ever runs - so they are re-exported `pub` from the crate root
//! (`lib.rs`) rather than sealed with the rest of this module.

use std::collections::{HashMap, HashSet};

use poot_graph_ir::analysis::{NumericalRewriteDecline, NumericalRewriteReason};
use poot_graph_ir::error::GraphValidationError;
#[cfg(test)]
use poot_graph_ir::graph::StateRole;
use poot_graph_ir::graph::{Eqn, Graph, Operand, Storage, ValidationChannel, ValueId, ValueMeta};
use poot_graph_ir::op::{
    BinOp, FusedOp, FusedOperand, FusedRegion, FusedStep, OpKind, RedOp, RowRegion, RowStep, UnOp,
};
use poot_graph_ir::types::Scalar;
use poot_tensor::DType;

mod attention_match;
mod budget;
mod cse_dce;
mod dtype;
mod fuse;
mod iota;
mod legalize;
mod packed;
mod packed_bind;
mod sample_token_large_vocab;
mod stage_split;

pub(crate) use budget::GraphBudget;
pub use legalize::{LegalizeError, legalize};
pub use sample_token_large_vocab::decompose_large_vocab_greedy;

#[cfg(any(test, feature = "test-support"))]
pub use attention_match::canonicalize;
pub use attention_match::canonicalize_with_declines;
pub use attention_match::flash_attention_capped;
pub use attention_match::rope_fusion;
pub use cse_dce::cse;
pub use cse_dce::dce;
#[cfg(any(test, feature = "test-support"))]
pub use cse_dce::dce_with_roots;
pub use dtype::collapse_reshape_chains;
pub use dtype::elide_noop_transposes;
#[cfg(any(test, feature = "test-support"))]
pub use fuse::FUSABLE_FLOAT_UNARY_OPS;
pub use fuse::{fuse, fuse_bias_epilogues};
pub use iota::fold_iota;
#[cfg(any(test, feature = "test-support"))]
pub use packed::PackedDequantErrorContext;
pub use packed::PackedDequantProductionError;
pub use packed::recognize_packed_contractions;
pub use packed::recognize_packed_row_gathers;
pub use packed::reject_packed_dequant_escapes;
#[cfg(test)]
pub(crate) use packed::reject_preexisting_packed_contractions;
pub use packed_bind::{
    PackedBindError, PackedConst, PackedLayout, WeightFormats, WeightFormatsError,
    bind_packed_weights,
};
pub use stage_split::BoundaryDescriptor;
#[cfg(test)]
pub(crate) use stage_split::{
    StageAssignment, StageSplitError, split_stages, split_stages_after_layers,
};

#[cfg(test)]
pub(crate) use dtype::transpose_is_noop;

/// The graph-to-graph passes alone, in the order [`crate::compile`]'s pipeline runs them, without
/// `compile`'s legalize/dtype-preparation/planning stages (which need a `Target`). The one shared
/// entry for pass-soundness and plan-shape tests that want the fused graph but no program; never a
/// production path (`compile` is the pipeline, Card 626).
#[cfg(any(test, feature = "test-support"))]
pub fn passes_without_target<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    fuse(&fuse_bias_epilogues(&dce(&flash_attention_capped(
        &rope_fusion(&fold_iota(&cse(&canonicalize(g)))),
        Some(65_535),
    ))))
}

#[cfg(test)]
mod packed_dequant_eval_soundness;
#[cfg(test)]
mod stage_split_eval_soundness;
#[cfg(test)]
mod tests;

// The real-model integration of these transforms (cse/dce/fuse on the qwen2 decode graph) lives in
// `poot-models` (it needs a model tracer, which depends on `poot-graph-ir` - a dependency this crate
// must not add).
