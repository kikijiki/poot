//! Graph-to-graph transforms: CSE, DCE, automatic fusion and dtype lowering (graph-architecture.md).
//! Kernel choice is not a transform: the planner records it (Card 557). [`crate::compile`] is their only caller (card 626, SC-001): this module is
//! private, so nothing outside this crate can name a pass or sequence its own pipeline, and
//! `poot_graph_ir::transform` (the pre-card-626 home) no longer exists at all. The read-only analyses
//! that used to live alongside these passes - [`poot_graph_ir::analysis`]'s `NumericsProperty`,
//! `dispatch_count`, `peak_transient_bytes` - stay public in
//! `poot-graph-ir`, since they rewrite nothing and a caller may legitimately want to measure a graph
//! without reaching a pass.
//!
//! [`packed_bind`]'s weight-carrier types (`bind_packed_weights`, `PackedConst`, `PackedLayout`,
//! `WeightFormats`) are not a `compile` pass - they bind a traced graph's packed-weight constants once,
//! at checkpoint load, before `compile` ever runs - so they are re-exported `pub` from the crate root
//! (`lib.rs`) rather than sealed with the rest of this module.

use std::collections::{HashMap, HashSet};

use poot_graph_ir::analysis::{NumericalRewriteDecline, NumericalRewriteReason};
use poot_graph_ir::graph::{EditError, Eqn, Graph, Operand, Storage, ValueId, ValueMeta};
use poot_graph_ir::op::{
    BinOp, ContractionHead, FusedOp, FusedOperand, FusedRegion, FusedStep, OpKind, RedOp,
    RowRegion, RowStep, UnOp,
};
use poot_graph_ir::types::{Scalar, TensorType};
use poot_tensor::DType;

mod attention_match;
mod budget;
mod cse_dce;
mod dtype;
pub(crate) mod fuse;
mod guard_witness;
mod iota;
mod legalize;
mod lora;
mod packed;
mod packed_bind;
mod pool;
mod routed_experts;
mod selection;
mod sharding;
pub(crate) mod stage_split;
mod widen;

/// `g` with its equations replaced by `eqns`, checked through the edit API: the one way a pass that
/// only rewrites equations produces its graph.
pub(crate) fn with_eqns(g: &Graph, eqns: Vec<Eqn>) -> Result<Graph, EditError> {
    let mut edit = g.edit();
    *edit.eqns_mut() = eqns;
    edit.commit()
}

pub(crate) use budget::GraphBudget;
pub(crate) use legalize::refuse_oversized_constants;
pub(crate) use legalize::{KvExtent, kv_extents};
pub use legalize::{LegalizeError, TOKEN_EMBED, legalize};
pub(crate) use lora::apply_lora;
pub use lora::{LORA_IDX, LoraError, LoraPlan};

#[cfg(any(test, feature = "test-support"))]
pub use attention_match::canonicalize;
pub(crate) use attention_match::canonicalize_with_declines;
pub use attention_match::{AttentionDecline, FusionDecline, FusionDeclineReason, RopeDecline};
#[cfg(any(test, feature = "test-support"))]
pub use attention_match::{flash_attention_capped, rope_fusion};
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use attention_match::{flash_attention_capped, rope_fusion};
#[cfg(any(test, feature = "test-support"))]
pub use cse_dce::cse;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use cse_dce::cse;
#[cfg(any(test, feature = "test-support"))]
pub use cse_dce::dce;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use cse_dce::dce;
#[cfg(any(test, feature = "test-support"))]
pub use dtype::collapse_reshape_chains;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use dtype::collapse_reshape_chains;
#[cfg(any(test, feature = "test-support"))]
pub use dtype::elide_noop_transposes;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use dtype::elide_noop_transposes;
#[cfg(any(test, feature = "test-support"))]
pub use fuse::fuse;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use fuse::fuse;
pub(crate) use fuse::fuse_contraction_epilogues;
pub(crate) use guard_witness::first_unobserved_clamp;
#[cfg(any(test, feature = "test-support"))]
pub use iota::fold_iota;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use iota::fold_iota;
pub use packed::PackedDequantErrorContext;
pub use packed::PackedDequantProductionError;
pub use packed::recognize_packed_contractions;
pub use packed::recognize_packed_row_gathers;
pub(crate) use packed::reject_packed_dequant_escapes;
pub use packed_bind::{
    ExpertStackRefusal, PackedBindError, PackedConst, PackedLayout, WeightFormats,
    WeightFormatsError, bind_packed_weights,
};
pub(crate) use pool::pool_experts;
pub use pool::{
    EXPERT_POOL, EXPERT_POOL_MAP, PoolError, PoolSite, PoolSpec, PooledWeight, routed_step_demand,
};
#[cfg(any(test, feature = "test-support"))]
pub use routed_experts::{RoutedExpertSite, routed_expert_sites};
pub(crate) use selection::form_selection;
pub(crate) use sharding::expand_ranks;
pub use sharding::{ShardError, ShardRule, ShardedGraph, shard};
pub use stage_split::BoundaryDescriptor;
pub use stage_split::StageSplitError;
pub(crate) use stage_split::{CutSite, StageAssignment, split_stages};
pub(crate) use widen::{fuse_index_casts, fuse_widening_casts};

/// The graph-to-graph passes alone, in the order [`crate::compile`]'s pipeline runs them, without
/// `compile`'s legalize/dtype-preparation/planning stages (which need a `Target`). The one shared
/// entry for pass-soundness and plan-shape tests that want the fused graph but no program; never a
/// production path (`compile` is the pipeline, Card 626).
#[cfg(any(test, feature = "test-support"))]
pub fn passes_without_target(g: &Graph) -> Result<Graph, EditError> {
    let g = cse(&canonicalize(g)?)?;
    let g = flash_attention_capped(&rope_fusion(&fold_iota(&g)?)?.0, Some(65_535))?.0;
    let g = form_selection(&g)?;
    let g = fuse_contraction_epilogues(&dce(&g)?, |_, _| true)?;
    fuse_index_casts(&fuse_widening_casts(&fuse(&g)?)?)
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
