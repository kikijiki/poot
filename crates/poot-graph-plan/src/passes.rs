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

use poot_graph_ir::analysis::{
    NumericalRewriteDecline, NumericalRewriteReason, NumericsError, NumericsProperty,
    implied_numerics_of,
};
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

/// What one pass of [`crate::compile`]'s pipeline declares about the device arithmetic of the ops it
/// produces (ADR-0101 decision 2). [`crate::compile`] runs a pass through its declaration, so a pass
/// cannot run undeclared, and checks the declaration against the pass's output with
/// [`Self::verify`]. The program publishes the strongest declaration among the passes that changed
/// its graph.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PassDeclaration {
    /// The pass's name as `compile` reports it.
    pub(crate) pass: &'static str,
    /// The numerics property the pass declares for the ops it produces.
    pub(crate) property: NumericsProperty,
    /// Whether an op is one the pass produces: [`Self::verify`] reads these ops in the output graph.
    produces: fn(&OpKind) -> bool,
}

impl PassDeclaration {
    const fn new(
        pass: &'static str,
        property: NumericsProperty,
        produces: fn(&OpKind) -> bool,
    ) -> Self {
        Self {
            pass,
            property,
            produces,
        }
    }

    /// Check the declaration against the strongest property among the ops the pass produced in `out`.
    /// Fails closed when the declaration is weaker than what the pass produced (a pass that
    /// reassociates while claiming `Exact`).
    pub(crate) fn verify(&self, out: &Graph) -> Result<(), NumericsError> {
        let implied = implied_numerics_of(out, self.produces);
        if self.property >= implied {
            return Ok(());
        }
        Err(NumericsError::UnderDeclared {
            pass: self.pass,
            declared: self.property,
            implied,
        })
    }
}

/// The declaration of every pass [`crate::compile`] runs, one constant per pass, named for it. A pass
/// added to `compile` takes its declaration as its first argument, so one without a row does not build.
pub(crate) mod declared {
    use super::{NumericsProperty, OpKind, PassDeclaration};

    fn nothing(_: &OpKind) -> bool {
        false
    }

    pub(crate) const CANONICALIZE: PassDeclaration =
        PassDeclaration::new("canonicalize", NumericsProperty::Exact, nothing);
    pub(crate) const CSE: PassDeclaration =
        PassDeclaration::new("cse", NumericsProperty::Exact, nothing);
    pub(crate) const FOLD_IOTA: PassDeclaration =
        PassDeclaration::new("fold_iota", NumericsProperty::Exact, nothing);
    pub(crate) const ROPE_FUSION: PassDeclaration =
        PassDeclaration::new("rope_fusion", NumericsProperty::Reassociating, |op| {
            matches!(op, OpKind::Rope { .. })
        });
    pub(crate) const FLASH_ATTENTION_CAPPED: PassDeclaration = PassDeclaration::new(
        "flash_attention_capped",
        NumericsProperty::Reassociating,
        |op| matches!(op, OpKind::Attention(_)),
    );
    pub(crate) const DCE: PassDeclaration =
        PassDeclaration::new("dce", NumericsProperty::Exact, nothing);
    // `compile` claims `ops::packed_linear`'s chains as `PackedContraction` right after `dce`, then
    // refuses every `PackedDequant` left unclaimed (the escape gate), so no later pass sees a packed
    // decode outside a contraction. The claim replaces a `PackedDequant` (already `Narrowing`) and its
    // `MatMul` with one `PackedContraction` (`Narrowing`).
    pub(crate) const RECOGNIZE_PACKED_CONTRACTIONS: PassDeclaration = PassDeclaration::new(
        "recognize_packed_contractions",
        NumericsProperty::Narrowing,
        |op| matches!(op, OpKind::PackedContraction { .. }),
    );
    // The row-gather claim (a quantized embedding lookup) after the contraction claim, before the gate.
    pub(crate) const RECOGNIZE_PACKED_ROW_GATHERS: PassDeclaration = PassDeclaration::new(
        "recognize_packed_row_gathers",
        NumericsProperty::Narrowing,
        |op| matches!(op, OpKind::PackedRowGather { .. }),
    );
    pub(crate) const REJECT_PACKED_DEQUANT_ESCAPES: PassDeclaration = PassDeclaration::new(
        "reject_packed_dequant_escapes",
        NumericsProperty::Exact,
        nothing,
    );
    // `compile` runs these four between `dce` and `fuse`, so the folds and the reduce legalization see
    // the traced `MatMul`/`Transpose`, `Cast`/`Gather` and non-last-axis `Reduce` chains before `fuse`
    // could absorb them into a region no target plans.
    pub(crate) const FOLD_DENSE_CONTRACTIONS: PassDeclaration = PassDeclaration::new(
        "fold_dense_contractions",
        NumericsProperty::Narrowing,
        |op| {
            matches!(
                op,
                OpKind::DenseContraction { .. } | OpKind::GroupedDenseContraction { .. }
            )
        },
    );
    pub(crate) const FOLD_DENSE_BF16_ROW_GATHERS: PassDeclaration = PassDeclaration::new(
        "fold_dense_bf16_row_gathers",
        NumericsProperty::Narrowing,
        |op| matches!(op, OpKind::DenseRowGather { .. }),
    );
    pub(crate) const LOWER_NONLAST_REDUCES: PassDeclaration =
        PassDeclaration::new("lower_nonlast_reduces", NumericsProperty::Exact, nothing);
    pub(crate) const WIDEN_MISMATCHED_MATMUL_DTYPES: PassDeclaration = PassDeclaration::new(
        "widen_mismatched_matmul_dtypes",
        NumericsProperty::Narrowing,
        nothing,
    );
    // The contraction epilogue rule folds a contraction and the pointwise chain that consumes it into one
    // equation. The lowering rounds the head through its declared dtype and every step through its own,
    // so the fused result is the unfused program's, bit for bit, on the same schedule: an unfused
    // contraction.
    pub(crate) const FUSE_CONTRACTION_EPILOGUES: PassDeclaration = PassDeclaration::new(
        "fuse_contraction_epilogues",
        NumericsProperty::Unfused,
        |op| matches!(op, OpKind::ContractionEpilogue { .. }),
    );
    pub(crate) const FUSE: PassDeclaration =
        PassDeclaration::new("fuse", NumericsProperty::Reassociating, |op| {
            matches!(op, OpKind::Fused(_) | OpKind::FusedRow(_))
        });
    // `fuse_widening_casts` rewrites a region's own steps: the `Cast` of a narrow weight becomes the
    // region's first step. Widening a narrow float is exact and no op kind is produced that was not
    // already counted, so the class of the result is read off its ops, as for every pass.
    pub(crate) const FUSE_WIDENING_CASTS: PassDeclaration =
        PassDeclaration::new("fuse_widening_casts", NumericsProperty::Exact, nothing);
    // `fuse_index_casts` makes the `Cast` of an I32 word a step of the pointwise region that reads or
    // writes it, wrapping a lone pointwise equation in a one-step region where it has none. The
    // conversion is the cast kernel's own, so the class is the fused regions it produces.
    pub(crate) const FUSE_INDEX_CASTS: PassDeclaration =
        PassDeclaration::new("fuse_index_casts", NumericsProperty::Reassociating, |op| {
            matches!(op, OpKind::Fused(_) | OpKind::FusedRow(_))
        });
    // `legalize` needs `Target::caps`; `compile` runs it as its own pass, between `dce` and `fuse`. Its
    // rewrite shapes: a hosted-embed `Gather` replaced by a device `MatMul`-free lookup, or an oversized
    // matmul weight split and rejoined by `Concat`.
    pub(crate) const LEGALIZE: PassDeclaration =
        PassDeclaration::new("legalize", NumericsProperty::Unfused, |op| {
            matches!(op, OpKind::MatMul | OpKind::Concat { .. })
        });
}

pub(crate) use budget::GraphBudget;
pub(crate) use legalize::refuse_oversized_constants;
pub use legalize::{LegalizeError, legalize};
pub(crate) use lora::apply_lora;
pub use lora::{LoraError, LoraPlan};

#[cfg(any(test, feature = "test-support"))]
pub use attention_match::canonicalize;
pub(crate) use attention_match::canonicalize_with_declines;
#[cfg(any(test, feature = "test-support"))]
pub use attention_match::flash_attention_capped;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use attention_match::flash_attention_capped;
#[cfg(any(test, feature = "test-support"))]
pub use attention_match::rope_fusion;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use attention_match::rope_fusion;
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
pub(crate) use guard_witness::{first_unobserved_clamp, producer_map};
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
pub use pool::{PoolError, PoolSite, PoolSpec, PooledWeight, routed_step_demand};
#[cfg(any(test, feature = "test-support"))]
pub use routed_experts::{RoutedExpertSite, routed_expert_sites};
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
    let g = flash_attention_capped(&rope_fusion(&fold_iota(&g)?)?, Some(65_535))?;
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
