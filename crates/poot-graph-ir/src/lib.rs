//! poot-graph-ir: a backend-neutral, flat-SSA, primitive tensor-op graph IR plus a tracer.
//!
//! The graph is the program: a model forward is traced once, run with type-only [`Traced`] handles, into
//! a [`Graph`] of [`OpKind`] operations. Model blocks (attention, norms, RoPE, MoE routing) are built
//! from these primitives as compositions under [`ops`]. Every value carries a host-computed [`TensorType`]
//! ([`OpKind::infer`] is the single source of truth) and no data. Plain host Rust: no GPU, no LLVM. Design
//! lineage: jaxpr / DynamicJaxprTrace.

pub mod analysis;
pub mod budget;
pub mod builder;
#[cfg(test)]
mod builder_budget_tests;
/// Composite ops defined by their decomposition into primitives (Card 556, ADR-0101 tier 1).
#[cfg(any(test, feature = "test-support"))]
pub mod decompose;
pub mod error;
pub mod graph;
/// The canonical in-graph index-bounds guard and its validation witness (card 372c).
pub mod index_guard;
pub mod op;
pub mod ops;
/// Graph constant names and source types of a packed linear: the `{linear_id}.packed_weight_source` /
/// `{linear_id}.packed_scale_source` spelling and the scale byte width each carrier is shaped by.
pub mod packed_source;
/// The interval analysis of I32 values and the declared slot ranges it starts from (POOT-726).
pub mod range;
pub mod rope_table;
/// The checked forward write map of an in-place row write.
mod slot_map;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod types;
pub mod weight_ref;

pub use budget::{AllocationSite, Attempted, BudgetResource, TraceBudgetExceeded, TraceLimits};
#[cfg(any(test, feature = "test-support"))]
pub use budget::{Charged, Ledger, LedgerEvent, Reservation};
pub use builder::{
    BlockTopKResult, Builder, LayerScope, Outputs, ScanResult, Traced, WitnessDeclaration,
};
#[cfg(any(test, feature = "test-support"))]
pub use builder::{BuilderAppendAccounting, BuilderAppendPlan, PreparedBuilderAppend};
pub use error::{
    BuilderCollection, BuilderError, BuilderValueNamespace, FusedOperandReason,
    GraphValidationError, Requirement, ScanReason, ShapeError,
};
pub use graph::{
    BoundaryId, ComputedConst, EditError, Eqn, ExecutionValidationFailure, Export, Graph,
    GraphEdit, GraphOutput, LayerIndex, LayerRuns, LayerTagError, MAX_VALIDATION_PACKET_BYTES,
    Operand, OutputKind, Slot, SlotKey, StatePair, StatePolicy, StateRole, StateRows, Storage,
    ValidationId, ValidationPacketEntry, ValidationPacketError, ValidationPacketLayout, ValueId,
    ValueMeta, ValueTable, Witness, WitnessLedger,
};
pub use index_guard::{
    GuardValues, GuardedIndex, IndexGuardEmitError, IndexGuardError, emit_index_guard,
    is_guard_witness,
};
pub use op::{
    AttentionAttrs, BinOp, BlockTopKAttrs, BlockTopKLayout, ContractionRows,
    DENSE_CONTRACTION_WEIGHT_DTYPES, DENSE_ROW_GATHER_SOURCE_DTYPES, KeyLayout, KvBound,
    MaskOperand, OpClass, OpKind, RedOp, ScanAttrs, ScanFamily, ScanLayout, SelectedAttentionAttrs,
    SelectedAttentionLayout, UnOp,
};
#[cfg(any(test, feature = "test-support"))]
pub use packed_source::packed_source_constants;
pub use packed_source::{PackedSourceName, packed_source_type, packed_stack_source_type};
pub use range::{IndexRange, RangeFacts, value_ranges};
pub use slot_map::{ForwardSlotMap, SlotMapError};
pub use types::{Scalar, TensorType};
pub use weight_ref::WeightRef;
