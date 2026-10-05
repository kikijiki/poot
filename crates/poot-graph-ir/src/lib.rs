//! poot-graph-ir: a backend-neutral, flat-SSA, primitive tensor-op graph IR plus a tracer.
//!
//! The graph is the program: a model forward is traced once, run with type-only [`Traced`] handles, into
//! a [`Graph`] of [`OpKind`] operations. Model blocks (attention, norms, RoPE, MoE routing) are built
//! from these primitives as compositions under [`ops`]. Every value carries a host-computed [`TensorType`]
//! ([`OpKind::infer`] is the single source of truth) and no data. Plain host Rust: no GPU, no LLVM. Design
//! lineage: jaxpr / DynamicJaxprTrace.

pub mod analysis;
pub mod builder;
/// Composite ops defined by their decomposition into primitives (Card 556, ADR-0101 tier 1).
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
pub mod rope_table;
#[cfg(test)]
mod test_support;
pub mod types;

pub use builder::{
    Builder, BuilderAppendAccounting, BuilderAppendPlan, LayerScope, PreparedBuilderAppend, Traced,
};
pub use error::{
    BuilderAppendError, BuilderCollection, BuilderValueNamespace, GraphValidationError, ShapeError,
};
pub use graph::{
    ComputedConst, Eqn, ExecutionValidationFailure, Graph, LayerIndex, MAX_VALIDATION_PACKET_BYTES,
    NoValidations, Operand, Slot, SlotKey, StatePair, StateRole, Storage, ValidationChannel,
    ValidationId, ValidationOutput, ValidationOutputs, ValidationPacketEntry,
    ValidationPacketError, ValidationPacketLayout, ValueId, ValueMeta,
};
pub use index_guard::{
    GuardedIndex, IndexBoundsGuard, IndexGuardError, IndexGuardRejection,
    recognize_index_bounds_guard,
};
pub use op::{
    BinOp, DENSE_CONTRACTION_WEIGHT_DTYPES, DENSE_ROW_GATHER_SOURCE_DTYPES, OpClass, OpKind, RedOp,
    UnOp,
};
pub use packed_source::{PackedSourceName, packed_source_constants, packed_source_type};
pub use types::{Scalar, TensorType};
