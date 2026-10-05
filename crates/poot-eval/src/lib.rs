//! poot-eval: a CPU eager reference executor for the graph IR.
//!
//! Evaluates a traced [`Graph`] on the host at f32: the reference of the oracle stack
//! (graph-architecture.md section 5, stage 1: decomposition correctness). It is not a fast path. It
//! lets primitive decompositions be checked against independent direct computations and GPU executors
//! be checked for bit-identity against it.
//!
//! One evaluator walk ([`eval`]): bind, preflight every equation's admission, evaluate, gate, publish.
//! Ordinary arithmetic storage is f32. Integer indices (token, pos) are exact in f32 (< 2^24), so a
//! scalar index is read back via `round`. [`Value`] is the typed storage-aware entry point, which also
//! carries the E4M3FN and packed-quant executor slices.

pub mod cast_authority;
pub mod exact_bf16;
pub mod exact_dense;
pub mod exact_value;
pub mod fp8;
pub mod observer;
mod weight_store;

pub use weight_store::{materialize_dense, take_dense};

pub use exact_bf16::ExactBf16Error;

use std::borrow::Cow;

use poot_graph_ir::PackedSourceName;
use poot_quant::PackedComponentRef;

pub(crate) use geometry::{broadcast_flat, gather_source_flat, strides, unravel};

mod error;
mod geometry;
mod operand;
mod ops;
mod resolve;
mod routing;
mod value;
mod walk;

pub use error::{CastFault, CastOperand, EvalError};
pub use exact_value::{ExactI32TensorView, ExactValue};
pub use ops::index_rule::{IndexFault, IndexFaultKind, IndexValue};
#[cfg(any(test, feature = "test-support"))]
pub use ops::movement::apply_movement;
pub use routing::{top_k_gate, top_k_ids, top_k_mask};
pub use value::{PackedEvalError, Value};
pub use walk::{EvalBudget, EvalOptions, Evaluation, eval};

#[cfg(test)]
mod tests;
