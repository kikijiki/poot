//! High-level op decompositions: the readable surface a model is authored against. Each has one
//! definition shared by the tracer, the oracle and every lowering (the `jax.nn` -> `lax` analog).

use crate::builder::{Builder, BuilderAppendPlan, Traced};

use crate::error::{BuilderAppendError, BuilderCollection};

use crate::graph::{Operand, Storage};

use crate::op::{BinOp, RedOp, UnOp};

use crate::packed_source::packed_source_constants;

use crate::types::{DType, Scalar, TensorType};

mod activation;
mod attention;
mod counter;
mod linear;
mod moe;
mod norm;
mod prefill;
mod rope;
pub mod sampling;
mod ssm;
mod u64_hash;

pub use activation::*;
pub use attention::*;
pub use linear::*;
pub use moe::*;
pub use norm::*;
pub use prefill::*;
pub use rope::*;
pub use ssm::*;
pub use u64_hash::*;

#[cfg(test)]
mod tests;
