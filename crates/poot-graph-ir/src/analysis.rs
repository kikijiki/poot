//! Read-only graph analyses: numerics classification (ADR-0101 decision 2) and the launch-cost
//! estimates (dispatch count, peak transient bytes). These rewrite nothing - a graph transform is a
//! pass, and every pass now lives in `poot-graph-plan` (card 626: `compile` is the one pipeline, so
//! this crate publishes no way to rewrite a graph outside it). What stays here is the pure functions
//! that read a graph's ops to classify them, independent of which crate runs the pass that produced
//! them, and the class-to-tolerance table; each pass declares its own numerics in `poot-graph-plan`.

use crate::graph::{Graph, ValueId};
#[cfg(any(test, feature = "test-support"))]
use crate::graph::{Operand, ValueTable};
use crate::op::OpKind;

mod numerics;
#[cfg(any(test, feature = "test-support"))]
mod stats;

pub use numerics::{
    NumericalRewriteDecline, NumericalRewriteReason, NumericsError, NumericsProperty, Tier2Class,
    implied_numerics, implied_numerics_of,
};
#[cfg(any(test, feature = "test-support"))]
pub use stats::{dispatch_count, peak_transient_bytes};
