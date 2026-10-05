//! Read-only graph analyses: numerics classification (ADR-0101 decision 2) and the launch-cost
//! estimates (dispatch count, peak transient bytes). These rewrite nothing - a graph transform is a
//! pass, and every pass now lives in `poot-graph-plan` (card 626: `compile` is the one pipeline, so
//! this crate publishes no way to rewrite a graph outside it). What stays here is the declarative
//! table of passes `compile` runs (keyed by name, not by calling into the pass modules - a
//! cross-crate reference the other way would be a dependency cycle) and the pure functions that read
//! a graph's ops to classify them, independent of which crate runs the pass that produced them.

use crate::graph::{Graph, Operand, ValidationChannel, ValueId};
use crate::op::OpKind;

mod numerics;
mod stats;

pub use numerics::{
    NumericalRewriteDecline, NumericalRewriteReason, NumericsError, NumericsProperty,
    PASS_DECLARATIONS, PassDeclaration, Tier2Class, implied_numerics, verify_pass_numerics,
};
pub use stats::{dispatch_count, peak_transient_bytes};
