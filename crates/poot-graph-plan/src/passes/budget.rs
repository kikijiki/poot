//! The graph-size budget every pass that grows a graph appends through (card 666).
//!
//! A pass that adds equations or values reserves each one here before it is pushed, so the limit
//! binds at the append and the graph never holds one element past it. A pass that cannot count its own
//! appends (the dtype-preparation passes live outside this module) is sized once from its input by
//! [`GraphBudget::reserve`] before it runs. [`GraphBudget::admit`] is the check on a finished graph: the
//! traced input, and the output of every pass, so a pass that grows without reserving is still refused
//! at its boundary.

use std::num::{NonZeroU64, NonZeroUsize};

use poot_graph_ir::TraceLimits;
use poot_graph_ir::graph::{Eqn, Graph, ValueId, ValueMeta};

use crate::{CompileLimits, ExpansionError, GraphResource};
use poot_graph_ir::ValueTable;

/// The value and equation caps of one compile, as the passes spend them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphBudget {
    max_values: NonZeroUsize,
    max_eqns: NonZeroUsize,
    max_literal_bytes: NonZeroU64,
    max_owned_bytes: NonZeroU64,
}

impl GraphBudget {
    pub(crate) fn of(limits: &CompileLimits) -> Self {
        Self {
            max_values: limits.max_intermediate_values,
            max_eqns: limits.max_intermediate_eqns,
            max_literal_bytes: limits.max_graph_literal_bytes,
            max_owned_bytes: limits.max_graph_owned_bytes,
        }
    }

    /// The limits of a tracer builder over a graph this compile reopens. A pass has no step, so the
    /// token bound (a model trace's) is the smallest nonzero count and is never consulted.
    pub(crate) fn trace_limits(&self) -> TraceLimits {
        TraceLimits {
            max_tokens: NonZeroUsize::MIN,
            max_values: self.max_values,
            max_eqns: self.max_eqns,
            max_literal_bytes: self.max_literal_bytes,
            max_owned_bytes: self.max_owned_bytes,
        }
    }

    fn limit(&self, resource: GraphResource) -> usize {
        match resource {
            GraphResource::Values => self.max_values.get(),
            GraphResource::Eqns => self.max_eqns.get(),
        }
    }

    /// `Ok` when `held` plus `adding` stays within the cap; the sum saturates, which no cap admits.
    fn check(
        &self,
        stage: &'static str,
        resource: GraphResource,
        held: usize,
        adding: usize,
    ) -> Result<(), ExpansionError> {
        let attempted = held.saturating_add(adding);
        let limit = self.limit(resource);
        if attempted > limit {
            return Err(ExpansionError {
                stage,
                resource,
                held,
                attempted,
                limit,
            });
        }
        Ok(())
    }

    /// A finished graph is within both caps.
    pub(crate) fn admit(&self, stage: &'static str, g: &Graph) -> Result<(), ExpansionError> {
        self.check(stage, GraphResource::Values, g.values().len(), 0)?;
        self.check(stage, GraphResource::Eqns, g.eqns().len(), 0)
    }

    /// A pass that will add up to `values` values and `eqns` equations to `g` fits, from a conservative
    /// bound the caller derived from `g` alone. Refused before the pass runs.
    pub(crate) fn reserve(
        &self,
        stage: &'static str,
        g: &Graph,
        values: usize,
        eqns: usize,
    ) -> Result<(), ExpansionError> {
        self.check(stage, GraphResource::Values, g.values().len(), values)?;
        self.check(stage, GraphResource::Eqns, g.eqns().len(), eqns)
    }

    /// Append `eqn` to the equations a pass is building, or refuse without appending.
    pub(crate) fn push_eqn(
        &self,
        stage: &'static str,
        eqns: &mut Vec<Eqn>,
        eqn: Eqn,
    ) -> Result<(), ExpansionError> {
        self.check(stage, GraphResource::Eqns, eqns.len(), 1)?;
        eqns.push(eqn);
        Ok(())
    }

    /// Append `meta` to the values a pass is building and return its id, or refuse without appending.
    pub(crate) fn push_value(
        &self,
        stage: &'static str,
        values: &mut ValueTable<ValueMeta>,
        meta: ValueMeta,
    ) -> Result<ValueId, ExpansionError> {
        self.check(stage, GraphResource::Values, values.len(), 1)?;
        Ok(values.push(meta))
    }
}
