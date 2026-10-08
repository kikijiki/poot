//! `poot-graph-ir`/`poot-graph-plan` test fixtures with no production caller (card 671: dead under the
//! strict dead-pub rule: a test reference is never a use).
//!
//! [`finish_with_validations`], [`finish_with_state_and_validations`], [`i32_constant`] and
//! [`validation_packet_layout`] are complete reimplementations composed from `Builder`/`Graph`'s own already
//! -pub, still-production-used methods (`finish_with_state`, `Graph::with_validations`, `Graph::validate`,
//! `ValidationPacketLayout::for_graph`, `Builder::append_plan`/`preflight_append`/`commit_append`) - the
//! originals' only crate-private dependency (`exact_i32_input`) turned out to be a thin wrapper over those
//! same public primitives. [`plan_eqn`] is likewise reimplemented by composing the superseding, still-pub
//! [`poot_graph_plan::plan_eqn_analyzed`] with a caller-owned [`poot_graph_plan::ExactI32StorageAnalysis`] -
//! byte-identical to what the deleted wrapper did internally. `plan_eqn`'s own siblings `plan_eqn_ws`/
//! `plan_eqn_views` (also deleted from poot-graph-plan) have no real caller left anywhere in the workspace -
//! every one of their test callers went straight to `plan_eqn_analyzed`/`plan_eqn_views_analyzed` instead, so
//! they are not reconstructed here.
//!
//! ADR-0113: `is_decode_gemv`/`is_tiled_gemm` had no production caller at all (production routes through
//! `poot-graph-plan`'s private `_in` twins) and were deleted; their cross-crate callers now assert on the
//! `Plan` [`poot_graph_plan::plan_eqn_analyzed`] returns instead of calling a test-only predicate.
//! `reassociation_class` and `per_eqn_plans_outside_compile` read `poot-graph-plan`'s private state
//! (`recorded_numerics`, a crate-private thread-local counter) and stay there under
//! `#[cfg(any(test, feature = "test-support"))]` (ADR-0113 point 2); their consumers call them directly
//! through that feature, not through a forwarder here (a forwarder whose only purpose is to be a non-test
//! reference is laundering, ADR-0113's Context).

use poot_graph_ir::{
    Builder, BuilderAppendError, Eqn, Graph, GraphValidationError, Storage, TensorType, Traced,
    ValidationId, ValidationOutput, ValidationPacketLayout,
};
use poot_graph_plan::{ExactI32StorageAnalysis, Plan, PlanError};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::DType;

/// Add one authoritative I32 constant through the transactional append boundary. Reimplements
/// `poot-graph-ir`'s deleted `Builder::i32_constant` from the same already-public append primitives its
/// private `exact_i32_input` helper used internally.
pub fn i32_constant(
    b: &Builder,
    name: &str,
    shape: Vec<usize>,
) -> Result<Traced, BuilderAppendError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| BuilderAppendError::ElementCountOverflow {
            name: name.to_string(),
            shape: shape.clone(),
        })?;
    let mut plan = b.append_plan(0);
    let value = plan.input_result(
        name.to_string(),
        TensorType::new(shape, DType::I32),
        Storage::Const,
    )?;
    let mut prepared = b.preflight_append(plan)?;
    let id = b.commit_append(&mut prepared)?;
    debug_assert_eq!(id, value.id);
    Ok(value)
}

/// Finalize with one validation output, no carried state. Reimplements `poot-graph-ir`'s deleted
/// `Builder::finish_with_validations`.
pub fn finish_with_validations(
    b: Builder,
    out: Traced,
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph<poot_graph_ir::ValidationOutputs>, GraphValidationError> {
    finish_with_state_and_validations(b, out, &[], validations)
}

/// Finalize with carried state and validation outputs. Reimplements `poot-graph-ir`'s deleted
/// `Builder::finish_with_state_and_validations` from `finish_with_state` + `Graph::with_validations` +
/// `Graph::validate`.
pub fn finish_with_state_and_validations(
    b: Builder,
    out: Traced,
    state: &[(Traced, Traced)],
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph<poot_graph_ir::ValidationOutputs>, GraphValidationError> {
    let g = b.finish_with_state(out, state).with_validations(
        validations
            .iter()
            .map(|&(id, name, value)| ValidationOutput {
                id,
                name: name.to_owned(),
                value: value.id,
            })
            .collect(),
    );
    g.validate()?;
    Ok(g)
}

/// A validated graph's packet layout. Reimplements `poot-graph-ir`'s deleted
/// `Graph::validation_packet_layout` via `ValidationPacketLayout::for_graph`.
pub fn validation_packet_layout(
    g: &Graph<poot_graph_ir::ValidationOutputs>,
) -> Result<ValidationPacketLayout, GraphValidationError> {
    ValidationPacketLayout::for_graph(g)
}

/// Kernel-body limits no fixture comes near, for a test that plans a kernel and is not about the limit.
pub fn roomy_body_limits() -> poot_kernelgen::BodyLimits {
    poot_kernelgen::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    }
}

/// Choose how to execute an eqn at `world_size = 1`, with no caller-owned graph-wide analysis. Reimplements
/// `poot-graph-plan`'s deleted `plan_eqn` by building the one-shot analysis the deleted wrapper built
/// internally, then calling the still-pub, still-production-used `plan_eqn_analyzed`.
pub fn plan_eqn(
    g: &Graph,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Plan, PlanError> {
    let analysis = ExactI32StorageAnalysis::new(g);
    poot_graph_plan::plan_eqn_analyzed(&analysis, g, eqn, backend, caps, &roomy_body_limits())
}
