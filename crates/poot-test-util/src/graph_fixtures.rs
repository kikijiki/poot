//! `poot-graph-ir`/`poot-graph-plan` test fixtures with no production caller (card 671: dead under the
//! strict dead-pub rule: a test reference is never a use).
//!
//! [`finish_with_validations`], [`finish_with_state_and_validations`], [`i32_constant`] and
//! [`validation_packet_layout`] are complete reimplementations composed from `Builder`/`Graph`'s own already
//! -pub, still-production-used methods (`finish_with_state`, `Graph::with_validations`, `Graph::validate`,
//! `ValidationPacketLayout::for_graph`, `Builder::append_plan`/`preflight_append`/`commit_append`) - the
//! originals' only crate-private dependency (`exact_i32_input`) turned out to be a thin wrapper over those
//! same public primitives. [`plan_eqn`] is [`poot_graph_plan::plan_eqn`] under [`roomy_body_limits`].
//!
//! ADR-0113: `is_decode_gemv`/`is_tiled_gemm` had no production caller at all (production routes through
//! `poot-graph-plan`'s private `_in` twins) and were deleted; their cross-crate callers now assert on the
//! `Plan` [`poot_graph_plan::plan_eqn`] returns instead of calling a test-only predicate.
//! `reassociation_class` and `per_eqn_plans_outside_compile` read `poot-graph-plan`'s private state
//! (`recorded_numerics`, a crate-private thread-local counter) and stay there under
//! `#[cfg(any(test, feature = "test-support"))]` (ADR-0113 point 2); their consumers call them directly
//! through that feature, not through a forwarder here (a forwarder whose only purpose is to be a non-test
//! reference is laundering, ADR-0113's Context).

use poot_graph_ir::{
    Builder, BuilderAppendError, Eqn, Graph, GraphValidationError, Storage, TensorType, Traced,
    ValidationId,
};
use poot_graph_plan::{Plan, PlanError};
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
    plan.input_result(
        name.to_string(),
        TensorType::new(shape, DType::I32),
        Storage::Const,
    )?;
    let mut prepared = b.preflight_append(plan)?;
    b.commit_append(&mut prepared)
}

/// Finalize with one validation output, no carried state. Reimplements `poot-graph-ir`'s deleted
/// `Builder::finish_with_validations`.
pub fn finish_with_validations(
    b: Builder,
    out: Traced,
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph, GraphValidationError> {
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
) -> Result<Graph, GraphValidationError> {
    let outputs = validations.iter().fold(
        poot_graph_ir::Outputs::new(out).with_state(state),
        |outputs, &(id, name, value)| outputs.with_witness(id, name, value),
    );
    b.finish(outputs).map_err(|error| match error {
        poot_graph_ir::EditError::Invalid(error) => error,
        other => panic!("a fresh builder has no baseline to break: {other}"),
    })
}

/// A new program derived from `g`: its tables and outputs as a draft with no baseline, for `change` to
/// edit before the check. A rewrite that must keep what `g` computes goes through [`Graph::edit`]; this is
/// for a fixture that wants a different graph (a retyped result, a malformed-looking-but-valid variant).
pub fn derive(
    g: &Graph,
    change: impl FnOnce(&mut poot_graph_ir::GraphEdit),
) -> Result<Graph, poot_graph_ir::EditError> {
    let mut draft = poot_graph_ir::GraphEdit::new();
    *draft.values_mut() = g.values().clone();
    *draft.inputs_mut() = g.inputs().to_vec();
    *draft.consts_mut() = g.consts().to_vec();
    *draft.slots_mut() = g.slots().to_vec();
    *draft.eqns_mut() = g.eqns().to_vec();
    draft.set_result(g.result());
    for witness in g.witnesses() {
        draft.push_witness(witness.id, witness.name, witness.value);
    }
    for export in g.exports() {
        draft.push_export(export);
    }
    for (input, next) in g.state() {
        draft.push_state(input, next);
    }
    change(&mut draft);
    draft.commit()
}

/// Recompute every equation's declared output type from its operands' types, in order. A fixture that
/// changes an operation (a different axis, another permutation) calls this so the graph stays well
/// formed and only the shape consequences of the change remain. A `MatMul` keeps an f32-accumulate
/// widening it already declares.
pub fn reinfer(draft: &mut poot_graph_ir::GraphEdit) {
    for index in 0..draft.eqns().len() {
        let eqn = draft.eqns()[index].clone();
        let ins: Vec<TensorType> = eqn
            .inputs
            .iter()
            .map(|operand| match operand {
                poot_graph_ir::Operand::Value(value) => draft.values()[*value].aval.clone(),
                poot_graph_ir::Operand::Lit(scalar) => scalar.ty(),
            })
            .collect();
        if let Ok(inferred) = eqn.op.infer(&ins) {
            let kept_widening = matches!(eqn.op, poot_graph_ir::OpKind::MatMul)
                && draft.values()[eqn.out].aval.dtype == poot_tensor::DType::F32
                && inferred.shape == draft.values()[eqn.out].aval.shape;
            if !kept_widening {
                draft.values_mut()[eqn.out].aval = inferred;
            }
        }
    }
}

/// `g` with `value`'s declared type replaced by `aval`, everything else unchanged: a fixture for a graph
/// shape no tracer emits (a matmul that accumulates and stores f32 over bf16 operands).
pub fn with_declared_type(
    g: &Graph,
    value: poot_graph_ir::ValueId,
    aval: TensorType,
) -> Result<Graph, poot_graph_ir::EditError> {
    derive(g, |draft| draft.values_mut()[value].aval = aval)
}

/// Kernel-body limits no fixture comes near, for a test that plans a kernel and is not about the limit.
pub fn roomy_body_limits() -> poot_kernelgen::BodyLimits {
    poot_kernelgen::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    }
}

/// Choose how to execute an eqn at `world_size = 1`: `poot-graph-plan`'s `plan_eqn` under
/// [`roomy_body_limits`].
pub fn plan_eqn(
    g: &Graph,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Plan, PlanError> {
    poot_graph_plan::plan_eqn(g, eqn, backend, caps, &roomy_body_limits())
}

// --- GPU-target accessors for tests --------------------------------------------------------------------------
//
// A test that builds a `Target::Gpu` fixture reads its backend and capabilities directly: it knows which arm
// it constructed. Production consumers match the enum, and the planner refuses a GPU-only lowering on an AIE
// target with `Capability::GpuOnly`; a test that reads these off an AIE target has a bug, so these panic.

/// The GPU capabilities of the GPU target fixture `target`.
pub fn gpu_caps(target: &poot_graph_plan::Target) -> poot_target::DeviceCaps {
    match target {
        poot_graph_plan::Target::Gpu { caps, .. } => *caps,
        poot_graph_plan::Target::Aie { .. } => {
            panic!("{target:?} is an AIE target: it has AieCaps, not GPU DeviceCaps")
        }
    }
}

/// The GPU capabilities of the GPU target fixture `target`, for a test that shrinks one limit.
pub fn gpu_caps_mut(target: &mut poot_graph_plan::Target) -> &mut poot_target::DeviceCaps {
    match target {
        poot_graph_plan::Target::Gpu { caps, .. } => caps,
        poot_graph_plan::Target::Aie { .. } => {
            panic!("an AIE target has AieCaps, not GPU DeviceCaps")
        }
    }
}

/// The GPU backend of the GPU target fixture `target`.
pub fn gpu_backend(target: &poot_graph_plan::Target) -> poot_target::Backend {
    match target {
        poot_graph_plan::Target::Gpu { backend, .. } => *backend,
        poot_graph_plan::Target::Aie { .. } => {
            panic!("{target:?} is an AIE target: it has no GPU backend")
        }
    }
}
