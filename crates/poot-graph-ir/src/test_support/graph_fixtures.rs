//! Graph fixtures with no production caller (card 671: dead under the strict dead-pub rule: a test reference
//! is never a use).
//!
//! [`finish_with_validations`], [`finish_with_state_and_validations`] and [`i32_constant`] are complete
//! reimplementations composed from `Builder`/`Graph`'s own already-pub, still-production-used methods
//! (`finish_with_state`, `Graph::with_validations`, `Graph::validate`, `Builder::append_plan`/
//! `preflight_append`/`commit_append`) - the originals' only crate-private dependency (`exact_i32_input`)
//! turned out to be a thin wrapper over those same public primitives. The planner half of the old fixture
//! module (`plan_eqn` and the GPU-target accessors) lives in `poot-graph-plan`'s `test_support` module.

use crate::{
    Builder, BuilderError, Graph, GraphValidationError, Storage, TensorType, Traced, ValidationId,
};
use poot_tensor::DType;

/// Add one authoritative I32 constant through the transactional append boundary. Reimplements
/// `poot-graph-ir`'s deleted `Builder::i32_constant` from the same already-public append primitives its
/// private `exact_i32_input` helper used internally.
pub fn i32_constant(b: &Builder, name: &str, shape: Vec<usize>) -> Result<Traced, BuilderError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| BuilderError::ElementCountOverflow {
            name: name.to_string(),
            shape: shape.clone(),
        })?;
    let mut plan = b
        .append_plan(0)
        .expect("a fixture builder has trace budget left");
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
        crate::Outputs::new(out).with_state(state),
        |outputs, &(id, name, value)| outputs.with_witness(id, name, value),
    );
    b.finish(outputs).map_err(|error| match error {
        crate::BuilderError::Invalid(edit) => match *edit {
            crate::EditError::Invalid(error) => error,
            other => panic!("a fresh builder has no baseline to break: {other}"),
        },
        other => panic!("a fresh builder refuses only an invalid graph: {other}"),
    })
}

/// A new program derived from `g`: its tables and outputs as a draft with no baseline, for `change` to
/// edit before the check. A rewrite that must keep what `g` computes goes through [`Graph::edit`]; this is
/// for a fixture that wants a different graph (a retyped result, a malformed-looking-but-valid variant).
pub fn derive(
    g: &Graph,
    change: impl FnOnce(&mut crate::GraphEdit),
) -> Result<Graph, crate::EditError> {
    let mut draft = crate::GraphEdit::new();
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
/// formed and only the shape consequences of the change remain.
pub fn reinfer(draft: &mut crate::GraphEdit) {
    for index in 0..draft.eqns().len() {
        let eqn = draft.eqns()[index].clone();
        let ins: Vec<TensorType> = eqn
            .inputs
            .iter()
            .map(|operand| match operand {
                crate::Operand::Value(value) => draft.values()[*value].aval.clone(),
                crate::Operand::Lit(scalar) => scalar.ty(),
            })
            .collect();
        let is_literal =
            |index: usize| matches!(eqn.inputs.get(index), Some(crate::Operand::Lit(_)));
        if let Ok(inferred) = eqn.op.infer(crate::Operands::new(&ins, &is_literal)) {
            draft.values_mut()[eqn.out].aval = inferred;
        }
    }
}
