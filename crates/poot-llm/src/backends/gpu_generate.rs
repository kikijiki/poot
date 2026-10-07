//! Executor-contract staging helpers: the target an executor drives, a compiled staged program and
//! the slot inputs of a bound graph.

use crate::error::{OptionExt, Result};

/// The one [`poot_graph_plan::Target`] `exec` drives (Card 546a's M3 contract: `Engine<D>` drives
/// exactly one device, so `target_set()` always names exactly one). `pub(crate)`: every backend's
/// consumer needs this same generic read, not just wgpu's (Card 549).
pub(crate) fn executor_target(exec: &dyn poot_executor::Executor) -> poot_graph_plan::Target {
    exec.target_set()
        .devices()
        .first()
        .expect("Card 546a's M3 Engine<D> always names exactly one device")
        .1
}

/// `g`, staged for the executor contract (always `Submission::Replay`: the contract admits no other
/// mode). Ordinary models load a zero-lane validation packet; no
/// production caller wires real decode-graph witnesses yet. Shared by every production entry program
/// (prefill and decode alike): the contract's `add_entry` shares carried state across entries by
/// (name, aval, storage), so a prefill entry's written K/V is the decode entry's input.
pub(crate) fn staged_program(
    g: &poot_graph_ir::Graph,
    target: poot_graph_plan::Target,
) -> std::result::Result<
    poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs>,
    poot_graph_plan::StagedCompileError,
> {
    let g = g.clone().with_validations(Vec::new());
    poot_graph_plan::compile_staged(
        &g,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &poot_graph_plan::CompileOptions {
            execution: poot_graph_plan::Submission::Replay,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
}

/// `bound`'s `Storage::Slot` entries only (Card 546a's contract binds every `Storage::Const`/
/// `Storage::State` value from the executable's `WeightStore`/carried state instead, through
/// `Runner::load_on` and the engine's own state buffers - never a per-step value), as the
/// `SlotKey`-keyed `StepInputs` the contract's `step` takes. Borrows `bound`'s tensors,
/// so the caller's `bind_decode` result must outlive the returned `StepInputs`. `pub(crate)`: every
/// backend's consumer needs this same generic bind, not just wgpu's (Card 549).
pub(crate) fn slot_step_inputs<'a>(
    g: &poot_graph_ir::Graph,
    bound: &'a std::collections::HashMap<poot_graph_ir::ValueId, poot_eval::Value>,
) -> Result<poot_executor::StepInputs<'a>> {
    let mut inputs = poot_executor::StepInputs::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if !matches!(meta.storage, poot_graph_ir::Storage::Slot(_)) {
            continue;
        }
        let key = meta
            .slot_key()
            .context("decode slot without a structured SlotKey")?
            .clone();
        let value = bound
            .get(&id)
            .context("bind_decode did not bind a declared slot")?;
        let tensor = value
            .as_host()
            .context("a slot value must be a dense tensor")?;
        inputs.push(key, tensor.shape(), tensor.view());
    }
    Ok(inputs)
}
