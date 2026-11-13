//! POOT-1571: a timed step's per-dispatch device times carry the plan's own dispatch labels when the step's
//! recording holds buffer copies between its dispatches.
//!
//! The engine names a device's per-dispatch times by their index among the entry's plan steps, so a device that
//! gives a copy (a state commit, an output snapshot) a time of its own, or an index, shifts every label after the
//! copy (POOT-1568: the ROCm device did, and a 'slow combine reduce' was another kernel's time). The fixture is cut
//! into two stages by a late input, so the first stage's state commit sits between the dispatches of the two stages.
//! The device is judged by what the engine prints: every dispatch row of the step names a plan step (none reads
//! `unknown`), and there is one row per plan dispatch.

use std::sync::Arc;

use poot_executor::{DeviceTime, ExecError, Executor, LateInputs, NoSync, StepInputs};
use poot_graph_ir::{BinOp, Builder, Outputs, Scalar, Slot, SlotKey, StateRole, TensorType, UnOp};
use poot_graph_plan::{
    CompileLimits, Target,
    multi_device::topology::DeviceId,
    staged::{DevicePlacement, ExpertPlacement, Partition, TargetSet, compile_staged},
};
use poot_quant::weights::WeightStore;
use poot_tensor::{DType, HostTensor};
use poot_weights::WeightMap;

/// The late input of the fixture: a constant, built once the first stage is on the device.
struct Late {
    keys: [SlotKey; 1],
}

impl LateInputs for Late {
    fn keys(&self) -> &[SlotKey] {
        &self.keys
    }

    fn build(&mut self) -> Result<Vec<HostTensor>, ExecError> {
        Ok(vec![HostTensor::f32(vec![2], vec![0.5, 0.5])])
    }
}

/// Step the fixture once on `exec`, which must be built with detailed device timing, and check its dispatch rows.
/// The fixture: stage one is `carried = state + x` (committed to the state) and `y = carried + 1` (exported), stage two is `-(y * 2 + late)`.
pub fn run_dispatch_labels_survive_the_stage_commit(
    exec: &mut dyn Executor,
    target: Target,
) -> Result<(), String> {
    let b = Builder::for_tests();
    let state = b
        .state_input("a", TensorType::f32(vec![2]), StateRole::Recurrent)
        .map_err(|e| e.to_string())?;
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![2]))
        .map_err(|e| e.to_string())?;
    let late = b
        .slot_named(Slot::Activation, "late", TensorType::f32(vec![2]))
        .map_err(|e| e.to_string())?;
    let carried = b.binary(BinOp::Add, state, x).map_err(|e| e.to_string())?;
    let y = b
        .binary_scalar(BinOp::Add, carried, Scalar::F32(1.0))
        .map_err(|e| e.to_string())?;
    let doubled = b
        .binary_scalar(BinOp::Mul, y, Scalar::F32(2.0))
        .map_err(|e| e.to_string())?;
    let shifted = b
        .binary(BinOp::Add, doubled, late)
        .map_err(|e| e.to_string())?;
    let out = b.unary(UnOp::Neg, shifted).map_err(|e| e.to_string())?;
    let boundary = poot_graph_ir::BoundaryId(y.id());
    let graph = b
        .finish(
            Outputs::new(out)
                .with_export(boundary, y)
                .with_state(&[(state, carried)]),
        )
        .map_err(|e| e.to_string())?;
    let program = compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &exec
            .compile_options(CompileLimits::STANDARD)
            .with_late(boundary),
    )
    .map_err(|e| format!("the fixture compiles: {e}"))?;
    if program.late_after().is_none() {
        return Err("the fixture is cut into two stages by its late input".into());
    }

    let store = WeightStore::builder().build();
    let map = WeightMap::builder(&store).build();
    let exe = exec
        .load_weights(Arc::new(store), Arc::new(map))
        .map_err(|e| e.to_string())?;
    let entry = exec.add_entry(exe, &program).map_err(|e| e.to_string())?;
    let x = [1.0f32, 2.0].map(f32::to_le_bytes).concat();
    let mut inputs = StepInputs::new();
    inputs.push(
        SlotKey::new(Slot::Activation, Some("x")),
        &[2],
        poot_tensor::HostView::new(DType::F32, 2, &x).map_err(|e| e.to_string())?,
    );
    let mut late = Late {
        keys: [SlotKey::new(Slot::Activation, Some("late"))],
    };
    let mark = exec.stats().timing.mark();
    let time = exec
        .step_late(exe, entry, &inputs, &mut late, &mut NoSync)
        .map_err(|e| e.to_string())?
        .device_time();
    let DeviceTime::Measured(measured) = time else {
        return Err("a timing-enabled device must report Measured".into());
    };
    let rows = measured.dispatches.len();
    let stats = exec.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry.raw()]);
    exec.unload(exe).map_err(|e| e.to_string())?;

    let named: u64 = report
        .buckets
        .iter()
        .filter(|(kind, _)| **kind != "unknown")
        .map(|(_, bucket)| bucket.dispatch_count)
        .sum();
    if report.buckets.contains_key("unknown") || named as usize != rows || rows == 0 {
        return Err(format!(
            "the step's {rows} dispatch times must each name a plan step, found {:?}",
            report.buckets
        ));
    }
    Ok(())
}
