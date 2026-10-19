//! The recurrence scan composite on a device (POOT-634, POOT-1052): the fixtures every backend's
//! `scan_contract` test runs, with the oracle passed in by the caller (this crate never depends on
//! `poot-eval`).
//!
//! - [`check_steps`]: consecutive windows of a scan through one entry, each step's packed result against the
//!   oracle's evaluation of the same graph (its decomposition), the carried state each side's own.
//! - [`streamed_in_pieces`]: a long scan stepped in short pieces through the entry's committed state, assembled into
//!   the one packed result a single launch over the whole sequence must equal.
//! - [`measure`]: median step time of a scan, for timing receipts.
//! - [`device_time_of_step`], [`timed_step`]: the [`DeviceTime`] a device reports for one scan step
//!   (POOT-1072), and that step's packed result.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use poot_executor::{DeviceTime, Executor, NoSync, StateScope, StepInputs};
use poot_graph_ir::analysis::Tier2Class;
use poot_graph_ir::test_support::scan::{ScanFixture, ScanGraph};
use poot_graph_ir::{Graph, ValueId};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, multi_device::topology::DeviceId,
    production_plan::Submission, staged::DevicePlacement, staged::ExpertPlacement,
    staged::Partition, staged::TargetSet, staged::compile_staged,
};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_tensor::HostTensor;

use crate::assert_close;

/// The oracle's evaluation of a graph over bound host tensors: its one output.
pub type Oracle<'a> = dyn Fn(&Graph, &HashMap<ValueId, HostTensor>) -> HostTensor + 'a;

const OPTIONS: CompileOptions = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);

/// The scan `whole` stepped in pieces of `piece` positions through one entry of `target` and assembled into the one packed result a single step over the
/// whole sequence has: per head, the `y` rows of every piece then the last piece's state rows.  The pieces are not
/// compared with the oracle here: the caller compares the assembled result with the single launch's, which
/// [`check_steps`] has compared with the oracle.
pub fn streamed_in_pieces(
    exec: &mut dyn Executor,
    target: Target,
    whole: ScanFixture,
    piece: usize,
    label: &str,
) -> Vec<f32> {
    assert_eq!(whole.len % piece, 0, "{label}: whole pieces");
    let steps = whole.len / piece;
    let results = run_steps(exec, target, whole, piece, steps, label, None);
    let (width, rows) = (whole.width, whole.state_rows_of());
    let piece_stride = (piece + rows) * width;
    let mut packed = Vec::with_capacity(whole.heads * (whole.len + rows) * width);
    for head in 0..whole.heads {
        for result in &results {
            let at = head * piece_stride;
            packed.extend_from_slice(&result[at..at + piece * width]);
        }
        let last = results.last().expect("at least one piece");
        let at = head * piece_stride + piece * width;
        packed.extend_from_slice(&last[at..at + rows * width]);
    }
    packed
}

/// Step `steps` consecutive windows of `piece` positions of `whole` through one entry of `target`, comparing
/// each step's packed result with the oracle's, whose carried state is the oracle's own previous `s_out`. The
/// device's state starts at the contract's zero, so the oracle's does too. Returns the device's packed
/// results, one per step.
pub fn check_steps(
    exec: &mut dyn Executor,
    target: Target,
    whole: ScanFixture,
    piece: usize,
    steps: usize,
    label: &str,
    oracle_eval: &Oracle<'_>,
) -> Vec<Vec<f32>> {
    run_steps(exec, target, whole, piece, steps, label, Some(oracle_eval))
}

fn run_steps(
    exec: &mut dyn Executor,
    target: Target,
    whole: ScanFixture,
    piece: usize,
    steps: usize,
    label: &str,
    oracle_eval: Option<&Oracle<'_>>,
) -> Vec<Vec<f32>> {
    let whole = ScanFixture {
        len: piece * steps,
        ..whole
    };
    let step = ScanFixture {
        len: piece,
        ..whole
    };
    let mut inputs = whole.inputs(31);
    let state = inputs.operands.last_mut().unwrap();
    state.iter_mut().for_each(|x| *x = 0.0);

    let scan = step.packed_slot_graph();
    let oracle = step.packed_graph();
    let (exe, entry, class) = load_graph_entry(exec, target, &scan.graph, label);

    let layout_rows = step.len + step.state_rows_of();
    let mut oracle_state = inputs.operands.last().unwrap().clone();
    let mut results = Vec::new();
    for index in 0..steps {
        let mut window = whole.window(&inputs, index * piece, piece);
        *window.operands.last_mut().unwrap() = oracle_state.clone();
        let got = step_once(exec, exe, entry, &step, &scan, &window)
            .unwrap_or_else(|e| panic!("{label}: step {index}: {e}"));
        if let Some(oracle_eval) = oracle_eval {
            let want = oracle_eval(&oracle.graph, &step.bind(&oracle, &window));
            assert_eq!(got.shape(), want.shape(), "{label}: step {index} shape");
            assert_close(
                got.as_f32().unwrap(),
                want.as_f32().unwrap(),
                class.tolerance(),
                &format!("{label}: step {index} packed result (y rows then s_out rows)"),
            );
            // The oracle's next carried state is the state rows of its own packed result.
            let want = want.as_f32().unwrap();
            let width = step.width;
            oracle_state = (0..step.heads)
                .flat_map(|h| {
                    let base = (h * layout_rows + step.len) * width;
                    want[base..base + step.state_rows_of() * width].to_vec()
                })
                .collect();
        }
        results.push(got.as_f32().unwrap().to_vec());
    }
    exec.unload(exe).unwrap();
    results
}

/// Load an executable with no weights and one entry for `graph` compiled for `target`, and the numerics
/// class that entry's program publishes: the class a result of it is compared within. A fixture graph
/// built here binds no weights: its operands are activation slots and carried state.
pub(crate) fn load_graph_entry(
    exec: &mut dyn Executor,
    target: Target,
    graph: &Graph,
    label: &str,
) -> (
    poot_executor::ExecutableId,
    poot_executor::EntryId,
    Tier2Class,
) {
    let store = WeightStore::builder().build();
    let map = WeightMap::builder(&store).build();
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let staged = compile_staged(
        graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &OPTIONS,
    )
    .unwrap_or_else(|e| panic!("{label}: the graph compiles: {e}"));
    let entry = exec.add_entry(exe, &staged).unwrap();
    (exe, entry, staged.numerics_class())
}

/// The numerics class the program of the scan `fixture` publishes on `target`: the class two device results of
/// that scan (a single launch and its stepped pieces) are compared within. Pieces and the whole differ in length
/// only, which the class does not read, so any length of one family and shape gives the one answer.
pub fn numerics_class(target: Target, fixture: &ScanFixture) -> Tier2Class {
    let scan = fixture.packed_slot_graph();
    compile_staged(
        &scan.graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &OPTIONS,
    )
    .unwrap_or_else(|e| panic!("the scan graph compiles: {e}"))
    .numerics_class()
}

/// One step of `entry` over `window`'s sequence operands; the packed result.
fn step_once(
    exec: &mut dyn Executor,
    exe: poot_executor::ExecutableId,
    entry: poot_executor::EntryId,
    step: &ScanFixture,
    scan: &ScanGraph,
    window: &poot_graph_ir::test_support::scan::ScanInputs,
) -> Result<HostTensor, poot_executor::ExecError> {
    let bound = step.bind(scan, window);
    let mut step_inputs = StepInputs::new();
    let sequence = &scan.operands[..scan.operands.len() - 1];
    for &id in sequence {
        let key = scan.graph.meta(id).slot_key().unwrap().clone();
        step_inputs.push(key, bound[&id].shape(), bound[&id].view());
    }
    exec.step(exe, entry, &step_inputs, &mut NoSync)?.to_host()
}

/// The [`DeviceTime`] `exec` reports for one step of the scan `fixture` on `target`: the entry is
/// loaded, stepped once, and unloaded. The device's own answer, so a backend built without device
/// timing returns `Unknown` and one built with it returns `Measured`.
pub fn device_time_of_step(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &ScanFixture,
) -> DeviceTime {
    timed_step(exec, target, fixture).0
}

/// One scan step's [`DeviceTime`] and its packed result, so a caller can compare a timed device's
/// result with an untimed one's (timing must not change results).
pub fn timed_step(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &ScanFixture,
) -> (DeviceTime, Vec<f32>) {
    let scan = fixture.packed_slot_graph();
    let (exe, entry, _) = load_graph_entry(exec, target, &scan.graph, "device time");
    let bound = fixture.bind(&scan, &fixture.inputs(7));
    let mut step_inputs = StepInputs::new();
    for &id in &scan.operands[..scan.operands.len() - 1] {
        let key = scan.graph.meta(id).slot_key().unwrap().clone();
        step_inputs.push(key, bound[&id].shape(), bound[&id].view());
    }
    let mut out = exec
        .step(exe, entry, &step_inputs, &mut NoSync)
        .expect("the scan steps");
    let time = out.device_time();
    let packed = out.to_host().expect("the packed result reads back");
    drop(out);
    exec.unload(exe).unwrap();
    (time, packed.as_f32().unwrap().to_vec())
}

/// One scan's step time: the host wall time of a step (inputs staged, output read back) and, where the
/// backend reports it, the device time of its dispatches.
#[derive(Clone, Copy, Debug)]
pub struct StepTime {
    /// Host wall milliseconds of the step, inputs staged and output read back.
    pub wall_ms: f64,
    /// Device milliseconds of the step's dispatches, or `None` when the backend reports no device time.
    pub device_ms: Option<f64>,
}

/// The median over `runs` steps of the scan `fixture` on `target`, after one warm-up step.
pub fn measure(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &ScanFixture,
    runs: usize,
) -> StepTime {
    let scan = fixture.packed_slot_graph();
    let (exe, entry, _) = load_graph_entry(exec, target, &scan.graph, "measure");
    let inputs = fixture.inputs(7);
    let bound = fixture.bind(&scan, &inputs);
    let mut samples = Vec::new();
    for run in 0..=runs {
        let mut step_inputs = StepInputs::new();
        for &id in &scan.operands[..scan.operands.len() - 1] {
            let key = scan.graph.meta(id).slot_key().unwrap().clone();
            step_inputs.push(key, bound[&id].shape(), bound[&id].view());
        }
        let start = Instant::now();
        let mut out = exec
            .step(exe, entry, &step_inputs, &mut NoSync)
            .expect("the scan steps");
        let device_ms = match out.device_time() {
            DeviceTime::Measured(m) => m.sum_of_dispatch_durations.map(|d| d.as_secs_f64() * 1e3),
            DeviceTime::Unknown => None,
        };
        out.to_host().expect("the packed result reads back");
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        exec.reset_state(exe, StateScope::All).unwrap();
        if run > 0 {
            samples.push(StepTime { wall_ms, device_ms });
        }
    }
    exec.unload(exe).unwrap();
    samples.sort_by(|a, b| a.wall_ms.total_cmp(&b.wall_ms));
    samples[samples.len() / 2]
}
