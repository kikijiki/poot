//! The live-capacity attention decode, stepped once through the executor contract: the one helper every
//! backend's live-attention rows call (wgpu, ROCm, PTX and the two-byte rows), so a fault in it turns all
//! of them red.

use std::sync::Arc;

use poot_eval::test_support::live_attention::{Inputs, LiveAttentionGraph};
use poot_executor::{ExecError, Executor, NoSync, StepInputs};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{Graph, Slot, Storage};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, multi_device::topology::DeviceId,
    staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::StagedProgram,
    staged::TargetSet, staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostView};

/// `graph` compiled for `target` on the executor contract's one shape: single stage, single device. Its own validations stay intact, so a validation under test is not stripped.
pub fn compile_live(graph: &Graph, target: Target) -> StagedProgram {
    compile_staged(
        graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .expect("the live-attention graph compiles")
}

/// One step of `built.graph`, compiled for `target`, through `exec`, validation channel included: the
/// output when the packet passes, the contract's error when it does not. The consts load as named dense F32
/// weights, `Slot::Pos` binds as the step's I32 input, and the entry and executable are torn down again.
pub fn live_step(
    exec: &mut dyn Executor,
    target: Target,
    built: &LiveAttentionGraph,
    inputs: &Inputs,
) -> Result<Vec<f32>, ExecError> {
    let pos = inputs.0[&built.pos]
        .as_i32()
        .expect("Slot::Pos binds I32 words")
        .to_vec();
    let mut outputs = live_steps(exec, target, built, inputs, &[pos])?;
    Ok(outputs.remove(0))
}

/// Several steps of `built.graph` through one entry of one executable: `positions[i]` is step `i`'s
/// `Slot::Pos` words (one per batch row). The first step records and replays in one call, every later one
/// is a pure replay, and the device's arena persists between them, so a step reads whatever the previous
/// step left in every slot it reuses. The consts load as named dense F32 weights from `inputs`; the entry
/// and executable are torn down again, also on an error.
pub fn live_steps(
    exec: &mut dyn Executor,
    target: Target,
    built: &LiveAttentionGraph,
    inputs: &Inputs,
    positions: &[Vec<i32>],
) -> Result<Vec<Vec<f32>>, ExecError> {
    let graph = &built.graph;
    let mut store = WeightStore::builder();
    let mut pos_key = None;
    for &id in graph.inputs() {
        let meta = graph.meta(id);
        match meta.storage {
            Storage::Const => {
                let tensor = &inputs.0[&id];
                let dense = DenseWeight::try_new(
                    DType::F32,
                    tensor.shape().to_vec(),
                    StoredBytes::from(tensor.view().bytes().to_vec()),
                )
                .unwrap();
                store
                    .insert(
                        meta.name.clone().expect("a named const"),
                        WeightEntry::Dense(dense),
                    )
                    .unwrap();
            }
            Storage::Slot(Slot::Pos) => {
                pos_key = Some((meta.slot_key().unwrap().clone(), meta.aval.shape.clone()));
            }
            other => panic!("unexpected live-attention input storage {other:?}"),
        }
    }
    let store = Arc::new(store.build());
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(graph);
    let program = compile_live(&named, target);
    let exe = exec
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let (key, shape) = pos_key.expect("the decode reads Slot::Pos");
    let mut results = Ok(Vec::new());
    for words in positions {
        let pos_bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut step_in = StepInputs::new();
        step_in.push(
            key.clone(),
            &shape,
            HostView::new(DType::I32, shape.iter().product(), &pos_bytes).unwrap(),
        );
        let result = exec
            .step(exe, entry, &step_in, &mut NoSync)
            .map(|mut output| {
                output
                    .read()
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect::<Vec<f32>>()
            });
        match (result, &mut results) {
            (Ok(output), Ok(outputs)) => outputs.push(output),
            (Err(error), _) => {
                results = Err(error);
                break;
            }
            (Ok(_), Err(_)) => break,
        }
    }
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    results
}
