//! The live-capacity attention decode, stepped once through the executor contract: the one helper every
//! backend's live-attention rows call (wgpu, ROCm, PTX and the two-byte rows), so a fault in it turns all
//! of them red.

use std::sync::Arc;

use poot_executor::{ExecError, Executor, HostView, NoSync, StepInputs};
use poot_graph_ir::{Graph, Slot, Storage, ValidationOutputs};
use poot_graph_plan::{
    CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_test_util::const_weights::ConstWeights;
use poot_test_util::live_attention::{Inputs, LiveAttentionGraph};

/// `graph` compiled for `target` on the executor contract's one shape: single stage, single device,
/// `Submission::Replay`. Its own validations stay intact, so a validation under test is not stripped.
pub fn compile_live(
    graph: &Graph<ValidationOutputs>,
    target: Target,
) -> StagedProgram<ValidationOutputs> {
    compile_staged(
        graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
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
    let graph = &built.graph;
    let mut store = WeightStore::builder();
    let mut pos_bytes = Vec::new();
    let mut pos_key = None;
    for &id in &graph.inputs {
        let meta = graph.meta(id);
        match meta.storage {
            Storage::Const => {
                let tensor = &inputs.0[&id];
                let dense = DenseWeight::try_new(
                    DType::F32,
                    tensor.shape().to_vec(),
                    Arc::from(tensor.view().bytes()),
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
                pos_bytes = inputs.0[&id]
                    .as_i32()
                    .unwrap()
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
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
    let mut step_in = StepInputs::new();
    step_in.push(
        key,
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
                .collect()
        });
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    result
}
