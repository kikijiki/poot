//! POOT-1040 on PTX: a guarded live-attention graph the matcher declines (no declared length witness, so the
//! length is not bounded) stays generic primitives, key ends in exact I32 included, and a live PTX device runs it.

use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, eval};
use poot_executor::{
    Device, Engine, ExecError, Executor, HostView, NoSync, StepInputs, WeightSource,
};
use poot_graph_ir::{Slot, Storage, ValidationOutputs};
use poot_graph_plan::{
    CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, KernelChoice,
    Partition, Submission, Target, TargetSet, compile_staged,
};
use poot_kernelgen::KernelRequest;
use poot_ptx_gpu::PtxDevice;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_tensor::DType;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::live_attention::{Inputs, LiveAttentionFixture, LiveAttentionGraph};

/// The graph compiled for the executor contract, then one step of it through `exec`.
fn live_step(
    exec: &mut dyn Executor,
    target: Target,
    built: &LiveAttentionGraph,
    inputs: &Inputs,
) -> Result<(Vec<f32>, bool), ExecError> {
    let graph: &poot_graph_ir::Graph<ValidationOutputs> = &built.graph;
    let program = compile_staged(
        graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            limits: CompileLimits::STANDARD,
        },
    )
    .expect("the declined live attention compiles for PTX");
    let planned_attention = program.stages().any(|(_, _, stage)| {
        stage.planned().any(|(eqn, _)| {
            matches!(
                stage.kernel_choice(eqn),
                KernelChoice::Generated(KernelRequest::Attention(_))
            )
        })
    });
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
    let exe = exec
        .load_weights(Arc::new(store.build()), WeightSource::ConstNames)
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
        })
        .map(|out| (out, planned_attention));
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    result
}

/// With NaN in every key and value past the length, the output is finite and within the attention tier of the
/// first-5-key oracle, and the plan has no attention kernel. Mutation: build `LiveKeys::new`'s key ends from an
/// F32 iota plus a `Cast` to I32 (`ops::attention_over_keys`); the plan refuses the F32 -> I32 cast.
#[test]
fn a_declined_live_attention_graph_runs_as_primitives_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 9,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: None,
    };
    let built = fixture.build(false);
    let inputs = fixture.inputs(&built, &[5], true);
    let (got, planned_attention) = live_step(&mut *exec, target, &built, &inputs)
        .expect("a declined graph runs on the device");
    assert!(!planned_attention, "the unproved graph stays generic");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = eval(
        &built.graph,
        &inputs.values(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let want = fixture.oracle(&inputs, &built, &[5]);
    for (label, reference) in [
        ("guarded graph", cpu.as_f32().unwrap()),
        ("first 5 keys", &want[..]),
    ] {
        for (i, (g, w)) in got.iter().zip(reference).enumerate() {
            assert!(
                (g - w).abs() <= 1e-3,
                "vs the {label}: element {i}: device {g} vs oracle {w}"
            );
        }
    }
}
