//! Device receipts for validation-bearing graphs on the executor contract.
//!
//! Each test compares the device outcome with the CPU oracle. Run serially on a real device with
//! `POOT_REQUIRE_WGPU=1`, which turns a missing device into a failure instead of a skip.

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalError, EvalOptions, Value};
use poot_executor::{Device, ExecError, Executor, HostView, NoSync, StepInputs};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::{
    BinOp, Builder, ExecutionValidationFailure, Graph, Scalar, Slot, SlotKey, StateRole,
    TensorType, Traced, UnOp, ValidationId, ValidationOutputs, ValidationPacketError, ValueId,
};
use poot_graph_plan::{
    CompileError, CompileOptions, DeviceId, DevicePlacement, DeviceWitnessRejection,
    ExpertPlacement, Partition, PlanError, StagedCompileError, StagedProgram, Submission, Target,
    TargetSet, compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_tensor::HostTensor;
use poot_test_util::const_weights::ConstWeights;

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Compile `g` (already `Graph<ValidationOutputs>`-typed, its own validations intact - unlike this
/// file's other migrated helpers, this one must NOT route through a `with_validations(Vec::new())`
/// type-coercion step, which would silently strip the very validations under test) for `target` with
/// `Submission::Replay` (Card 546b: the contract admits no other submission).
fn staged(g: &Graph<ValidationOutputs>, target: Target) -> StagedProgram<ValidationOutputs> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile the validation graph")
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// One slot's bytes, ready for [`StepInputs::push`].
struct SlotBind {
    key: SlotKey,
    shape: Vec<usize>,
    dtype: DType,
    bytes: Vec<u8>,
}

fn step_inputs(binds: &[SlotBind]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for b in binds {
        let elems = b.shape.iter().product();
        inputs.push(
            b.key.clone(),
            &b.shape,
            HostView::new(b.dtype, elems, &b.bytes).unwrap(),
        );
    }
    inputs
}

/// Split a `HashMap<ValueId, Value>` bind into the contract's two bind mechanisms, as in the other
/// migrated test files (`Storage::Const` -> a named [`WeightStore`] entry, `Storage::Slot` -> a
/// [`StepInputs`] row); `Storage::State` is left for the contract to zero-seed on first use.
fn split_consts_and_slots(
    g: &Graph<ValidationOutputs>,
    inputs: &HashMap<ValueId, Value>,
) -> (WeightStore, Vec<SlotBind>) {
    let mut builder = WeightStore::builder();
    let mut slots = Vec::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            poot_graph_ir::Storage::Const => {
                let name = m.name.clone().expect("named const");
                let t = inputs
                    .get(&id)
                    .and_then(Value::as_host)
                    .unwrap_or_else(|| panic!("missing const input {id:?}"));
                let bytes: Vec<u8> = t
                    .as_f32()
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let dense =
                    DenseWeight::try_new(DType::F32, t.shape().to_vec(), StoredBytes::from(bytes))
                        .unwrap();
                builder.insert(name, WeightEntry::Dense(dense)).unwrap();
            }
            poot_graph_ir::Storage::Slot(_) => {
                let t = inputs
                    .get(&id)
                    .and_then(Value::as_host)
                    .unwrap_or_else(|| panic!("missing slot input {id:?}"));
                let dtype = m.aval.dtype;
                let bytes: Vec<u8> = if dtype == DType::I32 {
                    t.as_i32()
                        .expect("I32-declared slot needs an I32 HostTensor")
                        .iter()
                        .flat_map(|&v| v.to_le_bytes())
                        .collect()
                } else {
                    t.as_f32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect()
                };
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: t.shape().to_vec(),
                    dtype,
                    bytes,
                });
            }
            poot_graph_ir::Storage::State | poot_graph_ir::Storage::Computed(_) => {}
            poot_graph_ir::Storage::Device => unreachable!(),
        }
    }
    (builder.build(), slots)
}

/// The contract's analogue of this file's old `gpu_failure`: unwraps an `ExecError::Validation`
/// packet failure into the same [`ExecutionValidationFailure`] the CPU oracle raises -
/// `poot_graph_ir::ValidationPacketError::Failure` wraps the identical struct
/// (`poot_executor`'s contract reuses `poot-graph-ir`'s validation types unchanged), so this stays a
/// precise lane-for-lane comparison, not a coarse "an error happened" check.
fn contract_failure<T>(result: Result<T, ExecError>) -> ExecutionValidationFailure {
    match result {
        Err(ExecError::Validation(packet)) => match *packet {
            ValidationPacketError::Failure(failure) => failure,
            other => panic!("expected a packet Failure, got {other:?}"),
        },
        Err(error) => panic!("expected ExecError::Validation, got {error:?}"),
        Ok(_) => panic!("expected a validation failure, but step published a result"),
    }
}

/// `1 - Ge(v, -MAX) * Ge(-v, -MAX)` per lane: 1 for NaN and both infinities, else 0.
fn nonfinite_flags(b: &Builder, v: Traced) -> Traced {
    let not_below = b.binary_scalar(BinOp::Ge, v, Scalar::F32(-f32::MAX));
    let negated = b.unary(UnOp::Neg, v);
    let not_above = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(-f32::MAX));
    let finite = b.binary(BinOp::Mul, not_below, not_above);
    let flipped = b.binary_scalar(BinOp::Mul, finite, Scalar::F32(-1.0));
    b.binary_scalar(BinOp::Add, flipped, Scalar::F32(1.0))
}

fn finite_input() -> Vec<f32> {
    vec![0.5, -1.25, 2.0, 3.0, -0.0, 1.0e-3, 7.0, -8.5]
}

fn cpu_failure(
    graph: &Graph<ValidationOutputs>,
    inputs: &HashMap<ValueId, Value>,
) -> ExecutionValidationFailure {
    match poot_eval::eval(graph, inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|e| e.output)
    {
        Err(EvalError::Validation(failure)) => failure,
        other => panic!("the CPU oracle must reject this input, got {other:?}"),
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn wgpu_validation_ge_ieee_edges_match_cpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let edges = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        -0.0,
        f32::from_bits(1),
        1.0,
    ];
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![8]));
    let flags = nonfinite_flags(&b, x);
    let ordered = b.binary(BinOp::Ge, x, x);
    let unordered = b.binary_scalar(BinOp::Mul, ordered, Scalar::F32(-1.0));
    let unordered = b.binary_scalar(BinOp::Add, unordered, Scalar::F32(1.0));
    let both = b.concat(0, &[flags, unordered]);
    // The first witness always passes and exposes nothing; the primary output carries every flag lane so
    // the device comparison results are checked bit for bit against the CPU oracle.
    let silent = b.binary_scalar(BinOp::Mul, both, Scalar::F32(0.0));
    let passing = poot_test_util::graph_fixtures::finish_with_validations(
        b,
        both,
        &[(ValidationId(23), "silent", silent)],
    )
    .unwrap();
    let inputs: HashMap<ValueId, Value> =
        HashMap::from([(x.id(), HostTensor::f32(vec![8], edges.clone()).into())]);
    let expected = poot_eval::eval(&passing, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let (store, slot_binds) = split_consts_and_slots(&passing, &inputs);
    let (bound, map) = ConstWeights::bind(&passing, &store);
    let program = staged(&bound, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let observed = f32_bytes(
        &exec
            .step(exe, entry, &step_in, &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    assert_eq!(bits(&observed), bits(expected.as_f32().unwrap()));
    assert_eq!(
        bits(expected.as_f32().unwrap()),
        bits(&[
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]),
        "oracle: nonfinite flags, then Ge(x, x) fails only for NaN"
    );

    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![8]));
    let ordered = b.binary(BinOp::Ge, x, x);
    let unordered = b.binary_scalar(BinOp::Mul, ordered, Scalar::F32(-1.0));
    let unordered = b.binary_scalar(BinOp::Add, unordered, Scalar::F32(1.0));
    let failing = poot_test_util::graph_fixtures::finish_with_validations(
        b,
        x,
        &[(ValidationId(24), "self-compare", unordered)],
    )
    .unwrap();
    let inputs: HashMap<ValueId, Value> =
        HashMap::from([(x.id(), HostTensor::f32(vec![8], edges).into())]);
    let (store, slot_binds) = split_consts_and_slots(&failing, &inputs);
    let (bound, map) = ConstWeights::bind(&failing, &store);
    let program = staged(&bound, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let result = exec.step(exe, entry, &step_in, &mut NoSync);
    assert_eq!(contract_failure(result), cpu_failure(&failing, &inputs));
}

#[test]
fn wgpu_validation_view_source_is_materialized() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![2, 4]));
    let y = b.binary(BinOp::Mul, x, x);
    let flags = nonfinite_flags(&b, y);
    // Without a witness this transpose would be a strided view of `flags`: its only consumer is a
    // two-value Binary.
    let transposed = b.transpose(flags, vec![1, 0]);
    let output = b.binary(BinOp::Add, transposed, transposed);
    let graph = poot_test_util::graph_fixtures::finish_with_validations(
        b,
        output,
        &[(ValidationId(25), "transposed", transposed)],
    )
    .unwrap();
    let mut data = vec![1.0; 8];
    data[1] = f32::MAX;
    let inputs: HashMap<ValueId, Value> =
        HashMap::from([(x.id(), HostTensor::f32(vec![2, 4], data).into())]);
    let expected = cpu_failure(&graph, &inputs);
    assert_eq!(expected.lane, 2, "flags[0][1] is transposed[1][0]");
    let (store, slot_binds) = split_consts_and_slots(&graph, &inputs);
    let (bound, map) = ConstWeights::bind(&graph, &store);
    let program = staged(&bound, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let result = exec.step(exe, entry, &step_in, &mut NoSync);
    assert_eq!(contract_failure(result), expected);
}

/// SC-001 (Card 673): a deliberately noncanonical witness is refused at `compile_staged`, before any
/// allocation or upload, with the planner's typed `PlanError::ValidationWitnessNotCanonical` naming
/// the witness and why it is not canonical. `state_out - state_out` is exactly zero for a finite
/// lane, but `state_out` is float arithmetic over graph inputs, so the device witness admission
/// rejects the head before the planner emits a program (and before any executor allocates).
///
/// Mutation: drop the `admit_device_witnesses` call in `compile`, immediately before `fuse`; the
/// graph compiles and the row goes red on the `expect_err`.
#[test]
fn wgpu_validation_prefill_refuses_noncanonical_witness_at_compile_staged() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let b = Builder::new();
    let state_in = b.state_input(
        "accumulator",
        TensorType::f32(vec![4]),
        StateRole::Recurrent,
    );
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![4]));
    let state_out = b.binary(BinOp::Add, state_in, x);
    let output = b.binary_scalar(BinOp::Mul, state_out, Scalar::F32(2.0));
    let noncanonical = b.binary(BinOp::Sub, state_out, state_out);
    let rejected = poot_test_util::graph_fixtures::finish_with_state_and_validations(
        b,
        output,
        &[(state_in, state_out)],
        &[(ValidationId(27), "float-difference", noncanonical)],
    )
    .unwrap();
    let error = compile_staged(
        &rejected,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect_err("a noncanonical witness must be refused at compile_staged");
    match error {
        StagedCompileError::Compile(CompileError::Plan(cause)) => match &*cause {
            PlanError::ValidationWitnessNotCanonical { id, value, reason } => {
                assert_eq!(*id, ValidationId(27), "the refusal names the declaration");
                assert_eq!(
                    *value,
                    noncanonical.id(),
                    "the refusal names the witness value"
                );
                assert_eq!(
                    reason,
                    &DeviceWitnessRejection::GraphInput {
                        value: state_in.id()
                    },
                    "the refusal names why v{value} is not a canonical device witness"
                );
            }
            other => panic!("expected a noncanonical-witness refusal, got {other:?}"),
        },
        other => panic!("expected a compile plan refusal, got {other:?}"),
    }
}

/// A graph whose sole output IS one named state value: a production entry an `Executor` steps like any
/// other, which happens to let the test read a state buffer through its real step output (same pattern
/// as `executor_contract.rs`'s `state_probe_graph` - no `read_state` accessor exists).
fn state_probe_graph(name: &str, aval: TensorType) -> Graph<ValidationOutputs> {
    let b = Builder::new();
    let state = b.state_input(name, aval, StateRole::Recurrent);
    poot_test_util::graph_fixtures::finish_with_state_and_validations(
        b,
        state,
        &[(state, state)],
        &[],
    )
    .unwrap()
}

/// The same prefill entry, with a canonical witness: a poisoned step fails exactly as the CPU oracle does.
/// Zero-seeded state is now the contract's own job (automatic on first use, Card 546b), not an explicit
/// seed upload as the deleted `run_resident_kv` needed; two independent executables give the "seeded" and "poisoned" steps
/// each a fresh zero state, since a second `add_entry` on the SAME executable would bind the identical
/// "accumulator" state by name and inherit the first step's already-advanced value instead.
#[test]
fn wgpu_validation_prefill_rejects_a_failing_witness() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let b = Builder::new();
    let state_in = b.state_input(
        "accumulator",
        TensorType::f32(vec![4]),
        StateRole::Recurrent,
    );
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![4]));
    let state_out = b.binary(BinOp::Add, state_in, x);
    let output = b.binary_scalar(BinOp::Mul, state_out, Scalar::F32(2.0));
    let witness = nonfinite_flags(&b, state_out);
    let graph = poot_test_util::graph_fixtures::finish_with_state_and_validations(
        b,
        output,
        &[(state_in, state_out)],
        &[(ValidationId(28), "prefill-finite", witness)],
    )
    .unwrap();

    // Executable 1: seeded from zero, finite input, must publish and must let the state probe read
    // back the advanced state.
    let step: HashMap<ValueId, Value> = HashMap::from([(
        x.id(),
        HostTensor::f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]).into(),
    )]);
    let (store, slot_binds) = split_consts_and_slots(&graph, &step);
    let (bound, map) = ConstWeights::bind(&graph, &store);
    let program = staged(&bound, target);
    let exe1 = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry1 = exec.add_entry(exe1, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let published = f32_bytes(
        &exec
            .step(exe1, entry1, &step_in, &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    assert_eq!(published, vec![2.0, 4.0, 6.0, 8.0], "seeded from zeros");

    let probe_graph = state_probe_graph("accumulator", TensorType::f32(vec![4]));
    let probe_program = staged(&probe_graph, target);
    let probe_entry = exec.add_entry(exe1, &probe_program).unwrap();
    let probed = f32_bytes(
        &exec
            .step(exe1, probe_entry, &StepInputs::new(), &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    assert_eq!(probed, vec![1.0, 2.0, 3.0, 4.0]);
    exec.remove_entry(exe1, probe_entry).unwrap();
    exec.remove_entry(exe1, entry1).unwrap();
    exec.unload(exe1).unwrap();

    // Executable 2: a fresh zero state, poisoned input, must fail exactly as the CPU oracle does.
    let poisoned: HashMap<ValueId, Value> = HashMap::from([(
        x.id(),
        HostTensor::f32(vec![4], vec![0.0, 0.0, f32::NAN, 0.0]).into(),
    )]);
    let (store, slot_binds) = split_consts_and_slots(&graph, &poisoned);
    let (_, map) = ConstWeights::bind(&graph, &store);
    let exe2 = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry2 = exec.add_entry(exe2, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let result = exec.step(exe2, entry2, &step_in, &mut NoSync);
    let failure = contract_failure(result);
    let mut cpu_inputs = poisoned.clone();
    cpu_inputs.insert(state_in.id(), HostTensor::f32(vec![4], vec![0.0; 4]).into());
    assert_eq!(failure, cpu_failure(&graph, &cpu_inputs));
    exec.remove_entry(exe2, entry2).unwrap();
    exec.unload(exe2).unwrap();
}

#[test]
fn wgpu_validation_plan_uses_prepared_graph_ids() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![8]));
    let y = b.binary(BinOp::Mul, x, x);
    let primary = nonfinite_flags(&b, y);
    // A structurally identical witness: CSE remaps it onto the primary flags, so its original producer
    // equations do not exist in the graph the executor runs.
    let witness = nonfinite_flags(&b, y);
    assert_ne!(primary.id(), witness.id());
    let graph = poot_test_util::graph_fixtures::finish_with_validations(
        b,
        primary,
        &[(ValidationId(26), "duplicate", witness)],
    )
    .unwrap();
    let mut data = finite_input();
    data[4] = f32::NAN;
    let inputs: HashMap<ValueId, Value> =
        HashMap::from([(x.id(), HostTensor::f32(vec![8], data).into())]);
    let (store, slot_binds) = split_consts_and_slots(&graph, &inputs);
    let (bound, map) = ConstWeights::bind(&graph, &store);
    let program = staged(&bound, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let result = exec.step(exe, entry, &step_in, &mut NoSync);
    assert_eq!(contract_failure(result), cpu_failure(&graph, &inputs));
}

// ---- live-capacity attention (POOT-730) ------------------------------------------------------------

use poot_executor_parity::live_attention::live_step;
use poot_test_util::live_attention::LiveAttentionFixture;

fn assert_within(label: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance,
            "{label}: element {i}: device {g} vs oracle {w}"
        );
    }
}

/// SC-002 on wgpu: capacity 4096, the length derived in the graph as 64 from `Pos`, NaN in every key and value
/// from position 64 on. The plan bounds the loop by the live length (`bound=live`), and the device output is
/// finite and within the attention tier of both the CPU evaluation of the guarded graph and the first-64-key
/// oracle. Mutation: bound the generated loop by the capacity (`KeyBound::Live` read as `lk` in
/// `attention_body`); the device consumes the poisoned tail and the output is NaN.
#[test]
fn wgpu_live_attention_never_reads_the_poisoned_tail() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 4096,
        dk: 64,
        dv: 64,
        shared_dim: None,
        softcap: None,
        dtype: DType::F32,
    };
    let built = fixture.build(true);
    let inputs = fixture.inputs(&built, &[64], true);
    let program = staged(&built.graph, target);
    let planned_live = program.stages().any(|(_, _, stage)| {
        stage.planned().any(|(eqn, _)| {
            matches!(stage.kernel_choice(eqn),
                poot_graph_plan::KernelChoice::Generated(poot_kernelgen::KernelRequest::Attention(spec))
                    if spec.bound == poot_kernelgen::KeyBound::Live)
        })
    });
    assert!(
        planned_live,
        "the plan bounds the attention loop by the live length"
    );
    let got = live_step(&mut *exec, target, &built, &inputs).expect("a valid length publishes");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = poot_eval::eval(
        &built.graph,
        &inputs.values(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    assert_within("vs the guarded graph", &got, cpu.as_f32().unwrap(), 1e-4);
    assert_within(
        "vs the first 64 keys",
        &got,
        &fixture.oracle(&inputs, &built, &[64]),
        1e-4,
    );
}

/// POOT-1041 on wgpu: a BF16 or F16 live-attention graph is refused at planning with the typed `DtypeLowering`
/// on its `Select`, never run through a kernel that reads the two-byte lanes as something else: SPIR-V has no
/// native two-byte lane to move. The compile happens against the real device's target. Mutation: drop the
/// SPIR-V exclusion from the two-byte arm of the `Select` planning (`planner/elementwise.rs`); the plan
/// succeeds and this row goes red.
#[test]
fn wgpu_live_attention_two_byte_branches_are_a_typed_refusal() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    for dtype in [DType::BF16, DType::F16] {
        let fixture = LiveAttentionFixture {
            batch: 1,
            hq: 2,
            hkv: 1,
            cap: 9,
            dk: 8,
            dv: 8,
            shared_dim: None,
            softcap: None,
            dtype,
        };
        let built = fixture.build(true);
        let refused = compile_staged(
            &built.graph,
            &TargetSet::single(DeviceId(0), target),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
        );
        let Err(StagedCompileError::Compile(CompileError::Plan(error))) = refused else {
            panic!("{dtype:?}: the two-byte select must be refused, got {refused:?}");
        };
        let PlanError::Refused(refusal) = *error else {
            panic!("{dtype:?}: expected a typed refusal, got {error:?}");
        };
        assert_eq!(
            refusal.missing,
            poot_graph_plan::Capability::DtypeLowering,
            "{dtype:?}"
        );
        assert!(
            matches!(refusal.op, poot_graph_ir::OpKind::Select),
            "{dtype:?}: {refusal:?}"
        );
    }
}

/// SC-004 on wgpu: lengths 0, -1 and capacity + 1 fail through the validation packet exactly as the CPU oracle
/// does, with nothing published; the endpoints 1 and capacity run and match the oracle. Mutation: widen the
/// guard's bound in `ops::attention_live` (`lk + 8`); capacity + 1 then publishes.
#[test]
fn wgpu_live_attention_rejects_an_out_of_range_length_and_runs_the_endpoints() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 9,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: Some(30.0),
        dtype: DType::F32,
    };
    let built = fixture.build(true);
    for len in [0, -1, 10] {
        let inputs = fixture.inputs(&built, &[len], false);
        let result = live_step(&mut *exec, target, &built, &inputs);
        assert_eq!(
            contract_failure(result),
            cpu_failure(&built.graph, &inputs.values()),
            "len {len}"
        );
    }
    for len in [1, 9] {
        let inputs = fixture.inputs(&built, &[len], true);
        let got = live_step(&mut *exec, target, &built, &inputs)
            .unwrap_or_else(|e| panic!("len {len}: {e:?}"));
        assert_within(
            &format!("len {len}"),
            &got,
            &fixture.oracle(&inputs, &built, &[len]),
            1e-4,
        );
    }
}

/// POOT-1040 on wgpu: the live-attention graph with no declared length witness is never proved, so the matcher
/// declines it and it stays the generic primitives, key ends in exact I32 included. The plan has no attention
/// kernel, the device runs it, and with NaN in every key and value past the length the output is finite and
/// within the attention tier of the first-5-key oracle. Mutation: build `LiveKeys::new`'s key ends from an F32
/// iota plus a `Cast` to I32 (`ops::attention_over_keys`); the plan refuses the F32 -> I32 cast.
#[test]
fn wgpu_live_attention_declined_graph_runs_as_primitives() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 9,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: None,
        dtype: DType::F32,
    };
    let built = fixture.build(false);
    let inputs = fixture.inputs(&built, &[5], true);
    let program = staged(&built.graph, target);
    let planned_attention = program.stages().any(|(_, _, stage)| {
        stage.planned().any(|(eqn, _)| {
            matches!(
                stage.kernel_choice(eqn),
                poot_graph_plan::KernelChoice::Generated(poot_kernelgen::KernelRequest::Attention(
                    _
                ))
            )
        })
    });
    assert!(
        !planned_attention,
        "the unproved graph stays generic primitives"
    );
    let got = live_step(&mut *exec, target, &built, &inputs)
        .expect("a declined graph runs on the device");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = poot_eval::eval(
        &built.graph,
        &inputs.values(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    assert_within("vs the guarded graph", &got, cpu.as_f32().unwrap(), 1e-4);
    assert_within(
        "vs the first 5 keys",
        &got,
        &fixture.oracle(&inputs, &built, &[5]),
        1e-4,
    );
}
