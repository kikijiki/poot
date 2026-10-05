//! Card 547b acceptance, PTX half of SC-002: a short value reusing a larger freed arena slot still
//! matches the CPU oracle through a real PTX dispatch (arena-slot reuse does not, by itself, corrupt
//! a real PTX dispatch - see `poot-gpu/tests/buffer_plan.rs`'s module doc for why these two
//! `pootc`-compiled `Add`/`Reduce` kernels do not themselves read the per-dispatch kernarg length at
//! runtime, their bound baked from the equation's own static shape). The decoupled-length contract
//! itself (review F1: `KernelArgs::pack`/`PtxContext::dispatch_dev` publish the plan's own length for
//! every argument, never `PtxBuffer::elem_count()`) is proven by
//! [`len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity_on_ptx`] below, driving
//! `PtxDevice` directly with a shape-generic probe body that genuinely reads its operand's runtime
//! `Len()` - mirroring `poot-gpu`/`poot-rocm-gpu`'s own siblings of the same name.

use std::sync::Arc;

use poot_executor::{Arg, BufferRole, Device, Dispatch, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::op::RedOp;
use poot_graph_ir::{Builder, TensorType, ValidationOutputs};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_ptx_gpu::device::PtxDevice;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_target::BufferStorage;
use poot_tensor::DType;
use poot_test_util::assert_close_rel;
use poot_test_util::device_skip::open_or_skip;

type Graph = poot_graph_ir::Graph<ValidationOutputs>;

const K: usize = 64;

fn require_ptx() -> Option<PtxDevice> {
    open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
}

fn fixture() -> (Graph, Vec<f32>, Vec<f32>, f32) {
    let a: Vec<f32> = (0..K).map(|i| 1.0 + i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..K).map(|i| -0.5 + i as f32 * 0.02).collect();
    let c = -5.0f32;

    let builder = Builder::new();
    let ta = builder.constant("a", TensorType::f32(vec![K]));
    let tb = builder.constant("b", TensorType::f32(vec![K]));
    let v1 = builder.binary(poot_graph_ir::op::BinOp::Add, ta, tb);
    let s1 = builder.reduce(RedOp::Sum, v1, 0, true);
    let tc = builder.constant("c", TensorType::f32(vec![1]));
    let v2 = builder.binary(poot_graph_ir::op::BinOp::Add, tc, tc);
    let s2 = builder.reduce(RedOp::Max, v2, 0, true);
    let out = builder.binary(poot_graph_ir::op::BinOp::Add, s1, s2);
    (builder.finish(out).with_validations(Vec::new()), a, b, c)
}

fn store(a: &[f32], b: &[f32], c: f32) -> Arc<WeightStore> {
    let mut wb = WeightStore::builder();
    for (name, bytes) in [("a", a.to_vec()), ("b", b.to_vec()), ("c", vec![c])] {
        let dense = DenseWeight::try_new(
            DType::F32,
            vec![bytes.len()],
            Arc::from(bytemuck_bytes(&bytes)),
        )
        .unwrap();
        wb.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    Arc::new(wb.build())
}

fn bytemuck_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

const UNFUSED_REPLAY: CompileOptions = CompileOptions {
    execution: Submission::Replay,
    fusion: FusionPolicy::MoeHangGuard,
    limits: poot_graph_plan::CompileLimits::STANDARD,
};

fn staged(g: &Graph, target: Target) -> StagedProgram<ValidationOutputs> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &UNFUSED_REPLAY,
    )
    .expect("the fixture compiles unfused")
}

fn oracle(a: &[f32], b: &[f32], c: f32) -> f32 {
    let s1: f32 = a.iter().zip(b).map(|(&x, &y)| x + y).sum();
    s1 + (c + c)
}

/// SC-002 (PTX half): a short value (`v2`, [1]) reusing a larger freed arena slot ([K], `v1`'s)
/// still matches the CPU oracle through a real PTX dispatch.
#[test]
fn a_short_value_reusing_a_larger_slot_matches_the_oracle_on_ptx() {
    let Some(device) = require_ptx() else {
        return;
    };
    let target = device.target();
    let (g, a, b, c) = fixture();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let exe = exec
        .load_weights(store(&a, &b, c), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &staged(&g, target)).unwrap();
    let mut out = exec
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let got = bytemuck_f32(&out.read().unwrap());
    assert_close_rel(&got, &[oracle(&a, &b, c)], 1e-5);
}

/// SC-002 (review F1, decoupled-length proof on real PTX hardware): mirrors `poot-gpu`/
/// `poot-rocm-gpu`'s own `len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity[_on_rocm]`
/// exactly, on `PtxDevice`. `poot_test_util::kernel_fixtures::len_probe_kernel`'s body reads
/// `Len(len_src)` at kernel-build time, so its compiled PTX has no shape baked in. `len_src` is
/// allocated with `SLOT_CAPACITY=64` device elements (an arena slot reused from a larger occupant),
/// but the dispatch publishes `Arg::elems = REAL_LEN = 4` for it. Drives `PtxDevice` directly through
/// the identical begin/dispatch/finish/replay/synchronize sequence `Engine::step` uses internally.
///
/// Mutation (card 547b, applied by hand, reverted, never left in the tree): in
/// `crates/poot-ptx-gpu/src/device.rs`'s `Device::dispatch`, build `in_lens` from
/// `d.inputs.iter().map(|a| a.buffer.elem_count())` instead of `a.elems` (the pre-547b placeholder).
/// Observed: the probe reports `64.0` (the buffer's full capacity) instead of `4.0`, and the row goes
/// red; reverting restored `4.0`.
#[test]
fn len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity_on_ptx() {
    let Some(mut device) = require_ptx() else {
        return;
    };
    const REAL_LEN: usize = 4;
    const SLOT_CAPACITY: usize = 64;

    let body = poot_test_util::kernel_fixtures::len_probe_kernel();
    let dir = poot_codegen::kernel_cache_root("ptx-sc002-len-probe", poot_codegen::Target::Nvptx);
    let out_path =
        poot_codegen::artifact_path(&dir, "sc002_len_probe", poot_codegen::Target::Nvptx);
    poot_codegen::compile(&body, poot_codegen::Target::Nvptx, &out_path)
        .expect("len_probe must compile to PTX");
    let bytes = std::fs::read(&out_path).unwrap();
    let compiled = poot_codegen::kernel_handle(&body, poot_codegen::Target::Nvptx, bytes);
    let kernel = device
        .load_kernel("sc002_len_probe", compiled)
        .expect("load the len_probe kernel");

    let len_src = device
        .allocate(BufferRole::Input, BufferStorage::f32(), SLOT_CAPACITY)
        .unwrap();
    device
        .write(&len_src, &bytemuck_bytes(&vec![0.0f32; SLOT_CAPACITY]))
        .unwrap();
    let out_buf = device
        .allocate(BufferRole::Output, BufferStorage::f32(), 1)
        .unwrap();

    device.begin(Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[Arg {
                buffer: &len_src,
                elems: REAL_LEN as u32,
            }],
            output: Arg {
                buffer: &out_buf,
                elems: 1,
            },
            threads: [1, 1, 1],
            workgroup: [1, 1, 1],
            work: 1,
        })
        .unwrap();
    let recording = device
        .finish()
        .unwrap()
        .expect("Submission::Replay always returns a recording");
    device
        .replay(&recording)
        .expect("replay only re-encodes metadata; it never runs the dispatch itself");
    device.synchronize().unwrap();

    let mut out_bytes = [0u8; 4];
    device.read(&out_buf, &mut out_bytes).unwrap();
    let probed_len = f32::from_le_bytes(out_bytes);
    assert_eq!(
        probed_len, REAL_LEN as f32,
        "Len(len_src) inside the kernel must equal the dispatch's own Arg::elems ({REAL_LEN}), never \
         len_src's SLOT_CAPACITY={SLOT_CAPACITY} device buffer capacity"
    );
}
