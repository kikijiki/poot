//! Card 547b acceptance, PTX half of SC-002: a short value reusing a larger freed arena slot still
//! matches the CPU oracle through a real PTX dispatch (arena-slot reuse does not, by itself, corrupt
//! a real PTX dispatch - see `poot-wgpu-device/tests/buffer_plan.rs`'s module doc for why these two
//! `pootc`-compiled `Add`/`Reduce` kernels do not themselves read the per-dispatch kernarg length at
//! runtime, their bound baked from the equation's own static shape). The decoupled-length contract
//! itself (review F1: `KernelArgs::pack`/`PtxContext::dispatch_dev` publish the plan's own length for
//! every argument, never `PtxBuffer::elem_count()`) is proven by
//! [`len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity_on_ptx`] below, driving
//! `PtxDevice` directly with a shape-generic probe body that genuinely reads its operand's runtime
//! `Len()` - mirroring `poot-wgpu-device`/`poot-rocm-device`'s own siblings of the same name.

use std::sync::Arc;

use poot_executor::{Arg, Device, Dispatch, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::op::RedOp;
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_ptx_device::device::PtxDevice;
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::device_skip::open_or_skip;
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::BufferStorage;
use poot_tensor::DType;
use poot_test_util::assert_close_rel;

type Graph = poot_graph_ir::Graph;

const K: usize = 64;

fn require_ptx() -> Option<PtxDevice> {
    open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
}

/// POOT-1100 LP-3 (ptx): `Device::write_with` puts every byte at its address across upload-ring chunks. A
/// device buffer of two `UPLOAD_CHUNK_BYTES` and an odd tail is written whole and again from an
/// unaligned offset, then read back.
/// Mutation: in `UploadRing::write` copy chunk `k` to `dst + done - 1`; the read-back differs and this
/// row goes red.
#[test]
fn lp3_ptx_write_with_is_exact_across_ring_chunks() {
    let Some(mut device) = require_ptx() else {
        return;
    };
    poot_executor_parity::write_probe::chunked_write_is_exact(
        &mut device,
        poot_ptx_runtime::UPLOAD_CHUNK_BYTES,
        3,
    );
}

fn fixture() -> (Graph, Vec<f32>, Vec<f32>, f32) {
    let a: Vec<f32> = (0..K).map(|i| 1.0 + i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..K).map(|i| -0.5 + i as f32 * 0.02).collect();
    let c = -5.0f32;

    let builder = Builder::for_tests();
    let ta = builder.constant("a", TensorType::f32(vec![K])).unwrap();
    let tb = builder.constant("b", TensorType::f32(vec![K])).unwrap();
    let v1 = builder
        .binary(poot_graph_ir::op::BinOp::Add, ta, tb)
        .unwrap();
    let s1 = builder.reduce(RedOp::Sum, v1, 0, true).unwrap();
    let tc = builder.constant("c", TensorType::f32(vec![1])).unwrap();
    let v2 = builder
        .binary(poot_graph_ir::op::BinOp::Add, tc, tc)
        .unwrap();
    let s2 = builder.reduce(RedOp::Max, v2, 0, true).unwrap();
    let out = builder
        .binary(poot_graph_ir::op::BinOp::Add, s1, s2)
        .unwrap();
    (builder.finish(out).unwrap(), a, b, c)
}

fn store(a: &[f32], b: &[f32], c: f32) -> Arc<WeightStore> {
    let mut wb = WeightStore::builder();
    for (name, bytes) in [("a", a.to_vec()), ("b", b.to_vec()), ("c", vec![c])] {
        let dense = DenseWeight::try_new(
            DType::F32,
            vec![bytes.len()],
            StoredBytes::from(bytemuck_bytes(&bytes)),
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

const REPLAY: CompileOptions = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);

fn staged(g: &Graph, target: Target) -> StagedProgram {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &REPLAY,
    )
    .expect("the fixture compiles")
}

fn oracle(a: &[f32], b: &[f32], c: f32) -> f32 {
    let s1: f32 = a.iter().zip(b).map(|(&x, &y)| x + y).sum();
    s1 + (c + c)
}

/// SC-002 (PTX half): a short value (`s2`, [1]) reusing a larger freed arena slot ([K], `v1`'s)
/// still matches the CPU oracle through a real PTX dispatch.
#[test]
fn a_short_value_reusing_a_larger_slot_matches_the_oracle_on_ptx() {
    let Some(device) = require_ptx() else {
        return;
    };
    let target = device.target();
    let (g, a, b, c) = fixture();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let store = store(&a, &b, c);
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(&g);
    let map = Arc::new(weights.map());
    let exe = exec.load_weights(store, map).unwrap();
    let entry = exec.add_entry(exe, &staged(&named, target)).unwrap();
    let mut out = exec
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let got = bytemuck_f32(&out.read().unwrap());
    assert_close_rel(&got, &[oracle(&a, &b, c)], 1e-5);
}

/// SC-002 (review F1, decoupled-length proof on real PTX hardware): mirrors `poot-wgpu-device`/
/// `poot-rocm-device`'s own `len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity[_on_rocm]`
/// exactly, on `PtxDevice`. `poot_kernel_ir::test_support::len_probe_kernel`'s body reads
/// `Len(len_src)` at kernel-build time, so its compiled PTX has no shape baked in. `len_src` is
/// allocated with `SLOT_CAPACITY=64` device elements (an arena slot reused from a larger occupant),
/// but the dispatch publishes `Arg::elems = REAL_LEN = 4` for it. Drives `PtxDevice` directly through
/// the identical begin/dispatch/finish/replay/synchronize sequence `Engine::step` uses internally.
///
/// Mutation (card 547b, applied by hand, reverted, never left in the tree): in
/// `crates/poot-ptx-device/src/device.rs`'s `Device::dispatch`, build `in_lens` from
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

    let body = poot_kernel_ir::test_support::len_probe_kernel();
    let artifact = poot_codegen::KernelCache::open(poot_codegen::Target::Nvptx)
        .load_or_compile(
            &body,
            poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
        )
        .expect("len_probe must compile to PTX");
    let compiled = poot_codegen::kernel_handle(&body, &artifact).unwrap();
    let kernel = device
        .load_kernel("sc002_len_probe", compiled)
        .expect("load the len_probe kernel");

    let len_src = device
        .allocate(BufferRole::Input, BufferStorage::f32(), SLOT_CAPACITY)
        .unwrap();
    poot_executor::write_slice(
        &mut device,
        &len_src,
        0,
        &bytemuck_bytes(&vec![0.0f32; SLOT_CAPACITY]),
    )
    .unwrap();
    let out_buf = device
        .allocate(BufferRole::Output, BufferStorage::f32(), 1)
        .unwrap();

    device.begin().unwrap();
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
            launch: poot_executor::LaunchKind::Grid {
                threads: [1, 1, 1],
                workgroup: [1, 1, 1],
            },
            work: 1,
        })
        .unwrap();
    let recording = device
        .finish()
        .unwrap()
        .expect("a device always returns a recording");
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

/// POOT-1117: a `LaunchKind::Aie` dispatch handed to the PTX device is the typed refusal
/// `PtxError::WrongKernelTarget` naming the AIE target, and the open capture is abortable. Mutation: let
/// `PtxDevice::dispatch` accept `LaunchKind::Aie` (bind its arm to a fixture grid); `dispatch` returns `Ok` and
/// the row goes red.
#[test]
fn an_aie_launch_is_a_typed_refusal_on_ptx() {
    use poot_ptx_device::PtxGpuError;
    use poot_target::{AieTargetId, TargetId};

    let Some(mut device) = require_ptx() else {
        return;
    };
    let body = poot_kernel_ir::test_support::len_probe_kernel();
    let artifact = poot_codegen::KernelCache::open(poot_codegen::Target::Nvptx)
        .load_or_compile(
            &body,
            poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
        )
        .expect("len_probe must compile to PTX");
    let compiled = poot_codegen::kernel_handle(&body, &artifact).unwrap();
    let kernel = device
        .load_kernel("aie_launch_probe", compiled)
        .expect("load the probe kernel");
    let input = device
        .allocate(BufferRole::Input, BufferStorage::f32(), 4)
        .unwrap();
    let output = device
        .allocate(BufferRole::Output, BufferStorage::f32(), 1)
        .unwrap();

    device.begin().unwrap();
    let error = device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[Arg {
                buffer: &input,
                elems: 4,
            }],
            output: Arg {
                buffer: &output,
                elems: 1,
            },
            launch: poot_executor::LaunchKind::Aie(poot_executor::AieInvocation::new(
                AieTargetId::Xdna2,
                std::num::NonZeroU32::new(1).unwrap(),
            )),
            work: 1,
        })
        .expect_err("a thread-grid device runs no AIE invocation");
    assert!(
        matches!(
            &error,
            PtxGpuError::Ptx(ptx)
                if matches!(
                    **ptx,
                    poot_ptx_runtime::PtxError::WrongKernelTarget {
                        actual: TargetId::Aie(AieTargetId::Xdna2)
                    }
                )
        ),
        "{error:?}"
    );
    device.abort().unwrap();
}

/// POOT-1205 SC-002: the engine refuses a dispatch whose asserted length exceeds its buffer by one element
/// with the shared `DispatchError::ArgExceedsBuffer`, before this device is handed the dispatch.
///
/// Mutation: remove the `check_arg_lengths` call in `Engine::walk` (`crates/poot-executor/src/engine.rs`);
/// the device records the dispatch and the step no longer fails, so this row goes red.
#[test]
fn an_over_length_dispatch_is_refused_by_the_engine_on_ptx() {
    let Some(device) = require_ptx() else {
        return;
    };
    poot_executor_parity::dispatch_length::over_length_dispatch_is_refused(device);
}
