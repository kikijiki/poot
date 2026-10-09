//! Card 553 acceptance for `VulkanDevice`, the raw-Vulkan implementation of the executor contract's
//! `poot_executor::Device`: SC-004 (a buffer moves across threads), SC-005 (capture once, replay many)
//! and the rows for the mechanisms this device owns that the parity table does not reach - dispatch
//! reads `Arg::elems`, `copy` is `vkCmdCopyBuffer`, a long recording is cut into several submissions,
//! kernel asserts surface at `synchronize`, device time is measured only when asked for. The parity
//! rows are in `tests/parity.rs`, the generation rows in `tests/generate.rs`. A row skips when no
//! Vulkan device opens (`POOT_REQUIRE_VULKAN=1` turns that into a failure); rows run serially under
//! `/tmp/poot-gpu.lock`.

use std::path::PathBuf;
use std::sync::Arc;

use poot_executor::{
    Arg, BufferRole, Device, Dispatch, Engine, Executor, ExecutorStats, NoSync, StepInputs,
};
use poot_graph_plan::{ImportedKernel, Submission};
use poot_runtime_common::{CompiledKernel, DeviceBackend};
use poot_target::{BufferStorage, ElementKind, LogicalDType};
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::weight_map_oracle::oracle_for;
use poot_test_util::{StepFixture, assert_close_rel};
use poot_vulkan_device::{VulkanDevice, VulkanKernel};

fn vulkan() -> Option<VulkanDevice> {
    open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new())
}

fn step_inputs(step: &[StepFixture]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for value in step {
        inputs.push(value.key.clone(), value.tensor.shape(), value.tensor.view());
    }
    inputs
}

fn compile_for_vulkan(body: &poot_kernel_ir::Body, name: &str) -> CompiledKernel {
    let dir: PathBuf = std::env::temp_dir().join("poot-vulkan-device-test");
    std::fs::create_dir_all(&dir).unwrap();
    let target = poot_codegen::Target::SpirvVulkan;
    let out = poot_codegen::artifact_path(&dir, name, target);
    poot_codegen::compile(body, target, &out).expect("compile SPIR-V");
    let spv = std::fs::read(&out).unwrap();
    poot_codegen::kernel_handle(body, target, spv)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// SC-005: a qwen2 decode entry on `Engine<VulkanDevice>` matches the CPU oracle at every one of 16
/// replayed steps, and capture is the default: `stats()` shows one recording, and the device itself one
/// native command-buffer recording, after the 16 steps.
///
/// Mutation: record every step (`needs_recording` in `Engine::step` always true); `recordings` is 16 and
/// the row goes red.
#[test]
fn sixteen_replayed_steps_record_once() {
    const STEPS: usize = 16;
    let Some(device) = vulkan() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let fixture = poot_executor_parity::qwen2_decode_fixture(STEPS);
    assert_eq!(fixture.steps.len(), STEPS);
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let exe = engine
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor_parity::fixture_weights(&fixture),
        )
        .unwrap();
    let entry = engine
        .add_entry(
            exe,
            &poot_executor_parity::staged(&fixture, target, Submission::Replay),
        )
        .unwrap();
    for (i, step) in fixture.steps.iter().enumerate() {
        let want = oracle(step);
        let got = engine
            .step(exe, entry, &step_inputs(step), &mut NoSync)
            .unwrap()
            .to_host()
            .unwrap();
        assert_close_rel(got.as_f32().unwrap(), want.as_f32().unwrap(), 5e-3);
        let ExecutorStats {
            recordings,
            replays,
            ..
        } = engine.stats();
        assert_eq!(recordings, 1, "step {i}: one entry records once");
        assert_eq!(replays, i as u64 + 1, "step {i}: every step replays");
    }
    assert_eq!(
        engine.device().context().graphs_recorded(),
        1,
        "the device recorded one native command-buffer graph for {STEPS} steps"
    );
}

/// SC-004: a `VulkanDevice` buffer moves into `std::thread::spawn` and reads back there.
///
/// Mutation: restore `Rc` for the buffer's shared owner (`DeviceBuffer::inner` and `DeviceOwner` handles
/// in `poot-vulkan-runtime`); this file stops compiling with `` `Rc<DeviceBufferInner>` cannot be sent
/// between threads safely `` (E0277), the row's required red signal (a soundness row is proven by the
/// compiler, ADR-0090).
#[test]
fn a_vulkan_buffer_moves_into_another_thread() {
    let Some(mut device) = vulkan() else {
        return;
    };
    let buffer = device
        .allocate(BufferRole::Input, BufferStorage::f32(), 4)
        .unwrap();
    device
        .write(&buffer, 0, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let clone = buffer.clone();
    let reader = std::thread::spawn(move || {
        let mut out = [0u8; 16];
        buffer.read_bytes(&mut out).unwrap();
        f32_values(&out)
    });
    assert_eq!(reader.join().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    // The other handle to the same allocation still reads it, and the device that made it still works.
    let mut out = [0u8; 16];
    device.read(&clone, &mut out).unwrap();
    assert_eq!(f32_values(&out), [1.0, 2.0, 3.0, 4.0]);
}

/// A loaded `add` kernel (`out[i] = a[i] + b[i]` under each argument's own length).
fn add_kernel(device: &mut VulkanDevice) -> VulkanKernel {
    let compiled = compile_for_vulkan(&poot_kernel_ir::fixtures::add_kernel(), "add");
    device.load_kernel("add", compiled).expect("load add")
}

fn f32_buffer(
    device: &mut VulkanDevice,
    role: BufferRole,
    values: &[f32],
) -> poot_vulkan_runtime::DeviceBuffer {
    let buffer = device
        .allocate(role, BufferStorage::f32(), values.len())
        .unwrap();
    device.write(&buffer, 0, &f32_bytes(values)).unwrap();
    buffer
}

/// Record `dst = lhs + rhs` over `n` elements into the open step.
fn record_add(
    device: &mut VulkanDevice,
    kernel: &VulkanKernel,
    lhs: &poot_vulkan_runtime::DeviceBuffer,
    rhs: &poot_vulkan_runtime::DeviceBuffer,
    dst: &poot_vulkan_runtime::DeviceBuffer,
    n: usize,
) {
    let elems = n as u32;
    device
        .dispatch(Dispatch {
            kernel,
            inputs: &[Arg { buffer: lhs, elems }, Arg { buffer: rhs, elems }],
            output: Arg { buffer: dst, elems },
            threads: [elems, 1, 1],
            workgroup: [64, 1, 1],
            work: n as u64,
        })
        .unwrap();
}

fn read_f32(device: &mut VulkanDevice, buffer: &poot_vulkan_runtime::DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; buffer.byte_len()];
    device.read(buffer, &mut bytes).unwrap();
    f32_values(&bytes)
}

/// Dispatch reads each argument's own element count, never the bound buffer's capacity: an `add` over
/// 4-element buffers told its operands hold 2 elements writes 2 outputs and leaves the rest of the output
/// buffer as it was.
///
/// Mutation: bind every argument's `buffer.elem_count()` in place of `Binding::elems` in
/// `Context::record_dispatch`'s length block; the kernel writes all 4 outputs and the row goes red.
#[test]
fn dispatch_reads_the_argument_element_count_not_the_buffer_capacity() {
    let Some(mut device) = vulkan() else {
        return;
    };
    let kernel = add_kernel(&mut device);
    let a = f32_buffer(&mut device, BufferRole::Input, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32_buffer(&mut device, BufferRole::Input, &[10.0, 20.0, 30.0, 40.0]);
    let out = f32_buffer(&mut device, BufferRole::Output, &[-1.0; 4]);
    device.begin(Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[
                Arg {
                    buffer: &a,
                    elems: 2,
                },
                Arg {
                    buffer: &b,
                    elems: 2,
                },
            ],
            output: Arg {
                buffer: &out,
                elems: 2,
            },
            threads: [4, 1, 1],
            workgroup: [64, 1, 1],
            work: 4,
        })
        .unwrap();
    let recording = device.finish().unwrap().expect("a recording");
    device.replay(&recording).unwrap();
    device.synchronize().unwrap();
    assert_eq!(read_f32(&mut device, &out), [11.0, 22.0, -1.0, -1.0]);
}

/// `copy` is `vkCmdCopyBuffer` in the command stream, ordered after the dispatch that wrote its source
/// and before the dispatch that reads its destination: `x + b` is copied into `y`, then `y + b` is
/// computed, so the second result sees the copy.
///
/// Mutation: drop the barrier `Context::record_copy` records before the copy (`barrier_before_command`);
/// the copy can read the source before the add wrote it.
#[test]
fn a_copy_is_ordered_between_the_dispatches_around_it() {
    let Some(mut device) = vulkan() else {
        return;
    };
    let kernel = add_kernel(&mut device);
    let n = 1024;
    let x = f32_buffer(&mut device, BufferRole::Input, &vec![1.0; n]);
    let b = f32_buffer(&mut device, BufferRole::Input, &vec![1.0; n]);
    let sum = f32_buffer(&mut device, BufferRole::Activation, &vec![0.0; n]);
    let copy = f32_buffer(&mut device, BufferRole::Activation, &vec![0.0; n]);
    let out = f32_buffer(&mut device, BufferRole::Output, &vec![0.0; n]);
    device.begin(Submission::Replay).unwrap();
    record_add(&mut device, &kernel, &x, &b, &sum, n);
    device.copy(&sum, &copy).unwrap();
    record_add(&mut device, &kernel, &copy, &b, &out, n);
    let recording = device.finish().unwrap().expect("a recording");
    device.replay(&recording).unwrap();
    device.synchronize().unwrap();
    assert_eq!(read_f32(&mut device, &out), vec![3.0; n]);
    assert_eq!(read_f32(&mut device, &copy), vec![2.0; n]);
}

/// `copy` moves every byte of the allocation for every element width, padding included: a 1-, 2- and
/// 4-byte element buffer with an element count that leaves a padded tail copies source bytes and source
/// padding over a destination pre-filled with a sentinel.
#[test]
fn copy_round_trips_every_element_width_with_no_sentinel_left() {
    let Some(mut device) = vulkan() else {
        return;
    };
    let storages = [
        (BufferStorage::raw_bytes(), 1usize),
        (
            BufferStorage::dense(ElementKind::Bf16, LogicalDType::Bf16),
            2,
        ),
        (BufferStorage::f32(), 4),
    ];
    for (storage, width) in storages {
        let elems = 7;
        let src = device.allocate(BufferRole::Input, storage, elems).unwrap();
        let dst = device.allocate(BufferRole::Output, storage, elems).unwrap();
        let payload: Vec<u8> = (0..elems * width).map(|i| i as u8 + 1).collect();
        device.write(&src, 0, &payload).unwrap();
        device.write(&dst, 0, &vec![0xAA; dst.byte_len()]).unwrap();
        device.begin(Submission::Replay).unwrap();
        device.copy(&src, &dst).unwrap();
        let recording = device.finish().unwrap().expect("a recording");
        device.replay(&recording).unwrap();
        device.synchronize().unwrap();
        let mut got = vec![0u8; dst.byte_len()];
        device.read(&dst, &mut got).unwrap();
        assert_eq!(
            &got[..payload.len()],
            &payload[..],
            "width {width}: payload"
        );
        assert!(
            got[payload.len()..].iter().all(|&byte| byte == 0),
            "width {width}: the source's zero padding, not the sentinel: {:?}",
            &got[payload.len()..]
        );
    }
}

/// A recording longer than one submission's budget is cut into several command buffers and replays to the
/// same result: 1030 chained adds cross the 1024-dispatch bound, so the graph holds two command buffers,
/// and the chained value is exact. Five dispatches each carrying 3M work units cross the work bound on
/// every dispatch after the first, so that graph holds five.
///
/// Mutation: never cut (`OpenStep::would_overflow` always false); both graphs hold one command buffer and
/// the row goes red on the segment counts.
#[test]
fn a_long_recording_is_cut_into_several_submissions_and_replays_exactly() {
    let Some(mut device) = vulkan() else {
        return;
    };
    let kernel = add_kernel(&mut device);
    let one = f32_buffer(&mut device, BufferRole::Input, &[1.0; 4]);
    let ping = f32_buffer(&mut device, BufferRole::Activation, &[0.0; 4]);
    let pong = f32_buffer(&mut device, BufferRole::Activation, &[0.0; 4]);
    let chain = |device: &mut VulkanDevice, dispatches: usize, work: u64| {
        device.write(&ping, 0, &f32_bytes(&[0.0; 4])).unwrap();
        device.begin(Submission::Replay).unwrap();
        for i in 0..dispatches {
            let (src, dst) = if i % 2 == 0 {
                (&ping, &pong)
            } else {
                (&pong, &ping)
            };
            device
                .dispatch(Dispatch {
                    kernel: &kernel,
                    inputs: &[
                        Arg {
                            buffer: src,
                            elems: 4,
                        },
                        Arg {
                            buffer: &one,
                            elems: 4,
                        },
                    ],
                    output: Arg {
                        buffer: dst,
                        elems: 4,
                    },
                    threads: [4, 1, 1],
                    workgroup: [64, 1, 1],
                    work,
                })
                .unwrap();
        }
        device.finish().unwrap().expect("a recording")
    };

    let long = chain(&mut device, 1030, 4);
    assert_eq!(long.dispatch_count(), 1030);
    assert_eq!(
        long.segment_count(),
        2,
        "1030 dispatches exceed one command buffer's 1024"
    );
    device.replay(&long).unwrap();
    device.synchronize().unwrap();
    // 1030 adds of 1: an even count ends back in `ping`.
    assert_eq!(read_f32(&mut device, &ping), vec![1030.0; 4]);

    let heavy = chain(&mut device, 5, 3_000_000);
    assert_eq!(
        heavy.segment_count(),
        5,
        "3M + 3M work exceeds a command buffer's 4M budget"
    );
    device.replay(&heavy).unwrap();
    device.synchronize().unwrap();
    // 5 adds of 1: an odd count ends in `pong`.
    assert_eq!(read_f32(&mut device, &pong), vec![5.0; 4]);
}

/// A kernel whose assert fires (`out[i] = data[idx[i]]` with `idx[2] == 4` over a 4-element `data`)
/// surfaces from `synchronize` as a fault the engine classifies, not from `replay` as a device failure,
/// and the next replay of a good input is clean.
///
/// Mutation: `VulkanDevice::classify_fault` always `None`; the fault is a generic device error and the
/// row goes red.
#[test]
fn a_kernel_assert_is_a_fault_raised_at_synchronize() {
    let Some(mut device) = vulkan() else {
        return;
    };
    let body = ImportedKernel::AssertTrap.body().clone();
    let kernel = device
        .load_kernel("assert_trap", compile_for_vulkan(&body, "assert_trap"))
        .expect("load assert_trap");
    let idx = device
        .allocate(BufferRole::Input, BufferStorage::i32(), 4)
        .unwrap();
    device
        .write(
            &idx,
            0,
            &[3i32, 0, 4, 1]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
    let data = f32_buffer(&mut device, BufferRole::Input, &[10.0, 20.0, 30.0, 40.0]);
    let out = f32_buffer(&mut device, BufferRole::Output, &[0.0; 4]);
    device.begin(Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[
                Arg {
                    buffer: &idx,
                    elems: 4,
                },
                Arg {
                    buffer: &data,
                    elems: 4,
                },
            ],
            output: Arg {
                buffer: &out,
                elems: 4,
            },
            threads: [4, 1, 1],
            workgroup: [64, 1, 1],
            work: 4,
        })
        .unwrap();
    let recording = device.finish().unwrap().expect("a recording");
    device
        .replay(&recording)
        .expect("a fired assert is not a replay failure");
    let error = device
        .synchronize()
        .expect_err("idx[2] == 4 is out of range for a 4-element data buffer");
    let (kernel_name, code) = device
        .classify_fault(&error)
        .expect("a kernel assert classifies as a fault");
    assert_eq!(kernel_name, "assert_trap");
    assert_ne!(
        code, 0,
        "a fired trap's code is never the no-fault sentinel"
    );

    // The error word is cleared for the next replay: the same recording over in-range indices is clean.
    device
        .write(
            &idx,
            0,
            &[3i32, 0, 2, 1]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
    device.replay(&recording).unwrap();
    device.synchronize().expect("no fault on in-range indices");
    assert_eq!(read_f32(&mut device, &out), [40.0, 10.0, 30.0, 20.0]);
}

/// The Weight and State roles charge the same live bytes on raw Vulkan and wgpu for the same decode
/// entry (the device's own counters, not the engine's tally).
///
/// Mutation: charge every allocation as `Activation` in `Context::alloc_storage`'s caller
/// (`VulkanDevice::allocate` passes a fixed role); the Weight row reads 0 on Vulkan and goes red.
#[test]
fn memory_counters_charge_the_roles_a_decode_entry_allocates_as_wgpu_does() {
    let Some(device) = vulkan() else {
        return;
    };
    let Some(wgpu) = open_or_skip(DeviceBackend::Wgpu, poot_gpu::device::WgpuDevice::new()) else {
        return;
    };
    let fixture = poot_executor_parity::qwen2_decode_fixture(8);
    let load = |target, device: &mut dyn Executor| {
        let exe = device
            .load_weights(
                Arc::new(fixture.store.clone()),
                poot_executor_parity::fixture_weights(&fixture),
            )
            .unwrap();
        device
            .add_entry(
                exe,
                &poot_executor_parity::staged(&fixture, target, Submission::Replay),
            )
            .unwrap();
        device.stats().memory
    };
    let (vulkan_target, wgpu_target) = (device.target(), wgpu.target());
    let (mut vulkan_engine, mut wgpu_engine) = (Engine::new(device), Engine::new(wgpu));
    let on_vulkan = load(vulkan_target, &mut vulkan_engine);
    let on_wgpu = load(wgpu_target, &mut wgpu_engine);
    let live = |memory: &[(BufferRole, poot_executor::MemoryCounterSnapshot)], role| {
        memory
            .iter()
            .find(|(r, _)| *r == role)
            .map_or(0, |(_, s)| s.live_bytes)
    };
    for role in [BufferRole::Weight, BufferRole::State] {
        assert!(
            live(&on_vulkan, role) > 0,
            "{role:?} is charged: {on_vulkan:?}"
        );
        assert_eq!(
            live(&on_vulkan, role),
            live(&on_wgpu, role),
            "{role:?} live bytes: vulkan {on_vulkan:?} wgpu {on_wgpu:?}"
        );
    }
}

/// POOT-1081: the raw Vulkan device reports what it could still allocate right now, read from
/// `VK_EXT_memory_budget` at the call, not the open-time ceiling repeated: it sits within the ceiling the
/// device reported at open, and it falls by about what a held allocation takes.
///
/// Mutation: `free_memory_bytes` returns `None` and the first assertion goes red; it returns the open-time
/// ceiling (cached, not read live) and the drop assertion goes red; it returns `u64::MAX` and the ceiling
/// assertion goes red.
#[test]
fn the_vulkan_device_reports_live_free_memory_within_its_open_time_ceiling() {
    const HELD_ELEMS: usize = 64 << 20; // 256 MiB of f32
    let Some(mut device) = vulkan() else {
        return;
    };
    let ceiling = device
        .target()
        .caps
        .memory_budget_bytes
        .expect("the vulkan device queried its heap budget at open");
    let free = device
        .free_memory_bytes()
        .expect("the vulkan device reports live free memory");
    assert!(
        free > 0,
        "a device with no free memory cannot run this test"
    );
    assert!(
        free <= ceiling,
        "free {free} is over the open-time ceiling {ceiling}"
    );
    let held = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), HELD_ELEMS)
        .unwrap();
    let after = device.free_memory_bytes().unwrap();
    let taken = (HELD_ELEMS * 4) as u64;
    assert!(
        free.saturating_sub(after) >= taken / 2,
        "holding {taken} bytes moved free memory from {free} to {after}"
    );
    drop(held);
}
