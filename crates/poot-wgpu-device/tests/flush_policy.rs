//! POOT-1232 SC-004: how many submits a wgpu stage costs follows the device's `watchdog_budget`, not a
//! constant. A device with a budget cuts a long recording into several submits; a device without one submits
//! the stage once. Driven through the production `begin`/`dispatch`/`finish`/`replay`, counting the context's
//! own compute submits. Rows run serially under `/tmp/poot-gpu.lock`.

use poot_executor::{Arg, Device, Dispatch};
use poot_runtime_common::device_skip::open_or_skip;
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::{BufferStorage, DeviceCaps};
use poot_wgpu_device::device::WgpuDevice;

/// What replaying one recording cost, and what it computed.
struct Replay {
    /// Compute submits the replay made.
    submits: usize,
    /// Blocking device polls the replay itself made, before any synchronize.
    waits: usize,
    /// The chained value: each add contributes 1.
    value: f32,
}

/// Record `dispatches` chained adds of `work` units each into one recording and replay it.
fn replay_chain(device: &mut WgpuDevice, dispatches: usize, work: u64) -> Replay {
    let body = poot_kernel_ir::fixtures::add_kernel();
    let dir = std::env::temp_dir().join("poot-wgpu-device-flush-policy");
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, "add", poot_codegen::Target::SpirvVulkan);
    let artifact =
        poot_codegen::compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out).unwrap();
    let kernel = device
        .load_kernel(
            "add",
            poot_codegen::kernel_handle(&body, &artifact).unwrap(),
        )
        .unwrap();
    let buffer = |device: &mut WgpuDevice, values: [f32; 4]| {
        let buffer = device
            .allocate(BufferRole::Activation, BufferStorage::f32(), 4)
            .unwrap();
        poot_executor::write_slice(device, &buffer, 0, bytemuck::cast_slice(&values)).unwrap();
        buffer
    };
    let one = buffer(device, [1.0; 4]);
    let ping = buffer(device, [0.0; 4]);
    let pong = buffer(device, [0.0; 4]);
    device.begin().unwrap();
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
                launch: poot_executor::LaunchKind::Grid {
                    threads: [4, 1, 1],
                    workgroup: [64, 1, 1],
                },
                work,
            })
            .unwrap();
    }
    let recording = device.finish().unwrap();
    let counters = device.context().counters();
    let before = counters.snapshot();
    device.replay(&recording).unwrap();
    let during = counters.snapshot();
    device.synchronize().unwrap();
    let submits = during.submits - before.submits;
    let waits = during.waits - before.waits;
    let last = if dispatches.is_multiple_of(2) {
        &ping
    } else {
        &pong
    };
    let mut bytes = [0u8; 16];
    device.read(last, &mut bytes).unwrap();
    Replay {
        submits,
        waits,
        value: f32::from_le_bytes(bytes[..4].try_into().unwrap()),
    }
}

/// A device with a watchdog budget cuts 1030 dispatches at the 1024-dispatch bound (two submits), and a heavy
/// dispatch cuts on the work bound (five 3M-unit dispatches: five submits); the chained value is exact either
/// way. With `watchdog_budget: None` the same recordings submit once.
///
/// Mutation: restore the 1,024 constant as an unconditional bound (`FlushPolicy::of` returns the watchdog policy
/// whatever the caps say): the no-budget rows submit twice and five times.
#[test]
fn a_stage_is_one_submit_without_a_watchdog_budget_and_cut_with_one() {
    let Some(mut device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let real = device.context().device_caps();
    let Replay { submits, value, .. } = replay_chain(&mut device, 1030, 4);
    let budgeted = real.watchdog_budget.is_some();
    eprintln!("real device: watchdog_budget {budgeted}, 1030 dispatches -> {submits} submits");
    assert_eq!(value, 1030.0);
    assert_eq!(
        submits,
        if budgeted { 2 } else { 1 },
        "the device's own policy"
    );

    let mut device = device.with_flush_caps(&DeviceCaps {
        watchdog_budget: None,
        ..real
    });
    let Replay { submits, value, .. } = replay_chain(&mut device, 1030, 4);
    assert_eq!((submits, value), (1, 1030.0), "no budget: one submit");
    let Replay { submits, value, .. } = replay_chain(&mut device, 5, 3_000_000);
    assert_eq!(
        (submits, value),
        (1, 5.0),
        "no budget: heavy dispatches still one submit"
    );

    let budget = poot_target::DeviceCaps::wgpu_rdna3_igpu().watchdog_budget;
    assert!(budget.is_some());
    let mut device = device.with_flush_caps(&DeviceCaps {
        watchdog_budget: budget,
        ..real
    });
    let Replay { submits, value, .. } = replay_chain(&mut device, 1030, 4);
    assert_eq!(
        (submits, value),
        (2, 1030.0),
        "budget: cut at 1024 dispatches"
    );
    let Replay { submits, value, .. } = replay_chain(&mut device, 5, 3_000_000);
    assert_eq!(
        (submits, value),
        (5, 5.0),
        "budget: cut on the 4M work bound"
    );
}

/// POOT-1435: a replay cut into several submits keeps two in flight and waits only for the oldest, so the
/// device always has the next submit queued while it runs the previous one. Five heavy dispatches are five
/// submits; the third needs the first retired, the fourth the second, the fifth the third: three waits, and
/// none for the last two, which the step's own synchronize collects. A wait after every submit (the queue
/// drained between runs) is four, and leaves the device idle for each submit's host turnaround.
///
/// Mutation: `IN_FLIGHT_SUBMITS` of 1 in `poot-wgpu-device`: four waits.
#[test]
fn a_cut_replay_keeps_two_submits_in_flight() {
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let real = device.context().device_caps();
    let budget = poot_target::DeviceCaps::wgpu_rdna3_igpu().watchdog_budget;
    assert!(budget.is_some());
    let mut device = device.with_flush_caps(&DeviceCaps {
        watchdog_budget: budget,
        ..real
    });
    let replay = replay_chain(&mut device, 5, 3_000_000);
    assert_eq!((replay.submits, replay.value), (5, 5.0));
    assert_eq!(replay.waits, 3, "the host waits only for the oldest submit");
    let replay = replay_chain(&mut device, 2, 3_000_000);
    assert_eq!(
        (replay.submits, replay.waits),
        (2, 0),
        "a window not yet full never waits"
    );
}
