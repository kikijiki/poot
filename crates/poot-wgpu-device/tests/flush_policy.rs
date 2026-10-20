//! POOT-1232 SC-004: how many submits a wgpu stage costs follows the device's `watchdog_budget`, not a
//! constant. A device with a budget cuts a long recording into several submits; a device without one submits
//! the stage once. Driven through the production `begin`/`dispatch`/`finish`/`replay`, counting the context's
//! own compute submits. Rows run serially under `/tmp/poot-gpu.lock`.

use poot_executor::{Arg, Device, Dispatch};
use poot_runtime_common::device_skip::open_or_skip;
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::{BufferStorage, DeviceCaps};
use poot_wgpu_device::device::WgpuDevice;

/// Record `dispatches` chained adds of `work` units each into one recording and replay it, returning the compute
/// submits the replay cost and the chained value (each add contributes 1).
fn replay_chain(device: &mut WgpuDevice, dispatches: usize, work: u64) -> (usize, f32) {
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
    let before = device.context().submits();
    device.replay(&recording).unwrap();
    device.synchronize().unwrap();
    let submits = device.context().submits() - before;
    let last = if dispatches.is_multiple_of(2) {
        &ping
    } else {
        &pong
    };
    let mut bytes = [0u8; 16];
    device.read(last, &mut bytes).unwrap();
    (submits, f32::from_le_bytes(bytes[..4].try_into().unwrap()))
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
    let (submits, value) = replay_chain(&mut device, 1030, 4);
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
    let (submits, value) = replay_chain(&mut device, 1030, 4);
    assert_eq!((submits, value), (1, 1030.0), "no budget: one submit");
    let (submits, value) = replay_chain(&mut device, 5, 3_000_000);
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
    let (submits, value) = replay_chain(&mut device, 1030, 4);
    assert_eq!(
        (submits, value),
        (2, 1030.0),
        "budget: cut at 1024 dispatches"
    );
    let (submits, value) = replay_chain(&mut device, 5, 3_000_000);
    assert_eq!(
        (submits, value),
        (5, 5.0),
        "budget: cut on the 4M work bound"
    );
}
