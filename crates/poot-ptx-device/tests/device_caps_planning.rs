//! Card 662, PTX half of SC-003: the capabilities a live NVIDIA device reports are the ones the planner's
//! finalizer reads. The caps come from
//! `PtxDevice::target()` (the `cuDeviceGetAttribute` queries), not from a fixture; a device without a
//! CUDA driver skips (and fails when `POOT_REQUIRE_PTX` is set).

use poot_executor::Device;
use poot_graph_plan::test_support::{gpu_backend, gpu_caps};
use poot_ptx_device::device::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_target::{Backend, FragmentDtype, FragmentLayout, Queried, SubgroupSupport};

/// A live device reports positive workgroup limits, a fixed 32-lane warp, and the matrix units of its compute
/// capability (never `Unknown`): on an sm_80+ part opaque bf16 and f16 fragments across 32 lanes.
///
/// Mutation: report `Queried::Unknown` from `caps_from_attributes`; the matrix-unit assertion goes red.
/// Owed on a pod (POOT-1431): this box has no NVIDIA device.
#[test]
fn a_live_ptx_device_reports_its_workgroup_subgroup_and_matrix_caps() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    assert_eq!(gpu_backend(&target), Backend::Nvptx);
    let caps = gpu_caps(&target);
    eprintln!("live ptx caps: {caps:?}");
    assert!(caps.max_workgroup_invocations >= 1);
    assert!(caps.max_workgroup_size.iter().all(|&extent| extent >= 1));
    assert_eq!(
        caps.subgroup,
        Queried::Known(SubgroupSupport::Present {
            min_size: 32,
            max_size: 32
        })
    );
    let units = caps
        .matrix_units
        .known()
        .expect("an NVIDIA device reports its matrix units from its compute capability");
    assert_eq!(
        units.iter().map(|c| c.dtype).collect::<Vec<_>>(),
        [FragmentDtype::Bf16, FragmentDtype::F16],
        "sm_80+ has bf16 and f16 wmma fragments"
    );
    assert!(
        units
            .iter()
            .all(|c| c.layout == FragmentLayout::Opaque && c.subgroup == 32),
        "the wmma API's fragment layout is opaque, held across a 32-lane warp: {units:?}"
    );
}

/// POOT-1065: the PTX device reports the driver's live free memory (`cuMemGetInfo`), within the total it
/// reported when it opened. Mutation: `free_memory_bytes` returns `None` and the first assertion goes
/// red; returns `u64::MAX` and the ceiling assertion goes red.
#[test]
fn the_ptx_device_reports_live_free_memory_within_its_open_time_ceiling() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let ceiling = gpu_caps(&device.target())
        .memory_budget_bytes
        .expect("the PTX device queried its total memory at open");
    let free = device
        .free_memory_bytes()
        .expect("the PTX device reports live free memory");
    assert!(
        free > 0,
        "a device with no free memory cannot run this test"
    );
    assert!(
        free <= ceiling,
        "free {free} is over the open-time ceiling {ceiling}"
    );
}
