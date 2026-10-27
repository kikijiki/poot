//! Raw Vulkan against wgpu: the one executor-contract row that opens both devices. It lives in poot-llm's
//! tests because poot-vulkan-device and poot-wgpu-device are same-rank siblings and neither may depend on the other,
//! even for tests. A row skips when a device does not open (`POOT_REQUIRE_VULKAN=1` and `POOT_REQUIRE_WGPU=1`
//! turn that into a failure); rows run serially under `/tmp/poot-gpu.lock`.

use std::sync::Arc;

use poot_executor::{Device, Engine, Executor};
use poot_runtime_common::device_skip::open_or_skip;
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_vulkan_device::VulkanDevice;

fn vulkan() -> Option<VulkanDevice> {
    open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new())
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
    let Some(wgpu) = open_or_skip(
        DeviceBackend::Wgpu,
        poot_wgpu_device::device::WgpuDevice::new(),
    ) else {
        return;
    };
    let fixture = poot_models::test_support::fixtures::qwen2_decode_fixture(8);
    let load = |target, device: &mut dyn Executor| {
        let exe = device
            .load_weights(
                Arc::new(fixture.store.clone()),
                poot_executor_parity::fixture_weights(&fixture),
            )
            .unwrap();
        device
            .add_entry(exe, &poot_executor_parity::staged(&fixture, target))
            .unwrap();
        device.stats().memory
    };
    let (vulkan_target, wgpu_target) = (device.target(), wgpu.target());
    let (mut vulkan_engine, mut wgpu_engine) = (Engine::new(device), Engine::new(wgpu));
    let on_vulkan = load(vulkan_target, &mut vulkan_engine);
    let on_wgpu = load(wgpu_target, &mut wgpu_engine);
    let live = |memory: &[(BufferRole, poot_runtime_common::MemoryCounterSnapshot)], role| {
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

/// A decode attention of `hq` heads over a `cap`-key cache of head width `d`, with K and V as activation
/// inputs so no constant size gate applies.
fn decode_attention(hq: usize, cap: usize, d: usize) -> poot_graph_ir::Graph {
    use poot_graph_ir::{Builder, Slot, TensorType};
    let b = Builder::for_tests();
    let q = b
        .slot_named(Slot::Activation, "q", TensorType::f32(vec![1, hq, 1, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    let k = b
        .slot_named(Slot::Activation, "k", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let v = b
        .slot_named(Slot::Activation, "v", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let out = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.125, mask).unwrap();
    b.finish(out).unwrap()
}

/// POOT-1232 SC-003: wgpu and raw Vulkan on the same GPU plan the same attention. Each device's own measured
/// target compiles the same decode attention; the workgroup memory, the workgroup limits and the lane count of the
/// planned schedule agree, at a value width where workgroup memory decides the lane count (256: 4 * lanes * 258
/// bytes must fit). Mutation: open the wgpu device with `wgpu::Limits::downlevel_defaults()` workgroup storage
/// (16 KiB) as it did before: wgpu plans 8 lanes where raw Vulkan plans 32, and the lane assertion goes red.
#[test]
fn the_same_gpu_plans_the_same_attention_lanes_on_wgpu_and_raw_vulkan() {
    use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, compile};
    use poot_kernelgen::KernelRequest;
    let Some(vulkan) = vulkan() else {
        return;
    };
    let Some(wgpu) = open_or_skip(
        DeviceBackend::Wgpu,
        poot_wgpu_device::device::WgpuDevice::new(),
    ) else {
        return;
    };
    let (vulkan_target, wgpu_target) = (vulkan.target(), wgpu.target());
    let caps = |target: &poot_graph_plan::Target| match target {
        poot_graph_plan::Target::Gpu { caps, .. } => *caps,
        other => panic!("a GPU target: {other:?}"),
    };
    let (on_vulkan, on_wgpu) = (caps(&vulkan_target), caps(&wgpu_target));
    eprintln!("vulkan caps {on_vulkan:?}\nwgpu caps {on_wgpu:?}");
    let wgpu_adapter = wgpu.identity().map(|identity| identity.adapter);
    if wgpu_adapter.as_deref() != Some(vulkan.context().device_name().as_str()) {
        eprintln!(
            "POOT-SKIP: the two runtimes did not open one GPU (vulkan {:?}, wgpu {wgpu_adapter:?})",
            vulkan.context().device_name()
        );
        return;
    }
    assert_eq!(on_vulkan.lds_bytes, on_wgpu.lds_bytes, "workgroup memory");
    assert_eq!(on_vulkan.max_workgroup_size, on_wgpu.max_workgroup_size);
    assert_eq!(
        on_vulkan.max_workgroup_invocations,
        on_wgpu.max_workgroup_invocations
    );
    let options = CompileOptions::new(CompileLimits::STANDARD);
    for d in [64, 256] {
        let graph = decode_attention(2, 1024, d);
        let lanes = |target: &poot_graph_plan::Target| {
            let program = compile(&graph, target, &options).expect("the decode attention plans");
            let (eqn, _) = program
                .planned()
                .find(|(eqn, _)| matches!(eqn.op, poot_graph_ir::OpKind::Attention(_)))
                .expect("an attention equation");
            let KernelChoice::Generated(KernelRequest::Attention(spec)) =
                program.kernel_choice(eqn)
            else {
                panic!("generated attention expected");
            };
            spec.schedule.lanes()
        };
        assert_eq!(
            lanes(&vulkan_target),
            lanes(&wgpu_target),
            "attention lanes at value width {d}: raw Vulkan and wgpu on one GPU"
        );
    }
}
