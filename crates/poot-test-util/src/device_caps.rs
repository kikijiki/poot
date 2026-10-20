//! Test-only [`poot_target::DeviceCaps`] fixtures (card 522): every production planner entry
//! point now requires a real caller-supplied `&DeviceCaps` (a runtime's `Context::device_caps()`), so a
//! test outside `poot-graph-plan` (which has its own `#[cfg(test)]`-gated in-crate twin of this same
//! table) reaches this cross-crate-safe dev-dependency instead. `poot-test-util` is `publish = false`
//! and used only as a `dev-dependency`, so Cargo structurally prevents this from ever reaching a
//! production build.

use poot_target::dram_channels::DramChannels;
use poot_target::{Backend, DeviceCaps, Queried, SubgroupSupport};

/// The [`DeviceCaps`] a test should use for a given [`Backend`] when it has no real measured device to
/// thread through: the values this box's own wgpu/RADV device measured (card 522), and the documented
/// ROCm/PTX fallbacks for a device with no confirmed watchdog budget or miscompile ceiling.
pub fn default_caps_for(backend: Backend) -> DeviceCaps {
    match backend {
        Backend::SpirvVulkan => DeviceCaps::wgpu_rdna3_igpu(),
        // A live ROCm context reads both from its agent: the arch carries the same values.
        Backend::AmdGcn(arch) => DeviceCaps {
            tensor_core: arch.tensor_core,
            subgroup: Queried::Known(SubgroupSupport::Present {
                min_size: arch.wave,
                max_size: arch.wave,
            }),
            dram_channels: DramChannels::of_device_name(arch.mcpu()),
            ..DeviceCaps::rocm_default()
        },
        Backend::Nvptx => DeviceCaps::ptx_default(),
    }
}

/// The values `poot-vulkan-runtime`'s raw Vulkan context measured on this box's RADV device (Strix Halo,
/// gfx1151) through `VkPhysicalDeviceLimits` and `VK_KHR_cooperative_matrix`: the reference caps of the raw
/// Vulkan planner column. It lowers to the same SPIR-V as [`DeviceCaps::wgpu_rdna3_igpu`] but is not the same
/// device description: wgpu grants only its downlevel workgroup-storage and per-dimension limits, so the raw
/// device has four times the workgroup memory, larger y and z workgroups and a 32-bit x grid.
/// `memory_budget_bytes` is `None`, as in every fixture. The watchdog budgets, the RADV buffer cap and the
/// tiled-GEMM ceiling are the vendor-scoped values the runtime applies to any AMD Vulkan device. The live
/// receipt `poot-vulkan-runtime/tests/device_caps.rs` fails when the device stops agreeing with this fixture.
pub fn vulkan_rdna3_igpu() -> DeviceCaps {
    DeviceCaps {
        lds_bytes: 65_536,
        max_workgroup_size: [1024, 1024, 1024],
        max_grid: [u32::MAX, 65_535, 65_535],
        ..DeviceCaps::wgpu_rdna3_igpu()
    }
}
