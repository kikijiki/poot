//! Test-only [`poot_target::DeviceCaps`] fixtures (card 522): every production planner entry
//! point now requires a real caller-supplied `&DeviceCaps` (a runtime's `Context::device_caps()`), so a
//! test outside `poot-graph-plan` (which has its own `#[cfg(test)]`-gated in-crate twin of this same
//! table) reaches this cross-crate-safe dev-dependency instead. `poot-test-util` is `publish = false`
//! and used only as a `dev-dependency`, so Cargo structurally prevents this from ever reaching a
//! production build.

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
            ..DeviceCaps::rocm_default()
        },
        Backend::Nvptx => DeviceCaps::ptx_default(),
    }
}
