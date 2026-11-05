//! Test-only [`poot_target::DeviceCaps`] fixtures (card 522): every production planner entry
//! point now requires a real caller-supplied `&DeviceCaps` (a runtime's `Context::device_caps()`), so a
//! test outside `poot-graph-plan` (which has its own `#[cfg(test)]`-gated in-crate twin of this same
//! table) reaches this cross-crate-safe dev-dependency instead. `poot-test-util` is `publish = false`
//! and used only as a `dev-dependency`, so Cargo structurally prevents this from ever reaching a
//! production build.

use poot_target::calibration::{Calibration, LaunchClass};
use poot_target::dram_channels::DramChannels;
use poot_target::{
    AMD_RADV_SAFE_MAX_BUFFER_BYTES, Backend, CooperativeMatrixConfig, DeviceCaps, E4m3Storage,
    FragmentConfig, FragmentDtype, FragmentLayout, FragmentShape, MatrixUnits, Queried,
    RegisterFile, RoundingConversions, STRIX_HALO_COMPUTE_UNITS, ShaderFeatures, SubgroupSupport,
};

/// The [`DeviceCaps`] a test should use for a given [`Backend`] when it has no real measured device to
/// thread through: the values this box's own wgpu/RADV device measured (card 522), and the documented
/// ROCm/PTX fallbacks for a device with no confirmed watchdog budget or miscompile ceiling.
pub fn default_caps_for(backend: Backend) -> DeviceCaps {
    match backend {
        Backend::SpirvVulkan => wgpu_rdna3_igpu(),
        // A live ROCm context reads both from its agent: the arch carries the same values.
        Backend::AmdGcn(arch) => DeviceCaps {
            matrix_units: Queried::Known(arch.matrix_units()),
            register_file: arch.register_file(),
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

/// The values measured on this box's wgpu/RADV device (Strix Halo, gfx1151) as of card 522: the
/// display-watchdog budgets cards 163/258/159 calibrated, and the card-095 tiled-GEMM RADV
/// miscompile ceiling, with the device's 40 compute units. A planner fixture (card 599's plan-summary
/// corpus among them): a runtime reports its own device's caps and plans an unmeasured one with unknown
/// compute units (POOT-1468), so no production code reads this.
pub fn wgpu_rdna3_igpu() -> DeviceCaps {
    let calibration = Calibration::STRIX_HALO_RADV;
    DeviceCaps {
        max_buffer_bytes: AMD_RADV_SAFE_MAX_BUFFER_BYTES,
        // Both Vulkan runtimes open the device with the adapter's own limits: RADV offers 64 KiB of
        // workgroup memory and 1024 invocations per workgroup axis.
        lds_bytes: 65_536,
        max_workgroup_size: [1024, 1024, 1024],
        max_workgroup_invocations: 1024,
        // RDNA3 runs 32- or 64-lane waves; wgpu reports the adapter's range.
        subgroup: Queried::Known(SubgroupSupport::Present {
            min_size: 32,
            max_size: 64,
        }),
        max_grid: [65_535, 65_535, 65_535],
        watchdog_budget: calibration.watchdog_budget,
        // wgpu leaves the pipeline's subgroup size to the driver, so only a one-subgroup workgroup is a
        // fragment collective.
        matrix_units: Queried::Known(MatrixUnits::of([FragmentConfig::m16n16k16(
            FragmentDtype::F16,
            FragmentLayout::Opaque,
            32,
            1,
        )])),
        // RADV's `VK_AMD_shader_core_properties` on Strix Halo: 256 VGPRs a lane, 768 wave64 VGPRs per SIMD,
        // two SIMDs a compute unit.
        register_file: Queried::Known(RegisterFile::amd_shader_core(256, 768, 2)),
        known_miscompiles: calibration.known_miscompiles,
        compute_units: Queried::Known(STRIX_HALO_COMPUTE_UNITS),
        memory_budget_bytes: None,
        dram_channels: Some(DramChannels::STRIX_HALO),
        shader_features: ShaderFeatures::ALL,
        e4m3_storage: E4m3Storage::PackedLanes,
        native_two_byte_floats: false,
        rounding_conversions: RoundingConversions::NONE,
        widens_bf16_decode_weight: false,
        peer_transport: false,
        decode_gemv_max_out_cols: None,
        launch_class: LaunchClass::Vulkan,
    }
}

/// The values `poot-vulkan-runtime`'s raw Vulkan context measured on this box's RADV device (Strix Halo,
/// gfx1151) through `VkPhysicalDeviceLimits` and `VK_KHR_cooperative_matrix`: the reference caps of the raw
/// Vulkan planner column. It is the same silicon [`wgpu_rdna3_igpu`] describes and differs only
/// in the dispatch grid (wgpu reports one workgroup-count limit for every axis, Vulkan reports each axis, so
/// x is a 32-bit grid) and in the shader features it requests. `memory_budget_bytes` is `None`, as in every fixture. The live receipt
/// `poot-vulkan-runtime/tests/device_caps.rs` fails when the device stops agreeing with this fixture.
pub fn vulkan_rdna3_igpu() -> DeviceCaps {
    DeviceCaps {
        max_grid: [u32::MAX, 65_535, 65_535],
        // The raw Vulkan device does not request the float32-atomics feature wgpu does.
        shader_features: ShaderFeatures {
            atomic_float32: false,
            ..ShaderFeatures::ALL
        },
        // Raw Vulkan requires 32-lane subgroups of its cooperative-matrix pipelines, so a workgroup runs
        // several fragment collectives.
        matrix_units: Queried::Known(MatrixUnits::cooperative_matrix(
            [CooperativeMatrixConfig {
                shape: FragmentShape::M16N16K16,
                ab_f16: true,
                result_f32: true,
            }],
            32,
            FragmentConfig::ANY_WAVES,
        )),
        ..wgpu_rdna3_igpu()
    }
}
