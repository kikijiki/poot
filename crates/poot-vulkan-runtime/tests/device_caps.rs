//! Card 522: `Context::device_caps` fills a plausible descriptor on the real
//! device, and `matrix_units` is a real `VK_KHR_cooperative_matrix` measurement, not a device-name guess -
//! confirmed by this box's real RDNA3.5 WMMA hardware reporting a matching config. A row without a Vulkan
//! device (or, for the RADV rows, without a RADV one) skips with `POOT-SKIP`, and fails when
//! `POOT_REQUIRE_VULKAN` requires the device.

use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::{open_or_skip, skip_unsupported};
use poot_vulkan_runtime::Context;

#[test]
fn device_caps_is_plausible_on_the_real_device() {
    let Some(ctx) = open_or_skip(DeviceBackend::Vulkan, Context::new()) else {
        return;
    };
    let caps = ctx.device_caps();
    eprintln!("raw-vulkan device_caps: {caps:?}");
    assert!(caps.max_buffer_bytes > 0);
    assert!(caps.lds_bytes > 0);
    assert!(caps.max_grid.iter().all(|&d| d > 0));
    // The calibration is keyed by device identity: a RADV device the table does not name reports none.
    let calibrated = poot_target::calibration::Calibration::of_vulkan_device(
        poot_target::calibration::VENDOR_AMD,
        &ctx.device_name(),
    );
    if ctx.device_name().to_ascii_lowercase().contains("radv") {
        assert_eq!(caps.watchdog_budget.is_some(), calibrated.known().is_some());
        assert_eq!(
            caps.known_miscompiles.tiled_gemm_max_workgroups.is_some(),
            calibrated.known().is_some()
        );
    }
}

/// POOT-1431: this box's iGPU is RDNA3.5 (gfx1151) behind RADV. `Context::device_caps` must report exactly one
/// matrix unit, measured from `VK_KHR_cooperative_matrix`: an opaque f16 16x16x16 fragment across 32 lanes. RADV
/// exposes no bf16 cooperative matrix (Mesa disables `VK_KHR_shader_bfloat16` on gfx11), so a bf16 config would
/// route bf16 operands to a body that cannot lower them.
/// Mutation: report a bf16 config (`MatrixUnits::of` with a `Bf16` `FragmentConfig`) from
/// `MatrixUnits::cooperative_matrix`; the dtype assertion goes red.
#[test]
fn raw_vulkan_reports_exactly_an_opaque_f16_fragment_on_this_box() {
    let Some(ctx) = open_or_skip(DeviceBackend::Vulkan, Context::new()) else {
        return;
    };
    if !ctx.device_name().to_ascii_lowercase().contains("radv") {
        skip_unsupported(
            DeviceBackend::Vulkan,
            "the expected RADV iGPU (WMMA assertion)",
        );
        return;
    }
    let units = ctx
        .device_caps()
        .matrix_units
        .known()
        .expect("a Vulkan device reports its matrix units");
    let configs: Vec<_> = units.iter().collect();
    assert_eq!(
        configs.iter().map(|c| c.dtype).collect::<Vec<_>>(),
        [poot_target::FragmentDtype::F16],
        "RADV offers f16 cooperative matrices only: {configs:?}"
    );
    let config = configs[0];
    assert_eq!(config.shape, poot_target::FragmentShape::M16N16K16);
    assert_eq!(config.layout, poot_target::FragmentLayout::Opaque);
    assert_eq!(config.subgroup, 32);
}

/// POOT-1231 SC-001: the reference caps the raw Vulkan planner column compiles against are this device's
/// own: every figure the fixture fixes (workgroup memory, workgroup and grid limits, subgroup range, matrix
/// hardware, compute units, the vendor-scoped RADV caps) is what the live context reports. A
/// non-RADV device is a skip (the fixture describes the RADV iGPU). Mutation: change a figure of
/// `poot_test_util::device_caps::vulkan_rdna3_igpu` (or point it back at the wgpu preset's `lds_bytes`): the field named in the
/// assertion differs from the live value.
#[test]
fn the_reference_caps_of_the_raw_vulkan_column_are_what_the_radv_device_reports() {
    let Some(ctx) = open_or_skip(DeviceBackend::Vulkan, Context::new()) else {
        return;
    };
    if !ctx.device_name().to_ascii_lowercase().contains("radv") {
        skip_unsupported(
            DeviceBackend::Vulkan,
            "a RADV device (the reference caps describe one)",
        );
        return;
    }
    let live = ctx.device_caps();
    let reference = poot_test_util::device_caps::vulkan_rdna3_igpu();
    // The live budget is an observation at open, not a property of the part.
    assert_eq!(
        poot_target::DeviceCaps {
            memory_budget_bytes: None,
            ..live
        },
        reference,
        "the raw Vulkan reference caps differ from the live device"
    );
}
