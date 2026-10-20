//! Card 522: `Context::device_caps` fills a plausible descriptor on the real
//! device, and `tensor_core` is a real `VK_KHR_cooperative_matrix` measurement, not a device-name guess -
//! confirmed by this box's real RDNA3.5 WMMA hardware reporting a matching config. Skips (passes) if no
//! Vulkan device is available.

use poot_vulkan_runtime::Context;

#[test]
fn device_caps_is_plausible_on_the_real_device() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
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

/// This box's iGPU is RDNA3.5 (gfx1151), which has WMMA matrix hardware. `Context::device_caps` must
/// report a confirmed tensor-core family here, measured from `VK_KHR_cooperative_matrix` - not `None`
/// (which a name-string classifier would give, since Mesa RADV's device name carries a marketing
/// codename, "RADV STRIX_HALO", not a gfx code) and not `UnknownNotExposedByApi` (which would mean the
/// query itself found no coop-matrix extension on this device, also wrong on this box).
#[test]
fn raw_vulkan_reports_wmma_class_support_on_this_box() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };
    if !ctx.device_name().to_ascii_lowercase().contains("radv") {
        eprintln!(
            "not the expected RADV iGPU ({:?}); skipping the WMMA-specific assertion",
            ctx.device_name()
        );
        return;
    }
    assert_eq!(
        ctx.device_caps().tensor_core,
        poot_target::TensorCoreSupport::Wmma16x16x16Rdna3,
        "this box's gfx1151 iGPU must report confirmed WMMA-class cooperative-matrix support"
    );
}

/// POOT-1231 SC-001: the reference caps the raw Vulkan planner column compiles against are this device's
/// own: every figure the fixture fixes (workgroup memory, workgroup and grid limits, subgroup range, matrix
/// hardware, compute units, the vendor-scoped RADV caps) is what the live context reports. Skips on a
/// non-RADV device (the fixture describes the RADV iGPU). Mutation: change a figure of
/// `poot_test_util::device_caps::vulkan_rdna3_igpu` (or point it back at the wgpu preset's `lds_bytes`): the field named in the
/// assertion differs from the live value.
#[test]
fn the_reference_caps_of_the_raw_vulkan_column_are_what_the_radv_device_reports() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };
    if !ctx.device_name().to_ascii_lowercase().contains("radv") {
        eprintln!("not a RADV device ({:?}); skipping", ctx.device_name());
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
