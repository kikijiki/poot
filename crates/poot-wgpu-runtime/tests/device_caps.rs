//! SC-004 (card 522): `Context::device_caps` compares field-for-field against a direct wgpu adapter
//! query taken independently of `Context::new`, not by re-reading the same cached struct fields. A missing
//! device is a `POOT-SKIP`, and a failure when `POOT_REQUIRE_WGPU` requires it.

use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::{open_or_skip, skip_unsupported};
use poot_wgpu_runtime::Context;

/// The same adapter selection `Context::new_async` makes (same power preference), queried directly
/// instead of through a `Context`, so this is an independent confirmation of what the device reports.
async fn raw_adapter() -> Option<wgpu::Adapter> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok()
}

#[test]
fn wgpu_device_caps_matches_a_direct_adapter_query() {
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let Some(adapter) = pollster::block_on(raw_adapter()) else {
        skip_unsupported(DeviceBackend::Wgpu, "a raw adapter query");
        return;
    };
    let info = adapter.get_info();
    let limits = adapter.limits();
    let caps = ctx.device_caps();

    // The device is opened with the adapter's own limits, so the workgroup memory and the per-axis workgroup
    // extents it grants are the adapter's, as raw Vulkan's are (POOT-1232).
    assert_eq!(
        caps.lds_bytes, limits.max_compute_workgroup_storage_size,
        "lds_bytes must be the adapter's own workgroup-storage limit, not the downlevel default"
    );
    assert_eq!(
        caps.max_workgroup_size,
        [
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z
        ],
        "workgroup extents are the adapter's own"
    );

    // This crate raises `max_compute_workgroups_per_dimension` to the adapter's own maximum, so the
    // granted device value must equal this direct query.
    assert_eq!(
        caps.max_grid, [limits.max_compute_workgroups_per_dimension; 3],
        "max_grid must equal the adapter's own max_compute_workgroups_per_dimension"
    );

    // `effective_max_buffer_size` is the raw device max held to the calibrated safe ceiling of the device's
    // identity, where one was measured (Card 453 D1; RADV on Strix Halo), else the raw device max; reproduce that
    // from an independently queried adapter to prove `device_caps` did not just echo a constant.
    let calibrated_ceiling = (info.backend == wgpu::Backend::Vulkan)
        .then(|| poot_target::calibration::Calibration::of_vulkan_device(info.vendor, &info.name))
        .and_then(poot_target::Queried::known)
        .and_then(|c| c.safe_max_buffer_bytes);
    let expected_max_buffer_bytes = calibrated_ceiling.map_or(limits.max_buffer_size, |safe| {
        limits.max_buffer_size.min(safe)
    });
    assert_eq!(
        caps.max_buffer_bytes, expected_max_buffer_bytes,
        "max_buffer_bytes must be the queried device max, RADV-capped where applicable"
    );

    // The performance figures follow the device's identity (Strix Halo measured, any other Vulkan device and every
    // non-Vulkan adapter on the portable prior), re-derived from the independently queried adapter.
    let expected_calibration = if info.backend == wgpu::Backend::Vulkan {
        poot_target::calibration::Calibrated::of_vulkan_device(info.vendor, &info.name)
    } else {
        poot_target::calibration::Calibrated::Portable
    };
    assert_eq!(caps.perf_calibration, expected_calibration);

    // The matrix units are measured from `VK_KHR_cooperative_matrix` (via wgpu's own adapter-level
    // query), never parsed from the adapter's device-name string (card 522 review). Re-derive
    // independently from a fresh adapter query so hardcoded units would disagree.
    // `Context::new_async` requests the cooperative-matrix feature exactly when the adapter advertises
    // it, so that feature is granted iff advertised; without it no unit is reported.
    let coopmat_granted = adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    let reported_f16_16x16x16 = adapter
        .cooperative_matrix_properties()
        .into_iter()
        .any(|c| {
            (c.m_size, c.n_size, c.k_size) == (16, 16, 16)
                && c.ab_type == wgpu::CooperativeScalarType::F16
                && c.cr_type == wgpu::CooperativeScalarType::F32
        });
    let expected_dtypes = if info.backend != wgpu::Backend::Vulkan {
        None
    } else if coopmat_granted && reported_f16_16x16x16 {
        Some(vec![poot_target::FragmentDtype::F16])
    } else {
        Some(vec![])
    };
    assert_eq!(
        caps.matrix_units
            .known()
            .map(|units| units.iter().map(|c| c.dtype).collect::<Vec<_>>()),
        expected_dtypes,
        "matrix_units must be measured from a fresh VK_KHR_cooperative_matrix query, not hardcoded"
    );
    // On this box (AMD Vulkan/RADV, RDNA3.5 WMMA hardware) the units are exactly one opaque f16 16x16x16
    // fragment across 32 lanes: RADV exposes no bf16 cooperative matrix (Mesa disables it on gfx11), and
    // reporting one would send bf16 operands to a body that cannot lower them.
    let is_amd_radv = info.backend == wgpu::Backend::Vulkan && info.vendor == 0x1002;
    if is_amd_radv {
        let units = caps
            .matrix_units
            .known()
            .expect("a Vulkan adapter reports its matrix units");
        let configs: Vec<_> = units.iter().collect();
        assert_eq!(
            configs.len(),
            1,
            "RADV offers one fragment config: {configs:?}"
        );
        let config = configs[0];
        assert_eq!(config.dtype, poot_target::FragmentDtype::F16);
        assert_eq!(config.shape, poot_target::FragmentShape::M16N16K16);
        assert_eq!(config.layout, poot_target::FragmentLayout::Opaque);
        assert_eq!(config.subgroup, 32);
    }

    // The watchdog budget and the tiled-GEMM miscompile ceiling come from the calibration table, keyed by the
    // device's identity, since no driver API reports either (they were calibrated from observed hangs and wrong
    // output): present exactly where the table names this device.
    let calibrated = (info.backend == wgpu::Backend::Vulkan)
        .then(|| poot_target::calibration::Calibration::of_vulkan_device(info.vendor, &info.name))
        .and_then(poot_target::Queried::known);
    assert_eq!(caps.watchdog_budget.is_some(), calibrated.is_some());
    assert_eq!(
        caps.known_miscompiles.tiled_gemm_max_workgroups.is_some(),
        calibrated.is_some()
    );
}
