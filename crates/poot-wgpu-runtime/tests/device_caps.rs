//! SC-004 (card 522): `Context::device_caps` compares field-for-field against a direct wgpu adapter
//! query taken independently of `Context::new`, not by re-reading the same cached struct fields. Skips
//! (passes) with no Vulkan adapter, like every other test in this crate.

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
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let Some(adapter) = pollster::block_on(raw_adapter()) else {
        eprintln!("no GPU (raw adapter query failed); skipping");
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

    // The tensor-core family is measured from `VK_KHR_cooperative_matrix` (via wgpu's own adapter-level
    // query), never parsed from the adapter's device-name string (card 522 review). Re-derive
    // independently from a fresh adapter query so a hardcoded family would disagree.
    // `Context::new_async` requests the cooperative-matrix feature exactly when the adapter advertises
    // it, so that feature is granted iff advertised; without it no family is reported.
    let coopmat_granted = adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    let expected_tensor_core = if info.backend != wgpu::Backend::Vulkan {
        poot_target::TensorCoreSupport::UnknownNotExposedByApi
    } else if !coopmat_granted {
        poot_target::TensorCoreSupport::None
    } else {
        adapter
            .cooperative_matrix_properties()
            .into_iter()
            .map(|c| {
                poot_target::TensorCoreSupport::from_cooperative_matrix_config(
                    c.m_size,
                    c.n_size,
                    c.k_size,
                    c.ab_type == wgpu::CooperativeScalarType::F16,
                    c.cr_type == wgpu::CooperativeScalarType::F32,
                )
            })
            .fold(poot_target::TensorCoreSupport::None, |acc, x| {
                acc.most_specific(x)
            })
    };
    assert_eq!(
        caps.tensor_core, expected_tensor_core,
        "tensor_core must be measured from a fresh VK_KHR_cooperative_matrix query, not a hardcoded family"
    );
    // On this box (AMD Vulkan/RADV, RDNA3.5 WMMA hardware) this must be a confirmed family, not `None`
    // or `UnknownNotExposedByApi` - the whole point of measuring instead of parsing a name string.
    let is_amd_radv = info.backend == wgpu::Backend::Vulkan && info.vendor == 0x1002;
    if is_amd_radv {
        assert_eq!(
            caps.tensor_core,
            poot_target::TensorCoreSupport::Wmma16x16x16Rdna3,
            "this box's RDNA3.5 iGPU must report confirmed WMMA-class support, not {:?}",
            caps.tensor_core
        );
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
