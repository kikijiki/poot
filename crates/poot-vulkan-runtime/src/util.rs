use crate::*;

/// The launch grid of `threads` / `wg` per axis, validated against the device's per-axis
/// `maxComputeWorkGroupCount`, or [`RuntimeError::GridCap`] with the requested grid and the X cap.
///
/// A grid over the limit is refused, never folded onto y: a kernel the planner emitted as a 1-D launch
/// reads only x, so a folded launch computes wrong results silently. The planner sizes grids from
/// `DeviceCaps::max_grid`, which this crate measures from the same limit, so a refusal here means the
/// caps and the device disagree.
pub(crate) fn resolve_groups(
    threads: [u32; 3],
    wg: [u32; 3],
    max_workgroup_count: [u32; 3],
) -> Result<[u32; 3], RuntimeError> {
    let groups = [
        threads[0].div_ceil(wg[0].max(1)),
        threads[1].div_ceil(wg[1].max(1)),
        threads[2].div_ceil(wg[2].max(1)),
    ];
    if groups
        .iter()
        .zip(max_workgroup_count)
        .any(|(&g, limit)| g == 0 || g > limit)
    {
        return Err(RuntimeError::GridCap(groups, max_workgroup_count[0]));
    }
    Ok(groups)
}

/// The X-thread extent of a launch: `groups[0] * wg[0]`. Shared codegen reconstructs
/// the linear index as `group_y * x_extent + global_id.x` from this value, which the runtime appends
/// to the length buffer at slot `param_count`.
pub(crate) fn x_extent(groups: [u32; 3], wg: [u32; 3]) -> u32 {
    groups[0].saturating_mul(wg[0].max(1))
}

/// Find a memory type index whose bit is set in `type_bits` (from `VkMemoryRequirements::memoryTypeBits`,
/// or `u32::MAX` for any type) and whose `propertyFlags` contain `required`. Prefers one that is also
/// `DEVICE_LOCAL` (the UMA/iGPU path, see the module doc), else any match.
pub(crate) fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mut fallback: Option<u32> = None;
    for i in 0..mem_props.memory_type_count {
        if type_bits & (1 << i) == 0 {
            continue;
        }
        let ty = mem_props.memory_types[i as usize];
        if !ty.property_flags.contains(required) {
            continue;
        }
        if ty
            .property_flags
            .contains(required | vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return Some(i);
        }
        if fallback.is_none() {
            fallback = Some(i);
        }
    }
    fallback
}

/// `vk::PhysicalDeviceProperties::device_name` is `[c_char; 256]` with `c_char == i8` on Linux; reinterpret as `u8` for UTF-8 decoding.
pub(crate) fn bytemuck_cast_i8_to_u8(s: &[std::ffi::c_char]) -> &[u8] {
    // SAFETY: `c_char` and `u8` have identical size/alignment on every platform this crate targets
    // (Linux: `c_char == i8`); this is a same-size, same-alignment reinterpret of a plain data array with
    // no padding.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) }
}
