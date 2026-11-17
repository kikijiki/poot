//! Vulkan property queries both SPIR-V runtimes make: `poot-wgpu-runtime` reads them through the wgpu-hal
//! adapter's raw handles and `poot-vulkan-runtime` through its own `ash` instance, so each query is written
//! once here, over the raw `ash` handles.

use ash::{amd, vk};
use poot_target::MemoryDomain;

/// The summed `VK_EXT_memory_budget` figures of a device's device-local heaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapBudget {
    /// What the OS lets this process use.
    pub budget: u64,
    /// What this process already holds.
    pub usage: u64,
}

impl HeapBudget {
    /// Device-local memory this process may still allocate: the budget less the usage.
    pub fn free(self) -> u64 {
        self.budget.saturating_sub(self.usage)
    }
}

/// The device-local heaps' budget and usage, in one `VK_EXT_memory_budget` read.
///
/// # Safety
///
/// `physical_device` must come from `instance`'s own enumeration, `instance` must be live, and the device must
/// implement `VK_EXT_memory_budget`.
pub unsafe fn device_local_heap_budget(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> HeapBudget {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: the caller's contract; `props2` and its chain outlive the call.
    unsafe { instance.get_physical_device_memory_properties2(physical_device, &mut props2) };
    let mem = props2.memory_properties;
    let device_local = (0..mem.memory_heap_count as usize).filter(|&i| {
        mem.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
    });
    device_local.fold(
        HeapBudget {
            budget: 0,
            usage: 0,
        },
        |sum, i| HeapBudget {
            budget: sum.budget + budget.heap_budget[i],
            usage: sum.usage + budget.heap_usage[i],
        },
    )
}

/// The memory domain of a Vulkan device (POOT-1273), read from its memory heaps and its device type.
///
/// A device is [`MemoryDomain::Host`] when it is an integrated (or CPU) part and a `DEVICE_LOCAL` heap has a
/// `HOST_VISIBLE` memory type: the heap the GPU allocates from is memory the host maps, because it is the
/// host's RAM. Every other device is [`MemoryDomain::Local`]: one whose `DEVICE_LOCAL` heaps carry no
/// host-visible type, and a discrete part even when it exposes its VRAM through a resizable BAR, since the
/// heap structure of a BAR-mapped VRAM and of an APU's shared heap is the same and only the device type
/// tells the host's own RAM from a window onto a separate memory.
pub fn memory_domain(
    device_type: vk::PhysicalDeviceType,
    memory: &vk::PhysicalDeviceMemoryProperties,
) -> MemoryDomain {
    let shares_host_ram = matches!(
        device_type,
        vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
    );
    let device_local_heap_is_host_visible = memory.memory_types
        [..memory.memory_type_count as usize]
        .iter()
        .any(|ty| {
            ty.property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && memory.memory_heaps[ty.heap_index as usize]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        });
    if shares_host_ram && device_local_heap_is_host_visible {
        MemoryDomain::Host
    } else {
        MemoryDomain::Local
    }
}

/// What AMD's shader-core properties say about a Vulkan device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmdShaderCore {
    /// Active compute units: `VK_AMD_shader_core_properties2`'s `activeComputeUnitCount`, else
    /// `VK_AMD_shader_core_properties`' engine x array x per-array product.
    pub compute_units: u32,
    /// `maxVgprAllocation`: the VGPRs one lane addresses.
    pub max_vgpr_allocation: u32,
    /// `vgprsPerSimd`: the SIMD's VGPRs, counted in wave64 registers.
    pub vgprs_per_simd: u32,
    /// `simdPerComputeUnit`.
    pub simd_per_compute_unit: u32,
}

/// A Vulkan device's AMD shader-core properties (`VK_AMD_shader_core_properties`, with
/// `VK_AMD_shader_core_properties2`'s active compute-unit count where the driver has it); `None` when the
/// driver exposes neither (every non-AMD device).
///
/// # Safety
///
/// `physical_device` must come from `instance`'s own enumeration, and `instance` must be live.
pub unsafe fn amd_vulkan_shader_core(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<AmdShaderCore> {
    // SAFETY: the caller's contract makes both handles valid; the enumeration and property queries
    // only read, and each property struct is chained only when its extension is listed as supported.
    unsafe {
        let extensions = instance
            .enumerate_device_extension_properties(physical_device)
            .ok()?;
        let supports = |name: &std::ffi::CStr| {
            extensions
                .iter()
                .any(|ext| ext.extension_name_as_c_str() == Ok(name))
        };
        if !supports(amd::shader_core_properties::NAME) {
            return None;
        }
        let mut core = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
        let mut core2 = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
        let has_core2 = supports(amd::shader_core_properties2::NAME);
        {
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            if has_core2 {
                props = props.push_next(&mut core2);
            }
            instance.get_physical_device_properties2(physical_device, &mut props);
        }
        let compute_units = if has_core2 {
            core2.active_compute_unit_count
        } else {
            core.shader_engine_count
                * core.shader_arrays_per_engine_count
                * core.compute_units_per_shader_array
        };
        Some(AmdShaderCore {
            compute_units,
            max_vgpr_allocation: core.max_vgpr_allocation,
            vgprs_per_simd: core.vgprs_per_simd,
            simd_per_compute_unit: core.simd_per_compute_unit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heaps and types of one device: `(heap flags, [type flags])` per heap, types numbered in order.
    fn memory(
        heaps: &[(vk::MemoryHeapFlags, &[vk::MemoryPropertyFlags])],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        for (heap, (flags, types)) in heaps.iter().enumerate() {
            props.memory_heaps[heap] = vk::MemoryHeap {
                size: 1 << 30,
                flags: *flags,
            };
            for ty in *types {
                props.memory_types[props.memory_type_count as usize] = vk::MemoryType {
                    property_flags: *ty,
                    heap_index: heap as u32,
                };
                props.memory_type_count += 1;
            }
        }
        props.memory_heap_count = heaps.len() as u32;
        props
    }

    const LOCAL: vk::MemoryHeapFlags = vk::MemoryHeapFlags::DEVICE_LOCAL;
    const NOT_LOCAL: vk::MemoryHeapFlags = vk::MemoryHeapFlags::empty();
    const DEVICE_LOCAL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const MAPPED: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    );
    const LOCAL_MAPPED: vk::MemoryPropertyFlags =
        vk::MemoryPropertyFlags::from_raw(DEVICE_LOCAL.as_raw() | MAPPED.as_raw());

    /// Strix Halo as vulkaninfo reports it (RADV, 2026-11-16): a host heap and one large `DEVICE_LOCAL` heap
    /// with host-visible types.
    fn unified_heaps() -> vk::PhysicalDeviceMemoryProperties {
        memory(&[
            (NOT_LOCAL, &[MAPPED]),
            (LOCAL, &[DEVICE_LOCAL, LOCAL_MAPPED]),
        ])
    }

    /// A discrete card without a resizable BAR: VRAM no host pointer reaches, plus a host heap.
    fn vram_heaps() -> vk::PhysicalDeviceMemoryProperties {
        memory(&[(LOCAL, &[DEVICE_LOCAL]), (NOT_LOCAL, &[MAPPED])])
    }

    #[test]
    fn an_integrated_device_with_a_mapped_device_local_heap_is_host() {
        assert_eq!(
            memory_domain(vk::PhysicalDeviceType::INTEGRATED_GPU, &unified_heaps()),
            MemoryDomain::Host
        );
    }

    #[test]
    fn a_device_whose_device_local_heaps_are_not_host_visible_is_local() {
        assert_eq!(
            memory_domain(vk::PhysicalDeviceType::DISCRETE_GPU, &vram_heaps()),
            MemoryDomain::Local
        );
        // The same heaps on a part that claims to be integrated still have no host-visible local type.
        assert_eq!(
            memory_domain(vk::PhysicalDeviceType::INTEGRATED_GPU, &vram_heaps()),
            MemoryDomain::Local
        );
    }

    #[test]
    fn a_discrete_device_with_a_resizable_bar_stays_local() {
        let rebar = memory(&[
            (LOCAL, &[DEVICE_LOCAL, LOCAL_MAPPED]),
            (NOT_LOCAL, &[MAPPED]),
        ]);
        assert_eq!(
            memory_domain(vk::PhysicalDeviceType::DISCRETE_GPU, &rebar),
            MemoryDomain::Local
        );
    }
}
