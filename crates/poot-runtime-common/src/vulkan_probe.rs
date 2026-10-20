//! Vulkan property queries both SPIR-V runtimes make: `poot-wgpu-runtime` reads them through the wgpu-hal
//! adapter's raw handles and `poot-vulkan-runtime` through its own `ash` instance, so each query is written
//! once here, over the raw `ash` handles.

use ash::{amd, vk};

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

/// A Vulkan device's active compute units from AMD's shader-core properties:
/// `VK_AMD_shader_core_properties2`'s `activeComputeUnitCount`, else `VK_AMD_shader_core_properties`'
/// engine x array x per-array product; `None` when the driver exposes neither (every non-AMD device).
///
/// # Safety
///
/// `physical_device` must come from `instance`'s own enumeration, and `instance` must be live.
pub unsafe fn amd_vulkan_compute_units(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<u32> {
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
        if supports(amd::shader_core_properties2::NAME) {
            let mut core = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            instance.get_physical_device_properties2(physical_device, &mut props);
            Some(core.active_compute_unit_count)
        } else if supports(amd::shader_core_properties::NAME) {
            let mut core = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            instance.get_physical_device_properties2(physical_device, &mut props);
            Some(
                core.shader_engine_count
                    * core.shader_arrays_per_engine_count
                    * core.compute_units_per_shader_array,
            )
        } else {
            None
        }
    }
}
