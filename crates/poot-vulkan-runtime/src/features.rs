use std::ffi::{CStr, c_char};

use crate::*;

/// The shader and storage features a physical device supports out of those poot's SPIR-V can declare:
/// the `Float16`/`Int8`/`Int16`/`Int64`/`Float64` arithmetic capabilities, 8- and 16-bit storage-buffer
/// access, the Vulkan memory model (cooperative-matrix modules require it), zero-initialized workgroup
/// memory (the SPIR-V backend emits initializers), subgroup-size control and cooperative matrices.
/// Everything supported is enabled at device creation, the way the wgpu runtime requests its optional
/// features; nothing unsupported is ever requested.
#[derive(Clone, Copy, Default)]
pub(crate) struct DeviceFeatures {
    shader_float64: bool,
    shader_int16: bool,
    shader_int64: bool,
    /// The device is Vulkan 1.2, so the 1.1 and 1.2 feature structs below were queried.
    vulkan12_core: bool,
    storage_buffer16: bool,
    shader_float16: bool,
    shader_int8: bool,
    storage_buffer8: bool,
    memory_model: bool,
    memory_model_device_scope: bool,
    subgroup_extended_types: bool,
    /// `VK_KHR_cooperative_matrix` is present (its query functions exist), whether or not its feature
    /// is supported.
    cooperative_matrix_extension: bool,
    cooperative_matrix: bool,
    zero_initialize_workgroup_memory: bool,
    /// `VK_EXT_subgroup_size_control` with `subgroupSizeControl` and `computeFullSubgroups`: a pipeline
    /// can require full subgroups of a chosen size.
    subgroup_size_control: bool,
}

impl DeviceFeatures {
    pub(crate) fn query(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        api_version: u32,
        extensions: &[vk::ExtensionProperties],
    ) -> Self {
        let has = |name: &CStr| {
            extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let vulkan12_core = api_version >= vk::API_VERSION_1_2;
        let has_coop = has(ash::khr::cooperative_matrix::NAME);
        let has_zero_init = has(ash::khr::zero_initialize_workgroup_memory::NAME);
        let has_size_control = has(ash::ext::subgroup_size_control::NAME);
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut zero_init = vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default();
        let mut size_control = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        if vulkan12_core {
            features2 = features2.push_next(&mut v11).push_next(&mut v12);
        }
        if has_coop {
            features2 = features2.push_next(&mut coop);
        }
        if has_zero_init {
            features2 = features2.push_next(&mut zero_init);
        }
        if has_size_control {
            features2 = features2.push_next(&mut size_control);
        }
        // SAFETY: `physical_device` is from `instance`'s own enumeration; the chain only holds structs
        // the device's API version (1.2) or its extension list says it knows, and outlives the call.
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        let core = features2.features;
        let on = |flag: vk::Bool32| flag == vk::TRUE;
        DeviceFeatures {
            shader_float64: on(core.shader_float64),
            shader_int16: on(core.shader_int16),
            shader_int64: on(core.shader_int64),
            vulkan12_core,
            storage_buffer16: vulkan12_core && on(v11.storage_buffer16_bit_access),
            shader_float16: vulkan12_core && on(v12.shader_float16),
            shader_int8: vulkan12_core && on(v12.shader_int8),
            storage_buffer8: vulkan12_core && on(v12.storage_buffer8_bit_access),
            memory_model: vulkan12_core && on(v12.vulkan_memory_model),
            memory_model_device_scope: vulkan12_core && on(v12.vulkan_memory_model_device_scope),
            subgroup_extended_types: vulkan12_core && on(v12.shader_subgroup_extended_types),
            cooperative_matrix_extension: has_coop,
            cooperative_matrix: has_coop && on(coop.cooperative_matrix),
            zero_initialize_workgroup_memory: has_zero_init
                && on(zero_init.shader_zero_initialize_workgroup_memory),
            subgroup_size_control: has_size_control
                && on(size_control.subgroup_size_control)
                && on(size_control.compute_full_subgroups),
        }
    }

    /// Whether `VK_KHR_cooperative_matrix` is present, so its properties can be queried.
    pub(crate) fn has_cooperative_matrix_extension(&self) -> bool {
        self.cooperative_matrix_extension
    }

    /// Whether the device can create a pipeline for a module that declares cooperative matrices: the
    /// feature is enabled, and the pipeline can require full subgroups of a chosen size (a pre-1.6
    /// module needs `REQUIRE_FULL_SUBGROUPS`, and a workgroup that is not a multiple of the device's
    /// default subgroup size needs a required size).
    pub(crate) fn cooperative_matrix_pipelines(&self) -> bool {
        self.cooperative_matrix && self.subgroup_size_control
    }

    /// Create the logical device with every supported feature enabled.
    ///
    /// # Safety
    ///
    /// `physical_device` must come from `instance`'s own enumeration, and `self` must be the result of
    /// [`DeviceFeatures::query`] on that device.
    pub(crate) unsafe fn create_device(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_infos: &[vk::DeviceQueueCreateInfo<'_>],
    ) -> Result<ash::Device, vk::Result> {
        let mut extension_names: Vec<*const c_char> = Vec::new();
        if self.cooperative_matrix {
            extension_names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        if self.zero_initialize_workgroup_memory {
            extension_names.push(ash::khr::zero_initialize_workgroup_memory::NAME.as_ptr());
        }
        if self.subgroup_size_control {
            extension_names.push(ash::ext::subgroup_size_control::NAME.as_ptr());
        }
        let core = vk::PhysicalDeviceFeatures::default()
            .shader_float64(self.shader_float64)
            .shader_int16(self.shader_int16)
            .shader_int64(self.shader_int64);
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(self.storage_buffer16);
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(self.shader_float16)
            .shader_int8(self.shader_int8)
            .storage_buffer8_bit_access(self.storage_buffer8)
            .vulkan_memory_model(self.memory_model)
            .vulkan_memory_model_device_scope(self.memory_model_device_scope)
            .shader_subgroup_extended_types(self.subgroup_extended_types);
        let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(self.cooperative_matrix);
        let mut zero_init = vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default()
            .shader_zero_initialize_workgroup_memory(self.zero_initialize_workgroup_memory);
        let mut size_control = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(self.subgroup_size_control)
            .compute_full_subgroups(self.subgroup_size_control);
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(queue_infos)
            .enabled_features(&core)
            .enabled_extension_names(&extension_names);
        if self.vulkan12_core {
            info = info.push_next(&mut v11).push_next(&mut v12);
        }
        if self.cooperative_matrix {
            info = info.push_next(&mut coop);
        }
        if self.zero_initialize_workgroup_memory {
            info = info.push_next(&mut zero_init);
        }
        if self.subgroup_size_control {
            info = info.push_next(&mut size_control);
        }
        // SAFETY: the caller's contract; `info` and the structs chained into it outlive the call.
        unsafe { instance.create_device(physical_device, &info, None) }
    }
}
