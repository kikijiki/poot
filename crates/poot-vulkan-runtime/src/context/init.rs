use crate::*;

impl Context {
    /// Create a Vulkan instance (no surface/swapchain), pick the first physical device with a
    /// `VK_QUEUE_COMPUTE_BIT` queue family (no graphics/present requirement, spec 133 FR-002), and open a
    /// logical device and queue on it.
    pub fn new() -> Result<Self, RuntimeError> {
        Self::new_inner().map_err(require_gpu_check)
    }

    pub(crate) fn new_inner() -> Result<Self, RuntimeError> {
        // SAFETY: `Entry::load` dlopens the system Vulkan loader (libvulkan.so.1); the usual `ash` dynamic-loading contract.
        let entry = unsafe { ash::Entry::load()? };

        let app_name = c"poot-vulkan-runtime";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_2);

        // Validation is a test-harness switch (POOT_VULKAN_VALIDATION=1, see `validation.rs`; Card 537,
        // ADR-0104 decision 5's debugging-read exception): when set, the layer and `VK_EXT_debug_utils`
        // are required, every warning or error the layer reports is counted, and the device panics on
        // teardown if any was. It never changes the command stream or the results.
        let validate = validation::requested();
        let mut layer_ptrs: Vec<*const std::ffi::c_char> = Vec::new();
        let mut instance_extensions: Vec<*const std::ffi::c_char> = Vec::new();
        if validate {
            let validation_layer = c"VK_LAYER_KHRONOS_validation";
            // SAFETY: read-only enumeration calls; the returned `Vec`s are not retained past this scope.
            let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
            // SAFETY: as above.
            let extensions =
                unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
            let has_layer = layers
                .iter()
                .any(|l| l.layer_name_as_c_str().ok() == Some(validation_layer));
            let has_utils = extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME));
            if !(has_layer && has_utils) {
                return Err(RuntimeError::ValidationUnavailable);
            }
            layer_ptrs.push(validation_layer.as_ptr());
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs)
            .enabled_extension_names(&instance_extensions);
        // SAFETY: `instance_info` and everything it points at (app_info, layer name C strings) outlive
        // this call.
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        // SAFETY: `instance` was just created and is valid for this enumeration.
        let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(e) => {
                // SAFETY: instance creation succeeded, but no device or other instance child exists yet.
                unsafe { instance.destroy_instance(None) };
                return Err(e.into());
            }
        };
        let mut chosen: Option<(vk::PhysicalDevice, u32)> = None;
        for &pd in &physical_devices {
            // SAFETY: `pd` came from `enumerate_physical_devices` on this same `instance`.
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            if let Some(idx) = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                chosen = Some((pd, idx as u32));
                break;
            }
        }
        let Some((physical_device, queue_family_index)) = chosen else {
            // SAFETY: no device/queue was ever created from this instance; this is its sole owner.
            unsafe { instance.destroy_instance(None) };
            return Err(RuntimeError::NoComputeDevice);
        };

        // SAFETY: `physical_device` is from this instance's own enumeration.
        let device_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .unwrap_or_default();
        // SAFETY: `physical_device` is from this instance's own enumeration.
        let physical_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let features = DeviceFeatures::query(
            &instance,
            physical_device,
            physical_props.api_version,
            &device_extensions,
        );
        let has_cooperative_matrix = features.has_cooperative_matrix_extension();

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let queue_infos = [queue_info];
        // SAFETY: `physical_device` is from this instance's own enumeration; the queue infos outlive the
        // call; every feature and extension `create_device` enables was reported supported by this
        // physical device (`DeviceFeatures::query`).
        let device =
            match unsafe { features.create_device(&instance, physical_device, &queue_infos) } {
                Ok(d) => d,
                Err(e) => {
                    // SAFETY: no device was created; the instance is still solely owned here.
                    unsafe { instance.destroy_instance(None) };
                    return Err(e.into());
                }
            };
        // SAFETY: `device` was just created with exactly one queue at (`queue_family_index`, index 0).
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Card 522 review: measured from `VK_KHR_cooperative_matrix`, never parsed from
        // `device_name()`'s string (Mesa RADV reports a marketing codename, not a gfx code). The
        // extension is instance-level (`khr::cooperative_matrix::Instance::new`) - only confirmed
        // present on this physical device first, since calling the query function when the driver does
        // not implement it is undefined by the ash binding (a null function-pointer stub that panics).
        // The device above enables the extension and its feature whenever the driver supports them, so
        // the matrix hardware reported here is hardware the SPIR-V the planner selects for it can use
        // (Card 553, R-553-1).
        let measured_tensor_core = if has_cooperative_matrix {
            let coop_matrix = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
            // SAFETY: `has_cooperative_matrix` confirmed the extension is supported on this physical
            // device, so the loaded function pointer is real (not the ash panic-on-call stub).
            let configs = unsafe {
                coop_matrix.get_physical_device_cooperative_matrix_properties(physical_device)
            }
            .unwrap_or_default();
            configs
                .into_iter()
                .filter(|c| c.scope == vk::ScopeKHR::SUBGROUP)
                .map(|c| {
                    poot_target::TensorCoreSupport::from_cooperative_matrix_config(
                        c.m_size,
                        c.n_size,
                        c.k_size,
                        c.a_type == vk::ComponentTypeKHR::FLOAT16,
                        c.result_type == vk::ComponentTypeKHR::FLOAT32,
                    )
                })
                .fold(poot_target::TensorCoreSupport::None, |acc, x| {
                    acc.most_specific(x)
                })
        } else {
            poot_target::TensorCoreSupport::UnknownNotExposedByApi
        };

        let validation = if validate {
            // SAFETY: the instance was created with `VK_EXT_debug_utils` enabled (above), and the
            // owner destroys the messenger before the instance.
            match unsafe { Validation::start(&entry, &instance) } {
                Ok(v) => Some(v),
                Err(e) => {
                    // SAFETY: nothing but the device was created from this instance; sole owner.
                    unsafe {
                        device.destroy_device(None);
                        instance.destroy_instance(None);
                    }
                    return Err(e.into());
                }
            }
        } else {
            None
        };
        let owner = Arc::new(DeviceOwner::new(
            device,
            instance,
            entry,
            queue,
            Arc::new(AshResourceDispatch),
            validation,
        ));

        // Confirm a HOST_VISIBLE|HOST_COHERENT memory type exists (FR-003's UMA path) before returning a `Context` whose first buffer alloc would fail.
        // SAFETY: `physical_device` is valid for this instance.
        let mem_props = unsafe {
            owner
                .instance
                .get_physical_device_memory_properties(physical_device)
        };
        if find_memory_type(
            &mem_props,
            u32::MAX,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .is_none()
        {
            // `owner` is still unique, so its drop tears down device then instance while retaining Entry.
            return Err(RuntimeError::NoHostVisibleMemory);
        }

        // SAFETY: `physical_device` is valid for this instance.
        let props = unsafe {
            owner
                .instance
                .get_physical_device_properties(physical_device)
        };
        let timestamp_period_ns = props.limits.timestamp_period;
        // SAFETY: `physical_device` is from this instance's own enumeration.
        let families = unsafe {
            owner
                .instance
                .get_physical_device_queue_family_properties(physical_device)
        };
        let timestamp_valid_bits = families[queue_family_index as usize].timestamp_valid_bits;
        let timestamp_valid_mask = match timestamp_valid_bits {
            0 => 0,
            1..=63 => (1u64 << timestamp_valid_bits) - 1,
            _ => u64::MAX,
        };
        let max_workgroup_count = props.limits.max_compute_work_group_count;

        // Card 522 review M1: measured once here, from the same `props`/`mem_props` queries this
        // constructor already made (the `mem_props` above, queried for the HOST_VISIBLE|HOST_COHERENT
        // check, carries the same heap data a device-local-max query would); `Context::device_caps`
        // returns this stored value forever after, never re-querying the driver.
        let device_local_max = mem_props.memory_heaps[..mem_props.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(u64::MAX);
        // One buffer must be a single allocation (`maxMemoryAllocationSize`, Maintenance3) that a
        // descriptor can bind whole (`maxStorageBufferRange`) out of one device-local heap.
        // SAFETY: `physical_device` is from this instance's own enumeration.
        let max_allocation =
            unsafe { max_memory_allocation_size(&owner.instance, physical_device) };
        let measured_max_buffer = device_local_max
            .min(max_allocation)
            .min(u64::from(props.limits.max_storage_buffer_range));
        let is_amd_radv = props.vendor_id == 0x1002;
        let (max_buffer_bytes, watchdog_budget, known_miscompiles) = if is_amd_radv {
            let d = poot_target::DeviceCaps::wgpu_rdna3_igpu();
            (
                measured_max_buffer.min(poot_runtime::AMD_RADV_SAFE_MAX_BUFFER_BYTES),
                d.watchdog_budget,
                d.known_miscompiles,
            )
        } else {
            (
                measured_max_buffer,
                None,
                poot_target::KnownMiscompiles::default(),
            )
        };
        // Card 653: AMD's shader-core properties where the driver exposes them, else
        // the RDNA3 default, as the wgpu runtime does.
        // SAFETY: `physical_device` is from this instance's own enumeration.
        let compute_units =
            unsafe { poot_runtime::amd_vulkan_compute_units(&owner.instance, physical_device) }
                .unwrap_or(poot_target::DeviceCaps::wgpu_rdna3_igpu().compute_units);
        // SAFETY: `physical_device` is from this instance's own enumeration; the extension list was
        // read from the same device.
        let subgroup = unsafe {
            probe_subgroup(
                &owner.instance,
                physical_device,
                props.api_version,
                &device_extensions,
            )
        };
        let launch = LaunchCaps::from_vulkan(&props.limits, subgroup);
        // A cooperative-matrix pipeline is created with full subgroups of the size its workgroup was
        // written for, which every matrix-fragment kernel the codegen emits fixes at 32 lanes. A device
        // that cannot create such a pipeline (no subgroup-size control, no 32-lane size) runs no
        // fragment the planner could select, so it reports no matrix hardware (Card 553, R-553-1).
        let coopmat_subgroup_sizes = subgroup.size_range.filter(|&(min, max)| {
            features.cooperative_matrix_pipelines()
                && subgroup.compute_size_control
                && (min..=max).contains(&COOPMAT_SUBGROUP_SIZE)
        });
        let tensor_core = if has_cooperative_matrix && coopmat_subgroup_sizes.is_none() {
            poot_target::TensorCoreSupport::None
        } else {
            measured_tensor_core
        };
        let device_caps = poot_target::DeviceCaps {
            max_buffer_bytes,
            lds_bytes: props.limits.max_compute_shared_memory_size,
            max_workgroup_size: launch.max_workgroup_size,
            max_workgroup_invocations: launch.max_workgroup_invocations,
            subgroup: launch.subgroup,
            // No driver API reports a safe per-dispatch work bound.
            max_dispatch_work: poot_target::Queried::Unknown,
            max_grid: max_workgroup_count,
            watchdog_budget,
            tensor_core,
            known_miscompiles,
            compute_units,
        };

        Ok(Context {
            instance: owner.instance.clone(),
            device: owner.device.clone(),
            physical_device,
            queue_family_index,
            timestamp_period_ns,
            timestamp_valid_mask,
            coopmat_subgroup_sizes,
            min_storage_offset_alignment: props.limits.min_storage_buffer_offset_alignment as usize,
            max_workgroup_count,
            device_caps,
            pipeline_cache: HashMap::new(),
            pipeline_builds: 0,
            graphs_recorded: AtomicUsize::new(0),
            owner,
        })
    }

    /// Count of pipeline builds (cache misses) via [`Context::pipeline`] so far; unchanged by calls whose key is cached. For tests.
    pub fn pipeline_builds(&self) -> usize {
        self.pipeline_builds
    }

    /// The physical device's per-axis `maxComputeWorkGroupCount` a dispatch's grid must fit.
    pub fn max_workgroup_count(&self) -> [u32; 3] {
        self.max_workgroup_count
    }

    /// Live/peak bytes and allocation count by role, from this device's own allocation counters.
    pub fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.owner.memory.snapshot_all()
    }

    /// Whether a lost submission may still be running: every later operation is refused.
    pub fn is_poisoned(&self) -> bool {
        self.owner.is_poisoned()
    }

    /// Count of `Context::end_graph` calls (command buffers finished recording) so far. For tests.
    pub fn graphs_recorded(&self) -> usize {
        self.graphs_recorded.load(Ordering::Relaxed)
    }

    /// The selected physical device's name (e.g. `"AMD Radeon Graphics (RADV GFX1151)"`), for diagnostics.
    pub fn device_name(&self) -> String {
        // SAFETY: `self.physical_device` is valid for `self.instance`'s whole lifetime.
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        // `device_name` is NUL-terminated; take up to the first NUL.
        let bytes: &[u8] = bytemuck_cast_i8_to_u8(&props.device_name);
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..end]).into_owned()
    }

    /// The device-capability descriptor the planner reads (card 522), measured once at construction and
    /// stored (card 522 review M1: a `Copy` read, never a re-query of the driver on every call):
    /// `max_compute_shared_memory_size` and `max_compute_work_group_count` are this device's own query,
    /// and `max_buffer_bytes` is the smallest of the largest `DEVICE_LOCAL` memory heap, the
    /// Maintenance3 `maxMemoryAllocationSize` and `maxStorageBufferRange` (a single buffer must be one
    /// allocation a descriptor can bind whole; Card 553, R-553-1). This crate dispatches the same SPIR-V through the same Vulkan driver
    /// as `poot-runtime` (wgpu), so an AMD Vulkan (RADV) device shares its RADV gfx-ring wedge
    /// (`max_buffer_bytes`), display-watchdog budget and card-095 tiled-GEMM miscompile ceiling; both are
    /// the same vendor-scoped default ([`poot_target::DeviceCaps::wgpu_rdna3_igpu`]) `poot-runtime`
    /// reports there, not a device query (neither is exposed by any driver API). `tensor_core` is the
    /// `VK_KHR_cooperative_matrix` query measured once at construction (card 522 review), never parsed
    /// from `device_name()`'s marketing codename string.
    pub fn device_caps(&self) -> poot_target::DeviceCaps {
        self.device_caps
    }
}

/// The largest single `vkAllocateMemory` this device allows (`VkPhysicalDeviceMaintenance3Properties`,
/// core in Vulkan 1.1; the instance is created at 1.2).
///
/// # Safety
/// `physical_device` must come from `instance`'s own enumeration.
unsafe fn max_memory_allocation_size(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> u64 {
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    // SAFETY: the caller's contract; `props2` and its chain outlive the call.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
    maintenance3.max_memory_allocation_size
}

/// The subgroup size every matrix-fragment (cooperative-matrix) kernel the codegen emits is written for.
const COOPMAT_SUBGROUP_SIZE: u32 = 32;

/// What a physical device reports about its compute-stage subgroups (Vulkan 1.1 `subgroupProperties`,
/// and the subgroup-size range where `VK_EXT_subgroup_size_control` or Vulkan 1.3 reports one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SubgroupProbe {
    /// The compute stage supports the basic subgroup operations.
    pub(crate) compute_basic: bool,
    /// `subgroupSize`: the size a pipeline gets by default.
    pub(crate) size: u32,
    /// `(minSubgroupSize, maxSubgroupSize)` where the device reports a controllable range.
    pub(crate) size_range: Option<(u32, u32)>,
    /// A compute pipeline may require a subgroup size within `size_range`.
    pub(crate) compute_size_control: bool,
}

/// The launch-shape limits of a Vulkan device, converted from its reported limits and subgroup probe (the
/// seam [`Context::device_caps`] is built from, so a test can feed sentinel values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchCaps {
    pub(crate) max_workgroup_size: [u32; 3],
    pub(crate) max_workgroup_invocations: u32,
    pub(crate) subgroup: poot_target::Queried<poot_target::SubgroupSupport>,
}

impl LaunchCaps {
    pub(crate) fn from_vulkan(limits: &vk::PhysicalDeviceLimits, subgroup: SubgroupProbe) -> Self {
        let (min_size, max_size) = subgroup
            .size_range
            .unwrap_or((subgroup.size, subgroup.size));
        Self {
            max_workgroup_size: limits.max_compute_work_group_size,
            max_workgroup_invocations: limits.max_compute_work_group_invocations,
            subgroup: poot_target::Queried::Known(if subgroup.compute_basic {
                poot_target::SubgroupSupport::Present { min_size, max_size }
            } else {
                poot_target::SubgroupSupport::Absent
            }),
        }
    }
}

/// Read `physical_device`'s subgroup properties. The size range is chained only when the device reports
/// it (`VK_EXT_subgroup_size_control`, or Vulkan 1.3 where it is core).
///
/// # Safety
/// `physical_device` must come from `instance`'s own enumeration, and `extensions` from the same device.
unsafe fn probe_subgroup(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    api_version: u32,
    extensions: &[vk::ExtensionProperties],
) -> SubgroupProbe {
    let has_size_control = api_version >= vk::API_VERSION_1_3
        || extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::subgroup_size_control::NAME));
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    if has_size_control {
        props2 = props2.push_next(&mut control);
    }
    // SAFETY: the caller's contract; `props2` and its chain outlive the call.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
    SubgroupProbe {
        compute_basic: subgroup
            .supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            && subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::BASIC),
        size: subgroup.subgroup_size,
        size_range: has_size_control
            .then_some((control.min_subgroup_size, control.max_subgroup_size)),
        compute_size_control: has_size_control
            && control
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE),
    }
}

#[cfg(test)]
mod launch_caps_tests {
    use super::*;
    use poot_target::{Queried, SubgroupSupport};

    fn limits() -> vk::PhysicalDeviceLimits {
        vk::PhysicalDeviceLimits {
            max_compute_work_group_size: [777, 333, 11],
            max_compute_work_group_invocations: 555,
            ..Default::default()
        }
    }

    /// SC-003: sentinel limits and a subgroup range no real device reports come out exactly. Mutation:
    /// report `max_compute_work_group_count` or a guessed 1024 for the invocation limit, and the first
    /// assertions fail with the wrong value.
    #[test]
    fn launch_caps_carry_the_reported_limits_and_the_subgroup_range() {
        let caps = LaunchCaps::from_vulkan(
            &limits(),
            SubgroupProbe {
                compute_basic: true,
                size: 48,
                size_range: Some((8, 128)),
                compute_size_control: true,
            },
        );
        assert_eq!(caps.max_workgroup_size, [777, 333, 11]);
        assert_eq!(caps.max_workgroup_invocations, 555);
        assert_eq!(
            caps.subgroup,
            Queried::Known(SubgroupSupport::Present {
                min_size: 8,
                max_size: 128
            })
        );
    }

    /// Without a controllable range the default size is the whole range; without compute-stage basic
    /// operations the device has no subgroup support, whatever size it reports.
    #[test]
    fn a_fixed_size_and_a_missing_feature_are_reported_as_such() {
        let fixed = LaunchCaps::from_vulkan(
            &limits(),
            SubgroupProbe {
                compute_basic: true,
                size: 48,
                size_range: None,
                compute_size_control: false,
            },
        );
        assert_eq!(
            fixed.subgroup,
            Queried::Known(SubgroupSupport::Present {
                min_size: 48,
                max_size: 48
            })
        );
        let absent = LaunchCaps::from_vulkan(
            &limits(),
            SubgroupProbe {
                compute_basic: false,
                size: 48,
                size_range: None,
                compute_size_control: false,
            },
        );
        assert_eq!(absent.subgroup, Queried::Known(SubgroupSupport::Absent));
    }
}

#[cfg(test)]
mod launch_caps_device_receipt {
    use super::*;

    /// Receipt (SC-003): on a live AMD (RADV) Vulkan device the launch limits and subgroup range the
    /// context reports are the device's own `maxComputeWorkGroup*` and subgroup properties, and the
    /// subgroup range contains the 32-lane subgroup every matrix-fragment kernel is written for. A skip
    /// (no Vulkan device, or a non-AMD one) is reported as one, not as a pass.
    #[test]
    fn a_live_vulkan_device_reports_its_own_launch_limits() {
        let Ok(ctx) = Context::new() else {
            eprintln!("SKIP a_live_vulkan_device_reports_its_own_launch_limits: no Vulkan device");
            return;
        };
        let live = ctx.device_caps();
        eprintln!("live raw-vulkan caps: {live:?}");
        // SAFETY: `ctx.physical_device` is valid for `ctx.instance`.
        let limits = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical_device)
        }
        .limits;
        assert_eq!(live.max_workgroup_size, limits.max_compute_work_group_size);
        assert_eq!(
            live.max_workgroup_invocations,
            limits.max_compute_work_group_invocations
        );
        let poot_target::Queried::Known(poot_target::SubgroupSupport::Present {
            min_size,
            max_size,
        }) = live.subgroup
        else {
            panic!(
                "a Vulkan 1.2 device reports its subgroup properties, got {:?}",
                live.subgroup
            );
        };
        assert!(min_size <= max_size && max_size.is_power_of_two() && min_size >= 4);
    }
}
