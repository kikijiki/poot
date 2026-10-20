//! Which physical device the context opens: every device with a compute queue family and the memory this
//! runtime's buffers need is a candidate, and the best-ranked candidate wins.

use crate::RuntimeError;
use crate::util::find_memory_type;
use ash::vk;

/// The memory type every buffer is allocated from: device-local memory the host can map and read without a
/// flush or an invalidate. A unified-memory part (an iGPU) exposes it on its one heap; a discrete part exposes
/// it only on the BAR-visible slice of VRAM, and a part with none is refused (a staging copy path is not
/// implemented, so a device without the type cannot run this runtime's buffers).
pub(crate) const BUFFER_MEMORY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()
        | vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

/// What selection reads of one enumerated physical device.
#[derive(Clone)]
pub(crate) struct DeviceFacts {
    pub(crate) device_type: vk::PhysicalDeviceType,
    pub(crate) queue_families: Vec<vk::QueueFamilyProperties>,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
}

/// The chosen device: its index in the enumeration and its compute queue family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pick {
    pub(crate) device: usize,
    pub(crate) queue_family: u32,
}

impl DeviceFacts {
    /// The first queue family with compute support (no graphics or present requirement).
    fn compute_family(&self) -> Option<u32> {
        self.queue_families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|i| i as u32)
    }

    /// Whether [`BUFFER_MEMORY`] exists on this device.
    fn has_buffer_memory(&self) -> bool {
        find_memory_type(&self.memory, u32::MAX, BUFFER_MEMORY).is_some()
    }

    /// The bytes of the largest device-local heap.
    fn local_heap_bytes(&self) -> u64 {
        self.memory.memory_heaps[..self.memory.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0)
    }

    /// Lower is preferred: a discrete GPU, then an integrated one, then a virtual one, and a software or
    /// unknown device last.
    fn type_rank(&self) -> u8 {
        match self.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 4,
            _ => 3,
        }
    }
}

/// Pick the device to open. A candidate has a compute queue family and [`BUFFER_MEMORY`]; candidates rank by
/// device type, then by device-local heap size, then by enumeration order. No candidate is a typed refusal
/// that says which requirement the enumerated devices failed.
pub(crate) fn pick_device(devices: &[DeviceFacts]) -> Result<Pick, RuntimeError> {
    let with_compute: Vec<(usize, u32)> = devices
        .iter()
        .enumerate()
        .filter_map(|(i, d)| d.compute_family().map(|family| (i, family)))
        .collect();
    if with_compute.is_empty() {
        return Err(RuntimeError::NoComputeDevice);
    }
    with_compute
        .into_iter()
        .filter(|&(i, _)| devices[i].has_buffer_memory())
        .min_by_key(|&(i, _)| {
            (
                devices[i].type_rank(),
                std::cmp::Reverse(devices[i].local_heap_bytes()),
                i,
            )
        })
        .map(|(device, queue_family)| Pick {
            device,
            queue_family,
        })
        .ok_or(RuntimeError::NoMappableDeviceLocalMemory)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family(flags: vk::QueueFlags) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: 1,
            ..Default::default()
        }
    }

    /// One heap of `bytes` and one memory type with `flags` on it.
    fn memory(
        flags: vk::MemoryPropertyFlags,
        heap: vk::MemoryHeapFlags,
        bytes: u64,
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut memory = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 1,
            memory_heap_count: 1,
            ..Default::default()
        };
        memory.memory_types[0] = vk::MemoryType {
            property_flags: flags,
            heap_index: 0,
        };
        memory.memory_heaps[0] = vk::MemoryHeap {
            size: bytes,
            flags: heap,
        };
        memory
    }

    fn device(
        device_type: vk::PhysicalDeviceType,
        queues: vk::QueueFlags,
        buffer_memory: bool,
        bytes: u64,
    ) -> DeviceFacts {
        let flags = if buffer_memory {
            BUFFER_MEMORY
        } else {
            // Host-visible but not device-local: the memory type of a discrete part's system-RAM heap.
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        };
        DeviceFacts {
            device_type,
            queue_families: vec![family(vk::QueueFlags::GRAPHICS), family(queues)],
            memory: memory(flags, vk::MemoryHeapFlags::DEVICE_LOCAL, bytes),
        }
    }

    const GIB: u64 = 1 << 30;

    /// A software rasterizer listed before a GPU does not win: devices rank by type, not enumeration order.
    /// Mutation: return the first candidate in `pick_device`: the CPU device is picked and this fails.
    #[test]
    fn a_gpu_outranks_a_software_device_listed_first() {
        let devices = [
            device(
                vk::PhysicalDeviceType::CPU,
                vk::QueueFlags::COMPUTE,
                true,
                8 * GIB,
            ),
            device(
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::QueueFlags::COMPUTE,
                true,
                8 * GIB,
            ),
        ];
        assert_eq!(
            pick_device(&devices).unwrap(),
            Pick {
                device: 1,
                queue_family: 1
            }
        );
    }

    /// Among devices of one type the larger device-local heap wins; a tie keeps enumeration order.
    #[test]
    fn the_larger_device_local_heap_breaks_a_type_tie() {
        let gpu = |gib| {
            device(
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::QueueFlags::COMPUTE,
                true,
                gib * GIB,
            )
        };
        let picked = pick_device(&[gpu(8), gpu(24), gpu(24)]).unwrap();
        assert_eq!(picked.device, 1);
    }

    /// A device without a compute queue family is never a candidate, however it ranks.
    #[test]
    fn a_device_without_a_compute_queue_is_skipped() {
        let devices = [
            device(
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::QueueFlags::GRAPHICS,
                true,
                24 * GIB,
            ),
            device(
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::QueueFlags::COMPUTE,
                true,
                8 * GIB,
            ),
        ];
        assert_eq!(pick_device(&devices).unwrap().device, 1);
        assert!(matches!(
            pick_device(&devices[..1]),
            Err(RuntimeError::NoComputeDevice)
        ));
    }

    /// A device with no DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory type is refused, not opened on the
    /// assumption of unified memory; a lesser device that has the type is chosen over it.
    /// Mutation: drop the `has_buffer_memory` filter in `pick_device`: the discrete device is picked and
    /// both assertions fail.
    #[test]
    fn a_device_without_mappable_device_local_memory_is_refused() {
        let discrete = device(
            vk::PhysicalDeviceType::DISCRETE_GPU,
            vk::QueueFlags::COMPUTE,
            false,
            24 * GIB,
        );
        assert!(matches!(
            pick_device(std::slice::from_ref(&discrete)),
            Err(RuntimeError::NoMappableDeviceLocalMemory)
        ));
        let integrated = device(
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            vk::QueueFlags::COMPUTE,
            true,
            8 * GIB,
        );
        assert_eq!(pick_device(&[discrete, integrated]).unwrap().device, 1);
    }
}
