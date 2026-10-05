use crate::*;

/// Card 537 (ADR-0104 decision 5): a test-harness convention, not a production execution choice - see
/// `poot_runtime::require::require_gpu_check`'s matching note (the wgpu counterpart this mirrors).
///
/// `POOT_REQUIRE_VULKAN=1` turns the "no Vulkan device: tests skip" convention into a failure: a
/// failed context open panics instead of returning the Err that skip guards turn into a silent pass
/// (`just test-device-vulkan` sets it). Failure path only; mirrors `poot-runtime::require_gpu_check`.
pub(crate) fn require_gpu_check<E: std::fmt::Display>(e: E) -> E {
    require_gpu_check_with(|name| std::env::var(name).ok(), e)
}

pub(crate) fn require_gpu_check_with<E: std::fmt::Display>(
    get: impl Fn(&str) -> Option<String>,
    e: E,
) -> E {
    poot_runtime_common::DeviceBackend::Vulkan.fail_if_required(get, e)
}

pub(crate) struct BufferResources {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) ptr: *mut u8,
}

/// Owns each buffer resource as soon as it is acquired, so an early return runs the same cleanup as a completed `DeviceBuffer`; `finish` transfers all handles.
pub(crate) struct PendingBuffer<'a> {
    pub(crate) owner: &'a DeviceOwner,
    pub(crate) buffer: Option<vk::Buffer>,
    pub(crate) memory: Option<vk::DeviceMemory>,
    pub(crate) mapped: bool,
}

impl PendingBuffer<'_> {
    pub(crate) fn finish(mut self, ptr: *mut u8) -> BufferResources {
        self.mapped = false;
        BufferResources {
            buffer: self.buffer.take().expect("buffer acquired before finish"),
            memory: self.memory.take().expect("memory acquired before finish"),
            ptr,
        }
    }
}

impl Drop for PendingBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: the guard owns every populated handle. Bound memory must remain alive until after its
        // buffer is destroyed; mapped memory is unmapped first.
        unsafe {
            if self.mapped {
                self.owner.dispatch.unmap_memory(
                    &self.owner.device,
                    self.memory.expect("mapped memory was acquired"),
                );
            }
            if let Some(buffer) = self.buffer.take() {
                self.owner
                    .dispatch
                    .destroy_buffer(&self.owner.device, buffer);
            }
            if let Some(memory) = self.memory.take() {
                self.owner.dispatch.free_memory(&self.owner.device, memory);
            }
        }
    }
}

/// Create, back, bind and map one storage buffer of exactly `byte_len` bytes: the only path to
/// `vkCreateBuffer`. A zero size is refused before the driver call (it violates
/// VUID-VkBufferCreateInfo-size-00912), and so is a size whose 4-byte element count does not fit the
/// `u32` the length buffer carries ([`DeviceBuffer::elem_count`]).
pub(crate) fn allocate_buffer_resources(
    owner: &DeviceOwner,
    byte_len: usize,
    choose_memory_type: impl FnOnce(vk::MemoryRequirements) -> Option<u32>,
) -> Result<BufferResources, RuntimeError> {
    if byte_len == 0 {
        return Err(RuntimeError::ZeroSizeBuffer);
    }
    if u32::try_from(byte_len / 4).is_err() {
        return Err(RuntimeError::BufferTooLarge { bytes: byte_len });
    }
    let mut pending = PendingBuffer {
        owner,
        buffer: None,
        memory: None,
        mapped: false,
    };
    let buf_info = vk::BufferCreateInfo::default()
        .size(byte_len as vk::DeviceSize)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: the owner's device and dispatch table stay alive for the guard's lifetime.
    let buffer = unsafe { owner.dispatch.create_buffer(&owner.device, &buf_info)? };
    pending.buffer = Some(buffer);
    // SAFETY: `buffer` was just created by the same dispatch table and is owned by `pending`.
    let reqs = unsafe {
        owner
            .dispatch
            .get_buffer_memory_requirements(&owner.device, buffer)
    };
    let memory_type_index = choose_memory_type(reqs).ok_or(RuntimeError::NoHostVisibleMemory)?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(memory_type_index);
    // SAFETY: `alloc_info` uses the requirements for `buffer`; the owner remains alive.
    let memory = unsafe { owner.dispatch.allocate_memory(&owner.device, &alloc_info)? };
    pending.memory = Some(memory);
    // SAFETY: both handles came from this owner and the allocation satisfies `buffer`'s requirements.
    unsafe {
        owner
            .dispatch
            .bind_buffer_memory(&owner.device, buffer, memory)?;
    }
    // SAFETY: the selected memory type is host visible, the memory is not already mapped, and the mapping
    // remains owned by `pending` until it is transferred or cleaned up.
    let ptr = unsafe { owner.dispatch.map_memory(&owner.device, memory)? } as *mut u8;
    pending.mapped = true;
    Ok(pending.finish(ptr))
}
