use crate::*;

impl Context {
    pub(crate) fn alloc_buffer(
        &self,
        byte_len: usize,
        storage: BufferStorage,
        role: BufferRole,
    ) -> Result<DeviceBuffer, RuntimeError> {
        self.owner.check_live("alloc_buffer")?;
        // SAFETY: `self.physical_device` is valid for `self.instance`.
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let resources = allocate_buffer_resources(&self.owner, byte_len, |reqs| {
            find_memory_type(
                &mem_props,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })?;
        let buffer = DeviceBuffer::from_resources(resources, byte_len, storage, role, &self.owner);
        // `vkAllocateMemory` contents are undefined; the contract hands out zero-filled buffers.
        buffer.zero()?;
        Ok(buffer)
    }

    /// A fresh, zero-filled, persistently mapped device buffer of `elems` native elements of
    /// `storage`, charged to `role` (Card 547a: the storage-typed, role-typed allocation the executor
    /// contract's memory service needs, instead of one allocator per dtype - the Vulkan per-dtype
    /// family this replaces had no consumer outside this crate's own tests; mirrors
    /// `poot_runtime::Context::alloc_storage`). `elems == 0` is refused with
    /// [`RuntimeError::ZeroSizeBuffer`] (`allocate_buffer_resources`'s own check). The allocation is
    /// rounded up to whole 4-byte words (a kernel addresses storage by word), so `elems` of a
    /// narrower element can leave padding at the end.
    pub fn alloc_storage(
        &self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<DeviceBuffer, RuntimeError> {
        let byte_len = elems
            .checked_mul(storage.element().byte_width())
            .and_then(|bytes| bytes.checked_next_multiple_of(4))
            .ok_or(RuntimeError::ElementCountOverflow { elems })?;
        self.alloc_buffer(byte_len, storage, role)
    }
}
