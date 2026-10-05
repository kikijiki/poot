use crate::*;

/// The native calls whose ordering is part of the ownership contract. Production forwards to ash; tests
/// substitute a dispatch table so failure and destruction paths need no Vulkan device.
///
/// # Safety
///
/// Each method has the contract of the `ash` call of the same name: the caller upholds that call's
/// Vulkan valid-usage rules (live handles that are children of `device`, no pending use of a handle
/// being destroyed, externally synchronized access). [`AshResourceDispatch`] forwards unchanged.
pub(crate) trait ResourceDispatch: Send + Sync {
    unsafe fn create_buffer(
        &self,
        device: &ash::Device,
        info: &vk::BufferCreateInfo<'_>,
    ) -> Result<vk::Buffer, vk::Result>;
    unsafe fn get_buffer_memory_requirements(
        &self,
        device: &ash::Device,
        buffer: vk::Buffer,
    ) -> vk::MemoryRequirements;
    unsafe fn allocate_memory(
        &self,
        device: &ash::Device,
        info: &vk::MemoryAllocateInfo<'_>,
    ) -> Result<vk::DeviceMemory, vk::Result>;
    unsafe fn bind_buffer_memory(
        &self,
        device: &ash::Device,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
    ) -> Result<(), vk::Result>;
    unsafe fn map_memory(
        &self,
        device: &ash::Device,
        memory: vk::DeviceMemory,
    ) -> Result<*mut c_void, vk::Result>;
    unsafe fn unmap_memory(&self, device: &ash::Device, memory: vk::DeviceMemory);
    unsafe fn destroy_buffer(&self, device: &ash::Device, buffer: vk::Buffer);
    unsafe fn free_memory(&self, device: &ash::Device, memory: vk::DeviceMemory);

    unsafe fn create_command_pool(
        &self,
        device: &ash::Device,
        info: &vk::CommandPoolCreateInfo<'_>,
    ) -> Result<vk::CommandPool, vk::Result>;
    unsafe fn allocate_command_buffers(
        &self,
        device: &ash::Device,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result>;
    unsafe fn create_fence(
        &self,
        device: &ash::Device,
        info: &vk::FenceCreateInfo<'_>,
    ) -> Result<vk::Fence, vk::Result>;
    unsafe fn begin_command_buffer(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        info: &vk::CommandBufferBeginInfo<'_>,
    ) -> Result<(), vk::Result>;
    unsafe fn end_command_buffer(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), vk::Result>;
    unsafe fn reset_fences(
        &self,
        device: &ash::Device,
        fences: &[vk::Fence],
    ) -> Result<(), vk::Result>;
    unsafe fn queue_submit(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<(), vk::Result>;
    unsafe fn wait_for_fences(
        &self,
        device: &ash::Device,
        fences: &[vk::Fence],
    ) -> Result<(), vk::Result>;
    unsafe fn destroy_fence(&self, device: &ash::Device, fence: vk::Fence);
    unsafe fn destroy_descriptor_pool(&self, device: &ash::Device, pool: vk::DescriptorPool);
    unsafe fn destroy_command_pool(&self, device: &ash::Device, pool: vk::CommandPool);

    unsafe fn destroy_pipeline(&self, device: &ash::Device, pipeline: vk::Pipeline);
    unsafe fn destroy_pipeline_layout(&self, device: &ash::Device, layout: vk::PipelineLayout);
    unsafe fn destroy_descriptor_set_layout(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
    );
    unsafe fn destroy_shader_module(&self, device: &ash::Device, module: vk::ShaderModule);
    unsafe fn destroy_device(&self, device: &ash::Device);
    unsafe fn destroy_instance(&self, instance: &ash::Instance);

    /// Test seam for the loader guard's final release. Production needs no call: dropping `ash::Entry` after `DeviceOwner::drop` closes the library guard.
    fn release_loader(&self) {}
}

pub(crate) struct AshResourceDispatch;

impl ResourceDispatch for AshResourceDispatch {
    unsafe fn create_buffer(
        &self,
        device: &ash::Device,
        info: &vk::BufferCreateInfo<'_>,
    ) -> Result<vk::Buffer, vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.create_buffer(info, None) }
    }
    unsafe fn get_buffer_memory_requirements(
        &self,
        device: &ash::Device,
        buffer: vk::Buffer,
    ) -> vk::MemoryRequirements {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.get_buffer_memory_requirements(buffer) }
    }
    unsafe fn allocate_memory(
        &self,
        device: &ash::Device,
        info: &vk::MemoryAllocateInfo<'_>,
    ) -> Result<vk::DeviceMemory, vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.allocate_memory(info, None) }
    }
    unsafe fn bind_buffer_memory(
        &self,
        device: &ash::Device,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
    ) -> Result<(), vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
    }
    unsafe fn map_memory(
        &self,
        device: &ash::Device,
        memory: vk::DeviceMemory,
    ) -> Result<*mut c_void, vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
    }
    unsafe fn unmap_memory(&self, device: &ash::Device, memory: vk::DeviceMemory) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.unmap_memory(memory) }
    }
    unsafe fn destroy_buffer(&self, device: &ash::Device, buffer: vk::Buffer) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_buffer(buffer, None) }
    }
    unsafe fn free_memory(&self, device: &ash::Device, memory: vk::DeviceMemory) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.free_memory(memory, None) }
    }

    unsafe fn create_command_pool(
        &self,
        device: &ash::Device,
        info: &vk::CommandPoolCreateInfo<'_>,
    ) -> Result<vk::CommandPool, vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.create_command_pool(info, None) }
    }
    unsafe fn allocate_command_buffers(
        &self,
        device: &ash::Device,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.allocate_command_buffers(info) }
    }
    unsafe fn create_fence(
        &self,
        device: &ash::Device,
        info: &vk::FenceCreateInfo<'_>,
    ) -> Result<vk::Fence, vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.create_fence(info, None) }
    }
    unsafe fn begin_command_buffer(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        info: &vk::CommandBufferBeginInfo<'_>,
    ) -> Result<(), vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.begin_command_buffer(command_buffer, info) }
    }
    unsafe fn end_command_buffer(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.end_command_buffer(command_buffer) }
    }
    unsafe fn reset_fences(
        &self,
        device: &ash::Device,
        fences: &[vk::Fence],
    ) -> Result<(), vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.reset_fences(fences) }
    }
    unsafe fn queue_submit(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<(), vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.queue_submit(queue, submits, fence) }
    }
    unsafe fn wait_for_fences(
        &self,
        device: &ash::Device,
        fences: &[vk::Fence],
    ) -> Result<(), vk::Result> {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.wait_for_fences(fences, true, u64::MAX) }
    }
    unsafe fn destroy_fence(&self, device: &ash::Device, fence: vk::Fence) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_fence(fence, None) }
    }
    unsafe fn destroy_descriptor_pool(&self, device: &ash::Device, pool: vk::DescriptorPool) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_descriptor_pool(pool, None) }
    }
    unsafe fn destroy_command_pool(&self, device: &ash::Device, pool: vk::CommandPool) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_command_pool(pool, None) }
    }

    unsafe fn destroy_pipeline(&self, device: &ash::Device, pipeline: vk::Pipeline) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_pipeline(pipeline, None) }
    }
    unsafe fn destroy_pipeline_layout(&self, device: &ash::Device, layout: vk::PipelineLayout) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_pipeline_layout(layout, None) }
    }
    unsafe fn destroy_descriptor_set_layout(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
    ) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_descriptor_set_layout(layout, None) }
    }
    unsafe fn destroy_shader_module(&self, device: &ash::Device, module: vk::ShaderModule) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_shader_module(module, None) }
    }
    unsafe fn destroy_device(&self, device: &ash::Device) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { device.destroy_device(None) }
    }
    unsafe fn destroy_instance(&self, instance: &ash::Instance) {
        // SAFETY: a pure forward: the caller upholds this trait method's contract, which is ash's.
        unsafe { instance.destroy_instance(None) }
    }
}
