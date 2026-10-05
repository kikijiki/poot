use crate::*;

/// Shared owner for the complete native parent chain. Every child that can escape `Context` holds one.
///
/// It also carries the device's poison flag (the Vulkan form of ROCm's Card 494/602 rule). Once a
/// submission's fence wait fails, or a submit fails without the spec's guarantee that nothing was
/// queued, the device may still be executing work that references any buffer, pipeline, pool or
/// command buffer of this device. From then on every child's `Drop` leaks its native objects instead of
/// destroying them, the owner leaks the device, instance and loader, and every operation that would
/// touch the device or a mapped buffer returns [`RuntimeError::Poisoned`].
///
/// The owner is `Send + Sync` (every field is: `ash`'s function tables and handles are plain data, the
/// poison flag is atomic, and the queue sits behind a mutex), so every child that holds an
/// `Arc<DeviceOwner>` - a buffer, a pipeline, a recorded graph - can move to another thread (R471-010).
pub(crate) struct DeviceOwner {
    pub(crate) device: ash::Device,
    pub(crate) instance: ash::Instance,
    /// Dropped after `DeviceOwner::drop` (and never, once poisoned), so all loaded functions remain
    /// callable through teardown.
    entry: ManuallyDrop<ash::Entry>,
    pub(crate) dispatch: Arc<dyn ResourceDispatch>,
    /// The one queue every submission goes through. Vulkan requires external synchronization of a
    /// queue, so a submit holds this lock, and a buffer or graph moved to another thread cannot race
    /// the owning context's submits.
    queue: Mutex<vk::Queue>,
    poisoned: AtomicBool,
    /// The validation-layer message counter, present only under `POOT_VULKAN_VALIDATION`; destroyed
    /// before the instance, and a nonzero count fails the teardown.
    validation: Option<Validation>,
    /// Live/peak bytes and allocation count by role (Card 547a): the one memory service's view of
    /// every allocation this device has made.
    pub(crate) memory: poot_runtime_common::MemoryCounters,
}

impl DeviceOwner {
    pub(crate) fn new(
        device: ash::Device,
        instance: ash::Instance,
        entry: ash::Entry,
        queue: vk::Queue,
        dispatch: Arc<dyn ResourceDispatch>,
        validation: Option<Validation>,
    ) -> Self {
        DeviceOwner {
            device,
            instance,
            entry: ManuallyDrop::new(entry),
            dispatch,
            queue: Mutex::new(queue),
            poisoned: AtomicBool::new(false),
            validation,
            memory: poot_runtime_common::MemoryCounters::new(),
        }
    }

    /// Whether a lost submission may still be running on this device (see the type doc).
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Refuse `op` once the device is poisoned.
    pub(crate) fn check_live(&self, op: &'static str) -> Result<(), RuntimeError> {
        if self.is_poisoned() {
            return Err(RuntimeError::Poisoned { op });
        }
        Ok(())
    }

    /// Submit `command_buffer` on the device queue, signalling `fence`, and wait for the fence: the one
    /// synchronous submission path every dispatch and replay uses.
    ///
    /// A submit that fails with `VK_ERROR_OUT_OF_HOST_MEMORY` or `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    /// leaves every referenced object unaffected (the `vkQueueSubmit` spec guarantees it), so it returns
    /// [`RuntimeError::Vk`] and the caller may destroy what it built. Any other submit failure
    /// (`VK_ERROR_DEVICE_LOST` is the spec's signal that it could not keep that guarantee) and any failed
    /// fence wait after a successful submit leave the work possibly pending: the device is poisoned and
    /// the call returns [`RuntimeError::SubmissionLost`], after which the caller must destroy nothing.
    ///
    /// # Safety
    ///
    /// `command_buffer` and `fence` are children of this device; `command_buffer` is executable and not
    /// pending, `fence` is unsignaled and not in use by another submission, and every object the
    /// command buffer references stays alive until this call returns (and forever after, when it
    /// returns `SubmissionLost`).
    pub(crate) unsafe fn submit_and_wait(
        &self,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<(), RuntimeError> {
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        let submitted = {
            let queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            // SAFETY: the caller upholds the command-buffer, fence and lifetime contract above, and
            // the queue (this device's own) is externally synchronized by the lock held here.
            unsafe {
                self.dispatch
                    .queue_submit(&self.device, *queue, &[submit_info], fence)
            }
        };
        match submitted {
            Ok(()) => {}
            Err(
                e @ (vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            ) => return Err(e.into()),
            Err(e) => {
                self.poisoned.store(true, Ordering::Release);
                return Err(RuntimeError::SubmissionLost(e));
            }
        }
        // SAFETY: `fence` is a child of this device that the submission above signals.
        if let Err(e) = unsafe { self.dispatch.wait_for_fences(&self.device, &[fence]) } {
            self.poisoned.store(true, Ordering::Release);
            return Err(RuntimeError::SubmissionLost(e));
        }
        Ok(())
    }
}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        if self.is_poisoned() {
            // Lost work may still be running on the device: destroying it, its instance, or unloading
            // the driver under it is invalid. Leak all three (the loader stays in `entry`).
            return;
        }
        // SAFETY: this is the final shared owner, so all buffer, graph, and cached-pipeline children have
        // already released their `Arc<DeviceOwner>`, and no submission is pending (none was lost). The
        // instance outlives its device, and the validation messenger (if any) goes between the two so
        // it also sees the device's teardown.
        unsafe { self.dispatch.destroy_device(&self.device) };
        if let Some(validation) = &mut self.validation {
            // SAFETY: the instance is destroyed right after this.
            unsafe { validation.destroy() };
        }
        // SAFETY: as above.
        unsafe { self.dispatch.destroy_instance(&self.instance) };
        self.dispatch.release_loader();
        // SAFETY: `entry` is dropped exactly once, here, after the last call through its functions.
        unsafe { ManuallyDrop::drop(&mut self.entry) };
        if let Some(validation) = &self.validation
            && validation.messages() != 0
            && !std::thread::panicking()
        {
            panic!(
                "the Vulkan validation layer reported {} warning(s) or error(s) (printed above)",
                validation.messages()
            );
        }
    }
}

/// The Vulkan objects of one compiled kernel (spec 133 P2 slice 1): shader module, descriptor-set
/// layout, pipeline layout and compute pipeline. Built once per kernel key by [`Context::pipeline`];
/// every recorded dispatch of it only builds a descriptor set. A recorded graph holds shared ownership
/// of every pipeline it references, so a pipeline may outlive the context cache and is destroyed before
/// its shared device owner.
pub struct Pipeline {
    pub(crate) module: vk::ShaderModule,
    pub(crate) dsl: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    /// What `dsl` was built for: the cache key's identity check (a key reused for a kernel that binds
    /// a different number of buffers, declares a trap, or has different code is refused).
    pub(crate) shape: PipelineShape,
    /// Last so pipeline children are destroyed before the shared device can be released.
    pub(crate) owner: Arc<DeviceOwner>,
}

/// The identity a cached pipeline answers for: the full SPIR-V module plus the binding shape its
/// descriptor-set layout was built from. Two kernels with equal shapes build equal pipelines.
pub(crate) struct PipelineShape {
    pub(crate) args: Box<[poot_runtime_common::ArgSchema]>,
    pub(crate) has_trap: bool,
    pub(crate) words: Box<[u32]>,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if self.owner.is_poisoned() {
            return; // A lost submission may still execute this pipeline: leak it.
        }
        // SAFETY: `owner` keeps the device and loader chain alive, and these handles have no other native
        // owner. Recorded graphs retain an `Arc<Pipeline>` for every baked pipeline reference, and
        // the device is not poisoned, so no submission that used this pipeline is still pending.
        unsafe {
            self.owner
                .dispatch
                .destroy_pipeline(&self.owner.device, self.pipeline);
            self.owner
                .dispatch
                .destroy_pipeline_layout(&self.owner.device, self.pipeline_layout);
            self.owner
                .dispatch
                .destroy_descriptor_set_layout(&self.owner.device, self.dsl);
            self.owner
                .dispatch
                .destroy_shader_module(&self.owner.device, self.module);
        }
    }
}
