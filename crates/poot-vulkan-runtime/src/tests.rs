use std::ffi::{CStr, c_char};
use std::mem::size_of;
use std::ptr;
use std::sync::atomic::AtomicU64;

use ash::vk::Handle;

use super::*;
use crate::graph::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FailPoint {
    CreateBuffer,
    AllocateMemory,
    BindBufferMemory,
    MapMemory,
    CreateCommandPool,
    AllocateCommandBuffer,
    CreateFence,
    BeginCommandBuffer,
    /// `vkQueueSubmit` fails with out-of-memory: the spec guarantees nothing was queued.
    QueueSubmitOutOfMemory,
    /// `vkQueueSubmit` fails with device loss: no guarantee that nothing was queued.
    QueueSubmitDeviceLost,
    /// `vkWaitForFences` fails after a successful submit.
    WaitForFences,
}

struct MockDispatch {
    events: Mutex<Vec<&'static str>>,
    fail_at: Mutex<Option<FailPoint>>,
    next_handle: AtomicU64,
    backing: Mutex<Box<[f32]>>,
}

impl MockDispatch {
    fn new(fail_at: Option<FailPoint>) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            fail_at: Mutex::new(fail_at),
            next_handle: AtomicU64::new(10),
            backing: Mutex::new(vec![0.0; 1 << 18].into_boxed_slice()),
        }
    }

    fn event(&self, name: &'static str) {
        self.events.lock().unwrap().push(name);
    }

    fn fail_at(&self) -> Option<FailPoint> {
        *self.fail_at.lock().unwrap()
    }

    fn set_fail_at(&self, point: Option<FailPoint>) {
        *self.fail_at.lock().unwrap() = point;
    }

    fn result<T: Handle>(&self, point: FailPoint) -> Result<T, ash::vk::Result> {
        if self.fail_at() == Some(point) {
            return Err(ash::vk::Result::ERROR_UNKNOWN);
        }
        Ok(T::from_raw(
            self.next_handle.fetch_add(1, Ordering::Relaxed),
        ))
    }

    fn events(&self) -> Vec<&'static str> {
        self.events.lock().unwrap().clone()
    }
}

impl ResourceDispatch for MockDispatch {
    unsafe fn create_buffer(
        &self,
        _device: &ash::Device,
        info: &ash::vk::BufferCreateInfo<'_>,
    ) -> Result<ash::vk::Buffer, ash::vk::Result> {
        self.event("create_buffer");
        // A zero size violates VUID-VkBufferCreateInfo-size-00912; the mock fails the call the way a
        // validating driver reports invalid usage.
        if info.size == 0 {
            return Err(ash::vk::Result::ERROR_VALIDATION_FAILED_EXT);
        }
        self.result(FailPoint::CreateBuffer)
    }

    unsafe fn get_buffer_memory_requirements(
        &self,
        _device: &ash::Device,
        _buffer: ash::vk::Buffer,
    ) -> ash::vk::MemoryRequirements {
        self.event("get_buffer_memory_requirements");
        ash::vk::MemoryRequirements {
            size: 64,
            alignment: 4,
            memory_type_bits: 1,
        }
    }

    unsafe fn allocate_memory(
        &self,
        _device: &ash::Device,
        _info: &ash::vk::MemoryAllocateInfo<'_>,
    ) -> Result<ash::vk::DeviceMemory, ash::vk::Result> {
        self.event("allocate_memory");
        self.result(FailPoint::AllocateMemory)
    }

    unsafe fn bind_buffer_memory(
        &self,
        _device: &ash::Device,
        _buffer: ash::vk::Buffer,
        _memory: ash::vk::DeviceMemory,
    ) -> Result<(), ash::vk::Result> {
        self.event("bind_buffer_memory");
        if self.fail_at() == Some(FailPoint::BindBufferMemory) {
            Err(ash::vk::Result::ERROR_UNKNOWN)
        } else {
            Ok(())
        }
    }

    unsafe fn map_memory(
        &self,
        _device: &ash::Device,
        _memory: ash::vk::DeviceMemory,
    ) -> Result<*mut c_void, ash::vk::Result> {
        self.event("map_memory");
        if self.fail_at() == Some(FailPoint::MapMemory) {
            return Err(ash::vk::Result::ERROR_UNKNOWN);
        }
        // The pointer stays valid after the lock is released: the boxed backing is never resized.
        Ok(self.backing.lock().unwrap().as_mut_ptr().cast())
    }

    unsafe fn unmap_memory(&self, _device: &ash::Device, _memory: ash::vk::DeviceMemory) {
        self.event("unmap_memory");
    }

    unsafe fn destroy_buffer(&self, _device: &ash::Device, _buffer: ash::vk::Buffer) {
        self.event("destroy_buffer");
    }

    unsafe fn free_memory(&self, _device: &ash::Device, _memory: ash::vk::DeviceMemory) {
        self.event("free_memory");
    }

    unsafe fn create_command_pool(
        &self,
        _device: &ash::Device,
        _info: &ash::vk::CommandPoolCreateInfo<'_>,
    ) -> Result<ash::vk::CommandPool, ash::vk::Result> {
        self.event("create_command_pool");
        self.result(FailPoint::CreateCommandPool)
    }

    unsafe fn allocate_command_buffers(
        &self,
        _device: &ash::Device,
        _info: &ash::vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<ash::vk::CommandBuffer>, ash::vk::Result> {
        self.event("allocate_command_buffers");
        self.result(FailPoint::AllocateCommandBuffer)
            .map(|command_buffer| vec![command_buffer])
    }

    unsafe fn create_fence(
        &self,
        _device: &ash::Device,
        _info: &ash::vk::FenceCreateInfo<'_>,
    ) -> Result<ash::vk::Fence, ash::vk::Result> {
        self.event("create_fence");
        self.result(FailPoint::CreateFence)
    }

    unsafe fn begin_command_buffer(
        &self,
        _device: &ash::Device,
        _command_buffer: ash::vk::CommandBuffer,
        _info: &ash::vk::CommandBufferBeginInfo<'_>,
    ) -> Result<(), ash::vk::Result> {
        self.event("begin_command_buffer");
        if self.fail_at() == Some(FailPoint::BeginCommandBuffer) {
            Err(ash::vk::Result::ERROR_UNKNOWN)
        } else {
            Ok(())
        }
    }

    unsafe fn end_command_buffer(
        &self,
        _device: &ash::Device,
        _command_buffer: ash::vk::CommandBuffer,
    ) -> Result<(), ash::vk::Result> {
        self.event("end_command_buffer");
        Ok(())
    }

    unsafe fn reset_fences(
        &self,
        _device: &ash::Device,
        _fences: &[ash::vk::Fence],
    ) -> Result<(), ash::vk::Result> {
        self.event("reset_fences");
        Ok(())
    }

    unsafe fn queue_submit(
        &self,
        _device: &ash::Device,
        _queue: ash::vk::Queue,
        _submits: &[ash::vk::SubmitInfo<'_>],
        _fence: ash::vk::Fence,
    ) -> Result<(), ash::vk::Result> {
        self.event("queue_submit");
        match self.fail_at() {
            Some(FailPoint::QueueSubmitOutOfMemory) => {
                Err(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
            }
            Some(FailPoint::QueueSubmitDeviceLost) => Err(ash::vk::Result::ERROR_DEVICE_LOST),
            _ => Ok(()),
        }
    }

    unsafe fn wait_for_fences(
        &self,
        _device: &ash::Device,
        _fences: &[ash::vk::Fence],
    ) -> Result<(), ash::vk::Result> {
        self.event("wait_for_fences");
        if self.fail_at() == Some(FailPoint::WaitForFences) {
            Err(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        } else {
            Ok(())
        }
    }

    unsafe fn destroy_fence(&self, _device: &ash::Device, _fence: ash::vk::Fence) {
        self.event("destroy_fence");
    }

    unsafe fn destroy_descriptor_pool(
        &self,
        _device: &ash::Device,
        _pool: ash::vk::DescriptorPool,
    ) {
        self.event("destroy_descriptor_pool");
    }

    unsafe fn destroy_command_pool(&self, _device: &ash::Device, _pool: ash::vk::CommandPool) {
        self.event("destroy_command_pool");
    }

    unsafe fn destroy_pipeline(&self, _device: &ash::Device, _pipeline: ash::vk::Pipeline) {
        self.event("destroy_pipeline");
    }

    unsafe fn destroy_pipeline_layout(
        &self,
        _device: &ash::Device,
        _layout: ash::vk::PipelineLayout,
    ) {
        self.event("destroy_pipeline_layout");
    }

    unsafe fn destroy_descriptor_set_layout(
        &self,
        _device: &ash::Device,
        _layout: ash::vk::DescriptorSetLayout,
    ) {
        self.event("destroy_descriptor_set_layout");
    }

    unsafe fn destroy_shader_module(&self, _device: &ash::Device, _module: ash::vk::ShaderModule) {
        self.event("destroy_shader_module");
    }

    unsafe fn destroy_device(&self, _device: &ash::Device) {
        self.event("destroy_device");
    }

    unsafe fn destroy_instance(&self, _instance: &ash::Instance) {
        self.event("destroy_instance");
    }

    fn release_loader(&self) {
        self.event("release_loader");
    }
}

unsafe extern "system" fn no_instance_proc_addr(
    _instance: ash::vk::Instance,
    _name: *const c_char,
) -> ash::vk::PFN_vkVoidFunction {
    None
}

fn mock_context(fail_at: Option<FailPoint>) -> (Context, Arc<MockDispatch>) {
    let mock = Arc::new(MockDispatch::new(fail_at));
    let static_fn = ash::StaticFn {
        get_instance_proc_addr: no_instance_proc_addr,
    };
    // SAFETY: `no_instance_proc_addr` has the loader signature and resolves nothing. No function of
    // this entry, instance or device is ever called: every native call goes through the mock dispatch.
    let entry = unsafe { ash::Entry::from_static_fn(static_fn) };
    // SAFETY: as above; the null function pointers are never called.
    let instance = unsafe {
        ash::Instance::load_with(|_name: &CStr| ptr::null(), ash::vk::Instance::from_raw(1))
    };
    // SAFETY: as above; the null function pointers are never called.
    let device =
        unsafe { ash::Device::load_with(|_name: &CStr| ptr::null(), ash::vk::Device::from_raw(2)) };
    let dispatch: Arc<dyn ResourceDispatch> = mock.clone();
    let owner = Arc::new(DeviceOwner::new(
        device,
        instance,
        entry,
        ash::vk::Queue::from_raw(3),
        dispatch,
        None,
    ));
    (
        Context {
            instance: owner.instance.clone(),
            device: owner.device.clone(),
            physical_device: ash::vk::PhysicalDevice::null(),
            queue_family_index: 0,
            timestamp_period_ns: 0.0,
            timestamp_valid_mask: 0,
            coopmat_subgroup_sizes: None,
            min_storage_offset_alignment: 256,
            max_workgroup_count: [65_535, 65_535, 65_535],
            device_caps: poot_target::DeviceCaps {
                max_buffer_bytes: u64::MAX,
                lds_bytes: 0,
                max_workgroup_size: [1, 1, 1],
                max_workgroup_invocations: 1,
                subgroup: poot_target::Queried::Unknown,
                max_dispatch_work: poot_target::Queried::Unknown,
                max_grid: [65_535, 65_535, 65_535],
                watchdog_budget: None,
                tensor_core: poot_target::TensorCoreSupport::UnknownNotExposedByApi,
                known_miscompiles: poot_target::KnownMiscompiles::default(),
                compute_units: poot_target::STRIX_HALO_COMPUTE_UNITS,
            },
            pipeline_cache: HashMap::new(),
            pipeline_builds: 0,
            graphs_recorded: AtomicUsize::new(0),
            owner,
        },
        mock,
    )
}

/// A pipeline of fake handles for a kernel of `data_buffers` f32 arguments, owned by `context`'s device.
fn mock_pipeline(context: &Context, data_buffers: usize) -> Arc<Pipeline> {
    Arc::new(Pipeline {
        module: ash::vk::ShaderModule::from_raw(20),
        dsl: ash::vk::DescriptorSetLayout::from_raw(21),
        pipeline_layout: ash::vk::PipelineLayout::from_raw(22),
        pipeline: ash::vk::Pipeline::from_raw(23),
        shape: PipelineShape::of(&test_kernel(data_buffers)).expect("a SPIR-V test kernel"),
        owner: Arc::clone(&context.owner),
    })
}

fn mock_buffer(context: &Context, byte_len: usize) -> Result<DeviceBuffer, RuntimeError> {
    mock_buffer_with_storage(context, byte_len, BufferStorage::f32())
}

/// `data` as little-endian bytes, for [`DeviceBuffer::write_bytes`] (Card 547a: the storage-typed
/// primitive replacing the deleted per-dtype `write_f32`).
fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// [`DeviceBuffer::read_bytes`] decoded as little-endian f32, for the deleted per-dtype `read_f32`'s
/// test callers.
fn read_f32_bytes(buffer: &DeviceBuffer) -> Result<Vec<f32>, RuntimeError> {
    let mut bytes = vec![0u8; buffer.elem_count() as usize * 4];
    buffer.read_bytes(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn mock_buffer_with_storage(
    context: &Context,
    byte_len: usize,
    storage: BufferStorage,
) -> Result<DeviceBuffer, RuntimeError> {
    let resources = allocate_buffer_resources(&context.owner, byte_len, |_reqs| Some(0))?;
    Ok(DeviceBuffer::from_resources(
        resources,
        byte_len,
        storage,
        BufferRole::Activation,
        &context.owner,
    ))
}

/// Card 527 (R484-001, R471-009): before this card a Vulkan buffer carried no element kind at all, so
/// any dtype bound cleanly. `bf16_packed()` canonicalizes on `ElementKind::I32` (review F2, the word a
/// packed-BF16 upload would use if this runtime ever packed); a packed-BF16 value and a plain dense I32
/// value share native element kind, so only the full `BufferStorage` comparison (not element kind
/// alone) tells them apart. `poot-vulkan-runtime` has no `*-gpu` binder crate yet (raw Vulkan's
/// executor is a later card, D4), so this calls `check_storage` directly on a real allocation.
///
/// Card 547a: `check_storage` is `#[cfg(test)]` (Card 547a: `read_bytes`/`write_bytes`, this
/// crate's own storage-typed primitives, are deliberately dtype-agnostic and never call it), so this is
/// evidence that the full storage contract check itself is correct, not evidence that a release bind
/// path enforces it - there is no such path on this backend yet. Wire `check_storage` into a real
/// bind-time entry point when D4 lands the Vulkan binder.
#[test]
fn bf16_packed_word_rejected_by_dense_i32_bind() {
    let (context, _mock) = mock_context(None);
    let buffer =
        mock_buffer_with_storage(&context, 4 * size_of::<i32>(), BufferStorage::bf16_packed())
            .expect("mock allocation");

    let error = buffer
        .check_storage("test_bind", BufferStorage::i32())
        .expect_err("packed bf16 lanes must not bind as dense I32");
    let RuntimeError::RepresentationMismatch {
        op: "test_bind",
        expected,
        actual,
    } = error
    else {
        panic!("expected RepresentationMismatch, got {error:?}");
    };
    assert_eq!(expected, BufferStorage::i32());
    assert_eq!(actual, BufferStorage::bf16_packed());
    assert_eq!(actual.element(), expected.element(), "same native element");
}

/// Card 527 (R471-009): a buffer allocated dense F32 bound as BF16 - this runtime carried no element
/// kind at all before this card, so any dtype was previously accepted.
#[test]
fn dense_f32_buffer_rejects_a_bf16_bind() {
    let (context, _mock) = mock_context(None);
    let buffer = mock_buffer(&context, 2 * size_of::<f32>()).expect("mock allocation");

    let error = buffer
        .check_storage("test_bind", BufferStorage::bf16())
        .expect_err("a dense F32 buffer must not bind as BF16");
    let RuntimeError::RepresentationMismatch {
        op: "test_bind", ..
    } = error
    else {
        panic!("expected RepresentationMismatch, got {error:?}");
    };
}

#[test]
fn buffer_and_clone_outlive_context_and_readback_does_not_alias_writes() {
    let (context, mock) = mock_context(None);
    let buffer = mock_buffer(&context, 3 * size_of::<f32>()).expect("mock allocation");
    buffer
        .write_bytes(&f32_bytes(&[1.0, 2.0, 3.0]))
        .expect("write");
    let clone = buffer.clone();
    let snapshot: Vec<f32> = read_f32_bytes(&buffer).expect("read back");

    drop(context);
    assert!(!mock.events().contains(&"destroy_device"));

    clone
        .write_bytes(&f32_bytes(&[4.0, 5.0, 6.0]))
        .expect("write");
    assert_eq!(snapshot, [1.0, 2.0, 3.0]);
    assert_eq!(read_f32_bytes(&buffer).expect("read back"), [4.0, 5.0, 6.0]);

    drop(buffer);
    assert_eq!(
        mock.events()
            .iter()
            .filter(|&&e| e == "unmap_memory")
            .count(),
        0
    );
    drop(clone);
    assert_eq!(
        &mock.events()[mock.events().len() - 6..],
        [
            "unmap_memory",
            "destroy_buffer",
            "free_memory",
            "destroy_device",
            "destroy_instance",
            "release_loader",
        ]
    );
}

#[test]
fn recorded_graph_directly_retains_owner_and_replays_after_context_drop() {
    let (context, mock) = mock_context(None);
    let graph = context.begin_graph(GraphTiming::Off).expect("mock graph");
    let graph = context.end_graph(graph).expect("end mock graph");

    drop(context);
    assert!(!mock.events().contains(&"destroy_device"));
    graph.replay().expect("synchronous mock replay");
    assert_eq!(
        &mock.events()[mock.events().len() - 3..],
        ["reset_fences", "queue_submit", "wait_for_fences"]
    );

    drop(graph);
    assert_eq!(
        &mock.events()[mock.events().len() - 5..],
        [
            "destroy_fence",
            "destroy_command_pool",
            "destroy_device",
            "destroy_instance",
            "release_loader",
        ]
    );
}

#[test]
fn recorded_graph_retains_buffers_and_pipelines_after_context_drop() {
    let (mut context, mock) = mock_context(None);
    let buffer = mock_buffer(&context, size_of::<f32>()).expect("mock allocation");
    let mut graph = context.begin_graph(GraphTiming::Off).expect("mock graph");
    graph.core.held.push(buffer);
    let pipeline = mock_pipeline(&context, 1);
    graph.core.pipelines.push(Arc::clone(&pipeline));
    context.pipeline_cache.insert("mock".into(), pipeline);
    let graph = context.end_graph(graph).expect("end mock graph");

    drop(context);
    assert!(!mock.events().contains(&"destroy_pipeline"));
    assert!(!mock.events().contains(&"destroy_device"));

    graph.replay().expect("synchronous mock replay");
    assert_eq!(
        &mock.events()[mock.events().len() - 3..],
        ["reset_fences", "queue_submit", "wait_for_fences"]
    );
    drop(graph);
    assert_eq!(
        &mock.events()[mock.events().len() - 12..],
        [
            "destroy_fence",
            "destroy_command_pool",
            "unmap_memory",
            "destroy_buffer",
            "free_memory",
            "destroy_pipeline",
            "destroy_pipeline_layout",
            "destroy_descriptor_set_layout",
            "destroy_shader_module",
            "destroy_device",
            "destroy_instance",
            "release_loader",
        ]
    );
}

#[test]
fn buffer_partial_failures_clean_up_exactly_acquired_resources() {
    let cases: &[(FailPoint, &[&str])] = &[
        (FailPoint::CreateBuffer, &["create_buffer"]),
        (
            FailPoint::AllocateMemory,
            &[
                "create_buffer",
                "get_buffer_memory_requirements",
                "allocate_memory",
                "destroy_buffer",
            ],
        ),
        (
            FailPoint::BindBufferMemory,
            &[
                "create_buffer",
                "get_buffer_memory_requirements",
                "allocate_memory",
                "bind_buffer_memory",
                "destroy_buffer",
                "free_memory",
            ],
        ),
        (
            FailPoint::MapMemory,
            &[
                "create_buffer",
                "get_buffer_memory_requirements",
                "allocate_memory",
                "bind_buffer_memory",
                "map_memory",
                "destroy_buffer",
                "free_memory",
            ],
        ),
    ];
    for &(fail_at, expected) in cases {
        let (context, mock) = mock_context(Some(fail_at));
        assert!(mock_buffer(&context, size_of::<f32>()).is_err());
        assert_eq!(mock.events(), expected, "failure at {fail_at:?}");
    }

    let (context, mock) = mock_context(None);
    assert!(allocate_buffer_resources(&context.owner, size_of::<f32>(), |_reqs| None).is_err());
    assert_eq!(
        mock.events(),
        [
            "create_buffer",
            "get_buffer_memory_requirements",
            "destroy_buffer"
        ]
    );
}

#[test]
fn graph_partial_failures_clean_up_exactly_acquired_resources() {
    let cases: &[(FailPoint, &[&str])] = &[
        (FailPoint::CreateCommandPool, &["create_command_pool"]),
        (
            FailPoint::AllocateCommandBuffer,
            &[
                "create_command_pool",
                "allocate_command_buffers",
                "destroy_command_pool",
            ],
        ),
        (
            FailPoint::CreateFence,
            &[
                "create_command_pool",
                "allocate_command_buffers",
                "create_fence",
                "destroy_command_pool",
            ],
        ),
        (
            FailPoint::BeginCommandBuffer,
            &[
                "create_command_pool",
                "allocate_command_buffers",
                "create_fence",
                "begin_command_buffer",
                "destroy_fence",
                "destroy_command_pool",
            ],
        ),
    ];
    for &(fail_at, expected) in cases {
        let (context, mock) = mock_context(Some(fail_at));
        assert!(context.begin_graph(GraphTiming::Off).is_err());
        assert_eq!(mock.events(), expected, "failure at {fail_at:?}");
    }
}

/// A recorded graph holding one buffer, finished and ready to replay, plus a second handle to the buffer.
fn mock_replayable_graph(
    fail_at: FailPoint,
) -> (Context, Arc<MockDispatch>, VulkanGraph, DeviceBuffer) {
    let (context, mock) = mock_context(None);
    let buffer = mock_buffer(&context, size_of::<f32>()).expect("mock allocation");
    let mut graph = context.begin_graph(GraphTiming::Off).expect("mock graph");
    graph.core.held.push(buffer.clone());
    let graph = context.end_graph(graph).expect("end mock graph");
    mock.set_fail_at(Some(fail_at));
    (context, mock, graph, buffer)
}

const DESTROY_EVENTS: &[&str] = &[
    "unmap_memory",
    "destroy_buffer",
    "free_memory",
    "destroy_fence",
    "destroy_descriptor_pool",
    "destroy_command_pool",
    "destroy_device",
    "destroy_instance",
    "release_loader",
];

/// M2 (Card 516b review): once a submitted replay's fence wait fails, the command buffer may still be
/// pending. Nothing it references is destroyed, the graph cannot be resubmitted, and mapped memory
/// cannot be read or written.
#[test]
fn failed_fence_wait_after_submit_poisons_and_leaks_everything() {
    let (context, mock, graph, buffer) = mock_replayable_graph(FailPoint::WaitForFences);
    let before = mock.events().len();
    let result = graph.replay();
    assert!(
        matches!(
            result,
            Err(RuntimeError::SubmissionLost(
                ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY
            ))
        ),
        "got {result:?}"
    );
    assert_eq!(
        &mock.events()[before..],
        ["reset_fences", "queue_submit", "wait_for_fences"]
    );

    let after_loss = mock.events().len();
    assert!(matches!(
        graph.replay(),
        Err(RuntimeError::Poisoned { op: "replay" })
    ));
    assert!(matches!(
        buffer.read_bytes(&mut [0u8; 4]),
        Err(RuntimeError::Poisoned { op: "read_bytes" })
    ));
    assert!(matches!(
        buffer.write_bytes(&[0u8; 4]),
        Err(RuntimeError::Poisoned { op: "write_bytes" })
    ));
    assert!(matches!(
        context.begin_graph(GraphTiming::Off),
        Err(RuntimeError::Poisoned { op: "begin_graph" })
    ));
    assert_eq!(
        mock.events().len(),
        after_loss,
        "a poisoned device makes no further native call"
    );

    drop(graph);
    drop(buffer);
    drop(context);
    let teardown = &mock.events()[after_loss..];
    assert!(
        teardown.iter().all(|event| !DESTROY_EVENTS.contains(event)),
        "nothing a lost submission could reference may be destroyed; got {teardown:?}"
    );
}

#[test]
fn submit_lost_to_device_loss_poisons() {
    let (_context, mock, graph, buffer) = mock_replayable_graph(FailPoint::QueueSubmitDeviceLost);
    let before = mock.events().len();
    assert!(matches!(
        graph.replay(),
        Err(RuntimeError::SubmissionLost(
            ash::vk::Result::ERROR_DEVICE_LOST
        ))
    ));
    assert_eq!(&mock.events()[before..], ["reset_fences", "queue_submit"]);
    assert!(matches!(
        buffer.read_bytes(&mut [0u8; 4]),
        Err(RuntimeError::Poisoned { .. })
    ));
}

/// An out-of-memory submit leaves nothing queued (the spec guarantees it): a typed error, no poison,
/// and the graph replays and tears down normally afterwards.
#[test]
fn submit_refused_for_memory_is_retired_not_poisoned() {
    let (context, mock, graph, buffer) = mock_replayable_graph(FailPoint::QueueSubmitOutOfMemory);
    assert!(matches!(
        graph.replay(),
        Err(RuntimeError::Vk(
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        ))
    ));
    mock.set_fail_at(None);
    graph
        .replay()
        .expect("the retired submission left the graph replayable");
    assert!(buffer.read_bytes(&mut [0u8; 4]).is_ok());
    drop((graph, buffer, context));
    assert_eq!(
        &mock.events()[mock.events().len() - 8..],
        [
            "destroy_fence",
            "destroy_command_pool",
            "unmap_memory",
            "destroy_buffer",
            "free_memory",
            "destroy_device",
            "destroy_instance",
            "release_loader",
        ]
    );
}

/// SC-003: a zero-size buffer request is refused with a typed error before `vkCreateBuffer`.
#[test]
fn zero_size_buffer_is_refused_before_the_driver_call() {
    let (context, mock) = mock_context(None);
    let result = mock_buffer(&context, 0);
    assert!(
        matches!(result, Err(RuntimeError::ZeroSizeBuffer)),
        "a zero-size buffer must be refused with ZeroSizeBuffer, got {:?}",
        result.as_ref().err()
    );
    assert_eq!(
        mock.events(),
        Vec::<&str>::new(),
        "no driver call may be made"
    );
}

#[test]
fn buffer_whose_element_count_exceeds_u32_is_refused() {
    let (context, mock) = mock_context(None);
    let bytes = (u32::MAX as usize + 1) * 4;
    let result = mock_buffer(&context, bytes);
    assert!(
        matches!(result, Err(RuntimeError::BufferTooLarge { bytes: b }) if b == bytes),
        "got {:?}",
        result.as_ref().err()
    );
    assert_eq!(mock.events(), Vec::<&str>::new());
}

fn test_kernel(data_buffers: usize) -> CompiledKernel {
    test_kernel_with_code(data_buffers, 0x0723_0203)
}

fn test_kernel_with_code(data_buffers: usize, first_word: u32) -> CompiledKernel {
    // Never reaches the driver: every test below is refused before a pipeline is built.
    let args: Vec<poot_runtime_common::ArgSchema> = (0..data_buffers)
        .map(|_| {
            poot_runtime_common::ArgSchema::new(
                poot_target::ElementKind::F32,
                poot_runtime_common::ArgAccess::Read,
            )
        })
        .collect();
    // SAFETY: test-only handle whose code is never dispatched to a real driver (every test below is
    // refused by the schema/foreign-object check first).
    unsafe {
        CompiledKernel::new(
            poot_target::Backend::SpirvVulkan,
            "main",
            poot_runtime_common::KernelCode::SpirvWords(Box::new([first_word])),
            args,
            false,
        )
    }
}

#[test]
fn dispatch_refuses_a_buffer_count_other_than_the_kernels() {
    let (context, _mock) = mock_context(None);
    let buffer = mock_buffer(&context, 4 * size_of::<f32>()).expect("mock allocation");
    let result = context.dispatch(&test_kernel(2), [64, 1, 1], [4, 1, 1], &[&buffer]);
    assert!(
        matches!(
            result,
            Err(RuntimeError::KernelArgs(
                poot_runtime_common::KernelArgError::Count {
                    expected: 2,
                    actual: 1
                }
            ))
        ),
        "got {result:?}"
    );
}

#[test]
fn dispatch_and_recording_refuse_objects_of_another_context() {
    let (context, _mock) = mock_context(None);
    let (other, _other_mock) = mock_context(None);
    let foreign = mock_buffer(&other, 4 * size_of::<f32>()).expect("mock allocation");
    let kernel = test_kernel(1);
    let result = context.dispatch(&kernel, [64, 1, 1], [4, 1, 1], &[&foreign]);
    assert!(
        matches!(result, Err(RuntimeError::ForeignObject("buffer"))),
        "got {result:?}"
    );

    let mut foreign_graph = other.begin_graph(GraphTiming::Off).expect("mock graph");
    let own = mock_buffer(&context, 4 * size_of::<f32>()).expect("mock allocation");
    let pipeline = mock_pipeline(&context, 1);
    let args = [Binding {
        buffer: &own,
        elems: 4,
    }];
    let result = context.record_dispatch(
        &mut foreign_graph,
        "k",
        &pipeline,
        [64, 1, 1],
        [4, 1, 1],
        &args,
    );
    assert!(
        matches!(result, Err(RuntimeError::ForeignObject("graph"))),
        "got {result:?}"
    );
    let mut own_graph = context.begin_graph(GraphTiming::Off).expect("mock graph");
    let foreign_pipeline = mock_pipeline(&other, 1);
    let result = context.record_dispatch(
        &mut own_graph,
        "k",
        &foreign_pipeline,
        [64, 1, 1],
        [4, 1, 1],
        &args,
    );
    assert!(
        matches!(result, Err(RuntimeError::ForeignObject("pipeline"))),
        "got {result:?}"
    );
    let result = context.record_copy(&mut own_graph, &own, &foreign);
    assert!(
        matches!(result, Err(RuntimeError::ForeignObject("buffer"))),
        "got {result:?}"
    );
    let result = context.end_graph(foreign_graph);
    assert!(
        matches!(result, Err(RuntimeError::ForeignObject("graph"))),
        "got {:?}",
        result.as_ref().err()
    );
    assert_eq!(context.graphs_recorded(), 0);
}

/// Review of Card 553: a dispatch told a buffer holds more elements than its allocation does is refused
/// before anything is recorded, whatever the buffer's own capacity is.
#[test]
fn a_binding_larger_than_its_buffer_is_refused() {
    let (context, _mock) = mock_context(None);
    let mut graph = context.begin_graph(GraphTiming::Off).expect("mock graph");
    let buffer = mock_buffer(&context, 4 * size_of::<f32>()).expect("mock allocation");
    let pipeline = mock_pipeline(&context, 1);
    let result = context.record_dispatch(
        &mut graph,
        "k",
        &pipeline,
        [64, 1, 1],
        [5, 1, 1],
        &[Binding {
            buffer: &buffer,
            elems: 5,
        }],
    );
    assert!(
        matches!(
            result,
            Err(RuntimeError::ArgExceedsBuffer {
                index: 0,
                elems: 5,
                byte_len: 16,
                ..
            })
        ),
        "got {result:?}"
    );
    assert_eq!(graph.core.dispatch_count, 0);
}

/// SC-001: a grid over the device's `maxComputeWorkGroupCount` is refused with the typed `GridCap`, before
/// anything is recorded, never folded onto y (a kernel planned as a 1-D launch reads only x, so a fold
/// computes wrong results silently). The mock context's limit is 65535 per axis.
///
/// Mutation: restore the fold in `resolve_groups` (`poot_runtime_common::fold_grid` over the limits); the
/// 66560-group launch becomes `[65535, 2, 1]`, the dispatch is recorded, and this row goes red (on a
/// real device it was observed diverging from the oracle by 5.769e-1 on a flash prefill).
#[test]
fn a_grid_over_the_device_limit_is_refused_not_folded() {
    let (context, _mock) = mock_context(None);
    let mut graph = context.begin_graph(GraphTiming::Off).expect("mock graph");
    let buffer = mock_buffer(&context, 4 * size_of::<f32>()).expect("mock allocation");
    let pipeline = mock_pipeline(&context, 1);
    let args = [Binding {
        buffer: &buffer,
        elems: 4,
    }];
    let result = context.record_dispatch(
        &mut graph,
        "k",
        &pipeline,
        [64, 1, 1],
        [66_560 * 64, 1, 1],
        &args,
    );
    assert!(
        matches!(result, Err(RuntimeError::GridCap([66_560, 1, 1], 65_535))),
        "got {result:?}"
    );
    assert_eq!(graph.core.dispatch_count, 0);
}

#[test]
fn a_cached_key_answers_for_one_kernel_only() {
    let (mut context, _mock) = mock_context(None);
    context
        .pipeline_cache
        .insert("k".into(), mock_pipeline(&context, 1));
    let result = context.pipeline("k", &test_kernel(3));
    assert!(
        matches!(&result, Err(RuntimeError::PipelineKeyConflict { key }) if key == "k"),
        "a key reused for a kernel with another argument schema is refused, got {:?}",
        result.as_ref().err()
    );
    let other_code = test_kernel_with_code(1, 0x0723_0204);
    let result = context.pipeline("k", &other_code);
    assert!(
        matches!(&result, Err(RuntimeError::PipelineKeyConflict { .. })),
        "a key reused for a kernel with other code is refused, got {:?}",
        result.as_ref().err()
    );
    assert!(context.pipeline("k", &test_kernel(1)).is_ok());
    assert_eq!(context.pipeline_builds(), 0);
}

#[test]
fn vulkan_context_open_is_required_by_its_own_variable_only() {
    use poot_runtime_common::DeviceBackend;
    let fails_open = |variable: &'static str| {
        std::panic::catch_unwind(|| {
            require_gpu_check_with(|name| (name == variable).then(|| "1".into()), "no device")
        })
        .is_err()
    };
    assert!(fails_open(DeviceBackend::Vulkan.variable()));
    assert!(
        !fails_open("POOT_REQUIRE_GPU"),
        "the retired all-backend switch must not require Vulkan"
    );
    for other in DeviceBackend::ALL
        .into_iter()
        .filter(|other| *other != DeviceBackend::Vulkan)
    {
        assert!(
            !fails_open(other.variable()),
            "{other:?} must not require Vulkan"
        );
    }
}
