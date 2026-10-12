use ash::vk;
use poot_target::{BufferStorage, ElementKind};

/// Every failure the raw-Vulkan runtime reports: loader and device setup, dispatch argument validation,
/// pipeline caching, submission loss and host transfers. Typed so a caller matches on the variant, never on
/// message text.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The Vulkan loader library could not be dlopened; the Vulkan SDK is not needed to build, but a
    /// loader must exist at run time.
    #[error("failed to load the Vulkan loader (libvulkan.so.1 or equivalent): {0}")]
    Loader(#[from] ash::LoadingError),
    /// A Vulkan entry point returned a failure code that has no more specific variant.
    #[error("vulkan call failed: {0}")]
    Vk(#[from] vk::Result),
    /// No physical device exposes a queue family with compute support.
    #[error("no Vulkan physical device with a compute queue family was found")]
    NoComputeDevice,
    /// The chosen device has no memory type that is both host-visible and host-coherent, which the
    /// staging and length buffers need.
    #[error("no Vulkan memory type is HOST_VISIBLE | HOST_COHERENT")]
    NoHostVisibleMemory,
    /// A dispatch grid is empty or exceeds `maxComputeWorkGroupCount`: `.0` is the requested grid, `.1`
    /// the x-dimension limit it was checked against.
    #[error("dispatch grid {0:?} is empty or exceeds maxComputeWorkGroupCount (x limit {1})")]
    GridCap([u32; 3], u32),
    /// `vkCreateBuffer` requires a non-zero size (VUID-VkBufferCreateInfo-size-00912).
    #[error("a Vulkan buffer must hold at least one byte")]
    ZeroSizeBuffer,
    /// The length buffer carries each element count as a `u32`, so a larger buffer has no exact bound.
    #[error("a {bytes}-byte buffer has more elements than a u32 element count can bound")]
    BufferTooLarge {
        /// Requested buffer size in bytes.
        bytes: usize,
    },
    /// A dispatch was given a [`poot_runtime_common::CompiledKernel`] compiled for a different backend
    /// (card 608): this runtime only ever dispatches a `SpirvVulkan` kernel.
    #[error(
        "Vulkan dispatch requires a SpirvVulkan-compiled kernel, got one compiled for {actual:?}"
    )]
    WrongKernelTarget {
        /// The target the rejected kernel was compiled for.
        actual: poot_target::TargetId,
    },
    /// The bound buffers did not meet the compiled kernel's argument schema (card 608, SC-002): a typed
    /// release-build error before any submission.
    #[error("kernel argument schema violation: {0}")]
    KernelArgs(#[from] poot_runtime_common::KernelArgError),
    /// A pipeline key was reused for a different kernel: other SPIR-V, argument schema or trap shape.
    #[error("pipeline key {key:?} is cached for a different kernel")]
    PipelineKeyConflict {
        /// The pipeline cache key that was reused.
        key: String,
    },
    /// A dispatch told a kernel a buffer holds more elements than its allocation does.
    #[error(
        "argument {index}: {elems} {element} elements exceed the {byte_len}-byte buffer bound to it"
    )]
    ArgExceedsBuffer {
        /// Position of the argument in the kernel's argument schema.
        index: usize,
        /// Element count the dispatch declared for the argument.
        elems: u32,
        /// Element kind the declared count is measured in.
        element: ElementKind,
        /// Byte size of the buffer actually bound.
        byte_len: usize,
    },
    /// A device-side buffer copy between buffers of different byte sizes.
    #[error("a copy needs equal sizes, got a {src}-byte source and a {dst}-byte destination")]
    CopySizeMismatch {
        /// Byte size of the source buffer.
        src: usize,
        /// Byte size of the destination buffer.
        dst: usize,
    },
    /// A buffer copy whose source and destination are the same buffer, which Vulkan forbids for
    /// overlapping regions.
    #[error("a buffer cannot be copied onto itself")]
    CopyOntoItself,
    /// A kernel's `Assert`/`Unreachable` fired during a one-shot [`Context::dispatch`].
    #[error("kernel {kernel:?} asserted (code {code})")]
    KernelAssertFailed {
        /// Name of the kernel that asserted.
        kernel: String,
        /// Body-local trap id the kernel wrote.
        code: u32,
    },
    /// A kernel declares cooperative matrices, which this device cannot create a pipeline for, or whose
    /// workgroup X is not a subgroup size it supports.
    #[error(
        "a cooperative-matrix kernel with workgroup X {local_size_x:?} cannot run on this device (subgroup sizes {supported:?})"
    )]
    CooperativeMatrixUnsupported {
        /// The kernel's declared workgroup X size, if it declared one.
        local_size_x: Option<u32>,
        /// The device's supported subgroup size range as `(min, max)`, if it reports one.
        supported: Option<(u32, u32)>,
    },
    /// `POOT_VULKAN_VALIDATION` asked for the Khronos validation layer and it cannot be enabled.
    #[error(
        "POOT_VULKAN_VALIDATION is set but the validation layer or VK_EXT_debug_utils is not available"
    )]
    ValidationUnavailable,
    /// The queue family writes no device timestamps (`timestampValidBits` is 0, or the period is 0).
    #[error("this Vulkan queue family writes no device timestamps")]
    TimestampsUnavailable,
    /// Buffers, graphs and pipelines are children of one `VkDevice`; a handle from another
    /// [`Context`] cannot be bound or recorded here.
    #[error("a {0} belongs to a different Vulkan context")]
    ForeignObject(&'static str),
    /// A submission may still be running after this error: its fence wait failed, or its submit failed
    /// without the guarantee that nothing was queued. The device is poisoned (see
    /// [`RuntimeError::Poisoned`]) and everything it could reference is leaked, not destroyed.
    #[error("a submitted command buffer was lost ({0}); the Vulkan device is poisoned")]
    SubmissionLost(vk::Result),
    /// A call was refused because an earlier submission was lost (see [`RuntimeError::SubmissionLost`]).
    #[error("{op} refused: an earlier submission was lost, so this Vulkan device is poisoned")]
    Poisoned {
        /// The refused operation, for the message.
        op: &'static str,
    },
    /// An f32 element count whose byte size does not fit `usize`.
    #[error("a {elems}-element f32 buffer overflows usize bytes")]
    ElementCountOverflow {
        /// The element count that overflowed.
        elems: usize,
    },
    /// A host write larger than the destination buffer.
    #[error("a {requested}-byte write exceeds the buffer's {capacity}-byte capacity")]
    WriteCapacity {
        /// Bytes the write asked to store.
        requested: usize,
        /// The buffer's capacity in bytes.
        capacity: usize,
    },
    /// A typed transfer was requested against a differently represented allocation (card 527: checked
    /// against the buffer's full storage contract - element kind, logical dtype and layout - not just
    /// element kind).
    #[error("{op} requires {expected} storage, but the buffer stores {actual}")]
    RepresentationMismatch {
        /// The transfer entry point that refused, for the message.
        op: &'static str,
        /// The storage contract the transfer needs.
        expected: BufferStorage,
        /// The storage contract the buffer actually carries.
        actual: BufferStorage,
    },
}
