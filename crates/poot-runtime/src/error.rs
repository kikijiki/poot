use crate::BufferStorage;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// A dispatch was given a [`poot_runtime_common::CompiledKernel`] compiled for a different backend
    /// (card 608): wgpu only ever dispatches a `SpirvVulkan` kernel.
    #[error(
        "wgpu dispatch requires a SpirvVulkan-compiled kernel, got one compiled for {actual:?}"
    )]
    WrongKernelTarget { actual: poot_target::Backend },
    /// The bound buffers did not meet the compiled kernel's argument schema (card 608, SC-002): a typed
    /// release-build error before any submission.
    #[error("kernel argument schema violation: {0}")]
    KernelArgs(#[from] poot_runtime_common::KernelArgError),
    #[error("no Vulkan adapter: {0}")]
    NoAdapter(String),
    #[error("device init failed: {0}")]
    DeviceInit(String),
    #[error("readback failed: {0}")]
    Readback(String),
    #[error(
        "dispatch grid {0:?} exceeds the device max workgroups per dimension ({1}); put the unbounded dim on x"
    )]
    GridCap([u32; 3], u32),
    #[error("readback segment {segment} requests {requested} words from a {available}-word buffer")]
    SegmentLength {
        segment: usize,
        requested: usize,
        available: usize,
    },
    #[error("readback segment byte layout overflowed at segment {segment}")]
    SegmentOverflow { segment: usize },
    #[error("cannot copy a {src}-word buffer into a {dst}-word buffer")]
    CopyLength { dst: u32, src: u32 },
    /// A kernel's `Assert` failed or it reached `Unreachable` on the device (card 531c, R468-007): the
    /// SpirvVulkan error word the kernel atomically wrote was nonzero. `code` is the body-local trap id
    /// (`poot_kernel_ir::Terminator::Trap`'s field); it is not a message so callers never match on text.
    #[error("kernel `{kernel}` trapped (assert/unreachable code {code})")]
    KernelAssertFailed { kernel: String, code: u32 },
    /// A typed transfer was requested against a differently represented allocation (card 527:
    /// R471-009/R484-001 - checked against the buffer's full storage contract, element kind, logical
    /// dtype and layout together, not element kind alone).
    #[error("{op} requires {expected} storage, but the buffer stores {actual}")]
    RepresentationMismatch {
        op: &'static str,
        expected: BufferStorage,
        actual: BufferStorage,
    },
    /// A checked byte range exceeded the buffer's byte capacity (card 527 review G2: a release check on
    /// every partial write, mirroring ROCm/PTX's `RangeOutOfBounds`).
    #[error(
        "{op} range [{byte_offset}, {byte_end}) exceeds the buffer's {byte_capacity}-byte capacity"
    )]
    RangeOutOfBounds {
        op: &'static str,
        byte_offset: usize,
        byte_end: usize,
        byte_capacity: usize,
    },
    /// Offset or length arithmetic overflowed before the write (card 527 review G2).
    #[error("{op} byte range overflow at element offset {element_offset} with {elements} elements")]
    RangeOverflow {
        op: &'static str,
        element_offset: usize,
        elements: usize,
    },
}
