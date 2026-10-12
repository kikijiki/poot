use poot_target::BufferStorage;
/// Every failure the wgpu runtime reports: adapter and device setup, dispatch argument validation, and
/// host transfers. Typed so a caller matches on the variant, never on message text.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// A dispatch was given a [`poot_runtime_common::CompiledKernel`] compiled for a different backend
    /// (card 608): wgpu only ever dispatches a `SpirvVulkan` kernel.
    #[error(
        "wgpu dispatch requires a SpirvVulkan-compiled kernel, got one compiled for {actual:?}"
    )]
    WrongKernelTarget {
        /// The target the rejected kernel was compiled for.
        actual: poot_target::TargetId,
    },
    /// The bound buffers did not meet the compiled kernel's argument schema (card 608, SC-002): a typed
    /// release-build error before any submission.
    #[error("kernel argument schema violation: {0}")]
    KernelArgs(#[from] poot_runtime_common::KernelArgError),
    /// wgpu found no adapter it can drive; the payload is the request failure's message.
    #[error("no Vulkan adapter: {0}")]
    NoAdapter(String),
    /// The adapter was found but opening the device (or the queue) on it failed; the payload is the
    /// driver's message.
    #[error("device init failed: {0}")]
    DeviceInit(String),
    /// Mapping or reading a staging buffer back to the host failed; the payload is the wgpu message.
    #[error("readback failed: {0}")]
    Readback(String),
    /// A dispatch grid exceeds the device's `max_compute_workgroups_per_dimension`: `.0` is the requested
    /// grid, `.1` the per-dimension limit it was checked against.
    #[error(
        "dispatch grid {0:?} exceeds the device max workgroups per dimension ({1}); put the unbounded dim on x"
    )]
    GridCap([u32; 3], u32),
    /// A readback segment asked for more words than the buffer it reads from holds.
    #[error("readback segment {segment} requests {requested} words from a {available}-word buffer")]
    SegmentLength {
        /// Index of the offending segment in the readback request.
        segment: usize,
        /// Words the segment asked for.
        requested: usize,
        /// Words the source buffer holds.
        available: usize,
    },
    /// The running byte offset of the packed readback layout overflowed `usize` at this segment.
    #[error("readback segment byte layout overflowed at segment {segment}")]
    SegmentOverflow {
        /// Index of the segment whose offset arithmetic overflowed.
        segment: usize,
    },
    /// A device-side copy between buffers of different word counts.
    #[error("cannot copy a {src}-word buffer into a {dst}-word buffer")]
    CopyLength {
        /// Word count of the destination buffer.
        dst: u32,
        /// Word count of the source buffer.
        src: u32,
    },
    /// A kernel's `Assert` failed or it reached `Unreachable` on the device (card 531c, R468-007): the
    /// SpirvVulkan error word the kernel atomically wrote was nonzero. `code` is the body-local trap id
    /// (`poot_kernel_ir::Terminator::Trap`'s field); it is not a message so callers never match on text.
    #[error("kernel `{kernel}` trapped (assert/unreachable code {code})")]
    KernelAssertFailed {
        /// Name of the kernel that trapped.
        kernel: String,
        /// Body-local trap id the kernel wrote.
        code: u32,
    },
    /// A typed transfer was requested against a differently represented allocation (card 527:
    /// R471-009/R484-001 - checked against the buffer's full storage contract, element kind, logical
    /// dtype and layout together, not element kind alone).
    #[error("{op} requires {expected} storage, but the buffer stores {actual}")]
    RepresentationMismatch {
        /// The transfer entry point that refused, for the message.
        op: &'static str,
        /// The storage contract the transfer needs.
        expected: BufferStorage,
        /// The storage contract the buffer actually carries.
        actual: BufferStorage,
    },
    /// A checked byte range exceeded the buffer's byte capacity (card 527 review G2: a release check on
    /// every partial write, mirroring ROCm/PTX's `RangeOutOfBounds`).
    #[error(
        "{op} range [{byte_offset}, {byte_end}) exceeds the buffer's {byte_capacity}-byte capacity"
    )]
    RangeOutOfBounds {
        /// The transfer entry point that refused, for the message.
        op: &'static str,
        /// First byte of the requested range.
        byte_offset: usize,
        /// One past the last byte of the requested range.
        byte_end: usize,
        /// The buffer's capacity in bytes.
        byte_capacity: usize,
    },
    /// A write the device cannot take: it moves whole 32-bit words, so `offset` and `len` must be
    /// multiples of four, except that a write reaching the end of the buffer may end mid-word.
    #[error("{op} at byte offset {offset} of {len} bytes is not word aligned")]
    Misaligned {
        /// The transfer entry point that refused, for the message.
        op: &'static str,
        /// Byte offset of the write.
        offset: usize,
        /// Byte length of the write.
        len: usize,
    },
    /// Offset or length arithmetic overflowed before the write (card 527 review G2).
    #[error("{op} byte range overflow at element offset {element_offset} with {elements} elements")]
    RangeOverflow {
        /// The transfer entry point that refused, for the message.
        op: &'static str,
        /// Element offset the write started at.
        element_offset: usize,
        /// Element count of the write.
        elements: usize,
    },
}
