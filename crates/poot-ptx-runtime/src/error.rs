use cudarc::driver::DriverError;
use poot_target::BufferStorage;

/// Every failure the NVIDIA/PTX runtime reports: CUDA driver errors, kernel-ABI and argument-schema
/// violations, host-transfer size/range/representation mismatches, and capture/peer-access misuse.
/// Each `op` field names the runtime entry point that refused, for the error message.
#[derive(Debug, thiserror::Error)]
pub enum PtxError {
    /// The CUDA driver (via `cudarc`) returned an error.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    /// The kernel text or entry name could not be loaded as a PTX module; carries the reason.
    #[error("invalid kernel text or entry name: {0}")]
    BadKernel(String),
    /// A dispatch was given a [`poot_runtime_common::CompiledKernel`] compiled for a different backend
    /// (card 608): this runtime only ever dispatches an `Nvptx`-compiled kernel.
    #[error("PTX dispatch requires an Nvptx-compiled kernel, got one compiled for {actual:?}")]
    WrongKernelTarget {
        /// The target the dispatched kernel was actually compiled for.
        actual: poot_target::TargetId,
    },
    /// The bound buffers did not meet the compiled kernel's argument schema (card 608, SC-002): a typed
    /// release-build error before any submission.
    #[error("kernel argument schema violation: {0}")]
    KernelArgs(#[from] poot_runtime_common::KernelArgError),
    /// The CUDA driver library could not be loaded on this host; carries the loader message.
    #[error("CUDA library unavailable: {0}")]
    CudaUnavailable(String),
    /// A host transfer's element count differs from the buffer's.
    #[error("{op} size mismatch: buffer has {have} elements, got {got}")]
    SizeMismatch {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Elements the buffer holds.
        have: usize,
        /// Elements the caller supplied.
        got: usize,
    },
    /// A buffer is longer than the `u32` element count the kernel ABI passes.
    #[error("{op} element count {elements} exceeds the native u32 kernel-length ABI")]
    ElementCountTooLarge {
        /// The runtime entry point that refused.
        op: &'static str,
        /// The element count that does not fit in `u32`.
        elements: usize,
    },
    /// `elements * element_bytes` overflowed `usize`.
    #[error("{op} byte size overflow: {elements} elements * {element_bytes} bytes")]
    ByteSizeOverflow {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Element count of the request.
        elements: usize,
        /// Bytes per element.
        element_bytes: usize,
    },
    /// The buffer's storage representation is not the one the operation requires.
    #[error("{op} requires {expected} storage, but the buffer stores {actual}")]
    RepresentationMismatch {
        /// The runtime entry point that refused.
        op: &'static str,
        /// The storage the operation requires.
        expected: BufferStorage,
        /// The storage the buffer holds.
        actual: BufferStorage,
    },
    #[error(
        "{op} range [{byte_offset}, {byte_end}) exceeds the buffer's {byte_capacity}-byte capacity"
    )]
    /// A byte range reaches past the buffer's capacity.
    RangeOutOfBounds {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Inclusive start of the requested range, in bytes.
        byte_offset: usize,
        /// Exclusive end of the requested range, in bytes.
        byte_end: usize,
        /// The buffer's capacity, in bytes.
        byte_capacity: usize,
    },
    /// Computing a byte range from an element offset and count overflowed `usize`.
    #[error("{op} byte range overflow at element offset {element_offset} with {elements} elements")]
    RangeOverflow {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Requested start, in elements.
        element_offset: usize,
        /// Requested length, in elements.
        elements: usize,
    },
    /// A byte payload is not a whole number of elements.
    #[error("{op} got {bytes} bytes, which is not a whole number of {element_bytes}-byte elements")]
    InvalidByteLength {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Length of the payload, in bytes.
        bytes: usize,
        /// Bytes per element of the buffer.
        element_bytes: usize,
    },
    /// Only kernel dispatches may run while a stream capture is open; see `PtxContext::begin_capture`.
    #[error("{op} refused: a stream capture is open on this context")]
    CaptureOpen {
        /// The runtime entry point that refused.
        op: &'static str,
    },
    /// A cross-device operation that only makes sense inside a stream capture was issued with none open.
    #[error("{op} refused: no stream capture is open on this context")]
    CaptureClosed {
        /// The runtime entry point that refused.
        op: &'static str,
    },
    /// An event, buffer or context of another owner was passed where this context's own is required.
    #[error("{op} refused: the handle belongs to another context")]
    ForeignHandle {
        /// The runtime entry point that refused.
        op: &'static str,
    },
    /// Direct access between two devices is not available: a copy between them would have to stage
    /// through the host, which a capture cannot hold.
    #[error("device {from} cannot access device {to}'s memory directly")]
    PeerUnavailable {
        /// CUDA ordinal of the device that would read.
        from: i32,
        /// CUDA ordinal of the device whose memory would be read.
        to: i32,
    },
    /// The destination's device was never enabled with `PtxContext::enable_peer_access`.
    #[error("copy_to_peer refused: direct access to the destination's device was not enabled")]
    PeerNotEnabled,
}
