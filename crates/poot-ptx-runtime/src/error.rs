use cudarc::driver::DriverError;

#[derive(Debug, thiserror::Error)]
pub enum PtxError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("invalid kernel text or entry name: {0}")]
    BadKernel(String),
    /// A dispatch was given a [`poot_runtime_common::CompiledKernel`] compiled for a different backend
    /// (card 608): this runtime only ever dispatches an `Nvptx`-compiled kernel.
    #[error("PTX dispatch requires an Nvptx-compiled kernel, got one compiled for {actual:?}")]
    WrongKernelTarget { actual: poot_target::Backend },
    /// The bound buffers did not meet the compiled kernel's argument schema (card 608, SC-002): a typed
    /// release-build error before any submission.
    #[error("kernel argument schema violation: {0}")]
    KernelArgs(#[from] poot_runtime_common::KernelArgError),
    #[error("CUDA library unavailable: {0}")]
    CudaUnavailable(String),
    #[error("{op} size mismatch: buffer has {have} elements, got {got}")]
    SizeMismatch {
        op: &'static str,
        have: usize,
        got: usize,
    },
    #[error("{op} element count {elements} exceeds the native u32 kernel-length ABI")]
    ElementCountTooLarge { op: &'static str, elements: usize },
    #[error("{op} byte size overflow: {elements} elements * {element_bytes} bytes")]
    ByteSizeOverflow {
        op: &'static str,
        elements: usize,
        element_bytes: usize,
    },
    #[error("{op} requires {expected} storage, but the buffer stores {actual}")]
    RepresentationMismatch {
        op: &'static str,
        expected: BufferStorage,
        actual: BufferStorage,
    },
    #[error(
        "{op} range [{byte_offset}, {byte_end}) exceeds the buffer's {byte_capacity}-byte capacity"
    )]
    RangeOutOfBounds {
        op: &'static str,
        byte_offset: usize,
        byte_end: usize,
        byte_capacity: usize,
    },
    #[error("{op} byte range overflow at element offset {element_offset} with {elements} elements")]
    RangeOverflow {
        op: &'static str,
        element_offset: usize,
        elements: usize,
    },
    #[error("{op} got {bytes} bytes, which is not a whole number of {element_bytes}-byte elements")]
    InvalidByteLength {
        op: &'static str,
        bytes: usize,
        element_bytes: usize,
    },
    /// Only kernel dispatches may run while a stream capture is open; see `PtxContext::begin_capture`.
    #[error("{op} refused: a stream capture is open on this context")]
    CaptureOpen { op: &'static str },
}

/// Native element representation retained by a PTX allocation (card 527: moved to `poot-target`, the
/// shared type every runtime's buffer handle now uses, so ROCm and PTX no longer keep independent
/// copies; `BufferElement` stays the name this crate's callers use). See [`poot_target::BufferStorage`]
/// for the fuller element-kind + logical-dtype + layout contract [`crate::PtxBuffer`] now carries.
pub use poot_target::ElementKind as BufferElement;
pub use poot_target::{BufferStorage, LogicalDType, StorageLayout};
