//! Error type for [`crate::device::RocmDevice`].

use poot_rocm_runtime::RocmError;
use poot_target::ArchError;
use thiserror::Error;

/// Errors from the ROCm executor [`Device`](poot_executor::Device) implementation: a thin wrapper
/// around [`RocmError`] plus the contract misuse this device refuses with a value instead of a panic.
/// Spec 384: an error enum never carries another poot error enum by value; every cause is boxed so this
/// type's size does not depend on the cause chain depth.
#[derive(Debug, Error)]
pub enum RocmGpuError {
    /// The ROCm runtime (HSA, queue, buffer or code-object layer) failed.
    #[error("rocm runtime: {0}")]
    Rocm(#[source] Box<RocmError>),
    /// The GPU reported an ISA name this device cannot classify into an architecture.
    #[error("rocm device: {0}")]
    Arch(#[source] Box<ArchError>),
    /// `begin` was called while a recording was already open.
    #[error("rocm device: begin inside an open recording")]
    AlreadyRecording,
    /// `dispatch`, `copy` or `finish` was called with no recording open.
    #[error("rocm device: {op} outside begin/finish")]
    NotRecording {
        /// The call that found no open recording.
        op: &'static str,
    },
    /// `copy` was called before the engine handed the device its copy kernel
    /// ([`poot_executor::Device::copy_mechanism`]).
    #[error("rocm device: copy before the copy kernel was loaded")]
    CopyKernelMissing,
    /// `copy` between buffers of different byte capacity.
    #[error("rocm device: copy of a {src}-byte buffer into a {dst}-byte buffer")]
    CopyLength {
        /// Byte capacity of the source.
        src: usize,
        /// Byte capacity of the destination.
        dst: usize,
    },
    /// `copy` between buffers whose byte capacity is not a whole number of `u32` words: the copy kernel moves
    /// whole words, so it would run past the end of the allocation. The runtime allocates whole words, so
    /// this is the runtime breaking its contract, not a caller error.
    #[error("rocm device: copy of {bytes} bytes, which is not a whole number of u32 words")]
    CopyNotWordAligned {
        /// Byte capacity of the buffers.
        bytes: usize,
    },
    /// A copy of more `u32` words than one launch addresses.
    #[error("rocm device: copy of {words} words exceeds the launch ABI")]
    CopyTooLarge {
        /// Words the copy would move.
        words: usize,
    },
}

impl From<RocmError> for RocmGpuError {
    fn from(error: RocmError) -> Self {
        RocmGpuError::Rocm(Box::new(error))
    }
}

impl From<ArchError> for RocmGpuError {
    fn from(error: ArchError) -> Self {
        RocmGpuError::Arch(Box::new(error))
    }
}
