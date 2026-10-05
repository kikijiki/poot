//! Error type for [`crate::device::RocmDevice`].

use poot_codegen::CompileError;
use poot_rocm_runtime::RocmError;
use thiserror::Error;

/// Errors from the ROCm executor [`Device`](poot_executor::Device) implementation: a thin wrapper
/// around [`RocmError`] plus the codegen [`CompileError`] the one generated copy kernel can raise, so
/// callers can match on either without pulling in both dependencies. Spec 384: an error enum never
/// carries another poot error enum by value; every cause is boxed so this type's size does not depend
/// on the cause chain depth.
#[derive(Debug, Error)]
pub enum RocmGpuError {
    #[error("rocm runtime: {0}")]
    Rocm(#[source] Box<RocmError>),
    #[error("copy kernel codegen: {0}")]
    Codegen(#[source] Box<CompileError>),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A dispatch or copy this device cannot express: a kernarg segment too small for its argument
    /// count, or a `Device::copy` byte length that is not a whole number of `u32` words.
    #[error("{0}")]
    Graph(String),
}

impl From<RocmError> for RocmGpuError {
    fn from(error: RocmError) -> Self {
        RocmGpuError::Rocm(Box::new(error))
    }
}

impl From<CompileError> for RocmGpuError {
    fn from(error: CompileError) -> Self {
        RocmGpuError::Codegen(Box::new(error))
    }
}
