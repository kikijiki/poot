//! Device-resident dispatch of poot-emitted AMDGPU kernels via the raw HSA Foundation runtime
//! (`libhsa-runtime64.so.1`) on the Strix Halo iGPU (Radeon 8060S, gfx1151). The AMD analogue of
//! `poot-ptx-runtime`. The library is loaded with `dlopen` and each entry point resolved with `dlsym`,
//! so the crate builds without ROCm installed (spec 063 FR-001, FR-004).
//!
//! - [`RocmContext::new`]: `hsa_init`, iterate agents, pick the first GPU, walk pools, create one queue
//!   for an immutable caller-selected role.
//! - [`RocmContext::allocate_f32`] / [`RocmContext::upload_f32`] / [`RocmContext::download_f32`]: f32
//!   device buffers. On Strix Halo the system pool is HMM/SVM, so upload/download are a
//!   `ptr::copy_nonoverlapping` into/out of a GPU-visible allocation.
//! - [`RocmContext::synchronize`]: drain the queue (spin on the queue's read index).
//! - HSACO loading and dispatch: an AQL packet is written, the doorbell rung, and a per-launch
//!   completion signal awaited. See the crate-private `ffi::Funcs` for the function pointers.
//!
//! Safety: no safe public item can cause undefined behavior, and every `unsafe` block states the invariant it
//! relies on in a `SAFETY:` comment (denied below).
//!
//! [`RocmContext::load_hsaco`] takes a [`CompiledKernel`] (card 608), not raw HSACO bytes: its only
//! constructor is `unsafe`, called only by `poot-codegen`, so safe code here cannot mint one.
//!
//! ```compile_fail,E0133
//! fn from_any_bytes(bytes: &[u8]) -> poot_runtime_common::CompiledKernel {
//!     poot_runtime_common::CompiledKernel::new(
//!         poot_target::TargetId::Gpu(poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151())),
//!         "add",
//!         poot_runtime_common::KernelCode::Hsaco(bytes.to_vec().into_boxed_slice()),
//!         Vec::new(),
//!         false,
//!     )
//! }
//! ```
//!
//! ```compile_fail,E0451
//! fn forge(code: poot_runtime_common::KernelCode) -> poot_runtime_common::CompiledKernel {
//!     poot_runtime_common::CompiledKernel {
//!         target: poot_target::TargetId::Gpu(poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151())),
//!         entry_point: "add".into(),
//!         code,
//!         args: Vec::new().into(),
//!         has_trap: false,
//!     }
//! }
//! ```

#![deny(clippy::undocumented_unsafe_blocks)]

mod bindings;

mod ffi;

mod publication;

mod aql;
mod buffer;
mod context;
mod discovery;
mod error;
mod hsaco;
mod owner;
mod queue;
mod replay;
mod transfer;
mod upload;

pub use aql::{DeviceRingReceiptProvenance, DispatchAxis, DispatchGeometryViolation};
pub use buffer::RocmBuffer;
pub use context::{DispatchSpan, RocmContext, RocmContextOptions, UploadStrategy};
pub use error::{DeviceFault, HsaStatus, RocmError};
pub use hsaco::{BoundKernarg, HsacoModule, KernelHandle};
pub use publication::{
    DeviceStoreFence, QueueHostLink, QueueHostTarget, QueuePublicationContract,
    QueuePublicationError, QueueRingMemory, take_issued_device_store_fence_count,
};
pub use queue::{QueueRingProvenance, QueueRole};

#[cfg(test)]
mod tests;
