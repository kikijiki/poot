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
//!   completion signal awaited. See [`ffi::Funcs`] for the function pointers.
//!
//! Safety: no safe public item can cause undefined behavior, and every `unsafe` block states the invariant it
//! relies on in a `SAFETY:` comment (denied below).
//!
//! [`RocmContext::load_hsaco`] takes a [`CompiledKernel`] (card 608), not raw HSACO bytes: its only
//! constructor is `unsafe`, called only by `poot-codegen`, so safe code here cannot mint one.
//!
//! ```compile_fail,E0133
//! fn from_any_bytes(bytes: &[u8]) -> poot_rocm_runtime::CompiledKernel {
//!     poot_rocm_runtime::CompiledKernel::new(
//!         poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
//!         "add",
//!         poot_runtime_common::KernelCode::Hsaco(bytes.to_vec().into_boxed_slice()),
//!         Vec::new(),
//!         false,
//!     )
//! }
//! ```
//!
//! ```compile_fail,E0451
//! fn forge(code: poot_runtime_common::KernelCode) -> poot_rocm_runtime::CompiledKernel {
//!     poot_rocm_runtime::CompiledKernel {
//!         target: poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
//!         entry_point: "add".into(),
//!         code,
//!         args: Vec::new().into(),
//!         has_trap: false,
//!     }
//! }
//! ```

#![deny(clippy::undocumented_unsafe_blocks)]

pub mod bindings;

pub mod ffi;

mod platform;

mod publication;

pub use publication::{
    DeviceStoreFence, QueueHostLink, QueueHostTarget, QueuePublicationContract,
    QueuePublicationError, QueueRingMemory, take_issued_device_store_fence_count,
};

use std::ffi::{OsString, c_char, c_void};

use std::sync::atomic::Ordering;

use std::sync::{Arc, Mutex, OnceLock};

use tracing::info;

use crate::bindings::{
    HSA_AGENT_INFO_DEVICE, HSA_AGENT_INFO_ISA, HSA_AGENT_INFO_NAME, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
    HSA_AGENT_INFO_VENDOR_NAME, HSA_AGENT_INFO_WAVEFRONT_SIZE, HSA_AMD_AGENT_INFO_BDFID,
    HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES,
    HSA_AMD_AGENT_INFO_DOMAIN, HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
    HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES, HSA_AMD_AGENT_INFO_PRODUCT_NAME,
    HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS,
    HSA_AMD_LINK_INFO_TYPE_PCIE, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED,
    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
    HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, HSA_AMD_MEMORY_POOL_INFO_LOCATION,
    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
    HSA_AMD_MEMORY_POOL_INFO_SIZE, HSA_AMD_MEMORY_POOL_LOCATION_CPU,
    HSA_AMD_MEMORY_POOL_LOCATION_GPU, HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU, HSA_AMD_SEGMENT_GLOBAL,
    HSA_DEVICE_TYPE_CPU, HSA_DEVICE_TYPE_GPU, HSA_ISA_INFO_NAME, HSA_ISA_INFO_NAME_LENGTH,
    HSA_QUEUE_FEATURE_KERNEL_DISPATCH, HSA_QUEUE_TYPE_COOPERATIVE, HSA_QUEUE_TYPE_MULTI,
    hsa_agent_t, hsa_amd_memory_pool_link_info_t, hsa_amd_memory_pool_t, hsa_isa_t, hsa_queue_t,
    hsa_status_t,
};

use object::read::elf::ElfFile;

use object::{LittleEndian, Object, ObjectSection, ObjectSymbol};

pub use crate::ffi::Funcs;

use crate::ffi::Hsa;

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

pub use aql::*;
pub use buffer::*;
pub use context::*;
pub use discovery::*;
pub use error::*;
pub use hsaco::*;
pub(crate) use owner::*;
/// The compiler-produced kernel handle [`RocmContext::load_hsaco`] takes (card 608): re-exported so
/// callers need not depend on `poot-runtime-common` directly just to name the type.
pub use poot_runtime_common::CompiledKernel;
pub use queue::*;

#[cfg(test)]
mod tests;
