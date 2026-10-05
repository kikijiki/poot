//! Dispatch poot-emitted NVPTX kernels on NVIDIA via cudarc's raw driver API (G3a).
//!
//! The CUDA analogue of `poot-runtime`'s wgpu surface, used by the PTX graph executor (G3b) and
//! capture/replay (G3c). Intermediates stay on the GPU across dispatches: upload inputs once, dispatch
//! reading/writing persistent [`PtxBuffer`]s, download only the final output. The kernel ABI matches
//! poot-codegen's NVPTX emitter and `poot-ptx-check`: each slice param lowers to `(ptr addrspace(1), i64 len)`
//! in declaration order, inputs first and the single writable output last.
//!
//! cudarc uses `dynamic-loading` (it dlopens `libcuda.so.1`), so this builds without a CUDA toolkit and only
//! runs on a machine with an NVIDIA driver. It is exercised on rented RunPod GPUs.
//!
//! Threading: a [`PtxContext`] sets its primary context current on the creating thread. The
//! primary context is shareable across threads (retained once per device, refcounted by the driver); only
//! the "current context" pointer is per-thread. Every owning handle this crate exports - [`PtxContext`],
//! [`PtxBuffer`] and [`PtxGraphExec`] - is `Send`: the shared native state behind them
//! ([`owner::PtxOwner`]) is `Arc`-backed, not `Rc`-backed, so a last-clone drop racing another thread's
//! clone/drop of the same allocation cannot corrupt the refcount, and `PtxOwner` itself is `unsafe impl
//! Send + Sync` (see its doc) because every driver call activates its context on the calling thread first
//! (`current_guard`/`with_current`) and its one piece of shared mutable state (`modules`) is a `Mutex`. A
//! handle built on one thread may therefore be moved to another (no production caller does this yet -
//! `poot-serve`'s batch scheduler does not construct a `PtxContext` today, card 549),
//! but whichever thread it lands on must call [`PtxContext::make_current`] before its first driver call,
//! or every call fails with `CUDA_ERROR_INVALID_CONTEXT`. A standalone [`PtxGraphExec`] is the exception:
//! it retains and scopes its originating context for launch and synchronization, then restores the
//! calling thread's prior context.
//! Device timing is the typed [`PtxContext::launch_timed`] span path; there is no label-keyed profiler.
//!
//! Safety: no safe public item can cause undefined behavior, and every `unsafe` block states the invariant it
//! relies on in a `SAFETY:` comment (denied below).
//!
//! [`PtxContext::dispatch_dev`] takes a [`CompiledKernel`] (card 608), not raw PTX text: its only
//! constructor is `unsafe`, called only by `poot-codegen`, so safe code here cannot mint one.
//!
//! ```compile_fail,E0133
//! fn from_any_text(text: &str) -> poot_ptx_runtime::CompiledKernel {
//!     poot_ptx_runtime::CompiledKernel::new(
//!         poot_target::Backend::Nvptx,
//!         "add",
//!         poot_runtime_common::KernelCode::Ptx(text.into()),
//!         Vec::new(),
//!         false,
//!     )
//! }
//! ```
//!
//! ```compile_fail,E0451
//! fn forge(code: poot_runtime_common::KernelCode) -> poot_ptx_runtime::CompiledKernel {
//!     poot_ptx_runtime::CompiledKernel {
//!         target: poot_target::Backend::Nvptx,
//!         entry_point: "add".into(),
//!         code,
//!         args: Vec::new().into(),
//!         has_trap: false,
//!     }
//! }
//! ```

#![deny(clippy::undocumented_unsafe_blocks)]

mod buffer;
mod context;
mod driver;
mod error;
mod graph;
mod owner;

pub use buffer::{AllocGuard, BufferRole, MemoryCounterSnapshot, PtxBuffer};
pub use context::{PtxContext, f32_to_bf16};
pub use error::{BufferElement, BufferStorage, LogicalDType, PtxError, StorageLayout};
pub use graph::{PtxGraphExec, PtxSpan};
/// The compiler-produced kernel handle every dispatch takes (card 608): re-exported so callers need not
/// depend on `poot-runtime-common` directly just to name the type.
pub use poot_runtime_common::CompiledKernel;

#[cfg(test)]
pub(crate) use {
    buffer::{allocate_owned, checked_buffer, checked_copy, checked_layout},
    cudarc::driver::{DriverError, result, sys},
    driver::CudaDriver,
    graph::{CaptureRetention, ModuleConstruction, create_event_pair, instantiate_graph},
    owner::{PtxOwner, construct_owner},
    std::collections::HashMap,
    std::sync::Arc,
};

#[cfg(test)]
pub(crate) use context::{KernelArgs, require_gpu_check_with};

#[cfg(test)]
mod tests;
