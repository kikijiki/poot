//! poot-vulkan-runtime: dispatch a compiled SPIR-V kernel via raw Vulkan (`ash`), no wgpu.
//!
//! Spec 133 P1: device, pipeline, and single dispatch from poot's existing `Target::SpirvVulkan` `.spv`
//! output (no naga, no re-emission). The binding ABI matches `poot-runtime`'s (wgpu) exactly, so the same
//! `.spv` dispatches unmodified on either backend (this crate's `run_add` test cross-checks both):
//!
//! - Descriptor set 0. Storage buffers at bindings `0..N-1` are the kernel's data buffers in the order
//!   the caller passes them: inputs first, then the single output last (as in
//!   `poot_runtime::Context::dispatch`; [`Context::dispatch`] takes the same shape).
//! - Binding `N` is a `[u32; N]` length buffer, one element per data buffer in the same order. Poot
//!   kernels bounds-check `i < len` against it, and `poot-runtime` builds the same buffer.
//! - Every binding is `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`, stage `COMPUTE` only (as `poot-runtime`'s
//!   `storage_entry`).
//! - The workgroup shape (`LocalSize`) is baked into the SPIR-V module; the caller supplies the matching
//!   `wg` and per-axis `threads` (output extent), so the grid is `ceil(threads / wg)` per axis, as in
//!   `poot-runtime::dispatch`'s `workgroup_count`.
//! - The entry point is `"main"` (poot-codegen's SpirvVulkan convention).
//!
//! Memory model (P1, spec 133 FR-003): RADV (the dev-box AMD iGPU) exposes a unified
//! `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` heap ([[dev-box-strix-halo]]). P1 always allocates from a
//! `HOST_VISIBLE | HOST_COHERENT` memory type (preferring one that is also `DEVICE_LOCAL`), maps it
//! persistently, and reads/writes through the mapped pointer: no staging buffer, no explicit copy, no
//! `vkFlush`/`vkInvalidateMappedMemoryRanges` (`HOST_COHERENT` guarantees visibility once the
//! submission's fence is signaled). FR-003's device-local + staging path for a discrete GPU without a
//! unified heap is out of scope for P1 (see `specs/133-raw-vulkan-backend/spec.md`, "Open questions").
//!
//! Pipelines (spec 133 P2 slice 1, card 139): [`Context::pipeline`] builds a pipeline/module once per
//! caller-supplied kernel key (like `poot-runtime`'s `CachedPipeline`, and the `Plan::Compute`/`ComputeMeta`
//! `key` the executor loads kernels under) and returns a shared [`Pipeline`]. A later call with the same
//! key returns the same pipeline, and is refused ([`RuntimeError::PipelineKeyConflict`]) if the kernel's
//! code, argument schema or trap shape differs, so a key can never dispatch another body's pipeline.
//! [`Context::dispatch`] (P1, no cache: builds and destroys every pipeline object per call) is unchanged
//! for `run_add` and `poot-vulkan-check`.
//!
//! Record and replay (spec 133 P2 slice 2a, card 139; Card 553, ADR-0043): [`VulkanGraph`] is the
//! record-once/replay-many primitive, the Vulkan analogue of CUDA graphs / ROCm AQL replay (see
//! [`Context::begin_graph`], [`Context::record_dispatch`], [`Context::record_copy`] and
//! [`Context::end_graph`]). `VkCommandBuffer`s are recorded once (each command's pipeline bind, descriptor
//! bind, and a conservative full memory barrier between consecutive commands) and re-`vkQueueSubmit`-ted
//! verbatim per replay, one fence wait after each. [`Context::cut_graph`] splits a long recording into
//! several command buffers so no single submission outlives the display watchdog. Each dispatch's length
//! block (and error word, for a kernel that can trap) lives in a few large host-visible buffers the graph
//! owns, so a graph of thousands of dispatches costs a handful of allocations. The graph retains every
//! buffer it binds and every pipeline it references (a freed buffer a captured dispatch references would
//! be a use-after-free, card 126). Descriptor sets are baked at record time and never updated afterward;
//! per-replay input data overwrites the same bound buffers in place via [`DeviceBuffer::write_bytes`]
//! (FR-009's persistently-mapped host-visible slot buffers). A dispatch is told each buffer's own element
//! count ([`Binding::elems`]), never the allocation's capacity.
//!
//! A replay reports a kernel assert that fired and, for a [`GraphTiming::PerDispatch`] graph, the device
//! time of each dispatch ([`Replay`]); it never waits for either beyond the fence its submissions already
//! wait on.
//!
//! Threads: buffers, pipelines and graphs are `Send + Sync` (the shared device owner is atomic and its
//! queue sits behind a lock), so a buffer moves to another thread whole (R471-010); a [`Context`] stays on
//! the thread that owns it.
//!
//! Safety: no safe public item can cause undefined behavior. A kernel reaches the device only as a
//! [`CompiledKernel`], whose `unsafe` constructor carries the SPIR-V validity and bounds contract; a
//! recording graph cannot be replayed (the `RecordingGraph`/`VulkanGraph` typestate); a buffer's
//! element count is fixed at allocation; a submission whose fence wait fails poisons the device, which
//! then leaks everything that work could reference and refuses every later operation (as ROCm does
//! after a timed-out wait); and every `unsafe` block states the invariant it relies on in a `SAFETY:`
//! comment (denied below).

#![deny(clippy::undocumented_unsafe_blocks)]

use std::collections::HashMap;

use std::ffi::c_void;

use std::mem::ManuallyDrop;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use std::time::Duration;

use ash::vk;

mod buffer;
mod context;
mod driver;
mod error;
mod features;
mod graph;
mod kernel;
mod owner;
mod resources;
mod spirv;
mod util;
mod validation;

pub use buffer::*;
pub use context::graph::Binding;
pub(crate) use driver::*;
pub use error::*;
pub(crate) use features::DeviceFeatures;
pub use graph::{
    DispatchTime, GraphDeviceTime, GraphTiming, KernelFault, RecordingGraph, Replay, VulkanGraph,
};
pub use kernel::*;
pub use owner::Pipeline;
pub(crate) use owner::*;
pub(crate) use resources::*;
pub(crate) use util::*;
pub(crate) use validation::Validation;

/// A Vulkan instance + compute-capable device + queue. Reusable across dispatches.
pub struct Context {
    /// Function-table clones for context-bound operations. The shared owner retains the underlying native
    /// objects and loader; these clones never destroy handles themselves.
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    timestamp_period_ns: f32,
    /// `timestampValidBits` of the queue family as a mask (0 when it writes no timestamps).
    timestamp_valid_mask: u64,
    /// The `[min, max]` subgroup size a cooperative-matrix pipeline may require, when the device can
    /// create such a pipeline at all (`None`: it cannot, and reports no matrix hardware).
    coopmat_subgroup_sizes: Option<(u32, u32)>,
    /// `minStorageBufferOffsetAlignment`: a descriptor binds a meta block at a multiple of this.
    min_storage_offset_alignment: usize,
    /// Per-axis `maxComputeWorkGroupCount` from the physical device (queried, not hardcoded).
    /// A dispatch whose grid exceeds these is refused ([`RuntimeError::GridCap`]), never folded.
    max_workgroup_count: [u32; 3],
    /// The device-capability descriptor, measured once at construction (card 522 review M1) - including
    /// matrix (tensor-core) hardware from the `VK_KHR_cooperative_matrix` query: [`Context::device_caps`]
    /// returns this stored value, never re-queries the driver.
    device_caps: poot_target::DeviceCaps,
    /// Pipeline/module cache keyed by the caller's kernel key (see the module doc), used by `Context::pipeline`.
    pipeline_cache: HashMap<String, Arc<Pipeline>>,
    /// Count of pipeline builds (cache misses) via `Context::pipeline`, like `poot-runtime`'s `pipeline_builds()`.
    pipeline_builds: usize,
    /// Count of `Context::end_graph` calls (command buffers finished recording); a graph captured once and replayed N times leaves this at 1.
    graphs_recorded: AtomicUsize,
    /// Last so the cache releases its pipeline children before this context releases its owner reference.
    owner: Arc<DeviceOwner>,
}

#[cfg(test)]
mod tests;
