//! Dispatch a compiled SPIR-V kernel on wgpu (Vulkan).
//!
//! Minimal wgpu backend for the un-fused graph executor: take a `.spv` (from poot-codegen), bind N
//! storage buffers plus a `[u32; N]` length buffer (the SPIR-V length source, at binding N), dispatch
//! `ceil(threads / workgroup_size)` workgroups, and read writable buffers back. Raw SPIR-V from `llc`
//! goes through the passthrough path (naga rejects it); validity is guaranteed out-of-band by
//! `spirv-val`. The wgpu usage follows the legacy runtime (wgpu 29).
//!
//! Every `unsafe` block states the invariant it relies on in a `SAFETY:` comment (denied below).
//!
//! # Kernel contract
//!
//! [`Context::dispatch`], [`Context::dispatch_dev`], [`Context::launch`] and
//! [`Context::build_cached_dispatch`] all take a [`CompiledKernel`] (card 608), not raw SPIR-V bytes: a
//! valid SPIR-V module following the binding ABI above, whose every access to data binding `i` stays
//! below the element count this crate writes at `lengths[i]`, is exactly what the type's `unsafe`
//! constructor requires. wgpu's SPIR-V passthrough validates neither on its own, so an arbitrary byte
//! string could make the device read or write outside its buffers - but `CompiledKernel`'s only
//! constructor is `unsafe`, called only by `poot-codegen`, so safe code here cannot mint one:
//!
//! ```compile_fail,E0133
//! fn from_any_bytes(spv: &[u8]) -> poot_runtime::CompiledKernel {
//!     poot_runtime::CompiledKernel::new(
//!         poot_runtime::Backend::SpirvVulkan,
//!         "main",
//!         poot_runtime::KernelCode::spirv_from_le_bytes(spv),
//!         Vec::new(),
//!         false,
//!     )
//! }
//! ```
//!
//! ```compile_fail,E0451
//! fn forge(code: poot_runtime::KernelCode) -> poot_runtime::CompiledKernel {
//!     poot_runtime::CompiledKernel {
//!         target: poot_runtime::Backend::SpirvVulkan,
//!         entry_point: "main".into(),
//!         code,
//!         args: Vec::new().into(),
//!         has_trap: false,
//!     }
//! }
//! ```
//!
//! Every dispatch checks the bound buffers' count and native element kind against
//! [`CompiledKernel::args`] before doing anything else (SC-002), returning
//! [`RuntimeError::KernelArgs`] on a mismatch.

#![deny(clippy::undocumented_unsafe_blocks)]

use std::borrow::Cow;

use std::cell::{Cell, RefCell};

use std::collections::HashMap;

use std::sync::atomic::{AtomicU64, Ordering};

use std::time::Duration;

use wgpu::util::DeviceExt;

/// The compiler-produced kernel handle every dispatch takes (card 608): re-exported so callers need not
/// depend on `poot-runtime-common` directly just to name the type.
pub use poot_runtime_common::{
    ArgAccess, ArgSchema, BufferRole, CompiledKernel, KernelCode, MemoryCounterSnapshot,
};
/// Re-exported so a `#[kernel]`-using crate need not add `poot-target` as its own dependency just to
/// name the backend a `CompiledKernel` is built for.
pub use poot_target::Backend;

mod buffer;
mod context;
mod error;
mod pipeline;
mod require;
mod timestamps;

pub use buffer::*;
pub use error::*;
pub use pipeline::*;
pub(crate) use require::*;
pub(crate) use timestamps::*;

/// Safe single-buffer cap for AMD RADV (Vulkan), in bytes: 1.25 GiB.
///
/// Observed 2026-09-25 on AMD Radeon 8060S (RADV STRIX_HALO): an lm_head chunk sitting at the
/// device `max_buffer_size` (2,147,483,647) wedged the gfx ring on its first dispatch, even
/// though `Device::create_buffer` accepted it. [`Context::effective_max_buffer_size`] is
/// therefore `min(device max_buffer_size, AMD_RADV_SAFE_MAX_BUFFER_BYTES)` on AMD Vulkan
/// adapters, and the raw device limit elsewhere. `compile`'s `legalize` pass (card 523a) consumes
/// the effective limit as `DeviceCaps::max_buffer_bytes` (card 522), hosting or splitting whatever
/// constant does not fit; this constant lives here, at the layer that owns device limits, not in a
/// model tracer.
pub const AMD_RADV_SAFE_MAX_BUFFER_BYTES: u64 = 1_342_177_280; // 1.25 GiB

/// A wgpu device + queue. Reusable across dispatches.
///
/// `device`/`queue` are private (card 608): the raw wgpu objects could be rewritten by safe code after
/// `poot-codegen` and this crate's own init read their limits, so external callers get no field access at
/// all (this crate's own dispatch/init/readback code, in the same module tree, still reaches them).
///
/// ```compile_fail,E0616
/// fn read(ctx: &poot_runtime::Context) -> &wgpu::Device {
///     &ctx.device
/// }
/// ```
pub struct Context {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Whether the device has `TIMESTAMP_QUERY` (per-dispatch device time). Without it, the Card 552
    /// typed device-timing path collects nothing (FR-005).
    timestamps: bool,
    /// Card 552: whether this context was built with [`Context::new_with_device_timing`] (review
    /// F1: read by [`Context::submit_encoded`], the one submit path, not a second method).
    device_timing: Cell<bool>,
    /// Card 552: the declared bound on concurrently un-drained [`PendingDeviceTiming`]
    /// resources; `0` means unbounded (never drains early for capacity). Set once at construction.
    max_in_flight_queries: Cell<usize>,
    /// Card 552: this step's running fold of every drained [`PendingDeviceTiming`] (an early
    /// capacity-triggered drain and the step's own final drain both land here), and the index the
    /// next registered chunk's dispatches start at within the step's recording.
    drained_device_timing: RefCell<DeviceTimingAccumulator>,
    next_dispatch_index: Cell<usize>,
    /// Card 552: readback buffers registered by [`Context::submit_encoded`] under the typed path,
    /// not yet drained. Each one's `map_async` is registered without its own poll, so collecting
    /// this detail adds no synchronization beyond the step's own `synchronize()` - whichever poll
    /// (an intermediate chunk boundary, an F3 capacity drain, or the final sync) happens to satisfy
    /// it first. [`Context::discard_device_timing`] clears this (and the accumulator and index)
    /// without reading it, for a step that fails before draining (review F2).
    pending_device_timing: RefCell<Vec<PendingDeviceTiming>>,
    /// Device `max_compute_workgroups_per_dimension` (queried at init, not hardcoded). Dispatch folds
    /// a 1-D grid over this cap onto Y via [`poot_runtime_common::fold_grid`].
    max_workgroups: u32,
    /// Raw device `max_buffer_size` (queried at init from wgpu `Limits`): the largest single buffer
    /// `Device::create_buffer` will accept.
    max_buffer_size: u64,
    /// Effective single-buffer limit for allocation planning: `min(max_buffer_size,
    /// AMD_RADV_SAFE_MAX_BUFFER_BYTES)` on AMD Vulkan (RADV), else `max_buffer_size`. Queried and
    /// derived at init (Card 453 D1); see the constant's doc for the observed RADV wedge.
    effective_max_buffer_size: u64,
    /// Device `max_compute_workgroup_storage_size` (queried at init): the LDS/shared-memory budget one
    /// dispatch may use. Card 522: feeds [`Context::device_caps`].
    lds_bytes: u32,
    /// Matrix-hardware family, measured from `Adapter::cooperative_matrix_properties()` (the real
    /// `VK_KHR_cooperative_matrix` query) on a Vulkan adapter; `UnknownNotExposedByApi` on any other
    /// backend, since this crate confirms only Vulkan's property list is meaningfully populated (card
    /// 522 — corrected from an earlier adapter-name-string classifier this doc described).
    /// wgpu's SPIR-V path does not itself emit AMD WMMA intrinsics (that is the ROCm/AmdGcn codegen path
    /// on the same hardware); this describes the device, not what this backend's codegen can reach.
    tensor_core: poot_target::TensorCoreSupport,
    /// Active compute units, probed from AMD's Vulkan shader-core properties at init (card 653); the
    /// RDNA3 default where the driver exposes no such query. Feeds [`Context::device_caps`].
    compute_units: u32,
    /// Same condition as [`Context::effective_max_buffer_size`]'s RADV scoping: the adapter is AMD
    /// (vendor `0x1002`) over Vulkan. The iGPU display watchdog and the card-095 tiled-GEMM miscompile
    /// are calibrated facts about that driver stack, not something a device query answers, so
    /// `device_caps` reports them as a vendor-scoped default exactly where this is true (Card 522 Scope).
    is_amd_radv: bool,
    /// G4: compute pipelines (shader module + bind-group layout + pipeline) bucketed by kernel key and
    /// matched by the exact module they were built from (card 656), built once and reused across
    /// tokens. `RefCell` for interior mutability on the `&self` dispatch path (single-threaded, like
    /// the PTX runtime's module cache).
    pipelines: RefCell<HashMap<String, Vec<CachedPipeline>>>,
    /// G4: count of pipeline builds (cache misses) and `queue.submit` calls, so a test can assert the
    /// cache is warm (0 builds after token 0) and one submit per token (SC-002).
    pipeline_builds: Counter,
    submits: Counter,
    native_submits: Counter,
    waits: Counter,
    dispatches: Counter,
    compute_passes: Counter,

    /// Card 156 phase 3/4: count of `create_buffer` (`alloc_f32`), `create_buffer_init` (length/meta
    /// buffers), and `create_bind_group` calls, so a test can assert the cached decode path makes none
    /// after token 0 (SC-001). Every dispatch path increments them; a test reads the delta across the
    /// measured step, as with `pipeline_builds`/`submits`.
    buffer_allocs: Counter,
    bind_group_creates: Counter,
    length_buffer_creates: Counter,
    /// Count of device-to-host readbacks, including the final logits readback; the delta shows a replay
    /// has no intermediate router-id readback.
    readbacks: Counter,
    /// Card 375f: the subset of `readbacks` that allocated a staging buffer for the call:
    /// [`Context::download_f32`], [`Context::read_bytes`] and [`Context::download_segments_u32`]. Card
    /// 547a: every one of them always allocates its own staging buffer, so this always
    /// equals `readbacks`; a session step's delta is still the number of result downloads it makes.
    /// Separate from `buffer_allocs`, which counts graph-visible allocations that the cached-decode
    /// assertions pin at zero per step.
    readback_staging_allocs: Cell<usize>,
    /// Card 547a: live/peak bytes and allocation count by role, the one memory service every runtime
    /// now keeps (`Engine::stats().memory` reads this through [`Context::memory`] /
    /// [`DeviceBuffer`]'s own guard, not an engine-side tally).
    pub(crate) memory: poot_runtime_common::MemoryCounters,
    /// Card 552: typed, purpose-tagged native call/transfer counters (the cross-backend telemetry
    /// contract, `poot_runtime_common::telemetry`), read through [`Context::execution_counters`].
    /// Independent of `pipeline_builds`/`submits`/etc above, which stay as this backend's own
    /// diagnostic gauges; this is the shape 548/549 match for their own backends.
    pub(crate) exec_counters: RefCell<poot_runtime_common::ExecutionCounters>,
    /// Card 531c: `dispatch_dev`/`submit_cached` have no
    /// per-dispatch sync, so an asserting dispatch's error word is staged here instead of read back
    /// immediately. `Context::stage_pending_faults` drains this and copies every entry into a small
    /// staging buffer as part of the next call that already syncs with the device (any `download_f32`/
    /// `download_segments_u32`/`read_bytes`/`flush_faults_and_wait` call, or `Context::dispatch`'s own
    /// sync). Real pending state, not a perf counter.
    pending_faults: RefCell<Vec<PendingFault>>,
}

/// A Vulkan device's active compute units from AMD's shader-core properties:
/// `VK_AMD_shader_core_properties2`'s `activeComputeUnitCount`, else `VK_AMD_shader_core_properties`'
/// engine x array x per-array product; `None` when the driver exposes neither (every non-AMD device).
/// One probe for both Vulkan runtimes (this crate's wgpu adapter and `poot-vulkan-runtime`'s ash
/// instance; card 653).
///
/// # Safety
///
/// `physical_device` must come from `instance`'s own enumeration, and `instance` must be live.
pub unsafe fn amd_vulkan_compute_units(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
) -> Option<u32> {
    use ash::{amd, vk};
    // SAFETY: the caller's contract makes both handles valid; the enumeration and property queries
    // only read, and each property struct is chained only when its extension is listed as supported.
    unsafe {
        let extensions = instance
            .enumerate_device_extension_properties(physical_device)
            .ok()?;
        let supports = |name: &std::ffi::CStr| {
            extensions
                .iter()
                .any(|ext| ext.extension_name_as_c_str() == Ok(name))
        };
        if supports(amd::shader_core_properties2::NAME) {
            let mut core = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            instance.get_physical_device_properties2(physical_device, &mut props);
            Some(core.active_compute_unit_count)
        } else if supports(amd::shader_core_properties::NAME) {
            let mut core = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            instance.get_physical_device_properties2(physical_device, &mut props);
            Some(
                core.shader_engine_count
                    * core.shader_arrays_per_engine_count
                    * core.compute_units_per_shader_array,
            )
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests;

pub use context::counters::{RuntimeCounterSnapshot, RuntimeCounters};

use context::counters::Counter;
