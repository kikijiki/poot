//! Hand-written HSA Foundation runtime and AMD-extension declarations: types, constants, and the C
//! declaration of every function the crate calls.
//!
//! Only what this crate uses is declared (function-pointer table in `src/ffi.rs`); bindgen over the
//! full HSA headers would emit ~250 KB of Rust and needs ~25 GB of RAM at peak. The HSA Foundation
//! ABI is a published spec, so these declarations and constants are stable across ROCm releases.
//!
//! References: ROCm 7.2.3 `include/hsa/hsa.h` and `include/hsa/hsa_ext_amd.h` (ROCR-Runtime
//! `runtime/hsa-runtime/inc/`).

// These type aliases mirror the HSA C API's snake_case names 1:1 so the bindings stay diffable
// against the C headers.
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};

// Status type

/// HSA status code (32-bit). Zero is success; nonzero is error class + specific error.
pub type hsa_status_t = u32;

/// `HSA_STATUS_SUCCESS = 0`, the only success code.
pub const HSA_STATUS_SUCCESS: hsa_status_t = 0;

// Opaque agent / queue / pool / signal handles (all 64-bit)

/// HSA agent handle. `handle == 0` means "no agent".
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct hsa_agent_t {
    pub handle: u64,
}

/// HSA ISA handle. Same layout as `hsa_agent_t` but a different object class: passing one where the
/// other is expected returns `HSA_STATUS_ERROR_INVALID_INDEX`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct hsa_isa_t {
    pub handle: u64,
}

/// HSA signal handle. A signal is a 64-bit value visible to both CPU and GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct hsa_signal_t {
    pub handle: u64,
}

/// HSA queue handle. A queue is a ring of AQL packets; opaque to the host.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hsa_queue_t {
    _private: [u8; 0],
}

/// AMD memory pool handle (AMD extension; distinct from the plain HSA region handle).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct hsa_amd_memory_pool_t {
    pub handle: u64,
}

/// Signal value type. 64-bit signed in `HSA_LARGE_MODEL` (which is what amdgcn uses).
pub type hsa_signal_value_t = i64;

/// Queue type enum (32-bit).
pub type hsa_queue_type32_t = u32;

// hsa.h agent info attribute ids

pub const HSA_AGENT_INFO_NAME: u32 = 0;
pub const HSA_AGENT_INFO_VENDOR_NAME: u32 = 1;
pub const HSA_AGENT_INFO_QUEUE_MAX_SIZE: u32 = 14;
pub const HSA_AGENT_INFO_WAVEFRONT_SIZE: u32 = 6;
/// Returns an `hsa_device_type_t` (CPU=0, GPU=1, DSP=2). Query this for device type, not
/// `HSA_AGENT_INFO_FEATURE` (a feature bitmask).
pub const HSA_AGENT_INFO_DEVICE: u32 = 17;
pub const HSA_AGENT_INFO_ISA: u32 = 19;

// hsa.h device types

pub const HSA_DEVICE_TYPE_CPU: u32 = 0;
pub const HSA_DEVICE_TYPE_GPU: u32 = 1;

// hsa.h queue types
// Values match installed ROCm `hsa.h` (`HSA_QUEUE_TYPE_MULTI = 0`, `SINGLE = 1`,
// `COOPERATIVE = 2`). Do not swap MULTI/SINGLE: ROCr's device-memory ring path and ordinary
// publication go through MULTI (=0).

pub const HSA_QUEUE_TYPE_MULTI: hsa_queue_type32_t = 0;
#[cfg(test)]
pub(crate) const HSA_QUEUE_TYPE_SINGLE: hsa_queue_type32_t = 1;
pub const HSA_QUEUE_TYPE_COOPERATIVE: hsa_queue_type32_t = 2;

// hsa.h queue feature bits

pub const HSA_QUEUE_FEATURE_KERNEL_DISPATCH: u32 = 1;

// hsa.h ISA info attribute ids

pub const HSA_ISA_INFO_NAME_LENGTH: u32 = 0;
pub const HSA_ISA_INFO_NAME: u32 = 1;

// hsa_ext_amd.h agent info attribute ids (extension range)

/// The agent's compute-unit count (`uint32_t`).
pub const HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT: u32 = 0xA002;
pub const HSA_AMD_AGENT_INFO_BDFID: u32 = 0xA006;
pub const HSA_AMD_AGENT_INFO_PRODUCT_NAME: u32 = 0xA009;
pub const HSA_AMD_AGENT_INFO_DOMAIN: u32 = 0xA00F;
pub const HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES: u32 = 0xA010;
/// Bit-mask (`uint8_t[8]`) of agent memory properties. Test bits with `hsa_flag_isset64` using
/// [`HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU`] as the bit index (the enum value is `1 << 0`, not a
/// mask against the first byte).
pub const HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES: u32 = 0xA114;
/// Bit index for `hsa_flag_isset64` on MEMORY_PROPERTIES (header value `1 << 0`). Do not `&` it
/// against the raw property bytes.
pub const HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU: u32 = 1 << 0;
/// "The amount of memory available in bytes across all global pools owned by the agent" (uint64_t),
/// per `hsa_ext_amd.h`. The ROCm/HSA analogue of CUDA's `cuMemGetInfo` free component; present at
/// slot 0xA015 in ROCm runtime headers 6.4.3, 7.2.3 and TheRock 7.13.0a20260515. There is no "used"
/// attribute; `RocmContext::vram_used_bytes` derives it as `total - avail` against the cached total
/// from `HSA_AMD_MEMORY_POOL_INFO_SIZE`.
pub const HSA_AMD_AGENT_INFO_MEMORY_AVAIL: u32 = 0xA015;

// hsa_ext_amd.h memory pool info attribute ids

pub const HSA_AMD_MEMORY_POOL_INFO_SEGMENT: u32 = 0;
pub const HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: u32 = 1;
/// Size of this pool in bytes (`size_t` in the header, read as `u64`; ROCm/HSA is 64-bit only).
/// Summed over every `HSA_AMD_SEGMENT_GLOBAL` pool of the GPU agent, this is the "total" half of the
/// VRAM metric (used = total - `HSA_AMD_AGENT_INFO_MEMORY_AVAIL`).
pub const HSA_AMD_MEMORY_POOL_INFO_SIZE: u32 = 2;
pub const HSA_AMD_MEMORY_POOL_INFO_LOCATION: u32 = 17;
pub const HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: u32 = 5;

// hsa_ext_amd.h segment + location enum values

pub const HSA_AMD_SEGMENT_GLOBAL: u32 = 0;
pub const HSA_AMD_MEMORY_POOL_LOCATION_CPU: u32 = 0;
pub const HSA_AMD_MEMORY_POOL_LOCATION_GPU: u32 = 1;

// hsa_ext_amd.h global-flag bitmasks

pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: u32 = 1;
pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED: u32 = 2;
pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED: u32 = 4;
pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED: u32 = 8;

// hsa_ext_amd.h agent<->pool relationship attrs + link types

pub const HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS: u32 = 1;
pub const HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO: u32 = 2;

pub const HSA_AMD_LINK_INFO_TYPE_PCIE: u32 = 2;

/// One hop on the path from an agent to a foreign memory pool (`hsa_amd_memory_pool_link_info_t`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hsa_amd_memory_pool_link_info_t {
    pub min_latency: u32,
    pub max_latency: u32,
    pub min_bandwidth: u32,
    pub max_bandwidth: u32,
    pub atomic_support_32bit: bool,
    pub atomic_support_64bit: bool,
    pub coherent_support: bool,
    pub link_type: u32,
    pub numa_distance: u32,
}

// AQL kernel dispatch packet (hsa.h `hsa_kernel_dispatch_packet_t`)
// 64-byte aligned, little-endian, layout fixed by the HSA spec. `kernel_object` is the EXEC symbol
// handle from `hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, ...)`.

/// 64-byte header packed into 16-bit fields. Bit layout (HSA spec):
///   bits  0..7  = HSA_PACKET_TYPE (= 2 for kernel dispatch)
///   bits  8..8  = HSA_PACKET_HEADER_BARRIER (0/1)
///   bits  9..10 = HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE (0/1/2)
///   bits 11..12 = HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE (0/1/2)
pub type hsa_packet_header_t = u16;

/// Bit field of `setup` for a kernel dispatch packet:
///   bits 0..1 = number of dimensions - 1 (1-D = 0, 2-D = 1, 3-D = 2)
pub type hsa_kernel_dispatch_packet_setup_t = u16;

// AQL kernel-dispatch packet. The HSA Foundation `hsa_kernel_dispatch_packet_s` is conditional on
// `HSA_LARGE_MODEL` (Strix Halo + amd64): the LITTLE_ENDIAN branch has an extra `reserved1` u32
// between `kernarg_address` and `reserved2`. Under LARGE_MODEL the layout is exactly 64 bytes:
//
//   0  uint16 header           32  u64 kernel_object
//   2  uint16 setup            40  ptr  kernarg_address
//   4  uint16 workgroup_size_x 48  u64  reserved2
//   6  uint16 workgroup_size_y 56  u64  completion_signal
//   8  uint16 workgroup_size_z
//  10  uint16 reserved0
//  12  u32 grid_size_x
//  16  u32 grid_size_y
//  20  u32 grid_size_z
//  24  u32 private_segment_size
//  28  u32 group_segment_size
//
// Including `reserved1` makes the struct 72 bytes, so writing a packet overflows 8 bytes into the
// next queue slot and the second dispatch fails with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct hsa_kernel_dispatch_packet_t {
    pub header: hsa_packet_header_t,
    pub setup: hsa_kernel_dispatch_packet_setup_t,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: *mut c_void,
    pub reserved2: u64,
    pub completion_signal: hsa_signal_t,
}

// The struct must be exactly 64 bytes (the AQL slot size) under HSA_LARGE_MODEL; otherwise
// `std::ptr::write(slot, packet)` overwrites the next slot's AQL packet.
const _: () = assert!(
    std::mem::size_of::<hsa_kernel_dispatch_packet_t>() == 64,
    "hsa_kernel_dispatch_packet_t must be 64 bytes (HSA_LARGE_MODEL); check the layout"
);

// C callback types (declared so ffi.rs's `unsafe extern "C" fn(...)` items type-check)

/// `hsa_iterate_agents` callback. Return 0 to continue, nonzero to stop.
pub type HsaAgentIterateCb =
    unsafe extern "C" fn(agent: hsa_agent_t, data: *mut c_void) -> hsa_status_t;

/// `hsa_amd_agent_iterate_memory_pools` callback. Return 0 to continue, nonzero to stop.
pub type HsaAmdMemoryPoolIterateCb =
    unsafe extern "C" fn(pool: hsa_amd_memory_pool_t, data: *mut c_void) -> hsa_status_t;

/// `hsa_queue_create` callback (the asynchronous error callback; `None` is passed).
pub type HsaQueueCreateCb =
    Option<unsafe extern "C" fn(status: hsa_status_t, queue: *mut hsa_queue_t, data: *mut c_void)>;

// Code-object / executable / signal handles (HSACO loading + AQL dispatch)

/// Code object reader (opaque; freed via `hsa_code_object_reader_destroy`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hsa_code_object_reader_t {
    pub handle: u64,
}

/// HSA executable handle. Created by `hsa_executable_create_alt`, loaded with one or more
/// code objects, and frozen before symbol queries are valid.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hsa_executable_t {
    pub handle: u64,
}

/// Loaded code object handle (returned by `hsa_executable_load_agent_code_object`); only passed back
/// to the runtime on free.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hsa_loaded_code_object_t {
    pub handle: u64,
}

/// Float rounding mode for the executable: `HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = 1`
/// (round-to-nearest-even).
pub type hsa_default_float_rounding_mode_t = u32;

/// Profile: `HSA_PROFILE_FULL = 1` (only full profile is supported for amdgcn kernels).
pub const HSA_PROFILE_FULL: u32 = 1;
pub const HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: u32 = 1;

/// `hsa_profile_t` enum, aliased to u32 for the C ABI.
pub type hsa_profile_t = u32;

/// `hsa_executable_symbol_t` is opaque (resolves to a kernel-object handle).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hsa_executable_symbol_t {
    pub handle: u64,
}

/// Executable symbol info attributes queried.
pub const HSA_EXECUTABLE_SYMBOL_INFO_TYPE: u32 = 0;
pub const HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH: u32 = 1;
pub const HSA_EXECUTABLE_SYMBOL_INFO_NAME: u32 = 2;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 11;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 13;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 14;

/// Symbol kinds (returned by `HSA_EXECUTABLE_SYMBOL_INFO_TYPE`): `HSA_SYMBOL_KIND_VARIABLE = 0`,
/// `HSA_SYMBOL_KIND_KERNEL = 1`, `HSA_SYMBOL_KIND_INDIRECT_FUNCTION = 2`.
pub const HSA_SYMBOL_KIND_KERNEL: u32 = 1;

/// Callback signature for `hsa_executable_iterate_symbols`.
pub type hsa_executable_iterate_symbols_cb_t = Option<
    unsafe extern "C" fn(hsa_executable_t, hsa_executable_symbol_t, *mut c_void) -> hsa_status_t,
>;

// hsa.h signal wait conditions (for hsa_signal_wait_scacquire's condition arg)

pub const HSA_SIGNAL_CONDITION_LT: u32 = 0;

// hsa.h wait states (for hsa_signal_wait_scacquire's state_hint arg)

pub const HSA_WAIT_STATE_BLOCKED: u32 = 1;

// Re-export c_char / c_void so callers need no separate `use`.
pub use std::ffi::{c_char as c_char_re, c_void as c_void_re};

// Silence unused-import warnings for the re-exports.
#[allow(dead_code)]
const _: *const c_char = std::ptr::null();

// C enum parameter types (each an `int`-sized enum in the headers)

pub type hsa_agent_info_t = u32;
pub type hsa_isa_info_t = u32;
pub type hsa_signal_condition_t = u32;
pub type hsa_wait_state_t = u32;
pub type hsa_amd_memory_pool_info_t = u32;
pub type hsa_amd_agent_memory_pool_info_t = u32;
pub type hsa_executable_symbol_info_t = u32;

// Header declarations of every HSA function the crate calls, transcribed from ROCm 7.2.3
// `hsa/hsa.h` and `hsa/hsa_ext_amd.h` with the header's parameter names. The crate never calls or
// links these items: `ffi.rs` checks at compile time that each `dlsym` function-pointer type equals
// the declaration here, so a wrong parameter or return type in the table fails the build.
unsafe extern "C" {
    pub(crate) fn hsa_init() -> hsa_status_t;
    pub(crate) fn hsa_shut_down() -> hsa_status_t;
    pub(crate) fn hsa_iterate_agents(
        callback: Option<HsaAgentIterateCb>,
        data: *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_agent_get_info(
        agent: hsa_agent_t,
        attribute: hsa_agent_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_isa_get_info_alt(
        isa: hsa_isa_t,
        attribute: hsa_isa_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_signal_create(
        initial_value: hsa_signal_value_t,
        num_consumers: u32,
        consumers: *const hsa_agent_t,
        signal: *mut hsa_signal_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_signal_destroy(signal: hsa_signal_t) -> hsa_status_t;
    pub(crate) fn hsa_signal_store_release(signal: hsa_signal_t, value: hsa_signal_value_t);
    pub(crate) fn hsa_signal_load_relaxed(signal: hsa_signal_t) -> hsa_signal_value_t;
    pub(crate) fn hsa_signal_wait_scacquire(
        signal: hsa_signal_t,
        condition: hsa_signal_condition_t,
        compare_value: hsa_signal_value_t,
        timeout_hint: u64,
        wait_state_hint: hsa_wait_state_t,
    ) -> hsa_signal_value_t;
    pub(crate) fn hsa_queue_create(
        agent: hsa_agent_t,
        size: u32,
        type_: hsa_queue_type32_t,
        callback: HsaQueueCreateCb,
        data: *mut c_void,
        private_segment_size: u32,
        group_segment_size: u32,
        queue: *mut *mut hsa_queue_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_queue_destroy(queue: *mut hsa_queue_t) -> hsa_status_t;
    pub(crate) fn hsa_queue_load_read_index_scacquire(queue: *const hsa_queue_t) -> u64;
    pub(crate) fn hsa_queue_load_read_index_relaxed(queue: *const hsa_queue_t) -> u64;
    pub(crate) fn hsa_queue_add_write_index_relaxed(queue: *const hsa_queue_t, value: u64) -> u64;
    pub(crate) fn hsa_queue_add_write_index_acq_rel(queue: *const hsa_queue_t, value: u64) -> u64;
    pub(crate) fn hsa_queue_store_write_index_relaxed(queue: *const hsa_queue_t, value: u64);
    pub(crate) fn hsa_amd_agent_iterate_memory_pools(
        agent: hsa_agent_t,
        callback: Option<HsaAmdMemoryPoolIterateCb>,
        data: *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_amd_memory_pool_get_info(
        memory_pool: hsa_amd_memory_pool_t,
        attribute: hsa_amd_memory_pool_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_amd_agent_memory_pool_get_info(
        agent: hsa_agent_t,
        memory_pool: hsa_amd_memory_pool_t,
        attribute: hsa_amd_agent_memory_pool_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_amd_memory_pool_allocate(
        memory_pool: hsa_amd_memory_pool_t,
        size: usize,
        flags: u32,
        ptr: *mut *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_amd_memory_pool_free(ptr: *mut c_void) -> hsa_status_t;
    pub(crate) fn hsa_amd_memory_async_copy(
        dst: *mut c_void,
        dst_agent: hsa_agent_t,
        src: *const c_void,
        src_agent: hsa_agent_t,
        size: usize,
        num_dep_signals: u32,
        dep_signals: *const hsa_signal_t,
        completion_signal: hsa_signal_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_amd_memory_lock(
        host_ptr: *mut c_void,
        size: usize,
        agents: *mut hsa_agent_t,
        num_agent: std::ffi::c_int,
        agent_ptr: *mut *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_amd_memory_unlock(host_ptr: *mut c_void) -> hsa_status_t;
    pub(crate) fn hsa_code_object_reader_create_from_memory(
        code_object: *const c_void,
        size: usize,
        code_object_reader: *mut hsa_code_object_reader_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_code_object_reader_destroy(
        code_object_reader: hsa_code_object_reader_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_executable_create_alt(
        profile: hsa_profile_t,
        default_float_rounding_mode: hsa_default_float_rounding_mode_t,
        options: *const c_char,
        executable: *mut hsa_executable_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_executable_load_agent_code_object(
        executable: hsa_executable_t,
        agent: hsa_agent_t,
        code_object_reader: hsa_code_object_reader_t,
        options: *const c_char,
        loaded_code_object: *mut hsa_loaded_code_object_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_executable_freeze(
        executable: hsa_executable_t,
        options: *const c_char,
    ) -> hsa_status_t;
    pub(crate) fn hsa_executable_destroy(executable: hsa_executable_t) -> hsa_status_t;
    pub(crate) fn hsa_executable_get_symbol_by_name(
        executable: hsa_executable_t,
        symbol_name: *const c_char,
        agent: *const hsa_agent_t,
        symbol: *mut hsa_executable_symbol_t,
    ) -> hsa_status_t;
    pub(crate) fn hsa_executable_symbol_get_info(
        executable_symbol: hsa_executable_symbol_t,
        attribute: hsa_executable_symbol_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;
    pub(crate) fn hsa_executable_iterate_symbols(
        executable: hsa_executable_t,
        callback: hsa_executable_iterate_symbols_cb_t,
        data: *mut c_void,
    ) -> hsa_status_t;
}
