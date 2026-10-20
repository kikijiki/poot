//! HSA Foundation runtime API, loaded dynamically from `libhsa-runtime64.so.1`.
//!
//! Functions are not called through `bindings::hsa_init()` etc., which would force link-time
//! resolution (spec 063 FR-004). Each is an `unsafe extern "C"` function pointer resolved once via
//! [`Hsa::load`]. The `bindings` module supplies the types and the header declaration of each
//! function; a compile-time check below ties every pointer type to its declaration without calling
//! or linking the symbol.

use std::ffi::{CStr, c_char, c_void};

use crate::bindings;
use poot_runtime_common::{Library, LoadScope};

/// Loaded HSA library + resolved function pointers. Lives inside [`RocmContext`](crate::RocmContext)
/// and is dropped after the queue / signals / pools so no symbol is invoked after unload.
pub(crate) struct Hsa {
    _lib: Library,
    pub(crate) funcs: Funcs,
}

impl Hsa {
    /// Try to dlopen libhsa-runtime64.so.1 and resolve every needed function. Returns
    /// `RocmError::LibraryNotFound` if the .so is not on the loader path, or
    /// `RocmError::Driver(HSA_STATUS_ERROR)` if a required symbol is missing.
    pub(crate) fn load() -> Result<Self, super::RocmError> {
        // SAFETY: each candidate names the ROCm HSA runtime, whose initializers are sound to run.
        let lib = unsafe {
            Library::open_first(
                &[
                    c"libhsa-runtime64.so.1",
                    c"libhsa-runtime64.so",
                    c"libhsa-runtime64.so.1.2.3",
                ],
                LoadScope::Global,
            )
        }
        .map_err(|error| super::RocmError::LibraryNotFound(error.to_string()))?;

        // SAFETY: `lib` was opened under an HSA runtime soname.
        let funcs = unsafe { Funcs::resolve(&lib) }.map_err(super::RocmError::MissingSymbol)?;
        Ok(Hsa { _lib: lib, funcs })
    }
}

macro_rules! fnptr {
    ($name:ident, $ret:ty, $( $arg:ty ),*) => {
        pub(crate) type $name = unsafe extern "C" fn($( $arg ),*) -> $ret;
    };
}

// Function pointer types, one per HSA function called. Signatures match the HSA headers;
// `Funcs::resolve` casts each `dlsym` result to the right type.
fnptr!(HsaInit, bindings::hsa_status_t,);
fnptr!(HsaShutDown, bindings::hsa_status_t,);
fnptr!(
    HsaIterateAgents,
    bindings::hsa_status_t,
    Option<unsafe extern "C" fn(bindings::hsa_agent_t, *mut c_void) -> bindings::hsa_status_t>,
    *mut c_void
);
fnptr!(
    HsaAgentGetInfo,
    bindings::hsa_status_t,
    bindings::hsa_agent_t,
    u32,
    *mut c_void
);
fnptr!(
    HsaAgentIterateRegions,
    bindings::hsa_status_t,
    bindings::hsa_agent_t,
    Option<bindings::HsaRegionIterateCb>,
    *mut c_void
);
fnptr!(
    HsaRegionGetInfo,
    bindings::hsa_status_t,
    bindings::hsa_region_t,
    u32,
    *mut c_void
);
fnptr!(
    HsaIsaGetInfoAlt,
    bindings::hsa_status_t,
    bindings::hsa_isa_t,
    u32,
    *mut c_void
);

fnptr!(HsaSystemGetInfo, bindings::hsa_status_t, u32, *mut c_void);
fnptr!(
    HsaAmdProfilingSetProfilerEnabled,
    bindings::hsa_status_t,
    *mut bindings::hsa_queue_t,
    std::ffi::c_int
);
fnptr!(
    HsaAmdProfilingGetDispatchTime,
    bindings::hsa_status_t,
    bindings::hsa_agent_t,
    bindings::hsa_signal_t,
    *mut bindings::hsa_amd_profiling_dispatch_time_t
);
fnptr!(
    HsaSignalCreate,
    bindings::hsa_status_t,
    bindings::hsa_signal_value_t,
    u32,
    *const bindings::hsa_agent_t,
    *mut bindings::hsa_signal_t
);
fnptr!(
    HsaSignalDestroy,
    bindings::hsa_status_t,
    bindings::hsa_signal_t
);
fnptr!(
    HsaSignalStoreRelease,
    (),
    bindings::hsa_signal_t,
    bindings::hsa_signal_value_t
);
fnptr!(
    HsaSignalLoadRelaxed,
    bindings::hsa_signal_value_t,
    bindings::hsa_signal_t
);
fnptr!(
    HsaSignalWaitScacquire,
    bindings::hsa_signal_value_t,
    bindings::hsa_signal_t,
    u32,
    bindings::hsa_signal_value_t,
    u64,
    u32
);

fnptr!(
    HsaQueueCreate,
    bindings::hsa_status_t,
    bindings::hsa_agent_t,
    u32,
    bindings::hsa_queue_type32_t,
    Option<unsafe extern "C" fn(bindings::hsa_status_t, *mut bindings::hsa_queue_t, *mut c_void)>,
    *mut c_void,
    u32,
    u32,
    *mut *mut bindings::hsa_queue_t
);
fnptr!(
    HsaQueueDestroy,
    bindings::hsa_status_t,
    *mut bindings::hsa_queue_t
);
fnptr!(
    HsaQueueLoadReadIndexScacquire,
    u64,
    *const bindings::hsa_queue_t
);
fnptr!(
    HsaQueueLoadReadIndexRelaxed,
    u64,
    *const bindings::hsa_queue_t
);
fnptr!(
    HsaQueueAddWriteIndexRelaxed,
    u64,
    *const bindings::hsa_queue_t,
    u64
);
fnptr!(
    HsaQueueAddWriteIndexAcqRel,
    u64,
    *const bindings::hsa_queue_t,
    u64
);

fnptr!(
    HsaAmdAgentIterateMemoryPools,
    bindings::hsa_status_t,
    bindings::hsa_agent_t,
    Option<
        unsafe extern "C" fn(
            bindings::hsa_amd_memory_pool_t,
            *mut c_void,
        ) -> bindings::hsa_status_t,
    >,
    *mut c_void
);
fnptr!(
    HsaAmdMemoryPoolGetInfo,
    bindings::hsa_status_t,
    bindings::hsa_amd_memory_pool_t,
    u32,
    *mut c_void
);
fnptr!(
    HsaAmdAgentMemoryPoolGetInfo,
    bindings::hsa_status_t,
    bindings::hsa_agent_t,
    bindings::hsa_amd_memory_pool_t,
    u32,
    *mut c_void
);
fnptr!(
    HsaAmdMemoryPoolAllocate,
    bindings::hsa_status_t,
    bindings::hsa_amd_memory_pool_t,
    usize,
    u32,
    *mut *mut c_void
);
fnptr!(HsaAmdMemoryPoolFree, bindings::hsa_status_t, *mut c_void);

// Card 125: discrete-GPU upload DMA (spec 120 FR-004/SC-004).
// `hsa_amd_memory_async_copy` is the device DMA transfer used when the destination pool is not
// host-visible. `hsa_amd_memory_lock`/`hsa_amd_memory_unlock` pin an arbitrary host pointer (a plain
// Rust slice) as a valid DMA source/dest: the copy requires agent-accessible memory, which
// unregistered heap memory is not until locked.
fnptr!(
    HsaAmdMemoryAsyncCopy,
    bindings::hsa_status_t,
    *mut c_void,
    bindings::hsa_agent_t,
    *const c_void,
    bindings::hsa_agent_t,
    usize,
    u32,
    *const bindings::hsa_signal_t,
    bindings::hsa_signal_t
);
// Zero-fill of a fresh allocation on the device: `hsa_amd_memory_fill(ptr, value, count)` writes `count`
// 32-bit words of `value`, so a new buffer is zeroed without a host buffer and a DMA.
fnptr!(
    HsaAmdMemoryFill,
    bindings::hsa_status_t,
    *mut c_void,
    u32,
    usize
);
fnptr!(
    HsaAmdMemoryLock,
    bindings::hsa_status_t,
    *mut c_void,
    usize,
    *mut bindings::hsa_agent_t,
    i32,
    *mut *mut c_void
);
fnptr!(HsaAmdMemoryUnlock, bindings::hsa_status_t, *mut c_void);

// Code-object loading + executable + symbol queries

fnptr!(
    HsaCodeObjectReaderCreateFromMemory,
    bindings::hsa_status_t,
    *const c_void,
    usize,
    *mut bindings::hsa_code_object_reader_t
);
fnptr!(
    HsaCodeObjectReaderDestroy,
    bindings::hsa_status_t,
    bindings::hsa_code_object_reader_t
);

fnptr!(
    HsaExecutableCreateAlt,
    bindings::hsa_status_t,
    bindings::hsa_profile_t,
    bindings::hsa_default_float_rounding_mode_t,
    *const c_char,
    *mut bindings::hsa_executable_t
);
fnptr!(
    HsaExecutableLoadAgentCodeObject,
    bindings::hsa_status_t,
    bindings::hsa_executable_t,
    bindings::hsa_agent_t,
    bindings::hsa_code_object_reader_t,
    *const c_char,
    *mut bindings::hsa_loaded_code_object_t
);
fnptr!(
    HsaExecutableFreeze,
    bindings::hsa_status_t,
    bindings::hsa_executable_t,
    *const c_char
);
fnptr!(
    HsaExecutableDestroy,
    bindings::hsa_status_t,
    bindings::hsa_executable_t
);

fnptr!(
    HsaAmdRegisterSystemEventHandler,
    bindings::hsa_status_t,
    bindings::HsaAmdSystemEventCb,
    *mut c_void
);
fnptr!(
    HsaExecutableGetSymbolByName,
    bindings::hsa_status_t,
    bindings::hsa_executable_t,
    *const c_char,
    *const bindings::hsa_agent_t,
    *mut bindings::hsa_executable_symbol_t
);
fnptr!(
    HsaExecutableSymbolGetInfo,
    bindings::hsa_status_t,
    bindings::hsa_executable_symbol_t,
    u32,
    *mut c_void
);
fnptr!(
    HsaExecutableIterateSymbols,
    bindings::hsa_status_t,
    bindings::hsa_executable_t,
    bindings::hsa_executable_iterate_symbols_cb_t,
    *mut c_void
);

// Each function-pointer type above must equal the header declaration in `bindings`: coercing the
// declared item to the pointer type fails to compile on any parameter or return mismatch. The
// constants are evaluated at compile time and never emitted, so no HSA symbol is linked.
macro_rules! declared_as {
    ($($ty:ident = $decl:ident;)*) => {
        $(const _: $ty = bindings::$decl;)*
    };
}

declared_as! {
    HsaInit = hsa_init;
    HsaShutDown = hsa_shut_down;
    HsaIterateAgents = hsa_iterate_agents;
    HsaAgentGetInfo = hsa_agent_get_info;
    HsaAgentIterateRegions = hsa_agent_iterate_regions;
    HsaRegionGetInfo = hsa_region_get_info;
    HsaIsaGetInfoAlt = hsa_isa_get_info_alt;
    HsaSystemGetInfo = hsa_system_get_info;
    HsaAmdProfilingSetProfilerEnabled = hsa_amd_profiling_set_profiler_enabled;
    HsaAmdProfilingGetDispatchTime = hsa_amd_profiling_get_dispatch_time;
    HsaSignalCreate = hsa_signal_create;
    HsaSignalDestroy = hsa_signal_destroy;
    HsaSignalStoreRelease = hsa_signal_store_release;
    HsaSignalLoadRelaxed = hsa_signal_load_relaxed;
    HsaSignalWaitScacquire = hsa_signal_wait_scacquire;
    HsaQueueCreate = hsa_queue_create;
    HsaQueueDestroy = hsa_queue_destroy;
    HsaQueueLoadReadIndexScacquire = hsa_queue_load_read_index_scacquire;
    HsaQueueLoadReadIndexRelaxed = hsa_queue_load_read_index_relaxed;
    HsaQueueAddWriteIndexRelaxed = hsa_queue_add_write_index_relaxed;
    HsaQueueAddWriteIndexAcqRel = hsa_queue_add_write_index_acq_rel;
    HsaAmdAgentIterateMemoryPools = hsa_amd_agent_iterate_memory_pools;
    HsaAmdMemoryPoolGetInfo = hsa_amd_memory_pool_get_info;
    HsaAmdAgentMemoryPoolGetInfo = hsa_amd_agent_memory_pool_get_info;
    HsaAmdMemoryPoolAllocate = hsa_amd_memory_pool_allocate;
    HsaAmdMemoryPoolFree = hsa_amd_memory_pool_free;
    HsaAmdMemoryAsyncCopy = hsa_amd_memory_async_copy;
    HsaAmdMemoryFill = hsa_amd_memory_fill;
    HsaAmdMemoryLock = hsa_amd_memory_lock;
    HsaAmdMemoryUnlock = hsa_amd_memory_unlock;
    HsaAmdRegisterSystemEventHandler = hsa_amd_register_system_event_handler;
    HsaCodeObjectReaderCreateFromMemory = hsa_code_object_reader_create_from_memory;
    HsaCodeObjectReaderDestroy = hsa_code_object_reader_destroy;
    HsaExecutableCreateAlt = hsa_executable_create_alt;
    HsaExecutableLoadAgentCodeObject = hsa_executable_load_agent_code_object;
    HsaExecutableFreeze = hsa_executable_freeze;
    HsaExecutableDestroy = hsa_executable_destroy;
    HsaExecutableGetSymbolByName = hsa_executable_get_symbol_by_name;
    HsaExecutableSymbolGetInfo = hsa_executable_symbol_get_info;
    HsaExecutableIterateSymbols = hsa_executable_iterate_symbols;
}

/// Resolve `name` in `lib` as a function pointer of type `T`.
///
/// # Safety
///
/// `T` must be the `unsafe extern "C" fn` type of the symbol's C declaration in `lib`.
unsafe fn resolve_fn<T: Copy>(lib: &Library, name: &CStr) -> Option<T> {
    assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<*mut c_void>(),
        "a resolved symbol must be a function pointer"
    );
    let address = lib.symbol(name)?;
    // SAFETY: `address` is the non-null address of the function `name`, and `T` is a function-pointer
    // type (same size as a data pointer, checked above) matching its declaration (caller contract).
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&address) })
}

/// All HSA entry points used, resolved once at [`Hsa::load`]. Each dispatch is one load plus an
/// indirect call, with no `dlsym` per call.
pub(crate) struct Funcs {
    /// `hsa_init`: initialises the HSA runtime (reference-counted).
    pub(crate) hsa_init: HsaInit,
    /// `hsa_shut_down`: releases one `hsa_init` reference.
    pub(crate) hsa_shut_down: HsaShutDown,
    /// `hsa_iterate_agents`: enumerates the CPU and GPU agents through a callback.
    pub(crate) hsa_iterate_agents: HsaIterateAgents,
    /// `hsa_agent_get_info`: reads an `HSA_AGENT_INFO_*` attribute of an agent.
    pub(crate) hsa_agent_get_info: HsaAgentGetInfo,
    /// `hsa_agent_iterate_regions`: enumerates an agent's regions (segments) through a callback.
    pub(crate) hsa_agent_iterate_regions: HsaAgentIterateRegions,
    /// `hsa_region_get_info`: reads an `HSA_REGION_INFO_*` attribute of a region.
    pub(crate) hsa_region_get_info: HsaRegionGetInfo,
    /// `hsa_isa_get_info_alt`: reads an `HSA_ISA_INFO_*` attribute of an ISA.
    pub(crate) hsa_isa_get_info_alt: HsaIsaGetInfoAlt,
    /// `hsa_system_get_info`: reads an `HSA_SYSTEM_INFO_*` attribute (timestamp frequency).
    pub(crate) hsa_system_get_info: HsaSystemGetInfo,
    /// `hsa_amd_profiling_set_profiler_enabled`: turns dispatch timestamping on or off for a queue.
    pub(crate) hsa_amd_profiling_set_profiler_enabled: HsaAmdProfilingSetProfilerEnabled,
    /// `hsa_amd_profiling_get_dispatch_time`: reads a completed dispatch's start and end ticks from its signal.
    pub(crate) hsa_amd_profiling_get_dispatch_time: HsaAmdProfilingGetDispatchTime,
    /// `hsa_signal_create`: creates a completion signal with an initial value.
    pub(crate) hsa_signal_create: HsaSignalCreate,
    /// `hsa_signal_destroy`: destroys a signal.
    pub(crate) hsa_signal_destroy: HsaSignalDestroy,
    /// `hsa_signal_store_release`: stores a signal value with release ordering.
    pub(crate) hsa_signal_store_release: HsaSignalStoreRelease,
    /// `hsa_signal_load_relaxed`: loads a signal value with relaxed ordering.
    pub(crate) hsa_signal_load_relaxed: HsaSignalLoadRelaxed,
    /// `hsa_signal_wait_scacquire`: waits on a signal condition with system-scope acquire ordering, bounded by a timeout.
    pub(crate) hsa_signal_wait_scacquire: HsaSignalWaitScacquire,
    /// `hsa_queue_create`: creates an AQL queue on an agent.
    pub(crate) hsa_queue_create: HsaQueueCreate,
    /// `hsa_queue_destroy`: destroys a queue.
    pub(crate) hsa_queue_destroy: HsaQueueDestroy,
    /// `hsa_queue_load_read_index_scacquire`: loads the queue read index with acquire ordering.
    pub(crate) hsa_queue_load_read_index_scacquire: HsaQueueLoadReadIndexScacquire,
    /// `hsa_queue_load_read_index_relaxed`: loads the queue read index with relaxed ordering.
    pub(crate) hsa_queue_load_read_index_relaxed: HsaQueueLoadReadIndexRelaxed,
    /// `hsa_queue_add_write_index_relaxed`: reserves packet slots by advancing the write index, relaxed.
    pub(crate) hsa_queue_add_write_index_relaxed: HsaQueueAddWriteIndexRelaxed,
    /// `hsa_queue_add_write_index_acq_rel`: reserves packet slots by advancing the write index, acquire-release.
    pub(crate) hsa_queue_add_write_index_acq_rel: HsaQueueAddWriteIndexAcqRel,
    /// `hsa_amd_agent_iterate_memory_pools`: enumerates an agent's memory pools through a callback.
    pub(crate) hsa_amd_agent_iterate_memory_pools: HsaAmdAgentIterateMemoryPools,
    /// `hsa_amd_memory_pool_get_info`: reads an `HSA_AMD_MEMORY_POOL_INFO_*` attribute of a pool.
    pub(crate) hsa_amd_memory_pool_get_info: HsaAmdMemoryPoolGetInfo,
    /// `hsa_amd_agent_memory_pool_get_info`: reads an agent-to-pool attribute (link hops, link info).
    pub(crate) hsa_amd_agent_memory_pool_get_info: HsaAmdAgentMemoryPoolGetInfo,
    /// `hsa_amd_memory_pool_allocate`: allocates memory from a pool.
    pub(crate) hsa_amd_memory_pool_allocate: HsaAmdMemoryPoolAllocate,
    /// `hsa_amd_memory_pool_free`: frees pool memory.
    pub(crate) hsa_amd_memory_pool_free: HsaAmdMemoryPoolFree,
    /// `hsa_amd_memory_async_copy`: starts an asynchronous (DMA) copy between agents, completing on a signal.
    pub(crate) hsa_amd_memory_async_copy: HsaAmdMemoryAsyncCopy,
    /// `hsa_amd_memory_fill`: fills device memory with a 32-bit pattern.
    pub(crate) hsa_amd_memory_fill: HsaAmdMemoryFill,
    /// `hsa_amd_memory_lock`: pins host memory and maps it for agent access.
    pub(crate) hsa_amd_memory_lock: HsaAmdMemoryLock,
    /// `hsa_amd_memory_unlock`: unpins memory pinned by `hsa_amd_memory_lock`.
    pub(crate) hsa_amd_memory_unlock: HsaAmdMemoryUnlock,
    /// `hsa_amd_register_system_event_handler`: registers the process's one GPU fault handler.
    pub(crate) hsa_amd_register_system_event_handler: HsaAmdRegisterSystemEventHandler,
    // code-object + executable + symbols + signals
    /// `hsa_code_object_reader_create_from_memory`: creates a code-object reader over an in-memory image.
    pub(crate) hsa_code_object_reader_create_from_memory: HsaCodeObjectReaderCreateFromMemory,
    /// `hsa_code_object_reader_destroy`: destroys a code-object reader.
    pub(crate) hsa_code_object_reader_destroy: HsaCodeObjectReaderDestroy,
    /// `hsa_executable_create_alt`: creates an executable with a profile and rounding mode.
    pub(crate) hsa_executable_create_alt: HsaExecutableCreateAlt,
    /// `hsa_executable_load_agent_code_object`: loads a code object into an executable for one agent.
    pub(crate) hsa_executable_load_agent_code_object: HsaExecutableLoadAgentCodeObject,
    /// `hsa_executable_freeze`: freezes an executable so its symbols can be queried.
    pub(crate) hsa_executable_freeze: HsaExecutableFreeze,
    /// `hsa_executable_destroy`: destroys an executable.
    pub(crate) hsa_executable_destroy: HsaExecutableDestroy,
    /// `hsa_executable_get_symbol_by_name`: looks a symbol up by name for an agent.
    pub(crate) hsa_executable_get_symbol_by_name: HsaExecutableGetSymbolByName,
    /// `hsa_executable_symbol_get_info`: reads an `HSA_EXECUTABLE_SYMBOL_INFO_*` attribute of a symbol.
    pub(crate) hsa_executable_symbol_get_info: HsaExecutableSymbolGetInfo,
    /// `hsa_executable_iterate_symbols`: enumerates an executable's symbols through a callback.
    pub(crate) hsa_executable_iterate_symbols: HsaExecutableIterateSymbols,
}

impl Funcs {
    /// Resolve every entry point via `dlsym`. `lib` is borrowed; the pointers stay valid for the
    /// lifetime of the library handle (the .so is held open).
    ///
    /// # Safety
    ///
    /// `lib` must be the HSA runtime library, so each symbol has the header declaration its field type
    /// is checked against (see `declared_as!`).
    unsafe fn resolve(lib: &Library) -> Result<Self, &'static str> {
        macro_rules! sym {
            ($p:ident) => {{
                let name = CStr::from_bytes_with_nul(concat!(stringify!($p), "\0").as_bytes()).unwrap();
                // SAFETY: the field's type equals the header declaration of `$p` (`declared_as!`), and
                // `lib` is the HSA runtime (caller contract).
                unsafe { resolve_fn(lib, name) }.ok_or(stringify!($p))?
            }};
        }
        Ok(Funcs {
            hsa_init: sym!(hsa_init),
            hsa_shut_down: sym!(hsa_shut_down),
            hsa_iterate_agents: sym!(hsa_iterate_agents),
            hsa_agent_get_info: sym!(hsa_agent_get_info),
            hsa_agent_iterate_regions: sym!(hsa_agent_iterate_regions),
            hsa_region_get_info: sym!(hsa_region_get_info),
            hsa_isa_get_info_alt: sym!(hsa_isa_get_info_alt),
            hsa_system_get_info: sym!(hsa_system_get_info),
            hsa_amd_profiling_set_profiler_enabled: sym!(hsa_amd_profiling_set_profiler_enabled),
            hsa_amd_profiling_get_dispatch_time: sym!(hsa_amd_profiling_get_dispatch_time),
            hsa_signal_create: sym!(hsa_signal_create),
            hsa_signal_destroy: sym!(hsa_signal_destroy),
            hsa_signal_store_release: sym!(hsa_signal_store_release),
            hsa_signal_load_relaxed: sym!(hsa_signal_load_relaxed),
            hsa_signal_wait_scacquire: sym!(hsa_signal_wait_scacquire),
            hsa_queue_create: sym!(hsa_queue_create),
            hsa_queue_destroy: sym!(hsa_queue_destroy),
            hsa_queue_load_read_index_scacquire: sym!(hsa_queue_load_read_index_scacquire),
            hsa_queue_load_read_index_relaxed: sym!(hsa_queue_load_read_index_relaxed),
            hsa_queue_add_write_index_relaxed: sym!(hsa_queue_add_write_index_relaxed),
            hsa_queue_add_write_index_acq_rel: sym!(hsa_queue_add_write_index_acq_rel),
            hsa_amd_agent_iterate_memory_pools: sym!(hsa_amd_agent_iterate_memory_pools),
            hsa_amd_memory_pool_get_info: sym!(hsa_amd_memory_pool_get_info),
            hsa_amd_agent_memory_pool_get_info: sym!(hsa_amd_agent_memory_pool_get_info),
            hsa_amd_memory_pool_allocate: sym!(hsa_amd_memory_pool_allocate),
            hsa_amd_memory_pool_free: sym!(hsa_amd_memory_pool_free),
            hsa_amd_memory_async_copy: sym!(hsa_amd_memory_async_copy),
            hsa_amd_memory_fill: sym!(hsa_amd_memory_fill),
            hsa_amd_memory_lock: sym!(hsa_amd_memory_lock),
            hsa_amd_memory_unlock: sym!(hsa_amd_memory_unlock),
            hsa_amd_register_system_event_handler: sym!(hsa_amd_register_system_event_handler),
            hsa_code_object_reader_create_from_memory: sym!(
                hsa_code_object_reader_create_from_memory
            ),
            hsa_code_object_reader_destroy: sym!(hsa_code_object_reader_destroy),
            hsa_executable_create_alt: sym!(hsa_executable_create_alt),
            hsa_executable_load_agent_code_object: sym!(hsa_executable_load_agent_code_object),
            hsa_executable_freeze: sym!(hsa_executable_freeze),
            hsa_executable_destroy: sym!(hsa_executable_destroy),
            hsa_executable_get_symbol_by_name: sym!(hsa_executable_get_symbol_by_name),
            hsa_executable_symbol_get_info: sym!(hsa_executable_symbol_get_info),
            hsa_executable_iterate_symbols: sym!(hsa_executable_iterate_symbols),
        })
    }
}
