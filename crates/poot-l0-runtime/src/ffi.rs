//! The Level Zero C ABI this runtime uses, loaded dynamically from `libze_loader.so.1`.
//!
//! No function is linked: each is an `unsafe extern "C"` pointer resolved once by [`Funcs::load`], so the
//! crate builds without a Level Zero installation. The structure layouts and enum values below are the ones
//! of `ze_api.h` (level-zero 1.28), pinned by the `abi` tests against the sizes the header gives.

use std::ffi::{CStr, c_char, c_void};

use crate::error::L0Error;
use crate::platform::LibHandle;

/// `ze_result_t`.
pub(crate) type ZeResult = u32;
/// An opaque Level Zero handle (`ze_driver_handle_t`, `ze_device_handle_t`, ...).
pub(crate) type Handle = *mut c_void;

pub(crate) const ZE_RESULT_SUCCESS: ZeResult = 0;
pub(crate) const ZE_INIT_FLAG_VPU_ONLY: u32 = 2;
/// `ZE_RESULT_NOT_READY`: a wait with a timeout expired before the queue drained.
pub(crate) const ZE_RESULT_NOT_READY: ZeResult = 1;
pub(crate) const ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 2;
/// `ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE`.
pub(crate) const ZE_QUEUE_GROUP_FLAG_COMPUTE: u32 = 1;

const STYPE_DEVICE_PROPERTIES: u32 = 0x3;
const STYPE_COMMAND_QUEUE_GROUP_PROPERTIES: u32 = 0x6;
const STYPE_CONTEXT_DESC: u32 = 0xd;
const STYPE_COMMAND_QUEUE_DESC: u32 = 0xe;
const STYPE_COMMAND_LIST_DESC: u32 = 0xf;
const STYPE_HOST_MEM_ALLOC_DESC: u32 = 0x16;
// The graph extension's own structure-type numbering (`ze_graph_ext.h`).
const STYPE_GRAPH_ARGUMENT_PROPERTIES: u32 = 0x4;
const STYPE_GRAPH_DESC_2: u32 = 0xE;
const STYPE_DEVICE_GRAPH_PROPERTIES_2: u32 = 0xF;
const STYPE_GRAPH_PROPERTIES_2: u32 = 0x10;

/// `ze_device_properties_t`: the driver writes the whole 368-byte structure, of which only the identity
/// fields are read here.
#[repr(C)]
pub(crate) struct DeviceProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub device_type: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    tail: [u8; 340],
}

impl DeviceProperties {
    pub(crate) fn new() -> Self {
        DeviceProperties {
            stype: STYPE_DEVICE_PROPERTIES,
            p_next: std::ptr::null_mut(),
            device_type: 0,
            vendor_id: 0,
            device_id: 0,
            tail: [0; 340],
        }
    }
}

/// `ze_command_queue_group_properties_t`.
#[repr(C)]
pub(crate) struct QueueGroupProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub flags: u32,
    pub max_memory_fill_pattern_size: usize,
    pub num_queues: u32,
}

impl QueueGroupProperties {
    pub(crate) fn new() -> Self {
        QueueGroupProperties {
            stype: STYPE_COMMAND_QUEUE_GROUP_PROPERTIES,
            p_next: std::ptr::null_mut(),
            flags: 0,
            max_memory_fill_pattern_size: 0,
            num_queues: 0,
        }
    }
}

/// `ze_context_desc_t`.
#[repr(C)]
pub(crate) struct ContextDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

impl ContextDesc {
    pub(crate) fn new() -> Self {
        ContextDesc {
            stype: STYPE_CONTEXT_DESC,
            p_next: std::ptr::null(),
            flags: 0,
        }
    }
}

/// `ze_command_queue_desc_t`.
#[repr(C)]
pub(crate) struct CommandQueueDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}

impl CommandQueueDesc {
    pub(crate) fn asynchronous(ordinal: u32) -> Self {
        CommandQueueDesc {
            stype: STYPE_COMMAND_QUEUE_DESC,
            p_next: std::ptr::null(),
            ordinal,
            index: 0,
            flags: 0,
            mode: ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
            priority: 0,
        }
    }
}

/// `ze_command_list_desc_t`.
#[repr(C)]
pub(crate) struct CommandListDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub ordinal: u32,
    pub flags: u32,
}

impl CommandListDesc {
    pub(crate) fn new(ordinal: u32) -> Self {
        CommandListDesc {
            stype: STYPE_COMMAND_LIST_DESC,
            p_next: std::ptr::null(),
            ordinal,
            flags: 0,
        }
    }
}

/// `ze_host_mem_alloc_desc_t`.
#[repr(C)]
pub(crate) struct HostMemAllocDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

impl HostMemAllocDesc {
    pub(crate) fn new() -> Self {
        HostMemAllocDesc {
            stype: STYPE_HOST_MEM_ALLOC_DESC,
            p_next: std::ptr::null(),
            flags: 0,
        }
    }
}

/// `ZE_GRAPH_FORMAT_NATIVE`: a blob the vendor compiler already produced.
pub(crate) const ZE_GRAPH_FORMAT_NATIVE: u32 = 0x1;
/// `ZE_GRAPH_FORMAT_NGRAPH_LITE`: OpenVINO IR, compiled by the driver's compiler on creation.
pub(crate) const ZE_GRAPH_FORMAT_NGRAPH_LITE: u32 = 0x2;
/// `ZE_GRAPH_FLAG_DISABLE_CACHING`: the driver's own disk cache is neither read nor written.
pub(crate) const ZE_GRAPH_FLAG_DISABLE_CACHING: u32 = 1;
/// `ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE`: the graph needs an `AppendGraphInitialize` before it runs.
pub(crate) const ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE: u32 = 1;
/// `ZE_GRAPH_STAGE_INITIALIZE`: the graph needs a `GraphInitialize` call before it runs.
pub(crate) const ZE_GRAPH_STAGE_INITIALIZE: u32 = 2;
/// `ZE_MAX_GRAPH_ARGUMENT_NAME`.
pub(crate) const GRAPH_ARGUMENT_NAME_LEN: usize = 256;

/// `ze_graph_desc_2_t`.
#[repr(C)]
pub(crate) struct GraphDesc2 {
    pub stype: u32,
    pub p_next: *const c_void,
    pub format: u32,
    pub input_size: usize,
    pub input: *const u8,
    pub build_flags: *const c_char,
    pub flags: u32,
}

impl GraphDesc2 {
    pub(crate) fn new(format: u32, input: &[u8], build_flags: &CStr, flags: u32) -> Self {
        GraphDesc2 {
            stype: STYPE_GRAPH_DESC_2,
            p_next: std::ptr::null(),
            format,
            input_size: input.len(),
            input: input.as_ptr(),
            build_flags: build_flags.as_ptr(),
            flags,
        }
    }
}

/// `ze_device_graph_properties_2_t`.
#[repr(C)]
pub(crate) struct DeviceGraphProperties2 {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub graph_extension_version: u32,
    pub compiler_major: u16,
    pub compiler_minor: u16,
    pub graph_formats_supported: u32,
    pub max_ov_opset_version_supported: u32,
    pub elf_version: [u32; 3],
    pub runtime_version: [u32; 3],
}

impl DeviceGraphProperties2 {
    pub(crate) fn new() -> Self {
        DeviceGraphProperties2 {
            stype: STYPE_DEVICE_GRAPH_PROPERTIES_2,
            p_next: std::ptr::null_mut(),
            graph_extension_version: 0,
            compiler_major: 0,
            compiler_minor: 0,
            graph_formats_supported: 0,
            max_ov_opset_version_supported: 0,
            elf_version: [0; 3],
            runtime_version: [0; 3],
        }
    }
}

/// `ze_graph_properties_2_t`.
#[repr(C)]
pub(crate) struct GraphProperties2 {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub num_graph_args: u32,
    pub init_stage_required: u32,
}

impl GraphProperties2 {
    pub(crate) fn new() -> Self {
        GraphProperties2 {
            stype: STYPE_GRAPH_PROPERTIES_2,
            p_next: std::ptr::null_mut(),
            num_graph_args: 0,
            init_stage_required: 0,
        }
    }
}

/// `ze_graph_argument_properties_t`.
#[repr(C)]
pub(crate) struct GraphArgumentProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub name: [c_char; GRAPH_ARGUMENT_NAME_LEN],
    pub kind: u32,
    pub dims: [u32; 5],
    pub network_precision: u32,
    pub network_layout: u32,
    pub device_precision: u32,
    pub device_layout: u32,
}

impl GraphArgumentProperties {
    pub(crate) fn new() -> Self {
        GraphArgumentProperties {
            stype: STYPE_GRAPH_ARGUMENT_PROPERTIES,
            p_next: std::ptr::null_mut(),
            name: [0; GRAPH_ARGUMENT_NAME_LEN],
            kind: 0,
            dims: [0; 5],
            network_precision: 0,
            network_layout: 0,
            device_precision: 0,
            device_layout: 0,
        }
    }
}

/// The leading entries of `ze_graph_dditable_ext_t`, the table the driver returns for `ZE_extension_graph`
/// (extension version 1.18). Only the entries this runtime calls are typed; the others hold their positions
/// as untyped pointers, because the table is read by offset. An entry a driver does not implement is null.
#[repr(C)]
pub(crate) struct GraphTable {
    _create: *const c_void,
    pub destroy: Option<unsafe extern "C" fn(Handle) -> ZeResult>,
    _get_properties: *const c_void,
    pub get_argument_properties:
        Option<unsafe extern "C" fn(Handle, u32, *mut GraphArgumentProperties) -> ZeResult>,
    pub set_argument_value: Option<unsafe extern "C" fn(Handle, u32, *const c_void) -> ZeResult>,
    pub append_graph_initialize:
        Option<unsafe extern "C" fn(Handle, Handle, Handle, u32, *mut Handle) -> ZeResult>,
    pub append_graph_execute:
        Option<unsafe extern "C" fn(Handle, Handle, Handle, Handle, u32, *mut Handle) -> ZeResult>,
    pub get_native_binary: Option<unsafe extern "C" fn(Handle, *mut usize, *mut u8) -> ZeResult>,
    _device_get_graph_properties: *const c_void,
    _get_argument_metadata: *const c_void,
    _get_argument_properties_2: *const c_void,
    _get_argument_properties_3: *const c_void,
    _query_network_create: *const c_void,
    pub query_network_destroy: Option<unsafe extern "C" fn(Handle) -> ZeResult>,
    pub query_network_get_supported_layers:
        Option<unsafe extern "C" fn(Handle, *mut usize, *mut c_char) -> ZeResult>,
    pub build_log_get_string:
        Option<unsafe extern "C" fn(Handle, *mut u32, *mut c_char) -> ZeResult>,
    pub create_2:
        Option<unsafe extern "C" fn(Handle, Handle, *const GraphDesc2, *mut Handle) -> ZeResult>,
    pub query_network_create_2:
        Option<unsafe extern "C" fn(Handle, Handle, *const GraphDesc2, *mut Handle) -> ZeResult>,
    _query_context_memory: *const c_void,
    pub device_get_graph_properties_2:
        Option<unsafe extern "C" fn(Handle, *mut DeviceGraphProperties2) -> ZeResult>,
    _get_native_binary_2: *const c_void,
    pub get_properties_2: Option<unsafe extern "C" fn(Handle, *mut GraphProperties2) -> ZeResult>,
    pub graph_initialize: Option<unsafe extern "C" fn(Handle) -> ZeResult>,
}

/// Declare the Level Zero function table: one field per entry point, typed by its C signature, resolved from
/// the loader by the symbol name. Every entry point returns a `ze_result_t`.
macro_rules! level_zero_functions {
    ($( $(#[$meta:meta])* $field:ident = $symbol:literal ( $( $arg:ty ),* ); )*) => {
        /// The resolved Level Zero entry points.
        pub(crate) struct Funcs {
            $( $(#[$meta])* pub $field: unsafe extern "C" fn($( $arg ),*) -> ZeResult, )*
        }

        impl Funcs {
            /// Resolve every entry point from `lib`, naming the first one it lacks.
            ///
            /// # Safety
            ///
            /// `lib` must be a Level Zero loader, whose exported symbols have the signatures declared here.
            unsafe fn resolve(lib: &LibHandle) -> Result<Self, L0Error> {
                Ok(Funcs {
                    $( $(#[$meta])* $field: {
                        let symbol = lib
                            .sym(
                                CStr::from_bytes_with_nul(concat!($symbol, "\0").as_bytes())
                                    .expect("a symbol literal has no interior NUL"),
                            )
                            .ok_or(L0Error::MissingSymbol($symbol))?;
                        // SAFETY: the caller vouches that the loader exports `$symbol` with this C
                        // signature; a function pointer and a data pointer have the same size here.
                        unsafe {
                            std::mem::transmute::<
                                *mut c_void,
                                unsafe extern "C" fn($( $arg ),*) -> ZeResult,
                            >(symbol)
                        }
                    }, )*
                })
            }
        }
    };
}

level_zero_functions! {
    init = "zeInit"(u32);
    driver_get = "zeDriverGet"(*mut u32, *mut Handle);
    device_get = "zeDeviceGet"(Handle, *mut u32, *mut Handle);
    device_get_properties = "zeDeviceGetProperties"(Handle, *mut DeviceProperties);
    device_get_queue_group_properties = "zeDeviceGetCommandQueueGroupProperties"(
        Handle, *mut u32, *mut QueueGroupProperties
    );
    context_create = "zeContextCreate"(Handle, *const ContextDesc, *mut Handle);
    context_destroy = "zeContextDestroy"(Handle);
    mem_alloc_host = "zeMemAllocHost"(
        Handle, *const HostMemAllocDesc, usize, usize, *mut *mut c_void
    );
    driver_get_extension_function_address = "zeDriverGetExtensionFunctionAddress"(
        Handle, *const c_char, *mut *mut c_void
    );
    mem_free = "zeMemFree"(Handle, *mut c_void);
    command_list_append_memory_copy = "zeCommandListAppendMemoryCopy"(
        Handle, *mut c_void, *const c_void, usize, Handle, u32, *mut Handle
    );
    command_queue_create = "zeCommandQueueCreate"(
        Handle, Handle, *const CommandQueueDesc, *mut Handle
    );
    command_list_create = "zeCommandListCreate"(
        Handle, Handle, *const CommandListDesc, *mut Handle
    );
    command_list_close = "zeCommandListClose"(Handle);
    command_queue_execute_command_lists = "zeCommandQueueExecuteCommandLists"(
        Handle, u32, *mut Handle, Handle
    );
    command_queue_synchronize = "zeCommandQueueSynchronize"(Handle, u64);
    command_list_destroy = "zeCommandListDestroy"(Handle);
    command_queue_destroy = "zeCommandQueueDestroy"(Handle);
}

/// The loaded loader library and the entry points resolved from it. The library handle is the last field, so
/// it is closed after the table is dropped.
pub(crate) struct Loader {
    pub funcs: Funcs,
    _lib: LibHandle,
}

impl Loader {
    /// `dlopen` the Level Zero loader by soname and resolve every entry point.
    pub(crate) fn load() -> Result<Self, L0Error> {
        Self::open(c"libze_loader.so.1")
    }

    /// `dlopen` the shared library `soname` and resolve every Level Zero entry point from it.
    pub(crate) fn open(soname: &CStr) -> Result<Self, L0Error> {
        // SAFETY: the loader's initializers are sound to run in any process; a non-loader library named
        // here fails symbol resolution below rather than being called.
        let lib = unsafe { LibHandle::open(soname) }.map_err(L0Error::LibraryNotFound)?;
        // SAFETY: the library was opened under a name the caller gives as a Level Zero loader.
        let funcs = unsafe { Funcs::resolve(&lib) }?;
        Ok(Loader { funcs, _lib: lib })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizes `ze_api.h` (level-zero 1.28.5, x86-64) gives these structures: the driver reads and writes
    /// them by this layout, so a field added, dropped or retyped here is a silent ABI break.
    #[test]
    fn descriptor_layouts_match_ze_api_h() {
        assert_eq!(size_of::<DeviceProperties>(), 368);
        assert_eq!(std::mem::offset_of!(DeviceProperties, vendor_id), 20);
        assert_eq!(size_of::<QueueGroupProperties>(), 40);
        assert_eq!(size_of::<ContextDesc>(), 24);
        assert_eq!(size_of::<CommandQueueDesc>(), 40);
        assert_eq!(size_of::<CommandListDesc>(), 24);
        assert_eq!(size_of::<HostMemAllocDesc>(), 24);
        assert_eq!(size_of::<GraphDesc2>(), 56);
        assert_eq!(std::mem::offset_of!(GraphDesc2, flags), 48);
        assert_eq!(size_of::<DeviceGraphProperties2>(), 56);
        assert_eq!(size_of::<GraphProperties2>(), 24);
        assert_eq!(size_of::<GraphArgumentProperties>(), 312);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties, dims), 276);
        assert_eq!(std::mem::offset_of!(GraphTable, destroy), 8);
        assert_eq!(
            std::mem::offset_of!(GraphTable, query_network_destroy),
            13 * 8
        );
        assert_eq!(
            std::mem::offset_of!(GraphTable, query_network_get_supported_layers),
            14 * 8
        );
        assert_eq!(std::mem::offset_of!(GraphTable, create_2), 16 * 8);
        assert_eq!(
            std::mem::offset_of!(GraphTable, query_network_create_2),
            17 * 8
        );
        assert_eq!(std::mem::offset_of!(GraphTable, graph_initialize), 22 * 8);
    }

    #[test]
    fn a_missing_loader_is_a_typed_library_error() {
        let Err(error) = Loader::open(c"libpoot-no-such-level-zero-loader.so.1") else {
            panic!("a library that does not exist loaded");
        };
        assert!(
            matches!(&error, L0Error::LibraryNotFound(message) if message.contains("libpoot-no-such-level-zero-loader")),
            "{error}"
        );
    }

    /// A library that loads but is not the loader fails on the first entry point it lacks, by name.
    #[test]
    fn a_library_without_the_entry_points_names_the_first_missing_symbol() {
        let Err(error) = Loader::open(c"libc.so.6") else {
            panic!("libc exports no Level Zero entry points");
        };
        assert!(matches!(error, L0Error::MissingSymbol("zeInit")), "{error}");
    }
}
