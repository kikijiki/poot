//! Minimal POSIX dlopen / dlsym wrapper. HSA symbols are resolved at runtime so the crate builds
//! without ROCm installed (spec 063 FR-004 / FR-001); the build never links `libhsa-runtime64.so.1`.
//!
//! `libdl` is part of glibc, so `#[link(name = "dl")]` is enough to find `dlerror` / `dlopen` /
//! `dlsym`. Same pattern as IREE's `libhsa` table (refs/iree-fresh/runtime/src/iree/hal/drivers/amdgpu/
//! util/libhsa.c) and LLVM OpenMP's `dynamic_hsa` (refs/llvm-project-rocm/offload/plugins-nextgen/
//! amdgpu/dynamic_hsa/hsa.cpp:25-138).

use std::ffi::{CStr, c_char, c_void};
use std::fmt;

unsafe extern "C" {
    #[link_name = "dlopen"]
    fn c_dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
    #[link_name = "dlsym"]
    fn c_dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    #[link_name = "dlclose"]
    fn c_dlclose(handle: *mut c_void) -> i32;
    #[link_name = "dlerror"]
    fn c_dlerror() -> *const c_char;
}

const RTLD_NOW: i32 = 2;
const RTLD_GLOBAL: i32 = 0x100;

/// A loaded shared library handle. Closing it on drop invalidates all derived function pointers, so
/// the runtime owner behind [`RocmContext`](crate::RocmContext) keeps it alive until every resource
/// using those symbols is gone.
pub struct LibHandle {
    handle: *mut c_void,
}

// SAFETY: the handle is an opaque token for a process-wide loaded library. `dlsym` and `dlclose` are
// thread-safe, `dlclose` runs once (in `Drop`, with exclusive access), and no method mutates it.
unsafe impl Send for LibHandle {}
// SAFETY: as for `Send`: the only shared-reference method, `sym`, calls the thread-safe `dlsym`.
unsafe impl Sync for LibHandle {}

impl LibHandle {
    /// Try to dlopen `name`. `name` may be an absolute path or a soname (the loader walks
    /// `LD_LIBRARY_PATH` / `DT_RUNPATH` etc. as usual).
    ///
    /// # Safety
    ///
    /// Loading a library runs its initializers; `name` must be a library whose initializers are
    /// sound to run in this process.
    pub unsafe fn open(name: &CStr) -> Result<Self, DlError> {
        // SAFETY: `name` is NUL-terminated and outlives the call; the caller vouches for the library.
        unsafe {
            let h = c_dlopen(name.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
            if h.is_null() {
                Err(DlError(format!(
                    "dlopen({:?}) failed: {}",
                    name,
                    last_error()
                )))
            } else {
                Ok(LibHandle { handle: h })
            }
        }
    }

    /// Resolve `symbol` to a raw function pointer. Returns `None` if the symbol is missing.
    pub fn sym(&self, symbol: &CStr) -> Option<*mut c_void> {
        // SAFETY: `handle` is a live `dlopen` handle (closed only in `Drop`), and `symbol` is
        // NUL-terminated and outlives the call. Using the returned address is the caller's concern.
        unsafe {
            let p = c_dlsym(self.handle, symbol.as_ptr());
            if p.is_null() { None } else { Some(p) }
        }
    }
}

impl Drop for LibHandle {
    fn drop(&mut self) {
        // SAFETY: `handle` came from a successful `dlopen` and is closed exactly once, here. Every owner
        // of a function pointer resolved from it keeps this handle alive (see `Hsa`).
        unsafe {
            let _ = c_dlclose(self.handle);
        }
    }
}

/// Error from `dlopen`. Carries the OS-supplied reason (e.g. "libhsa-runtime64.so.1: cannot
/// open shared object file: No such file or directory").
#[derive(Debug)]
pub struct DlError(pub String);

impl fmt::Display for DlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DlError {}

fn last_error() -> String {
    // SAFETY: `dlerror` returns null or a NUL-terminated thread-local string valid until the next `dl*`
    // call on this thread, and it is copied out before any.
    unsafe {
        let p = c_dlerror();
        if p.is_null() {
            "<no dlerror>".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}
