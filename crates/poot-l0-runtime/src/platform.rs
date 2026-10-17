//! A `dlopen`ed shared library, closed on drop.

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;

/// A loaded shared library. Closing it on drop invalidates every function pointer resolved from it, so
/// [`Loader`](crate::ffi::Loader) keeps it alive for as long as the pointers are.
pub(crate) struct LibHandle(NonNull<c_void>);

// SAFETY: the handle is an opaque token for a process-wide loaded library. `dlsym` and `dlclose` are
// thread-safe, `dlclose` runs once (in `Drop`, with exclusive access), and no method mutates the handle.
unsafe impl Send for LibHandle {}
// SAFETY: as for `Send`: the only shared-reference method, `sym`, calls the thread-safe `dlsym`.
unsafe impl Sync for LibHandle {}

impl LibHandle {
    /// `dlopen` `soname` (a soname the loader walks `LD_LIBRARY_PATH` for, or an absolute path). The error is
    /// the OS message.
    ///
    /// # Safety
    ///
    /// Loading a library runs its initializers; `soname` must name a library whose initializers are sound to
    /// run in this process.
    pub(crate) unsafe fn open(soname: &CStr) -> Result<Self, String> {
        // SAFETY: `soname` is NUL-terminated and outlives the call; the caller vouches for the library.
        let handle = unsafe { libc::dlopen(soname.as_ptr(), libc::RTLD_NOW) };
        NonNull::new(handle).map(LibHandle).ok_or_else(|| {
            // SAFETY: `dlerror` returns null or a NUL-terminated thread-local string valid until the next
            // `dl*` call on this thread; it is copied out before any.
            let message = unsafe { libc::dlerror() };
            if message.is_null() {
                format!("dlopen({soname:?}) failed")
            } else {
                // SAFETY: non-null and NUL-terminated, per `dlerror`.
                unsafe { CStr::from_ptr(message) }
                    .to_string_lossy()
                    .into_owned()
            }
        })
    }

    /// The address of `symbol`, or `None` when the library does not export it.
    pub(crate) fn sym(&self, symbol: &CStr) -> Option<*mut c_void> {
        // SAFETY: `self.0` is a live `dlopen` handle (closed only in `Drop`) and `symbol` is NUL-terminated
        // and outlives the call. Using the returned address is the caller's concern.
        let address = unsafe { libc::dlsym(self.0.as_ptr(), symbol.as_ptr()) };
        (!address.is_null()).then_some(address)
    }
}

impl Drop for LibHandle {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful `dlopen` and is closed exactly once, here.
        unsafe {
            libc::dlclose(self.0.as_ptr());
        }
    }
}
