//! The one `dlopen` wrapper the runtimes that resolve a vendor library at run time share (ROCm's HSA
//! runtime, Level Zero's loader), so a crate builds without the vendor stack installed.

use std::ffi::{CStr, c_void};
use std::fmt;
use std::ptr::NonNull;

/// Whether the symbols of a loaded library become visible to libraries loaded after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadScope {
    /// The library's symbols resolve only through its own handle.
    Local,
    /// The library's symbols are also available to libraries loaded later (`RTLD_GLOBAL`), which the HSA
    /// runtime's own plugins need.
    Global,
}

impl LoadScope {
    fn flags(self) -> libc::c_int {
        match self {
            LoadScope::Local => libc::RTLD_NOW,
            LoadScope::Global => libc::RTLD_NOW | libc::RTLD_GLOBAL,
        }
    }
}

/// Why a library could not be loaded: the message the OS loader gave (for example
/// "libhsa-runtime64.so.1: cannot open shared object file: No such file or directory").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlError(String);

impl fmt::Display for DlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DlError {}

/// A `dlopen`ed shared library, closed on drop. Closing it invalidates every address resolved from it, so
/// whoever holds a resolved function pointer keeps the `Library` alive for as long as the pointer is.
pub struct Library(NonNull<c_void>);

// SAFETY: the handle is an opaque token for a process-wide loaded library. `dlsym` and `dlclose` are
// thread-safe, `dlclose` runs once (in `Drop`, with exclusive access), and no method mutates the handle.
unsafe impl Send for Library {}
// SAFETY: as for `Send`: the only shared-reference method, `symbol`, calls the thread-safe `dlsym`.
unsafe impl Sync for Library {}

impl Library {
    /// `dlopen` `soname` (a soname the loader walks `LD_LIBRARY_PATH` for, or an absolute path).
    ///
    /// # Safety
    ///
    /// Loading a library runs its initializers; `soname` must name a library whose initializers are sound
    /// to run in this process.
    pub unsafe fn open(soname: &CStr, scope: LoadScope) -> Result<Self, DlError> {
        // SAFETY: `soname` is NUL-terminated and outlives the call; the caller vouches for the library.
        let handle = unsafe { libc::dlopen(soname.as_ptr(), scope.flags()) };
        NonNull::new(handle).map(Library).ok_or_else(|| {
            // SAFETY: `dlerror` returns null or a NUL-terminated thread-local string valid until the next
            // `dl*` call on this thread; it is copied out before any.
            let message = unsafe { libc::dlerror() };
            DlError(if message.is_null() {
                format!("dlopen({soname:?}) failed")
            } else {
                // SAFETY: non-null and NUL-terminated, per `dlerror`.
                unsafe { CStr::from_ptr(message) }
                    .to_string_lossy()
                    .into_owned()
            })
        })
    }

    /// The first of `sonames` that loads, or the last loader message when none does.
    ///
    /// # Safety
    ///
    /// As for [`Library::open`], for every name in `sonames`.
    pub unsafe fn open_first(sonames: &[&CStr], scope: LoadScope) -> Result<Self, DlError> {
        let mut last = DlError("no library name given".to_string());
        for soname in sonames {
            // SAFETY: the caller vouches for every name in `sonames`.
            match unsafe { Self::open(soname, scope) } {
                Ok(library) => return Ok(library),
                Err(error) => last = error,
            }
        }
        Err(last)
    }

    /// The address of `name`, or `None` when the library does not export it. Using the address is the
    /// caller's concern, and it is valid only while this `Library` is.
    pub fn symbol(&self, name: &CStr) -> Option<*mut c_void> {
        // SAFETY: `self.0` is a live `dlopen` handle (closed only in `Drop`) and `name` is NUL-terminated
        // and outlives the call.
        let address = unsafe { libc::dlsym(self.0.as_ptr(), name.as_ptr()) };
        (!address.is_null()).then_some(address)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful `dlopen` and is closed exactly once, here.
        unsafe {
            libc::dlclose(self.0.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_reports_the_loader_message() {
        // SAFETY: the name resolves to no file, so nothing is loaded and no initializer runs.
        let error = unsafe {
            Library::open(c"libpoot-no-such-library.so.0", LoadScope::Local)
                .err()
                .expect("no such library")
        };
        assert!(
            error.to_string().contains("libpoot-no-such-library"),
            "{error}"
        );
    }

    #[test]
    fn open_first_skips_a_missing_name_and_resolves_a_symbol() {
        // SAFETY: libc is already loaded in every process, and its initializers are sound to run again.
        let library = unsafe {
            Library::open_first(
                &[c"libpoot-no-such-library.so.0", c"libc.so.6"],
                LoadScope::Local,
            )
        }
        .expect("libc.so.6 loads");
        assert!(library.symbol(c"getpid").is_some());
        assert!(library.symbol(c"poot_no_such_symbol").is_none());
    }

    #[test]
    fn open_first_reports_the_last_loader_message_when_none_loads() {
        // SAFETY: neither name resolves to a file.
        let error = unsafe {
            Library::open_first(
                &[c"libpoot-first-missing.so.0", c"libpoot-last-missing.so.0"],
                LoadScope::Global,
            )
            .err()
            .expect("neither loads")
        };
        assert!(
            error.to_string().contains("libpoot-last-missing"),
            "{error}"
        );
    }
}
