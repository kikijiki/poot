//! The XRT FFI boundary: the one module of this crate with `unsafe`. It wraps the C++ shim
//! (`xrt_shim.cpp`) in three RAII handles, [`XrtDevice`], [`XrtBo`] and [`XrtKernel`], each freed exactly
//! once on drop, and turns every shim failure into the typed [`XrtError`] carrying XRT's own message.
//!
//! None of the handles is `Send` or `Sync`: a device and everything it allocated live on one thread, which is
//! what makes the unsynchronized view [`XrtBo`] gives of its mapping sound.
//!
//! XRT's objects are reference-counted: a hardware context keeps the device it was opened on alive, so no
//! drop order between these wrappers can leave a dangling reference.

use std::ffi::{CString, c_char, c_void};
use std::ptr::NonNull;

/// The error buffer length the shim writes into, NUL included (`kErrLen` in `xrt_shim.cpp`).
const ERR_LEN: usize = 512;

unsafe extern "C" {
    fn poot_xrt_device_open(index: u32, err: *mut c_char) -> *mut c_void;
    fn poot_xrt_device_free(handle: *mut c_void);
    fn poot_xrt_bo_alloc(device: *mut c_void, bytes: usize, err: *mut c_char) -> *mut c_void;
    fn poot_xrt_bo_free(handle: *mut c_void);
    fn poot_xrt_bo_map(handle: *mut c_void) -> *mut c_void;
    fn poot_xrt_bo_sync(
        handle: *mut c_void,
        to_device: i32,
        size: usize,
        offset: usize,
        err: *mut c_char,
    ) -> i32;
    fn poot_xrt_kernel_load(
        device: *mut c_void,
        xclbin: *const u8,
        xclbin_bytes: usize,
        name: *const c_char,
        instructions: *const u8,
        instruction_bytes: usize,
        arg_count: *mut u32,
        err: *mut c_char,
    ) -> *mut c_void;
    fn poot_xrt_kernel_free(handle: *mut c_void);
    fn poot_xrt_kernel_run(
        handle: *mut c_void,
        bos: *const *mut c_void,
        n: usize,
        timeout_us: u64,
        err: *mut c_char,
    ) -> i32;
}

/// An XRT call failed. The text is XRT's own (the exception's `what()`), prefixed by the shim with the
/// operation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct XrtError {
    message: String,
}

impl XrtError {
    fn from_buffer(buffer: &[u8; ERR_LEN]) -> Self {
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(ERR_LEN);
        Self {
            message: String::from_utf8_lossy(&buffer[..end]).into_owned(),
        }
    }

    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// An open XDNA device (`xrt::device`).
pub(crate) struct XrtDevice(NonNull<c_void>);

impl XrtDevice {
    /// Open the NPU at `index` (0 is `/dev/accel/accel0`).
    pub(crate) fn open(index: u32) -> Result<Self, XrtError> {
        let mut err = [0u8; ERR_LEN];
        // SAFETY: `err` is a writable `ERR_LEN`-byte buffer, as the shim requires.
        let handle = unsafe { poot_xrt_device_open(index, err.as_mut_ptr().cast()) };
        NonNull::new(handle)
            .map(Self)
            .ok_or_else(|| XrtError::from_buffer(&err))
    }

    /// A zero-filled host-mapped buffer object of `bytes` bytes.
    pub(crate) fn alloc(&self, bytes: usize) -> Result<XrtBo, XrtError> {
        let mut err = [0u8; ERR_LEN];
        // SAFETY: `self.0` is a live device handle; `err` is an `ERR_LEN`-byte buffer.
        let handle = unsafe { poot_xrt_bo_alloc(self.0.as_ptr(), bytes, err.as_mut_ptr().cast()) };
        let handle = NonNull::new(handle).ok_or_else(|| XrtError::from_buffer(&err))?;
        // SAFETY: `handle` is a live buffer object; its mapping stays valid until it is freed.
        let map = unsafe { poot_xrt_bo_map(handle.as_ptr()) }.cast::<u8>();
        match NonNull::new(map) {
            Some(map) => Ok(XrtBo {
                handle,
                map,
                len: bytes,
            }),
            None => {
                // SAFETY: `handle` was just allocated and is not used again.
                unsafe { poot_xrt_bo_free(handle.as_ptr()) };
                Err(XrtError::new(
                    "bo alloc: the buffer object has no host mapping",
                ))
            }
        }
    }

    /// Register `xclbin`, open a hardware context and the kernel `name`, and upload `instructions`.
    pub(crate) fn load_kernel(
        &self,
        xclbin: &[u8],
        name: &str,
        instructions: &[u8],
    ) -> Result<XrtKernel, XrtError> {
        let name_c = CString::new(name)
            .map_err(|_| XrtError::new("kernel load: the kernel name holds a NUL byte"))?;
        let mut err = [0u8; ERR_LEN];
        let mut arg_count = 0u32;
        // SAFETY: the pointers and lengths describe live slices; `name_c` is NUL-terminated; `arg_count`
        // and `err` are writable for what the shim writes.
        let handle = unsafe {
            poot_xrt_kernel_load(
                self.0.as_ptr(),
                xclbin.as_ptr(),
                xclbin.len(),
                name_c.as_ptr(),
                instructions.as_ptr(),
                instructions.len(),
                &mut arg_count,
                err.as_mut_ptr().cast(),
            )
        };
        NonNull::new(handle)
            .map(|handle| XrtKernel { handle, arg_count })
            .ok_or_else(|| XrtError::from_buffer(&err))
    }
}

impl Drop for XrtDevice {
    fn drop(&mut self) {
        // SAFETY: the handle came from `poot_xrt_device_open` and is freed exactly once, here.
        unsafe { poot_xrt_device_free(self.0.as_ptr()) };
    }
}

/// One host-mapped buffer object. The mapping is the buffer's host view: the device reads and writes the same
/// pages, and [`XrtBo::sync_to_device`] / [`XrtBo::sync_from_device`] make each side's writes visible to the
/// other.
pub(crate) struct XrtBo {
    handle: NonNull<c_void>,
    map: NonNull<u8>,
    len: usize,
}

impl XrtBo {
    /// The buffer's logical length in bytes (what `allocate` was asked for).
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Hand `fill` the mapping's `range` to write. The caller syncs the range to the device afterwards.
    /// `range` must lie inside the buffer.
    pub(crate) fn write(&self, range: std::ops::Range<usize>, fill: impl FnOnce(&mut [u8])) {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "range outside the buffer"
        );
        // SAFETY: the range is inside the mapping, which holds at least `len` bytes and lives as long as
        // `self`. The handles are not `Sync`, so this is the only thread that can reach the mapping, and no
        // slice of it is alive across this call: every accessor here creates one for its own duration.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.map.as_ptr().add(range.start), range.len())
        };
        fill(bytes);
    }

    /// Copy the mapping's `offset..offset + out.len()` into `out`. The range must lie inside the buffer.
    pub(crate) fn read(&self, offset: usize, out: &mut [u8]) {
        assert!(
            offset <= self.len && out.len() <= self.len - offset,
            "range outside the buffer"
        );
        // SAFETY: as for `write`; `out` is a distinct host allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.map.as_ptr().add(offset),
                out.as_mut_ptr(),
                out.len(),
            );
        }
    }

    /// Copy the whole of `src` over this buffer's mapping. Both are `len` bytes long.
    pub(crate) fn copy_from(&self, src: &XrtBo) {
        assert_eq!(
            self.len, src.len,
            "copy between buffers of different lengths"
        );
        // SAFETY: both mappings hold `len` bytes; `copy` handles `src` and `self` being the same buffer.
        unsafe { std::ptr::copy(src.map.as_ptr(), self.map.as_ptr(), self.len) };
    }

    /// Make the host's writes to `offset..offset + size` visible to the device.
    pub(crate) fn sync_to_device(&self, offset: usize, size: usize) -> Result<(), XrtError> {
        self.sync(true, offset, size)
    }

    /// Make the device's writes to `offset..offset + size` visible to the host.
    pub(crate) fn sync_from_device(&self, offset: usize, size: usize) -> Result<(), XrtError> {
        self.sync(false, offset, size)
    }

    fn sync(&self, to_device: bool, offset: usize, size: usize) -> Result<(), XrtError> {
        let mut err = [0u8; ERR_LEN];
        // SAFETY: `handle` is live; XRT bounds-checks the range itself and reports a bad one as an error.
        let status = unsafe {
            poot_xrt_bo_sync(
                self.handle.as_ptr(),
                i32::from(to_device),
                size,
                offset,
                err.as_mut_ptr().cast(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(XrtError::from_buffer(&err))
        }
    }
}

impl Drop for XrtBo {
    fn drop(&mut self) {
        // SAFETY: the handle came from `poot_xrt_bo_alloc` and is freed exactly once, here.
        unsafe { poot_xrt_bo_free(self.handle.as_ptr()) };
    }
}

/// A loaded kernel: its registered xclbin, hardware context, kernel object and instruction buffer.
pub(crate) struct XrtKernel {
    handle: NonNull<c_void>,
    arg_count: u32,
}

impl XrtKernel {
    /// The number of arguments the xclbin declares for this kernel (opcode, instruction buffer, its length
    /// and the data buffers).
    pub(crate) fn arg_count(&self) -> u32 {
        self.arg_count
    }

    /// Run the kernel once over `buffers` (argument slots 3.., in order) and wait up to `timeout_us`
    /// microseconds for a successful completion. A timeout and a non-success completion are both errors.
    pub(crate) fn run(&self, buffers: &[&XrtBo], timeout_us: u64) -> Result<(), XrtError> {
        let handles: Vec<*mut c_void> = buffers.iter().map(|bo| bo.handle.as_ptr()).collect();
        let mut err = [0u8; ERR_LEN];
        // SAFETY: `handle` is a live kernel; `handles` holds live buffer handles that outlive this call.
        let status = unsafe {
            poot_xrt_kernel_run(
                self.handle.as_ptr(),
                handles.as_ptr(),
                handles.len(),
                timeout_us,
                err.as_mut_ptr().cast(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(XrtError::from_buffer(&err))
        }
    }
}

impl Drop for XrtKernel {
    fn drop(&mut self) {
        // SAFETY: the handle came from `poot_xrt_kernel_load` and is freed exactly once, here.
        unsafe { poot_xrt_kernel_free(self.handle.as_ptr()) };
    }
}
