//! The GPU half: OpenCL SPIR-V modules, device buffers and kernel launches (`Target::SpirvOpenCl`).
//!
//! No production crate consumes it yet (the Intel GPU `Device` is a later card), so it sits behind the
//! sanctioned `test-support` feature (POOT-ADR-0113), enabled by this crate's own dev-dependency edge. The first
//! consumer removes the gate. Every launch and copy here waits for the queue.

use std::ffi::c_void;
use std::ptr::null_mut;

use crate::context::{ALLOC_ALIGNMENT, L0Context};
use crate::error::{L0Error, check};
use crate::ffi::{CommandListDesc, DeviceMemAllocDesc, GroupCount, Handle};
use crate::module::{L0Kernel, L0Module};

/// One kernel argument, bound by index in the kernel's parameter order.
pub enum KernelArg<'a> {
    /// A device buffer's address (a `ptr addrspace(1)` parameter).
    Buffer(&'a L0Buffer<'a>),
    /// A 64-bit integer (an `i64` slice length).
    U64(u64),
}

/// A device-memory allocation (`zeMemAllocDevice`), freed on drop.
pub struct L0Buffer<'c> {
    pub(crate) context: &'c L0Context,
    pub(crate) ptr: *mut c_void,
    len_bytes: usize,
}

impl<'c> L0Buffer<'c> {
    pub(crate) fn new(context: &'c L0Context, ptr: *mut c_void, len_bytes: usize) -> Self {
        L0Buffer {
            context,
            ptr,
            len_bytes,
        }
    }

    /// Copy `host` to the device. The slice must be exactly as long as the buffer.
    pub fn upload_f32(&self, host: &[f32]) -> Result<(), L0Error> {
        let host_bytes = size_of_val(host);
        if host_bytes != self.len_bytes {
            return Err(L0Error::TransferLength {
                host: host_bytes,
                device: self.len_bytes,
            });
        }
        self.context
            .copy(self.ptr, host.as_ptr().cast(), self.len_bytes)
    }

    /// Copy the buffer to `host`. The slice must be exactly as long as the buffer.
    pub fn download_f32(&self, host: &mut [f32]) -> Result<(), L0Error> {
        let host_bytes = size_of_val(host);
        if host_bytes != self.len_bytes {
            return Err(L0Error::TransferLength {
                host: host_bytes,
                device: self.len_bytes,
            });
        }
        self.context
            .copy(host.as_mut_ptr().cast(), self.ptr, self.len_bytes)
    }
}

impl Drop for L0Buffer<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `zeMemAllocDevice` on this context and is freed exactly once, here; the
        // borrow of the context keeps it alive until now. A failed free leaves nothing to recover.
        let result = unsafe { (self.context.funcs().mem_free)(self.context.handle(), self.ptr) };
        if let Err(error) = check("zeMemFree", result) {
            tracing::warn!(%error, "leaking a Level Zero device allocation");
        }
    }
}

impl L0Context {
    /// Allocate `len` `f32`s of device memory, uninitialized.
    pub fn alloc_f32(&self, len: usize) -> Result<L0Buffer<'_>, L0Error> {
        let bytes = len * size_of::<f32>();
        let desc = DeviceMemAllocDesc::new();
        let mut ptr: *mut c_void = null_mut();
        // SAFETY: the context and device are live and `desc` outlives the call.
        let result = unsafe {
            (self.funcs().mem_alloc_device)(
                self.handle(),
                &desc,
                bytes,
                ALLOC_ALIGNMENT,
                self.device(),
                &mut ptr,
            )
        };
        check("zeMemAllocDevice", result)?;
        Ok(L0Buffer::new(self, ptr, bytes))
    }

    /// Compile a SPIR-V module (`Target::SpirvOpenCl` output, little-endian bytes) for the device.
    pub fn load_module(&self, spirv: &[u8]) -> Result<L0Module<'_>, L0Error> {
        L0Module::create(self, spirv)
    }

    /// Launch `kernel` over `groups` work-groups with `args` bound by index, and wait for it to finish.
    pub fn dispatch(
        &self,
        kernel: &mut L0Kernel<'_>,
        groups: [u32; 3],
        args: &[KernelArg<'_>],
    ) -> Result<(), L0Error> {
        if !self.same(kernel.context()) {
            return Err(L0Error::ForeignResource { what: "kernel" });
        }
        if usize::try_from(kernel.num_args) != Ok(args.len()) {
            return Err(L0Error::ArgumentCount {
                expected: kernel.num_args,
                given: args.len(),
            });
        }
        for (index, arg) in args.iter().enumerate() {
            let index = u32::try_from(index).expect("a kernel has fewer than 2^32 parameters");
            match arg {
                KernelArg::Buffer(buffer) => {
                    if !self.same(buffer.context) {
                        return Err(L0Error::ForeignResource { what: "buffer" });
                    }
                    self.set_argument(kernel, index, &buffer.ptr)?;
                }
                KernelArg::U64(value) => self.set_argument(kernel, index, value)?,
            }
        }
        let [x, y, z] = groups;
        let count = GroupCount { x, y, z };
        self.submit(|list| {
            // SAFETY: `list` is open, `kernel` is live with its arguments bound, and `count` outlives the
            // call; no signal or wait events are used.
            let result = unsafe {
                (self.funcs().command_list_append_launch_kernel)(
                    list,
                    kernel.kernel,
                    &count,
                    null_mut(),
                    0,
                    null_mut(),
                )
            };
            check("zeCommandListAppendLaunchKernel", result)
        })
    }

    fn set_argument<T>(&self, kernel: &L0Kernel<'_>, index: u32, value: &T) -> Result<(), L0Error> {
        // SAFETY: `kernel` is live and the driver copies `size_of::<T>()` bytes out of `value` during the call.
        let result = unsafe {
            (self.funcs().kernel_set_argument_value)(
                kernel.kernel,
                index,
                size_of::<T>(),
                std::ptr::from_ref(value).cast(),
            )
        };
        check("zeKernelSetArgumentValue", result)
    }

    /// Copy `len` bytes from `src` to `dst`, one of which is device memory of this context, and wait.
    pub(crate) fn copy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        len: usize,
    ) -> Result<(), L0Error> {
        self.submit(|list| {
            // SAFETY: `list` is open; the caller guarantees `dst` and `src` are valid for `len` bytes (a
            // device allocation of this context and a host slice of that length) until this call returns,
            // which it does only after the queue has drained.
            let result = unsafe {
                (self.funcs().command_list_append_memory_copy)(
                    list,
                    dst,
                    src,
                    len,
                    null_mut(),
                    0,
                    null_mut(),
                )
            };
            check("zeCommandListAppendMemoryCopy", result)
        })
    }

    /// Record one command list with `record`, close it, execute it on the queue and wait for the queue.
    fn submit(&self, record: impl FnOnce(Handle) -> Result<(), L0Error>) -> Result<(), L0Error> {
        let desc = CommandListDesc::new(self.ordinal());
        let mut list = null_mut();
        // SAFETY: the context and device are live and `desc` outlives the call.
        check("zeCommandListCreate", unsafe {
            (self.funcs().command_list_create)(self.handle(), self.device(), &desc, &mut list)
        })?;
        let list = CommandList {
            context: self,
            list,
        };
        record(list.list)?;
        // SAFETY: `list` is a live, open command list.
        check("zeCommandListClose", unsafe {
            (self.funcs().command_list_close)(list.list)
        })?;
        let mut lists = [list.list];
        // SAFETY: the queue is live, `lists` holds one closed list, and no fence is used.
        check("zeCommandQueueExecuteCommandLists", unsafe {
            (self.funcs().command_queue_execute_command_lists)(
                self.queue(),
                1,
                lists.as_mut_ptr(),
                null_mut(),
            )
        })?;
        // SAFETY: the queue is live; `u64::MAX` waits without a timeout.
        check("zeCommandQueueSynchronize", unsafe {
            (self.funcs().command_queue_synchronize)(self.queue(), u64::MAX)
        })
    }
}

/// A command list destroyed on drop, so an error while recording or executing it does not leak it.
struct CommandList<'c> {
    context: &'c L0Context,
    list: Handle,
}

impl Drop for CommandList<'_> {
    fn drop(&mut self) {
        // SAFETY: `list` came from a successful `zeCommandListCreate` and is destroyed exactly once, here.
        // Dropping after `zeCommandQueueSynchronize` returned means the queue no longer uses it.
        let result = unsafe { (self.context.funcs().command_list_destroy)(self.list) };
        if let Err(error) = check("zeCommandListDestroy", result) {
            tracing::warn!(%error, "leaking a Level Zero command list");
        }
    }
}
