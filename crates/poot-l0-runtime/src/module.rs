use std::ffi::CString;
use std::ptr::null_mut;

use crate::context::L0Context;
use crate::error::{L0Error, check};
use crate::ffi::{Handle, KernelDesc, KernelProperties, ModuleDesc};

/// A SPIR-V module the driver compiled for the device (`zeModuleCreate`), destroyed on drop.
pub struct L0Module<'c> {
    pub(crate) context: &'c L0Context,
    module: Handle,
}

impl<'c> L0Module<'c> {
    pub(crate) fn create(context: &'c L0Context, spirv: &[u8]) -> Result<Self, L0Error> {
        if !spirv.len().is_multiple_of(4) {
            return Err(L0Error::MisalignedSpirv { len: spirv.len() });
        }
        let desc = ModuleDesc::spirv(spirv);
        let mut module = null_mut();
        // SAFETY: `desc` and the SPIR-V bytes it points into outlive the call; the context and device are
        // live; the build log out-pointer is optional and null.
        let result = unsafe {
            (context.funcs().module_create)(
                context.handle(),
                context.device(),
                &desc,
                &mut module,
                null_mut(),
            )
        };
        check("zeModuleCreate", result)?;
        Ok(L0Module { context, module })
    }

    /// The kernel `name` of this module, with the work-group size `group_size` every launch uses.
    pub fn kernel(&self, name: &str, group_size: [u32; 3]) -> Result<L0Kernel<'_>, L0Error> {
        let name = CString::new(name)?;
        let desc = KernelDesc::named(&name);
        let mut kernel = null_mut();
        // SAFETY: `module` is live (destroyed only on drop, after every kernel borrowing it) and `desc` and
        // the name it points to outlive the call.
        let result =
            unsafe { (self.context.funcs().kernel_create)(self.module, &desc, &mut kernel) };
        check("zeKernelCreate", result)?;
        let mut kernel = L0Kernel {
            module: self,
            kernel,
            num_args: 0,
        };
        let mut properties = KernelProperties::new();
        // SAFETY: `kernel` is live and `properties` is a correctly sized, typed structure.
        let result =
            unsafe { (self.context.funcs().kernel_get_properties)(kernel.kernel, &mut properties) };
        check("zeKernelGetProperties", result)?;
        kernel.num_args = properties.num_kernel_args;
        let [x, y, z] = group_size;
        // SAFETY: `kernel` is a live kernel handle, owned by the value that destroys it on drop.
        let result =
            unsafe { (self.context.funcs().kernel_set_group_size)(kernel.kernel, x, y, z) };
        check("zeKernelSetGroupSize", result)?;
        Ok(kernel)
    }
}

impl Drop for L0Module<'_> {
    fn drop(&mut self) {
        // SAFETY: `module` came from a successful `zeModuleCreate` and is destroyed exactly once, here, after
        // every kernel (which borrows it) is gone.
        let result = unsafe { (self.context.funcs().module_destroy)(self.module) };
        if let Err(error) = check("zeModuleDestroy", result) {
            tracing::warn!(%error, "leaking a Level Zero module");
        }
    }
}

/// One kernel of a module (`zeKernelCreate`), destroyed on drop.
pub struct L0Kernel<'m> {
    module: &'m L0Module<'m>,
    pub(crate) kernel: Handle,
    /// The kernel's parameter count, as the driver reports it.
    pub(crate) num_args: u32,
}

impl L0Kernel<'_> {
    pub(crate) fn context(&self) -> &L0Context {
        self.module.context
    }
}

impl Drop for L0Kernel<'_> {
    fn drop(&mut self) {
        // SAFETY: `kernel` came from a successful `zeKernelCreate` and is destroyed exactly once, here, while
        // the module it borrows is still alive.
        let result = unsafe { (self.module.context.funcs().kernel_destroy)(self.kernel) };
        if let Err(error) = check("zeKernelDestroy", result) {
            tracing::warn!(%error, "leaking a Level Zero kernel");
        }
    }
}
