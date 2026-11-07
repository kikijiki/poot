//! [`PtxCompilerProbe`]: what the CUDA driver's PTX JIT makes of a kernel, asked of the same compiler that will run it
//! (POOT-1505, POOT-1509). A PTX module is loaded exactly as [`crate::PtxContext`] loads one to dispatch it
//! (`cuModuleLoadData` on the textual image, no JIT options), its function's registers and local memory are read, and
//! the module is unloaded again.

use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::{DriverError, result, sys};

use crate::context::PtxContext;
use crate::error::PtxError;
use crate::owner::PtxOwner;

/// What the JIT allocated for one function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionRegisters {
    /// `CU_FUNC_ATTRIBUTE_NUM_REGS`: registers per thread.
    pub registers: u32,
    /// `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`: per-thread local memory, where a spilled register goes.
    pub local_bytes: u32,
}

/// Why a module has no [`FunctionRegisters`].
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    /// The driver's JIT refused the module or cannot compile it for this device.
    #[error("the CUDA JIT refused the module: {0}")]
    Refused(String),
    /// The device or driver failed outside compiling: a context, an attribute query, a lost device.
    #[error(transparent)]
    Device(#[from] PtxError),
}

/// A handle on one device's compiler, usable from any thread: each call binds the device's context to the calling
/// thread and restores what was current before, so a planner thread with no context of its own can ask. It shares the
/// device's context with the [`PtxContext`] it came from and keeps it alive.
pub struct PtxCompilerProbe {
    owner: Arc<PtxOwner>,
    release: u32,
    compute_capability: (u32, u32),
}

impl PtxContext {
    /// The handle on this device's PTX compiler, with the driver release and compute capability read once here.
    pub fn compiler_probe(&self) -> Result<PtxCompilerProbe, PtxError> {
        let owner = Arc::clone(&self.owner);
        let (release, compute_capability) = owner.with_current(|_| {
            let mut release = 0;
            // SAFETY: `release` is a live `int` the call writes.
            unsafe { sys::cuDriverGetVersion(&mut release) }.result()?;
            let attribute = |a| {
                // SAFETY: `owner.device` is the device the context was retained on.
                unsafe { result::device::get_attribute(owner.device, a) }
            };
            let major =
                attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            let minor =
                attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            Ok((release as u32, (major as u32, minor as u32)))
        })?;
        Ok(PtxCompilerProbe {
            owner,
            release,
            compute_capability,
        })
    }
}

impl std::fmt::Debug for PtxCompilerProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PtxCompilerProbe")
            .field("release", &self.release)
            .field("compute_capability", &self.compute_capability)
            .finish_non_exhaustive()
    }
}

/// Whether a load error is the JIT saying it cannot compile this image (as opposed to the device failing).
fn is_refusal(error: DriverError) -> bool {
    use sys::CUresult as R;
    matches!(
        error.0,
        R::CUDA_ERROR_INVALID_PTX
            | R::CUDA_ERROR_INVALID_IMAGE
            | R::CUDA_ERROR_INVALID_SOURCE
            | R::CUDA_ERROR_NO_BINARY_FOR_GPU
            | R::CUDA_ERROR_UNSUPPORTED_PTX_VERSION
            | R::CUDA_ERROR_NOT_FOUND
    )
}

impl PtxCompilerProbe {
    /// `cuDriverGetVersion`'s value, `1000 * major + 10 * minor`.
    pub fn driver_release(&self) -> u32 {
        self.release
    }

    /// The device's compute capability, `(major, minor)`: what the JIT compiles for.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Compile `ptx` with the driver and report `entry`'s registers and local memory. The module is unloaded
    /// before this returns, on every path.
    pub fn function_registers(
        &self,
        ptx: &str,
        entry: &str,
    ) -> Result<FunctionRegisters, ProbeError> {
        let (Ok(ptx), Ok(entry)) = (CString::new(ptx), CString::new(entry)) else {
            return Err(ProbeError::Refused(
                "the kernel text or entry name holds a NUL byte".to_string(),
            ));
        };
        crate::owner::OwnerCounters::bump(&self.owner.counters.register_probes);
        let outcome = self.owner.with_current(|_| {
            // SAFETY: `ptx` is a NUL-terminated PTX image that outlives the call.
            let module = match unsafe { result::module::load_data(ptx.as_ptr().cast()) } {
                Ok(module) => module,
                Err(error) if is_refusal(error) => return Ok(Err(error)),
                Err(error) => return Err(error),
            };
            let read = || {
                // SAFETY: `module` was just loaded and is unloaded below.
                let function = unsafe { result::module::get_function(module, entry.clone()) }?;
                let attribute = |a| {
                    // SAFETY: `function` belongs to the loaded `module`.
                    unsafe { result::function::get_function_attribute(function, a) }
                };
                Ok(FunctionRegisters {
                    registers: attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS)?
                        as u32,
                    local_bytes: attribute(
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    )? as u32,
                })
            };
            let registers = read();
            // SAFETY: `module` was loaded above and is unloaded once.
            let _ = unsafe { result::module::unload(module) };
            match registers {
                Ok(registers) => Ok(Ok(registers)),
                Err(error) if is_refusal(error) => Ok(Err(error)),
                Err(error) => Err(error),
            }
        })?;
        outcome.map_err(|error| ProbeError::Refused(error.to_string()))
    }
}
