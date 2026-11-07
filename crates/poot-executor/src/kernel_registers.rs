//! What a device's compiler allocates for a kernel the engine compiled (POOT-1505), and the engine's side of the
//! planner's compile feedback. A device is mechanism: it compiles a kernel it was handed with its own compiler and
//! reports the registers and local memory ([`KernelRegisters`]). The engine owns every kernel, so it lowers a probe's
//! body (through its kernel cache, as it lowers a plan's kernels) and asks the device about the result: that pair is
//! the [`CompiledRegisters`] the planner is given.

use std::sync::Arc;

use poot_codegen::{CompiledKernel, KernelCache};
use poot_graph_plan::{
    CompiledRegisters, FeedbackError, RegisterFeedback, RegisterProbe, RegisterVerdict,
    VerdictCache,
};
use poot_target::KernelCompiler;

/// The directory, beside a kernel cache's entries, that keeps its register verdicts.
const VERDICTS: &str = "register-verdicts";

/// Why a device has no registers for a kernel.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum KernelRegistersError {
    /// The device's compiler refused the kernel.
    #[error("{0}")]
    Refused(String),
    /// The device or its driver failed outside compiling: a context, a query.
    #[error("{0}")]
    Device(String),
}

/// A device's compiler, asked what a compiled kernel allocates (PTX: the driver's JIT and `cuFuncGetAttribute`).
pub trait KernelRegisters: Send + Sync + std::fmt::Debug {
    /// The compiler that compiles this device's kernels: part of what names a plan made with its verdicts.
    fn compiler(&self) -> KernelCompiler;

    /// Compile `kernel` as the device will when a plan runs it, and report its registers. Callable from any thread.
    fn registers(&self, kernel: &CompiledKernel) -> Result<RegisterVerdict, KernelRegistersError>;
}

/// A probe lowered by the engine and compiled by the device.
#[derive(Debug)]
struct LoweredRegisters {
    device: Arc<dyn KernelRegisters>,
    kernels: KernelCache,
    max_artifact_bytes: std::num::NonZeroU64,
}

impl CompiledRegisters for LoweredRegisters {
    fn compiler(&self) -> KernelCompiler {
        self.device.compiler()
    }

    fn compile(&self, probe: &RegisterProbe) -> Result<RegisterVerdict, FeedbackError> {
        let name = || probe.body.name.clone();
        let refused = |reason: String| FeedbackError::Compile {
            name: name(),
            reason,
        };
        let artifact = self
            .kernels
            .load_or_compile(&probe.body, self.max_artifact_bytes)
            .map_err(|error| refused(error.to_string()))?;
        let kernel = poot_codegen::kernel_handle(&probe.body, &artifact)
            .map_err(|error| refused(error.to_string()))?;
        self.device.registers(&kernel).map_err(|error| match error {
            KernelRegistersError::Refused(reason) => refused(reason),
            KernelRegistersError::Device(reason) => FeedbackError::Device {
                name: name(),
                reason,
            },
        })
    }
}

/// The planner's feedback from `device`'s compiler: probes lowered through `kernels`, their verdicts kept on disk
/// beside the cache's entries (in the cache directory of the codegen epoch, so a codegen or toolchain change starts
/// them afresh) and in memory only where the cache is disabled.
pub(crate) fn register_feedback(
    device: Arc<dyn KernelRegisters>,
    kernels: KernelCache,
    max_artifact_bytes: std::num::NonZeroU64,
) -> RegisterFeedback {
    let verdicts = kernels.dir().map(|dir| dir.join(VERDICTS));
    let lowered = LoweredRegisters {
        device,
        kernels,
        max_artifact_bytes,
    };
    match verdicts {
        Some(dir) => RegisterFeedback::new(VerdictCache::new(lowered, dir)),
        None => RegisterFeedback::new(lowered),
    }
}

/// What `device` brings to a compile under `limits` (POOT-1505): its NPU lowering and, where its compiler can be
/// asked ([`crate::Device::kernel_registers`]), the register feedback an engine over it builds, probes lowered
/// through the device target's kernel cache with their verdicts beside it. For a tool that plans for a device before
/// an engine owns it; an engine answers [`crate::Executor::device_options`] with the same facts.
pub fn device_facts<D: crate::Device + ?Sized>(
    device: &D,
    limits: poot_graph_plan::CompileLimits,
) -> poot_graph_plan::DeviceFacts {
    let register_feedback = device.kernel_registers().and_then(|registers| {
        let kernels = poot_codegen::Target::try_from(device.target().id())
            .ok()
            .map(KernelCache::open)?;
        Some(register_feedback(
            registers,
            kernels,
            limits.max_artifact_bytes,
        ))
    });
    poot_graph_plan::DeviceFacts {
        npu: device.region_loader().map(crate::RegionLoader::lowering),
        register_feedback,
    }
}
