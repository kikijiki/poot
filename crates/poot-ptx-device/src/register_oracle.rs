//! [`PtxKernelRegisters`]: the CUDA driver's answer to what a kernel allocates (POOT-1505, POOT-1509). The engine hands
//! it a kernel it compiled; the driver's JIT compiles the PTX the way it will when a plan runs it
//! (`cuModuleLoadData`, no options), and the registers and local memory are what `cuFuncGetAttribute` reports. The
//! device lowers nothing: the engine owns every kernel (POOT-1235).

use poot_codegen::{CompiledKernel, KernelCode};
use poot_executor::{KernelRegisters, KernelRegistersError};
use poot_graph_plan::RegisterVerdict;
use poot_ptx_runtime::{ProbeError, PtxCompilerProbe, PtxContext, PtxError};
use poot_target::KernelCompiler;

/// The CUDA driver's compiler, asked how many registers a kernel allocates. Usable from any thread: every call binds
/// the device's context to the calling thread.
#[derive(Debug)]
pub struct PtxKernelRegisters {
    compiler: PtxCompilerProbe,
    identity: KernelCompiler,
}

impl PtxKernelRegisters {
    /// The compiler of the device `ctx` drives.
    pub fn new(ctx: &PtxContext) -> Result<Self, PtxError> {
        let compiler = ctx.compiler_probe()?;
        let identity = KernelCompiler::CudaDriver {
            release: compiler.driver_release(),
            compute_capability: compiler.compute_capability(),
        };
        Ok(Self { compiler, identity })
    }
}

impl KernelRegisters for PtxKernelRegisters {
    fn compiler(&self) -> KernelCompiler {
        self.identity
    }

    fn registers(&self, kernel: &CompiledKernel) -> Result<RegisterVerdict, KernelRegistersError> {
        let KernelCode::Ptx(ptx) = kernel.code() else {
            return Err(KernelRegistersError::Refused(
                "the kernel is not PTX".to_string(),
            ));
        };
        match self.compiler.function_registers(ptx, kernel.entry_point()) {
            Ok(registers) => Ok(RegisterVerdict {
                registers: registers.registers,
                local_bytes: registers.local_bytes,
            }),
            Err(ProbeError::Refused(reason)) => Err(KernelRegistersError::Refused(reason)),
            Err(ProbeError::Device(error)) => Err(KernelRegistersError::Device(error.to_string())),
        }
    }
}
