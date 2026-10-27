//! The one handle a runtime takes for an NPU region: the validated artifact wrapped as a `CompiledKernel`.
//!
//! An artifact is only constructible through its own validator (an xclbin, a whole instruction stream and a
//! self-consistent shape ABI for an AMD region; dense F16, F32 and I32 bindings of non-empty shape for an Intel
//! graph), so the argument schema derived here describes it exactly. This is the sanctioned unsafe constructor
//! the compiler's own byte-kernel handle (`poot_codegen::kernel_handle`) is for the GPU backends.

use poot_codegen::{ArgSchema, CompiledKernel, KernelCode};
use poot_npu_artifacts::{AieLinkedArtifact, VpuGraphArtifact};
use poot_target::{ArgAccess, TargetId};

/// The [`CompiledKernel`] for an AIE region's linked `artifact`. Its arguments are the artifact's shape
/// ABI, binding for binding: the host side of the region. Which core runs which body was checked tile by
/// tile when the artifact was compiled (`compile_aie`), and the artifact itself is only constructible
/// through the validator, so there is nothing left to refuse here.
pub fn aie_kernel_handle(artifact: AieLinkedArtifact) -> CompiledKernel {
    let args: Box<[ArgSchema]> = artifact
        .abi()
        .bindings()
        .iter()
        .map(|binding| ArgSchema::new(binding.storage().element(), binding.access()))
        .collect();
    let entry_point: Box<str> = artifact.entry().into();
    let target = TargetId::Aie(artifact.target());
    // SAFETY: `artifact` came out of the artifact validator (an xclbin, a whole instruction stream and a
    // self-consistent shape ABI), and `args` is that ABI's bindings in order with their element kinds and
    // write flags, so the schema describes the artifact exactly. Every tile body was checked against the
    // recipe the ABI belongs to at compile time, and a trapping body was refused there, so `has_trap` is
    // false. The generator's proved access ranges are inside every binding's extent (checked at
    // construction).
    unsafe { CompiledKernel::new(target, entry_point, KernelCode::Aie(artifact), args, false) }
}

/// The [`CompiledKernel`] for an Intel NPU region's `artifact`. Its arguments are the artifact's inputs and then
/// its outputs: the region reads the first and writes the rest.
pub fn vpu_kernel_handle(artifact: VpuGraphArtifact) -> CompiledKernel {
    let args: Box<[ArgSchema]> = artifact
        .inputs()
        .iter()
        .map(|b| ArgSchema::new(b.storage().element(), ArgAccess::Read))
        .chain(
            artifact
                .outputs()
                .iter()
                .map(|b| ArgSchema::new(b.storage().element(), ArgAccess::Write)),
        )
        .collect();
    let target = TargetId::Vpu(artifact.target());
    // SAFETY: `artifact` is only constructible through `VpuGraphArtifact::new`, which admits only dense F16,
    // F32 and I32 bindings of non-empty shape, and `args` is exactly its inputs (read) then outputs
    // (written) in order, so the schema describes the artifact. The graph has no trap word, and the device
    // checks the compiled graph's own argument list against these bindings before it executes.
    unsafe {
        CompiledKernel::new(
            target,
            "poot_region",
            KernelCode::Vpu(artifact),
            args,
            false,
        )
    }
}
