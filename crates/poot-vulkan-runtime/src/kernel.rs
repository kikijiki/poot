use crate::*;

/// The compiler-produced kernel handle every dispatch takes (card 608): re-exported so callers need not
/// depend on `poot-runtime-common` directly just to name the type. This crate's entry points that read a
/// kernel ([`Context::dispatch`], [`Context::pipeline`]) take a `&CompiledKernel`, and every dispatch
/// ([`Context::dispatch`], [`Context::record_dispatch`]) checks the bound buffers' count and native
/// element kind against [`CompiledKernel::args`] before doing anything else (SC-002); its fields are
/// private and its only constructor is `unsafe`, called only by
/// `poot-codegen`, so safe code outside the compiler cannot mint one:
///
/// ```compile_fail,E0133
/// fn from_any_bytes(spv: &[u8]) -> poot_vulkan_runtime::CompiledKernel {
///     poot_vulkan_runtime::CompiledKernel::new(
///         poot_target::Backend::SpirvVulkan,
///         "main",
///         poot_runtime_common::KernelCode::spirv_from_le_bytes(spv),
///         Vec::new(),
///         false,
///     )
/// }
/// ```
///
/// ```compile_fail,E0451
/// fn forge(code: poot_runtime_common::KernelCode) -> poot_vulkan_runtime::CompiledKernel {
///     poot_vulkan_runtime::CompiledKernel {
///         target: poot_target::Backend::SpirvVulkan,
///         entry_point: "main".into(),
///         code,
///         args: Vec::new().into(),
///         has_trap: false,
///     }
/// }
/// ```
pub use poot_runtime_common::CompiledKernel;

/// The SPIR-V words of a compiler-produced `kernel`, or a typed error if it was compiled for another
/// backend (card 608): the one check every entry point that reads a `CompiledKernel`'s code runs before
/// touching it.
pub(crate) fn spirv_words(kernel: &CompiledKernel) -> Result<&[u32], RuntimeError> {
    match kernel.code() {
        poot_runtime_common::KernelCode::SpirvWords(words) => Ok(words.as_ref()),
        _ => Err(RuntimeError::WrongKernelTarget {
            actual: kernel.target(),
        }),
    }
}
