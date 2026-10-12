use crate::RuntimeError;
use poot_runtime_common::CompiledKernel;

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
