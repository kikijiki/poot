//! A raw word-for-word copy kernel: the executor engine's state-commit copy for a device whose recording
//! has no buffer-copy command (`CopyMechanism::Kernel`, ROCm: HSA's copy engine is a queue of its own, not
//! an AQL packet the batched replay could order against). The engine compiles it for the device's target
//! and hands it over as a `CopyKernel`; the device launches it as one dispatch in the recording. One
//! thread per `u32` word, so it is dtype-agnostic: the caller always sizes the dispatch by word count,
//! regardless of the buffer's logical dtype.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index_x;

pub fn __poot_kernel_copy_words(src: &[u32], dst: &mut [u32]) {
    let i = thread_index_x();
    if i < dst.len() {
        dst[i] = src[i];
    }
}
