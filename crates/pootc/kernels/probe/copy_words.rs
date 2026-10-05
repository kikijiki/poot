//! A raw word-for-word copy kernel: ROCm's `Device::copy` expresses the executor
//! contract's two-phase commit copy as one generated kernel dispatch (an AQL packet the batched
//! capture/replay model can record), never a raw HSA memcpy mid-recording. One thread per `u32`
//! word, so it is dtype-agnostic: the caller always sizes the dispatch by word count, regardless of
//! the buffer's logical dtype.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_copy_words(src: &[u32], dst: &mut [u32]) {
    let i = thread_index();
    if i < dst.len() {
        dst[i] = src[i];
    }
}
