//! A kernel-authored type that merely has `Range`, `Option` and `Iter` in its name is an ordinary (unsupported)
//! type: `pootc` rejects it by name instead of treating it as loop plumbing and mapping it to nothing.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index_x;

struct RangeOptionIter {
    start: usize,
    end: usize,
}

pub fn __poot_kernel_range_like_type(out: &mut [f32]) {
    let i = thread_index_x();
    let span = RangeOptionIter { start: 0, end: 4 };
    if i < out.len() {
        out[i] = (span.end - span.start) as f32;
    }
}
