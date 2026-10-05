//! A looping kernel: per-row sum over COLS columns via a `while` loop (the serial-reduce shape, like
//! `poot_test_util::kernel_fixtures::reduce_last`, a thin f32 wrapper over kernelgen's own
//! `reduce_last_dt`). Tests that the importer handles a loop (header SwitchInt + back-edge
//! Goto + an accumulator) without the `for`-range iterator machinery.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

const COLS: usize = 4;

pub fn __poot_kernel_rowsum(x: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r < out.len() {
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < COLS {
            acc = acc + x[r * COLS + j];
            j = j + 1;
        }
        out[r] = acc;
    }
}
