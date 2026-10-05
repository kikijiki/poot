//! Parallel argmax, stage 2 of 2 (card 657): one workgroup reduces the `(value, index)` partials that
//! `argmax_partials.rs` wrote (`partials.len() / 2` of them; an index of `-1.0` is an empty chunk) with the
//! same rule, larger value first and the smaller index on an equal value, and writes the winning index (as
//! `f32`) to `out[0]`. No candidate anywhere (every logit NaN or `-inf`) yields index 0, as the CPU oracle's
//! scan from `-inf` does. See `argmax_partials.rs` for the tie and NaN contract.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{local_index, wg_read, wg_write, workgroup_barrier};
const WORKGROUP_SIZE: usize = 256;
pub fn __poot_kernel_argmax_finalize(partials: &[f32], out: &mut [f32]) {
    let lane = local_index();
    let groups = partials.len() / 2;
    let mut bv = 0.0f32;
    let mut bi = -1.0f32;
    let mut k = lane;
    while k < groups {
        let cv = partials[2 * k];
        let ci = partials[2 * k + 1];
        if ci >= 0.0 && (bi < 0.0 || cv > bv || (cv == bv && ci < bi)) {
            bv = cv;
            bi = ci;
        }
        k = k + WORKGROUP_SIZE;
    }
    wg_write(0, lane, bv);
    wg_write(1, lane, bi);
    workgroup_barrier();
    if lane == 0 {
        let mut fv = 0.0f32;
        let mut fi = -1.0f32;
        let mut m = 0usize;
        while m < WORKGROUP_SIZE {
            let cv = wg_read(0, m);
            let ci = wg_read(1, m);
            if ci >= 0.0 && (fi < 0.0 || cv > fv || (cv == fv && ci < fi)) {
                fv = cv;
                fi = ci;
            }
            m = m + 1;
        }
        if fi < 0.0 {
            out[0] = 0.0;
        } else {
            out[0] = fi;
        }
    }
}
