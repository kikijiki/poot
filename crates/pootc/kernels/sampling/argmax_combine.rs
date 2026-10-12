//! Greedy sampling, `SampleRule::Greedy`, stage 2 of 2 (POOT-1146): one workgroup of 64 lanes per row
//! reduces that row's `groups` partials (`argmax_chunk.rs`) to the row's `(token, non_finite_index)`.
//! `dims = [groups, vocab]`; `partials` is `[B, groups, 3]` of `(best_value, best_index, bad_index)`;
//! `out` is `[B, 2]` I32.
//!
//! Partials combine by the same rule stage 1 reduces its lanes with: the larger value wins and, on an
//! exact tie, the smaller index. A partial with `best_index < 0` holds no finite candidate and never
//! wins. `bad_index` is the smallest non-finite index over the partials (`vocab` when none). `token` is
//! forced to `0` when the row has a non-finite logit or no finite candidate, so the output is fully
//! determined even though the driver never commits it (R482-005).
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, wg_read, wg_write, workgroup_barrier};
const W: usize = 64;

pub fn __poot_kernel_argmax_combine(partials: &[f32], dims: &[u32], out: &mut [i32]) {
    let groups = dims[0] as usize;
    let vocab = dims[1] as usize;
    let b = group_index();
    let lane = local_index();

    let mut best_val = 0.0f32;
    let mut best_idx = -1.0f32;
    let mut bad_idx = vocab as f32;
    let mut g = lane;
    while g < groups {
        let p = (b * groups + g) * 3;
        let cv = partials[p];
        let ci = partials[p + 1];
        if ci >= 0.0 && (best_idx < 0.0 || cv > best_val || (cv == best_val && ci < best_idx)) {
            best_val = cv;
            best_idx = ci;
        }
        let cb = partials[p + 2];
        if cb < bad_idx {
            bad_idx = cb;
        }
        g = g + W;
    }
    wg_write(0, lane, best_val);
    wg_write(1, lane, best_idx);
    wg_write(2, lane, bad_idx);
    workgroup_barrier();
    if lane == 0 {
        let mut bv = 0.0f32;
        let mut bi = -1.0f32;
        let mut nb = vocab as f32;
        let mut k = 0usize;
        while k < W {
            let cv = wg_read(0, k);
            let ci = wg_read(1, k);
            if ci >= 0.0 && (bi < 0.0 || cv > bv || (cv == bv && ci < bi)) {
                bv = cv;
                bi = ci;
            }
            let cb = wg_read(2, k);
            if cb < nb {
                nb = cb;
            }
            k = k + 1;
        }
        let non_finite = nb < (vocab as f32);
        let token = if non_finite || bi < 0.0 { 0.0f32 } else { bi };
        let flag = if non_finite { nb } else { -1.0f32 };
        out[b * 2] = token as i32;
        out[b * 2 + 1] = flag as i32;
    }
}
