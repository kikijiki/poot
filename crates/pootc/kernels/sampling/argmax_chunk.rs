//! Greedy sampling, `SampleRule::Greedy`, stage 1 of 2 (POOT-1146): one workgroup per (row, chunk) reduces
//! one contiguous `chunk` of that row's `vocab` logits to a partial `(best_value, best_index, bad_index)`.
//! `argmax_combine.rs` reduces a row's partials to the row's `(token, non_finite_index)`. A decode step's
//! one 151,936-wide row is `ceil(vocab / chunk)` workgroups that run in parallel; the single-workgroup
//! body this replaces scanned the whole row with 64 lanes (about 750 us at that width on gfx1151).
//!
//! `logits` is flat `[B, vocab]` row-major; `dims = [vocab, chunk, groups]` with `groups = ceil(vocab /
//! chunk)` (at least one). The grid is `groups` workgroups along X and `B` along Y: `group_index()` is
//! the chunk, `group_index_y()` the row. Group `g` of row `b` owns `[g * chunk, min((g + 1) * chunk,
//! vocab))` and writes its three words at `partials[(b * groups + g) * 3 ..]`.
//!
//! The pick is the one `poot_eval`'s `SampleToken { Greedy }` defines, split without changing it:
//! - `best_idx` starts at the sentinel `-1.0` ("no finite candidate yet": an actual `-inf` start value
//!   cannot ride in a `.kir.json` asset, since JSON has no infinity). A finite value replaces the running
//!   best only on a strict `>`, so a lane keeps the first (lowest-index) maximum of its strided slice.
//! - `bad_idx` starts at `vocab` (an index no real element has) and takes the smallest index whose logit
//!   is non-finite. The importer admits no `is_finite` method call, so finiteness is the bounds check
//!   `v >= f32::MIN && v <= f32::MAX` (true for every finite value, false for NaN and `+-inf` alike: NaN
//!   compares false against both bounds, which is why this is two `&&`-ed comparisons).
//! - Two candidates combine by the larger value and, on an exact tie (`-0.0 == 0.0` included), the smaller
//!   index: an associative, commutative rule, so the lane, chunk and group split cannot change the answer.
//!
//! Indices travel as `f32` (LDS and the partials are `f32`-only), exact below 2^24 logits.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{
    group_index, group_index_y, local_index, wg_read, wg_write, workgroup_barrier,
};
const W: usize = 64;

pub fn __poot_kernel_argmax_chunk(logits: &[f32], dims: &[u32], partials: &mut [f32]) {
    let vocab = dims[0] as usize;
    let chunk = dims[1] as usize;
    let groups = dims[2] as usize;
    let row = group_index_y();
    let part = group_index();
    let lane = local_index();
    let start = part * chunk;
    let mut end = start + chunk;
    if end > vocab {
        end = vocab;
    }
    let base = row * vocab;

    let mut best_val = 0.0f32;
    let mut best_idx = -1.0f32;
    let mut bad_idx = vocab as f32;
    let mut j = start + lane;
    while j < end {
        let v = logits[base + j];
        if v >= -3.4028235e38f32 && v <= 3.4028235e38f32 {
            if best_idx < 0.0 || v > best_val {
                best_val = v;
                best_idx = j as f32;
            }
        } else if (j as f32) < bad_idx {
            bad_idx = j as f32;
        }
        j = j + W;
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
        let p = (row * groups + part) * 3;
        partials[p] = bv;
        partials[p + 1] = bi;
        partials[p + 2] = nb;
    }
}
