//! Parallel argmax, stage 1 of 2 (card 657): every workgroup reduces one contiguous chunk of the logits to a
//! `(value, index)` partial. `argmax_finalize.rs` reduces the partials to the winning index. The launch is
//! `groups = partials.len() / 2` workgroups (`poot_graph_plan::sampler_bodies::ArgmaxLaunch`); group `g` owns
//! `[g * chunk, (g + 1) * chunk)` with `chunk = ceil(n / groups)`, and its 256 lanes stride over the chunk.
//!
//! The result is the CPU oracle's greedy pick bit for bit: the first index of the maximum, where a value is a
//! candidate iff the oracle's scan (`x > best`, from `best = -inf`) could take it, i.e. `x >= f32::MIN`. So a
//! NaN or `-inf` logit is never picked, and a row with no candidate yields index 0. The `-inf` start state
//! cannot ride in a KIR asset (JSON has no infinity), so "no candidate yet" is an index of `-1.0` instead.
//! Two partials combine by the larger value, and on an equal value (`-0.0 == 0.0` included) by the smaller
//! index: an associative, commutative rule, so the chunk and lane split cannot change the answer.
//!
//! Indices travel as `f32` (LDS is `f32`-only), exact below 2^24 logits.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, wg_read, wg_write, workgroup_barrier};
const WORKGROUP_SIZE: usize = 256;
pub fn __poot_kernel_argmax_partials(logits: &[f32], partials: &mut [f32]) {
    let lane = local_index();
    let g = group_index();
    let n = logits.len();
    let groups = partials.len() / 2;
    let divisor = groups.max(1);
    let chunk = (n + divisor - 1) / divisor;
    let start = (g * chunk).min(n);
    let end = (start + chunk).min(n);
    // This lane's first maximum over its strided slice of the chunk (indices ascend, so strict `>` keeps
    // the first).
    let mut found = false;
    let mut best_val = 0.0f32;
    let mut best_idx = 0usize;
    let mut j = start + lane;
    while j < end {
        let v = logits[j];
        // `-3.4028235e38` is `f32::MIN` (the importer does not lower that path const): false for NaN and
        // `-inf`, true for every value the oracle's scan from `-inf` can take.
        if v >= -3.4028235e38f32 && (!found || v > best_val) {
            found = true;
            best_val = v;
            best_idx = j;
        }
        j = j + WORKGROUP_SIZE;
    }
    wg_write(0, lane, best_val);
    if found {
        wg_write(1, lane, best_idx as f32);
    } else {
        wg_write(1, lane, -1.0);
    }
    workgroup_barrier();
    if lane == 0 && g < groups {
        let mut bv = 0.0f32;
        let mut bi = -1.0f32;
        let mut k = 0usize;
        while k < WORKGROUP_SIZE {
            let cv = wg_read(0, k);
            let ci = wg_read(1, k);
            if ci >= 0.0 && (bi < 0.0 || cv > bv || (cv == bv && ci < bi)) {
                bv = cv;
                bi = ci;
            }
            k = k + 1;
        }
        partials[2 * g] = bv;
        partials[2 * g + 1] = bi;
    }
}
