//! Batched greedy sampling, `SampleRule::Greedy` (card 551a, R472-007, fixing the three bugs
//! deval.md section 8 names in the Card 447 consolidation this replaces). One workgroup per row
//! (`b = group_index()`), 64 lanes strided-scanning that row's `vocab` logits into LDS, a barrier, then
//! lane 0 reduces. `logits` is flat `[B, vocab]` row-major (`B` derived from `logits.len()/vocab`);
//! `dims[0]` carries `vocab`. Output is `[B, 2]` I32: `(token, non_finite_index)`.
//!
//! Per lane: `best_idx` starts at the sentinel `-1.0` ("no finite candidate yet" - an actual `-inf` start
//! value cannot ride in a `.kir.json` asset, since JSON has no infinity; `-1.0` plays the same role
//! `argmax_partials.rs`/`argmax_finalize.rs` already use). A finite value only replaces the running best
//! on a strict `>` (R472-007's old `-1.0e30` sentinel wrongly rejected every real value below it; `-1.0`
//! as an INDEX sentinel, checked separately from the VALUE comparison, has no such gap - the very first
//! finite value a lane sees always becomes its candidate). `bad_idx` starts at `vocab` (an index no real
//! element has) and takes the smallest index whose logit is non-finite: the importer admits no
//! `is_finite`/bit-cast method call, so finiteness is the bounds check `v >= f32::MIN && v <= f32::MAX`
//! (true for every finite value, false for NaN and `+-inf` alike - NaN compares false against both
//! bounds, which is exactly why this is two `&&`-ed comparisons and not a negated range check).
//!
//! Cross-lane reduce, in both passes: larger value wins; on an exact tie, the smaller index wins
//! (`cv == bv && ci < bi`) - deterministic regardless of lane/scan order. `token` is forced to `0` when
//! `non_finite_index >= 0` (the row has a non-finite logit), so the output is fully determined even
//! though the driver never commits it (card 551b, R482-005).
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, wg_read, wg_write, workgroup_barrier};
const W: usize = 64;

pub fn __poot_kernel_argmax_batched(logits: &[f32], dims: &[u32], out: &mut [i32]) {
    let b = group_index();
    let vocab = dims[0] as usize;
    let lane = local_index();
    let base = b * vocab;

    let mut best_val = 0.0f32;
    let mut best_idx = -1.0f32;
    let mut bad_idx = vocab as f32;
    let mut j = lane;
    while j < vocab {
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
        let non_finite = nb < (vocab as f32);
        let token = if non_finite || bi < 0.0 { 0.0f32 } else { bi };
        let flag = if non_finite { nb } else { -1.0f32 };
        out[b * 2] = token as i32;
        out[b * 2 + 1] = flag as i32;
    }
}
