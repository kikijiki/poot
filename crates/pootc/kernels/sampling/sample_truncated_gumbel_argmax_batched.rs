//! Top-k + temperature/min-p sampling, `SampleRule::GumbelTopK` (card 551a, R472-007):
//! `sample_gumbel_argmax_batched`'s Gumbel-max pass preceded by a threshold-bisection pass that finds the
//! top-k logit threshold `t_k` (the k-th largest logit in the row) without a sort/cumsum/candidate set.
//! min-p and top-k are both monotone-in-logit truncations (each keeps `logit >= some threshold`), so
//! folding the higher threshold into the Gumbel floor makes top-k just another floor source. Top-p is not
//! folded in here (see `sample_topp_gumbel_argmax_batched.rs`): its float mass sum is reduction-order-
//! dependent, so it needs the separate exact-integer accumulation that kernel builds. Top-k's count
//! (`count(logit >= mid)`) is an exact integer in f32 for vocab `< 2^24`, so it stays bit-exact against a
//! CPU oracle running the identical bisection.
//!
//! Shape: one workgroup per row, 64 lanes, the same LDS-reduce idiom as `sample_gumbel_argmax_batched.rs`:
//!   1. row max+min pass (values only) plus the lowest non-finite index, one strided scan (R-551a-2).
//!   2. bisection (skipped when `top_k <= 0`): 30 fixed iterations narrowing `[lo, hi]` from
//!      `[min_logit, max_logit]`. Each iteration: `mid = (lo+hi)*0.5` (IEEE-deterministic, so a CPU oracle
//!      taking the same steps lands on the same bits); each lane counts its strided elements `>= mid`; the
//!      64 counts LDS-reduce (summed) to a total; if `total >= top_k` then `lo = mid`, else `hi = mid`. A
//!      barrier after the branch guards the count array against a write-after-read hazard across
//!      iterations. After 30 iterations `lo` has converged to the k-th largest logit;
//!      `count(logit >= lo) >= top_k` holds by construction.
//!   3. `floor = max(min_p_floor, t_k)` where `t_k = lo` (or `min_logit` when top-k is disabled), and
//!      `min_p_floor` is the `max_logit + floor_offset` formula `sample_gumbel_argmax_batched` uses.
//!   4. Gumbel pass: identical to `sample_gumbel_argmax_batched`, fed this kernel's floor, using the
//!      caller-supplied `noise` (no in-kernel hash).
//!
//! `top_k` is its own I32 operand (one per row; `<= 0` disables top-k truncation).
//! `params[b,0..3] = [inv_temp, floor_offset, noise_scale]`, as `sample_gumbel_argmax_batched`.
//! Sentinels, as `sample_gumbel_argmax_batched.rs`: `f32::MIN`/`f32::MAX` for the value-only max/min
//! reduce (no "found" bookkeeping needed - no finite value can be outside that range), `-1.0`/`vocab`
//! index sentinels for the non-finite-index and perturbed-argmax reduces. Finiteness is the bounds check
//! `v >= f32::MIN && v <= f32::MAX` (see `argmax_batched.rs`).
#![crate_type = "lib"]
use poot_kernel_intrinsics::{
    group_index, local_index, no_contract_add, wg_read, wg_write, workgroup_barrier,
};
const W: usize = 64;

pub fn __poot_kernel_sample_truncated_gumbel_argmax_batched(
    logits: &[f32],
    noise: &[f32],
    params: &[f32],
    top_k: &[i32],
    dims: &[u32],
    out: &mut [i32],
) {
    let b = group_index();
    let vocab = dims[0] as usize;
    let lane = local_index();
    let base = b * vocab;
    let inv_temp = params[b * 3];
    let floor_offset = params[b * 3 + 1];
    let noise_scale = params[b * 3 + 2];
    let k_limit = top_k[b] as f32;

    // Pass 1: the row's max and min logit (values only) and the lowest non-finite index, one strided
    // scan. Array 0 carries per-lane maxima, array 1 per-lane minima, array 2 the per-lane non-finite
    // index.
    let mut local_max = -3.4028235e38f32; // f32::MIN
    let mut local_min = 3.4028235e38f32; // f32::MAX
    let mut local_bad = vocab as f32;
    let mut j = lane;
    while j < vocab {
        let v = logits[base + j];
        if v >= -3.4028235e38f32 && v <= 3.4028235e38f32 {
            if v > local_max {
                local_max = v;
            }
            if v < local_min {
                local_min = v;
            }
        } else if (j as f32) < local_bad {
            local_bad = j as f32;
        }
        j = j + W;
    }
    wg_write(0, lane, local_max);
    wg_write(1, lane, local_min);
    wg_write(2, lane, local_bad);
    workgroup_barrier();
    if lane == 0 {
        let mut m = -3.4028235e38f32;
        let mut n = 3.4028235e38f32;
        let mut nb = vocab as f32;
        let mut k = 0usize;
        while k < W {
            let cv = wg_read(0, k);
            if cv > m {
                m = cv;
            }
            let cn = wg_read(1, k);
            if cn < n {
                n = cn;
            }
            let cb = wg_read(2, k);
            if cb < nb {
                nb = cb;
            }
            k = k + 1;
        }
        wg_write(0, 0, m);
        wg_write(1, 0, n);
        wg_write(2, 0, nb);
    }
    workgroup_barrier();
    let max_logit = wg_read(0, 0);
    let min_logit = wg_read(1, 0);
    let non_finite_idx = wg_read(2, 0);

    // Pass 2: bisect for t_k (the k-th largest logit) on [min_logit, max_logit]. Skipped (lo stays
    // min_logit, no truncation) when top_k <= 0.
    let mut lo = min_logit;
    let mut hi = max_logit;
    if k_limit > 0.0f32 {
        let mut iter = 0usize;
        while iter < 30 {
            let mid = (lo + hi) * 0.5f32;
            let mut local_count = 0.0f32;
            let mut jc = lane;
            while jc < vocab {
                if logits[base + jc] >= mid {
                    local_count = local_count + 1.0f32;
                }
                jc = jc + W;
            }
            wg_write(3, lane, local_count);
            workgroup_barrier();
            if lane == 0 {
                let mut total = 0.0f32;
                let mut kk = 0usize;
                while kk < W {
                    total = total + wg_read(3, kk);
                    kk = kk + 1;
                }
                wg_write(3, 0, total);
            }
            workgroup_barrier();
            let total_count = wg_read(3, 0);
            if total_count >= k_limit {
                lo = mid;
            } else {
                hi = mid;
            }
            // Guard array 3 against a write-after-read hazard: the next iteration's wg_write(3, lane, ..)
            // must not race this iteration's wg_read(3, 0) above.
            workgroup_barrier();
            iter = iter + 1;
        }
    }
    let t_k = lo;
    let min_p_floor = max_logit + floor_offset;
    let floor = if min_p_floor > t_k { min_p_floor } else { t_k };

    // Pass 3: Gumbel-perturbed argmax over the kept (>= floor) indices, as in
    // sample_gumbel_argmax_batched's pass 2, fed this kernel's (possibly top-k-tightened) floor. The
    // multiply feeding the add is two separate roundings, kept unfused on every backend by the explicit
    // `no_contract_add` marker (card 675, ADR 0114 tier 1).
    let mut best_val = 0.0f32;
    let mut best_idx = -1.0f32;
    let mut j2 = lane;
    while j2 < vocab {
        let v = logits[base + j2];
        if v >= floor {
            let scaled = v * inv_temp;
            let noise_term = noise_scale * noise[base + j2];
            let perturbed = no_contract_add(scaled, noise_term);
            if best_idx < 0.0 || perturbed > best_val {
                best_val = perturbed;
                best_idx = j2 as f32;
            }
        }
        j2 = j2 + W;
    }
    wg_write(0, lane, best_val);
    wg_write(1, lane, best_idx);
    workgroup_barrier();
    if lane == 0 {
        let mut bv = 0.0f32;
        let mut bi = -1.0f32;
        let mut k2 = 0usize;
        while k2 < W {
            let cv = wg_read(0, k2);
            let ci = wg_read(1, k2);
            if ci >= 0.0 && (bi < 0.0 || cv > bv || (cv == bv && ci < bi)) {
                bv = cv;
                bi = ci;
            }
            k2 = k2 + 1;
        }
        let non_finite = non_finite_idx < (vocab as f32);
        let picked = if bi < 0.0 { 0.0f32 } else { bi };
        let token = if non_finite { 0.0f32 } else { picked };
        let flag = if non_finite { non_finite_idx } else { -1.0f32 };
        out[b * 2] = token as i32;
        out[b * 2 + 1] = flag as i32;
    }
}
