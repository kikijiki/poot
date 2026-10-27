//! Top-p (nucleus) + top-k + temperature/min-p sampling, `SampleRule::GumbelTopKTopP` (card 551a,
//! R472-007, R-551a-5): extends `sample_truncated_gumbel_argmax_batched` (top-k) with a top-p bisection
//! after the min-p/top-k floor, so temperature / min-p / top-k / top-p all sample in one dispatch
//! (penalties, logit-bias, logprobs, and guided decoding stay a host fallback, card 551b).
//!
//! Why top-p needs an exact integer mass: min-p and top-k are monotone-in-logit truncations found by an
//! exact-integer bisection (a logit floor, or a `count(logit >= mid)` that is an exact integer in f32),
//! so a GPU tree-reduce and a CPU sequential reduce land on the same bits. Top-p's boundary is a
//! probability-mass sum whose float value depends on reduction order, which can flip the last token in
//! the nucleus and break the bit-exact oracle (tier 2 on the noise/membership boundary, ADR-0101). Fix:
//! each token's mass is the integer `q = round(exp((l-max)/T) * 2^24)` (`poot-eval`'s
//! `top_p_token_mass`, at most `2^24`), so a token down to `2^-25` of the top token keeps a nonzero mass
//! and the tail of a flat distribution is summed, not rounded away. Integer addition is exactly
//! associative, so the sum is order-independent.
//!
//! The sum is a two-limb integer in base `2^24` (`hi * 2^24 + lo`, `lo < 2^24`), because the row total
//! (up to `2^24 * vocab`) does not fit an f32 and the workgroup arrays hold f32. Each lane carries its
//! own two limbs (each f32-exact: `lo < 2^24`, `hi <= vocab / 64`), writes them to arrays 3 and 4, and
//! after a barrier every lane folds the 64 partials in `u32` and normalizes the carry, so there is no
//! vocabulary bound below the `u32` limbs. No scratch, no atomics (R-551a-5): the bisection is entirely
//! workgroup-local (one workgroup per row).
//!
//! The threshold is the one rounded quantity, and it is rounded identically on the host: `Z` is the
//! total converted to f32 once (`hi * 2^24` is exact, the add rounds once, as the host's `u64 as f32`),
//! `tp_thresh = round(top_p * Z)` in f32, then split back into the same two limbs (exact: a division by
//! a power of two and a subtraction). The compared `mass_above >= tp_thresh` is exact integer order.
//!
//! Shape: one workgroup per row, 64 lanes, same LDS-reduce idiom as `sample_truncated_gumbel_argmax_batched`:
//!   1. row max/min pass (values only) + the lowest non-finite index (R-551a-2), as in the top-k kernel.
//!   2. top-k bisection -> `t_k`; `floor1 = max(min_p_floor, t_k)`, as in the top-k kernel.
//!   3. Top-p step (three cases matching the CPU sampler's `top_p < 1.0` gate and its `cum >= top_p` cut,
//!      which fires on the first token when `top_p <= 0`):
//!      - `top_p <= 0`: collapse to top-1 (`floor = max_logit`). This must not be treated as "disabled":
//!        skipping it would sample from the unrestricted min-p/top-k distribution, disagreeing with the
//!        CPU sampler's top-1 cut.
//!      - `0 < top_p < 1`: bisection.
//!        a. `Z_trunc` = the exact two-limb quantized mass over `logit >= floor1`.
//!        b. `tp_thresh = round(top_p * Z_trunc)` (identical f32 arithmetic on GPU and CPU).
//!        c. bisect `t_p` over `[floor1, max_logit]` (30 iterations): each iteration LDS-sums the
//!           quantized mass over `logit >= mid`; if `mass_above >= tp_thresh` then `lo = mid`, else
//!           `hi = mid`. `t_p = lo` (the tightest logit whose kept mass still covers `top_p`).
//!        d. `floor = max(floor1, t_p)` (`= t_p`, since `lo` starts at `floor1`).
//!      - `top_p >= 1`: disabled, `floor` stays `floor1`.
//!   4. Gumbel-perturbed argmax over `logit >= floor`, as in the top-k kernel, fed this kernel's
//!      (possibly top-p-tightened) floor.
//!
//! `top_k` is its own I32 operand. `params[b,0..4] = [inv_temp, floor_offset, noise_scale, top_p]`
//! (one extra column versus the other rules).
#![crate_type = "lib"]
use poot_kernel_attr::kernel;
use poot_kernel_intrinsics::{
    group_index_x, local_index_x, no_contract_add, wg_read, wg_write, workgroup_barrier,
};
const W: usize = 64;

#[kernel(workgroup(64), lds(f32, 64), lds(f32, 64), lds(f32, 64), lds(f32, 64), lds(f32, 64))]
pub fn sample_topp_gumbel_argmax_batched(
    logits: &[f32],
    noise: &[f32],
    params: &[f32],
    top_k: &[i32],
    dims: &[u32],
    out: &mut [i32],
) {
    let b = group_index_x();
    let vocab = dims[0] as usize;
    let lane = local_index_x();
    let base = b * vocab;
    let inv_temp = params[b * 4];
    let floor_offset = params[b * 4 + 1];
    let noise_scale = params[b * 4 + 2];
    let top_p = params[b * 4 + 3];
    let k_limit = top_k[b] as f32;

    // Pass 1: the row's max/min logit (values only) and the lowest non-finite index, as in the top-k
    // kernel.
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
            // The oracle's `MASKED_LOGIT` (f32::MIN) is an excluded token, not a candidate floor.
            if v < local_min && v > -3.4028235e38f32 {
                local_min = v;
            }
        } else if (j as f32) < local_bad {
            local_bad = j as f32;
        }
        j = j + W;
    }
    wg_write::<f32>(0, lane, local_max);
    wg_write::<f32>(1, lane, local_min);
    wg_write::<f32>(2, lane, local_bad);
    workgroup_barrier();
    if lane == 0 {
        let mut m = -3.4028235e38f32;
        let mut n = 3.4028235e38f32;
        let mut nb = vocab as f32;
        let mut k = 0usize;
        while k < W {
            let cv = wg_read::<f32>(0, k);
            if cv > m {
                m = cv;
            }
            let cn = wg_read::<f32>(1, k);
            if cn < n {
                n = cn;
            }
            let cb = wg_read::<f32>(2, k);
            if cb < nb {
                nb = cb;
            }
            k = k + 1;
        }
        wg_write::<f32>(0, 0, m);
        wg_write::<f32>(1, 0, n);
        wg_write::<f32>(2, 0, nb);
    }
    workgroup_barrier();
    let max_logit = wg_read::<f32>(0, 0);
    let min_logit = wg_read::<f32>(1, 0);
    let non_finite_idx = wg_read::<f32>(2, 0);

    // Pass 2: top-k bisection for t_k on [min_logit, max_logit] (skipped when top_k <= 0), as in the
    // top-k kernel. floor1 folds min-p and top-k into one logit floor.
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
            wg_write::<f32>(3, lane, local_count);
            workgroup_barrier();
            if lane == 0 {
                let mut total = 0.0f32;
                let mut kk = 0usize;
                while kk < W {
                    total = total + wg_read::<f32>(3, kk);
                    kk = kk + 1;
                }
                wg_write::<f32>(3, 0, total);
            }
            workgroup_barrier();
            let total_count = wg_read::<f32>(3, 0);
            if total_count >= k_limit {
                lo = mid;
            } else {
                hi = mid;
            }
            workgroup_barrier();
            iter = iter + 1;
        }
    }
    let t_k = lo;
    let min_p_floor = max_logit + floor_offset;
    let floor1 = if min_p_floor > t_k { min_p_floor } else { t_k };

    // Pass 3: top-p (nucleus) bisection for t_p on [floor1, max_logit]. Skipped when top_p is outside
    // the open interval (0,1); floor then stays floor1. The mass is the two-limb fixed-point integer the
    // module doc describes, in LDS arrays 3 and 4 (array 3 is reused from pass 2 - every write there
    // happened under a barrier this pass's first write is ordered after).
    let mut floor = floor1;
    if top_p <= 0.0f32 {
        // top_p <= 0: collapse to top-1 (only the max-logit token survives). Matches the CPU sampler's
        // nucleus cut of exactly 1 token. Do not take the disabled shortcut: it would leave `floor` at
        // `floor1` and sample from the unrestricted min-p/top-k distribution.
        floor = max_logit;
    } else if top_p < 1.0f32 {
        // Z_trunc: the exact quantized mass over the post-min-p/top-k set (logit >= floor1), as the
        // (hi, lo) base-2^24 limbs `mass_limbs` describes, then the f32 `Z` the threshold is rounded from.
        let mut z_hi = 0u32;
        let mut z_lo = 0u32;
        let mut jz = lane;
        let mut acc_hi = 0u32;
        let mut acc_lo = 0u32;
        while jz < vocab {
            let v = logits[base + jz];
            if v >= floor1 {
                let q = (((v - max_logit) * inv_temp).exp() * 16777216.0f32).round() as u32;
                let sum = acc_lo + q;
                if sum >= 16777216u32 {
                    acc_hi = acc_hi + 1u32;
                    acc_lo = sum - 16777216u32;
                } else {
                    acc_lo = sum;
                }
            }
            jz = jz + W;
        }
        wg_write::<f32>(3, lane, acc_hi as f32);
        wg_write::<f32>(4, lane, acc_lo as f32);
        workgroup_barrier();
        let mut kk = 0usize;
        while kk < W {
            z_hi = z_hi + wg_read::<f32>(3, kk) as u32;
            z_lo = z_lo + wg_read::<f32>(4, kk) as u32;
            kk = kk + 1;
        }
        z_hi = z_hi + (z_lo >> 24);
        z_lo = z_lo & 16777215u32;
        workgroup_barrier();
        // `top_p_threshold`: the f32 product, rounded. `Z` is the correctly rounded f32 of the exact
        // integer (`hi * 2^24` is exact; the add rounds once), as the host's `u64 as f32` is.
        let z_f = (z_hi as f32) * 16777216.0f32 + (z_lo as f32);
        let tp_thresh = (top_p * z_f).round();
        let thr_hi_f = (tp_thresh / 16777216.0f32).floor();
        let thr_hi = thr_hi_f as u32;
        let thr_lo = (tp_thresh - thr_hi_f * 16777216.0f32) as u32;

        let mut tlo = floor1;
        let mut thi = max_logit;
        let mut it = 0usize;
        while it < 30 {
            let mid = (tlo + thi) * 0.5f32;
            let mut m_hi = 0u32;
            let mut m_lo = 0u32;
            let mut lane_hi = 0u32;
            let mut lane_lo = 0u32;
            let mut jm = lane;
            while jm < vocab {
                let v = logits[base + jm];
                if v >= mid {
                    let q = (((v - max_logit) * inv_temp).exp() * 16777216.0f32).round() as u32;
                    let sum = lane_lo + q;
                    if sum >= 16777216u32 {
                        lane_hi = lane_hi + 1u32;
                        lane_lo = sum - 16777216u32;
                    } else {
                        lane_lo = sum;
                    }
                }
                jm = jm + W;
            }
            wg_write::<f32>(3, lane, lane_hi as f32);
            wg_write::<f32>(4, lane, lane_lo as f32);
            workgroup_barrier();
            let mut k3 = 0usize;
            while k3 < W {
                m_hi = m_hi + wg_read::<f32>(3, k3) as u32;
                m_lo = m_lo + wg_read::<f32>(4, k3) as u32;
                k3 = k3 + 1;
            }
            m_hi = m_hi + (m_lo >> 24);
            m_lo = m_lo & 16777215u32;
            if m_hi > thr_hi || (m_hi == thr_hi && m_lo >= thr_lo) {
                tlo = mid;
            } else {
                thi = mid;
            }
            workgroup_barrier();
            it = it + 1;
        }
        let t_p = tlo;
        floor = if floor1 > t_p { floor1 } else { t_p };
    }
    // top_p >= 1.0: top-p disabled; floor stays floor1.

    // Pass 4: Gumbel-perturbed argmax over the kept (>= floor) indices, as in the top-k kernel, fed this
    // kernel's (possibly top-p-tightened) floor. The multiply feeding the add is two separate roundings,
    // kept unfused on every backend by the explicit `no_contract_add` marker (card 675, ADR 0114 tier 1).
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
    wg_write::<f32>(0, lane, best_val);
    wg_write::<f32>(1, lane, best_idx);
    workgroup_barrier();
    if lane == 0 {
        let mut bv = 0.0f32;
        let mut bi = -1.0f32;
        let mut k2 = 0usize;
        while k2 < W {
            let cv = wg_read::<f32>(0, k2);
            let ci = wg_read::<f32>(1, k2);
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
