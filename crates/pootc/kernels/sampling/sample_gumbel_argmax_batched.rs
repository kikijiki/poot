//! Temperature + min-p sampling via the Gumbel-max trick, `SampleRule::Gumbel` (card 551a, R472-007):
//! the sampling sibling of `argmax_batched.rs`, same one-workgroup-per-row / 64-lane shape, but it
//! argmaxes a perturbed row `logits[b,i]*inv_temp + noise_scale*noise[b,i]`. By the Gumbel-max identity
//! (`argmax_i(l_i/T + g_i)` is distributed as `softmax(l/T)` for iid standard-Gumbel `g_i`) this is an
//! exact draw from `softmax(logits/T)` when `noise` already carries standard-Gumbel values (the caller's
//! graph composes `RandomUniform` + `gumbel(u) = -log(-log(u))`, so this kernel never hashes or draws
//! noise itself - card 551a moves that off-device-opaque step into the graph, one semantic definition for
//! every backend).
//!
//! min-p (keep tokens with probability `>= min_p * top_prob`) is folded in as a logit floor
//! `max_logit + floor_offset`, computed from a first row-max pass (values only, no index) before the
//! Gumbel pass. `params[b,0..3] = [inv_temp, floor_offset, noise_scale]`: `inv_temp = 1/T` and
//! `floor_offset = T*ln(min_p)` (or a very negative value when `min_p <= 0`) are host-precomputed once
//! per request; `noise_scale` is `0` for a greedy row sharing a mixed-batch dispatch (with `inv_temp = 1`
//! that row's perturbed value reduces exactly to its logit) and `1` otherwise.
//!
//! The row-max pass also finds the lowest non-finite index (`!is_finite`, via the exponent bits), in the
//! same scan (R-551a-2): output is `[B, 2]` I32 `(token, non_finite_index)`, `token` forced to `0` when
//! `non_finite_index >= 0`.
//!
//! Sentinels (JSON has no infinity, so a literal `-inf`/`+inf` start value cannot ride in a `.kir.json`
//! asset): the row-max-value-only reduce starts from `f32::MIN` (the true least finite value - unlike the
//! old, arbitrary `-1.0e30` cutoff this replaces, no finite value can ever be below it, so a plain strict
//! `>` has no gap) and needs no extra "found" bookkeeping; the two index-carrying reduces (the
//! non-finite-index MIN and the final perturbed-argmax) use `-1.0`/`vocab` index sentinels, exactly as
//! `argmax_batched.rs` does, with the same `(value desc, index asc)` tie rule for the perturbed argmax.
//! Finiteness itself is the bounds check `v >= f32::MIN && v <= f32::MAX` (the importer admits no
//! `is_finite`/bit-cast method call; see `argmax_batched.rs` for why this is two `&&`-ed comparisons and
//! not a negated range check).
//!
//! Precondition (host-enforced, not checked here): `inv_temp > 0` for every row (a `0`-noise_scale row
//! stays well-defined with `inv_temp = 1`).
#![crate_type = "lib"]
use poot_kernel_intrinsics::{
    group_index, local_index, no_contract_add, wg_read, wg_write, workgroup_barrier,
};
const W: usize = 64;

pub fn __poot_kernel_sample_gumbel_argmax_batched(
    logits: &[f32],
    noise: &[f32],
    params: &[f32],
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

    // Pass 1: the row's max logit (value only - no tie-break needed, there is no index to report) and
    // the lowest non-finite index, in one strided scan.
    let mut local_max = -3.4028235e38f32; // f32::MIN: the true least finite value
    let mut local_bad = vocab as f32;
    let mut j = lane;
    while j < vocab {
        let v = logits[base + j];
        if v >= -3.4028235e38f32 && v <= 3.4028235e38f32 {
            if v > local_max {
                local_max = v;
            }
        } else if (j as f32) < local_bad {
            local_bad = j as f32;
        }
        j = j + W;
    }
    wg_write(0, lane, local_max);
    wg_write(1, lane, local_bad);
    workgroup_barrier();
    if lane == 0 {
        let mut m = -3.4028235e38f32;
        let mut nb = vocab as f32;
        let mut k = 0usize;
        while k < W {
            let cv = wg_read(0, k);
            if cv > m {
                m = cv;
            }
            let cb = wg_read(1, k);
            if cb < nb {
                nb = cb;
            }
            k = k + 1;
        }
        wg_write(0, 0, m);
        wg_write(1, 0, nb);
    }
    workgroup_barrier();
    let max_logit = wg_read(0, 0);
    let non_finite_idx = wg_read(1, 0);
    let floor = max_logit + floor_offset;

    // Pass 2: Gumbel-perturbed argmax over the kept (>= floor) indices. `noise` is an input (no in-kernel
    // hash); the multiply feeding the add is two separate roundings, kept unfused on every backend by the
    // explicit `no_contract_add` marker (card 675, ADR 0114 tier 1).
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
