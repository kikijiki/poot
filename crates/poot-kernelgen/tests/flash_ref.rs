//! Lock down the online-softmax decode algorithm on the CPU (no GPU) before translating it into the
//! `flash_attention_decode` Body kernel. The kernel uses a private `o[D]` array, so it is NVPTX-only (rejected
//! on SpirvVulkan) and can only run on RunPod; verifying the algorithm here means the RunPod run confirms
//! lowering, not math.
//!
//! `flash_decode_ref` is the exact recurrence the kernel emits; `direct_decode` is the textbook two-pass
//! `softmax(scale*qkᵀ + mask) @ v` (what `ops::attention_masked` decomposes to). They must match.

use poot_test_util::assert_close;

/// Online-softmax decode attention, one (q) token over `cap` cached keys, per head, with GQA. Layout:
/// q[h*D + d], k/v[kv*cap*D + t*D + d], mask[t] (additive), out[h*D + d]. `kv = h / n_rep`.
/// This is the precise sequence of scalar ops the Body kernel emits (one thread per head h).
#[allow(clippy::too_many_arguments)]
fn flash_decode_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    hq: usize,
    n_rep: usize,
    cap: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kv = h / n_rep;
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut o = vec![0.0f32; d];
        for t in 0..cap {
            // s = scale * (q[h] . k[kv,t]) + mask[t]
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[kv * cap * d + t * d + dd];
            }
            s = s * scale + mask[t];
            let m_new = m.max(s);
            let corr = (m - m_new).exp(); // exp(-inf) = 0 on the first (m = -inf) step
            let e = (s - m_new).exp();
            l = l * corr + e;
            for dd in 0..d {
                o[dd] = o[dd] * corr + e * v[kv * cap * d + t * d + dd];
            }
            m = m_new;
        }
        for dd in 0..d {
            out[h * d + dd] = o[dd] / l;
        }
    }
    out
}

/// Textbook two-pass decode attention (the decomposed reference): scores -> softmax -> weighted V.
#[allow(clippy::too_many_arguments)]
fn direct_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    hq: usize,
    n_rep: usize,
    cap: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kv = h / n_rep;
        let mut scores = vec![0.0f32; cap];
        for t in 0..cap {
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[kv * cap * d + t * d + dd];
            }
            scores[t] = s * scale + mask[t];
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - m).exp();
            denom += *sc;
        }
        for dd in 0..d {
            let mut acc = 0.0f32;
            for t in 0..cap {
                acc += scores[t] / denom * v[kv * cap * d + t * d + dd];
            }
            out[h * d + dd] = acc;
        }
    }
    out
}

/// Deterministic pseudo-random fill in [-1, 1).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

#[test]
fn flash_decode_matches_direct() {
    // GQA decode shapes; an additive mask with the last few slots masked off (t > pos).
    let (hq, n_rep, cap, d) = (4usize, 2usize, 6usize, 8usize);
    let hkv = hq / n_rep;
    let scale = 1.0 / (d as f32).sqrt();
    let q = fill(hq * d, 1);
    let k = fill(hkv * cap * d, 2);
    let v = fill(hkv * cap * d, 3);
    // pos = 3: keys 0..=3 valid, 4..6 masked.
    let mask: Vec<f32> = (0..cap)
        .map(|t| if t <= 3 { 0.0 } else { -1.0e9 })
        .collect();

    let flash = flash_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale);
    let direct = direct_decode(&q, &k, &v, &mask, hq, n_rep, cap, d, scale);
    assert_close(&flash, &direct, 1e-5);
}

#[test]
fn flash_decode_no_mask_matches_direct() {
    // all keys valid (mask all zero) - the unmasked online softmax.
    let (hq, n_rep, cap, d) = (3usize, 1usize, 5usize, 4usize);
    let hkv = hq / n_rep;
    let scale = 0.5;
    let q = fill(hq * d, 7);
    let k = fill(hkv * cap * d, 8);
    let v = fill(hkv * cap * d, 9);
    let mask = vec![0.0f32; cap];
    let flash = flash_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale);
    let direct = direct_decode(&q, &k, &v, &mask, hq, n_rep, cap, d, scale);
    assert_close(&flash, &direct, 1e-5);
}
