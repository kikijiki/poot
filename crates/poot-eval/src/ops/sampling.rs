//! The CPU oracle for the sampling primitives (card 551a, R472-007, deval.md section 8): F32 logits
//! only, every other dtype is a typed `Unsupported` at the `walk` dispatch (checked by `OpKind::infer`
//! before any graph reaches here). Mirrors the device bodies' formulas exactly (`poot-graph-plan`'s
//! `sample_gumbel_argmax_batched.rs` and siblings): the row-max/min and the tie-broken argmax are a
//! genuine reduction (order-independent - a lowest-index-on-tie ascending scan agrees with the device's
//! distributed tree reduce by construction, no lane-order simulation needed); the top-k/top-p bisection
//! steps are deterministic IEEE arithmetic run in the identical order, so a sequential CPU loop lands on
//! the same bits as the device's 64-lane LDS-reduced count/mass.

use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::hash32;

use poot_tensor::HostTensor;

use super::i32_lane;
use crate::EvalError;

/// `OpKind::RandomUniform`'s oracle: `seed` is `[rows]` I32 (u32 bits); output is `[rows, cols]` F32.
/// Bit-for-bit what `random_uniform.rs` computes (tier 1, ADR-0101).
pub(crate) fn random_uniform(seed: &HostTensor, cols: usize) -> Result<HostTensor, EvalError> {
    let seed_words = i32_lane(seed, "random_uniform")?;
    let rows = seed_words.len();
    let mut data = vec![0.0f32; rows * cols];
    for (r, &s) in seed_words.iter().enumerate() {
        let s = s as u32;
        for c in 0..cols {
            let idx = (r * cols + c) as u32;
            let x = hash32(s, idx);
            let mantissa = (x >> 9) as i32;
            data[r * cols + c] = (2.0f32 * (mantissa as f32) + 1.0f32) * 5.9604645e-8f32; // 2^-24
        }
    }
    let mut shape = seed.shape().to_vec();
    shape.push(cols);
    Ok(HostTensor::f32(shape, data))
}

/// The finite stand-in a caller writes for a token it excluded from the row (a guided-decoding mask).
/// `SampleToken` flags a non-finite logit, so a masked token cannot be `-inf`; it is the lowest finite
/// f32 instead, and the bisection floor ignores it so top-k and top-p search the real logits only.
pub const MASKED_LOGIT: f32 = f32::MIN;

/// Fixed-point scale of a token's top-p mass: `2^24` units for the top token (probability ratio 1).
/// A token whose probability is at least `2^-25` of the top token keeps a nonzero mass, so the tail of a
/// flat distribution is summed instead of rounded away.
const TOP_P_MASS_UNITS: f32 = 16_777_216.0;

/// One token's top-p mass in units of `2^-24` of the top token: `round(exp((logit - max) * inv_temp) *
/// 2^24)`, at most `2^24`. The one definition of a token's nucleus mass: the oracle's bisection, the
/// host sampler's candidate cut and the device kernel all sum these integers, so the sum is exact and
/// independent of reduction order.
pub fn top_p_token_mass(logit: f32, max_logit: f32, inv_temp: f32) -> u32 {
    (((logit - max_logit) * inv_temp).exp() * TOP_P_MASS_UNITS).round() as u32
}

/// The mass the nucleus must reach: `round(top_p * total)` with the `f32` product the device kernel also
/// computes (`total` rounded to f32 first). Only this threshold is rounded; the masses compared against
/// it are exact integers.
pub fn top_p_threshold(total: u64, top_p: f32) -> u64 {
    (top_p * total as f32).round() as u64
}

/// One row's `(token, non_finite_index)`. `params` is `(inv_temp, floor_offset, noise_scale)`; `top_k`
/// is `None` when the rule has no top-k operand, `Some(k)` otherwise (`k <= 0` disables it, matching the
/// kernel); `top_p` is `None` unless `rule` is `GumbelTopKTopP`.
fn sample_one_row(
    logits: &[f32],
    noise: Option<&[f32]>,
    params: Option<(f32, f32, f32)>,
    top_k: Option<i32>,
    top_p: Option<f32>,
) -> (i32, i32) {
    let mut max_logit = f32::MIN;
    let mut min_logit = f32::MAX;
    let mut non_finite_idx: Option<usize> = None;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() {
            if v > max_logit {
                max_logit = v;
            }
            // `MASKED_LOGIT` is a row the caller excluded, not a candidate: counting it would make the
            // top-k / top-p bisection start from a floor no real logit approaches (R2-C1-002).
            if v < min_logit && v != MASKED_LOGIT {
                min_logit = v;
            }
        } else if non_finite_idx.is_none() {
            non_finite_idx = Some(i);
        }
    }
    let flag = non_finite_idx.map_or(-1, |i| i as i32);

    let Some((inv_temp, floor_offset, noise_scale)) = params else {
        // Greedy: token = min { i : logits[i] == max(logits) } over finite logits, lowest index on tie
        // (an ascending scan with strict `>` keeps the first occurrence).
        let mut best_idx = -1i32;
        let mut best_val = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_finite() && (best_idx < 0 || v > best_val) {
                best_val = v;
                best_idx = i as i32;
            }
        }
        let token = if non_finite_idx.is_some() || best_idx < 0 {
            0
        } else {
            best_idx
        };
        return (token, flag);
    };

    let noise = noise.expect("sample_one_row: params implies a Gumbel-family rule with noise");
    // t_k starts (and, with top-k disabled or absent, stays) at `min_logit` - a finite lower
    // bound every finite logit is `>=` by construction - exactly as the device kernels' own `lo`
    // does (`sample_truncated_gumbel_argmax_batched.rs`, `sample_topp_gumbel_argmax_batched.rs`).
    // `floor1` (the pre-top-p floor) is therefore always finite, never `-inf`: starting the
    // top-p bisection below from an unconditional `max_logit + floor_offset` (`-inf` whenever
    // min-p is disabled too) breaks IEEE averaging (`(-inf + finite) * 0.5 == -inf`), pinning the
    // bisection at `-inf` forever and silently disabling top-p (card 677, found by card 551b's
    // review).
    let mut lo = min_logit;
    if let Some(k) = top_k
        && (k as f32) > 0.0
    {
        let mut hi = max_logit;
        for _ in 0..30 {
            let mid = (lo + hi) * 0.5f32;
            let count = logits.iter().filter(|&&v| v >= mid).count() as f32;
            if count >= k as f32 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    let t_k = lo;
    let min_p_floor = max_logit + floor_offset;
    let mut floor = if min_p_floor > t_k { min_p_floor } else { t_k };
    if let Some(tp) = top_p {
        let floor1 = floor;
        if tp <= 0.0f32 {
            floor = max_logit;
        } else if tp < 1.0f32 {
            let mass_at = |thresh: f32| -> u64 {
                logits
                    .iter()
                    .filter(|&&v| v >= thresh)
                    .map(|&v| u64::from(top_p_token_mass(v, max_logit, inv_temp)))
                    .sum()
            };
            let tp_thresh = top_p_threshold(mass_at(floor1), tp);
            let mut tlo = floor1;
            let mut thi = max_logit;
            for _ in 0..30 {
                let mid = (tlo + thi) * 0.5f32;
                if mass_at(mid) >= tp_thresh {
                    tlo = mid;
                } else {
                    thi = mid;
                }
            }
            let t_p = tlo;
            floor = if floor1 > t_p { floor1 } else { t_p };
        }
        // tp >= 1.0: top-p disabled, floor stays floor1.
    }

    let mut best_idx = -1i32;
    let mut best_val = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if v >= floor {
            let scaled = v * inv_temp;
            let perturbed = scaled + noise_scale * noise[i];
            if best_idx < 0 || perturbed > best_val {
                best_val = perturbed;
                best_idx = i as i32;
            }
        }
    }
    let token = if non_finite_idx.is_some() || best_idx < 0 {
        0
    } else {
        best_idx
    };
    (token, flag)
}

/// `OpKind::SampleToken`'s oracle. `logits`/`noise` are `[rows, vocab]` F32 (`noise` absent for
/// `Greedy`); `params` is `[rows, 3]` F32 (`[rows, 4]` for `GumbelTopKTopP`, `+top_p`); `top_k` is
/// `[rows]` I32. Output is `[rows, 2]` I32.
pub(crate) fn sample_token(
    rule: SampleRule,
    logits: &HostTensor,
    noise: Option<&HostTensor>,
    params: Option<&HostTensor>,
    top_k: Option<&HostTensor>,
) -> Result<HostTensor, EvalError> {
    let rank = logits.shape().len();
    let vocab = logits.shape()[rank - 1];
    let rows: usize = logits.shape()[..rank - 1].iter().product();
    let logits_data = logits.to_f32()?;
    let noise_data = noise.map(|t| t.to_f32()).transpose()?;
    let params_data = params.map(|t| t.to_f32()).transpose()?;
    let top_k_words = match top_k {
        Some(t) => Some(i32_lane(t, "sample_token")?),
        None => None,
    };
    let params_cols = if matches!(rule, SampleRule::GumbelTopKTopP) {
        4
    } else {
        3
    };

    let mut out = vec![0i32; rows * 2];
    for r in 0..rows {
        let row_logits = &logits_data[r * vocab..(r + 1) * vocab];
        let row_noise = noise_data.as_ref().map(|d| &d[r * vocab..(r + 1) * vocab]);
        let row_params = params_data
            .as_ref()
            .map(|d| &d[r * params_cols..r * params_cols + params_cols]);
        let params_tuple = row_params.map(|p| (p[0], p[1], p[2]));
        let k = top_k_words.as_ref().map(|w| w[r]);
        let tp = if matches!(rule, SampleRule::GumbelTopKTopP) {
            row_params.map(|p| p[3])
        } else {
            None
        };
        let (token, flag) = sample_one_row(row_logits, row_noise, params_tuple, k, tp);
        out[r * 2] = token;
        out[r * 2 + 1] = flag;
    }
    let mut shape = logits.shape()[..rank - 1].to_vec();
    shape.push(2);
    Ok(HostTensor::i32(shape, out))
}
