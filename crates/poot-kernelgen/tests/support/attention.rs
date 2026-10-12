//! The textbook attention the generated flash-attention kernel must equal, shared by the integration binaries
//! that interpret or run it.

use poot_kernelgen::AttentionSpec;

/// The textbook attention the generated kernel must equal, in `f64`: per `(b, h, i)`,
/// `softmax(softcap(q . k * scale) + mask) @ v` over every key, in one pass over the whole row (no running
/// maximum, no rescale). `q [B,Hq,Lq,D]`, `k` and `v [B,Hkv,Lk,D]`, `mask [B,Hm,Lq,Lk]`, GQA through
/// `h / n_rep`. A masked score of `-inf` weighs exactly zero; a row with no finite score is `0 / 0`, NaN.
/// The dense, every-key case of [`attention_reference_keys`].
pub(crate) fn attention_reference(
    spec: &AttentionSpec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
) -> Vec<f32> {
    attention_reference_keys(spec, q, k, None, v, mask, None)
}

/// [`attention_reference`] over a split key (`k` the first `dk - s` channels of each key, `k_shared [B,1,Lk,s]`
/// the last `s`, shared by every head) and a live bound (`len [B]`: batch row `b` visits only its first
/// `clamp(len[b], 0, Lk)` keys, an empty row being `0 / 0`, NaN). `v` is `[B,Hkv,Lk,Dv]` and the output
/// `[B,Hq,Lq,Dv]`. Keys past a row's bound are never read, so poison there cannot reach the result.
pub(crate) fn attention_reference_keys(
    spec: &AttentionSpec,
    q: &[f32],
    k: &[f32],
    k_shared: Option<&[f32]>,
    v: &[f32],
    mask: &[f32],
    len: Option<&[i32]>,
) -> Vec<f32> {
    use poot_kernelgen::{KeyLayout, MaskKind, ScoreTransform};
    let (hq, lq, lk, dk, dv) = (spec.hq, spec.lq, spec.lk, spec.dk, spec.dv);
    let hkv = hq / spec.n_rep;
    let shared_dim = match spec.key {
        KeyLayout::Single => 0,
        KeyLayout::Split { shared_dim } => shared_dim,
    };
    let head_dim = dk - shared_dim;
    let mask_heads = match spec.mask {
        MaskKind::Broadcast => 1,
        MaskKind::PerHead => hq,
    };
    let (scale, softcap) = match spec.score {
        ScoreTransform::Scale { scale_bits } => (f32::from_bits(scale_bits), None),
        ScoreTransform::ScaleSoftcap {
            scale_bits,
            cap_bits,
        } => (f32::from_bits(scale_bits), Some(f32::from_bits(cap_bits))),
    };
    let mut out = vec![0.0f32; spec.batch * hq * lq * dv];
    for b in 0..spec.batch {
        let keys = len.map_or(lk, |len| len[b].clamp(0, lk as i32) as usize);
        for h in 0..hq {
            let kv_row = (b * hkv + h / spec.n_rep) * lk;
            let mask_plane = (b * mask_heads + if mask_heads == 1 { 0 } else { h }) * lq * lk;
            for i in 0..lq {
                let q_row = ((b * hq + h) * lq + i) * dk;
                let scores: Vec<f64> = (0..keys)
                    .map(|t| {
                        let mut dot: f64 = (0..head_dim)
                            .map(|c| {
                                f64::from(q[q_row + c]) * f64::from(k[(kv_row + t) * head_dim + c])
                            })
                            .sum();
                        if let Some(shared) = k_shared {
                            let shared_row = (b * lk + t) * shared_dim;
                            dot += (0..shared_dim)
                                .map(|c| {
                                    f64::from(q[q_row + head_dim + c])
                                        * f64::from(shared[shared_row + c])
                                })
                                .sum::<f64>();
                        }
                        let mut score = dot * f64::from(scale);
                        if let Some(cap) = softcap {
                            score = f64::from(cap) * (score / f64::from(cap)).tanh();
                        }
                        score + f64::from(mask[mask_plane + i * lk + t])
                    })
                    .collect();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                let out_row = ((b * hq + h) * lq + i) * dv;
                for c in 0..dv {
                    let acc: f64 = (0..keys)
                        .map(|t| weights[t] * f64::from(v[(kv_row + t) * dv + c]))
                        .sum();
                    out[out_row + c] = (acc / denominator) as f32;
                }
            }
        }
    }
    out
}
