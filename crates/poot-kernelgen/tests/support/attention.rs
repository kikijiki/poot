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
    attention_reference_keys(spec, q, k, v, mask, KeyExtras::default())
}

/// The operands beyond `q`, `k`, `v` and `mask` that a key layout or bound adds; each is `None` when the spec
/// does not read it.
#[derive(Clone, Copy, Default)]
pub(crate) struct KeyExtras<'a> {
    /// The shared part of a split key, `[B,1,Lk,s]`.
    pub(crate) k_shared: Option<&'a [f32]>,
    /// The live length of each batch row.
    pub(crate) len: Option<&'a [i32]>,
    /// The `[B,Lk]` slot map of a paged key.
    pub(crate) read: Option<&'a [i32]>,
    /// The sink logit of each query head, `[Hq]`: one more position in every row's softmax that weighs
    /// into the maximum and the normalizer and has no value.
    pub(crate) sinks: Option<&'a [f32]>,
    /// The `[B,Lq]` query positions of a causal mask, which reads no mask tensor: key `t` is visible to row
    /// `i` of batch row `b` when `t <= max(pos, 0)` and, under a window, within it.
    pub(crate) positions: Option<&'a [i32]>,
}

/// [`attention_reference`] over a split key (`k` the first `dk - s` channels of each key, `k_shared [B,1,Lk,s]`
/// the last `s`, shared by every head) and a live bound (`len [B]`: batch row `b` visits only its first
/// `clamp(len[b], 0, Lk)` keys, an empty row being `0 / 0`, NaN). `v` is `[B,Hkv,Lk,Dv]` and the output
/// `[B,Hq,Lq,Dv]`. Keys past a row's bound are never read, so poison there cannot reach the result. A paged
/// key (`read [B,Lk]`, a slot per position) reads `k [S,Hkv,Dk]` and `v [S,Hkv,Dv]` as pools: key `t` of row `b`
/// and key head `g` is pool row `read[b * Lk + t] * Hkv + g`. With `sinks`, each row's softmax also holds the
/// head's sink logit, which is never a key: a row with no visible key divides by the sink's mass alone.
pub(crate) fn attention_reference_keys(
    spec: &AttentionSpec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    extras: KeyExtras<'_>,
) -> Vec<f32> {
    let KeyExtras {
        k_shared,
        len,
        read,
        sinks,
        positions,
    } = extras;
    use poot_kernelgen::{KeyLayout, MaskKind, ScoreTransform};
    let (hq, lq, lk, dk, dv) = (spec.hq, spec.lq, spec.lk, spec.dk, spec.dv);
    let hkv = hq / spec.n_rep;
    let shared_dim = match spec.key {
        KeyLayout::Single | KeyLayout::Paged => 0,
        KeyLayout::Split { shared_dim } => shared_dim,
    };
    let head_dim = dk - shared_dim;
    let mask_heads = match spec.mask {
        MaskKind::Broadcast | MaskKind::Causal { .. } => 1,
        MaskKind::PerHead => hq,
        MaskKind::Blocks { .. } => unreachable!("the dense reference takes an additive mask"),
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
            let kv_head = h / spec.n_rep;
            let kv_row = (b * hkv + kv_head) * lk;
            // Row `t` of this head's keys and of its values: the dense plane, or the pool row of its slot.
            let key_at = |t: usize| match read {
                Some(read) => read[b * lk + t] as usize * hkv + kv_head,
                None => kv_row + t,
            };
            let mask_plane = (b * mask_heads + if mask_heads == 1 { 0 } else { h }) * lq * lk;
            for i in 0..lq {
                let q_row = ((b * hq + h) * lq + i) * dk;
                let scores: Vec<f64> = (0..keys)
                    .map(|t| {
                        let mut dot: f64 = (0..head_dim)
                            .map(|c| {
                                f64::from(q[q_row + c]) * f64::from(k[key_at(t) * head_dim + c])
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
                        match (spec.mask, positions) {
                            (MaskKind::Causal { window }, Some(positions)) => {
                                let position = positions[b * lq + i].max(0) as usize;
                                if t > position || window.is_some_and(|w| position - t >= w) {
                                    f64::NEG_INFINITY
                                } else {
                                    score
                                }
                            }
                            _ => score + f64::from(mask[mask_plane + i * lk + t]),
                        }
                    })
                    .collect();
                let sink = sinks.map(|sinks| f64::from(sinks[h]));
                let top = scores
                    .iter()
                    .copied()
                    .chain(sink)
                    .fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let denominator: f64 =
                    weights.iter().sum::<f64>() + sink.map_or(0.0, |sink| (sink - top).exp());
                let out_row = ((b * hq + h) * lq + i) * dv;
                for c in 0..dv {
                    let acc: f64 = (0..keys)
                        // A hidden key weighs exactly zero and its value row is never read.
                        .filter(|&t| weights[t] != 0.0)
                        .map(|t| weights[t] * f64::from(v[key_at(t) * dv + c]))
                        .sum();
                    out[out_row + c] = (acc / denominator) as f32;
                }
            }
        }
    }
    out
}
