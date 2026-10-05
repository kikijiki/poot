//! One shared home for the plain-loop reference implementations the family CPU-oracle tests check
//! their tracers against (R474-014).
//!
//! These are deliberately independent of the graph ops they verify, but they must not be re-written
//! once per family: a wrong term in one copy would go unnoticed in the others. Each function here is
//! the single definition, so a mutation shows up in every family that uses it.

/// ALiBi slopes (Press et al.; HF `bloom` `get_slopes`), independent of `poot_llm::graphs::alibi_slopes`
/// (no poot-llm dependency); power-of-2 `n_heads` only.
pub(crate) fn alibi_slopes_pow2(n_heads: usize) -> Vec<f32> {
    assert!(n_heads.is_power_of_two());
    let start = 2f32.powf(-8.0 / n_heads as f32);
    (0..n_heads).map(|i| start.powi(i as i32 + 1)).collect()
}

/// The `[1, n_heads, l, l]` ALiBi prefill mask, mirroring `poot_llm::graphs::alibi_prefill_mask`.
pub(crate) fn alibi_prefill_mask_ref(n_heads: usize, l: usize) -> Vec<f32> {
    let slopes = alibi_slopes_pow2(n_heads);
    let mut out = vec![0.0f32; n_heads * l * l];
    for hh in 0..n_heads {
        for i in 0..l {
            for j in 0..l {
                out[(hh * l + i) * l + j] = if j <= i {
                    -slopes[hh] * (i as f32 - j as f32)
                } else {
                    -1.0e30
                };
            }
        }
    }
    out
}

/// LayerNorm with a bias: `(x - mean) / sqrt(var + eps) * w + b`.
pub(crate) fn layernorm_ref(x: &[f32], w: &[f32], b: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let den = (var + eps).sqrt();
    (0..n).map(|i| (x[i] - mean) / den * w[i] + b[i]).collect()
}

/// Bias-free LayerNorm: `(x - mean) / sqrt(var + eps) * w`.
pub(crate) fn layernorm_no_bias_ref(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let den = (var + eps).sqrt();
    (0..n).map(|i| (x[i] - mean) / den * w[i]).collect()
}

/// DeepSeek's interleaved-pair rotation (module docs, item 6): pairs `(x[2i], x[2i+1])`, not the
/// half-split `rope_ref` used elsewhere. `cos`/`sin` are position-major `[max_pos, half]`.
pub(crate) fn rope_interleaved_ref(
    row: &[f32],
    cos: &[f32],
    sin: &[f32],
    pos: usize,
    half: usize,
) -> Vec<f32> {
    let c = &cos[pos * half..(pos + 1) * half];
    let s = &sin[pos * half..(pos + 1) * half];
    let mut out = vec![0.0f32; 2 * half];
    for i in 0..half {
        let (a, bb) = (row[2 * i], row[2 * i + 1]);
        out[2 * i] = a * c[i] - bb * s[i];
        out[2 * i + 1] = a * s[i] + bb * c[i];
    }
    out
}
