#[cfg(test)]
use super::*;
#[cfg(test)]
use poot_graph_ir::ops::swiglu;

/// `min(x, limit)` from `UnOp::Neg` + `BinOp::Max` (`poot_graph_ir` has no `Min`/`Clamp` primitive).
#[cfg(test)]
pub(crate) fn clamp_max(b: &Builder, x: Traced, limit: f32) -> Traced {
    let neg_x = b.unary(UnOp::Neg, x);
    let clipped = b.binary_scalar(BinOp::Max, neg_x, Scalar::F32(-limit));
    b.unary(UnOp::Neg, clipped)
}

/// `clamp(x, -limit, limit)`, real `DeepseekV4MLP.forward`'s `up_proj` path (FR-006).
#[cfg(test)]
pub(crate) fn clamp_sym(b: &Builder, x: Traced, limit: f32) -> Traced {
    let upper = clamp_max(b, x, limit);
    b.binary_scalar(BinOp::Max, upper, Scalar::F32(-limit))
}

/// Sum over axis `rank-2` (the first of the two trailing `hc` axes), keepdim. `x` must have rank `>= 2`.
#[cfg(test)]
pub(crate) fn sum_penultimate_keepdim(b: &Builder, x: Traced) -> Traced {
    let r = b.aval(x).shape.len();
    b.reduce(RedOp::Sum, x, r - 2, true)
}

/// Manifold-Constrained Hyper-Connection mapping (`DeepseekV4HyperConnection.forward`). From a widened residual
/// `streams[1,s,hc,hidden]` computes `pre` (per-stream collapse weight, `[1,s,hc]`), `post` (output-placement
/// weight in `[0,2]`, `[1,s,hc]`), `comb` (`hc x hc` doubly-stochastic mixer, `[1,s,hc,hc]`), and
/// `collapsed = sum_hc(pre * streams)` (`[1,s,hidden]`) for the sublayer. `fn_w[hc*hidden, (2+hc)*hc]`,
/// `base[(2+hc)*hc]`, `scale[3]` are this module's weight-constant naming. `s` is the sequence length (`1` for
/// decode).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn hyper_connection(
    b: &Builder,
    streams: Traced,
    fn_w: Traced,
    base: Traced,
    scale: Traced,
    hc: usize,
    hidden: usize,
    s: usize,
    rms_eps: f32,
    hc_eps: f32,
    sinkhorn_iters: usize,
) -> (Traced, Traced, Traced) {
    let flat = b.reshape(streams, vec![1, s, hc * hidden]);
    let normed = rmsnorm_no_weight(b, flat, rms_eps);
    let mix = linear(b, normed, fn_w, None); // [1,s,(2+hc)*hc]

    let pre_w = b.slice(mix, 2, 0, hc);
    let post_w = b.slice(mix, 2, hc, 2 * hc);
    let comb_w = b.slice(mix, 2, 2 * hc, (2 + hc) * hc); // [1,s,hc*hc]
    let pre_b = b.slice(base, 0, 0, hc);
    let post_b = b.slice(base, 0, hc, 2 * hc);
    let comb_b = b.reshape(b.slice(base, 0, 2 * hc, (2 + hc) * hc), vec![hc, hc]);
    let pre_scale = b.slice(scale, 0, 0, 1);
    let post_scale = b.slice(scale, 0, 1, 2);
    let comb_scale = b.slice(scale, 0, 2, 3);

    let pre_logits = b.binary(BinOp::Add, b.binary(BinOp::Mul, pre_w, pre_scale), pre_b);
    let pre = b.binary_scalar(BinOp::Add, sigmoid(b, pre_logits), Scalar::F32(hc_eps));

    let post_logits = b.binary(BinOp::Add, b.binary(BinOp::Mul, post_w, post_scale), post_b);
    let post = b.binary_scalar(BinOp::Mul, sigmoid(b, post_logits), Scalar::F32(2.0));

    let comb_w4 = b.reshape(comb_w, vec![1, s, hc, hc]);
    let comb_logits = b.binary(
        BinOp::Add,
        b.binary(BinOp::Mul, comb_w4, comb_scale),
        comb_b,
    );
    let comb0 = b.binary_scalar(BinOp::Add, softmax(b, comb_logits), Scalar::F32(hc_eps));

    let col_sum = sum_penultimate_keepdim(b, comb0); // [1,s,1,hc]
    let mut comb = b.binary(
        BinOp::Div,
        comb0,
        b.binary_scalar(BinOp::Add, col_sum, Scalar::F32(hc_eps)),
    );
    for _ in 0..sinkhorn_iters.saturating_sub(1) {
        let row_sum = b.reduce(RedOp::Sum, comb, 3, true); // [1,s,hc,1] - already trailing, no transpose
        comb = b.binary(
            BinOp::Div,
            comb,
            b.binary_scalar(BinOp::Add, row_sum, Scalar::F32(hc_eps)),
        );
        let col_sum = sum_penultimate_keepdim(b, comb);
        comb = b.binary(
            BinOp::Div,
            comb,
            b.binary_scalar(BinOp::Add, col_sum, Scalar::F32(hc_eps)),
        );
    }

    let pre_r = b.reshape(pre, vec![1, s, hc, 1]);
    let weighted = b.binary(BinOp::Mul, pre_r, streams); // [1,s,hc,hidden]
    let collapsed = b.reduce(RedOp::Sum, weighted, 2, false); // [1,s,hidden]

    (post, comb, collapsed)
}

/// Combine a sublayer's single-stream output back into the widened residual (`DeepseekV4DecoderLayer.forward`):
/// `post[..,None] * sublayer_out[...,None,:] + matmul(comb^T, residual)`. Every one of the `hc` streams gets its
/// own scaled copy of the sublayer output, plus an `hc x hc` mix of the residual through `comb` (transposed,
/// since `comb` is doubly stochastic but not symmetric).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn hyper_connection_combine(
    b: &Builder,
    post: Traced,
    comb: Traced,
    sublayer_out: Traced,
    residual: Traced,
    hc: usize,
    s: usize,
    hidden: usize,
) -> Traced {
    let post_r = b.reshape(post, vec![1, s, hc, 1]);
    let sub_r = b.reshape(sublayer_out, vec![1, s, 1, hidden]);
    let outer = b.binary(BinOp::Mul, post_r, sub_r); // broadcast -> [1,s,hc,hidden]
    let comb_t = b.transpose(comb, vec![0, 1, 3, 2]);
    let mix = b.matmul(comb_t, residual); // [1,s,hc,hc] @ [1,s,hc,hidden] -> [1,s,hc,hidden]
    b.binary(BinOp::Add, outer, mix)
}

/// Final HC-stream collapse (`DeepseekV4HyperHead.forward`): the normalize-project-sigmoid-collapse chain of
/// [`hyper_connection`]'s `pre` branch, standalone (mix width `hc`, no `post`/`comb`). Runs once before the
/// closing RMSNorm.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hyper_head(
    b: &Builder,
    streams: Traced,
    fn_w: Traced,
    base: Traced,
    scale: Traced,
    hc: usize,
    hidden: usize,
    s: usize,
    rms_eps: f32,
    hc_eps: f32,
) -> Traced {
    let flat = b.reshape(streams, vec![1, s, hc * hidden]);
    let normed = rmsnorm_no_weight(b, flat, rms_eps);
    let mixes = linear(b, normed, fn_w, None); // [1,s,hc]
    let logits = b.binary(BinOp::Add, b.binary(BinOp::Mul, mixes, scale), base);
    let pre = b.binary_scalar(BinOp::Add, sigmoid(b, logits), Scalar::F32(hc_eps));
    let pre_r = b.reshape(pre, vec![1, s, hc, 1]);
    let weighted = b.binary(BinOp::Mul, pre_r, streams);
    b.reduce(RedOp::Sum, weighted, 2, false) // [1,s,hidden]
}

/// Shared DeepSeek-V4/GLM5Next clamped SwiGLU composition.
#[cfg(test)]
pub(crate) fn deepseek4_clamped_swiglu(
    b: &Builder,
    gate: Traced,
    up: Traced,
    limit: f32,
) -> Traced {
    let gate = clamp_max(b, gate, limit);
    let up = clamp_sym(b, up, limit);
    swiglu(b, gate, up)
}

/// One mHC site's projection in the layout [`hyper_connection`] contracts against. The checkpoint stores
/// `[mix, streams * hidden]`, so the transpose is an equation here. The row is widened to f32 because
/// `hyper_connection` (shared with GLM-5.3-Flash) does f32 elementwise work on it, as
/// `Glm5NextTextTrace::mhc_map` does.
#[cfg(test)]
pub(crate) fn v4_mhc_projection(b: &Builder, spec: &V4DenseSourceSpec) -> Traced {
    b.transpose(v4_source_f32(b, spec), vec![1, 0])
}
